"""Generate a document and questions with the model, then answer them with Parallel Prompting.

Three stages, all run with Parallel Prompting (PP) on one model:

  1. Document: the model writes an outline of the topic, then writes every
     section at once (one PP stream per section) until the document reaches
     --doc-tokens.
  2. Questions: the model writes --num-questions questions about the document
     at once (one stream per question), each about a different part of it.
  3. Answers: every question is answered with PP, --parallel-size questions per
     request. With --compare, stock SGLang (no PP) answers the same prompts too,
     one request per question with the document prefix cached, once per
     --baselines backend: Triton + CUDA graphs (the same kernels PP uses, so
     the difference is PP alone) and FlashInfer + CUDA graphs (stock's fastest).

Every prompt has a default and can be overridden. Instead of generating, you can
give your own document (--doc-file) or questions (--questions-file).

Answering time excludes the one-time document prefill on both sides, as in the
paper's benchmarks: stock SGLang has the document in its prefix cache and PP has
it prefilled once. Both sides run once untimed, then once timed.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time

from common import DOC_TEMPLATE, QUESTION_TEMPLATE

EXAMPLES = """\
examples (times measured on one A100-PCIe-40GB, CodeLlama-7B-Instruct):
  # Defaults: a 4k-token document, 64 questions, answers up to 64 tokens;
  # compared with stock Triton + CUDA graphs and FlashInfer + CUDA graphs
  python pp_bench/demo_gen.py --cuda-graph --compare

  # Only the same-kernel baseline (stock Triton + CUDA graphs, no PP)
  python pp_bench/demo_gen.py --cuda-graph --compare --baselines triton

  # PP's best regime: long document, many questions, short answers.
  # Measured: PP 3.2s; stock SGLang without PP: Triton + CUDA graphs 19.1s (6.0x),
  # FlashInfer + CUDA graphs 11.7s (3.7x); 240-247 of 256 answers token-identical
  python pp_bench/demo_gen.py --doc-tokens 12000 --num-questions 256 --answer-tokens 32 \\
      --parallel-size 32 --cuda-graph --compare

  # Your own topic and lengths
  python pp_bench/demo_gen.py --topic "the James Webb Space Telescope" \\
      --doc-tokens 8000 --num-questions 128 --answer-tokens 48 --cuda-graph --compare

  # Your own prompt ({topic}, {sections}, {outline}, {words}, {i}, {title},
  # {excerpt} are filled in where the default prompt uses them; see --help above)
  python pp_bench/demo_gen.py --topic "sourdough baking" \\
      --question-prompt 'Write one tricky "why" question about "{excerpt}". [/INST]'

  # Save a run, then reuse its document and questions (nothing is generated)
  python pp_bench/demo_gen.py --doc-tokens 12000 --num-questions 256 --save run.json
  python3 -c "import json; d = json.load(open('run.json')); \\
      open('doc.txt', 'w').write(d['doc']); open('qs.txt', 'w').write('\\n'.join(d['questions']))"
  python pp_bench/demo_gen.py --doc-file doc.txt --questions-file qs.txt \\
      --answer-tokens 32 --parallel-size 64 --cuda-graph --compare

tips:
  - The speedup grows with document length x number of questions and shrinks with
    long answers (each stream also scans its sibling streams' tokens). Short
    documents with few questions run at about stock speed.
  - --parallel-size between 16 and 64 is usually best; 0 puts every question in
    one request.
  - Keep --doc-tokens under ~15000 for a 16k-context model such as CodeLlama.
"""


# Default prompts use the Llama-2 / CodeLlama-Instruct format ([INST] ... [/INST]).
# Placeholders in braces are filled in by the script.
DEFAULTS = dict(
    topic="the history and engineering of the Panama Canal",
    outline_prompt=(
        "[INST] Write the outline of a long, detailed encyclopedia article about {topic}: "
        "exactly {sections} distinct section titles, each on a different subtopic, "
        "numbered, one per line, and nothing else. [/INST]"
    ),
    doc_prompt=(
        "[INST] You are writing a long, detailed encyclopedia article about {topic}. "
        "Its outline is:\n{outline}\n\nWrite only the section named below, in about "
        "{words} words of plain prose with concrete facts, names, dates and numbers. "
        "Do not repeat the title, and do not write any other section.\n\n"
    ),
    section_prompt="Section to write: {i}. {title} [/INST]",
    question_prompt=(
        'Write one specific question that the passage answers, about the part that begins '
        'with "{excerpt}". Reply with the question only. [/INST]'
    ),
)


def newline_id(tok):
    """Token id of a bare newline, if the tokenizer has one (answers stop there)."""
    ids = tok.encode("\n", add_special_tokens=False)
    return ids[-1] if ids and tok.decode(ids[-1:]) == "\n" else None


def stop_ids(tok, eos_ids, args):
    nl = None if args.no_newline_stop else newline_id(tok)
    return sorted(set(eos_ids) | ({nl} if nl is not None else set()))


# ----------------------------------------------------------------------- PP side


def pp_engine(args):
    # A PP request holds the document plus P * (question + answer) slots in one
    # KV row, so the row can be longer than the model's context, while positions
    # stay below document + one stream (see bench_pp_vs_sglang.py).
    os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    from sglang.srt.parallel_prompt.engine import ParallelPromptEngine
    from sglang.srt.server_args import ServerArgs

    return ParallelPromptEngine(
        ServerArgs(
            model_path=args.model_path,
            attention_backend="triton",
            sampling_backend="pytorch",
            disable_cuda_graph=True,  # PP captures its own graphs
            disable_radix_cache=True,
            mem_fraction_static=args.mem_fraction_static,
            context_length=65536,
            dtype="bfloat16",
        ),
        cuda_graph=args.cuda_graph,
    )


def pp_generate(engine, prefix, streams, max_new_tokens, stop=None, parallel_size=None):
    """One shared prefix, one PP stream per suffix; returns the decoded outputs."""
    tok = engine.tokenizer
    doc = tok.encode(prefix)
    qs = [tok.encode(s, add_special_tokens=False) for s in streams]
    out = engine.generate([doc], [qs], max_new_tokens, stop_token_ids=stop,
                          parallel_size=parallel_size)
    return [tok.decode(o, skip_special_tokens=True) for o in out[0]]


def make_document(engine, args):
    n_sec = max(1, math.ceil(args.doc_tokens / args.section_tokens))
    topic = args.topic
    tic = time.perf_counter()
    outline = pp_generate(
        engine, args.outline_prompt.format(topic=topic, sections=n_sec), ["\n1."],
        max_new_tokens=24 * n_sec,
    )[0]
    titles = []
    for line in ("1." + outline).splitlines():
        m = re.match(r"\s*\d+[.)]\s*(.+)", line)
        t = m.group(1).strip().strip("*").rstrip(".") if m else ""
        if t and t not in titles:
            titles.append(t)
    titles = (titles + [f"More about {topic} (part {k})" for k in range(1, n_sec + 1)])[:n_sec]
    print(f"[doc] outline: {len(titles)} sections", flush=True)

    # Every section is written at once, one PP stream per section; each one
    # sees the whole outline, so they cover different ground.
    outline_text = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(titles))
    prefix = args.doc_prompt.format(topic=topic, outline=outline_text,
                                    words=int(args.section_tokens * 0.6))
    bodies = pp_generate(
        engine, prefix,
        [args.section_prompt.format(i=i + 1, title=t, topic=topic) for i, t in enumerate(titles)],
        max_new_tokens=args.section_tokens,
    )
    parts = []
    for i, (t, b) in enumerate(zip(titles, bodies)):
        b = b.strip()
        if not b.endswith("."):  # cut a section that hit the length cap mid-sentence
            end = b.rfind(". ")
            b = b[: end + 1] if end > 0 else b
        parts.append(f"Section {i + 1}: {t}\n{b}")
    doc = f"{topic[0].upper()}{topic[1:]}\n\n" + "\n\n".join(parts)
    n_tok = len(engine.tokenizer.encode(doc))
    print(f"[doc] {n_tok} tokens in {time.perf_counter() - tic:.1f}s "
          f"({len(titles)} sections written in parallel)", flush=True)
    if n_tok < 0.8 * args.doc_tokens:
        print(f"[doc] shorter than --doc-tokens {args.doc_tokens}: sections ended early; "
              f"raise --doc-tokens to get more sections", flush=True)
    return doc


def make_questions(engine, args, doc):
    words = doc.split()
    n = args.num_questions
    tic = time.perf_counter()
    # Question i is about the part of the document starting at word i * len / n.
    excerpts = [" ".join(words[i * len(words) // n:][:8]) for i in range(n)]
    tok = engine.tokenizer
    nl = newline_id(tok)
    raw = pp_generate(
        engine, "[INST] " + DOC_TEMPLATE.format(doc=doc),
        [args.question_prompt.format(i=i + 1, excerpt=e) for i, e in enumerate(excerpts)],
        max_new_tokens=args.question_tokens,
        stop=engine.default_stop_ids() + ([nl] if nl is not None else []),
    )
    qs = []
    for r, e in zip(raw, excerpts):
        q = r.strip().split("\n")[0].strip()
        if "?" in q:
            q = q[: q.index("?") + 1]
        qs.append(q if len(q) > 3 else f'What does the passage say about "{e}"?')
    print(f"[questions] {n} questions in {time.perf_counter() - tic:.1f}s "
          f"(written in parallel)", flush=True)
    return qs


def pp_answer(engine, args, doc, questions):
    tok = engine.tokenizer
    d = tok.encode(DOC_TEMPLATE.format(doc=doc))
    qs = [tok.encode(QUESTION_TEMPLATE.format(q=q), add_special_tokens=False) for q in questions]
    stop = stop_ids(tok, engine.default_stop_ids(), args)
    # P larger than the number of questions would only add padding streams.
    P = min(args.parallel_size or len(qs), len(qs))
    for timed in (False, True):  # the untimed run compiles kernels / captures graphs
        cache = engine.prefill_docs([d])
        try:
            import torch

            torch.cuda.synchronize()
            tic = time.perf_counter()
            out = engine.generate_cached(cache, [qs], args.answer_tokens, stop, P)
            torch.cuda.synchronize()
            wall = time.perf_counter() - tic
        finally:
            engine.release_docs(cache)
    st = engine.stats
    print(f"[answers] PP: {wall:.2f}s ({len(qs)} questions, P={st.parallel_size}, "
          f"{st.num_reqs} request(s))", flush=True)
    stop_set = set(stop)
    ids = [[x for x in o if x not in stop_set] for o in out[0]]
    return ids, wall, len(d)


def run_pp(args, out_path):
    engine = pp_engine(args)
    doc = open(args.doc_file).read() if args.doc_file else make_document(engine, args)
    if args.questions_file:
        questions = [l.strip() for l in open(args.questions_file) if l.strip()]
    else:
        questions = make_questions(engine, args, doc)
    ids, wall, doc_len = pp_answer(engine, args, doc, questions)
    tok = engine.tokenizer
    json.dump({"doc": doc, "doc_tokens": doc_len, "questions": questions, "pp_ids": ids,
               "pp_answers": [tok.decode(a, skip_special_tokens=True) for a in ids],
               "pp_wall": wall}, open(out_path, "w"))


# -------------------------------------------------------------------- stock side

BASELINES = {"triton": "Triton + CUDA graphs", "flashinfer": "FlashInfer + CUDA graphs"}


def run_stock(args, path):
    import sglang as sgl
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    data = json.load(open(path))
    tok = get_tokenizer(args.model_path)
    d = tok.encode(DOC_TEMPLATE.format(doc=data["doc"]))
    prompts = [d + tok.encode(QUESTION_TEMPLATE.format(q=q), add_special_tokens=False)
               for q in data["questions"]]
    # CUDA graphs on, and captured up to the full batch: by default SGLang
    # captures decode graphs only up to a small batch size, and a larger batch
    # then decodes without graphs.
    engine = sgl.Engine(model_path=args.model_path, attention_backend=args.stock_backend,
                        disable_cuda_graph=False, cuda_graph_max_bs_decode=len(prompts),
                        mem_fraction_static=args.mem_fraction_static, dtype="bfloat16",
                        log_level="error")
    stop = stop_ids(tok, [tok.eos_token_id], args)
    params = {"temperature": 0, "max_new_tokens": args.answer_tokens, "stop_token_ids": stop}
    engine.generate(input_ids=[d], sampling_params=dict(params, max_new_tokens=1))  # cache doc
    for timed in (False, True):
        tic = time.perf_counter()
        out = engine.generate(input_ids=prompts, sampling_params=params)
        wall = time.perf_counter() - tic
    engine.shutdown()
    ids = [[x for x in o["output_ids"] if x not in set(stop)] for o in out]
    data.setdefault("stock", {})[args.stock_backend] = dict(
        ids=ids, wall=wall, answers=[tok.decode(a, skip_special_tokens=True) for a in ids])
    json.dump(data, open(path, "w"))
    print(f"[answers] stock SGLang, {BASELINES[args.stock_backend]}: {wall:.2f}s", flush=True)


# ------------------------------------------------------------------------- main


def main():
    p = argparse.ArgumentParser(
        description=__doc__, epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = p.add_argument_group("document (stage 1)")
    g.add_argument("--topic", default=DEFAULTS["topic"], help="what the document is about (default: %(default)s)")
    g.add_argument("--doc-tokens", type=int, default=4096, help="target document length in tokens (default: %(default)s; keep under ~15000 for a 16k-context model)")
    g.add_argument("--section-tokens", type=int, default=512, help="tokens per section; sections = doc-tokens / this (default: %(default)s)")
    g.add_argument("--outline-prompt", default=DEFAULTS["outline_prompt"], help="asks for the section titles; uses {topic}, {sections}")
    g.add_argument("--doc-prompt", default=DEFAULTS["doc_prompt"], help="shared prefix of every section; uses {topic}, {outline}, {words}")
    g.add_argument("--section-prompt", default=DEFAULTS["section_prompt"], help="per-section suffix; uses {i}, {title}, {topic}")
    g.add_argument("--doc-file", help="use this text file as the document instead of generating one")
    g = p.add_argument_group("questions (stage 2)")
    g.add_argument("--num-questions", type=int, default=64, help="how many questions to write (default: %(default)s)")
    g.add_argument("--question-tokens", type=int, default=32, help="max tokens per generated question (default: %(default)s)")
    g.add_argument("--question-prompt", default=DEFAULTS["question_prompt"], help="asks for one question; follows \"[INST] <document>\"; uses {i}, {excerpt} (the first words of the part it is about)")
    g.add_argument("--questions-file", help="use these questions (one per line) instead of generating them")
    g = p.add_argument_group("answers (stage 3)")
    g.add_argument("--answer-tokens", type=int, default=64, help="max tokens per answer (default: %(default)s)")
    g.add_argument("--parallel-size", type=int, default=32, help="questions per PP request, P; 0 = all in one (default: %(default)s)")
    g.add_argument("--compare", action="store_true", help="also answer with stock SGLang (no PP) and compare time and answers")
    g.add_argument("--no-newline-stop", action="store_true", help="let answers run past their first line (default: an answer stops at a newline)")
    g.add_argument("--baselines", nargs="+", default=["triton", "flashinfer"], choices=list(BASELINES), help="stock SGLang backends for --compare, each with CUDA graphs: triton = same kernels as PP, flashinfer = stock's fastest (default: both)")
    g = p.add_argument_group("engine and output")
    g.add_argument("--model-path", default="codellama/CodeLlama-7b-Instruct-hf", help="(default: %(default)s)")
    g.add_argument("--mem-fraction-static", type=float, default=0.8, help="(default: %(default)s)")
    g.add_argument("--cuda-graph", action="store_true", help="capture PP decode steps as CUDA graphs (faster)")
    g.add_argument("--show", type=int, default=5, help="print this many question/answer pairs (default: %(default)s)")
    g.add_argument("--save", help="write the document, questions and answers to this JSON file")
    p.add_argument("--phase", choices=["pp", "stock"], help=argparse.SUPPRESS)
    p.add_argument("--state", help=argparse.SUPPRESS)
    p.add_argument("--stock-backend", choices=list(BASELINES), help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.phase == "pp":
        return run_pp(args, args.state)
    if args.phase == "stock":
        return run_stock(args, args.state)

    # Each engine runs in its own process, one after the other, so they never
    # share GPU memory.
    state = args.save or tempfile.mktemp(suffix=".json")
    base = [sys.executable, os.path.abspath(__file__)] + sys.argv[1:] + ["--state", state]
    if subprocess.run(base + ["--phase", "pp"]).returncode:
        sys.exit("PP run failed")
    for b in args.baselines if args.compare else []:
        if subprocess.run(base + ["--phase", "stock", "--stock-backend", b]).returncode:
            sys.exit(f"stock {b} run failed")
    r = json.load(open(state))
    stock = r.get("stock", {})

    print(f"\n=== Document ({r['doc_tokens']} tokens), first 600 characters\n{r['doc'][:600]} ...")
    print(f"\n=== {len(r['questions'])} questions, first {args.show} with answers")
    for i in range(min(args.show, len(r["questions"]))):
        print(f"\nQ{i + 1}: {r['questions'][i]}\n  PP:    {r['pp_answers'][i].strip()}")
        for b, st in stock.items():
            same = st["ids"][i] == r["pp_ids"][i]
            print(f"  {b + ':':<11} " + ("(identical)" if same else st["answers"][i].strip()))
    print(f"\n=== Answering {len(r['questions'])} questions, up to {args.answer_tokens} tokens each")
    n = len(r["questions"])
    pp_name = "PP, Triton + CUDA graphs" if args.cuda_graph else "PP, Triton, no graphs"
    print(f"{'config':<40}{'time':>8}{'PP speedup':>12}{'same answers as PP':>20}")
    print(f"{pp_name:<40}{r['pp_wall']:>7.2f}s{'':>12}{'':>20}")
    for b, st in stock.items():
        same = sum(a == c for a, c in zip(st["ids"], r["pp_ids"]))
        print(f"{'stock SGLang, ' + BASELINES[b]:<40}{st['wall']:>7.2f}s"
              f"{st['wall'] / r['pp_wall']:>11.2f}x{f'{same}/{n}':>20}")
    if stock:
        print("Answers that differ do so from bf16 rounding: both are greedy decodes of the same prompt.")
    if args.save:
        print(f"saved to {args.save}")
    else:
        os.remove(state)


if __name__ == "__main__":
    main()
