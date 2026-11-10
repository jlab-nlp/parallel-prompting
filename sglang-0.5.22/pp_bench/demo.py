"""Try Parallel Prompting by hand: one document, several questions, answered together.

    python pp_bench/demo.py                                  # built-in passage and questions
    python pp_bench/demo.py --doc-file my.txt -q "Who ...?" -q "When ...?"
    python pp_bench/demo.py --compare                        # also run stock SGLang (no PP)

PP answers every question in one decoding loop: the document is prefilled once,
then each step emits one token per question (stream). --parallel-size P caps how
many questions share one request (default: all of them). --compare runs stock
SGLang without PP (one request per question) on the same prompts, once per
--baselines backend, each with CUDA graphs and in its own subprocess: triton
uses the same kernels as PP, flashinfer is stock's fastest. It prints the
answers, the times, and whether they match.
"""

import argparse
import json
import subprocess
import sys
import time

import torch

from common import DOC_TEMPLATE, QUESTION_TEMPLATE, add_server_args, make_server_args

from sglang.srt.utils.hf_transformers_utils import get_tokenizer

PASSAGE = """\
The Golden Gate Bridge is a suspension bridge that spans the Golden Gate, the
strait connecting San Francisco Bay to the Pacific Ocean. Construction began in
January 1933 and the bridge opened to traffic in May 1937. Its chief engineer was
Joseph Strauss, while much of the structural design is credited to Charles Ellis
and the Art Deco styling to architect Irving Morrow. The main span is 1,280 meters
long, and when the bridge opened it was the longest suspension bridge main span
in the world. The two towers rise 227 meters above the water. The bridge is
painted a color known as International Orange, chosen partly because it stays
visible in the fog that often covers the strait. It carries U.S. Route 101 and
California State Route 1 between San Francisco and Marin County.
"""

QUESTIONS = [
    "When did the bridge open to traffic?",
    "Who was the chief engineer?",
    "How long is the main span?",
    "What is the name of the bridge's color?",
    "Why was that color chosen?",
    "Which two places does the bridge connect?",
    "How tall are the towers?",
    "Who designed the Art Deco styling?",
]


def prompts(tok, doc_text, questions):
    doc = tok.encode(DOC_TEMPLATE.format(doc=doc_text.strip()))
    qs = [tok.encode(QUESTION_TEMPLATE.format(q=q), add_special_tokens=False) for q in questions]
    return doc, qs


def newline_id(tok):
    """Token id of a bare newline, if the tokenizer has one (answers stop there)."""
    ids = tok.encode("\n", add_special_tokens=False)
    return ids[-1] if ids and tok.decode(ids[-1:]) == "\n" else None


def stop_ids(tok, eos_ids, args):
    nl = None if args.no_newline_stop else newline_id(tok)
    return sorted(set(eos_ids) | ({nl} if nl is not None else set()))


def cut_at_stop(ids, stop):
    return ids[: next((k for k, x in enumerate(ids) if x in stop), len(ids))]


BASELINES = {"triton": "Triton + CUDA graphs", "flashinfer": "FlashInfer + CUDA graphs"}


def run_stock(args, doc_text, questions):
    """Stock SGLang: one request per question (the document is a shared prefix)."""
    import sglang as sgl

    tok = get_tokenizer(args.model_path)
    doc, qs = prompts(tok, doc_text, questions)
    engine = sgl.Engine(
        model_path=args.model_path,
        attention_backend=args.stock_backend,
        disable_cuda_graph=False,
        cuda_graph_max_bs_decode=max(len(qs), 1),  # graphs cover the whole batch
        mem_fraction_static=args.mem_fraction_static,
        dtype=args.dtype,
        log_level="error",
    )
    stop = stop_ids(tok, [tok.eos_token_id], args)
    params = {"temperature": 0, "max_new_tokens": args.max_new_tokens, "stop_token_ids": stop}
    engine.generate(input_ids=[doc], sampling_params=dict(params, max_new_tokens=1))  # cache doc
    for _ in range(2):  # the first run warms up kernels; time the second
        tic = time.perf_counter()
        out = engine.generate(input_ids=[doc + q for q in qs], sampling_params=params)
        wall = time.perf_counter() - tic
    engine.shutdown()
    return [cut_at_stop(o["output_ids"], set(stop)) for o in out], wall


def run_pp(args, doc_text, questions):
    from sglang.srt.parallel_prompt.engine import ParallelPromptEngine

    engine = ParallelPromptEngine(make_server_args(args), cuda_graph=args.cuda_graph)
    tok = engine.tokenizer
    doc, qs = prompts(tok, doc_text, questions)
    stop = stop_ids(tok, engine.default_stop_ids(), args)
    P = args.parallel_size or None
    # First run compiles kernels (and captures the CUDA graph); time the second.
    # As on the stock side, the document is prefilled before the clock starts.
    for _ in range(2):
        cache = engine.prefill_docs([doc])
        try:
            torch.cuda.synchronize()
            tic = time.perf_counter()
            out = engine.generate_cached(cache, [qs], args.max_new_tokens, stop, P)
            torch.cuda.synchronize()
            wall = time.perf_counter() - tic
        finally:
            engine.release_docs(cache)
    st = engine.stats
    print(f"[pp] {len(qs)} questions, P={st.parallel_size}, {st.num_reqs} request(s), "
          f"{st.decode_steps} decode steps, {wall:.2f}s", flush=True)
    return [cut_at_stop(a, set(stop)) for a in out[0]], wall, tok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_server_args(parser)
    parser.set_defaults(model_path="codellama/CodeLlama-7b-Instruct-hf")
    for a in parser._actions:  # --model-path has a default here
        if a.dest == "model_path":
            a.required = False
    parser.add_argument("--doc-file", help="text file to use as the document")
    parser.add_argument("-q", "--question", action="append", help="a question (repeatable)")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--parallel-size", type=int, default=0, help="0 = all questions at once")
    parser.add_argument("--cuda-graph", action="store_true", help="capture PP decode as a CUDA graph")
    parser.add_argument("--compare", action="store_true", help="also run stock SGLang (no PP) and diff")
    parser.add_argument("--baselines", nargs="+", default=["triton", "flashinfer"],
                        choices=list(BASELINES),
                        help="stock backends for --compare, each with CUDA graphs (default: both)")
    parser.add_argument("--stock-backend", choices=list(BASELINES), help=argparse.SUPPRESS)
    parser.add_argument("--no-newline-stop", action="store_true",
                        help="let answers run past their first line (default: stop at a newline)")
    parser.add_argument("--phase", choices=["pp", "stock"], default="pp", help=argparse.SUPPRESS)
    args = parser.parse_args()

    doc_text = open(args.doc_file).read() if args.doc_file else PASSAGE
    questions = args.question or QUESTIONS

    if args.phase == "stock":  # child process of --compare
        ans, wall = run_stock(args, doc_text, questions)
        print("STOCK_JSON " + json.dumps({"answers": ans, "wall": wall}))
        return

    stock = {}
    for b in args.baselines if args.compare else []:
        # Each stock engine runs first, in its own process, so no two engines
        # ever share GPU memory.
        cmd = [sys.executable, __file__, "--phase", "stock", "--stock-backend", b] + sys.argv[1:]
        cmd.remove("--compare")
        res = subprocess.run(cmd, capture_output=True, text=True)
        line = next((l for l in res.stdout.splitlines() if l.startswith("STOCK_JSON ")), None)
        if line is None:
            sys.exit(f"stock {b} run failed:\n" + res.stderr[-3000:])
        stock[b] = json.loads(line[len("STOCK_JSON "):])

    answers, wall, tok = run_pp(args, doc_text, questions)
    for i, (q, a) in enumerate(zip(questions, answers)):
        print(f"\nQ{i}: {q}\n  PP:         {tok.decode(a).strip()}")
        for b, st in stock.items():
            s = st["answers"][i]
            print(f"  {b + ':':<11} " + ("(identical)" if s == a else tok.decode(s).strip()))
    if stock:
        n = len(questions)
        print(f"\n{'config':<40}{'time':>8}{'PP speedup':>12}{'same answers as PP':>20}")
        print(f"{'PP, Triton' + (' + CUDA graphs' if args.cuda_graph else ', no graphs'):<40}"
              f"{wall:>7.2f}s")
        for b, st in stock.items():
            same = sum(x == y for x, y in zip(st["answers"], answers))
            print(f"{'stock SGLang, ' + BASELINES[b]:<40}{st['wall']:>7.2f}s"
                  f"{st['wall'] / wall:>11.2f}x{f'{same}/{n}':>20}")


if __name__ == "__main__":
    main()
