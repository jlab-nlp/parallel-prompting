"""Answer quality of Parallel Prompting vs stock SGLang on SQuAD v1.1 dev.

    python pp_bench/eval_squad.py --articles 2 --out pp_bench/out/acc_smoke   # smoke test
    python pp_bench/eval_squad.py --out pp_bench/out/acc_squad_a100           # all 48 articles

Each Wikipedia article (its paragraphs concatenated in order) is one shared
document and all of its questions are the streams. An article longer than
--max-doc-tokens is split into chunks of consecutive paragraphs, and each
question goes to the chunk holding its paragraph. Every config sees the same
token ids (document + question), decodes greedily for at most
--max-new-tokens, and stops at EOS or a newline.

Configs, each in its own subprocess so no two engines share the GPU:
  pp          Parallel Prompting, Triton + CUDA graphs, P = --parallel-size
  flashinfer  stock SGLang, FlashInfer + CUDA graphs, document prefix-cached
  triton      stock SGLang, Triton + CUDA graphs, document prefix-cached

Each config runs the whole workload twice: an untimed warmup pass (kernel
compiles, CUDA graph captures), then the timed pass. Per chunk the document is
prefilled before the clock starts, as in demo.py; the time covers question
prefill plus answer decoding. Writes raw_<config>.jsonl per config, then
per_question.jsonl and summary.md with the official SQuAD v1.1 EM/F1, paired
bootstrap CIs for PP minus stock, and answer agreement.
"""

import argparse
import collections
import json
import os
import random
import re
import string
import subprocess
import sys
import time

from demo import cut_at_stop, newline_id

from sglang.srt.utils.hf_transformers_utils import get_tokenizer

DOC_PROMPT = "[INST] Read the document below.\n\nDocument:\n{doc}\n\n"
QUESTION_PROMPT = (
    "Answer the question with a short phrase copied from the document.\n"
    "Question: {q} [/INST]"
)
CONFIGS = {
    "pp": "PP, Triton + CUDA graphs",
    "flashinfer": "stock SGLang, FlashInfer + CUDA graphs",
    "triton": "stock SGLang, Triton + CUDA graphs",
}


# ------------------------------------------------------------------ workload


def load_articles():
    """SQuAD v1.1 dev as [(title, [paragraph], [(qid, question, answers, paragraph index)])]."""
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad", split="validation")
    arts = collections.OrderedDict()
    for ex in ds:
        title, paras, qs = arts.setdefault(ex["title"], (ex["title"], [], []))
        if ex["context"] not in paras:
            paras.append(ex["context"])
        qs.append((ex["id"], ex["question"], ex["answers"]["text"], paras.index(ex["context"])))
    return list(arts.values())


def pick_articles(articles, n, seed):
    if not n or n >= len(articles):
        return articles
    keep = set(random.Random(seed).sample(range(len(articles)), n))
    return [a for i, a in enumerate(articles) if i in keep]


def build_groups(tok, args):
    """Chunks of consecutive paragraphs (<= max_doc_tokens document tokens), each
    with its questions: [dict(title, chunk, doc ids, questions, question ids)]."""
    articles = pick_articles(load_articles(), args.articles, args.seed)
    groups = []
    for title, paras, qs in articles:
        lens = [len(tok.encode(p + "\n\n", add_special_tokens=False)) for p in paras]
        base = len(tok.encode(DOC_PROMPT.format(doc="")))
        chunks, cur, size = [], [], base
        for i, n in enumerate(lens):
            if cur and size + n > args.max_doc_tokens:
                chunks.append(cur)
                cur, size = [], base
            cur.append(i)
            size += n
        chunks.append(cur)
        chunk_of = {p: c for c, ps in enumerate(chunks) for p in ps}
        for c, ps in enumerate(chunks):
            doc_text = "\n\n".join(paras[p] for p in ps)
            mine = [q for q in qs if chunk_of[q[3]] == c]
            if not mine:
                continue
            groups.append(
                {
                    "title": title,
                    "chunk": c,
                    "n_chunks": len(chunks),
                    "doc": tok.encode(DOC_PROMPT.format(doc=doc_text)),
                    "qs": [
                        tok.encode(QUESTION_PROMPT.format(q=q[1].strip()), add_special_tokens=False)
                        for q in mine
                    ],
                    "meta": [{"id": q[0], "question": q[1], "answers": q[2]} for q in mine],
                }
            )
    return groups


def stop_ids(tok, eos):
    nl = newline_id(tok)
    return sorted(set(eos) | ({nl} if nl is not None else set()))


# ------------------------------------------------------------------ configs


def run_pp(args, groups):
    import torch

    from common import make_server_args
    from sglang.srt.parallel_prompt.engine import ParallelPromptEngine

    # As in bench_pp_vs_sglang.py: a PP request holds the document plus
    # P * (question + output) slots in one req_to_token row, sized from the
    # context length; positions stay below document + one stream.
    os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    args.context_length = args.pp_row_len
    engine = ParallelPromptEngine(make_server_args(args), cuda_graph=True)
    tok = engine.tokenizer
    stop = stop_ids(tok, engine.default_stop_ids())
    passes = []
    for p in range(2):  # warmup, timed
        outs, wall = [], 0.0
        for g in groups:
            cache = engine.prefill_docs([g["doc"]])
            try:
                torch.cuda.synchronize()
                tic = time.perf_counter()
                out = engine.generate_cached(
                    cache, [g["qs"]], args.max_new_tokens, stop, args.parallel_size
                )
                torch.cuda.synchronize()
                wall += time.perf_counter() - tic
            finally:
                engine.release_docs(cache)
            outs.append([cut_at_stop(a, set(stop)) for a in out[0]])
        print(f"[pp] pass {p}: {wall:.2f}s", flush=True)
        passes.append((outs, wall))
    return passes


def run_stock(args, groups, backend):
    import sglang as sgl

    tok = get_tokenizer(args.model_path)
    engine = sgl.Engine(
        model_path=args.model_path,
        attention_backend=backend,
        disable_cuda_graph=False,
        cuda_graph_max_bs_decode=max(len(g["qs"]) for g in groups),  # whole batch
        mem_fraction_static=args.mem_fraction_static,
        dtype=args.dtype,
        log_level="error",
    )
    stop = stop_ids(tok, [tok.eos_token_id])
    params = {"temperature": 0, "max_new_tokens": args.max_new_tokens, "stop_token_ids": stop}
    passes = []
    for p in range(2):  # warmup, timed
        engine.flush_cache()  # the timed pass must not reuse the warmup's questions
        outs, wall = [], 0.0
        for g in groups:
            engine.generate(input_ids=[g["doc"]], sampling_params=dict(params, max_new_tokens=1))
            tic = time.perf_counter()
            out = engine.generate(input_ids=[g["doc"] + q for q in g["qs"]], sampling_params=params)
            wall += time.perf_counter() - tic
            outs.append([cut_at_stop(o["output_ids"], set(stop)) for o in out])
        print(f"[{backend}] pass {p}: {wall:.2f}s", flush=True)
        passes.append((outs, wall))
    engine.shutdown()
    return passes


def run_phase(args):
    tok = get_tokenizer(args.model_path)
    groups = build_groups(tok, args)
    passes = run_pp(args, groups) if args.phase == "pp" else run_stock(args, groups, args.phase)
    (warm, _), (outs, wall) = passes
    n_same = sum(a == b for gw, go in zip(warm, outs) for a, b in zip(gw, go))
    path = os.path.join(args.out, f"raw_{args.phase}.jsonl")
    with open(path, "w") as f:
        for g, go in zip(groups, outs):
            for m, ids in zip(g["meta"], go):
                f.write(json.dumps({"id": m["id"], "title": g["title"], "chunk": g["chunk"],
                                    "doc_tokens": len(g["doc"]), "n_streams": len(g["qs"]),
                                    "output_ids": ids,
                                    "answer": tok.decode(ids, skip_special_tokens=True).strip()})
                        + "\n")
    with open(os.path.join(args.out, f"time_{args.phase}.json"), "w") as f:
        json.dump({"config": args.phase, "wall_s": wall, "warmup_wall_s": passes[0][1],
                   "warmup_vs_timed_identical": n_same,
                   "n": sum(len(g["qs"]) for g in groups)}, f)


# ------------------------------------------------------------------ metrics


def normalize_answer(s):
    """Official SQuAD v1.1 normalization."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred, gt):
    p, g = normalize_answer(pred).split(), normalize_answer(gt).split()
    common = collections.Counter(p) & collections.Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def em_score(pred, gt):
    return float(normalize_answer(pred) == normalize_answer(gt))


def summarize(args, groups, configs):
    import numpy as np

    raw = {}
    for c in configs:
        with open(os.path.join(args.out, f"raw_{c}.jsonl")) as f:
            raw[c] = {r["id"]: r for r in map(json.loads, f)}
    times = {c: json.load(open(os.path.join(args.out, f"time_{c}.json"))) for c in configs}
    metas = [m for g in groups for m in g["meta"]]
    ids = [m["id"] for m in metas]
    em = {c: np.array([max(em_score(raw[c][m["id"]]["answer"], a) for a in m["answers"])
                       for m in metas]) for c in configs}
    f1 = {c: np.array([max(f1_score(raw[c][m["id"]]["answer"], a) for a in m["answers"])
                       for m in metas]) for c in configs}
    with open(os.path.join(args.out, "per_question.jsonl"), "w") as f:
        for i, m in enumerate(metas):
            r = {"id": m["id"], "question": m["question"], "gold": m["answers"],
                 "title": raw[configs[0]][m["id"]]["title"]}
            for c in configs:
                r[c] = {"answer": raw[c][m["id"]]["answer"], "em": em[c][i], "f1": f1[c][i]}
            f.write(json.dumps(r) + "\n")

    n = len(ids)
    rng = np.random.default_rng(args.seed)
    boot = rng.integers(0, n, size=(args.bootstrap, n))
    gpu = times[configs[0]].get("gpu", args.gpu)
    lines = [
        "# SQuAD v1.1 dev: Parallel Prompting vs stock SGLang 0.5.22", "",
        f"- GPU: {gpu}", f"- Model: {args.model_path} ({args.dtype})",
        f"- Articles: {len({g['title'] for g in groups})}, document chunks: {len(groups)}, "
        f"questions: {n}" + ("" if not args.articles else
                             f" (subset of {args.articles} articles, seed {args.seed})"),
        f"- Document tokens per chunk: max {max(len(g['doc']) for g in groups)}, "
        f"mean {np.mean([len(g['doc']) for g in groups]):.0f}; streams per chunk: "
        f"max {max(len(g['qs']) for g in groups)}, mean {np.mean([len(g['qs']) for g in groups]):.0f}",
        f"- Chunk limit {args.max_doc_tokens} document tokens; PP parallel size P = {args.parallel_size}",
        f"- Greedy, max {args.max_new_tokens} new tokens, stop at EOS or newline",
        "- Prompt: `" + DOC_PROMPT.replace("\n", "\\n") + "` + `"
        + QUESTION_PROMPT.replace("\n", "\\n") + "`", "",
        "| config | EM | F1 | answer time (s) | identical to PP |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in configs:
        same = sum(raw[c][i]["output_ids"] == raw["pp"][i]["output_ids"] for i in ids) \
            if "pp" in raw else 0
        lines.append(f"| {CONFIGS[c]} | {100 * em[c].mean():.2f} | {100 * f1[c].mean():.2f} | "
                     f"{times[c]['wall_s']:.2f} | "
                     + ("-" if c == "pp" else f"{same}/{n} ({100 * same / n:.1f}%)") + " |")
    if "pp" in raw:
        lines += ["", f"Paired bootstrap over questions ({args.bootstrap} resamples), "
                  "PP minus stock, in points:", "",
                  "| stock config | F1 diff [95% CI] | EM diff [95% CI] |", "|---|---:|---:|"]
        for c in configs:
            if c == "pp":
                continue
            cells = []
            for m in (f1, em):
                d = m["pp"] - m[c]
                bs = d[boot].mean(1)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                cells.append(f"{100 * d.mean():+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]")
            lines.append(f"| {CONFIGS[c]} | {cells[0]} | {cells[1]} |")
    lines += ["", "Determinism: timed pass identical to warmup pass for " + ", ".join(
        f"{c} {times[c]['warmup_vs_timed_identical']}/{n}" for c in configs) + "."]
    if args.articles:
        lines += ["", "Articles: " + ", ".join(sorted({g["title"] for g in groups}))]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(text)
    print(text)


# ------------------------------------------------------------------ main


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-path", default="codellama/CodeLlama-7b-Instruct-hf")
    parser.add_argument("--mem-fraction-static", type=float, default=0.8)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--out", required=True)
    parser.add_argument("--articles", type=int, default=0, help="0 = all 48 dev articles")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-doc-tokens", type=int, default=8000)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--parallel-size", type=int, default=32)
    parser.add_argument("--pp-row-len", type=int, default=32768)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--phase", choices=list(CONFIGS), help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.gpu = ""
    os.makedirs(args.out, exist_ok=True)

    if args.phase:
        run_phase(args)
        return

    if not args.summarize_only:
        import torch

        args.gpu = torch.cuda.get_device_name(0)
        with open(os.path.join(args.out, "gpu.txt"), "w") as f:
            f.write(args.gpu + "\n")
        # Stock configs first, then PP, each in its own process.
        for c in sorted(args.configs, key=lambda c: c == "pp"):
            cmd = [sys.executable, os.path.abspath(__file__), "--phase", c] + sys.argv[1:]
            tic = time.perf_counter()
            res = subprocess.run(cmd, capture_output=True, text=True)
            with open(os.path.join(args.out, f"log_{c}.txt"), "w") as f:
                f.write(res.stdout + "\n--- stderr ---\n" + res.stderr)
            if res.returncode:
                sys.exit(f"{c} failed (see log_{c}.txt):\n" + res.stderr[-3000:])
            print(f"{c}: done in {time.perf_counter() - tic:.0f}s", flush=True)
    elif os.path.exists(os.path.join(args.out, "gpu.txt")):
        args.gpu = open(os.path.join(args.out, "gpu.txt")).read().strip()
    tok = get_tokenizer(args.model_path)
    summarize(args, build_groups(tok, args), args.configs)


if __name__ == "__main__":
    main()
