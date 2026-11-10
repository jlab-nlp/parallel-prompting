"""Speed sweep: stock SGLang vs. Parallel Prompting on identical workloads.

Six configurations, all with the documents already in the KV cache when the
timer starts (stock: radix cache pre-filled by a 1-token request per document;
PP: documents prefilled once and shared by every request of their group):

    flashinfer        stock SGLang, FlashInfer attention, no CUDA graphs
    flashinfer-graph  stock SGLang, FlashInfer attention, CUDA graphs
    triton            stock SGLang, Triton attention, no CUDA graphs
    triton-graph      stock SGLang, Triton attention, CUDA graphs
    pp                Parallel Prompting, Triton attention, no CUDA graphs
    pp-graph          Parallel Prompting, Triton attention, CUDA graphs

Every stream emits exactly --output-len tokens (EOS ignored). Each run uses
fresh documents drawn from a seed (doc length, Q, rep), so the same rep gives
every config the same inputs (paired comparison). Per workload:

    rep 0         short untimed warmup (PP: one per parallel size P, which also
                  captures that P's CUDA graph)
    rep 1         PP only: calibration. Every P is timed once ("phase": "calib")
                  and the fastest is kept.
    --eval-reps   timed runs ("phase": "eval"); PP runs only the calibrated P,
                  so the reported P is never chosen on the runs it is scored on.

Without --eval-reps, reps 2..--repeats+1 are used. Runs already in --out are
skipped, so a killed sweep can be resumed. One JSON line per timed run.

    python pp_bench/bench_pp_vs_sglang.py --config pp-graph --model-path ...
"""

import argparse
import json
import os
import time

from common import add_server_args, bench_groups, make_server_args

from sglang.srt.hf_transformers_utils import get_tokenizer

STOCK = {
    "flashinfer": ("flashinfer", False),
    "flashinfer-graph": ("flashinfer", True),
    "triton": ("triton", False),
    "triton-graph": ("triton", True),
}


def seed_of(doc_tokens, n_q, rep):
    return doc_tokens * 1000 + n_q * 10 + rep


WARMUP_LEN = 8


def run_stock(args, tok, emit):
    import sglang as sgl

    backend, graph = STOCK[args.config]
    max_bs = args.groups * max(args.questions)
    engine = sgl.Engine(
        model_path=args.model_path,
        attention_backend=backend,
        disable_cuda_graph=not graph,
        cuda_graph_max_bs=max_bs if graph else None,
        mem_fraction_static=args.mem_fraction_static,
        context_length=args.context_length,
        dtype=args.dtype,
        log_level="error",
    )
    params = {"temperature": 0, "max_new_tokens": args.output_len, "ignore_eos": True}
    for doc_tokens in args.doc_tokens:
        for n_q in args.questions:
            for rep in [0] + args.eval_reps:
                if rep and emit.done(doc_tokens, n_q, rep, "eval", None):
                    continue
                docs, qs = bench_groups(tok, args.groups, n_q, doc_tokens, seed_of(doc_tokens, n_q, rep))
                prompts = [d + q for d, g in zip(docs, qs) for q in g]
                engine.generate(input_ids=docs, sampling_params=dict(params, max_new_tokens=1))
                run_params = dict(params, max_new_tokens=WARMUP_LEN) if rep == 0 else params
                tic = time.perf_counter()
                out = engine.generate(input_ids=prompts, sampling_params=run_params)
                wall = time.perf_counter() - tic
                if rep == 0:
                    continue  # warmup
                emit(doc_tokens, n_q, rep, wall,
                     sum(o["meta_info"]["completion_tokens"] for o in out), {"phase": "eval"})
    engine.shutdown()


def run_pp(args, tok, emit):
    import torch

    from sglang.srt.parallel_prompt.engine import ParallelPromptEngine

    # A request holds the document plus P * (question + output) KV slots in one
    # req_to_token row, which SGLang sizes to the context length; position ids
    # stay below the document + one stream, so only the row width grows. RoPE is
    # sized from the HF config, not from this.
    if args.pp_row_len:
        os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
        args.context_length = args.pp_row_len
    engine = ParallelPromptEngine(make_server_args(args), cuda_graph=args.config == "pp-graph")

    def timed(doc_tokens, n_q, rep, p, out_len):
        docs, qs = bench_groups(tok, args.groups, n_q, doc_tokens, seed_of(doc_tokens, n_q, rep))
        cache = engine.prefill_docs(docs)
        try:
            torch.cuda.synchronize()
            tic = time.perf_counter()
            out = engine.generate_cached(cache, qs, out_len, stop_token_ids=[], parallel_size=p)
            torch.cuda.synchronize()
            return time.perf_counter() - tic, sum(len(o) for g in out for o in g)
        finally:
            engine.release_docs(cache)

    # The engine runs requests in waves when the KV pool is short, and the wave
    # size depends on the output length; warming up at full length captures the
    # CUDA graphs of the timed run's wave shapes before the clock starts.
    warmup_len = args.output_len if args.warmup_full else WARMUP_LEN

    def extra(p, phase):
        s = engine.stats
        return {"phase": phase, "parallel_size": p, "num_reqs": s.num_reqs,
                "prefill_s": s.prefill_s, "decode_s": s.decode_s,
                "step_ms": 1000 * s.decode_s / max(s.decode_steps, 1)}

    for doc_tokens in args.doc_tokens:
        for n_q in args.questions:
            sizes = [p for p in args.parallel_sizes if p <= n_q and n_q % p == 0]
            calib = {}
            for p in sizes:
                prev = emit.done(doc_tokens, n_q, 1, "calib", p)
                if prev:
                    calib[p] = prev["wall_s"]
                    continue
                try:
                    timed(doc_tokens, n_q, 0, p, warmup_len)  # warmup + graph capture
                    wall, n = timed(doc_tokens, n_q, 1, p, args.output_len)
                except MemoryError as e:
                    print(f"skip doc={doc_tokens} Q={n_q} P={p}: {e}")
                    continue
                calib[p] = wall
                emit(doc_tokens, n_q, 1, wall, n, extra(p, "calib"))
            if not calib:
                continue
            best = min(calib, key=calib.get)
            print(f"doc={doc_tokens} Q={n_q}: calibrated P={best}", flush=True)
            timed(doc_tokens, n_q, 0, best, warmup_len)
            for rep in args.eval_reps:
                if emit.done(doc_tokens, n_q, rep, "eval", best):
                    continue
                wall, n = timed(doc_tokens, n_q, rep, best, args.output_len)
                emit(doc_tokens, n_q, rep, wall, n, extra(best, "eval"))


def main():
    parser = argparse.ArgumentParser()
    add_server_args(parser)
    parser.add_argument("--config", choices=list(STOCK) + ["pp", "pp-graph"], required=True)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--questions", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--doc-tokens", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--parallel-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--warmup-full", action="store_true",
                        help="PP warmup at the full output length (for runs that need waves)")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--eval-reps", type=int, nargs="+", default=None)
    parser.add_argument("--pp-row-len", type=int, default=65536,
                        help="PP only: req_to_token row width (0 = context length)")
    parser.add_argument("--out", default="pp_bench/results.jsonl")
    args = parser.parse_args()

    if args.eval_reps is None:
        args.eval_reps = list(range(2, args.repeats + 2))
    assert min(args.eval_reps) >= 2, "reps 0 and 1 are warmup and calibration"
    tok = get_tokenizer(args.model_path)

    done = {}
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            key = (r["config"], r["groups"], r["output_len"], r["doc_tokens"], r["questions"],
                   r["rep"], r.get("phase"), r.get("parallel_size"))
            done[key] = r

    def emit(doc_tokens, n_q, rep, wall, tokens, extra):
        res = dict(config=args.config, model=args.model_path, groups=args.groups,
                   doc_tokens=doc_tokens, questions=n_q, output_len=args.output_len,
                   rep=rep, wall_s=wall, output_tokens=tokens,
                   tokens_per_s=tokens / wall, **extra)
        print(json.dumps(res), flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(res) + "\n")

    def is_done(doc_tokens, n_q, rep, phase, p):
        hit = done.get((args.config, args.groups, args.output_len, doc_tokens, n_q, rep, phase, p))
        if hit is None and phase == "eval" and p is not None:
            # PP eval runs at the calibrated P; any eval P counts as done.
            hit = next((r for k, r in done.items()
                        if k[:7] == (args.config, args.groups, args.output_len,
                                     doc_tokens, n_q, rep, phase)), None)
        return hit

    emit.done = is_done

    (run_pp if args.config.startswith("pp") else run_stock)(args, tok, emit)


if __name__ == "__main__":
    main()
