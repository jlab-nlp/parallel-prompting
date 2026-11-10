"""Real-benchmark runs: wall time and answer quality per engine, on the work
files written by bench_data.py (QuALITY, LooGLE, ...) or qa_data.py (SQuAD, QuAC, DROP).

One config, one timed pass per process, so every pass starts with a cold
prefix cache:

    load engine -> warmup on the train-split calibration set -> clear the
    prefix cache where the engine can -> timed pass over the full validation
    set (token ids in, answer ids out, all prefill included)

Configs (the workload, prompts and decoding are identical for all):

    sgl-on / sgl-off        stock SGLang, FlashInfer, CUDA graphs, radix cache on / off
    sgl-triton-on           stock SGLang, Triton (PP's backend), radix cache on
    pp-on                   Parallel Prompting, Triton + CUDA graphs; each document
                            is prefilled once and shared by all its requests
    pp-off                  same requests, but every request of P questions
                            prefills its own copy of the document
    pp-radix                pp-on plus SGLang's radix prefix cache over documents:
                            a prefix shared across documents (instruction and
                            few-shot block) is prefilled once (cache flushed
                            before every timed pass)
    pp-cb                   pp-on with continuous batching (parallel_prompt/
                            continuous.py): finished requests are retired and
                            waiting ones admitted at every stop check, no waves
    pp-cb-radix             pp-cb plus the radix prefix cache over documents

PP-family configs also run one untimed pass over the validation set before the
timed one; it captures the CUDA graph of every batch shape the timed pass uses
(stock engines capture theirs at startup). They hold no state across passes.

    python pp_bench/eval_qa.py --work <out>/work/<model>/quality.json.gz \\
        --model-path <path> --config pp-radix --pass-index 1 --out <dir>
"""

import argparse
import json
import os
import subprocess
import sys
import time

from qa_data import load_work

PP_CONFIGS = {"pp-on", "pp-off", "pp-radix", "pp-cb", "pp-cb-radix"}
CONFIGS = ["sgl-on", "sgl-off", "sgl-triton-on"] + sorted(PP_CONFIGS)
CALIB_P = [4, 8, 16, 0]  # 0 = all questions of a document in one request


def prompts(groups):
    return [g["doc"] + q for g in groups for q in g["qs"]]


def qids(groups):
    return [i for g in groups for i in g["qids"]]


# ------------------------------------------------------------------ SGLang


def run_sgl(args, work):
    import sglang as sgl

    mem = {"mem_fraction_static": args.mem_fraction_static} if args.mem_fraction_static else {}
    if args.sgl_max_running:
        mem["max_running_requests"] = args.sgl_max_running
    engine = sgl.Engine(model_path=args.model_path, dtype=args.dtype, log_level="error",
                        attention_backend="triton" if "triton" in args.config else "flashinfer",
                        disable_radix_cache=args.config == "sgl-off", **mem)
    params = {"temperature": 0, "top_p": 1, "max_new_tokens": work["max_new_tokens"]}

    def gen(groups):
        return [o["output_ids"] for o in engine.generate(input_ids=prompts(groups),
                                                         sampling_params=params)]

    gen(work["calib_groups"])
    engine.flush_cache()
    tic = time.perf_counter()
    outs = gen(work["groups"])
    wall = time.perf_counter() - tic
    engine.shutdown()
    return outs, wall, {"engine": f"sglang {sgl.__version__}", "cache_flushed": True,
                        "max_running_requests": args.sgl_max_running or "sglang default"}


# ------------------------------------------------------------------ PP family


def calls(groups, P, pool, max_q_tokens):
    """Split groups into engine calls whose documents fit in 40% of the KV pool
    and whose question prefill (padded to P streams) stays below max_q_tokens."""
    out, cur, doc_t, q_t = [], [], 0, 0
    for g in groups:
        p = P or len(g["qs"])
        n_req = -(-len(g["qs"]) // p)
        gq = sum(len(q) for q in g["qs"]) + (n_req * p - len(g["qs"])) * len(g["qs"][-1])
        if cur and (doc_t + len(g["doc"]) > 0.4 * pool or q_t + gq > max_q_tokens):
            out.append(cur)
            cur, doc_t, q_t = [], 0, 0
        cur.append(g)
        doc_t += len(g["doc"])
        q_t += gq
    return out + ([cur] if cur else [])


def split_per_request(groups, P):
    """pp-off: one group per request of P questions, each with its own document."""
    out = []
    for g in groups:
        k = P or len(g["qs"])
        out += [{"doc": g["doc"], "qs": g["qs"][c : c + k], "qids": g["qids"][c : c + k]}
                for c in range(0, len(g["qs"]), k)]
    return out


def run_pp(args, work):
    import torch

    import sglang
    from common import make_server_args
    from sglang.srt.parallel_prompt.engine import DocCache, ParallelPromptEngine

    os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    args.context_length = args.pp_row_len
    cls = ParallelPromptEngine
    if args.config.startswith("pp-cb"):
        from sglang.srt.parallel_prompt.continuous import ContinuousPPEngine as cls
    engine = cls(make_server_args(args), cuda_graph=True,
                 prefix_cache=args.config.endswith("radix"))
    engine.max_prefill_tokens = args.max_prefill_tokens  # question prefill chunks too
    engine.compact_finished = not args.no_compact
    stop = engine.default_stop_ids()
    steps = work["max_new_tokens"]
    pool = engine.kv_alloc.available_size()

    def prefill(docs):
        lens, slots, nodes, chunk = [], [], [], []
        for d in docs + [None]:
            if chunk and (d is None or sum(map(len, chunk)) + len(d) > args.max_prefill_tokens):
                c = engine.prefill_docs(chunk)
                lens += c.lens
                slots += c.slots
                nodes += c.nodes  # prefix-cache locks, released with the merged cache
                chunk = []
            if d is not None:
                chunk.append(d)
        return DocCache(lens=lens, slots=slots, nodes=nodes)

    def gen(groups, P):
        # Requests are padded to P streams; ordering groups by question count
        # keeps P = all (0) from padding small documents up to large ones.
        order = sorted(range(len(groups)), key=lambda i: -len(groups[i]["qs"]))
        gs = [groups[i] for i in order]
        if args.config == "pp-off":
            gs = split_per_request(gs, P)
        outs = {}
        engine.flush_prefix_cache()  # every pass starts cold
        hits0 = engine.prefix_hit_tokens
        sched = dict(calls=0, reqs=0, waves=0, refills=0, decode_steps=0, active_req_steps=0,
                     row_steps=0, batch_cap=[])
        torch.cuda.synchronize()
        tic = time.perf_counter()
        for call in calls(gs, P, pool, args.max_prefill_tokens):
            cache = prefill([g["doc"] for g in call])
            try:
                res = engine.generate_cached(cache, [g["qs"] for g in call], steps,
                                             stop_token_ids=stop, parallel_size=P or None)
            finally:
                engine.release_docs(cache)
            s = engine.stats
            sched["calls"] += 1
            sched["reqs"] += s.num_reqs
            sched["batch_cap"].append(s.batch_cap)
            for k in ("waves", "refills", "decode_steps", "active_req_steps", "row_steps"):
                sched[k] += getattr(s, k)
            for g, r in zip(call, res):
                outs.update(zip(g["qids"], r))
        torch.cuda.synchronize()
        gen.hit_tokens = engine.prefix_hit_tokens - hits0
        gen.sched = sched
        return [outs[i] for i in qids(groups)], time.perf_counter() - tic

    if args.parallel_size is not None:
        cands = [args.parallel_size]
    else:
        cands = CALIB_P
    def fits(P):
        # The timed groups can be larger than the calibration groups, so check
        # the row against them.
        row = engine.req_to_token.shape[1]
        for g in work["groups"]:
            n = min(P, len(g["qs"])) if P else len(g["qs"])
            q = sorted(map(len, g["qs"]), reverse=True)[:n]
            if len(g["doc"]) + sum(q) + n * steps > row:
                return False
        return True

    calib = {}
    for P in cands:
        if not fits(P):
            print(f"skip P={P}: the timed groups do not fit one request row", flush=True)
            continue
        try:
            gen(work["calib_groups"], P)  # warmup, graph capture
            if len(cands) > 1:  # a fixed P (--parallel-size) is not calibrated
                calib[P] = gen(work["calib_groups"], P)[1]
            else:
                calib[P] = None
        except MemoryError as e:  # P streams do not fit one request row
            print(f"skip P={P}: {e}", flush=True)
    if not calib:
        raise MemoryError("no parallel size fits; raise --pp-row-len")
    P = min(calib, key=lambda k: calib[k] or 0)

    # Untimed warm pass over the timed groups: captures the timed pass's batch shapes.
    gen(work["groups"], P)
    outs, wall = gen(work["groups"], P)
    return outs, wall, {"engine": f"sglang {sglang.__version__} + {cls.__name__}",
                        "cache_flushed": True, "parallel_size": P,
                        "parallel_size_fixed": args.parallel_size is not None,
                        "compact_finished": engine.compact_finished,
                        "prefix_hit_tokens": gen.hit_tokens if engine.tree is not None else None,
                        "doc_tokens": sum(len(g["doc"]) for g in work["groups"]),
                        "sched": gen.sched,
                        "calib_s": {str(k): v for k, v in calib.items() if v is not None}}


# ------------------------------------------------------------------ main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config", required=True, choices=CONFIGS)
    parser.add_argument("--label", default=None, help="file label (default: config)")
    parser.add_argument("--pass-index", type=int, default=1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mem-fraction-static", type=float, default=0.88)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--parallel-size", type=int, default=None,
                        help="PP: fixed P (0 = all); default: calibrate over 4/8/16/all")
    parser.add_argument("--max-prefill-tokens", type=int, default=8192,
                        help="PP: tokens per prefill forward (stock engines chunk prefill too)")
    parser.add_argument("--pp-row-len", type=int, default=32768)
    parser.add_argument("--limit-groups", type=int, default=0, help="smoke test: first N groups")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-compact", action="store_true",
                        help="PP: keep finished requests in the decode batch until their wave ends")
    parser.add_argument("--sgl-max-running", type=int, default=0,
                        help="stock SGLang: max_running_requests (0 = engine default)")
    args = parser.parse_args()

    if args.config.startswith("sgl") and not args.child:
        # Stock SGLang can run out of GPU memory at the shared memory fraction
        # (e.g. Qwen's 152k-token vocabulary on large batches); an OOM kills
        # the process, so retry once in a new one with SGLang's own default
        # fraction (0 = engine default), which is recorded in the result.
        # If that also fails (Qwen on a 48 GB A6000: fp32 logits of ~4k running
        # requests over 152k tokens), cap running requests at 1024, a usual
        # serving setting.
        cmd = [sys.executable, os.path.abspath(__file__)] + sys.argv[1:] + ["--child"]
        rc = 1
        for extra in ([], ["--mem-fraction-static", "0"],
                      ["--mem-fraction-static", "0", "--sgl-max-running", "1024"]):
            if extra:
                print(f"retrying {args.config} with {' '.join(extra)}", flush=True)
            rc = subprocess.run(cmd + extra).returncode
            if rc == 0:
                break
        sys.exit(rc)

    work = load_work(args.work)
    if args.limit_groups:  # smoke tests
        work["groups"] = work["groups"][: args.limit_groups]
        work["calib_groups"] = work["calib_groups"][: args.limit_groups]
    label = args.label or args.config
    os.makedirs(args.out, exist_ok=True)
    run = run_pp if args.config in PP_CONFIGS else run_sgl
    outs, wall, meta = run(args, work)

    import torch

    ids = qids(work["groups"])
    assert len(outs) == len(ids), (len(outs), len(ids))
    stem = os.path.join(args.out, f"{label}.p{args.pass_index}")
    with open(stem + ".raw.jsonl", "w") as f:
        for i, o in zip(ids, outs):
            f.write(json.dumps({"id": i, "output_ids": [int(x) for x in o]}) + "\n")
    res = dict(meta, label=label, config=args.config, dataset=work["dataset"],
               model=args.model_path, pass_index=args.pass_index, wall_s=wall, n=len(ids),
               output_tokens=sum(len(o) for o in outs),
               gpu=torch.cuda.get_device_name(0),
               mem_fraction_static=args.mem_fraction_static or "sglang default",
               max_prefill_tokens=args.max_prefill_tokens if args.config in PP_CONFIGS else None)
    with open(stem + ".time.json", "w") as f:
        json.dump(res, f)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
