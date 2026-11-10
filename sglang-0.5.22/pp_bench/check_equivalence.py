"""End-to-end check (SGLang v0.5.22): Parallel Prompting vs. stock SGLang.

The reference is the unmodified serving path: `sgl.Engine` with the Triton
backend greedy-decodes every [Doc + q_i] as its own request. It runs in a child
process first (so its GPU memory is released), then the PP engine runs here.
With exact masking the outputs should match token-for-token, up to bf16 noise.

    python pp_bench/check_equivalence.py --model-path codellama/CodeLlama-7b-Instruct-hf
"""

import argparse
import json
import subprocess
import sys
import tempfile

from common import add_server_args, make_server_args, squad_groups, synthetic_groups

from sglang.srt.utils.hf_transformers_utils import get_tokenizer


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def workload(args, tok):
    if args.workload == "squad":
        return squad_groups(tok, args.groups, args.max_questions)
    return synthetic_groups(tok, args.groups, args.max_questions, args.doc_tokens)


def run_stock(args):
    import sglang as sgl

    tok = get_tokenizer(args.model_path)
    docs, questions = workload(args, tok)
    prompts = [d + q for d, qs in zip(docs, questions) for q in qs]
    engine = sgl.Engine(
        model_path=args.model_path,
        attention_backend="triton",
        mem_fraction_static=args.mem_fraction_static,
        dtype=args.dtype,
        log_level="error",
    )
    stop = sorted({tok.eos_token_id})
    out = engine.generate(
        input_ids=prompts,
        sampling_params={"temperature": 0, "max_new_tokens": args.max_new_tokens},
    )
    engine.shutdown()
    ref = []
    for o in out:
        ids = o["output_ids"]
        cut = next((k + 1 for k, x in enumerate(ids) if x in stop), len(ids))
        ref.append(ids[:cut])
    with open(args.ref_out, "w") as f:
        json.dump(ref, f)


def main():
    parser = argparse.ArgumentParser()
    add_server_args(parser)
    parser.add_argument("--workload", choices=["squad", "synthetic"], default="squad")
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--max-questions", type=int, default=8)
    parser.add_argument("--doc-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--parallel-size", type=int, nargs="+", default=[0],
                        help="streams per request; 0 = all questions of a group")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--no-compact", action="store_true",
                        help="keep finished requests in the decode batch (earlier behaviour)")
    parser.add_argument("--prefix-cache", action="store_true",
                        help="PP engine with SGLang's radix prefix cache over documents; "
                        "later parallel sizes then reuse the cached documents")
    parser.add_argument("--engine", choices=["pp", "cb"], default="pp",
                        help="cb: continuous batching (parallel_prompt/continuous.py)")
    parser.add_argument("--show", type=int, default=3)
    parser.add_argument("--out", default=None)
    parser.add_argument("--phase", choices=["both", "stock"], default="both")
    parser.add_argument("--ref-out", default=None)
    args = parser.parse_args()

    if args.phase == "stock":
        run_stock(args)
        return

    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        subprocess.run(
            [sys.executable, __file__, *sys.argv[1:], "--phase", "stock", "--ref-out", f.name],
            check=True,
        )
        ref = json.load(open(f.name))

    from sglang.srt.parallel_prompt.continuous import ContinuousPPEngine
    from sglang.srt.parallel_prompt.engine import ParallelPromptEngine

    cls = {"pp": ParallelPromptEngine, "cb": ContinuousPPEngine}[args.engine]
    engine = cls(make_server_args(args), cuda_graph=args.cuda_graph,
                 prefix_cache=args.prefix_cache)
    engine.compact_finished = not args.no_compact
    tok = engine.tokenizer
    docs, questions = workload(args, tok)
    for p in args.parallel_size:
        hits = engine.prefix_hit_tokens
        pp = engine.generate(docs, questions, args.max_new_tokens, parallel_size=p or None)
        if args.prefix_cache:
            print(f"prefix-cache hit tokens: {engine.prefix_hit_tokens - hits} of "
                  f"{sum(len(d) for d in docs)} document tokens")
        print(f"== {args.engine} parallel size {p or 'all'}  cuda graph {args.cuda_graph}  "
              f"batch compactions so far {engine.compactions}")
        compare(tok, [o for g in pp for o in g], ref, args)


def compare(tok, pp_flat, ref, args):
    exact, divs, records = 0, [], []
    for i, (a, b) in enumerate(zip(pp_flat, ref)):
        d = first_divergence(a, b)
        exact += d is None
        if d is not None:
            divs.append(d)
        records.append(
            {"i": i, "pp": tok.decode(a), "ref": tok.decode(b), "first_divergence": d}
        )
    n = len(ref)
    print(f"streams: {n}  exact match: {exact}/{n} ({100 * exact / n:.1f}%)")
    if divs:
        divs.sort()
        print(f"first divergence (tokens) min/median: {divs[0]}/{divs[len(divs) // 2]}")
    for r in records[: args.show]:
        print(json.dumps(r, ensure_ascii=False))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(records, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
