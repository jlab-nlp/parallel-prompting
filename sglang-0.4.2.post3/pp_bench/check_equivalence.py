"""End-to-end check: Parallel Prompting outputs vs. independent decoding.

Both use the same ModelRunner and greedy decoding. The reference runs every
[Doc + q_i] as its own request through the unmodified SGLang extend/decode path.
With exact masking the outputs should match token-for-token, up to bf16 noise.

    python pp_bench/check_equivalence.py --model-path codellama/CodeLlama-7b-Instruct-hf
"""

import argparse
import json

from common import add_server_args, make_server_args, squad_groups, synthetic_groups

from sglang.srt.parallel_prompt.engine import ParallelPromptEngine


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


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
    parser.add_argument("--show", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    engine = ParallelPromptEngine(make_server_args(args), cuda_graph=args.cuda_graph)
    tok = engine.tokenizer
    if args.workload == "squad":
        docs, questions = squad_groups(tok, args.groups, args.max_questions)
    else:
        docs, questions = synthetic_groups(
            tok, args.groups, args.max_questions, args.doc_tokens
        )

    prompts = [d + q for d, qs in zip(docs, questions) for q in qs]
    ref = engine.generate_independent(prompts, args.max_new_tokens)
    for p in args.parallel_size:
        pp = engine.generate(docs, questions, args.max_new_tokens, parallel_size=p or None)
        print(f"== parallel size {p or 'all'}  cuda graph {args.cuda_graph}")
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
