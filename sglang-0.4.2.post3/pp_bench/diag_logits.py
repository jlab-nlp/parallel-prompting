"""Diagnose PP vs. independent divergences by comparing next-token logits.

Teacher-forces both paths on the same token sequences: the PP engine decodes
greedily, and the reference re-runs each [Doc + q_i + answer prefix] from
scratch, so every step compares logits for identical contexts. A masking or
position bug shows up as large logit gaps everywhere; bf16 noise shows up as
small gaps, with flips only where the reference top-2 margin is tiny.

    python pp_bench/diag_logits.py --model-path codellama/CodeLlama-7b-Instruct-hf
"""

import argparse

import torch
from common import add_server_args, make_server_args, squad_groups

from sglang.srt.parallel_prompt.engine import ParallelPromptEngine


def main():
    parser = argparse.ArgumentParser()
    add_server_args(parser)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--max-questions", type=int, default=8)
    parser.add_argument("--steps", type=int, default=24)
    args = parser.parse_args()

    engine = ParallelPromptEngine(make_server_args(args))
    docs, questions = squad_groups(engine.tokenizer, args.groups, args.max_questions)

    # Record PP logits of the prefill and of every decode step (eager mode).
    # One request per group, padded to P streams: row of (g, s) is g * P + s.
    pp_logits = []
    engine.logits_hook = lambda logits: pp_logits.append(logits.float().cpu())
    pp_out = engine.generate(docs, questions, args.steps, stop_token_ids=[])
    engine.logits_hook = None
    P = max(len(qs) for qs in questions)

    rows = {}
    for t, step in enumerate(pp_logits):
        for g, qs in enumerate(questions):
            for s in range(len(qs)):
                rows[(g, s, t)] = step[g * P + s]

    max_diffs, flips, flip_margins, margins_all = [], 0, [], []
    for t in range(args.steps):
        keys, prompts = [], []
        for g, qs in enumerate(questions):
            for s, q in enumerate(qs):
                keys.append((g, s))
                prompts.append(docs[g] + q + pp_out[g][s][:t])
        ref_logits = []
        orig_fwd = engine._forward

        def capture_ref(batch):
            mwb = batch.get_model_worker_batch()
            from sglang.srt.model_executor.forward_batch_info import ForwardBatch

            fb = ForwardBatch.init_new(mwb, engine.model_runner)
            logits = engine.model_runner.forward(fb).next_token_logits.float()
            ref_logits.append(logits.cpu())
            return torch.argmax(logits, dim=-1).tolist()

        engine._forward = capture_ref
        engine.generate_independent(prompts, 1, stop_token_ids=[])
        engine._forward = orig_fwd
        ref = ref_logits[0]
        for i, (g, s) in enumerate(keys):
            a, b = rows[(g, s, t)], ref[i]
            max_diffs.append((a - b).abs().max().item())
            top2 = torch.topk(b, 2).values
            margin = (top2[0] - top2[1]).item()
            margins_all.append(margin)
            if a.argmax() != b.argmax():
                flips += 1
                flip_margins.append(margin)

    md = sorted(max_diffs)
    n = len(md)
    print(f"compared {n} (stream, step) logit rows over {args.steps} teacher-forced steps")
    print(
        f"max |logit diff|: median {md[n // 2]:.3f}  p99 {md[int(n * 0.99)]:.3f}  max {md[-1]:.3f}"
    )
    print(f"argmax flips: {flips}/{n}")
    if flip_margins:
        print("reference top1-top2 margin at flips:", [round(m, 3) for m in flip_margins])
    ma = sorted(margins_all)
    print(f"reference margin overall: median {ma[n // 2]:.3f}")


if __name__ == "__main__":
    main()
