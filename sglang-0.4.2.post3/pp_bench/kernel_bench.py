"""Attention-only timing of one decode step, per layer.

PP: R requests, each with a cached prefix of L tokens (document + earlier
stream tokens) and P new tokens, through the Triton extend kernel with the
stream-id mask. Stock: the same R*P streams as separate sequences of L tokens
through the Triton decode kernel (each stream re-reads the document).

    python pp_bench/kernel_bench.py
"""

import itertools

import torch
import triton

import sglang.srt.layers.attention.triton_ops.extend_attention as ext
from sglang.srt.layers.attention.triton_ops.decode_attention import (
    decode_attention_fwd,
)

DEV, DT = "cuda", torch.bfloat16
H, D = 32, 128  # CodeLlama-7B: 32 query and 32 KV heads


def bench(fn):
    return triton.testing.do_bench(fn, warmup=20, rep=100)


def pp_case(R, P, L, cfg):
    slots = R * (L + P) + 1
    kb = torch.randn(slots, H, D, device=DEV, dtype=DT)
    vb = torch.randn(slots, H, D, device=DEV, dtype=DT)
    sid = torch.full((slots,), -1, dtype=torch.int32, device=DEV)
    q = torch.randn(R * P, H, D, device=DEV, dtype=DT)
    k = torch.randn(R * P, H, D, device=DEV, dtype=DT)
    v = torch.randn(R * P, H, D, device=DEV, dtype=DT)
    o = torch.empty_like(q)
    qo = torch.arange(0, R * P + 1, P, dtype=torch.int32, device=DEV)
    kvp = torch.arange(0, R * L + 1, L, dtype=torch.int32, device=DEV)
    kvi = torch.arange(1, R * L + 1, dtype=torch.int32, device=DEV)
    q_sid = torch.arange(P, dtype=torch.int32, device=DEV).repeat(R)
    ext.SMALL_EXTEND_CONFIG = cfg
    return bench(lambda: ext.extend_attention_fwd(
        q, k, v, o, kb, vb, qo, kvp, kvi, None, None, P, D**-0.5,
        q_sid=q_sid, slot_sid=sid))


def stock_case(R, P, L, splits=8):
    n = R * P
    kb = torch.randn(R * L + 1, H, D, device=DEV, dtype=DT)
    vb = torch.randn(R * L + 1, H, D, device=DEV, dtype=DT)
    q = torch.randn(n, H, D, device=DEV, dtype=DT)
    o = torch.empty_like(q)
    # every stream of request r reads request r's L tokens
    kvi = torch.cat([torch.arange(1 + r * L, 1 + (r + 1) * L) for r in range(R) for _ in range(P)])
    kvi = kvi.to(torch.int32).to(DEV)
    kvp = torch.arange(0, n * L + 1, L, dtype=torch.int32, device=DEV)
    logits = torch.empty(n, H, splits, D + 1, dtype=torch.float32, device=DEV)
    return bench(lambda: decode_attention_fwd(q, kb, vb, o, kvp, kvi, logits, splits, D**-0.5))


def main():
    configs = [None, (16, 64, 4), (16, 128, 4), (16, 64, 2), (32, 64, 4), (16, 32, 4)]
    print("R  P   L     stock_decode  " + "  ".join(f"{str(c):>13}" for c in configs))
    for (G, Q), L in itertools.product([(4, 16), (4, 64)], [1024, 4096]):
        for P in [1, 4, 16, 64]:
            if P > Q:
                continue
            R = G * Q // P
            # the decode kernel's int32 offsets overflow beyond ~500K pool slots
            stock = stock_case(R, P, L) if R * P * L <= 500_000 else float("nan")
            row = [stock] + [pp_case(R, P, L, c) for c in configs]
            print(f"{R:<3}{P:<4}{L:<6}" + "  ".join(f"{t * 1000:>11.0f}us" for t in row))


if __name__ == "__main__":
    main()
