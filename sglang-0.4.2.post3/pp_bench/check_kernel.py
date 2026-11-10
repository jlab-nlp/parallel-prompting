"""GPU check: Triton extend kernel with the Parallel Prompting mask vs. SDPA.

Covers a prefill step (no cached prefix) and decode steps (cached prefix plus one
new token per active stream), batched over several requests, with GQA. Both mask
modes are checked: the dense custom mask and the in-kernel stream-id mask.

    python pp_bench/check_kernel.py
"""

import torch

from sglang.srt.layers.attention.triton_ops.extend_attention import (
    extend_attention_fwd,
)
from sglang.srt.parallel_prompt.mask import (
    build_mask,
    decode_layout,
    make_input,
    prefill_layout,
)

DEV = "cuda"
DTYPE = torch.bfloat16
H_Q, H_KV, D = 32, 8, 128


def reference(q, k_all, v_all, mask):
    # q: [E, Hq, D], k_all/v_all: [L, Hkv, D], mask: [E, L] bool
    rep = H_Q // H_KV
    k = k_all.repeat_interleave(rep, dim=1).transpose(0, 1)  # [Hq, L, D]
    v = v_all.repeat_interleave(rep, dim=1).transpose(0, 1)
    out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1).float(),
        k.float(),
        v.float(),
        attn_mask=mask[None],
    )
    return out.transpose(0, 1)


def run_case(layouts):
    """layouts: list of (stream_ids, prefix_len, extend_len)."""
    pool = sum(s.numel() for s, _, _ in layouts)
    k_buf = torch.randn(pool, H_KV, D, device=DEV, dtype=DTYPE)
    v_buf = torch.randn(pool, H_KV, D, device=DEV, dtype=DTYPE)
    slot_sid = torch.full((pool,), -1, dtype=torch.int32, device=DEV)
    qs, ks, vs, kv_idx, refs, q_sid = [], [], [], [], [], []
    qo_indptr, kv_indptr = [0], [0]
    pt = 0
    for sids, pre, ext in layouts:
        L = pre + ext
        slots = torch.arange(pt, pt + L, device=DEV)
        pt += L
        q = torch.randn(ext, H_Q, D, device=DEV, dtype=DTYPE)
        # The kernel reads new K/V from k_extend and writes nothing to the buffer,
        # so keep the buffer copy in sync for the reference.
        k_ext, v_ext = k_buf[slots[pre:]], v_buf[slots[pre:]]
        qs.append(q)
        ks.append(k_ext)
        vs.append(v_ext)
        kv_idx.append(slots[:pre])
        slot_sid[slots] = sids
        q_sid.append(sids[pre:])
        qo_indptr.append(qo_indptr[-1] + ext)
        kv_indptr.append(kv_indptr[-1] + pre)
        mask = build_mask(sids, pre, ext).view(ext, L)
        refs.append(reference(q, k_buf[slots], v_buf[slots], mask))

    inp = make_input(
        stream_ids=[s for s, _, _ in layouts],
        prefix_lens=[p for _, p, _ in layouts],
        extend_lens=[e for _, _, e in layouts],
        positions=[torch.zeros(e, device=DEV) for _, _, e in layouts],
        logit_rows=[torch.zeros(1, device=DEV) for _ in layouts],
    )
    q = torch.cat(qs)
    ref = torch.cat(refs)
    errs = []
    for stream_ids in (False, True):
        o = torch.empty_like(q)
        extend_attention_fwd(
            q,
            torch.cat(ks).contiguous(),
            torch.cat(vs).contiguous(),
            o,
            k_buf,
            v_buf,
            torch.tensor(qo_indptr, dtype=torch.int32, device=DEV),
            torch.tensor(kv_indptr, dtype=torch.int32, device=DEV),
            torch.cat(kv_idx).to(torch.int32),
            None if stream_ids else inp.custom_mask,
            None if stream_ids else inp.mask_offsets,
            max(e for _, _, e in layouts),
            D**-0.5,
            q_sid=torch.cat(q_sid) if stream_ids else None,
            slot_sid=slot_sid if stream_ids else None,
        )
        errs.append((o.float() - ref).abs().max().item())
    return max(errs)


def main():
    torch.manual_seed(0)
    worst = 0.0
    for doc_len, q_lens_list in [
        (300, [[7, 12, 5, 9], [3, 3]]),
        (1000, [[20] * 16, [11, 4, 30]]),
        (37, [[1] * 64]),
    ]:
        groups = []
        for q_lens in q_lens_list:
            sids, _, _, next_pos = prefill_layout(doc_len, q_lens, device=DEV)
            groups.append((sids, next_pos, list(range(len(q_lens)))))
        # prefill
        err = run_case([(g[0], 0, g[0].numel()) for g in groups])
        print(f"doc={doc_len} prefill: max abs err {err:.4f}")
        worst = max(worst, err)
        # a few decode steps, dropping a stream each time
        for step in range(4):
            layouts, new_groups = [], []
            for sids, next_pos, active in groups:
                pre = sids.numel()
                sids, _, _ = decode_layout(sids, active, next_pos, device=DEV)
                layouts.append((sids, pre, len(active)))
                for s in active:
                    next_pos[s] += 1
                new_groups.append((sids, next_pos, active[:-1] or active))
            groups = new_groups
            err = run_case(layouts)
            print(f"doc={doc_len} decode step {step}: max abs err {err:.4f}")
            worst = max(worst, err)
    print(f"worst max abs err {worst:.4f}")
    assert worst < 3e-2, "Triton PP mask output does not match SDPA reference"
    print("check_kernel: passed")


if __name__ == "__main__":
    main()
