"""CPU checks for the Parallel Prompting mask/position builders.

    python pp_bench/test_mask.py
"""

import random

import torch

from sglang.srt.parallel_prompt.mask import (
    DOC_STREAM,
    build_mask,
    decode_layout,
    make_input,
    prefill_layout,
    reference_mask,
)


def check_prefill(doc_len, q_lens):
    sids, pos, rows, next_pos = prefill_layout(doc_len, q_lens)
    total = doc_len + sum(q_lens)
    assert sids.tolist() == [DOC_STREAM] * doc_len + [
        s for s, n in enumerate(q_lens) for _ in range(n)
    ]
    expect_pos = list(range(doc_len)) + [
        doc_len + j for n in q_lens for j in range(n)
    ]
    assert pos.tolist() == expect_pos
    assert rows.tolist() == [doc_len + sum(q_lens[: s + 1]) - 1 for s in range(len(q_lens))]
    assert next_pos == [doc_len + n for n in q_lens]
    got = build_mask(sids, 0, total).view(total, total).tolist()
    assert got == reference_mask(sids.tolist(), 0)
    return sids, next_pos


def check_decode(sids, next_pos, n_streams, steps):
    active = list(range(n_streams))
    for _ in range(steps):
        prefix = sids.numel()
        sids, pos, rows = decode_layout(sids, active, next_pos)
        assert pos.tolist() == [next_pos[s] for s in active]
        assert rows.tolist() == list(range(len(active)))
        ext = len(active)
        got = build_mask(sids, prefix, ext).view(ext, prefix + ext).tolist()
        assert got == reference_mask(sids.tolist(), prefix)
        for s in active:
            next_pos[s] += 1
        if len(active) > 1 and random.random() < 0.3:
            active.pop(random.randrange(len(active)))


def check_batch_offsets():
    a = prefill_layout(3, [2, 1])
    b = prefill_layout(2, [1, 1, 2])
    inp = make_input(
        stream_ids=[a[0], b[0]],
        prefix_lens=[0, 0],
        extend_lens=[6, 6],
        positions=[a[1], b[1]],
        logit_rows=[a[2], b[2]],
    )
    assert inp.mask_offsets.tolist() == [0, 36]
    assert inp.custom_mask.numel() == 72
    assert inp.logit_index.tolist() == [4, 5, 8, 9, 11]
    assert inp.positions.tolist() == [0, 1, 2, 3, 4, 3, 0, 1, 2, 2, 2, 3]


def main():
    random.seed(0)
    torch.manual_seed(0)
    for _ in range(50):
        doc_len = random.randint(1, 12)
        q_lens = [random.randint(1, 5) for _ in range(random.randint(1, 6))]
        sids, next_pos = check_prefill(doc_len, q_lens)
        check_decode(sids, next_pos, len(q_lens), steps=random.randint(1, 6))
    check_batch_offsets()
    # Doc tokens must never see query tokens; query tokens never see other queries.
    sids, _, _, _ = prefill_layout(2, [2, 2])
    m = build_mask(sids, 0, 6).view(6, 6)
    assert not m[:2, 2:].any()
    assert not m[4:, 2:4].any() and not m[2:4, 4:].any()
    print("test_mask: all checks passed")


if __name__ == "__main__":
    main()
