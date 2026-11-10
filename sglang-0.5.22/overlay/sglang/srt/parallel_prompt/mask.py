"""Attention masks and positions for Parallel Prompting.

One request holds a shared document followed by N independent query streams:

    [Doc][q_0][q_1]...[q_{N-1}]  then each decode step appends  [a_0^t]...[a_{N-1}^t]

Every cached token carries a stream id (-1 for the document). A query token of
stream s may attend to key token k iff k is not after it and k belongs to the
document or to stream s. Positions of every stream start right after the
document, so each stream sees the same positions as if it were decoded alone.

The masks follow the layout of the Triton extend kernel
(`triton_ops/extend_attention.py`): for each request, a flattened bool matrix of
shape [extend_len, prefix_len + extend_len], concatenated over the batch, with
`mask_offsets[i]` giving where request i starts.

The engine instead passes stream ids (`q_sid`, `slot_sid`) and lets the kernel
apply the same rule; the dense mask is kept for tests and as a reference.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch

DOC_STREAM = -1


@dataclass
class ParallelPromptInput:
    """Carried in `ForwardBatch.spec_info` for a Parallel Prompting extend step."""

    # [num_extend_tokens] RoPE positions, overriding the default contiguous ones.
    positions: torch.Tensor
    # Flattened bool mask, see module docstring.
    custom_mask: Optional[torch.Tensor] = None
    # [bs] start of each request's mask in `custom_mask`.
    mask_offsets: Optional[torch.Tensor] = None
    # Rows of the extend hidden states to turn into logits.
    logit_index: Optional[torch.Tensor] = None
    # Alternative to the dense mask, computed inside the kernel (fixed-size, so
    # CUDA-graph friendly): stream id of every extend token, and of every slot
    # of the KV pool.
    q_sid: Optional[torch.Tensor] = None
    slot_sid: Optional[torch.Tensor] = None


def build_mask(
    stream_ids: torch.Tensor, prefix_len: int, extend_len: int
) -> torch.Tensor:
    """Mask for one request. `stream_ids` covers all prefix + extend tokens."""
    seq_len = prefix_len + extend_len
    assert stream_ids.numel() == seq_len
    rows = torch.arange(prefix_len, seq_len, device=stream_ids.device)
    cols = torch.arange(seq_len, device=stream_ids.device)
    row_sid = stream_ids[prefix_len:]
    causal = cols[None, :] <= rows[:, None]
    same_stream = (stream_ids[None, :] == DOC_STREAM) | (
        stream_ids[None, :] == row_sid[:, None]
    )
    return (causal & same_stream).flatten()


def make_input(
    stream_ids: List[torch.Tensor],
    prefix_lens: List[int],
    extend_lens: List[int],
    positions: List[torch.Tensor],
    logit_rows: List[torch.Tensor],
) -> ParallelPromptInput:
    """Pack per-request pieces into one batch. `logit_rows` are local to each
    request's extend tokens."""
    device = stream_ids[0].device
    masks, offsets, rows = [], [], []
    mask_pt = tok_pt = 0
    for sid, pre, ext, lr in zip(stream_ids, prefix_lens, extend_lens, logit_rows):
        assert pre > 0 or sid[0] == DOC_STREAM, "first token must be document"
        masks.append(build_mask(sid, pre, ext))
        offsets.append(mask_pt)
        rows.append(lr + tok_pt)
        mask_pt += ext * (pre + ext)
        tok_pt += ext
    return ParallelPromptInput(
        positions=torch.cat(positions).to(torch.int64),
        custom_mask=torch.cat(masks),
        mask_offsets=torch.tensor(offsets, dtype=torch.int64, device=device),
        logit_index=torch.cat(rows).to(torch.int64),
    )


def prefill_layout(doc_len: int, q_lens: List[int], device=None):
    """Stream ids, positions and logit rows for the prefill of one request.

    Returns (stream_ids, positions, logit_rows, next_pos) where `next_pos[s]` is
    the position of the first answer token of stream s.
    """
    assert doc_len > 0
    sids = [torch.full((doc_len,), DOC_STREAM, dtype=torch.int32)]
    pos = [torch.arange(doc_len)]
    logit_rows, next_pos = [], []
    pt = doc_len
    for s, n in enumerate(q_lens):
        assert n > 0
        sids.append(torch.full((n,), s, dtype=torch.int32))
        pos.append(torch.arange(doc_len, doc_len + n))
        pt += n
        logit_rows.append(pt - 1)
        next_pos.append(doc_len + n)
    return (
        torch.cat(sids).to(device),
        torch.cat(pos).to(device),
        torch.tensor(logit_rows, device=device),
        next_pos,
    )


def decode_layout(
    stream_ids: torch.Tensor,
    active: List[int],
    next_pos: List[int],
    device=None,
):
    """Extend `stream_ids` with one token per active stream.

    Returns (stream_ids, positions, logit_rows) for this step; the caller
    advances `next_pos` after the step.
    """
    new = torch.tensor(active, dtype=torch.int32, device=device)
    positions = torch.tensor([next_pos[s] for s in active], device=device)
    logit_rows = torch.arange(len(active), device=device)
    return torch.cat([stream_ids, new]), positions, logit_rows


def reference_mask(stream_ids: List[int], prefix_len: int) -> List[List[bool]]:
    """Plain-Python mask used by the tests."""
    seq_len = len(stream_ids)
    out = []
    for q in range(prefix_len, seq_len):
        row = []
        for k in range(seq_len):
            row.append(
                k <= q
                and (stream_ids[k] == DOC_STREAM or stream_ids[k] == stream_ids[q])
            )
        out.append(row)
    return out
