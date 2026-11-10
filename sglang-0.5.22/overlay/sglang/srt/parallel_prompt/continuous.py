"""Parallel Prompting with continuous batching (eval_qa config `pp-cb`).

ParallelPromptEngine runs the requests of a call in waves: a wave is
prefilled, decoded until its requests stop, and only then is the next wave
admitted. Compaction shrinks a wave as its requests finish, but the KV slots
and batch places they free stay idle until the whole wave is done.

This engine keeps one running batch and refills it, like SGLang's scheduler
admits requests as memory frees: at every stop check (each
STOP_CHECK_INTERVAL decode steps, or earlier when a request reaches its token
budget) finished requests are retired, their slots and rows freed, and
waiting requests are prefilled into the free places. Requests no longer
advance in lockstep, so each keeps its own step count t_r: its decode state
is prefix0 + P*t_r and next_pos + t_r, rebuilt at every check.

Unchanged from the base engine: requests of P streams over the shared
document KV, each request's slots allocated up front at admission
(q_len + P*steps), greedy decoding, and one CUDA graph per (batch size, P).
The batch is rounded up to a power of two; its empty places run a dummy
request on a small scratch row (outputs dropped), so few shapes are captured.
"""

import os
import time
from collections import deque
from typing import List, Optional, Sequence

import torch

from sglang.srt.parallel_prompt.engine import (
    STOP_CHECK_INTERVAL,
    DocCache,
    GenerateStats,
    ParallelPromptEngine,
)


class ContinuousPPEngine(ParallelPromptEngine):
    @torch.no_grad()
    def generate_cached(
        self,
        cache: DocCache,
        questions: Sequence[Sequence[List[int]]],
        max_new_tokens: int,
        stop_token_ids: Optional[List[int]] = None,
        parallel_size: Optional[int] = None,
    ) -> List[List[List[int]]]:
        if max_new_tokens <= 1:  # prefill only: nothing to schedule
            return super().generate_cached(
                cache, questions, max_new_tokens, stop_token_ids, parallel_size
            )
        assert len(cache.lens) == len(questions)
        stop = sorted(
            set(self.default_stop_ids() if stop_token_ids is None else stop_token_ids)
        )
        runner = self.model_runner
        dev = self.device
        P = parallel_size or max(len(qs) for qs in questions)
        T = max_new_tokens
        steps = T - 1

        reqs = []  # (group, [question index or None per stream]), as the base engine
        for g, qs in enumerate(questions):
            for c in range(0, len(qs), P):
                idx = list(range(c, min(c + P, len(qs))))
                reqs.append((g, idx + [None] * (P - len(idx))))
        N = len(reqs)
        q_len = []
        for g, idx in reqs:  # padded streams repeat the last question
            last = max(i for i in idx if i is not None)
            q_len.append(sum(len(questions[g][last if i is None else i]) for i in idx))
        need = [q + P * steps for q in q_len]  # slots a request holds while running

        # Batch capacity: the base engine's wave size (power of two, fits the
        # free pool at the largest request), so graph shapes match pp-on's.
        max_q = max(len(q) for qs in questions for q in qs)
        cap = max(1, min(N, self._free_slots() // (P * (max_q + steps))))
        cap = 1 << (cap.bit_length() - 1)
        cap = min(cap, int(os.environ.get("SGLANG_PP_MAX_WAVE", 0)) or cap)

        # hist[r*P + s, t]: token t of stream s of request r (-1 = not yet
        # decoded). The last P rows belong to the dummy request.
        hist = torch.full((N * P + P, T), -1, dtype=torch.int32, device=dev)
        hist[N * P :] = 0
        stopped = torch.zeros(N * P + P, dtype=torch.bool, device=dev)
        real = torch.tensor(
            [i is not None for _, idx in reqs for i in idx] + [False] * P, device=dev
        )
        stop_t = torch.tensor(stop, dtype=torch.int32, device=dev)
        ar_p = torch.arange(P)

        # Dummy request: one scratch row that a segment (at most
        # STOP_CHECK_INTERVAL steps) decodes into from position 0; every empty
        # batch place points at it and writes the same slots.
        dummy_row = runner.req_to_token_pool.alloc_rows(1)
        if dummy_row is None:
            raise MemoryError("not enough request slots")
        dummy_row = dummy_row[0]
        scratch = self._alloc(P * STOP_CHECK_INTERVAL)
        if scratch is None:
            runner.req_to_token_pool.free_rows([dummy_row])
            raise MemoryError("KV pool too small for this workload")
        self.req_to_token[dummy_row, : scratch.numel()] = scratch.to(torch.int32)
        self.slot_sid[scratch.to(torch.int64)] = (
            torch.arange(P, dtype=torch.int32).repeat(STOP_CHECK_INTERVAL).to(dev)
        )

        row = [None] * N
        own = [None] * N
        prefix0 = [0] * N
        next_pos = [None] * N
        t_r = [0] * N  # decode steps run by each request
        queue = deque(range(N))
        active: List[int] = []
        stats = GenerateStats(num_reqs=N, parallel_size=P, batch_cap=cap, waves=1)

        def admit():
            """Prefill waiting requests (FCFS) into free batch places while
            their slots and rows fit."""
            free = self._free_slots()
            rows_free = runner.req_to_token_pool.available_size()
            new = []
            while queue and len(active) + len(new) < cap and len(new) < rows_free:
                r = queue[0]
                if need[r] > free:
                    break
                free -= need[r]
                new.append(queue.popleft())
            if not new:
                if not active:  # nothing running will free memory
                    raise MemoryError("KV pool too small for this workload")
                return
            torch.cuda.synchronize()
            tic = time.perf_counter()
            rows_, own_, _, pre_, pos_, first = self._prefill_reqs(
                cache, questions, [reqs[r] for r in new], P, steps
            )
            cols = (torch.tensor(new)[:, None] * P + ar_p).view(-1).to(dev)
            hist[cols, 0] = first.to(hist.dtype)
            stopped[cols] = torch.isin(first, stop_t)
            for k, r in enumerate(new):
                row[r], own[r], prefix0[r] = rows_[k], own_[k], pre_[k]
                next_pos[r] = pos_[k * P : (k + 1) * P]
            if active or stats.decode_steps:  # not the first admission
                stats.refills += len(new)
            active.extend(new)
            torch.cuda.synchronize()
            stats.prefill_s += time.perf_counter() - tic

        def retire(done: List[int]):
            for r in done:
                self.kv_alloc.free(own[r])
                own[r] = None
            runner.req_to_token_pool.free_rows([row[r] for r in done])

        def load():
            """Decode state of the running batch (padded with the dummy
            request to a power of two); returns it and the flat hist index
            of each stream's next token."""
            n = len(active)
            R = 1 << (n - 1).bit_length()
            pad = R - n
            st = self._step_state(R, P)
            st.req_rows.copy_(torch.tensor([row[r] for r in active] + [dummy_row] * pad))
            st.prefix.copy_(
                torch.tensor([prefix0[r] + P * t_r[r] for r in active] + [0] * pad)
            )
            st.positions.copy_(
                torch.tensor(
                    [p + t_r[r] for r in active for p in next_pos[r]] + [0] * (P * pad)
                )
            )
            cols = torch.tensor(
                [r * P + s for r in active for s in range(P)] + list(range(N * P, N * P + P)) * pad
            )
            base = torch.tensor([t_r[r] for r in active for _ in range(P)] + [0] * (P * pad))
            cols, base = cols.to(dev), base.to(dev)
            st.tokens.copy_(hist[cols, base])
            if self.cuda_graph and st.graph is None:
                self._capture(st)
            return st, cols, cols * T + base + 1

        try:
            admit()
            flat = hist.view(-1)
            n_steps = 0
            while active:
                # One segment: up to the next stop check, never past the token
                # budget of any running request (its slots end there).
                L = min(STOP_CHECK_INTERVAL, min(steps - t_r[r] for r in active))
                tic = time.perf_counter()
                st, cols, lin = load()
                for j in range(L):
                    if st.graph is not None:
                        st.graph.replay()
                    else:
                        self._step(st)
                    flat[lin + j] = st.tokens
                n_steps += L
                stats.decode_steps = n_steps
                stats.active_req_steps += L * len(active)
                stats.row_steps += L * (1 << (len(active) - 1).bit_length())
                for r in active:
                    t_r[r] += L
                if stop:
                    seg = flat[lin[:, None] + torch.arange(L, device=dev)]
                    stopped[cols] |= torch.isin(seg, stop_t).any(1)
                n = len(active)
                fin = ((stopped | ~real)[cols[: n * P]].view(n, P).all(1)).cpu().tolist()
                torch.cuda.synchronize()
                stats.decode_s += time.perf_counter() - tic
                done = [r for r, f in zip(active, fin) if f or t_r[r] == steps]
                if done:
                    retire(done)
                    active[:] = [r for r in active if own[r] is not None]
                    self.compactions += 1
                    if queue:
                        admit()
            stats.decode_steps = n_steps
        finally:
            retire(active)
            self.kv_alloc.free(scratch)
            runner.req_to_token_pool.free_rows([dummy_row])

        outputs = [[None] * len(qs) for qs in questions]
        stop_set = set(stop)
        hist = hist[: N * P].tolist()
        for r, (g, idx) in enumerate(reqs):
            for s, i in enumerate(idx):
                if i is None:
                    continue
                o = hist[r * P + s][: t_r[r] + 1]
                cut = next((k + 1 for k, x in enumerate(o) if x in stop_set), len(o))
                outputs[g][i] = o[:cut]
        stats.output_tokens = sum(len(o) for g in outputs for o in g)
        self.stats = stats
        return outputs
