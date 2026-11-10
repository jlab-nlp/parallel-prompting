"""Offline Parallel Prompting engine on top of SGLang's ModelRunner (v0.5.22).

Port of the 0.4.2.post3 engine; the algorithm is unchanged.

Each group is one shared document plus Q questions. The questions are split into
chunks of `parallel_size` (P) streams; each chunk is one SGLang request whose
KV row starts with the group's document slots (prefilled once and shared by all
chunks, i.e. prefix caching), followed by its own question and answer slots:

    row = [Doc (shared)][q_0]..[q_{P-1}][a_0^1..a_{P-1}^1][a_0^2..a_{P-1}^2]...

A decode step extends every request by P tokens (one per stream) as a plain
EXTEND batch. The Parallel Prompting mask is applied inside the Triton extend
kernel from stream ids (`q_sid` per new token, `slot_sid` per KV slot), and
every stream gets its own positions starting after the document.

All slots are allocated up front and streams that finish keep decoding (their
extra tokens are cut off), so a decode step has a fixed shape and does no CPU
work: it can run eagerly or be captured once as a CUDA graph per (R, P).

Like benchmark/one_batch.py, this drives the ModelRunner directly (no
scheduler). Requires `--attention-backend triton` and page size 1.
"""

import os
import time
from array import array
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from sglang.benchmark.one_batch import _init_process_global_configs, load_model
from sglang.kernels.ops.kvcache.kv_indices import create_flashinfer_kv_indices_triton
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.layers.attention.triton_backend import ForwardMetadata
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.parallel_prompt.mask import DOC_STREAM, ParallelPromptInput
from sglang.srt.runtime_context import SpawnRanks, publish, spawn_world_rank
from sglang.srt.server_args import PortArgs, ServerArgs

# How often (in decode steps) to check on the CPU whether every stream has
# emitted a stop token. Only used when stop tokens are given.
STOP_CHECK_INTERVAL = 16


@dataclass
class DocCache:
    """KV of documents prefilled once, shared by every chunk of their group."""

    lens: List[int]
    slots: List[torch.Tensor]
    # With the prefix cache: the locked radix tree node of each document,
    # which owns its slots (released by unlocking, not freeing).
    nodes: List = field(default_factory=list)


@dataclass
class GenerateStats:
    doc_prefill_s: float = 0.0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    decode_steps: int = 0
    output_tokens: int = 0
    num_reqs: int = 0
    parallel_size: int = 0
    batch_cap: int = 0  # wave size (pp) or running-batch capacity (pp-cb)
    waves: int = 0  # pp: waves run; pp-cb: 1
    refills: int = 0  # pp-cb: requests admitted after the first admission
    active_req_steps: int = 0  # pp-cb: sum over decode steps of running real requests
    row_steps: int = 0  # sum over decode steps of batch rows run (incl. padding)


def _metadata(max_extend_len, kv_indptr, kv_indices, qo_indptr, q_sid, slot_sid):
    return ForwardMetadata(
        attn_logits=None,
        attn_lse=None,
        max_extend_len=max_extend_len,
        num_kv_splits=None,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        qo_indptr=qo_indptr,
        custom_mask=None,
        mask_indptr=None,
        window_kv_indptr=None,
        window_kv_indices=None,
        window_num_kv_splits=None,
        window_kv_offsets=None,
        q_sid=q_sid,
        slot_sid=slot_sid,
    )


class ParallelPromptEngine:
    def __init__(
        self, server_args: ServerArgs, cuda_graph: bool = False, prefix_cache: bool = False
    ):
        assert (
            server_args.attention_backend == "triton"
        ), "Parallel Prompting is only wired for the triton backend"
        assert server_args.tp_size == 1, "prototype supports tp_size=1"
        # Same setup as benchmark/one_batch.py's correctness_test.
        server_args.resolve_once()
        _set_envs_and_config(server_args)
        port_args = PortArgs.init_new(server_args)
        publish(
            server_args,
            role="scheduler",
            ranks=SpawnRanks(
                world_rank=spawn_world_rank(server_args, tp_rank=0, pp_rank=0),
                gpu_id=0,
            ),
        )
        _init_process_global_configs()
        bench_runner, self.tokenizer = load_model(server_args, port_args, 0, 0)
        self.model_runner = bench_runner.torch_runner
        runner = self.model_runner
        backend = runner.attn_backend
        assert type(backend).__name__ == "TritonAttnBackend", type(backend)
        assert getattr(backend, "page_size", 1) == 1, "requires page size 1"
        self.device = runner.device
        self.cuda_graph = cuda_graph
        self.stats = GenerateStats()
        # Called with the logits of every eager PP forward (prefill, then each
        # decode step); used by diag_logits.py.
        self.logits_hook: Optional[Callable[[torch.Tensor], None]] = None

        self.req_to_token = runner.req_to_token_pool.req_to_token
        self.kv_alloc = runner.token_to_kv_pool_allocator
        # SGLang's radix tree over document token ids (page size 1). Documents
        # that share a prefix (e.g. an instruction and few-shot block, or the
        # same document asked again) reuse its KV; streams are never cached.
        self.tree = (
            RadixCache(
                CacheInitParams(
                    disable=False,
                    req_to_token_pool=runner.req_to_token_pool,
                    token_to_kv_pool_allocator=self.kv_alloc,
                    page_size=1,
                )
            )
            if prefix_cache
            else None
        )
        self.prefix_hit_tokens = 0
        # Question prefill is split into forwards of at most this many tokens.
        self.max_prefill_tokens = 16384
        # Drop requests whose streams have all emitted a stop token from the
        # decode batch (see _decode_wave); False keeps them, so a wave decodes
        # until its longest stream stops.
        self.compact_finished = True
        self.compactions = 0  # times a decode batch was shrunk (diagnostics)
        num_slots = runner.token_to_kv_pool.get_key_buffer(0).shape[0]
        self.slot_sid = torch.full(
            (num_slots,), DOC_STREAM, dtype=torch.int32, device=self.device
        )
        self._ctx = ForwardContext(attn_backend=backend)
        self._steps: Dict[Tuple[int, int], SimpleNamespace] = {}
        self._graph_pool = None

    def default_stop_ids(self) -> List[int]:
        eos = self.model_runner.model_config.hf_eos_token_id
        ids = set(eos) if eos else set()
        if self.tokenizer.eos_token_id is not None:
            ids.add(self.tokenizer.eos_token_id)
        return sorted(ids)

    # ------------------------------------------------------------------ API

    @torch.no_grad()
    def generate(
        self,
        docs: Sequence[List[int]],
        questions: Sequence[Sequence[List[int]]],
        max_new_tokens: int,
        stop_token_ids: Optional[List[int]] = None,
        parallel_size: Optional[int] = None,
    ) -> List[List[List[int]]]:
        """Greedy-decode every question of every group.

        Returns output token ids indexed as [group][question]."""
        torch.cuda.synchronize()
        tic = time.perf_counter()
        cache = self.prefill_docs(docs)
        torch.cuda.synchronize()
        doc_s = time.perf_counter() - tic
        try:
            out = self.generate_cached(
                cache, questions, max_new_tokens, stop_token_ids, parallel_size
            )
        finally:
            self.release_docs(cache)
        self.stats.doc_prefill_s = doc_s
        return out

    def _free_slots(self) -> int:
        """Slots available to allocate, counting unlocked prefix-cache entries."""
        n = self.kv_alloc.available_size()
        return n + self.tree.evictable_size() if self.tree is not None else n

    def _alloc(self, n: int):
        if self.tree is not None and self.kv_alloc.available_size() < n:
            self.tree.evict(EvictParams(num_tokens=n - self.kv_alloc.available_size()))
        return self.kv_alloc.alloc(n)

    def flush_prefix_cache(self) -> None:
        """Drop every unlocked prefix-cache entry (cold start for a timed pass)."""
        if self.tree is not None:
            self.tree.evict(EvictParams(num_tokens=self.tree.evictable_size()))

    @torch.no_grad()
    def prefill_docs(self, docs: Sequence[List[int]]) -> DocCache:
        """Prefill each document once; its KV is shared by all its questions."""
        if self.tree is not None:
            return self._prefill_docs_cached(docs)
        lens = [len(d) for d in docs]
        slots = []
        for n in lens:
            s = self._alloc(n)
            if s is None:
                raise MemoryError("KV pool too small for the documents")
            slots.append(s)
        out_cache_loc = torch.cat(slots).to(torch.int64)
        self.slot_sid[out_cache_loc] = DOC_STREAM
        dev = self.device
        qo = torch.tensor([0] + lens, device=dev).cumsum(0).to(torch.int32)
        self._extend(
            input_ids=torch.tensor(
                [t for d in docs for t in d], dtype=torch.int32, device=dev
            ),
            positions=torch.cat([torch.arange(n, device=dev) for n in lens]),
            out_cache_loc=out_cache_loc,
            qo_indptr=qo,
            kv_indptr=torch.zeros(len(lens) + 1, dtype=torch.int32, device=dev),
            kv_indices=torch.empty(0, dtype=torch.int32, device=dev),
            max_extend_len=max(lens),
            logit_index=(qo[1:] - 1).to(torch.int64),
            q_sid=None,  # documents are plain causal
        )
        return DocCache(lens=lens, slots=slots)

    def _prefill_docs_cached(self, docs: Sequence[List[int]]) -> DocCache:
        """prefill_docs with the radix tree: each document reuses its longest
        cached prefix and prefills only the rest. A document is inserted
        before it is computed, so a later document of the same batch that
        shares its prefix reuses those slots too (their KV is written before
        attention runs in the same forward, as in vLLM's prefix cache)."""
        dev = self.device
        lens, slots, nodes = [], [], []
        tails = []  # (token ids, first position, cached prefix slots, new slots)
        for d in docs:
            key = RadixKey(array("q", d))
            m = self.tree.match_prefix(MatchPrefixParams(key=key))
            hit = m.device_prefix_len
            # Keep the matched path while allocating (allocation may evict).
            self.tree.inc_lock_ref(m.last_device_node)
            prefix = self.tree.path_device_indices(m.last_device_node)[:hit]
            new = self._alloc(len(d) - hit) if hit < len(d) else prefix[:0]
            self.tree.dec_lock_ref(m.last_device_node)
            if new is None:
                raise MemoryError("KV pool too small for the documents")
            new = new.to(torch.int64)
            row = torch.cat([prefix.to(torch.int64), new])
            r = self.tree.insert(InsertParams(key=key, value=row))
            assert r.prefix_len == hit, (r.prefix_len, hit)
            self.tree.inc_lock_ref(r.last_device_node)
            nodes.append(r.last_device_node)
            lens.append(len(d))
            slots.append(row)
            self.prefix_hit_tokens += hit
            if hit < len(d):
                tails.append((d[hit:], hit, prefix, new))
        if tails:
            out_cache_loc = torch.cat([t[3] for t in tails])
            self.slot_sid[out_cache_loc] = DOC_STREAM
            ext = [len(t[0]) for t in tails]
            qo = torch.tensor([0] + ext, device=dev).cumsum(0).to(torch.int32)
            self._extend(
                input_ids=torch.tensor(
                    [x for t in tails for x in t[0]], dtype=torch.int32, device=dev
                ),
                positions=torch.cat(
                    [torch.arange(t[1], t[1] + len(t[0]), device=dev) for t in tails]
                ),
                out_cache_loc=out_cache_loc,
                qo_indptr=qo,
                kv_indptr=torch.tensor([0] + [len(t[2]) for t in tails], device=dev)
                .cumsum(0)
                .to(torch.int32),
                kv_indices=torch.cat([t[2] for t in tails]).to(torch.int32),
                max_extend_len=max(ext),
                logit_index=(qo[1:] - 1).to(torch.int64),
                q_sid=None,  # documents are plain causal
            )
        return DocCache(lens=lens, slots=slots, nodes=nodes)

    def release_docs(self, cache: DocCache) -> None:
        # With the prefix cache the tree owns the document slots: release by
        # unlocking. Freeing them here would hand cached KV to new requests.
        assert self.tree is None or len(cache.nodes) == len(cache.lens), (
            "prefix-cache DocCache lost its tree nodes")
        if cache.nodes:
            for node in cache.nodes:
                self.tree.dec_lock_ref(node)
            return
        self.kv_alloc.free(torch.cat(cache.slots))

    @torch.no_grad()
    def generate_cached(
        self,
        cache: DocCache,
        questions: Sequence[Sequence[List[int]]],
        max_new_tokens: int,
        stop_token_ids: Optional[List[int]] = None,
        parallel_size: Optional[int] = None,
    ) -> List[List[List[int]]]:
        """Answer the questions of each cached document, `parallel_size` streams
        per request (default: all questions of a group in one request)."""
        assert len(cache.lens) == len(questions)
        stop = sorted(
            set(self.default_stop_ids() if stop_token_ids is None else stop_token_ids)
        )
        runner = self.model_runner
        dev = self.device
        P = parallel_size or max(len(qs) for qs in questions)
        steps = max_new_tokens - 1

        # Split each group into chunks of P streams; pad the last chunk with
        # copies of its last question (their outputs are dropped).
        reqs = []  # (group, [question index or None per stream])
        for g, qs in enumerate(questions):
            for c in range(0, len(qs), P):
                idx = list(range(c, min(c + P, len(qs))))
                reqs.append((g, idx + [None] * (P - len(idx))))

        # Run the requests in waves that fit the free KV pool, as stock SGLang's
        # scheduler admits only what fits. The wave size is rounded down to a
        # power of two so it stays the same across inputs of similar size, and
        # equal-size waves share one CUDA graph.
        max_q = max(len(q) for qs in questions for q in qs)
        need = P * (max_q + steps)
        wave = max(1, min(len(reqs), self._free_slots() // need))
        wave = 1 << (wave.bit_length() - 1)
        # Tests can force small waves (e.g. SGLANG_PP_MAX_WAVE=2) on any pool size.
        wave = min(wave, int(os.environ.get("SGLANG_PP_MAX_WAVE", 0)) or wave)
        hist = []
        stats = GenerateStats(num_reqs=len(reqs), parallel_size=P, batch_cap=wave)
        for w in range(0, len(reqs), wave):
            stats.waves += 1
            h, pre, dec, n = self._run_wave(
                cache, questions, reqs[w : w + wave], P, max_new_tokens, stop
            )
            hist += h
            stats.prefill_s += pre
            stats.decode_s += dec
            stats.decode_steps += n
            stats.row_steps += self._wave_row_steps

        outputs = [[None] * len(qs) for qs in questions]
        stop_set = set(stop)
        for r, (g, idx) in enumerate(reqs):
            for s, i in enumerate(idx):
                if i is None:
                    continue
                o = hist[r * P + s]
                cut = next((k + 1 for k, x in enumerate(o) if x in stop_set), len(o))
                outputs[g][i] = o[:cut]
        stats.output_tokens = sum(len(o) for g in outputs for o in g)
        self.stats = stats
        return outputs

    def _run_wave(self, cache, questions, reqs, P, max_new_tokens, stop):
        """Prefill and decode one wave of requests; returns each stream's tokens
        (request-major), prefill and decode seconds, and decode steps."""
        runner = self.model_runner
        dev = self.device
        steps = max_new_tokens - 1
        torch.cuda.synchronize()
        tic = time.perf_counter()
        R = len(reqs)
        rows, own_slots, stream_qlens, prefix0, next_pos, first = self._prefill_reqs(
            cache, questions, reqs, P, steps
        )

        hist = torch.empty((max_new_tokens, R * P), dtype=torch.int32, device=dev)
        hist[0] = first
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - tic

        tic = time.perf_counter()
        n_steps = 0
        self._wave_row_steps = 0
        if steps > 0:
            n_steps = self._decode_wave(
                reqs, rows, own_slots, stream_qlens, prefix0, next_pos, first, hist, stop
            )
        torch.cuda.synchronize()
        decode_s = time.perf_counter() - tic

        hist = hist[: n_steps + 1].T.tolist()
        self.kv_alloc.free(torch.cat(own_slots))
        runner.req_to_token_pool.free_rows(rows)
        return hist, prefill_s, decode_s, n_steps

    def _prefill_reqs(self, cache, questions, reqs, P, steps):
        """Allocate a row and the question and answer slots (`q_len + P*steps`)
        of each request and prefill its questions; returns the rows, slots,
        question length of each stream, each request's prefix length, each
        stream's next position, and each stream's first token."""
        runner = self.model_runner
        dev = self.device
        R = len(reqs)
        rows = runner.req_to_token_pool.alloc_rows(R)
        if rows is None:
            raise MemoryError("not enough request slots")

        ids, pos, sids, own_slots, prefix0, kv_idx, logit_rows = [], [], [], [], [], [], []
        stream_qlens = []
        next_pos = []
        tok_pt = 0
        try:
            for (g, idx), row in zip(reqs, rows):
                doc_len = cache.lens[g]
                last = max(i for i in idx if i is not None)
                qs = [questions[g][last if i is None else i] for i in idx]
                q_len = sum(len(q) for q in qs)
                if doc_len + q_len + P * steps > self.req_to_token.shape[1]:
                    raise MemoryError("request row too short; raise --pp-row-len")
                own = self._alloc(q_len + P * steps)
                if own is None:
                    raise MemoryError("KV pool too small for this workload")
                own_slots.append(own)
                self.req_to_token[row, :doc_len] = cache.slots[g].to(torch.int32)
                self.req_to_token[row, doc_len : doc_len + own.numel()] = own.to(
                    torch.int32
                )
                row_sid = [
                    torch.full((len(q),), s, dtype=torch.int32) for s, q in enumerate(qs)
                ]
                row_sid.append(torch.arange(P, dtype=torch.int32).repeat(steps))
                self.slot_sid[own.to(torch.int64)] = torch.cat(row_sid).to(dev)
                for s, q in enumerate(qs):
                    ids.extend(q)
                    pos.append(torch.arange(doc_len, doc_len + len(q)))
                    sids.append(row_sid[s])
                    tok_pt += len(q)
                    logit_rows.append(tok_pt - 1)
                    next_pos.append(doc_len + len(q))
                    stream_qlens.append(len(q))
                kv_idx.append(cache.slots[g])
                prefix0.append(doc_len + q_len)
        except MemoryError:
            if own_slots:
                self.kv_alloc.free(torch.cat(own_slots))
            runner.req_to_token_pool.free_rows(rows)
            raise

        q_lens = [p - cache.lens[g] for p, (g, _) in zip(prefix0, reqs)]
        # Prefill the questions in forwards of at most max_prefill_tokens
        # tokens (whole requests), as stock engines chunk prefill: a wave of
        # many requests over one shared prefix can otherwise run out of
        # activation memory. Requests are independent, so this is exact.
        firsts, a, toff = [], 0, 0
        while a < R:
            b, n = a + 1, q_lens[a]
            while b < R and n + q_lens[b] <= self.max_prefill_tokens:
                n += q_lens[b]
                b += 1
            sl = slice(a * P, b * P)  # streams of requests a..b-1
            firsts.append(self._extend(
                input_ids=torch.tensor(ids[toff : toff + n], dtype=torch.int32, device=dev),
                positions=torch.cat(pos[sl]).to(dev),
                out_cache_loc=torch.cat(
                    [o[:k] for o, k in zip(own_slots[a:b], q_lens[a:b])]
                ).to(torch.int64),
                qo_indptr=torch.tensor([0] + q_lens[a:b], device=dev).cumsum(0).to(torch.int32),
                kv_indptr=torch.tensor([0] + [cache.lens[g] for g, _ in reqs[a:b]], device=dev)
                .cumsum(0)
                .to(torch.int32),
                kv_indices=torch.cat(kv_idx[a:b]).to(torch.int32),
                max_extend_len=max(q_lens[a:b]),
                logit_index=torch.tensor(
                    [x - toff for x in logit_rows[sl]], dtype=torch.int64, device=dev
                ),
                q_sid=torch.cat(sids[sl]).to(dev),
            ).argmax(-1))
            a, toff = b, toff + n
        return rows, own_slots, stream_qlens, prefix0, next_pos, torch.cat(firsts)

    def _decode_wave(
        self, reqs, rows, own_slots, stream_qlens, prefix0, next_pos, first, hist, stop
    ) -> int:
        """Decode a prefilled wave in place into `hist` (row 0 holds the first
        tokens); returns the number of decode steps run."""
        dev = self.device
        R, P = len(reqs), len(reqs[0][1])
        rows_t = torch.tensor(rows, dtype=torch.int64)
        prefix0_t = torch.tensor(prefix0, dtype=torch.int64)
        next_pos_t = torch.tensor(next_pos, dtype=torch.int64).view(R, P)
        real = torch.tensor(
            [i is not None for _, idx in reqs for i in idx], device=dev
        ).view(R, P)
        stop_t = torch.tensor(stop, dtype=torch.int32, device=dev)
        ar_p = torch.arange(P)

        def load(sel, t):
            """Decode state of requests `sel` after t steps. Requests advance
            in lockstep (P slots and one position per step), so the state is
            rebuilt from the wave's start values; returns it and the hist
            columns of its streams."""
            self.compactions += t > 0
            st = self._step_state(len(sel), P)
            st.req_rows.copy_(rows_t[sel])
            st.prefix.copy_(prefix0_t[sel] + P * t)
            st.positions.copy_((next_pos_t[sel] + t).view(-1))
            cols = (sel[:, None] * P + ar_p).view(-1).to(dev)
            st.tokens.copy_(hist[t, cols])
            if self.cuda_graph and st.graph is None:
                self._capture(st)
            return st, cols

        sel = torch.arange(R)
        st, cols = load(sel, 0)
        steps = hist.shape[0] - 1
        n_steps = 0
        self._wave_row_steps = 0
        for t in range(1, steps + 1):
            self._wave_row_steps += len(sel)
            if st.graph is not None:
                st.graph.replay()
            else:
                self._step(st)
            hist[t, cols] = st.tokens
            n_steps = t
            if stop and (t % STOP_CHECK_INTERVAL == 0 or t == steps):
                done = torch.isin(hist[: t + 1], stop_t).any(0).view(R, P) | ~real
                req_done = done.all(1).cpu()
                if bool(req_done.all()):
                    break
                live = (~req_done).nonzero().flatten()
                # Drop requests whose streams have all stopped, once at most
                # half the batch is still running: the batch shrinks to the
                # next power of two (topped up with finished requests, which
                # decode into their own slots and are cut afterwards), so a
                # request stops at the longest of its own P answers, not of
                # the whole wave, and only a few graph shapes are captured.
                # Never after the last step: nothing is left to decode, and a new
                # graph's warmup step would run past the allocated slots.
                if self.compact_finished and t < steps and len(live) <= len(sel) // 2:
                    n = 1 << (len(live) - 1).bit_length()
                    # Pad with finished requests of the current batch: their
                    # hist columns are current (those dropped earlier are not).
                    pad = sel[req_done[sel]][: n - len(live)]
                    sel = torch.cat([live, pad])
                    st, cols = load(sel, t)
        return n_steps

    def _decode_loop(self, step, reqs, hist, stop) -> int:
        """Run `step` (returns the new token of every stream) until `hist` is
        full or every real stream has emitted a stop token."""
        dev = self.device
        real = torch.tensor(
            [i is not None for _, idx in reqs for i in idx], device=dev
        )
        stop_t = torch.tensor(stop, dtype=torch.int32, device=dev)
        steps = hist.shape[0] - 1
        n_steps = 0
        for t in range(1, steps + 1):
            hist[t] = step()
            n_steps = t
            if stop and (t % STOP_CHECK_INTERVAL == 0 or t == steps):
                done = torch.isin(hist[: t + 1], stop_t).any(0)
                if bool((done | ~real).all()):
                    break
        return n_steps

    # ------------------------------------------------------------ internals

    def _forward_batch(self, bs, input_ids, positions, out_cache_loc, spec_info):
        return ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=None,
            seq_lens=None,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=0,
            positions=positions,
            spec_info=spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

    def _run_model(self, input_ids, positions, fb, metadata):
        self.model_runner.attn_backend.forward_metadata = metadata
        with forward_context(self._ctx):
            out = self.model_runner.model.forward(input_ids, positions, fb)
        return out.next_token_logits

    def _extend(
        self,
        input_ids,
        positions,
        out_cache_loc,
        qo_indptr,
        kv_indptr,
        kv_indices,
        max_extend_len,
        logit_index,
        q_sid,
    ) -> torch.Tensor:
        """One eager EXTEND forward with explicit attention metadata; returns
        the logits of the `logit_index` rows."""
        slot_sid = self.slot_sid if q_sid is not None else None
        spec = ParallelPromptInput(
            positions=positions, logit_index=logit_index, q_sid=q_sid, slot_sid=slot_sid
        )
        fb = self._forward_batch(
            qo_indptr.numel() - 1, input_ids, positions, out_cache_loc, spec
        )
        md = _metadata(max_extend_len, kv_indptr, kv_indices, qo_indptr, q_sid, slot_sid)
        logits = self._run_model(input_ids, positions, fb, md)
        if self.logits_hook is not None and q_sid is not None:
            self.logits_hook(logits)
        return logits

    def _step_state(self, R: int, P: int) -> SimpleNamespace:
        """Static buffers of a decode step with R requests of P streams."""
        key = (R, P)
        if key in self._steps:
            return self._steps[key]
        dev = self.device
        n = R * P
        st = SimpleNamespace(
            R=R,
            P=P,
            graph=None,
            req_rows=torch.zeros(R, dtype=torch.int64, device=dev),
            prefix=torch.zeros(R, dtype=torch.int64, device=dev),
            tokens=torch.zeros(n, dtype=torch.int32, device=dev),
            positions=torch.zeros(n, dtype=torch.int64, device=dev),
            out_cache_loc=torch.zeros(n, dtype=torch.int64, device=dev),
            kv_indptr=torch.zeros(R + 1, dtype=torch.int32, device=dev),
            kv_indices=torch.zeros(
                R * self.req_to_token.shape[1], dtype=torch.int32, device=dev
            ),
            qo_indptr=torch.arange(0, n + 1, P, dtype=torch.int32, device=dev),
            q_sid=torch.arange(P, dtype=torch.int32, device=dev).repeat(R),
            ar_p=torch.arange(P, dtype=torch.int64, device=dev),
        )
        spec = ParallelPromptInput(
            positions=st.positions,
            logit_index=torch.arange(n, dtype=torch.int64, device=dev),
            q_sid=st.q_sid,
            slot_sid=self.slot_sid,
        )
        st.fb = self._forward_batch(R, st.tokens, st.positions, st.out_cache_loc, spec)
        st.metadata = _metadata(
            P, st.kv_indptr, st.kv_indices, st.qo_indptr, st.q_sid, self.slot_sid
        )
        self._steps[key] = st
        return st

    def _step(self, st: SimpleNamespace) -> None:
        """One decode step, entirely on the GPU and in place on `st`."""
        st.kv_indptr[1:].copy_(torch.cumsum(st.prefix, 0))
        create_flashinfer_kv_indices_triton[(st.R,)](
            self.req_to_token,
            st.req_rows,
            st.prefix,
            st.kv_indptr,
            None,
            st.kv_indices,
            self.req_to_token.stride(0),
        )
        loc = self.req_to_token[st.req_rows[:, None], st.prefix[:, None] + st.ar_p]
        st.out_cache_loc.copy_(loc.view(-1))
        logits = self._run_model(st.tokens, st.positions, st.fb, st.metadata)
        if self.logits_hook is not None and st.graph is None:
            self.logits_hook(logits)
        st.tokens.copy_(torch.argmax(logits, dim=-1))
        st.positions.add_(1)
        st.prefix.add_(st.P)

    def _capture(self, st: SimpleNamespace) -> None:
        """Capture `_step` as a CUDA graph. Warmup steps write the KV of the
        first decode step, which the first real step rewrites identically."""
        saved = [t.clone() for t in (st.prefix, st.positions, st.tokens)]

        def restore():
            for t, s in zip((st.prefix, st.positions, st.tokens), saved):
                t.copy_(s)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._step(st)
                restore()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._graph_pool):
            self._step(st)
        restore()
        self._graph_pool = graph.pool()
        st.graph = graph
