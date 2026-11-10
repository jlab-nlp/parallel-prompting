"""Offline Parallel Prompting engine on top of SGLang's ModelRunner.

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

Like bench_one_batch, this drives the ModelRunner directly (no scheduler).
Requires `--attention-backend triton`.
"""

import os
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from sglang.bench_one_batch import load_model
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.layers.attention.flashinfer_backend import (
    create_flashinfer_kv_indices_triton,
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.parallel_prompt.mask import DOC_STREAM, ParallelPromptInput
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

# How often (in decode steps) to check on the CPU whether every stream has
# emitted a stop token. Only used when stop tokens are given.
STOP_CHECK_INTERVAL = 16


@dataclass
class DocCache:
    """KV of documents prefilled once, shared by every chunk of their group."""

    lens: List[int]
    slots: List[torch.Tensor]


@dataclass
class GenerateStats:
    doc_prefill_s: float = 0.0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    decode_steps: int = 0
    output_tokens: int = 0
    num_reqs: int = 0
    parallel_size: int = 0


class ParallelPromptEngine:
    def __init__(self, server_args: ServerArgs, cuda_graph: bool = False):
        assert (
            server_args.attention_backend == "triton"
        ), "Parallel Prompting is only wired for the triton backend"
        assert server_args.tp_size == 1, "prototype supports tp_size=1"
        _set_envs_and_config(server_args)
        port_args = PortArgs.init_new(server_args)
        self.model_runner, self.tokenizer = load_model(server_args, port_args, 0)
        self.device = self.model_runner.device
        self.cuda_graph = cuda_graph
        self.stats = GenerateStats()
        # Called with the logits of every eager PP forward (prefill, then each
        # decode step); used by diag_logits.py.
        self.logits_hook: Optional[Callable[[torch.Tensor], None]] = None

        runner = self.model_runner
        self.req_to_token = runner.req_to_token_pool.req_to_token
        num_slots = runner.token_to_kv_pool.get_key_buffer(0).shape[0]
        self.slot_sid = torch.full(
            (num_slots,), DOC_STREAM, dtype=torch.int32, device=self.device
        )
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

    @torch.no_grad()
    def prefill_docs(self, docs: Sequence[List[int]]) -> DocCache:
        """Prefill each document once; its KV is shared by all its questions."""
        pool = self.model_runner.token_to_kv_pool
        lens = [len(d) for d in docs]
        slots = []
        for n in lens:
            s = pool.alloc(n)
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

    def release_docs(self, cache: DocCache) -> None:
        self.model_runner.token_to_kv_pool.free(torch.cat(cache.slots))

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
        wave = max(1, min(len(reqs), runner.token_to_kv_pool.available_size() // need))
        wave = 1 << (wave.bit_length() - 1)
        # Tests can force small waves (e.g. SGLANG_PP_MAX_WAVE=2) on any pool size.
        wave = min(wave, int(os.environ.get("SGLANG_PP_MAX_WAVE", 0)) or wave)
        hist = []
        stats = GenerateStats(num_reqs=len(reqs), parallel_size=P)
        for w in range(0, len(reqs), wave):
            h, pre, dec, n = self._run_wave(
                cache, questions, reqs[w : w + wave], P, max_new_tokens, stop
            )
            hist += h
            stats.prefill_s += pre
            stats.decode_s += dec
            stats.decode_steps += n

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
        rows = runner.req_to_token_pool.alloc(R)
        if rows is None:
            raise MemoryError("not enough request slots")

        ids, pos, sids, own_slots, prefix0, kv_idx, logit_rows = [], [], [], [], [], [], []
        next_pos = []
        tok_pt = 0
        for (g, idx), row in zip(reqs, rows):
            doc_len = cache.lens[g]
            last = max(i for i in idx if i is not None)
            qs = [questions[g][last if i is None else i] for i in idx]
            q_len = sum(len(q) for q in qs)
            own = runner.token_to_kv_pool.alloc(q_len + P * steps)
            if own is None:
                raise MemoryError("KV pool too small for this workload")
            own_slots.append(own)
            self.req_to_token[row, :doc_len] = cache.slots[g]
            self.req_to_token[row, doc_len : doc_len + own.numel()] = own
            row_sid = [torch.full((len(q),), s, dtype=torch.int32) for s, q in enumerate(qs)]
            row_sid.append(torch.arange(P, dtype=torch.int32).repeat(steps))
            self.slot_sid[own.to(torch.int64)] = torch.cat(row_sid).to(dev)
            for s, q in enumerate(qs):
                ids.extend(q)
                pos.append(torch.arange(doc_len, doc_len + len(q)))
                sids.append(row_sid[s])
                tok_pt += len(q)
                logit_rows.append(tok_pt - 1)
                next_pos.append(doc_len + len(q))
            kv_idx.append(cache.slots[g])
            prefix0.append(doc_len + q_len)

        q_lens = [p - cache.lens[g] for p, (g, _) in zip(prefix0, reqs)]
        first = self._extend(
            input_ids=torch.tensor(ids, dtype=torch.int32, device=dev),
            positions=torch.cat(pos).to(dev),
            out_cache_loc=torch.cat(
                [o[:n] for o, n in zip(own_slots, q_lens)]
            ).to(torch.int64),
            qo_indptr=torch.tensor([0] + q_lens, device=dev).cumsum(0).to(torch.int32),
            kv_indptr=torch.tensor([0] + [cache.lens[g] for g, _ in reqs], device=dev)
            .cumsum(0)
            .to(torch.int32),
            kv_indices=torch.cat(kv_idx).to(torch.int32),
            max_extend_len=max(q_lens),
            logit_index=torch.tensor(logit_rows, dtype=torch.int64, device=dev),
            q_sid=torch.cat(sids).to(dev),
        ).argmax(-1)

        hist = torch.empty((max_new_tokens, R * P), dtype=torch.int32, device=dev)
        hist[0] = first
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - tic

        tic = time.perf_counter()
        n_steps = 0
        if steps > 0:
            st = self._step_state(R, P)
            st.req_rows.copy_(torch.tensor(rows, dtype=torch.int64))
            st.prefix.copy_(torch.tensor(prefix0, dtype=torch.int64))
            st.positions.copy_(torch.tensor(next_pos, dtype=torch.int64))
            st.tokens.copy_(first)
            if self.cuda_graph and st.graph is None:
                self._capture(st)
            real = torch.tensor(
                [i is not None for _, idx in reqs for i in idx], device=dev
            )
            stop_t = torch.tensor(stop, dtype=torch.int32, device=dev)
            for t in range(1, max_new_tokens):
                if st.graph is not None:
                    st.graph.replay()
                else:
                    self._step(st)
                hist[t] = st.tokens
                n_steps = t
                if stop and (t % STOP_CHECK_INTERVAL == 0 or t == steps):
                    done = torch.isin(hist[: t + 1], stop_t).any(0)
                    if bool((done | ~real).all()):
                        break
        torch.cuda.synchronize()
        decode_s = time.perf_counter() - tic

        hist = hist[: n_steps + 1].T.tolist()
        runner.token_to_kv_pool.free(torch.cat(own_slots))
        for row in rows:
            runner.req_to_token_pool.free(row)
        return hist, prefill_s, decode_s, n_steps

    @torch.no_grad()
    def generate_independent(
        self,
        prompts: Sequence[List[int]],
        max_new_tokens: int,
        stop_token_ids: Optional[List[int]] = None,
    ) -> List[List[int]]:
        """Reference: greedy-decode each prompt as its own request through the
        unmodified SGLang extend/decode path (as in bench_one_batch)."""
        stop = set(self.default_stop_ids() if stop_token_ids is None else stop_token_ids)
        runner = self.model_runner
        reqs = []
        for i, ids in enumerate(prompts):
            req = Req(
                rid=i,
                origin_input_text="",
                origin_input_ids=list(ids),
                sampling_params=SamplingParams(
                    temperature=0, max_new_tokens=max_new_tokens
                ),
            )
            req.prefix_indices = []
            req.fill_ids = req.origin_input_ids
            req.extend_input_len = len(req.fill_ids)
            reqs.append(req)
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=runner.req_to_token_pool,
            token_to_kv_pool=runner.token_to_kv_pool,
            tree_cache=None,
            model_config=runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            enable_custom_logit_processor=False,
        )
        batch.prepare_for_extend()
        outputs = [[t] for t in self._forward(batch)]
        for _ in range(max_new_tokens - 1):
            if all(o[-1] in stop for o in outputs):
                break
            batch.output_ids = torch.tensor(
                [o[-1] for o in outputs], dtype=torch.int32, device=self.device
            )
            batch.prepare_for_decode()
            for o, t in zip(outputs, self._forward(batch)):
                o.append(t)
        for req, seq_len in zip(reqs, batch.seq_lens.tolist()):
            runner.token_to_kv_pool.free(
                runner.req_to_token_pool.req_to_token[req.req_pool_idx, :seq_len]
            )
            runner.req_to_token_pool.free(req.req_pool_idx)
        trimmed = []
        for o in outputs:
            cut = next((i + 1 for i, t in enumerate(o) if t in stop), len(o))
            trimmed.append(o[:cut])
        return trimmed

    # ------------------------------------------------------------ internals

    def _forward_batch(self, bs, input_ids, positions, out_cache_loc, spec_info):
        runner = self.model_runner
        return ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=None,
            seq_lens=None,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=0,
            positions=positions,
            req_to_token_pool=runner.req_to_token_pool,
            token_to_kv_pool=runner.token_to_kv_pool,
            attn_backend=runner.attn_backend,
            spec_info=spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

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
        spec = ParallelPromptInput(
            positions=positions,
            logit_index=logit_index,
            q_sid=q_sid,
            slot_sid=self.slot_sid if q_sid is not None else None,
        )
        fb = self._forward_batch(
            qo_indptr.numel() - 1, input_ids, positions, out_cache_loc, spec
        )
        self.model_runner.attn_backend.forward_metadata = (
            None,
            max_extend_len,
            kv_indptr,
            kv_indices,
            qo_indptr,
            None,
            None,
            spec.q_sid,
            spec.slot_sid,
        )
        logits = self.model_runner.model.forward(input_ids, positions, fb)
        logits = logits.next_token_logits
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
        st.metadata = (
            None,
            P,
            st.kv_indptr,
            st.kv_indices,
            st.qo_indptr,
            None,
            None,
            st.q_sid,
            self.slot_sid,
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
        self.model_runner.attn_backend.forward_metadata = st.metadata
        logits = self.model_runner.model.forward(st.tokens, st.positions, st.fb)
        logits = logits.next_token_logits
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

    def _forward(self, batch: ScheduleBatch) -> List[int]:
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        logits_output = self.model_runner.forward(forward_batch)
        return torch.argmax(logits_output.next_token_logits, dim=-1).tolist()
