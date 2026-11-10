"""CPU check of ContinuousPPEngine's bookkeeping against ParallelPromptEngine
(no GPU or SGLang install needed; only torch):

    python pp_bench/test_cb_mock.py

The model is replaced by a deterministic fake whose next token depends on the
KV a stream can see (document + its own slots, via req_to_token and slot_sid),
its current token and its position. So any mistake in rows, prefixes,
positions, slot layout, hist indexing or token budgets changes the outputs.
"""

import importlib.util
import os
import random
import sys
import types
from unittest import mock

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "overlay", "sglang", "srt",
                    "parallel_prompt")
DOC = -1
V = 97

torch.cuda.synchronize = lambda: None
for name in [
    "sglang", "sglang.benchmark", "sglang.benchmark.one_batch", "sglang.kernels",
    "sglang.kernels.ops", "sglang.kernels.ops.kvcache", "sglang.kernels.ops.kvcache.kv_indices",
    "sglang.srt", "sglang.srt.entrypoints", "sglang.srt.entrypoints.engine",
    "sglang.srt.layers", "sglang.srt.layers.attention",
    "sglang.srt.layers.attention.triton_backend", "sglang.srt.mem_cache",
    "sglang.srt.mem_cache.base_prefix_cache", "sglang.srt.mem_cache.cache_init_params",
    "sglang.srt.mem_cache.radix_cache", "sglang.srt.model_executor",
    "sglang.srt.model_executor.forward_batch_info",
    "sglang.srt.model_executor.forward_context", "sglang.srt.parallel_prompt",
    "sglang.srt.parallel_prompt.mask", "sglang.srt.runtime_context", "sglang.srt.server_args",
]:
    sys.modules[name] = mock.MagicMock()
sys.modules["sglang.srt.parallel_prompt.mask"].DOC_STREAM = DOC


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, file))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


E = load("sglang.srt.parallel_prompt.engine", "engine.py")
C = load("sglang.srt.parallel_prompt.continuous", "continuous.py")


class Alloc:
    def __init__(self, n):
        self.n = n
        self.free_l = list(range(1, n))  # slot 0 is never handed out
        self.used = torch.zeros(n, dtype=torch.bool)

    def available_size(self):
        return len(self.free_l)

    def alloc(self, k):
        if k > len(self.free_l):
            return None
        idx, self.free_l = self.free_l[:k], self.free_l[k:]
        assert not self.used[idx].any()
        self.used[idx] = True
        return torch.tensor(idx, dtype=torch.int64)

    def free(self, t):
        idx = t.flatten().tolist()
        assert self.used[idx].all(), "double free"
        self.used[idx] = False
        self.free_l += idx


class Rows:
    def __init__(self, n):
        self.free_slots = list(range(n))

    def available_size(self):
        return len(self.free_slots)

    def alloc_rows(self, k):
        if k > len(self.free_slots):
            return None
        r, self.free_slots = self.free_slots[:k], self.free_slots[k:]
        return r

    def free_rows(self, rows):
        assert not set(rows) & set(self.free_slots), "row double free"
        self.free_slots += rows


class Fake:
    """Fake model shared by both engines (mixed in before the engine)."""

    def setup(self, n_slots, n_rows, row_len):
        self.device = "cpu"
        self.cuda_graph = False
        self.tree = None
        self.prefix_hit_tokens = 0
        self.max_prefill_tokens = 64
        self.compact_finished = True
        self.compactions = 0
        self.logits_hook = None
        self.kv_alloc = Alloc(n_slots)
        pool = Rows(n_rows)
        self.model_runner = types.SimpleNamespace(req_to_token_pool=pool)
        self.req_to_token = torch.zeros((n_rows, row_len), dtype=torch.int32)
        self.slot_sid = torch.full((n_slots,), DOC, dtype=torch.int32)
        self.kv_tok = torch.zeros(n_slots, dtype=torch.int64)
        self._steps = {}
        self._graph_pool = None
        self.stats = E.GenerateStats()
        return self

    def default_stop_ids(self):
        return [0]

    def make_docs(self, docs):
        slots = []
        for d in docs:
            s = self.kv_alloc.alloc(len(d))
            self.kv_tok[s] = torch.tensor(d)
            self.slot_sid[s] = DOC
            slots.append(s)
        return E.DocCache(lens=[len(d) for d in docs], slots=slots)

    def _extend(self, input_ids, positions, out_cache_loc, qo_indptr, kv_indptr,
                kv_indices, max_extend_len, logit_index, q_sid):
        assert self.kv_alloc.used[out_cache_loc].all()
        self.kv_tok[out_cache_loc] = input_ids.to(torch.int64)
        li = logit_index
        tok = (input_ids[li].to(torch.int64) * 13 + positions[li] * 5 + 2) % V
        # Depend on the request's document too (its kv prefix).
        out = []
        qo = qo_indptr.tolist()
        kvp = kv_indptr.tolist()
        for k, x in enumerate(li.tolist()):
            r = next(i for i in range(len(qo) - 1) if qo[i] <= x < qo[i + 1])
            doc = kv_indices[kvp[r] : kvp[r + 1]].to(torch.int64)
            out.append((int(tok[k]) + int(self.kv_tok[doc].sum())) % V)
        return torch.nn.functional.one_hot(torch.tensor(out), V).float()

    def _step(self, st):
        rows = st.req_rows.tolist()
        new = []
        for b, row in enumerate(rows):
            pre = int(st.prefix[b])
            locs = self.req_to_token[row, pre : pre + st.P].to(torch.int64)
            assert self.kv_alloc.used[locs].all(), "writes outside allocated slots"
            assert (self.slot_sid[locs] == torch.arange(st.P, dtype=torch.int32)).all()
            toks = st.tokens[b * st.P : (b + 1) * st.P].to(torch.int64)
            self.kv_tok[locs] = toks
            vis = self.req_to_token[row, : pre + st.P].to(torch.int64)
            assert self.kv_alloc.used[vis].all(), "reads outside allocated slots"
            sid = self.slot_sid[vis]
            for s in range(st.P):
                m = (sid == DOC) | (sid == s)
                S = int(self.kv_tok[vis[m]].sum())
                pos = int(st.positions[b * st.P + s])
                new.append((S + int(toks[s]) * 31 + pos * 7 + 1) % V)
        st.tokens.copy_(torch.tensor(new, dtype=torch.int32))
        st.positions.add_(1)
        st.prefix.add_(st.P)


class FakePP(Fake, E.ParallelPromptEngine):
    pass


class FakeCB(Fake, C.ContinuousPPEngine):
    pass


def run(cls, docs, questions, T, stop, P, n_slots, n_rows, wave):
    eng = object.__new__(cls).setup(n_slots, n_rows, 512)
    if wave:
        os.environ["SGLANG_PP_MAX_WAVE"] = str(wave)
    else:
        os.environ.pop("SGLANG_PP_MAX_WAVE", None)
    cache = eng.make_docs(docs)
    free0 = eng.kv_alloc.available_size()
    rows0 = eng.model_runner.req_to_token_pool.available_size()
    out = eng.generate_cached(cache, questions, T, stop_token_ids=stop, parallel_size=P)
    assert eng.kv_alloc.available_size() == free0, "leaked slots"
    assert eng.model_runner.req_to_token_pool.available_size() == rows0, "leaked rows"
    return out, eng


def main():
    rng = random.Random(0)
    n = 0
    for trial in range(60):
        G = rng.randint(1, 5)
        docs = [[rng.randrange(1, V) for _ in range(rng.randint(3, 20))] for _ in range(G)]
        questions = [[[rng.randrange(1, V) for _ in range(rng.randint(1, 6))]
                      for _ in range(rng.randint(1, 9))] for _ in range(G)]
        T = rng.choice([1, 2, 3, 7, 20, 40])
        stop = rng.choice([[], [0], [0, 5, 11], [3, 7, 13, 17, 23]])
        P = rng.choice([None, 1, 2, 3, 4])
        wave = rng.choice([0, 1, 2, 4])
        n_slots = rng.choice([4000, 600])
        try:
            ref, _ = run(FakePP, docs, questions, T, stop, P, n_slots, 64, wave)
        except MemoryError:
            continue
        out, eng = run(FakeCB, docs, questions, T, stop, P, n_slots, 64, wave)
        assert out == ref, (trial, T, stop, P, wave, out, ref)
        n += 1
    print(f"{n} random cases: pp-cb outputs == pp-on outputs, no leaked slots or rows")

    # Memory-limited admission: the pool fits fewer requests than the wave size.
    docs = [[5] * 10]
    questions = [[[rng.randrange(1, V)] * 3 for _ in range(12)]]
    ref, _ = run(FakePP, docs, questions, 30, [0, 5], 1, 4000, 64, 0)
    for n_slots in (10 + 1 + 16 + 33 * 2 + 5, 10 + 1 + 16 + 33 * 5 + 3):
        out, eng = run(FakeCB, docs, questions, 30, [0, 5], 1, n_slots, 64, 0)
        assert out == ref
        print(f"pool {n_slots} slots: equal, {eng.compactions} refills")
    # Nothing fits: MemoryError, not a silent drop.
    try:
        run(FakeCB, docs, questions, 30, [0], 1, 10 + 1 + 16 + 20, 64, 0)
        raise AssertionError("expected MemoryError")
    except MemoryError as e:
        print("too-small pool raises:", e)


if __name__ == "__main__":
    main()
