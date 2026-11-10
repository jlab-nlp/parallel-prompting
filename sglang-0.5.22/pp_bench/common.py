"""Shared workload helpers for the Parallel Prompting checks and benchmarks."""

import argparse
import os
from typing import List, Tuple

from sglang.srt.server_args import ServerArgs

HERE = os.path.dirname(os.path.abspath(__file__))
WAR_AND_PEACE = os.environ.get(
    "PP_HAYSTACK", os.path.join(HERE, "..", "..", "data", "war_and_peace.txt")
)

DOC_TEMPLATE = (
    "Read the passage and answer the question briefly.\n\nPassage:\n{doc}\n\n"
)
QUESTION_TEMPLATE = "Question: {q}\nAnswer:"


def add_server_args(parser: argparse.ArgumentParser):
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mem-fraction-static", type=float, default=0.8)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16")


def make_server_args(args) -> ServerArgs:
    return ServerArgs(
        model_path=args.model_path,
        attention_backend="triton",
        sampling_backend="pytorch",
        disable_cuda_graph=True,
        disable_radix_cache=True,
        mem_fraction_static=args.mem_fraction_static,
        context_length=args.context_length,
        dtype=args.dtype,
    )


def squad_groups(tokenizer, n_groups: int, max_q: int) -> Tuple[List, List]:
    """SQuAD v1 validation, grouped by shared context."""
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad", split="validation")
    by_ctx = {}
    for ex in ds:
        by_ctx.setdefault(ex["context"], []).append(ex["question"])
    docs, questions = [], []
    for ctx, qs in by_ctx.items():
        if len(qs) < 2:
            continue
        docs.append(tokenizer.encode(DOC_TEMPLATE.format(doc=ctx)))
        questions.append(
            [
                tokenizer.encode(QUESTION_TEMPLATE.format(q=q), add_special_tokens=False)
                for q in qs[:max_q]
            ]
        )
        if len(docs) == n_groups:
            break
    return docs, questions


def synthetic_groups(tokenizer, n_groups: int, n_q: int, doc_tokens: int):
    """Haystack from war_and_peace.txt; questions vary only in a numeric id so
    every stream has the same length (matches the paper's synthetic setup)."""
    with open(WAR_AND_PEACE, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    ids = tokenizer.encode(text[: doc_tokens * 8], add_special_tokens=False)
    docs, questions = [], []
    for g in range(n_groups):
        start = g * doc_tokens
        chunk = tokenizer.decode(ids[start : start + doc_tokens])
        docs.append(tokenizer.encode(DOC_TEMPLATE.format(doc=chunk))[: doc_tokens + 32])
        questions.append(
            [
                tokenizer.encode(
                    QUESTION_TEMPLATE.format(
                        q=f"Summarize the main point of paragraph {i + 1}."
                    ),
                    add_special_tokens=False,
                )
                for i in range(n_q)
            ]
        )
    return docs, questions


_CORPUS = {}


def bench_groups(tokenizer, n_groups: int, n_q: int, doc_tokens: int, seed: int):
    """Like synthetic_groups, but every (seed, group) gets its own document: a
    distinct war_and_peace.txt slice behind a unique tag, exactly `doc_tokens`
    tokens long. Benchmarks use a fresh seed per repeat, so a radix cache can
    only reuse the document it was explicitly given, never an earlier run."""
    key = id(tokenizer)
    if key not in _CORPUS:
        with open(WAR_AND_PEACE, encoding="utf-8", errors="ignore") as f:
            _CORPUS[key] = tokenizer.encode(f.read(), add_special_tokens=False)
    corpus = _CORPUS[key]
    span = len(corpus) - doc_tokens
    docs, questions = [], []
    for g in range(n_groups):
        start = ((seed * n_groups + g) * 7919 * doc_tokens) % span
        head = tokenizer.encode(
            f"Read the passage (document {seed}.{g}) and answer the question "
            "briefly.\n\nPassage:\n"
        )
        tail = tokenizer.encode("\n\n", add_special_tokens=False)
        docs.append(head + corpus[start : start + doc_tokens] + tail)
        questions.append(
            [
                tokenizer.encode(
                    QUESTION_TEMPLATE.format(
                        q=f"Summarize the main point of paragraph {i + 1}."
                    ),
                    add_special_tokens=False,
                )
                for i in range(n_q)
            ]
        )
    return docs, questions
