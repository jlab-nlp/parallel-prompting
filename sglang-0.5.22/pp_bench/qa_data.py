"""SQuAD / QuAC / DROP work files for eval_qa.py, and the work-file reader and
SQuAD scoring shared with bench_data.py and make_qa_table.py.

The prompts are the paper's original 5-shot prompts, copied verbatim into
qa_prompts.json. Every prompt is split into a shared prefix (instruction,
few-shot block and the document) and a per-question suffix:

    prefix = few-shot block + document        tokenized once, with BOS
    suffix = "Q[0]: {question}\\n\\nA[0]:"     tokenized without special tokens

and every engine is given exactly prefix_ids + suffix_ids, so all engines see
the same token ids. Questions are grouped by their document (SQuAD context,
QuAC dialogue, DROP passage); a group is one PP document with its questions.

    python pp_bench/qa_data.py --model-path <hf id> --out <out>/work

writes <out>/<model tag>/<dataset>.json.gz with the validation groups, the
gold answers and a train-split calibration/warmup set. Built once per model
in one environment, then read by every engine, so tokenizer versions of the
different images do not matter.

Scoring matches the originals: the decoded answer is cut at the first blank
line and scored with the SQuAD v2 metric (no-answer probability 0), as
`evaluate.load("squad_v2")` does; reimplemented here so it runs offline.
"""

import argparse
import collections
import gzip
import json
import os
import re
import string

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = json.load(open(os.path.join(HERE, "qa_prompts.json"), encoding="utf-8"))
SUFFIX = "Q[0]: {q}\n\nA[0]:"
MAX_NEW_TOKENS = {"squad": 10, "quac": 25, "drop": 10}
DATASETS = list(MAX_NEW_TOKENS)
CALIB_QUESTIONS = 1024


def model_tag(model_path):
    return os.path.basename(model_path.rstrip("/")).lower()


def work_path(root, model_path, dataset):
    return os.path.join(root, model_tag(model_path), f"{dataset}.json.gz")


def load_work(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ datasets


def _squad(split):
    from datasets import load_dataset

    for ex in load_dataset("rajpurkar/squad", split=split):
        yield ex["context"], PROMPTS["squad"] + ex["context"] + "\n\n", ex["question"], \
            ex["id"], {"text": ex["answers"]["text"], "answer_start": ex["answers"]["answer_start"]}


def _drop(split):
    from datasets import load_dataset

    for ex in load_dataset("ucinlp/drop", split=split):
        yield ex["passage"], PROMPTS["drop"] + ex["passage"] + "\n\n", ex["question"], \
            ex["query_id"], {"text": ex["answers_spans"]["spans"], "answer_start": [0]}


def _quac(split):
    from datasets import load_dataset

    # The allenai/quac loading script no longer runs; the hub's parquet
    # conversion holds the same dialogues.
    ds = load_dataset("allenai/quac", split=split, revision="refs/convert/parquet")
    for ex in ds:
        prefix = PROMPTS["quac"].format(
            title=ex["wikipedia_page_title"], section=ex["section_title"],
            background=ex["background"], context=ex["context"])
        for i, q in enumerate(ex["questions"]):
            yield ex["dialogue_id"], prefix, q, ex["turn_ids"][i], {
                "text": ex["answers"]["texts"][i], "answer_start": ex["answers"]["answer_starts"][i]}


LOADERS = {"squad": _squad, "quac": _quac, "drop": _drop}


def build_groups(tok, dataset, split, max_questions=None):
    """[{doc: ids, qs: [ids], qids: [str]}] in order of first appearance, plus
    {qid: gold answers}."""
    groups, refs = collections.OrderedDict(), {}
    n = 0
    for key, prefix, q, qid, ans in LOADERS[dataset](split):
        if max_questions is not None and n >= max_questions and key not in groups:
            break
        if qid in refs:
            # DROP validation repeats one query id; the original squad_v2 scoring
            # kept one of the two (its total was 9534), so do the same.
            continue
        g = groups.get(key)
        if g is None:
            g = groups[key] = {"doc": tok.encode(prefix), "qs": [], "qids": []}
        g["qs"].append(tok.encode(SUFFIX.format(q=q), add_special_tokens=False))
        g["qids"].append(qid)
        refs[qid] = ans
        n += 1
    return list(groups.values()), refs


# ------------------------------------------------------------------ metric
# Official SQuAD v2.0 evaluation (what evaluate's squad_v2 wraps), with a
# no-answer probability of 0 for every prediction.


def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _tokens(s):
    return normalize_answer(s).split() if s else []


def _f1(gold, pred):
    g, p = _tokens(gold), _tokens(pred)
    if not g or not p:
        return float(g == p)
    same = sum((collections.Counter(g) & collections.Counter(p)).values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def score(pred, answers):
    """(EM, F1) of one prediction against its gold answers, in [0, 1]."""
    golds = [a for a in answers["text"] if normalize_answer(a)] or [""]
    em = max(float(normalize_answer(a) == normalize_answer(pred)) for a in golds)
    return em, max(_f1(a, pred) for a in golds)


def answer_text(tok, output_ids):
    """Decoded answer, cut at the first blank line as the original scripts did."""
    return tok.decode(output_ids, skip_special_tokens=True).split("\n\n")[0]


# ------------------------------------------------------------------ main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="tokenizer: hub id or local path")
    parser.add_argument("--model-id", default=None, help="hub id (names the output dir)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model_id = args.model_id or args.model_path
    for d in args.datasets:
        groups, refs = build_groups(tok, d, "validation")
        calib, _ = build_groups(tok, d, "train", max_questions=CALIB_QUESTIONS)
        path = work_path(args.out, model_id, d)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        work = {"dataset": d, "model": model_id, "tokenizer_path": args.model_path, "max_new_tokens": MAX_NEW_TOKENS[d],
                "eos_token_id": tok.eos_token_id, "groups": groups, "refs": refs,
                "calib_groups": calib}
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(work, f)
        nq = sum(len(g["qs"]) for g in groups)
        doc = [len(g["doc"]) for g in groups]
        print(f"{d}: {len(groups)} groups, {nq} questions, doc tokens mean "
              f"{sum(doc) / len(doc):.0f} max {max(doc)}, max questions/group "
              f"{max(len(g['qs']) for g in groups)} -> {path}", flush=True)


if __name__ == "__main__":
    main()
