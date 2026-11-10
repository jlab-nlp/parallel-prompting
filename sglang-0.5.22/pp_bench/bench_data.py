"""Work files for the real benchmarks of the paper (Section 5.5): QuALITY and LooGLE.

Same format as qa_data.py (read by eval_qa.py): each group is one shared
prefix (instruction + document) with its questions, tokenized once so every
engine gets identical token ids:

    prefix = instruction + document      tokenized once, with BOS
    suffix = the question part           tokenized without special tokens

| name     | source                                                 | groups              | questions | answers           |
|----------|--------------------------------------------------------|---------------------|-----------|-------------------|
| quality  | nyu-mll/quality GitHub, v1.0.1 htmlstripped dev        | 115 articles        | 2,086     | 4-way MC letter   |
| loogle   | HF bigai-nlco/LooGLE, shortdep_qa, test                | 105 Wikipedia docs  | 1,951     | short free-form   |

Continuous-batching study (Table 4): `quality_short` and `quality_mixed` are
QuALITY in the model's chat template, so the model ends its answer with EOS
(the plain prompt above never stops before the budget), both with a 512-token
budget. `quality_short` asks for the letter only; `quality_mixed` asks every
other article's questions for a short explanation ending in "Answer: X", so
requests (whose streams all come from one article) differ widely in length.

Prompts: LooGLE uses its official prompt (config/task2prompt.json) and answer
budget of 300 tokens (task2maxlen.json). QuALITY has no official LLM prompt;
ours is below, with a 16-token budget.

Every engine generates to EOS or the budget (no stop strings: the PP engine
stops on token ids only); answers are cut afterwards at the first blank line.
LooGLE is scored by token recall (also token F1 and ROUGE-L), QuALITY by accuracy.

Groups whose prompt (prefix + longest question + answer budget) does not fit
--max-context tokens are dropped and counted in the work file, e.g. most
LooGLE documents need a 32k-context model.

    python pp_bench/bench_data.py --model-path <hf id or path> --model-id <hf id> \\
        --out pp_bench/out/bench/work --max-context 32768
"""

import argparse
import collections
import gzip
import json
import os
import re
import string
import urllib.request

SUFFIX_NEWLINE_CUT = "\n\n"
QUALITY_URL = ("https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1/"
               "QuALITY.v1.0.1.htmlstripped.{split}")
LETTERS = "ABCDEFGHIJKLMNOP"

SPECS = {
    "quality": {"max_new_tokens": 16, "metric": "mc_letter"},
    "loogle": {"max_new_tokens": 300, "metric": "free_form"},
    "quality_short": {"max_new_tokens": 512, "metric": "mc_final", "chat": True},
    "quality_mixed": {"max_new_tokens": 512, "metric": "mc_final", "chat": True},
}
DATASETS = list(SPECS)
CALIB_QUESTIONS = 512


def _fetch(url, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, os.path.basename(url.split("?")[0]))
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


# ------------------------------------------------------------------ loaders
# Each yields (group key, prefix text, suffix text, question id, gold).

QUALITY_PREFIX = (
    "Read the article below and answer the multiple-choice question about it.\n\n"
    "Article:\n{article}\n\n"
)


def _quality(split, cache):
    path = _fetch(QUALITY_URL.format(split="dev" if split == "eval" else "train"), cache)
    for line in open(path, encoding="utf-8"):
        art = json.loads(line)
        prefix = QUALITY_PREFIX.format(article=art["article"].strip())
        for q in art["questions"]:
            opts = "".join(f"{LETTERS[i]}. {o}\n" for i, o in enumerate(q["options"]))
            suffix = (f"Question: {q['question']}\nOptions:\n{opts}"
                      "Answer with the letter of the correct option.\nAnswer:")
            yield art["article_id"], prefix, suffix, q["question_unique_id"], {
                "letter": LETTERS[q["gold_label"] - 1], "n_options": len(q["options"])}


QUALITY_SHORT = "Answer with only the letter of the correct option."
QUALITY_EXPLAIN = ("Explain your reasoning in a few sentences, then end with \"Answer: X\", "
                   "where X is the letter of the correct option.")


def _quality_chat(split, cache, mixed):
    """QuALITY questions with an instruction per article (chat template added
    in build_groups): every other article asks for an explanation if mixed."""
    path = _fetch(QUALITY_URL.format(split="dev" if split == "eval" else "train"), cache)
    for k, line in enumerate(open(path, encoding="utf-8")):
        art = json.loads(line)
        prefix = QUALITY_PREFIX.format(article=art["article"].strip())
        instr = QUALITY_EXPLAIN if mixed and k % 2 else QUALITY_SHORT
        for q in art["questions"]:
            opts = "".join(f"{LETTERS[i]}. {o}\n" for i, o in enumerate(q["options"]))
            yield art["article_id"], prefix, f"Question: {q['question']}\nOptions:\n{opts}{instr}", \
                q["question_unique_id"], {"letter": LETTERS[q["gold_label"] - 1],
                                          "n_options": len(q["options"])}


LOOGLE_PREFIX = "Please answer the question based on the long texts below. \n{input}\n"


def _loogle(split, cache):
    from datasets import load_dataset

    # LooGLE has only a test split: calibrate on the long-dependency questions
    # (other questions about the same kind of documents; untimed).
    cfg = "shortdep_qa" if split == "eval" else "longdep_qa"
    for ex in load_dataset("bigai-nlco/LooGLE", cfg, split="test"):
        yield ex["doc_id"], LOOGLE_PREFIX.format(input=ex["context"]), \
            f"Question: {ex['question']}\nAnswer: ", ex["id"], {"text": [ex["answer"]]}


LOADERS = {"quality": _quality, "loogle": _loogle,
           "quality_short": lambda split, cache: _quality_chat(split, cache, False),
           "quality_mixed": lambda split, cache: _quality_chat(split, cache, True)}


def chat_split(tok, prefix, suffix):
    """The chat-template prompt for prefix + suffix as one user turn, split
    back into the shared part (template head + prefix) and the question part
    (suffix + template tail)."""
    full = tok.apply_chat_template([{"role": "user", "content": prefix + suffix}],
                                   tokenize=False, add_generation_prompt=True)
    i = full.index(prefix) + len(prefix)
    return full[:i], full[i:]


def build_groups(tok, dataset, split, cache, max_context, max_questions=None):
    """Groups that fit max_context, gold answers, and how many groups /
    questions were dropped for not fitting."""
    groups, refs, prefix_ids = collections.OrderedDict(), {}, {}
    budget = SPECS[dataset]["max_new_tokens"]
    for key, prefix, suffix, qid, gold in LOADERS[dataset](split, cache):
        if qid in refs:
            continue
        if SPECS[dataset].get("chat"):
            prefix, suffix = chat_split(tok, prefix, suffix)
        if key not in groups:
            if max_questions is not None and len(refs) >= max_questions:
                break
            groups[key] = {"doc": tok.encode(prefix), "qs": [], "qids": []}
        g = groups[key]
        g["qs"].append(tok.encode(suffix, add_special_tokens=False))
        g["qids"].append(qid)
        refs[qid] = gold
    kept, dropped_g, dropped_q = [], 0, 0
    for g in groups.values():
        if len(g["doc"]) + max(map(len, g["qs"])) + budget > max_context:
            dropped_g += 1
            dropped_q += len(g["qs"])
            for i in g["qids"]:
                refs.pop(i)
        else:
            kept.append(g)
    return kept, refs, {"groups": dropped_g, "questions": dropped_q}


# ------------------------------------------------------------------ metrics


def normalize_answer(s):
    """SQuAD normalization."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred, gold):
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    same = sum((collections.Counter(p) & collections.Counter(g)).values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def token_recall(pred, gold):
    """Share of the gold answer's tokens that appear in the prediction."""
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not g:
        return float(not p)
    return sum((collections.Counter(p) & collections.Counter(g)).values()) / len(g)


def rouge_l(pred, gold):
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    prev = [0] * (len(g) + 1)
    for x in p:
        cur = [0]
        for j, y in enumerate(g):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(g)
    return 2 * prec * rec / (prec + rec)


def final_letter(text):
    """The answer letter of an explained answer: "answer is (X)", else
    "Answer: X", else the last standalone letter."""
    m = re.search(r"answer is \(?([A-J])\)?", text)
    if m:
        return m.group(1)
    m = re.search(r".*[aA]nswer:\s*([A-J])", text)
    if m:
        return m.group(1)
    m = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", text, re.DOTALL)
    return m.group(0) if m else None


def answer_text(tok, dataset, output_ids):
    text = tok.decode(output_ids, skip_special_tokens=True)
    if SPECS[dataset].get("chat"):  # ends at EOS; explanations may hold blank lines
        return text.strip()
    return text.split(SUFFIX_NEWLINE_CUT)[0].strip()


def score(dataset, pred, gold):
    """Per-question scores in [0, 1], as {metric name: value}."""
    metric = SPECS[dataset]["metric"]
    if metric == "mc_letter":
        m = re.search(r"\b([A-D])\b", pred)
        return {"acc": float(bool(m) and m.group(1) == gold["letter"])}
    if metric == "mc_final":
        return {"acc": float(final_letter(pred) == gold["letter"])}
    if metric == "free_form":
        return {"recall": max(token_recall(pred, a) for a in gold["text"]),
                "f1": max(token_f1(pred, a) for a in gold["text"]),
                "rouge_l": max(rouge_l(pred, a) for a in gold["text"])}
    raise ValueError(metric)


# ------------------------------------------------------------------ main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="tokenizer: hub id or local path")
    parser.add_argument("--model-id", default=None, help="hub id (names the output dir)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--max-context", type=int, required=True,
                        help="model context length: groups that do not fit are dropped")
    parser.add_argument("--cache", default=os.path.join(os.environ.get("HF_HOME", "/tmp"),
                                                        "bench_downloads"))
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model_id = args.model_id or args.model_path
    tag = os.path.basename(model_id.rstrip("/")).lower()
    for d in args.datasets:
        groups, refs, dropped = build_groups(tok, d, "eval", args.cache, args.max_context)
        calib, _, _ = build_groups(tok, d, "calib", args.cache, args.max_context,
                                   max_questions=CALIB_QUESTIONS)
        path = os.path.join(args.out, tag, f"{d}.json.gz")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        work = {"dataset": d, "model": model_id, "tokenizer_path": args.model_path,
                "max_new_tokens": SPECS[d]["max_new_tokens"], "metric": SPECS[d]["metric"],
                "max_context": args.max_context, "dropped": dropped,
                "eos_token_id": tok.eos_token_id, "groups": groups, "refs": refs,
                "calib_groups": calib}
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(work, f)
        nq = sum(len(g["qs"]) for g in groups)
        doc = [len(g["doc"]) for g in groups] or [0]
        print(f"{d}: {len(groups)} groups, {nq} questions (dropped {dropped['groups']} groups / "
              f"{dropped['questions']} questions over {args.max_context} tokens), prefix tokens "
              f"mean {sum(doc) / len(doc):.0f} max {max(doc)}, questions/group mean "
              f"{nq / max(len(groups), 1):.1f} max {max([len(g['qs']) for g in groups] or [0])}, "
              f"calibration {sum(len(g['qs']) for g in calib)} questions -> {path}", flush=True)


if __name__ == "__main__":
    main()
