"""Summarize eval_qa.py runs: one table per GPU / model / dataset.

    python pp_bench/make_qa_table.py <out> [--work <out>/work]

Reads <root>/<gpu>/<model>/<dataset>/*.time.json and *.raw.jsonl and writes
summary.md next to them and <root>/summary.md with every cell. Scores are
from pass 1; time is the mean over passes. Answers are decoded with the
model's tokenizer here, the same way for every engine.
"""

import argparse
import collections
import glob
import json
import os

import bench_data
from qa_data import answer_text, load_work, score

ORDER = ["sgl-off", "sgl-on", "sgl-triton-on", "pp-off", "pp-on", "pp-radix", "pp-cb", "pp-cb-radix"]
BASELINES = [o for o in ORDER if not o.startswith("pp")]
DESC = {
    "sgl-on / sgl-off": "stock SGLang (FlashInfer), radix cache on / off",
    "sgl-triton-on": "stock SGLang, Triton backend (PP's backend), radix cache on",
    "pp-off": "PP, document per request (prefilled once per request of P questions)",
    "pp-on": "PP, document shared across requests (prefilled once per document)",
    "pp-radix": "PP, document shared, plus SGLang's radix prefix cache across documents",
    "pp-cb": "pp-on with continuous batching (no waves; requests admitted as others finish)",
    "pp-cb-radix": "pp-cb plus SGLang's radix prefix cache across documents",
}
# Cells deliberately not run (dataset, config) -> reason, listed under the table.
NOT_RUN = {}


def load_cell(d):
    runs = collections.defaultdict(dict)
    for t in glob.glob(os.path.join(d, "*.time.json")):
        r = json.load(open(t))
        raw = t[: -len(".time.json")] + ".raw.jsonl"
        r["outputs"] = {x["id"]: x["output_ids"] for x in map(json.loads, open(raw))}
        runs[r["label"]][r["pass_index"]] = r
    return runs


def cell_table(d, work, tok):
    runs = load_cell(d)
    refs = work["refs"]
    # Questions every run answered (all of them, unless a smoke test ran a subset).
    done = set.intersection(*(set(r[min(r)]["outputs"]) for r in runs.values()))
    ids = [i for g in work["groups"] for i in g["qids"] if i in done]
    labels = sorted(runs, key=lambda l: (ORDER.index(l) if l in ORDER else len(ORDER), l))
    rows, times = [], {}
    ref_out = runs.get("sgl-on", {}).get(1, {}).get("outputs")
    for l in labels:
        passes = runs[l]
        p1 = passes[min(passes)]
        if "metric" in work:  # bench_data.py benchmarks: their own metric
            ds = work["dataset"]
            per_q = [bench_data.score(ds, bench_data.answer_text(tok, ds, p1["outputs"][i]), refs[i])
                     for i in ids]
            keys = sorted({k for q in per_q for k in q})
            scores = {k: 100 * sum(q[k] for q in per_q if k in q) /
                      max(1, sum(k in q for q in per_q)) for k in keys}
            em_f1 = [(q.get("acc", 0.0), q.get("f1", q.get("acc", 0.0))) for q in per_q]
        else:
            ans = {i: answer_text(tok, p1["outputs"][i]) for i in ids}
            em_f1 = [score(ans[i], refs[i]) for i in ids]
            scores = None
        t = sum(p["wall_s"] for p in passes.values()) / len(passes)
        times[l] = t
        same_passes = all(p["outputs"] == p1["outputs"] for p in passes.values())
        agree = (sum(p1["outputs"][i] == ref_out[i] for i in ids) / len(ids)) if ref_out else None
        rows.append({"label": l, "time": t, "passes": len(passes),
                     "em": 100 * sum(e for e, _ in em_f1) / len(ids),
                     "f1": 100 * sum(f for _, f in em_f1) / len(ids), "n": len(p1["outputs"]),
                     "scores": scores, "hits": p1.get("prefix_hit_tokens"),
                     "P": p1.get("parallel_size"), "engine": p1["engine"],
                     "deterministic": same_passes, "agree_sgl_on": agree,
                     "flushed": p1.get("cache_flushed"), "gpu": p1["gpu"], "node": p1.get("node"),
                     "notes": notes(passes.values())})
    return rows, times


def notes(passes):
    """Settings that differ from the shared protocol, over all passes."""
    out = set()
    for p in passes:
        if p.get("mem_fraction_static") not in (0.88, None):
            out.add(f"mem {p['mem_fraction_static']}")
        if p.get("max_running_requests") not in ("sglang default", None):
            out.add(f"max running {p['max_running_requests']}")
        if p.get("max_prefill_tokens") is None and p["config"] in ("pp-on", "pp-off"):
            out.add("prefill chunk 16384")
        if p.get("cache_flushed") is False:
            out.add("no cache reset (fresh process)")
        if p.get("parallel_size_fixed"):
            out.add(f"P fixed at {p['parallel_size']} (not calibrated)")
    return ", ".join(sorted(out))


def render(title, rows, times, n_total, pending=None):
    lines = [f"### {title}", "",
             "| config | engine | time (s) | passes | F1 | EM | n | P | same answers as sgl-on | notes |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        agree = "-" if r["agree_sgl_on"] is None else f"{100 * r['agree_sgl_on']:.1f}%"
        if r["scores"]:  # bench_data metrics replace F1 / EM
            metric = ", ".join(f"{k} {v:.2f}" for k, v in r["scores"].items()) + " | -"
        else:
            metric = f"{r['f1']:.2f} | {r['em']:.2f}"
        hits = "" if r["hits"] is None else f"prefix-cache hit tokens {r['hits']}"
        notes = ", ".join(x for x in (r["notes"], hits) if x)
        lines.append(f"| {r['label']} | {r['engine']} | {r['time']:.1f} | {r['passes']} | "
                     f"{metric} | {r['n']}{'' if r['n'] == n_total else ' (!)'} | "
                     f"{r['P'] if r['P'] is not None else '-'} | {agree} | {notes} |")
    for pp in ("pp-cb-radix", "pp-cb", "pp-radix", "pp-on", "pp-off"):
        if pp in times:
            sp = ", ".join(f"{b} {times[b] / times[pp]:.2f}x" for b in BASELINES if b in times)
            lines.append(f"\n{pp} speedup: {sp}")
    if pending:
        lines.append(f"\n{pending}")
    nodes = sorted({f"{r['gpu']} @ {r['node'] or '?'}" for r in rows})
    lines += ["", "GPU / node: " + "; ".join(nodes),
              "Nondeterministic across passes: " + (", ".join(
                  r["label"] for r in rows if not r["deterministic"]) or "none"), ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--work", default=None, help="default: <root>/qa_work")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    work_root = args.work or os.path.join(args.root, "qa_work")

    from transformers import AutoTokenizer

    out = ["# " + (args.title or "SQuAD / QuAC / DROP rerun"), "", "Configs:", ""]
    out += [f"- `{k}`: {v}" for k, v in DESC.items()] + [""]
    for gpu in sorted(os.listdir(args.root)):
        gdir = os.path.join(args.root, gpu)
        if gpu == "qa_work" or not os.path.isdir(gdir):
            continue
        for model in sorted(os.listdir(gdir)):
            for ds in ("squad", "quac", "drop") + tuple(bench_data.DATASETS):
                d = os.path.join(gdir, model, ds)
                if not glob.glob(os.path.join(d, "*.time.json")):
                    continue
                w = glob.glob(os.path.join(work_root, model, f"{ds}.json.gz"))[0]
                work = load_work(w)
                try:
                    tok = AutoTokenizer.from_pretrained(work["tokenizer_path"])
                except OSError:
                    tok = AutoTokenizer.from_pretrained(work["model"])
                rows, times = cell_table(d, work, tok)
                pending = [v for (k, c), v in NOT_RUN.items()
                           if k == ds and not glob.glob(os.path.join(d, f"{c}.p*.time.json"))]
                text = render(f"{gpu} / {model} / {ds}", rows, times, len(work["refs"]),
                              "\n".join(pending) or None)
                open(os.path.join(d, "summary.md"), "w").write(text)
                out.append(text)
    open(os.path.join(args.root, "summary.md"), "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
