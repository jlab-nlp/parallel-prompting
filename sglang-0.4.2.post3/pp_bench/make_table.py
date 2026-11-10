"""Summarize bench_pp_vs_sglang.py results as Markdown tables.

    python pp_bench/make_table.py pp_bench/out/sweep_a100/results.jsonl
"""

import json
import statistics
import sys
from collections import defaultdict

CONFIGS = [
    ("flashinfer", "1. SGLang + PC (FlashInfer)"),
    ("flashinfer-graph", "2. SGLang + PC (FlashInfer, graphs)"),
    ("triton", "3. SGLang + PC (Triton)"),
    ("triton-graph", "4. SGLang + PC (Triton, graphs)"),
    ("pp", "5. SGLang + PC + PP (Triton)"),
    ("pp-graph", "6. SGLang + PC + PP (Triton, graphs)"),
]


def main(path):
    runs = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        key = (r["config"], r["doc_tokens"], r["questions"], r.get("parallel_size"))
        runs[key].append(r)
    med = {k: statistics.median(x["wall_s"] for x in v) for k, v in runs.items()}
    tokens = {k: v[0]["output_tokens"] for k, v in runs.items()}
    workloads = sorted({(d, q) for _, d, q, _ in med})

    def best_pp(cfg, d, q):
        cands = [(t, p) for (c, dd, qq, p), t in med.items() if (c, dd, qq) == (cfg, d, q)]
        return min(cands) if cands else (None, None)

    print("### Wall time (s), median of repeats; PP at its best parallel size P\n")
    head = ["Doc tokens", "Q/doc"] + [n for _, n in CONFIGS] + [
        "6 vs best stock", "6 vs 4 (same kernels)"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for d, q in workloads:
        row = [str(d), str(q)]
        times = {}
        for c, _ in CONFIGS:
            if c.startswith("pp"):
                t, p = best_pp(c, d, q)
                times[c] = t
                row.append("–" if t is None else f"{t:.2f} (P={p})")
            else:
                t = med.get((c, d, q, None))
                times[c] = t
                row.append("–" if t is None else f"{t:.2f}")
        stock = [times[c] for c, _ in CONFIGS[:4] if times[c] is not None]
        pp = times["pp-graph"]
        row.append(f"{min(stock) / pp:.2f}x" if pp and stock else "–")
        row.append(f"{times['triton-graph'] / pp:.2f}x" if pp and times["triton-graph"] else "–")
        print("| " + " | ".join(row) + " |")

    print("\n### Throughput (output tokens/s)\n")
    head = ["Doc tokens", "Q/doc"] + [n.split(". ", 1)[0] for _, n in CONFIGS]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for d, q in workloads:
        row = [str(d), str(q)]
        for c, _ in CONFIGS:
            if c.startswith("pp"):
                t, p = best_pp(c, d, q)
                n = tokens.get((c, d, q, p))
            else:
                t = med.get((c, d, q, None))
                n = tokens.get((c, d, q, None))
            row.append("–" if t is None else f"{n / t:,.0f}")
        print("| " + " | ".join(row) + " |")

    for cfg in ("pp-graph", "pp"):
        sizes = sorted({p for (c, _, _, p) in med if c == cfg})
        if not sizes:
            continue
        print(f"\n### {dict(CONFIGS)[cfg]}: wall time (s) by parallel size P\n")
        head = ["Doc tokens", "Q/doc"] + [f"P={p}" for p in sizes]
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
        for d, q in workloads:
            best = best_pp(cfg, d, q)[1]
            row = [str(d), str(q)]
            for p in sizes:
                t = med.get((cfg, d, q, p))
                cell = "–" if t is None else f"{t:.2f}"
                row.append(f"**{cell}**" if p == best else cell)
            print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main(sys.argv[1])
