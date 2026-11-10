"""Consistency report for a calibrated sweep (bench_pp_vs_sglang.py --eval-reps).

Answers "is config 6 (PC + PP, Triton, graphs) faster on every input, or did it
get lucky?" using only eval runs, whose PP parallel size was picked on a
separate calibration seed. Runs with the same (doc tokens, Q, rep) saw the same
documents in every config, so they are compared pairwise.

    python pp_bench/make_stats.py pp_bench/out/sweep_long/*/results.jsonl
"""

import json
import math
import random
import statistics
import sys
from collections import defaultdict

from make_table import CONFIGS

TARGET = "pp-graph"
STOCK = ["flashinfer", "flashinfer-graph", "triton", "triton-graph"]
NAMES = dict(CONFIGS)
SHORT = {c: n.split(". ", 1)[0] for c, n in CONFIGS}
# Two-sided 95% Student-t quantiles by degrees of freedom.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042}


def t975(df):
    return T975.get(df) or T975[max(k for k in T975 if k <= df)]


def geomean(xs):
    return math.exp(statistics.fmean(math.log(x) for x in xs))


def boot_ci(xs, n=10000, seed=0):
    """95% percentile-bootstrap CI of the geometric mean."""
    rng = random.Random(seed)
    logs = [math.log(x) for x in xs]
    means = sorted(statistics.fmean(rng.choices(logs, k=len(logs))) for _ in range(n))
    return math.exp(means[int(0.025 * n)]), math.exp(means[int(0.975 * n) - 1])


def sign_test_p(wins, n):
    """One-sided exact sign test: P(>= wins of n | each config equally likely to win)."""
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n


def main(paths):
    rows = [json.loads(line) for p in paths for line in open(p)]
    ev = [r for r in rows if r.get("phase") == "eval"]
    wall = defaultdict(dict)  # (doc, q) -> config -> rep -> seconds
    pp_p = defaultdict(set)
    for r in ev:
        wall[(r["doc_tokens"], r["questions"])].setdefault(r["config"], {})[r["rep"]] = r["wall_s"]
        if r["config"] == TARGET:
            pp_p[(r["doc_tokens"], r["questions"])].add(r["parallel_size"])
    workloads = sorted(wall)
    configs = [c for c, _ in CONFIGS if any(c in wall[w] for w in workloads)]
    others = [c for c in configs if c != TARGET]
    groups = {r["groups"] for r in ev}
    print(f"{len(ev)} eval runs, groups/run {sorted(groups)}, "
          f"output {sorted({r['output_len'] for r in ev})} tokens/stream\n")

    print("### Wall time (s): mean ± 95% CI over eval seeds (CV%)\n")
    head = ["Doc tokens", "Q/doc", "n"] + [NAMES[c] for c in configs]
    print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
    for w in workloads:
        n = len(wall[w].get(TARGET, {}))
        cells = []
        for c in configs:
            xs = list(wall[w].get(c, {}).values())
            if len(xs) < 2:
                cells.append("–")
                continue
            m, s = statistics.fmean(xs), statistics.stdev(xs)
            h = t975(len(xs) - 1) * s / math.sqrt(len(xs))
            p = " P=" + "/".join(map(str, sorted(pp_p[w]))) if c == TARGET else ""
            cells.append(f"{m:.2f} ± {h:.2f} ({100 * s / m:.1f}%){p}")
        print("| " + " | ".join([str(w[0]), str(w[1]), str(n)] + cells) + " |")

    print(f"\n### Paired speedup of {SHORT[TARGET]} over each config: "
          "geometric mean [95% bootstrap CI], (min over seeds)\n")
    head = ["Doc tokens", "Q/doc"] + [NAMES[c] for c in others]
    print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
    pooled = defaultdict(list)
    for w in workloads:
        t = wall[w].get(TARGET, {})
        cells = []
        for c in others:
            reps = sorted(set(t) & set(wall[w].get(c, {})))
            if not reps:
                cells.append("–")
                continue
            sp = [wall[w][c][r] / t[r] for r in reps]
            pooled[c].extend(sp)
            lo, hi = boot_ci(sp)
            cells.append(f"{geomean(sp):.2f}x [{lo:.2f}, {hi:.2f}] ({min(sp):.2f})")
        print("| " + " | ".join([str(w[0]), str(w[1])] + cells) + " |")
    cells = []
    for c in others:
        sp = pooled[c]
        lo, hi = boot_ci(sp) if sp else (0, 0)
        cells.append(f"**{geomean(sp):.2f}x** [{lo:.2f}, {hi:.2f}] ({min(sp):.2f})" if sp else "–")
    print("| **all** | | " + " | ".join(cells) + " |")

    print(f"\n### Does {SHORT[TARGET]} win every paired run?\n")
    head = ["Doc tokens", "Q/doc", f"wins vs all {len(others)} other(s)", "slowest 6 (s)",
            "fastest stock run (s)", "worst-case margin"]
    print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
    wins = total = 0
    for w in workloads:
        t = wall[w].get(TARGET, {})
        reps = [r for r in t if all(r in wall[w].get(c, {}) for c in others)]
        win = sum(t[r] < min(wall[w][c][r] for c in others) for r in reps)
        wins, total = wins + win, total + len(reps)
        stock_runs = [x for c in STOCK for x in wall[w].get(c, {}).values()]
        if not t or not stock_runs:
            continue
        worst, best = max(t.values()), min(stock_runs)
        print(f"| {w[0]} | {w[1]} | {win}/{len(reps)} | {worst:.2f} | {best:.2f} | "
              f"{best / worst:.2f}x |")
    print(f"\nPaired runs where {SHORT[TARGET]} beat all {len(others)} other config(s): "
          f"**{wins}/{total}**. One-sided exact sign test against a coin flip versus "
          f"the best competitor: p = {sign_test_p(wins, total):.1e}.")
    print("\"Worst-case margin\" = fastest run of any stock config on that workload / "
          f"slowest run of {SHORT[TARGET]}; > 1 means the ranges do not overlap at all.")

    calib = [r for r in rows if r.get("phase") == "calib"]
    for cfg in ("pp-graph", "pp"):
        cr = [r for r in calib if r["config"] == cfg]
        if not cr:
            continue
        sizes = sorted({r["parallel_size"] for r in cr})
        print(f"\n### {NAMES[cfg]}: calibration seed, wall time (s) by P "
              "(bold = P used for eval)\n")
        head = ["Doc tokens", "Q/doc"] + [f"P={p}" for p in sizes]
        print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
        by = {(r["doc_tokens"], r["questions"], r["parallel_size"]): r["wall_s"] for r in cr}
        for d, q in sorted({(r["doc_tokens"], r["questions"]) for r in cr}):
            vals = {p: by.get((d, q, p)) for p in sizes}
            best = min((v, p) for p, v in vals.items() if v is not None)[1]
            cells = ["–" if v is None else (f"**{v:.2f}**" if p == best else f"{v:.2f}")
                     for p, v in vals.items()]
            print("| " + " | ".join([str(d), str(q)] + cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
