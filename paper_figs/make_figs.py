"""Insight figures for the paper from the PP-vs-SGLang sweeps (eval runs only).

Every speedup is paired: the same seed (same documents and questions) is timed
under both configs on the same GPU, and the per-seed ratios are summarised by
their geometric mean with a 95% bootstrap CI. Cells with no data are skipped, so
this can be rerun while sweeps are still going.

    python paper_figs/make_figs.py            # writes paper_figs/fig*.{pdf,png} + summary.md
"""

import json
import math
import os
import random
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper_figs")
RES = os.path.join(ROOT, "results")
# (label, {stage: [results.jsonl ...]}) per (SGLang version, GPU).
CELLS = [
    ("0.4.2 · A100", {"core": [f"{RES}/sglang-0.4.2_a100/A/results.jsonl",
                               f"{RES}/sglang-0.4.2_a100/B/results.jsonl"]}),
    ("0.4.2 · A6000", f"{RES}/sglang-0.4.2_a6000"),
    ("0.5.22 · A100", f"{RES}/sglang-0.5.22_a100"),
    ("0.5.22 · A6000", f"{RES}/sglang-0.5.22_a6000"),
]
STAGES = ["core", "scaling", "groups_4", "groups_16", "outlen_32", "outlen_512", "outlen_1024"]
STOCK_GRAPH = ["triton-graph", "flashinfer-graph"]
# Categorical slots 1-4 of the default validated palette (blue, orange, aqua, yellow).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 2, "lines.markersize": 6, "savefig.bbox": "tight", "savefig.dpi": 200,
})


def cell_paths(spec):
    if isinstance(spec, dict):
        return spec
    return {s: [f"{spec}/{s}/results.jsonl"] for s in STAGES}


def load(paths):
    """{(config, doc, Q): {rep: wall}} and calibration rows, from eval/calib runs."""
    wall, calib = defaultdict(dict), []
    for p in paths:
        if not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            if r.get("phase") == "eval":
                wall[(r["config"], r["doc_tokens"], r["questions"])][r["rep"]] = r["wall_s"]
            elif r.get("phase") == "calib":
                calib.append(r)
    return wall, calib


DATA = {label: {s: load(ps) for s, ps in cell_paths(spec).items()} for label, spec in CELLS}


def geomean(xs):
    return math.exp(statistics.fmean(math.log(x) for x in xs))


def boot_ci(xs, n=4000, seed=0):
    rng = random.Random(seed)
    logs = [math.log(x) for x in xs]
    m = sorted(statistics.fmean(rng.choices(logs, k=len(logs))) for _ in range(n))
    return math.exp(m[int(0.025 * n)]), math.exp(m[int(0.975 * n) - 1])


def ratios(wall, base, doc=None, q=None):
    """Per-seed base/pp-graph wall ratios over matching workloads. base may be
    'best' = fastest stock graph config on that seed (only seeds where all ran)."""
    out = []
    for (c, d, qq), pp in wall.items():
        if c != "pp-graph" or (doc and d != doc) or (q and qq != q):
            continue
        bases = [base] if base != "best" else STOCK_GRAPH
        runs = [wall.get((b, d, qq), {}) for b in bases]
        if base == "best" and any(not r for r in runs):
            runs = [r for r in runs if r]  # cell without FlashInfer: best of what ran
        for rep, t in pp.items():
            if runs and all(rep in r for r in runs):
                out.append(min(r[rep] for r in runs) / t)
    return out


def summary(xs):
    if not xs:
        return None
    lo, hi = boot_ci(xs)
    return geomean(xs), lo, hi, len(xs)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)


LINES = []


def note(s=""):
    LINES.append(s)
    print(s)


def fig1_portability():
    """Main idea: the gain is not a quirk of one SGLang version or one GPU."""
    note("## Fig 1: speedup of PP (Triton, graphs) on 8k/16k docs x 64-256 questions\n")
    note("| Cell | vs Triton + graphs | vs FlashInfer + graphs | paired runs | PP wins |")
    note("|---|---|---|---|---|")
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    width = 0.36
    for j, (base, name) in enumerate([("triton-graph", "vs stock Triton + graphs"),
                                      ("flashinfer-graph", "vs stock FlashInfer + graphs")]):
        for i, (label, _) in enumerate(CELLS):
            s = summary(ratios(DATA[label]["core"][0], base))
            if not s:
                continue
            g, lo, hi, _ = s
            x = i + (j - 0.5) * width
            labelled = name in ax.get_legend_handles_labels()[1]
            ax.bar(x, g, width - 0.04, color=COLORS[j], label=None if labelled else name)
            ax.errorbar(x, g, [[g - lo], [hi - g]], color=INK, lw=1, capsize=2)
            ax.text(x, hi + 0.15, f"{g:.1f}×", ha="center", va="bottom", fontsize=7.5, color=INK)
    for label, _ in CELLS:
        w = DATA[label]["core"][0]
        t, f = summary(ratios(w, "triton-graph")), summary(ratios(w, "flashinfer-graph"))
        best = ratios(w, "best")
        if not t:
            continue
        fs = f"{f[0]:.2f}× [{f[1]:.2f}, {f[2]:.2f}]" if f else "not run"
        note(f"| {label} | {t[0]:.2f}× [{t[1]:.2f}, {t[2]:.2f}] | {fs} | {len(best)} | "
             f"{sum(x > 1 for x in best)}/{len(best)} |")
    ax.axhline(1, color=MUTED, lw=1, ls="--")
    ax.set_xticks(range(len(CELLS)), [c for c, _ in CELLS])
    ax.set_ylabel("Speedup of PP (×, paired)")
    ax.set_title("PP's speedup holds on both SGLang versions and both GPUs", loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    save(fig, "fig1_portability")


def present(stage):
    return [c for c, _ in CELLS if DATA[c].get(stage, ({}, []))[0]]


def fig2_scaling():
    """Main idea: the gain grows with the KV that stock decode re-reads (doc x Q)."""
    cells = present("scaling")
    if not cells:
        return
    note("\n## Fig 2: speedup vs best stock (graphs) by document length\n")
    fig, axes = plt.subplots(1, len(cells), figsize=(2.9 * len(cells), 2.6), sharey=True, squeeze=False)
    for ax, cell in zip(axes[0], cells):
        wall = {**DATA[cell]["scaling"][0], **DATA[cell]["core"][0]}
        for k, q in enumerate([16, 64, 256]):
            pts = []
            for d in sorted({d for (_, d, qq) in wall if qq == q}):
                s = summary(ratios(wall, "best", d, q))
                if s:
                    pts.append((d, *s[:3]))
                    note(f"- {cell}, {d} tokens, Q={q}: {s[0]:.2f}× [{s[1]:.2f}, {s[2]:.2f}]")
            if not pts:
                continue
            xs, g, lo, hi = zip(*pts)
            ax.fill_between(xs, lo, hi, color=COLORS[k], alpha=0.18, lw=0)
            ax.plot(xs, g, "-o", color=COLORS[k], label=f"{q} questions/doc",
                    markeredgecolor="white", markeredgewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks([512, 1024, 2048, 4096, 8192, 16000], ["0.5k", "1k", "2k", "4k", "8k", "16k"])
        ax.minorticks_off()
        ax.axhline(1, color=MUTED, lw=1, ls="--")
        ax.set_title(cell, loc="left")
        ax.set_xlabel("Document length (tokens)")
    axes[0][0].set_ylabel("Speedup vs best stock (×)")
    axes[0][0].legend(frameon=False, loc="upper left")
    fig.suptitle("The gain grows with document length × questions per document",
                 x=0.01, ha="left", fontsize=10, y=1.03)
    save(fig, "fig2_scaling")


def fig3_outlen():
    """Main idea: the gain shrinks for long answers. A PP request's KV row holds
    every stream's generated tokens, so each decode step also scans (and masks
    out) its siblings' tokens, which grow with output length."""
    stages = {32: "outlen_32", 128: "core", 512: "outlen_512", 1024: "outlen_1024"}
    cells = [c for c in present("outlen_512")]
    if not cells:
        return
    note("\n## Fig 3: speedup vs best stock by output length (8192 tokens, 128 questions)\n")
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for k, cell in enumerate(cells):
        pts = []
        for n, st in stages.items():
            s = summary(ratios(DATA[cell].get(st, ({}, []))[0], "best", 8192, 128))
            if s:
                pts.append((n, *s[:3]))
                note(f"- {cell}, {n} output tokens: {s[0]:.2f}× [{s[1]:.2f}, {s[2]:.2f}]")
        if pts:
            xs, g, lo, hi = zip(*pts)
            ax.fill_between(xs, lo, hi, color=COLORS[k], alpha=0.18, lw=0)
            ax.plot(xs, g, "-o", color=COLORS[k], label=cell, markeredgecolor="white", markeredgewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(stages), [str(n) for n in stages])
    ax.minorticks_off()
    ax.axhline(1, color=MUTED, lw=1, ls="--")
    ax.set_xlabel("Output tokens per question")
    ax.set_ylabel("Speedup vs best stock (×)")
    ax.set_title("Long answers shrink the gain: each stream also scans its siblings' tokens",
                 loc="left")
    ax.legend(frameon=False)
    save(fig, "fig3_outlen")


def fig4_parallel_size():
    """Main idea: wall time is U-shaped in P, with the same sweet spot everywhere."""
    cells = [c for c, _ in CELLS if any(r["config"] == "pp-graph" for r in DATA[c]["core"][1])]
    if not cells:
        return
    note("\n## Fig 4: best parallel size P on the calibration seed (16000-token docs)\n")
    fig, axes = plt.subplots(1, len(cells), figsize=(2.5 * len(cells), 2.5), sharey=True, squeeze=False)
    for ax, cell in zip(axes[0], cells):
        rows = [r for r in DATA[cell]["core"][1] if r["config"] == "pp-graph" and r["doc_tokens"] == 16000]
        for k, q in enumerate([64, 128, 256]):
            by_p = defaultdict(list)  # two calibration passes in some cells: average them
            for r in rows:
                if r["questions"] == q:
                    by_p[r["parallel_size"]].append(r["wall_s"])
            pts = sorted((p, statistics.fmean(w)) for p, w in by_p.items())
            if not pts:
                continue
            best = min(w for _, w in pts)
            ps, ws = zip(*pts)
            ax.plot(ps, [w / best for w in ws], "-o", color=COLORS[k], label=f"{q} questions",
                    markeredgecolor="white", markeredgewidth=1)
            note(f"- {cell}, Q={q}: best P={ps[ws.index(best)]}")
        ax.set_xscale("log", base=2)
        ax.set_xticks([4, 16, 64, 256], ["4", "16", "64", "256"])
        ax.minorticks_off()
        ax.set_title(cell, loc="left")
        ax.set_xlabel("Streams per request P")
    axes[0][0].set_ylabel("Wall time / best")
    axes[0][0].legend(frameon=False)
    fig.suptitle("Choosing P: a U-shaped curve with its minimum at 16–32 on every setup",
                 x=0.01, ha="left", fontsize=10, y=1.03)
    save(fig, "fig4_parallel_size")


def fig5_groups():
    """Main idea: batching many documents does not erase the gain."""
    stages = {1: "scaling", 4: "groups_4", 16: "groups_16"}
    cells = present("groups_4")
    if not cells:
        return
    note("\n## Fig 5: speedup vs best stock by documents per batch (4096 tokens, 64 questions)\n")
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for k, cell in enumerate(cells):
        pts = []
        for g, st in stages.items():
            s = summary(ratios(DATA[cell].get(st, ({}, []))[0], "best", 4096, 64))
            if s:
                pts.append((g, *s[:3]))
                note(f"- {cell}, {g} docs/batch: {s[0]:.2f}× [{s[1]:.2f}, {s[2]:.2f}]")
        if pts:
            xs, gm, lo, hi = zip(*pts)
            ax.fill_between(xs, lo, hi, color=COLORS[k], alpha=0.18, lw=0)
            ax.plot(xs, gm, "-o", color=COLORS[k], label=cell, markeredgecolor="white", markeredgewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(stages), [str(g) for g in stages])
    ax.minorticks_off()
    ax.axhline(1, color=MUTED, lw=1, ls="--")
    ax.set_xlabel("Documents per batch")
    ax.set_ylabel("Speedup vs best stock (×)")
    ax.set_title("The gain survives batching many documents", loc="left")
    ax.legend(frameon=False)
    save(fig, "fig5_groups")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig1_portability()
    fig2_scaling()
    fig3_outlen()
    fig4_parallel_size()
    fig5_groups()
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(LINES) + "\n")
