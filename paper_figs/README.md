# Figures: Parallel Prompting (PP) inside SGLang

Model: CodeLlama-7B-Instruct, bf16. Every speedup is **paired**: the same
documents and questions (same seed) are timed under each config on the same
GPU. Results are the geometric mean of the per-seed ratios with a 95% bootstrap
CI. PP's P is chosen on a separate calibration seed and never on the seeds it is
scored on. "Best stock" means the faster of stock Triton + CUDA graphs and stock
FlashInfer + CUDA graphs on that seed.

Regenerate from the repo root with `python paper_figs/make_figs.py` (needs matplotlib). It writes
`fig*.pdf`/`fig*.png` and `summary.md`, which has every number behind each figure.

| File (paper figure) | Main idea | Experiment (stage in `pp_bench/run_figs.sh`) |
|---|---|---|
| `fig1_portability` (Figure 3) | The speedup isn't a quirk of one SGLang version or one GPU. | `core`: 8k/16k-token docs × 64/128/256 questions, 6 seeds in two passes with reversed config order, on a 2×2 grid of {SGLang 0.4.2, 0.5.22} × {A100-80GB, RTX A6000} |
| `fig2_scaling` (Figure 4, left) | The gain grows with the KV that stock decode re-reads, i.e. document length × questions per document. | `scaling`: 0.5k–4k-token docs × 16/64/256 questions, plus the 8k/16k docs from `core` |
| `fig3_outlen` (Figure 4, right) | Long answers shrink the gain. A PP request's KV row also holds its sibling streams' generated tokens, so each decode step scans P × (answer so far) extra tokens, which it masks out. A stream-major KV layout, or a smaller P for long answers, would reduce this. | `outlen_*`: 32/512/1024 output tokens (128 comes from `core`), 8k-token doc × 128 questions |
| `fig4_parallel_size` (Figure 5) | Wall time is U-shaped in P, with the best P between 16 and 64 on every setup, so P needs little tuning. | calibration runs of `core` (every P timed on a held-out seed) |
| `fig5_groups` (Figure 6, Appendix A) | Batching many documents doesn't erase the gain. | `groups_*`: 1/4/16 docs per batch, 4k-token docs × 64 questions |

Data per cell (version × GPU) is in [`../results/`](../results/README.md).
