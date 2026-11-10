# Benchmark results

Synthetic folders hold the raw output of `pp_bench/bench_pp_vs_sglang.py`: one JSON line per timed run, with
`config`, `doc_tokens`, `questions`, `rep` (the seed), `phase` (`calib` or `eval`),
`wall_s` and, for PP, the parallel size. `gpu.txt` names the GPU each folder ran on.
Every folder was produced on a single GPU, so all paired comparisons inside it are
on the same device. Model: CodeLlama-7B-Instruct, bf16, except in `sglang-0.5.22_models/`.

| Folder | SGLang · GPU | Produced by | Contents |
|---|---|---|---|
| `sglang-0.4.2_a100/{A,B}` | 0.4.2.post3 · A100-SXM4-80GB | `run_sweep_long.sh <out> A` / `B` | long-document sweep, all six configs; pass A = seeds 2–4, pass B = seeds 5–7, in reversed config order |
| `sglang-0.4.2_a6000/` | 0.4.2.post3 · RTX A6000 | `run_figs.sh`, then `run_figs_fix.sh` | one subfolder per figure stage (`core`, `scaling`, `groups_*`, `outlen_*`) |
| `sglang-0.5.22_a100/` | 0.5.22 · A100-SXM4-80GB | `run_figs.sh`, then `run_figs_fix.sh` | same stages; `waves_check.log` is the correctness check of waved PP |
| `sglang-0.5.22_a6000/` | 0.5.22 · RTX A6000 | `run_figs.sh`, then `run_figs_fix.sh` | same stages; the run ended partway through the final `groups_*` rerun |
| `sglang-0.5.22_models/<model>_<gpu>/` | 0.5.22 · RTX A6000 (Qwen2.5-32B: A100-SXM4-80GB) | `run_models_0522.sh` | model sweep with stock FlashInfer + graphs, stock Triton + graphs and PP + graphs; `stats.md` is the `make_stats.py` report, and `check_*.log` are the correctness checks run on that model |
| `sglang-0.5.22_a6000_squad/` | 0.5.22 · RTX A6000 | `eval_squad.py` | SQuAD v1.1 dev: `summary.md` (EM, F1, paired CIs, agreement), `time_*.json` (answer time per config), `per_question.jsonl` (every answer) |
| `sglang-0.5.22_a100_real/` | 0.5.22 · A100-SXM4-80GB | `run_bench.sh`, `make_qa_table.py` | QuALITY and LooGLE, Qwen2.5-7B-Instruct: per config and pass, `<config>.p<n>.time.json` (wall time, P, prefix-cache hits, schedule) and `summary.md` (time, accuracy or F1, agreement with stock SGLang); `loogle_metrics.log` has LooGLE's token recall, F1 and ROUGE-L (`make_qa_table.py` now reports them too). Configs: `sgl-on` (FlashInfer), `sgl-triton-on`, `pp-radix`, `pp-cb-radix` |
| `sglang-0.5.22_a6000_cb/` | 0.5.22 · RTX A6000 | `run_cb_study.sh` | continuous-batching study (Table 4) on QuALITY in the chat template (`quality_short`: letter answers; `quality_mixed`: every other article explains first), `pp-radix` vs `pp-cb-radix`, P = 4; `sched` in each `time.json` counts decode steps, refills and running requests |

The `stats*.md` files are `make_stats.py` reports for those runs.

Notes:

- On the 0.4.2 · A6000 setup P is capped at 128 (`LONG_P=4,8,16,32,64,128`).
- PP points that do not fit the KV pool in one go run in waves. Those points were
  rerun by `run_figs_fix.sh` with `--warmup-full`, so the CUDA graph of every wave
  shape is captured before timing.
- Some PP settings are skipped as "KV pool too small" (for example 16 documents of
  4096 tokens on the A6000, where the documents alone exceed the pool). The figure
  script only compares runs that both configs have for the same seed.
