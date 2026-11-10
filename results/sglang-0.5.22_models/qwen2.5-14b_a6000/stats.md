72 eval runs, groups/run [1], output [128] tokens/stream

### Wall time (s): mean ± 95% CI over eval seeds (CV%)

| Doc tokens | Q/doc | n | 2. SGLang + PC (FlashInfer, graphs) | 4. SGLang + PC (Triton, graphs) | 6. SGLang + PC + PP (Triton, graphs) |
|---|---|---|---|---|---|
| 8192 | 64 | 6 | 13.32 ± 0.02 (0.1%) | 14.34 ± 0.02 (0.2%) | 9.20 ± 0.05 (0.5%) P=16 |
| 8192 | 256 | 6 | 38.41 ± 0.18 (0.5%) | 43.71 ± 0.52 (1.1%) | 19.76 ± 0.15 (0.7%) P=32 |
| 16000 | 64 | 6 | 20.56 ± 0.22 (1.0%) | 22.67 ± 0.58 (2.4%) | 11.78 ± 0.03 (0.2%) P=16 |
| 16000 | 256 | 6 | 65.31 ± 0.30 (0.4%) | 74.50 ± 1.28 (1.6%) | 26.37 ± 0.03 (0.1%) P=32 |

### Paired speedup of 6 over each config: geometric mean [95% bootstrap CI], (min over seeds)

| Doc tokens | Q/doc | 2. SGLang + PC (FlashInfer, graphs) | 4. SGLang + PC (Triton, graphs) |
|---|---|---|---|
| 8192 | 64 | 1.45x [1.44, 1.45] (1.44) | 1.56x [1.55, 1.57] (1.55) |
| 8192 | 256 | 1.94x [1.93, 1.96] (1.92) | 2.21x [2.19, 2.23] (2.18) |
| 16000 | 64 | 1.75x [1.73, 1.76] (1.72) | 1.92x [1.89, 1.96] (1.85) |
| 16000 | 256 | 2.48x [2.47, 2.48] (2.47) | 2.82x [2.79, 2.86] (2.76) |
| **all** | | **1.87x** [1.73, 2.02] (1.44) | **2.08x** [1.91, 2.26] (1.55) |

### Does 6 win every paired run?

| Doc tokens | Q/doc | wins vs all 2 other(s) | slowest 6 (s) | fastest stock run (s) | worst-case margin |
|---|---|---|---|---|---|
| 8192 | 64 | 6/6 | 9.25 | 13.31 | 1.44x |
| 8192 | 256 | 6/6 | 19.89 | 38.23 | 1.92x |
| 16000 | 64 | 6/6 | 11.81 | 20.30 | 1.72x |
| 16000 | 256 | 6/6 | 26.41 | 64.99 | 2.46x |

Paired runs where 6 beat all 2 other config(s): **24/24**. One-sided exact sign test against a coin flip versus the best competitor: p = 6.0e-08.
"Worst-case margin" = fastest run of any stock config on that workload / slowest run of 6; > 1 means the ranges do not overlap at all.

### 6. SGLang + PC + PP (Triton, graphs): calibration seed, wall time (s) by P (bold = P used for eval)

| Doc tokens | Q/doc | P=4 | P=8 | P=16 | P=32 | P=64 | P=128 | P=256 |
|---|---|---|---|---|---|---|---|---|
| 8192 | 64 | 15.86 | 11.40 | **9.18** | 9.80 | 12.35 | – | – |
| 8192 | 256 | 51.87 | 32.86 | 23.28 | **19.90** | 24.48 | 27.11 | 36.70 |
| 16000 | 64 | 25.44 | 16.27 | **11.79** | 12.54 | 16.00 | – | – |
| 16000 | 256 | 90.19 | 51.60 | 33.17 | **26.32** | 32.38 | 34.68 | 44.33 |
