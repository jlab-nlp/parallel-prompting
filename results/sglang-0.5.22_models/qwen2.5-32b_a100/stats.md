36 eval runs, groups/run [1], output [128] tokens/stream

### Wall time (s): mean ± 95% CI over eval seeds (CV%)

| Doc tokens | Q/doc | n | 2. SGLang + PC (FlashInfer, graphs) | 4. SGLang + PC (Triton, graphs) | 6. SGLang + PC + PP (Triton, graphs) |
|---|---|---|---|---|---|
| 8192 | 64 | 6 | 11.16 ± 0.26 (2.2%) | 12.84 ± 0.09 (0.6%) | 9.63 ± 0.01 (0.1%) P=16 |
| 8192 | 256 | 6 | 33.54 ± 0.46 (1.3%) | 40.40 ± 0.20 (0.5%) | 24.42 ± 0.09 (0.3%) P=32 |

### Paired speedup of 6 over each config: geometric mean [95% bootstrap CI], (min over seeds)

| Doc tokens | Q/doc | 2. SGLang + PC (FlashInfer, graphs) | 4. SGLang + PC (Triton, graphs) |
|---|---|---|---|
| 8192 | 64 | 1.16x [1.14, 1.18] (1.12) | 1.33x [1.33, 1.34] (1.32) |
| 8192 | 256 | 1.37x [1.36, 1.39] (1.36) | 1.65x [1.65, 1.66] (1.64) |
| **all** | | **1.26x** [1.20, 1.32] (1.12) | **1.49x** [1.40, 1.57] (1.32) |

### Does 6 win every paired run?

| Doc tokens | Q/doc | wins vs all 2 other(s) | slowest 6 (s) | fastest stock run (s) | worst-case margin |
|---|---|---|---|---|---|
| 8192 | 64 | 6/6 | 9.65 | 10.84 | 1.12x |
| 8192 | 256 | 6/6 | 24.57 | 33.11 | 1.35x |

Paired runs where 6 beat all 2 other config(s): **12/12**. One-sided exact sign test against a coin flip versus the best competitor: p = 2.4e-04.
"Worst-case margin" = fastest run of any stock config on that workload / slowest run of 6; > 1 means the ranges do not overlap at all.

### 6. SGLang + PC + PP (Triton, graphs): calibration seed, wall time (s) by P (bold = P used for eval)

| Doc tokens | Q/doc | P=4 | P=8 | P=16 | P=32 | P=64 | P=128 |
|---|---|---|---|---|---|---|---|
| 8192 | 64 | 15.19 | 12.28 | **9.65** | 10.62 | 14.88 | – |
| 8192 | 256 | 51.16 | 34.27 | 28.16 | **24.66** | 32.62 | 47.91 |
