72 eval runs, groups/run [1], output [128] tokens/stream

### Wall time (s): mean ± 95% CI over eval seeds (CV%)

| Doc tokens | Q/doc | n | 2. SGLang + PC (FlashInfer, graphs) | 4. SGLang + PC (Triton, graphs) | 6. SGLang + PC + PP (Triton, graphs) |
|---|---|---|---|---|---|
| 8192 | 64 | 6 | 5.09 ± 0.01 (0.1%) | 5.51 ± 0.01 (0.2%) | 4.54 ± 0.01 (0.2%) P=16 |
| 8192 | 256 | 6 | 13.15 ± 0.03 (0.2%) | 15.27 ± 0.01 (0.1%) | 9.44 ± 0.01 (0.1%) P=32 |
| 16000 | 64 | 6 | 7.07 ± 0.01 (0.1%) | 7.85 ± 0.02 (0.2%) | 5.81 ± 0.01 (0.2%) P=16 |
| 16000 | 256 | 6 | 20.34 ± 0.04 (0.2%) | 24.54 ± 0.02 (0.1%) | 12.65 ± 0.03 (0.3%) P=32 |

### Paired speedup of 6 over each config: geometric mean [95% bootstrap CI], (min over seeds)

| Doc tokens | Q/doc | 2. SGLang + PC (FlashInfer, graphs) | 4. SGLang + PC (Triton, graphs) |
|---|---|---|---|
| 8192 | 64 | 1.12x [1.12, 1.12] (1.12) | 1.21x [1.21, 1.22] (1.21) |
| 8192 | 256 | 1.39x [1.39, 1.39] (1.39) | 1.62x [1.62, 1.62] (1.61) |
| 16000 | 64 | 1.22x [1.22, 1.22] (1.21) | 1.35x [1.35, 1.35] (1.35) |
| 16000 | 256 | 1.61x [1.60, 1.61] (1.60) | 1.94x [1.94, 1.94] (1.93) |
| **all** | | **1.32x** [1.25, 1.40] (1.12) | **1.51x** [1.40, 1.62] (1.21) |

### Does 6 win every paired run?

| Doc tokens | Q/doc | wins vs all 2 other(s) | slowest 6 (s) | fastest stock run (s) | worst-case margin |
|---|---|---|---|---|---|
| 8192 | 64 | 6/6 | 4.55 | 5.08 | 1.12x |
| 8192 | 256 | 6/6 | 9.46 | 13.11 | 1.39x |
| 16000 | 64 | 6/6 | 5.82 | 7.06 | 1.21x |
| 16000 | 256 | 6/6 | 12.69 | 20.27 | 1.60x |

Paired runs where 6 beat all 2 other config(s): **24/24**. One-sided exact sign test against a coin flip versus the best competitor: p = 6.0e-08.
"Worst-case margin" = fastest run of any stock config on that workload / slowest run of 6; > 1 means the ranges do not overlap at all.

### 6. SGLang + PC + PP (Triton, graphs): calibration seed, wall time (s) by P (bold = P used for eval)

| Doc tokens | Q/doc | P=4 | P=8 | P=16 | P=32 | P=64 | P=128 | P=256 |
|---|---|---|---|---|---|---|---|---|
| 8192 | 64 | 7.07 | 5.65 | **4.60** | 4.93 | 6.43 | – | – |
| 8192 | 256 | 21.23 | 14.03 | 10.18 | **9.49** | 12.38 | 13.62 | 18.60 |
| 16000 | 64 | 10.75 | 7.99 | **5.78** | 6.32 | 8.48 | – | – |
| 16000 | 256 | 35.91 | 21.63 | 14.06 | **12.67** | 16.68 | 17.56 | 22.65 |
