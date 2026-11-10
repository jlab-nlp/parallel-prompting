72 eval runs, groups/run [1], output [128] tokens/stream

### Wall time (s): mean ± 95% CI over eval seeds (CV%)

| Doc tokens | Q/doc | n | 2. SGLang + PC (FlashInfer, graphs) | 4. SGLang + PC (Triton, graphs) | 6. SGLang + PC + PP (Triton, graphs) |
|---|---|---|---|---|---|
| 8192 | 64 | 6 | 1.97 ± 0.01 (0.3%) | 2.06 ± 0.00 (0.2%) | 2.11 ± 0.04 (1.8%) P=16 |
| 8192 | 256 | 6 | 6.60 ± 0.07 (1.0%) | 7.35 ± 0.10 (1.3%) | 4.05 ± 0.05 (1.2%) P=32 |
| 16000 | 64 | 6 | 3.20 ± 0.01 (0.3%) | 3.64 ± 0.02 (0.4%) | 3.36 ± 0.07 (1.8%) P=16 |
| 16000 | 256 | 6 | 11.37 ± 0.03 (0.3%) | 12.86 ± 0.04 (0.3%) | 6.05 ± 0.07 (1.1%) P=32 |

### Paired speedup of 6 over each config: geometric mean [95% bootstrap CI], (min over seeds)

| Doc tokens | Q/doc | 2. SGLang + PC (FlashInfer, graphs) | 4. SGLang + PC (Triton, graphs) |
|---|---|---|---|
| 8192 | 64 | 0.93x [0.92, 0.95] (0.91) | 0.97x [0.96, 0.99] (0.96) |
| 8192 | 256 | 1.63x [1.62, 1.64] (1.62) | 1.82x [1.80, 1.82] (1.79) |
| 16000 | 64 | 0.95x [0.94, 0.97] (0.93) | 1.08x [1.07, 1.10] (1.06) |
| 16000 | 256 | 1.88x [1.86, 1.90] (1.85) | 2.13x [2.11, 2.14] (2.10) |
| **all** | | **1.28x** [1.13, 1.46] (0.91) | **1.42x** [1.25, 1.62] (0.96) |

### Does 6 win every paired run?

| Doc tokens | Q/doc | wins vs all 2 other(s) | slowest 6 (s) | fastest stock run (s) | worst-case margin |
|---|---|---|---|---|---|
| 8192 | 64 | 0/6 | 2.14 | 1.96 | 0.91x |
| 8192 | 256 | 6/6 | 4.10 | 6.50 | 1.58x |
| 16000 | 64 | 0/6 | 3.43 | 3.18 | 0.93x |
| 16000 | 256 | 6/6 | 6.13 | 11.34 | 1.85x |

Paired runs where 6 beat all 2 other config(s): **12/24**. One-sided exact sign test against a coin flip versus the best competitor: p = 5.8e-01.
"Worst-case margin" = fastest run of any stock config on that workload / slowest run of 6; > 1 means the ranges do not overlap at all.

### 6. SGLang + PC + PP (Triton, graphs): calibration seed, wall time (s) by P (bold = P used for eval)

| Doc tokens | Q/doc | P=4 | P=8 | P=16 | P=32 | P=64 | P=128 | P=256 |
|---|---|---|---|---|---|---|---|---|
| 8192 | 64 | 3.32 | 2.17 | **2.11** | 2.55 | 4.29 | – | – |
| 8192 | 256 | 9.12 | 6.11 | 4.69 | **3.94** | 5.49 | 6.17 | 8.78 |
| 16000 | 64 | 6.21 | 3.67 | **3.35** | 4.10 | 6.63 | – | – |
| 16000 | 256 | 17.06 | 10.69 | 7.59 | **5.94** | 8.21 | 8.49 | 11.09 |
