36 eval runs, groups/run [1], output [128] tokens/stream

### Wall time (s): mean ± 95% CI over eval seeds (CV%)

| Doc tokens | Q/doc | n | 2. SGLang + PC (FlashInfer, graphs) | 4. SGLang + PC (Triton, graphs) | 6. SGLang + PC + PP (Triton, graphs) |
|---|---|---|---|---|---|
| 8192 | 64 | 6 | 32.56 ± 0.14 (0.4%) | 32.98 ± 0.19 (0.5%) | 8.92 ± 0.03 (0.3%) P=16 |
| 8192 | 256 | 6 | 123.38 ± 0.24 (0.2%) | 113.57 ± 1.09 (0.9%) | 35.91 ± 0.05 (0.1%) P=16 |

### Paired speedup of 6 over each config: geometric mean [95% bootstrap CI], (min over seeds)

| Doc tokens | Q/doc | 2. SGLang + PC (FlashInfer, graphs) | 4. SGLang + PC (Triton, graphs) |
|---|---|---|---|
| 8192 | 64 | 3.65x [3.64, 3.66] (3.63) | 3.70x [3.68, 3.71] (3.67) |
| 8192 | 256 | 3.44x [3.43, 3.44] (3.43) | 3.16x [3.14, 3.18] (3.12) |
| **all** | | **3.54x** [3.48, 3.60] (3.43) | **3.42x** [3.28, 3.56] (3.12) |

### Does 6 win every paired run?

| Doc tokens | Q/doc | wins vs all 2 other(s) | slowest 6 (s) | fastest stock run (s) | worst-case margin |
|---|---|---|---|---|---|
| 8192 | 64 | 6/6 | 8.97 | 32.34 | 3.61x |
| 8192 | 256 | 6/6 | 35.96 | 112.22 | 3.12x |

Paired runs where 6 beat all 2 other config(s): **12/12**. One-sided exact sign test against a coin flip versus the best competitor: p = 2.4e-04.
"Worst-case margin" = fastest run of any stock config on that workload / slowest run of 6; > 1 means the ranges do not overlap at all.

### 6. SGLang + PC + PP (Triton, graphs): calibration seed, wall time (s) by P (bold = P used for eval)

| Doc tokens | Q/doc | P=4 | P=8 | P=16 | P=32 | P=64 |
|---|---|---|---|---|---|---|
| 8192 | 64 | 14.23 | 10.64 | **8.89** | 9.39 | 11.26 |
| 8192 | 256 | 58.44 | 43.27 | **35.94** | 37.76 | 45.38 |
