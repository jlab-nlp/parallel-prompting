### a100 / qwen2.5-7b-instruct / loogle

| config | engine | time (s) | passes | F1 | EM | n | P | same answers as sgl-on | notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| sgl-on | sglang 0.5.22 | 403.6 | 2 | f1 17.29, rouge_l 16.61 | - | 1761 | - | 100.0% |  |
| sgl-triton-on | sglang 0.5.22 | 568.0 | 2 | f1 17.41, rouge_l 16.70 | - | 1761 | - | 23.9% |  |
| pp-radix | sglang 0.5.22 + ParallelPromptEngine | 508.3 | 2 | f1 17.36, rouge_l 16.63 | - | 1761 | 0 | 24.8% | prefix-cache hit tokens 1212 |
| pp-cb-radix | sglang 0.5.22 + ContinuousPPEngine | 507.3 | 2 | f1 17.33, rouge_l 16.61 | - | 1761 | 0 | 24.2% | prefix-cache hit tokens 1212 |

pp-cb-radix speedup: sgl-on 0.80x, sgl-triton-on 1.12x

pp-radix speedup: sgl-on 0.79x, sgl-triton-on 1.12x

GPU / node: NVIDIA A100-SXM4-80GB
Nondeterministic across passes: none
