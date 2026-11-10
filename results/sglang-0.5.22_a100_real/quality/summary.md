### a100 / qwen2.5-7b-instruct / quality

| config | engine | time (s) | passes | F1 | EM | n | P | same answers as sgl-on | notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| sgl-on | sglang 0.5.22 | 133.2 | 2 | acc 76.13 | - | 2086 | - | 100.0% |  |
| sgl-triton-on | sglang 0.5.22 | 155.9 | 2 | acc 76.22 | - | 2086 | - | 82.6% |  |
| pp-radix | sglang 0.5.22 + ParallelPromptEngine | 73.6 | 2 | acc 75.89 | - | 2086 | 0 | 82.8% | prefix-cache hit tokens 1946 |
| pp-cb-radix | sglang 0.5.22 + ContinuousPPEngine | 73.7 | 2 | acc 75.89 | - | 2086 | 0 | 83.1% | prefix-cache hit tokens 1946 |

pp-cb-radix speedup: sgl-on 1.81x, sgl-triton-on 2.12x

pp-radix speedup: sgl-on 1.81x, sgl-triton-on 2.12x

GPU / node: NVIDIA A100-SXM4-80GB
Nondeterministic across passes: none
