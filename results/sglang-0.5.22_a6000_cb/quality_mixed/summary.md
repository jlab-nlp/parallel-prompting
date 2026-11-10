### a6000 / qwen2.5-7b-instruct / quality_mixed

| config | engine | time (s) | passes | F1 | EM | n | P | same answers as sgl-on | notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| pp-radix | sglang 0.5.22 + ParallelPromptEngine | 356.5 | 2 | acc 75.55 | - | 2086 | 4 | - | P fixed at 4 (not calibrated), prefix-cache hit tokens 4681 |
| pp-cb-radix | sglang 0.5.22 + ContinuousPPEngine | 292.1 | 2 | acc 75.36 | - | 2086 | 4 | - | P fixed at 4 (not calibrated), prefix-cache hit tokens 4681 |

pp-cb-radix speedup: 

pp-radix speedup: 

GPU / node: NVIDIA RTX A6000
Nondeterministic across passes: none
