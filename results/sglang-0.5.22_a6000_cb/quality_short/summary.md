### a6000 / qwen2.5-7b-instruct / quality_short

| config | engine | time (s) | passes | F1 | EM | n | P | same answers as sgl-on | notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| pp-radix | sglang 0.5.22 + ParallelPromptEngine | 161.0 | 2 | acc 75.89 | - | 2086 | 4 | - | P fixed at 4 (not calibrated), prefix-cache hit tokens 4681 |
| pp-cb-radix | sglang 0.5.22 + ContinuousPPEngine | 161.0 | 2 | acc 75.89 | - | 2086 | 4 | - | P fixed at 4 (not calibrated), prefix-cache hit tokens 4681 |

pp-cb-radix speedup: 

pp-radix speedup: 

GPU / node: NVIDIA RTX A6000
Nondeterministic across passes: none
