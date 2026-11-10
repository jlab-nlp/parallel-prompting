# SQuAD v1.1 dev: Parallel Prompting vs stock SGLang 0.5.22

- GPU: NVIDIA RTX A6000
- Model: codellama/CodeLlama-7b-Instruct-hf (bfloat16)
- Articles: 48, document chunks: 71, questions: 10570
- Document tokens per chunk: max 7999, mean 5538; streams per chunk: max 766, mean 149
- Chunk limit 8000 document tokens; PP parallel size P = 32
- Greedy, max 32 new tokens, stop at EOS or newline
- Prompt: `[INST] Read the document below.\n\nDocument:\n{doc}\n\n` + `Answer the question with a short phrase copied from the document.\nQuestion: {q} [/INST]`

| config | EM | F1 | answer time (s) | identical to PP |
|---|---:|---:|---:|---:|
| PP, Triton + CUDA graphs | 21.51 | 45.14 | 237.59 | - |
| stock SGLang, FlashInfer + CUDA graphs | 21.50 | 45.25 | 406.48 | 9895/10570 (93.6%) |
| stock SGLang, Triton + CUDA graphs | 21.55 | 45.18 | 409.17 | 9861/10570 (93.3%) |

Paired bootstrap over questions (10000 resamples), PP minus stock, in points:

| stock config | F1 diff [95% CI] | EM diff [95% CI] |
|---|---:|---:|
| stock SGLang, FlashInfer + CUDA graphs | -0.11 [-0.25, +0.03] | +0.01 [-0.15, +0.16] |
| stock SGLang, Triton + CUDA graphs | -0.04 [-0.20, +0.12] | -0.04 [-0.22, +0.13] |

Determinism: timed pass identical to warmup pass for pp 10570/10570, flashinfer 10570/10570, triton 10570/10570.
