#!/usr/bin/env bash
# SGLang v0.5.22: stock Triton + CUDA graphs vs. Parallel Prompting (Triton,
# CUDA graphs), long documents, held-out P calibration. Both passes run here, on
# one GPU, with reversed config order (pass A seeds 2-4, pass B seeds 5-7), so
# every paired comparison is on the same device. Resumable.
#   bash pp_bench/run_sweep_0522.sh <out_dir> [model]
set -u
OUT=${1:-pp_bench/out/sweep_0522}
MODEL=${2:-codellama/CodeLlama-7b-Instruct-hf}
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,uuid --format=csv,noheader > "$OUT/gpu.txt"
run() {  # <pass> <reps> <configs...>
  local pass=$1 reps=$2; shift 2
  for c in "$@"; do
    echo "=== pass $pass $c $(date)"
    python3 pp_bench/bench_pp_vs_sglang.py --config $c --model-path "$MODEL" --groups 1 \
      --doc-tokens 8192 16000 --questions 64 128 256 --parallel-sizes 4 8 16 32 64 128 256 \
      --output-len 128 --mem-fraction-static 0.88 --eval-reps $reps \
      --out "$OUT/results.jsonl" >> "$OUT/$c.log" 2>&1
    echo "=== pass $pass $c exit $? $(date)"
  done
}
run A "2 3 4" pp-graph triton-graph
run B "5 6 7" triton-graph pp-graph
echo SWEEP_DONE
