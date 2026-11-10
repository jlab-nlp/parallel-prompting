#!/usr/bin/env bash
# Long-document, many-question sweep with held-out P calibration. Two passes
# with reversed config order (so GPU drift over time cannot favour one config):
# pass A scores eval seeds 2-4, pass B seeds 5-7. Each pass writes its own
# directory and can run on its own GPU; every seed stays on one GPU, so paired
# comparisons hold. Resumable.
#   bash pp_bench/run_sweep_long.sh <out_dir> <A|B> [model]
set -u
OUT=${1:-pp_bench/out/sweep_long}/$2
MODEL=${3:-codellama/CodeLlama-7b-Instruct-hf}
mkdir -p "$OUT"
case $2 in
  A) REPS="2 3 4"; ORDER="pp-graph pp triton-graph triton flashinfer-graph flashinfer" ;;
  B) REPS="5 6 7"; ORDER="flashinfer flashinfer-graph triton triton-graph pp pp-graph" ;;
esac
nvidia-smi --query-gpu=name,uuid --format=csv,noheader > "$OUT/gpu.txt"
for c in $ORDER; do
  echo "=== pass $2 $c $(date)"
  python3 pp_bench/bench_pp_vs_sglang.py --config $c --model-path "$MODEL" --groups 1 \
    --doc-tokens 8192 16000 --questions 64 128 256 --parallel-sizes 4 8 16 32 64 128 256 \
    --output-len 128 --eval-reps $REPS --out "$OUT/results.jsonl" >> "$OUT/$c.log" 2>&1
  echo "=== pass $2 $c exit $? $(date)"
done
echo SWEEP_DONE
