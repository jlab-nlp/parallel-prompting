#!/usr/bin/env bash
# Model-size sweep on SGLang v0.5.22: PP (Triton + CUDA graphs) vs. stock
# FlashInfer + CUDA graphs and Triton + CUDA graphs on a reduced core grid, with
# the paper's methodology (rep 0 warmup, rep 1 P calibration, eval reps paired
# across configs). Pass A runs seeds 2-4, pass B seeds 5-7 in reversed config
# order. Resumable. First runs the PP correctness checks on this model.
#   bash pp_bench/run_models_0522.sh <out_dir> <model> [doc tokens...]
set -u
OUT=$1
MODEL=$2
shift 2
DOCS=${*:-8192 16000}
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader > "$OUT/gpu.txt"
echo "=== checks $(date)"
python3 pp_bench/check_kernel.py > "$OUT/check_kernel.log" 2>&1
echo "=== check_kernel exit $?"
python3 pp_bench/check_equivalence.py --model-path "$MODEL" --groups 4 --max-questions 8 \
  --max-new-tokens 32 --parallel-size 0 4 --cuda-graph > "$OUT/check_equivalence.log" 2>&1
echo "=== check_equivalence exit $?"
run() {  # <pass> <reps> <configs...>
  local pass=$1 reps=$2; shift 2
  for c in "$@"; do
    echo "=== pass $pass $c $(date)"
    python3 pp_bench/bench_pp_vs_sglang.py --config $c --model-path "$MODEL" --groups 1 \
      --doc-tokens $DOCS --questions 64 256 --parallel-sizes 4 8 16 32 64 128 256 \
      --output-len 128 --mem-fraction-static 0.88 --eval-reps $reps \
      --out "$OUT/results.jsonl" >> "$OUT/$c.log" 2>&1
    echo "=== pass $pass $c exit $? $(date)"
  done
}
run A "2 3 4" pp-graph flashinfer-graph triton-graph
echo PASS_A_DONE
run B "5 6 7" triton-graph flashinfer-graph pp-graph
echo SWEEP_DONE
