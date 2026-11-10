#!/usr/bin/env bash
# Follow-up to run_figs.sh once the PP engine runs requests in waves that fit
# the KV pool (before that, PP skipped workloads whose streams did not all fit).
#   1. check that waved outputs still match the reference (SGLANG_PP_MAX_WAVE=2
#      forces several waves);
#   2. rerun PP where it was skipped, warming up at full output length so the
#      wave shapes' CUDA graphs are captured before timing;
#   3. resume anything run_figs.sh left unfinished.
#   bash pp_bench/run_figs_fix.sh <out_dir> [model]   (env: MEM, LONG_P)
set -u
OUT=${1:-pp_bench/out/figs}
MODEL=${2:-codellama/CodeLlama-7b-Instruct-hf}
MEMARG=${MEM:+--mem-fraction-static $MEM}
mkdir -p "$OUT"
echo "=== waves check $(date)"
SGLANG_PP_MAX_WAVE=2 python3 pp_bench/check_equivalence.py --model-path "$MODEL" $MEMARG \
  --groups 8 --max-questions 8 --max-new-tokens 64 --parallel-size 0 2 --cuda-graph \
  > "$OUT/waves_check.log" 2>&1
echo "=== waves check exit $? $(date)"
grep -E "^== |exact match" "$OUT/waves_check.log"
pp() {  # <stage> <reps> <bench args...>
  local stage=$1 reps=$2; shift 2
  python3 -c "import torch; torch.ones(1, device='cuda')" > /dev/null 2>&1 ||
    { echo "GPU_BROKEN before $stage $(date)"; echo SWEEP_DONE; exit 1; }
  echo "=== $stage pp-graph (waves) $(date)"
  python3 pp_bench/bench_pp_vs_sglang.py --config pp-graph --model-path "$MODEL" $MEMARG \
    --warmup-full --eval-reps $reps --out "$OUT/$stage/results.jsonl" "$@" >> "$OUT/$stage/pp-graph.log" 2>&1
  echo "=== $stage pp-graph exit $? $(date)"
}
LONG_P=${LONG_P:-4,8,16,32,64,128,256}
pp core "2 3 4 5 6 7" --groups 1 --doc-tokens 8192 16000 --questions 64 128 256 \
  --parallel-sizes ${LONG_P//,/ } --output-len 128
for g in 4 16; do
  pp groups_$g "2 3 4" --groups $g --doc-tokens 4096 --questions 64 --parallel-sizes 4 8 16 32 64 --output-len 128
done
for n in 512 32 1024; do
  pp outlen_$n "2 3 4" --groups 1 --doc-tokens 8192 --questions 128 --parallel-sizes 8 16 32 64 --output-len $n
done
bash pp_bench/run_figs.sh "$OUT" "$MODEL"
