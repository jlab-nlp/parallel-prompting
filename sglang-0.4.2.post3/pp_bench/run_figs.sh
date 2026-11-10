#!/usr/bin/env bash
# Experiments behind the paper's figures, ordered by priority so a run that is
# stopped early still has the most important ones. Each stage writes its
# own results.jsonl (stats never mix workloads). Resumable: rerun to continue.
#   S1 core     Fig 1 (portable across versions/GPUs) + Fig 4 (choosing P):
#               8192/16000-token docs x 64/128/256 questions, two passes with
#               reversed config order (seeds 2-4, then 5-7)
#   S2 scaling  Fig 2 (gain grows with doc length x questions): 512-4096-token
#               docs x 16/64/256 questions (8192/16000 come from S1)
#   S3 groups   Fig 5 (gain survives batching many documents): 4 and 16 docs
#               per batch, 4096 tokens x 64 questions (1 doc comes from S2)
#   S4 outlen   Fig 3 (gain comes from decode): 32/512/1024 output tokens,
#               8192 tokens x 128 questions (128 comes from S1)
#   S5 nograph  6-config ablation on the S1 workloads (configs without graphs)
#   bash pp_bench/run_figs.sh <out_dir> [model]
# Env: MEM=0.88 sets the mem fraction; LONG_P=4,8,...,128 sets the P list for
# the long-document stages (default up to 256).
set -u
OUT=${1:-pp_bench/out/figs}
MODEL=${2:-codellama/CodeLlama-7b-Instruct-hf}
MEMARG=${MEM:+--mem-fraction-static $MEM}
GRAPH="pp-graph triton-graph flashinfer-graph"
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,uuid --format=csv,noheader > "$OUT/gpu.txt"
run() {  # <stage> <reps> <configs> <bench args...>
  local stage=$1 reps=$2 configs=$3; shift 3
  mkdir -p "$OUT/$stage"
  for c in $configs; do
    # A GPU that hit a launch failure can stay broken for every later process;
    # stop rather than record a column of failures.
    python3 -c "import torch; torch.ones(1, device='cuda')" > /dev/null 2>&1 ||
      { echo "GPU_BROKEN before $stage $c $(date)"; echo SWEEP_DONE; exit 1; }
    echo "=== $stage $c $(date)"
    python3 pp_bench/bench_pp_vs_sglang.py --config $c --model-path "$MODEL" $MEMARG \
      --eval-reps $reps --out "$OUT/$stage/results.jsonl" "$@" >> "$OUT/$stage/$c.log" 2>&1
    echo "=== $stage $c exit $? $(date)"
  done
}
LONG_P=${LONG_P:-4,8,16,32,64,128,256}
LONG="--groups 1 --doc-tokens 8192 16000 --questions 64 128 256 --parallel-sizes ${LONG_P//,/ } --output-len 128"
run core "2 3 4" "$GRAPH" $LONG
run core "5 6 7" "flashinfer-graph triton-graph pp-graph" $LONG
run scaling "2 3 4" "$GRAPH" --groups 1 --doc-tokens 512 1024 2048 4096 --questions 16 64 256 \
  --parallel-sizes 2 4 8 16 32 64 128 --output-len 128
for g in 4 16; do
  run groups_$g "2 3 4" "$GRAPH" --groups $g --doc-tokens 4096 --questions 64 \
    --parallel-sizes 4 8 16 32 64 --output-len 128
done
for n in 512 32 1024; do
  run outlen_$n "2 3 4" "$GRAPH" --groups 1 --doc-tokens 8192 --questions 128 \
    --parallel-sizes 8 16 32 64 --output-len $n
done
run core "2 3 4" "pp triton flashinfer" $LONG
echo SWEEP_DONE
