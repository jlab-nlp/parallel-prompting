#!/usr/bin/env bash
# Equivalence checks for continuous batching (pp-cb, pp-cb-radix), each next to
# the gate of the config it extends, on one model:
#   bash pp_bench/run_cb_gate.sh <out_dir> <model path>
# Every check_equivalence run compares against stock SGLang (Triton); the
# new engine should match about as often as the engine it extends.
set -uo pipefail
OUT=$1
MP=$2
mkdir -p "$OUT"
cd "$(dirname "$0")/.."
nvidia-smi --query-gpu=name,uuid --format=csv,noheader > "$OUT/gpu.txt"
python3 pp_bench/test_cb_mock.py > "$OUT/mock.log" 2>&1
COMMON=(--model-path "$MP" --groups 8 --max-questions 8 --max-new-tokens 64)
run() {  # <name> <args...>
  local name=$1; shift
  [ -f "$OUT/$name.log" ] && grep -q "exact match" "$OUT/$name.log" && return
  echo "== $name $*" | tee -a "$OUT/driver.log"
  python3 pp_bench/check_equivalence.py "${COMMON[@]}" "$@" --out "$OUT/$name.json" \
    > "$OUT/$name.log" 2>&1 || echo "$name failed ($?)" | tee -a "$OUT/driver.log"
}
run pp            --engine pp      --parallel-size 1 4 0 --cuda-graph
run cb            --engine cb      --parallel-size 1 4 0 --cuda-graph
run cb-eager      --engine cb      --parallel-size 1 4 0
# A wave of 2 requests forces refills on every group.
export SGLANG_PP_MAX_WAVE=2
run cb-wave2      --engine cb      --parallel-size 1 4 --cuda-graph
unset SGLANG_PP_MAX_WAVE
run pp-radix      --engine pp      --parallel-size 1 4 --cuda-graph --prefix-cache
run cb-radix      --engine cb      --parallel-size 1 4 --cuda-graph --prefix-cache
grep -H "exact match\|== \|Error" "$OUT"/*.log > "$OUT/summary.txt"
echo done >> "$OUT/driver.log"
