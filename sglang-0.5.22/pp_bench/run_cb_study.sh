#!/bin/bash
# Continuous-batching study (Table 4): pp-radix vs pp-cb-radix on QuALITY with
# uniform short answers (quality_short) and mixed short / explained answers
# (quality_mixed), both in the chat template with a 512-token budget, P fixed
# at 4, same node, two cold passes each (bench_data.py docstring).
#
#   bash pp_bench/run_cb_study.sh <out root> <gpu tag> <hf model id>
#
# Env: CONFIGS (default "pp-radix pp-cb-radix").
set -u
ROOT=$1
GPU=$2
M=$3
cd "$(dirname "$0")/.."
MP=$(ls -d "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${M//\//--}/snapshots/"*/ 2>/dev/null | head -1)
MP=${MP:-$M}  # not cached: let the engines download it
TAG=$(basename "$M" | tr '[:upper:]' '[:lower:]')
gpu_ok() { python3 -c "import torch; torch.ones(1, device='cuda')" > /dev/null 2>&1; }

python3 pp_bench/test_cb_mock.py > "$ROOT/test_cb_mock.log" 2>&1 \
  || { echo "MOCK_FAILED"; tail -20 "$ROOT/test_cb_mock.log"; exit 1; }
echo "mock test passed"

for d in quality_short quality_mixed; do
  [ -f "$ROOT/work/$TAG/$d.json.gz" ] ||
    HF_HUB_OFFLINE=0 python3 pp_bench/bench_data.py --model-path "$MP" --model-id "$M" \
      --out "$ROOT/work" --max-context 32768 --datasets $d --cache /tmp/bench_downloads
done
python3 - "$ROOT/work/$TAG/quality_mixed.json.gz" "$MP" <<'EOF'
import gzip, json, sys
from transformers import AutoTokenizer
w = json.load(gzip.open(sys.argv[1], "rt"))
tok = AutoTokenizer.from_pretrained(sys.argv[2])
for g in w["groups"][:2]:
    print("doc tail:", repr(tok.decode(g["doc"][-12:])))
    print("question:", repr(tok.decode(g["qs"][0])))
EOF

for d in quality_short quality_mixed; do
  W="$ROOT/work/$TAG/$d.json.gz"
  OUT="$ROOT/$GPU/$TAG/$d"
  mkdir -p "$OUT"
  for c in ${CONFIGS:-pp-radix pp-cb-radix}; do
    for p in 1 2; do
      [ -f "$OUT/$c.p$p.time.json" ] && continue
      gpu_ok || { echo "GPU_BROKEN before $TAG $d $c $(date)"; echo SWEEP_DONE; exit 1; }
      echo "=== $TAG $d $c pass $p $(date)"
      python3 pp_bench/eval_qa.py --work "$W" --model-path "$MP" --config $c --pass-index $p \
        --out "$OUT" --pp-row-len 65536 --parallel-size 4 >> "$OUT/$c.log" 2>&1
      echo "=== $TAG $d $c pass $p exit $? $(date)"
    done
  done
done
echo SWEEP_DONE
