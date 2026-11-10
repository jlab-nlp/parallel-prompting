#!/usr/bin/env bash
# Real benchmarks of the paper (Table 3; bench_data.py: QuALITY, LooGLE): stock
# SGLang and PP, with SGLang's radix prefix cache. In the lmsysorg/sglang:v0.5.22
# image after `bash install.sh 0.5.22`.
# First checks pp-radix against stock SGLang (check_equivalence.py), then runs
# every (dataset, config) for two cold-start passes; one
# if the first took over 30 min. Resumable.
#   bash pp_bench/run_bench.sh <root> <gpu tag> <max context> <hf model id...>
# Env: DATASETS, CONFIGS, LIMIT_GROUPS (smoke test)
set -u
ROOT=$1
GPU=$2
CTX=$3
shift 3
DATASETS=${DATASETS:-quality loogle}
CONFIGS=${CONFIGS:-sgl-on sgl-triton-on pp-radix pp-cb-radix}
LIMIT=${LIMIT_GROUPS:-0}
export HF_HUB_OFFLINE=1
cd "$(dirname "$0")/.."  # <root> must be absolute
snap() {
  local d
  d=$(ls -d "$HF_HOME/hub/models--${1//\//--}/snapshots/"*/ 2>/dev/null | head -1)
  if [ -n "$d" ] && ls "$d" | grep -q safetensors; then echo "$d"; return; fi
  HF_HUB_OFFLINE=0 python3 -c "from huggingface_hub import snapshot_download as s; print(s('$1', cache_dir='/tmp/hf_home/hub', allow_patterns=['*.json', '*.safetensors', 'tokenizer*']))" | tail -1
}
gpu_ok() { python3 -c "import torch; torch.ones(1, device='cuda')" > /dev/null 2>&1; }

for M in "$@"; do
  MP=$(snap "$M")
  TAG=$(basename "$M" | tr '[:upper:]' '[:lower:]')
  [ -n "$MP" ] || { echo "no local snapshot of $M"; continue; }
  for d in $DATASETS; do
    [ -f "$ROOT/work/$TAG/$d.json.gz" ] ||
      HF_HUB_OFFLINE=0 HF_DATASETS_CACHE=/tmp/hf_datasets \
        python3 pp_bench/bench_data.py --model-path "$MP" --model-id "$M" \
          --out "$ROOT/work" --max-context $CTX --datasets $d --cache /tmp/bench_downloads
  done

  G="$ROOT/$GPU/$TAG/radix_gate"
  mkdir -p "$G"
  if [ ! -f "$G/verdict" ]; then
    for v in "" --prefix-cache; do
      python3 pp_bench/check_equivalence.py --engine pp $v --model-path "$MP" --groups 4 \
        --max-questions 8 --max-new-tokens 32 --parallel-size 1 4 --cuda-graph \
        > "$G/pp${v:+-radix}.log" 2>&1
    done
    python3 - "$G" <<'EOF'
import os, re, sys
g = sys.argv[1]
def pct(name):
    m = re.findall(r"exact match: \d+/\d+ \(([\d.]+)%\)", open(os.path.join(g, name)).read())
    return min(map(float, m)) if len(m) == 2 else -1.0
pp, radix = pct("pp.log"), pct("pp-radix.log")
ok = pp >= 0 and radix >= 0 and radix >= pp - 5
open(os.path.join(g, "verdict"), "w").write(("pass" if ok else "fail") + f" pp {pp} pp-radix {radix}\n")
print("radix gate:", "pass" if ok else "fail", pp, radix)
EOF
  fi
  RADIX_OK=$(cut -d' ' -f1 "$G/verdict")

  for d in $DATASETS; do
    W="$ROOT/work/$TAG/$d.json.gz"
    [ -f "$W" ] || { echo "no work file $W"; continue; }
    OUT="$ROOT/$GPU/$TAG/$d"
    mkdir -p "$OUT"
    for c in $CONFIGS; do
      if [[ $c == *radix* && $RADIX_OK != pass ]]; then
        echo "skip $c ($M $d): radix gate failed"
        continue
      fi
      for p in 1 2; do
        [ -f "$OUT/$c.p$p.time.json" ] && continue
        if [ $p = 2 ] && python3 -c "import json,sys; sys.exit(json.load(open('$OUT/$c.p1.time.json'))['wall_s'] <= 1800)"; then
          continue
        fi
        gpu_ok || { echo "GPU_BROKEN before $TAG $d $c $(date)"; echo SWEEP_DONE; exit 1; }
        echo "=== $TAG $d $c pass $p $(date)"
        python3 pp_bench/eval_qa.py --work "$W" --model-path "$MP" --config $c --pass-index $p \
          --out "$OUT" --limit-groups $LIMIT --pp-row-len 65536 >> "$OUT/$c.log" 2>&1
        echo "=== $TAG $d $c pass $p exit $? $(date)"
      done
    done
  done
done
echo SWEEP_DONE
