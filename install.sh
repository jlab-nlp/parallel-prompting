#!/usr/bin/env bash
# Install Parallel Prompting onto an existing, unmodified SGLang installation by
# copying this repo's overlay files over it.
#
#   bash install.sh 0.5.22               # into the SGLang that python3 imports
#   bash install.sh 0.4.2.post3 <dir>    # into <dir>, the `sglang` package dir
#
# Use the matching official image (lmsysorg/sglang:v0.5.22 or
# lmsysorg/sglang:v0.4.2.post3-cu124); it has the exact kernels and deps.
# The three edited SGLang files are only overwritten if they are still the
# stock files of that version (or already this repo's versions).
set -euo pipefail
VER=${1:?usage: install.sh <0.5.22|0.4.2.post3> [sglang package dir]}
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/sglang-$VER/overlay/sglang"
[ -d "$SRC" ] || { echo "unknown version $VER" >&2; exit 1; }
# cd / so a checkout in the current directory cannot shadow the installed package.
DEST=${2:-$(cd / && python3 -c "import os, sglang; print(os.path.dirname(sglang.__file__))")}

declare -A STOCK_MD5
case $VER in
  0.5.22) STOCK_MD5=(
    [kernels/ops/attention/extend_attention.py]=44f9e5b8dfd38c178025ded3cb898179
    [srt/layers/attention/triton_backend.py]=028edfe30d58efbd65911a4c4ea6d72f
    [srt/layers/logits_processor.py]=9433e6158a89f870554e981708fced12) ;;
  0.4.2.post3) STOCK_MD5=(
    [srt/layers/attention/triton_ops/extend_attention.py]=ad7c42e15463bda7d95aed2ef38edf2d
    [srt/layers/attention/triton_backend.py]=9dfb1a2a2bd55723b338fee584c89512
    [srt/layers/logits_processor.py]=37bdd186ed368cfe884ea0ed483cc867) ;;
esac

for f in "${!STOCK_MD5[@]}"; do
  have=$(md5sum "$DEST/$f" | cut -d' ' -f1)
  if [ "$have" != "${STOCK_MD5[$f]}" ] && ! cmp -s "$SRC/$f" "$DEST/$f"; then
    echo "$DEST/$f is not stock SGLang $VER; refusing to overwrite" >&2
    exit 1
  fi
done
for f in "${!STOCK_MD5[@]}"; do cp "$SRC/$f" "$DEST/$f"; done
mkdir -p "$DEST/srt/parallel_prompt"
cp "$SRC"/srt/parallel_prompt/*.py "$DEST/srt/parallel_prompt/"
echo "installed Parallel Prompting into $DEST"
if [ -z "${2:-}" ]; then
  cd / && python3 -c "import sglang, sglang.srt.parallel_prompt.engine; print('sglang', sglang.__version__, 'OK')"
fi
