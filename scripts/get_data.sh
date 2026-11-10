#!/usr/bin/env bash
# Download the haystack text used by the synthetic long-document workloads
# (War and Peace, Project Gutenberg eBook #2600, public domain) to data/.
# The SQuAD workloads load their data through Hugging Face `datasets` instead.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
curl -fsSL https://www.gutenberg.org/cache/epub/2600/pg2600.txt -o data/war_and_peace.txt
echo "wrote data/war_and_peace.txt ($(wc -c < data/war_and_peace.txt) bytes)"
