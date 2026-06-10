#!/usr/bin/env bash
set -euo pipefail

URL="https://data.hplt-project.org/three/sorted/eng_Latn/9_99.jsonl.zst"
OUT="en_sample.jsonl"

curl -sSL "$URL" | zstd -dc | head -n 1000 > "$OUT"

echo "Wrote $(wc -l < "$OUT") lines to $OUT"