#!/usr/bin/env bash
# Fetch the exact llama.cpp conversion tooling that matches the llama.cpp
# revision vendored by llama-cpp-sys-2 0.1.160.
#
#   llama-cpp-rs  commit : 0dd5814c9b061930965d144c1653cd4d05054873
#   llama.cpp     commit : d81235049384534c167caea52b85a694f6103d14
set -euo pipefail

LC_SHA=d81235049384534c167caea52b85a694f6103d14
DEST="${1:-$HOME/models/convert}"
API="https://api.github.com/repos/ggml-org/llama.cpp/contents"
RAW="https://raw.githubusercontent.com/ggml-org/llama.cpp/$LC_SHA"

mkdir -p "$DEST"
cd "$DEST"

fetch_dir() {
  local dir="$1"          # e.g. conversion
  local names
  names=$(curl -sf --max-time 60 "$API/$dir?ref=$LC_SHA" \
    | python3 -c 'import json,sys
d=json.load(sys.stdin)
if not isinstance(d,list): raise SystemExit(d)
for f in d:
    if f["type"]=="file" and f["name"].endswith(".py"): print(f["name"])')
  mkdir -p "$dir"
  local n
  for n in $names; do
    curl -sfL --max-time 60 "$RAW/$dir/$n" -o "$dir/$n"
    printf '  %s/%s\n' "$dir" "$n"
  done
}

fetch_dir conversion
fetch_dir gguf-py/gguf
curl -sfL --max-time 60 "$RAW/convert_hf_to_gguf.py" -o convert_hf_to_gguf.py

echo "== done =="
ls -l convert_hf_to_gguf.py
ls conversion | wc -l
ls gguf-py/gguf | wc -l
