#!/usr/bin/env bash
# Create kb-* aliases of Ollama models with an 8k context window.
# They share the original weights (no extra disk). Ollama's 4k default
# would silently truncate long Georgian prompts and thinking traces.
# Usage: scripts/make_ollama_aliases.sh [model ...]
set -euo pipefail
MODELS=("$@")
(( ${#MODELS[@]} )) || MODELS=(qwen3:8b llama3.1:8b aya-expanse:8b gpt-oss:20b)
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
for m in "${MODELS[@]}"; do
  printf 'FROM %s\nPARAMETER num_ctx 8192\n' "$m" > "$tmp"
  name="kb-${m/:/-}"
  ollama create "$name" -f "$tmp" >/dev/null && echo "created $name (from $m, num_ctx 8192)"
done
