#!/usr/bin/env bash
# Full benchmark, one model in memory at a time, smallest first.
#
# Safeguards for a 24 GB Mac:
#   * every model is unloaded before the next one loads (Ollama and LM Studio)
#   * 8k context (16k for the reasoning group): Ollama kb-* aliases from
#     scripts/make_ollama_aliases.sh, `lms load --context-length` for LM Studio
#   * fit check: an Ollama model that isn't 100% on the GPU is skipped
#   * speed guard (in kabench): a model that keeps decoding < 3 tok/s, or whose
#     server keeps timing out, exits with code 3; its remaining tracks are skipped
#   * time budgets: 75 min per model (150 for the reasoning subset) and 10 h for
#     the whole run; an over-budget track is killed, finished tracks are kept
#   * all models are unloaded when the script exits, however it exits
#
# Usage: scripts/run_queue.sh [model ...]   (default: every model below)
#        TRACK=agent scripts/run_queue.sh aya-expanse-8b   (rerun one track only)
# Log:   .cache/logs/queue-summary.txt lists what finished, was skipped or aborted
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p .cache/logs
SUMMARY=.cache/logs/queue-summary.txt
: > "$SUMMARY"

# name | backend | backend model id | track set (full / reasoning)
QUEUE=(
  # 1. Tiny
  "nemotron3-nano-4b|lmstudio|nvidia/nemotron-3-nano-4b|full"
  "ministral3-3b|lmstudio|mistralai/ministral-3-3b|full"
  "gemma4-e4b|lmstudio|google/gemma-4-e4b|full"
  # 2. 8B, Q4 and Q8 back to back
  "qwen3-8b|ollama|kb-qwen3-8b|full"
  "qwen3-8b-q8|ollama|kb-qwen3-8b-q8_0|full"
  "llama3.1-8b|ollama|kb-llama3.1-8b|full"
  "llama3.1-8b-q8|ollama|kb-llama3.1-8b-instruct-q8_0|full"
  "aya-expanse-8b|ollama|kb-aya-expanse-8b|full"
  "aya-expanse-8b-q8|ollama|kb-aya-expanse-8b-q8_0|full"
  "granite3.2-8b|lmstudio|ibm/granite-3.2-8b|full"
  # 3. 12B, 2-bit 27B, MoE
  "gemma4-12b-qat|lmstudio|google/gemma-4-12b-qat|full"
  "bonsai-27b-2bit|lmstudio|prism-ml/bonsai-27b|full"
  "lfm2-24b-a2b|lmstudio|liquid/lfm2-24b-a2b|full"
  "gpt-oss-20b|ollama|kb-gpt-oss-20b|full"
  # 4. Reasoning group: fixed subset, 16k context
  "qwen3-8b-think|ollama|kb-qwen3-8b-16k|reasoning"
  "deepseek-r1-qwen3-8b|lmstudio|deepseek/deepseek-r1-0528-qwen3-8b|reasoning"
  # Opt-in, skipped unless named (e.g. scripts/run_queue.sh phi4-reasoning-plus):
  #   phi4-mini-reasoning is math-specialised; on Georgian it thinks in loops until the budget runs out
  #   phi4-reasoning-plus (14B) takes ~3-4 h on its own
  "phi4-mini-reasoning|lmstudio|microsoft/phi-4-mini-reasoning|reasoning|optin"
  "phi4-reasoning-plus|lmstudio|microsoft/phi-4-reasoning-plus|reasoning|optin"
  # Opt-in big models (16-18 GB): stop Docker, raise iogpu.wired_limit_mb to 22528 and set
  # LM Studio's loading guardrails to Off first; LM Studio over-estimates them at ~21 GiB
  "qwen3.6-27b|lmstudio|qwen/qwen3.6-27b|full|optin"
  "qwen3.8-27b|lmstudio|qwen/qwen3.8-27b|full|optin"
  "devstral-small-2-24b|lmstudio|mistralai/devstral-small-2-2512|full|optin"
  "glm4.7-flash|lmstudio|zai-org/glm-4.7-flash|full|optin,ctx=4096"  # 8k leaves ~4% RAM free
  "gemma4-26b-qat|lmstudio|google/gemma-4-26b-a4b-qat|full|optin"
  "qwen3.6-27b-think|lmstudio|qwen/qwen3.6-27b|reasoning|optin"
)

RUN_DEADLINE=$(( $(date +%s) + 10 * 3600 ))
log() { echo "=== $(date +%H:%M) $*"; }
note() { log "$*"; echo "$(date +%H:%M) $*" >> "$SUMMARY"; }

unload_all() {
  ollama ps | awk 'NR>1 {print $1}' | xargs -r -n1 ollama stop 2>/dev/null
  lms unload --all >/dev/null 2>&1
}

load() {  # load <backend> <id> <context>; non-zero if it can't be loaded fully on the GPU
  local backend=$1 id=$2 ctx=$3
  if [[ $backend == ollama ]]; then
    curl -sf http://localhost:11434/api/generate -d "{\"model\":\"$id\",\"keep_alive\":\"30m\"}" >/dev/null || return 1
    ollama ps | grep -q "100% GPU" || { ollama ps; return 2; }
  else
    lms load "$id" --context-length "$ctx" --parallel 1 --gpu max -y >/dev/null 2>&1 || return 1
  fi
}

kill_tree() {  # kill_tree <pid> <signal>: signal a process and all its descendants
  local child
  for child in $(pgrep -P "$1"); do kill_tree "$child" "$2"; done
  kill -"$2" "$1" 2>/dev/null
}

run_track() {  # run_track <deadline-epoch> <kabench args...>; 3 = too slow, 124 = over budget
  local deadline=$1; shift
  uv run kabench "$@" &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    if (( $(date +%s) > deadline )); then
      kill_tree "$pid" TERM; sleep 5; kill_tree "$pid" KILL
      wait "$pid" 2>/dev/null
      return 124
    fi
    sleep 5
  done
  wait "$pid"
}

run_model() {
  local name=$1 backend=$2 id=$3 set=$4 flags=${5:-} ctx=8192 budget=75
  [[ $set == reasoning ]] && ctx=16384 && budget=150
  [[ $flags =~ ctx=([0-9]+) ]] && ctx=${BASH_REMATCH[1]}
  local deadline=$(( $(date +%s) + budget * 60 ))
  (( deadline > RUN_DEADLINE )) && deadline=$RUN_DEADLINE
  unload_all
  log "$name ($backend: $id, $set, ctx $ctx) loading"
  load "$backend" "$id" "$ctx"
  case $? in
    0) ;;
    2) note "SKIPPED $name: does not fit fully on the GPU"; return ;;
    *) note "SKIPPED $name: failed to load"; return ;;
  esac
  log "$name loaded, free memory: $(memory_pressure | tail -1 | grep -o '[0-9]*%')"

  local -a tracks
  if [[ -n ${TRACK:-} ]]; then
    tracks=("$TRACK -m $name")
  elif [[ $set == reasoning ]]; then
    tracks=("grammar -m $name --variants zero_shot --sample 30" "comprehension -m $name --n 30" "agent -m $name")
  else
    tracks=("grammar -m $name" "comprehension -m $name" "agent -m $name")
  fi
  for t in "${tracks[@]}"; do
    # shellcheck disable=SC2086
    run_track "$deadline" $t
    case $? in
      3) note "ABORTED $name during '${t%% *}': too slow"; return ;;
      124) note "ABORTED $name during '${t%% *}': over its ${budget}-min budget (finished tracks kept)"; return ;;
    esac
  done
  # Tokenization is identical across quantizations / thinking modes: measure once per weights
  [[ -n ${TRACK:-} || $name == *-q8 || $name == *-think ]] || run_track "$deadline" tokens -m "$name" --live-only
  note "DONE $name"
}

trap unload_all EXIT
lms server start >/dev/null 2>&1
SELECTED=("$@")
for entry in "${QUEUE[@]}"; do
  IFS='|' read -r name backend id set flags <<< "$entry"
  if (( $(date +%s) >= RUN_DEADLINE )); then note "STOPPED: 10 h run budget used up"; break; fi
  if (( ${#SELECTED[@]} )); then
    [[ " ${SELECTED[*]} " == *" $name "* ]] || continue
  elif [[ ${flags:-} == optin* ]]; then
    continue
  fi
  run_model "$name" "$backend" "$id" "$set" "${flags:-}"
done
unload_all

log "report"
uv run kabench tokens   # refresh offline tokenizer table
uv run kabench report
log "finished"
cat "$SUMMARY"
