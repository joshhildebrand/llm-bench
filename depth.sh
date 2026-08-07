#!/usr/bin/env bash
# Context-DEPTH ladder: decode tok/s as a function of how full the KV cache is.
#
# sweep.sh and ladder.sh both measure decode from an EMPTY cache (prompts/decode.txt
# is 77 tokens). Every row in results.csv is therefore a best case. The laguna notes
# already flag the problem: laguna-xs is 103.5 tok/s empty but 16.1 at 86k depth, so
# an empty-cache row overstates agent behaviour by ~6x.
#
# This walks one loaded model across increasing prompt depths at a FIXED allocated
# context, so the only thing changing is how many tokens are already in the cache.
# That is the number that matters for an agent harness (Hermes Agent enforces
# MINIMUM_CONTEXT_LENGTH = 64_000 and will sit at 40-80k depth routinely).
#
# One load per model, N decode measurements -- unlike sweep.sh which reloads per row.
#
# Usage:
#   ./depth.sh <model-key> <quant-tag> <gguf-rel-path> <ctx> [thinking:allow|no_think]
# Example:
#   ./depth.sh qwen3-coder-30b-a3b-instruct iq4_xs \
#     unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-IQ4_XS.gguf \
#     131072 no_think
set -uo pipefail
cd "$(dirname "$0")"

MODEL_KEY="${1:?usage: depth.sh <model-key> <quant> <gguf> <ctx> [thinking]}"
QUANT="${2:?quant tag}"
GGUF="${3:?gguf path relative to LM Studio models dir}"
CTX="${4:?allocated context length}"
THINKING="${5:-allow}"

ID="bench"
OUT="${OUT:-results/results.csv}"
HOST="${LMS_HOST:-http://localhost:1234}"
RUNS="${RUNS:-3}"; WARMUP="${WARMUP:-1}"; MAXTOK="${MAXTOK:-256}"
THREADS="${THREADS:-8}"; KC="${KC:-q8_0}"; VC="${VC:-$KC}"
MTP="${MTP:-false}"                 # default OFF: MTP regressed 2026-07-14, see models.json
CPU_EXPERTS="${CPU_EXPERTS:-}"      # empty = let LM Studio place (resident models)
OFFLOAD="${OFFLOAD:-}"              # set 1.0 for gpt-oss: auto ratio silently spills to CPU
MODEL_NAME="${MODEL_NAME:-$MODEL_KEY}"
VRAMLOG="${VRAMLOG:-results/vram-depth.log}"

# depth rung -> prompt file. decode.txt is the empty-cache control that every
# existing row in results.csv used, so the first rung is directly comparable.
DEPTHS="${DEPTHS:-empty:prompts/decode.txt 2k:prompts/prefill_2k.txt 11k:prompts/prefill_16k.txt 46k:prompts/prefill_64k.txt 69k:prompts/prefill_96k.txt}"

wait_ready() {
  for _ in $(seq 1 120); do
    curl -s "$HOST/api/v0/models" 2>/dev/null | grep -q "\"$ID\"" && return 0
    sleep 2
  done
  echo "[depth] ERROR: $ID not ready" >&2; return 1
}

vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd, -; }

mkdir -p "$(dirname "$VRAMLOG")"
echo "[depth] $MODEL_NAME $QUANT ctx=$CTX kv=$KC/$VC thinking=$THINKING mtp=$MTP"

sets=(--set ctx="$CTX" --set threads="$THREADS" --set kcache="$KC" --set vcache="$VC"
      --set flash=true --set mtp="$MTP" --set kv_to_gpu=true)
[ -n "$CPU_EXPERTS" ] && sets+=(--set cpu_experts="$CPU_EXPERTS")
[ -n "$OFFLOAD" ] && sets+=(--set offload="$OFFLOAD")
python3 apply_config.py --gguf "$GGUF" "${sets[@]}" || exit 1

lms unload --all >/dev/null 2>&1
if ! timeout 900 lms load "$MODEL_KEY" --identifier "$ID" --parallel 1 -y >/dev/null 2>&1; then
  echo "[depth] LOAD FAILED for $MODEL_KEY @ ctx=$CTX -- skipping" >&2
  echo "$(date +%H:%M:%S) $MODEL_NAME ctx=$CTX LOAD_FAILED" >>"$VRAMLOG"
  exit 1
fi
wait_ready || exit 1

v="$(vram)"
echo "$(date +%H:%M:%S) $MODEL_NAME $QUANT ctx=$CTX kv=$KC vram_mib=$v" >>"$VRAMLOG"
echo "[depth] VRAM after load (MiB per gpu): $v"

for rung in $DEPTHS; do
  name="${rung%%:*}"; file="${rung#*:}"
  [ -f "$file" ] || { echo "[depth] missing $file, skipping $name" >&2; continue; }
  echo "=== [depth] $MODEL_NAME :: depth=$name ==="
  python3 bench.py --model "$ID" --model-name "$MODEL_NAME" --thinking "$THINKING" \
    --quant "$QUANT" --ctx "$CTX" --parallel 1 --gpu max --flash on \
    --mtp "$([ "$MTP" = "true" ] && echo on || echo off)" \
    --kv-quant "$KC" --threads "$THREADS" --label "depth_${name}_ctx$((CTX/1024))k" \
    --runs "$RUNS" --warmup "$WARMUP" --mode decode --prompt "$file" \
    --max-tokens "$MAXTOK" --concurrency 1 --out "$OUT"
done

echo "[depth] done -> $OUT (vram: $VRAMLOG)"
