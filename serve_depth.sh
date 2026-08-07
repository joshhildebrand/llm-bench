#!/usr/bin/env bash
# Serving profile at REALISTIC KV depth: prefill cost vs prompt size, and
# decode vs concurrent streams when each stream is already deep.
#
# depth.sh answers "how fast does one agent decode as its context fills".
# This answers the two questions that follow from it:
#
#   1. PREFILL CURVE. Agent turns are prefill-heavy -- file contents, tool
#      results, re-sent schemas. The repo only ever measured prefill at 16k from
#      an empty cache, so the cost of ingesting a big turn is unknown.
#
#   2. CONCURRENCY AT DEPTH. The existing concurrency ladders used a 77-token
#      prompt, so they measured N nearly-empty slots. That is not what N agents
#      look like. CRITICAL: KV is a UNIFIED pool (kv_unified=true) -- n_ctx is
#      shared across all slots, NOT per slot. N streams at depth D need N*D
#      tokens allocated, so CTX must be >= CONC_MAX * depth or the run thrashes.
#      The script refuses to start if that budget does not hold.
#
# Usage:
#   ./serve_depth.sh <model-key> <quant> <gguf-rel-path> <ctx> <parallel> [thinking]
set -uo pipefail
cd "$(dirname "$0")"

MODEL_KEY="${1:?usage: serve_depth.sh <model-key> <quant> <gguf> <ctx> <parallel> [thinking]}"
QUANT="${2:?quant tag}"
GGUF="${3:?gguf path relative to LM Studio models dir}"
CTX="${4:?allocated context length}"
PARALLEL="${5:?KV slots to load with}"
THINKING="${6:-allow}"

ID="bench"
OUT="${OUT:-results/results.csv}"
HOST="${LMS_HOST:-http://localhost:1234}"
RUNS="${RUNS:-3}"; WARMUP="${WARMUP:-1}"; MAXTOK="${MAXTOK:-256}"
THREADS="${THREADS:-8}"; KC="${KC:-q8_0}"; VC="${VC:-$KC}"
MTP="${MTP:-false}"; CPU_EXPERTS="${CPU_EXPERTS:-}"; OFFLOAD="${OFFLOAD:-}"
MODEL_NAME="${MODEL_NAME:-$MODEL_KEY}"
VRAMLOG="${VRAMLOG:-results/vram-depth.log}"

# prefill curve: measured prompt sizes (bench.py prefill mode defeats the cache)
PREFILL_SET="${PREFILL_SET:-2k:prompts/prefill_2k.txt 11k:prompts/prefill_16k.txt 46k:prompts/prefill_64k.txt 69k:prompts/prefill_96k.txt}"
# concurrency at depth: every stream ingests this prompt, then decodes
CONC_PROMPT="${CONC_PROMPT:-prompts/prefill_16k.txt}"
CONC_DEPTH_TOK="${CONC_DEPTH_TOK:-11044}"      # measured tokens in CONC_PROMPT
CONC="${CONC:-1 2 4 8}"

_maxc=1; for c in $CONC; do [ "$c" -gt "$_maxc" ] && _maxc="$c"; done
need=$(( _maxc * CONC_DEPTH_TOK + _maxc * MAXTOK ))
if [ "$need" -gt "$CTX" ]; then
  echo "[serve] ABORT: unified KV needs ${need} tokens for ${_maxc} streams at ${CONC_DEPTH_TOK} depth," >&2
  echo "[serve]        but ctx is only ${CTX}. Raise ctx or lower CONC/CONC_PROMPT." >&2
  exit 2
fi
if [ "$PARALLEL" -lt "$_maxc" ]; then
  echo "[serve] ABORT: --parallel $PARALLEL < max concurrency $_maxc; extra streams would queue." >&2
  exit 2
fi

wait_ready() {
  for _ in $(seq 1 120); do
    curl -s "$HOST/api/v0/models" 2>/dev/null | grep -q "\"$ID\"" && return 0
    sleep 2
  done
  echo "[serve] ERROR: $ID not ready" >&2; return 1
}
vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd, -; }

echo "[serve] $MODEL_NAME $QUANT ctx=$CTX parallel=$PARALLEL kv=$KC/$VC conc='$CONC' (needs ${need} tok)"

sets=(--set ctx="$CTX" --set threads="$THREADS" --set kcache="$KC" --set vcache="$VC"
      --set flash=true --set mtp="$MTP" --set kv_to_gpu=true)
[ -n "$CPU_EXPERTS" ] && sets+=(--set cpu_experts="$CPU_EXPERTS")
[ -n "$OFFLOAD" ] && sets+=(--set offload="$OFFLOAD")
python3 apply_config.py --gguf "$GGUF" "${sets[@]}" || exit 1

lms unload --all >/dev/null 2>&1
if ! timeout 900 lms load "$MODEL_KEY" --identifier "$ID" --parallel "$PARALLEL" -y >/dev/null 2>&1; then
  echo "[serve] LOAD FAILED for $MODEL_KEY @ ctx=$CTX parallel=$PARALLEL" >&2
  echo "$(date +%H:%M:%S) $MODEL_NAME ctx=$CTX p=$PARALLEL LOAD_FAILED" >>"$VRAMLOG"
  exit 1
fi
wait_ready || exit 1
v="$(vram)"; echo "$(date +%H:%M:%S) $MODEL_NAME $QUANT ctx=$CTX p=$PARALLEL kv=$KC vram_mib=$v" >>"$VRAMLOG"
echo "[serve] VRAM after load (MiB per gpu): $v"

mtp_flag="$([ "$MTP" = "true" ] && echo on || echo off)"
common=(--model "$ID" --model-name "$MODEL_NAME" --thinking "$THINKING" --quant "$QUANT"
        --ctx "$CTX" --parallel "$PARALLEL" --gpu max --mtp "$mtp_flag" --flash on
        --kv-quant "$KC" --threads "$THREADS" --runs "$RUNS" --warmup "$WARMUP")

# 1. prefill curve
for rung in $PREFILL_SET; do
  name="${rung%%:*}"; file="${rung#*:}"
  [ -f "$file" ] || continue
  echo "=== [serve] $MODEL_NAME :: prefill @ $name ==="
  python3 bench.py "${common[@]}" --mode prefill --prompt "$file" \
    --label "pfcurve_${name}_ctx$((CTX/1024))k" --out "$OUT"
done

# 2. concurrency at depth
for c in $CONC; do
  echo "=== [serve] $MODEL_NAME :: ${c} stream(s) @ depth ${CONC_DEPTH_TOK} ==="
  python3 bench.py "${common[@]}" --mode decode --prompt "$CONC_PROMPT" \
    --max-tokens "$MAXTOK" --concurrency "$c" \
    --label "concdepth_${CONC_DEPTH_TOK}tok_p${PARALLEL}_ctx$((CTX/1024))k" --out "$OUT"
done

echo "[serve] done -> $OUT"
