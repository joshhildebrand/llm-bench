#!/usr/bin/env bash
# Config sweep driver for llama.cpp's llama-server (the LM Studio-free path).
#
# Same job as sweep.sh, different server: for each config row it starts
# llama-server with the requested flags, waits for /health, benchmarks decode +
# prefill via bench.py --api llamacpp, then stops it. Results append to the same
# results/results.csv with the same schema, so llama.cpp rows and LM Studio rows
# are directly comparable.
#
# Why a separate driver: LM Studio isn't available everywhere (headless servers,
# no GUI, no license), and on a GPU-less box the whole `lms load` + advanced-config
# path is meaningless. llama-server takes every knob on the command line, so this
# script needs no apply_config.py equivalent.
#
# NOTE ON CONTEXT: llama.cpp's -c is the TOTAL KV context, divided across the -np
# slots. LM Studio's context is PER slot. This script takes per-slot ctx (matching
# the CSV column and LM Studio semantics) and passes -c (ctx * parallel).
#
# Usage:
#   ./sweep_llamacpp.sh <model-name> <quant-tag> <gguf-abs-path> [thinking]
# Example:
#   ./sweep_llamacpp.sh qwen3.6-35b-a3b-mtp q4_k_xl \
#     /opt/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
set -uo pipefail
cd "$(dirname "$0")"

MODEL_NAME="${1:?usage: sweep_llamacpp.sh <model-name> <quant-tag> <gguf-abs-path> [thinking]}"
QUANT="${2:?quant tag, e.g. q4_k_xl (CSV column only; the file is chosen by path)}"
GGUF="${3:?absolute path to the .gguf}"
THINKING="${4:-allow}"

ID="bench"                                   # stable API alias across quants
OUT="results/results.csv"
PORT="${PORT:-8080}"
HOST="http://127.0.0.1:${PORT}"
SERVER="${SERVER:-/opt/llama.cpp/build/bin/llama-server}"
# Speculative decoding. llama.cpp DOES support the MTP head baked into models like
# Qwen3.6 (common_speculative_impl_draft_mtp), but it is OFF by default -- the
# default is --spec-type none, and on a default load the server silently drops the
# GGUF's blk.N.nextn.* tensors with "unused tensor ... ignoring". Set
# SPEC_TYPE=draft-mtp to actually use them; no separate draft model is needed.
# Two MTP packagings exist in the wild and they need different plumbing:
#   in-model  (Qwen3.6): the head ships as blk.N.nextn.* inside the main GGUF ->
#             SPEC_TYPE alone is enough.
#   separate  (Gemma 4): the head is its own small file (mtp-*.gguf, ~0.1 GB) ->
#             also set DRAFT_MODEL to that file, passed as -md.
SPEC_TYPE="${SPEC_TYPE:-}"
DRAFT_MAX="${DRAFT_MAX:-4}"
DRAFT_MODEL="${DRAFT_MODEL:-}"
RUNS="${RUNS:-3}"; WARMUP="${WARMUP:-1}"; MAXTOK="${MAXTOK:-256}"
CONC="${CONC:-1}"                            # concurrency levels (space-separated)
_maxc=1; for _c in $CONC; do [ "$_c" -gt "$_maxc" ] && _maxc="$_c"; done
PARALLEL="${PARALLEL:-$_maxc}"
PREFILL="${PREFILL:-1}"                      # also run the prefill probe
# Prefill is compute-bound and reads EVERY expert, so on a CPU-only box it costs
# orders of magnitude more wall-clock than decode (a 16k prompt can take minutes).
# It therefore gets its own prompt + run budget rather than inheriting RUNS/WARMUP.
PREFILL_PROMPT="${PREFILL_PROMPT:-prompts/prefill_16k.txt}"
PREFILL_RUNS="${PREFILL_RUNS:-$RUNS}"
PREFILL_WARMUP="${PREFILL_WARMUP:-$WARMUP}"
LOGDIR="${LOGDIR:-/tmp/llama-sweep-logs}"; mkdir -p "$LOGDIR"

# Config matrix. One row = one benchmarked config.
# Fields: ctx|threads|threads_batch|kcache|vcache|flash|label
# ctx is PER SLOT (see note above). threads is the decode thread count -- on a
# bandwidth-bound MoE this is the knob that matters most after quant.
#
# Override without editing this file by exporting CONFIG_ROWS, one row per line:
#   CONFIG_ROWS=$'32768|7|14|q8_0|q8_0|on|t7\n32768|14|14|q8_0|q8_0|on|t14' ./sweep_llamacpp.sh ...
CONFIGS=(
  "32768|14|14|q8_0|q8_0|on|ctx32k_t14"
)
if [ -n "${CONFIG_ROWS:-}" ]; then
  mapfile -t CONFIGS <<<"$CONFIG_ROWS"
fi

SRV_PID=""

stop_server() {
  [ -n "$SRV_PID" ] || return 0
  kill "$SRV_PID" 2>/dev/null
  for _ in $(seq 1 40); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 0.5; done
  kill -9 "$SRV_PID" 2>/dev/null
  wait "$SRV_PID" 2>/dev/null
  SRV_PID=""
}
trap 'stop_server' EXIT INT TERM

wait_ready() {
  for _ in $(seq 1 300); do
    # Bail out early if the server died (bad quant, OOM, unsupported arch).
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      echo "[sweep] ERROR: llama-server exited during load" >&2; return 1
    fi
    if curl -s "$HOST/health" 2>/dev/null | grep -q '"status":"ok"'; then return 0; fi
    sleep 2
  done
  echo "[sweep] ERROR: server not ready after timeout" >&2; return 1
}

run_config() {
  local ctx="$1" threads="$2" tbatch="$3" kc="$4" vc="$5" flash="$6" label="$7"
  local total_ctx=$(( ctx * PARALLEL ))
  echo "=== [sweep] $QUANT :: $label (ctx=$ctx x$PARALLEL=$total_ctx t=$threads tb=$tbatch kv=$kc/$vc fa=$flash) ==="

  local log="$LOGDIR/${QUANT}_${label}.log"
  local spec_args=() mtp_col="off"
  if [ -n "$SPEC_TYPE" ]; then
    # NOTE: --draft-max/--draft-min were REMOVED upstream; the current spelling is
    # --spec-draft-n-max. Passing the old name makes llama-server exit at argument
    # parsing, which looks exactly like a failed model load.
    spec_args=(--spec-type "$SPEC_TYPE" --spec-draft-n-max "$DRAFT_MAX")
    mtp_col="$SPEC_TYPE:d$DRAFT_MAX"
    # Models that ship the MTP head as a separate file (Gemma 4) need it as -md.
    if [ -n "$DRAFT_MODEL" ]; then
      spec_args+=(-md "$DRAFT_MODEL")
      mtp_col="${mtp_col}:ext"
    fi
  fi
  # --no-mmap: load weights into anonymous RAM. On ZFS, mmap'd GGUF reads fight
  # the ARC and make the first-touch cost show up inside the measurement window;
  # a straight read into RSS is both faster to steady-state and reproducible.
  "$SERVER" -m "$GGUF" --alias "$ID" --host 127.0.0.1 --port "$PORT" \
    -c "$total_ctx" -np "$PARALLEL" -t "$threads" -tb "$tbatch" \
    -ctk "$kc" -ctv "$vc" -fa "$flash" --no-mmap --no-webui \
    "${spec_args[@]}" \
    > "$log" 2>&1 &
  SRV_PID=$!

  if ! wait_ready; then
    echo "[sweep] ERROR: load failed for $label; skipping (see $log)" >&2
    tail -20 "$log" >&2
    stop_server; return 1
  fi

  local common=(--api llamacpp --host "$HOST"
    --model "$ID" --model-name "$MODEL_NAME" --thinking "$THINKING" --quant "$QUANT"
    --ctx "$ctx" --parallel "$PARALLEL" --gpu none --mtp "$mtp_col"
    --flash "$flash" --kv-quant "$kc" --threads "$threads"
    --label "$label")

  for c in $CONC; do
    python3 bench.py "${common[@]}" --runs "$RUNS" --warmup "$WARMUP" \
      --mode decode --prompt prompts/decode.txt \
      --max-tokens "$MAXTOK" --concurrency "$c" --out "$OUT"
  done
  [ "$PREFILL" = "1" ] && python3 bench.py "${common[@]}" \
    --runs "$PREFILL_RUNS" --warmup "$PREFILL_WARMUP" \
    --mode prefill --prompt "$PREFILL_PROMPT" --out "$OUT"

  stop_server
}

echo "[sweep] model=$MODEL_NAME quant=$QUANT gguf=$GGUF parallel=$PARALLEL conc='$CONC'"
for row in "${CONFIGS[@]}"; do
  IFS='|' read -r ctx threads tbatch kc vc flash label <<<"$row"
  run_config "$ctx" "$threads" "$tbatch" "$kc" "$vc" "$flash" "$label"
done
echo "[sweep] done -> $OUT"
