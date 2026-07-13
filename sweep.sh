#!/usr/bin/env bash
# Config sweep driver for the LM Studio tok/s benchmark.
#
# For each config row it: unloads all models, loads the target with the given
# CLI knobs (`lms load`), waits until the server answers, then runs bench.py for
# decode + prefill and (optionally) bench_parallel.py. Results append to CSV.
#
# CLI-settable knobs live in the CONFIGS rows below. GUI/preset-only knobs
# (flash attention, KV-cache quant, force-MoE-experts-to-CPU, thread count) can
# NOT be set from `lms load` -- set them once in the LM Studio UI, then export
# FLASH / KV_QUANT / THREADS so they're recorded in the CSV for this batch:
#
#   FLASH=on KV_QUANT=q8_0 THREADS=8 ./sweep.sh qwen3.6-35b-a3b-mtp q8_k_xl
#
# Usage: ./sweep.sh <model-key> <quant-label> [thinking:allow|no_think]
set -uo pipefail
cd "$(dirname "$0")"

MODEL="${1:?usage: sweep.sh <model-key> <quant-label> [thinking]}"
QUANT="${2:?provide a quant label for the CSV, e.g. q8_k_xl}"
THINKING="${3:-allow}"

FLASH="${FLASH:-unset}"        # GUI knob, recorded only
KV_QUANT="${KV_QUANT:-unset}"  # GUI knob, recorded only
THREADS="${THREADS:-unset}"    # GUI knob, recorded only

OUT="results/results.csv"
TPUT="results/throughput.csv"
RUNS="${RUNS:-3}"
WARMUP="${WARMUP:-1}"
MAXTOK="${MAXTOK:-256}"
HOST="${LMS_HOST:-http://localhost:1234}"

# Config matrix: "ctx|parallel|gpu|mtp|label". Edit freely per phase.
# mtp = on|off (adds --speculative-draft-mtp). gpu = off|max|0..1.
CONFIGS=(
  "131072|1|max|on|ctx128k_p1_gpumax_mtp"
  "262144|1|max|on|ctx256k_p1_gpumax_mtp"
  "131072|1|max|off|ctx128k_p1_gpumax_nomtp"
)
# Concurrency levels for the throughput pass (set THROUGHPUT=1 to enable).
THROUGHPUT="${THROUGHPUT:-0}"
CONCURRENCY_LEVELS=(2 4)

wait_ready() {
  for _ in $(seq 1 60); do
    if curl -sf "$HOST/api/v0/models" >/dev/null 2>&1; then
      # confirm the target model reports loaded
      if curl -s "$HOST/api/v0/models" | grep -q "\"$MODEL\""; then return 0; fi
    fi
    sleep 2
  done
  echo "[sweep] ERROR: model $MODEL not ready after timeout" >&2
  return 1
}

run_config() {
  local ctx="$1" par="$2" gpu="$3" mtp="$4" label="$5"
  echo "=== [sweep] $label  (ctx=$ctx parallel=$par gpu=$gpu mtp=$mtp flash=$FLASH kv=$KV_QUANT threads=$THREADS) ==="

  lms unload --all >/dev/null 2>&1
  local mtp_flag="--no-speculative-draft-mtp"
  [ "$mtp" = "on" ] && mtp_flag="--speculative-draft-mtp"
  if ! lms load "$MODEL" --gpu "$gpu" -c "$ctx" --parallel "$par" $mtp_flag -y >/dev/null 2>&1; then
    echo "[sweep] ERROR: lms load failed for $label; skipping" >&2
    return 1
  fi
  wait_ready || return 1

  local common=(--model "$MODEL" --thinking "$THINKING" --quant "$QUANT"
    --ctx "$ctx" --parallel "$par" --gpu "$gpu" --mtp "$mtp"
    --flash "$FLASH" --kv-quant "$KV_QUANT" --threads "$THREADS"
    --label "$label" --runs "$RUNS" --warmup "$WARMUP")

  python3 bench.py "${common[@]}" --mode decode  --prompt prompts/decode.txt      --max-tokens "$MAXTOK" --out "$OUT"
  python3 bench.py "${common[@]}" --mode prefill --prompt prompts/prefill_16k.txt --out "$OUT"

  if [ "$THROUGHPUT" = "1" ] && [ "$par" -gt 1 ]; then
    for c in "${CONCURRENCY_LEVELS[@]}"; do
      [ "$c" -le "$par" ] || continue
      python3 bench_parallel.py --model "$MODEL" --thinking "$THINKING" --prompt prompts/decode.txt \
        --concurrency "$c" --max-tokens "$MAXTOK" --quant "$QUANT" --ctx "$ctx" --parallel "$par" \
        --gpu "$gpu" --flash "$FLASH" --kv-quant "$KV_QUANT" --threads "$THREADS" \
        --label "${label}_c${c}" --mtp "$mtp" --out "$TPUT"
    done
  fi
}

echo "[sweep] model=$MODEL quant=$QUANT thinking=$THINKING flash=$FLASH kv=$KV_QUANT threads=$THREADS"
for row in "${CONFIGS[@]}"; do
  IFS='|' read -r ctx par gpu mtp label <<<"$row"
  run_config "$ctx" "$par" "$gpu" "$mtp" "$label"
done
echo "[sweep] done. Results in $OUT"
[ "$THROUGHPUT" = "1" ] && echo "[sweep] throughput in $TPUT"
