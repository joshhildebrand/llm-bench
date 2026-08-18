#!/usr/bin/env bash
# DENSE-model sweep driver. Identical to sweep.sh except for the CONFIGS matrix.
#
# WHY THIS EXISTS: sweep.sh's matrix sweeps cpu_experts (0.85/0.70/0.55), which only
# means anything for a Mixture-of-Experts model. A DENSE model streams ALL weights every
# token -- there are no experts to push to CPU, so those rows would be four identical
# measurements wearing different labels.
#
# The dense levers are different:
#   1. the largest weight quant that still fits, and
#   2. KV-cache quant, which is what buys back the VRAM to reach FULL context.
# So this matrix holds cpu_experts=auto and sweeps kcache/vcache x context instead.
#
# Usage:
#   ./sweep_dense.sh <model-key> <quant-tag> <gguf-rel-path> [thinking:allow|no_think]
# Example:
#   ./sweep_dense.sh muse-glimmer-30b q4_k_xl #     unsloth/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-UD-Q4_K_XL.gguf
set -uo pipefail
cd "$(dirname "$0")"

MODEL_KEY="${1:?usage: sweep.sh <model-key> <quant-tag> <gguf-rel-path> [thinking]}"
QUANT="${2:?quant tag, e.g. q4_k_xl (used for @<quant> load selector + CSV)}"
GGUF="${3:?gguf path relative to LM Studio models dir}"
THINKING="${4:-allow}"

ID="bench"                       # stable API identifier across quants
OUT="results/results.csv"
HOST="${LMS_HOST:-http://localhost:1234}"
RUNS="${RUNS:-3}"; WARMUP="${WARMUP:-1}"; MAXTOK="${MAXTOK:-256}"
CONC="${CONC:-1 2 4 8}"          # concurrency levels tested per config (space-separated)
# Load with enough KV slots for the largest concurrency, unless PARALLEL is set.
# NOTE: KV scales with parallel*context; at high concurrency use a smaller ctx so
# the KV still fits VRAM (single-stream can use full context, the ladder a serving one).
_maxc=1; for _c in $CONC; do [ "$_c" -gt "$_maxc" ] && _maxc="$_c"; done
PARALLEL="${PARALLEL:-$_maxc}"
PREFILL="${PREFILL:-1}"          # also run 16k prefill per config (0 to skip)

# Advanced-param matrix. One row = one benchmarked config.
# Fields: ctx|cpu_experts|threads|kv_to_gpu|draft_max|kcache|vcache|label
# cpu_experts: 1=all experts on CPU (frees VRAM); lower pushes experts onto GPU.
# Use "auto" for cpu_experts to let LM Studio pick (omit the override).
CONFIGS=(
  # ctx|cpu_experts|threads|kv_to_gpu|draft_max|kcache|vcache|label
  # --- KV-quant ladder at FULL context: which one lets 128k fit at all? ---
  "131072|auto|8|false|2|q8_0|q8_0|ctx128k_kv-q8"
  "131072|auto|8|false|2|f16|f16|ctx128k_kv-f16"
  "131072|auto|8|false|2|q4_0|q4_0|ctx128k_kv-q4"
  # --- context ladder at the safe KV quant: cost of full ctx vs shorter ---
  "65536|auto|8|false|2|q8_0|q8_0|ctx64k_kv-q8"
  "32768|auto|8|false|2|q8_0|q8_0|ctx32k_kv-q8"
  # --- secondary levers, only meaningful once a fitting config is known ---
  "131072|auto|16|false|2|q8_0|q8_0|ctx128k_kv-q8_t16"
  "131072|auto|8|true|2|q8_0|q8_0|ctx128k_kv-q8_kvgpu"
)

wait_ready() {
  for _ in $(seq 1 90); do
    if curl -s "$HOST/api/v0/models" 2>/dev/null | grep -q "\"$ID\""; then return 0; fi
    sleep 2
  done
  echo "[sweep] ERROR: $ID not ready after timeout" >&2; return 1
}

run_config() {
  local ctx="$1" experts="$2" threads="$3" kvgpu="$4" draft="$5" kc="$6" vc="$7" label="$8"
  echo "=== [sweep] $QUANT :: $label (ctx=$ctx experts=$experts t=$threads kv_gpu=$kvgpu draft=$draft kv=$kc/$vc) ==="

  local sets=(--set ctx="$ctx" --set threads="$threads" --set kv_to_gpu="$kvgpu"
    --set draft_max="$draft" --set kcache="$kc" --set vcache="$vc"
    --set flash=true --set mtp=true)
  [ "$experts" != "auto" ] && sets+=(--set cpu_experts="$experts")
  python3 apply_config.py --gguf "$GGUF" "${sets[@]}" || return 1

  lms unload --all >/dev/null 2>&1
  if ! lms load "${MODEL_KEY}@${QUANT}" --identifier "$ID" --parallel "$PARALLEL" -y >/dev/null 2>&1; then
    echo "[sweep] ERROR: load failed for $label; skipping" >&2; return 1
  fi
  wait_ready || return 1

  local common=(--model "$ID" --model-name "$MODEL_KEY" --thinking "$THINKING" --quant "$QUANT"
    --ctx "$ctx" --parallel "$PARALLEL" --gpu max --mtp on
    --flash on --kv-quant "$kc" --threads "$threads"
    --label "$label" --runs "$RUNS" --warmup "$WARMUP")

  # Decode at each requested concurrency (CONC space-separated; 1=single-stream).
  # The load's --parallel must be >= the largest concurrency, or extra streams queue.
  for c in $CONC; do
    python3 bench.py "${common[@]}" --mode decode --prompt prompts/decode.txt \
      --max-tokens "$MAXTOK" --concurrency "$c" --out "$OUT"
  done
  [ "$PREFILL" = "1" ] && python3 bench.py "${common[@]}" --mode prefill --prompt prompts/prefill_16k.txt --out "$OUT"
}

echo "[sweep] model=$MODEL_KEY quant=$QUANT gguf=$GGUF thinking=$THINKING parallel=$PARALLEL"
for row in "${CONFIGS[@]}"; do
  IFS='|' read -r ctx experts threads kvgpu draft kc vc label <<<"$row"
  run_config "$ctx" "$experts" "$threads" "$kvgpu" "$draft" "$kc" "$vc" "$label"
done
echo "[sweep] done -> $OUT"
