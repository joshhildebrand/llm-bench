#!/usr/bin/env bash
# Advanced-parameter sweep driver for the LM Studio tok/s benchmark.
#
# For each config row it: writes the advanced load params into LM Studio's
# per-model config (apply_config.py), unloads, loads the target quant under a
# stable identifier, waits until healthy, then benchmarks decode + prefill.
# Results append to results/results.csv.
#
# Advanced params (flash attn, KV-cache quant, MoE-expert-CPU ratio, threads,
# draft depth, batch size) are NOT settable via `lms load` -- apply_config.py
# writes them to the concrete per-model config the server reads on load.
#
# Usage:
#   ./sweep.sh <model-key> <quant-tag> <gguf-rel-path> [thinking:allow|no_think]
# Example:
#   ./sweep.sh qwen3.6-35b-a3b-mtp q4_k_xl \
#     unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
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
PARALLEL="${PARALLEL:-1}"        # single-stream by default
PREFILL="${PREFILL:-1}"          # also run 16k prefill per config (0 to skip)

# Advanced-param matrix. One row = one benchmarked config.
# Fields: ctx|cpu_experts|threads|kv_to_gpu|draft_max|kcache|vcache|label
# cpu_experts: 1=all experts on CPU (frees VRAM); lower pushes experts onto GPU.
# Use "auto" for cpu_experts to let LM Studio pick (omit the override).
CONFIGS=(
  "131072|auto|8|false|2|q8_0|q8_0|ctx128k_expauto_t8"
  "131072|0.85|8|false|2|q8_0|q8_0|ctx128k_exp0.85_t8"
  "131072|0.70|8|false|2|q8_0|q8_0|ctx128k_exp0.70_t8"
  "131072|0.55|8|false|2|q8_0|q8_0|ctx128k_exp0.55_t8"
  "131072|auto|16|false|2|q8_0|q8_0|ctx128k_expauto_t16"
  "131072|auto|8|false|3|q8_0|q8_0|ctx128k_expauto_t8_draft3"
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

  python3 bench.py "${common[@]}" --mode decode --prompt prompts/decode.txt --max-tokens "$MAXTOK" --out "$OUT"
  [ "$PREFILL" = "1" ] && python3 bench.py "${common[@]}" --mode prefill --prompt prompts/prefill_16k.txt --out "$OUT"
}

echo "[sweep] model=$MODEL_KEY quant=$QUANT gguf=$GGUF thinking=$THINKING parallel=$PARALLEL"
for row in "${CONFIGS[@]}"; do
  IFS='|' read -r ctx experts threads kvgpu draft kc vc label <<<"$row"
  run_config "$ctx" "$experts" "$threads" "$kvgpu" "$draft" "$kc" "$vc" "$label"
done
echo "[sweep] done -> $OUT"
