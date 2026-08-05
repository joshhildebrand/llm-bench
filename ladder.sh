#!/usr/bin/env bash
# cpu_experts ladder for a hybrid (doesn't-fit-VRAM) model, at a FIXED context.
#
# sweep.sh sweeps a preset config matrix and hardcodes MTP on; this instead walks a
# single knob -- numCpuExpertLayersRatio -- and records the VRAM each rung actually
# consumed, which is the number you need to find the rung before the model overfills.
# Use it for models where no quant fits VRAM and expert *placement* is the only lever.
#
# Usage:
#   ./ladder.sh <model-key> <quant-tag> <gguf-rel-path> <ctx> <ratios...>
# Example:
#   ./ladder.sh laguna-s-2.1 ud_q2_k_xl \
#     unsloth/Laguna-S-2.1-GGUF/Laguna-S-2.1-UD-Q2_K_XL.gguf 262144 1.0 0.85 0.70
set -uo pipefail
cd "$(dirname "$0")"

MODEL_KEY="${1:?usage: ladder.sh <model-key> <quant> <gguf> <ctx> <ratios...>}"
QUANT="${2:?quant tag}"
GGUF="${3:?gguf path relative to LM Studio models dir}"
CTX="${4:?context length}"
shift 4
RATIOS=("$@")

ID="bench"
OUT="results/results.csv"
HOST="${LMS_HOST:-http://localhost:1234}"
RUNS="${RUNS:-3}"; WARMUP="${WARMUP:-1}"; MAXTOK="${MAXTOK:-256}"
THREADS="${THREADS:-8}"; KC="${KC:-q8_0}"; VC="${VC:-q8_0}"
PREFILL="${PREFILL:-1}"; PREFILL_PROMPT="${PREFILL_PROMPT:-prompts/prefill_16k.txt}"
VRAMLOG="${VRAMLOG:-results/vram-ladder.log}"

wait_ready() {
  for _ in $(seq 1 120); do
    curl -s "$HOST/api/v0/models" 2>/dev/null | grep -q "\"$ID\"" && return 0
    sleep 2
  done
  echo "[ladder] ERROR: $ID not ready" >&2; return 1
}

# Per-GPU used MiB, comma separated -- the signal for "which rung overfilled".
vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd, -; }

mkdir -p "$(dirname "$VRAMLOG")"
echo "[ladder] $MODEL_KEY $QUANT ctx=$CTX ratios=${RATIOS[*]}"
for r in "${RATIOS[@]}"; do
  label="ctx$((CTX/1024))k_exp${r}"
  echo "=== [ladder] $label ==="

  sets=(--set ctx="$CTX" --set threads="$THREADS" --set kcache="$KC" --set vcache="$VC"
        --set flash=true --set mtp=false --set kv_to_gpu=true
        --set keep_in_ram=false --set mmap=true)
  [ "$r" != "auto" ] && sets+=(--set cpu_experts="$r")
  python3 apply_config.py --gguf "$GGUF" "${sets[@]}" >/dev/null 2>&1 || continue

  lms unload --all >/dev/null 2>&1
  if ! timeout 900 lms load "$MODEL_KEY" --identifier "$ID" --parallel 1 -y >/dev/null 2>&1; then
    echo "[ladder] LOAD FAILED at cpu_experts=$r (likely VRAM cap) -- stopping ladder" >&2
    echo "$(date +%H:%M:%S) $label LOAD_FAILED" >>"$VRAMLOG"
    break
  fi
  wait_ready || break

  v="$(vram)"
  echo "$(date +%H:%M:%S) $label vram_mib=$v" >>"$VRAMLOG"
  echo "[ladder] VRAM after load (MiB per gpu): $v"

  common=(--model "$ID" --model-name "$MODEL_KEY" --thinking allow --quant "$QUANT"
          --ctx "$CTX" --parallel 1 --gpu max --mtp off --flash on
          --kv-quant "$KC" --threads "$THREADS" --label "$label"
          --runs "$RUNS" --warmup "$WARMUP")

  python3 bench.py "${common[@]}" --mode decode --prompt prompts/decode.txt \
      --max-tokens "$MAXTOK" --concurrency 1 --out "$OUT"
  [ "$PREFILL" = "1" ] && python3 bench.py "${common[@]}" --mode prefill \
      --prompt "$PREFILL_PROMPT" --out "$OUT"
done
echo "[ladder] done -> $OUT (vram: $VRAMLOG)"
