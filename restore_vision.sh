#!/usr/bin/env bash
# Re-enable the Qwen3.6 vision projector (needed for image inputs). Note: with
# vision on, the Q4 128k config will not fit on the 8 GB GPU -- reduce context or
# GPU offload (numCpuExpertLayersRatio -> higher) to load it, at a speed cost.
set -euo pipefail
D="$HOME/.lmstudio/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF"
if [ -f "$D/mmproj-F32.gguf.disabled" ]; then
  mv "$D/mmproj-F32.gguf.disabled" "$D/mmproj-F32.gguf"
  echo "vision projector restored."
else
  echo "nothing to restore (mmproj-F32.gguf.disabled not found)."
fi
