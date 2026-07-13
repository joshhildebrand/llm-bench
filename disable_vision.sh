#!/usr/bin/env bash
# Disable the Qwen3.6 vision projector so text-only inference frees ~1.8 GB VRAM.
# Required for the Q4 config to load at 128k context on an 8 GB GPU (otherwise the
# server SIGABRTs with cudaMalloc OOM inside clip_init). Reversible: restore_vision.sh
set -euo pipefail
D="$HOME/.lmstudio/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF"
if [ -f "$D/mmproj-F32.gguf" ]; then
  mv "$D/mmproj-F32.gguf" "$D/mmproj-F32.gguf.disabled"
  echo "vision projector disabled (text-only). Restore with ./restore_vision.sh"
elif [ -f "$D/mmproj-F32.gguf.disabled" ]; then
  echo "already disabled."
else
  echo "no mmproj file found in $D"
fi
