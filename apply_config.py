#!/usr/bin/env python3
"""Edit LM Studio's per-model load config so advanced params can be swept.

LM Studio stores each model's load config at:
  ~/.lmstudio/.internal/user-concrete-model-default-config/<repo>/<file>.gguf.json
The server reads this when the model is loaded. `lms load` only exposes a few
flags (gpu/context/parallel/mtp); the *advanced* knobs below live only here.

Supported short keys -> full LM Studio keys:
  ctx           -> llm.load.contextLength                          (int)
  threads       -> llm.load.llama.cpuThreadPoolSize                (int)
  offload       -> llm.load.llama.acceleration.offloadRatio        (float 0..1)
  kv_to_gpu     -> llm.load.offloadKVCacheToGpu                     (bool)
  cpu_experts   -> llm.load.numCpuExpertLayersRatio                (float 0..1; 1=all experts on CPU)
  flash         -> llm.load.llama.flashAttention                   (bool)
  mtp           -> llm.load.llama.speculativeDecoding.draftMtp     (bool)
  draft_max     -> llm.load.llama.speculativeDecoding.draftMaxTokens (int)
  draft_min     -> llm.load.llama.speculativeDecoding.draftMinTokens (int)
  draft_prob    -> llm.load.llama.speculativeDecoding.draftMinContinueProbability (float)
  kcache        -> llm.load.llama.kCacheQuantizationType           (str: f16|q8_0|q4_0 or "off")
  vcache        -> llm.load.llama.vCacheQuantizationType           (str: f16|q8_0|q4_0 or "off")
  pbatch        -> llm.load.llama.physicalBatchSize                (int)
  keep_in_ram   -> llm.load.llama.keepModelInMemory                (bool; false = don't mlock weights)
  mmap          -> llm.load.llama.tryMmap                          (bool; true = page cache, reclaimable)

Usage:
  python3 apply_config.py --gguf unsloth/Qwen3.6-.../...-Q4_K_XL.gguf \
      --set cpu_experts=0.8 --set threads=8 --set draft_max=3 [--print]

NOTE on hub-resolved models: entries that `lms ls` shows as "publisher/name
(N variants)" (e.g. mistralai/devstral-small-2-2512) do NOT read the config at
their physical gguf path — they key it as <publisher>/<model-key>.json. Pass
--gguf "mistralai/devstral-small-2-2512" (no .gguf) for those, and verify the
load picked it up via `lms ps` (CONTEXT column). Physical-path GGUFs
(unsloth/..., bartowski/...) use <repo-dir>/<file>.gguf.json as documented.

Writes a .bak the first time it touches a file. Use --show to dump current config.
"""
import argparse
import json
import os
import sys

CFG_ROOT = os.path.expanduser(
    "~/.lmstudio/.internal/user-concrete-model-default-config")

# short key -> (full key, type, wrapped?)
KEYMAP = {
    "ctx": ("llm.load.contextLength", "int", False),
    "threads": ("llm.load.llama.cpuThreadPoolSize", "int", False),
    "offload": ("llm.load.llama.acceleration.offloadRatio", "float", False),
    "kv_to_gpu": ("llm.load.offloadKVCacheToGpu", "bool", False),
    "cpu_experts": ("llm.load.numCpuExpertLayersRatio", "float", False),
    "flash": ("llm.load.llama.flashAttention", "bool", False),
    "mtp": ("llm.load.llama.speculativeDecoding.draftMtp", "bool", False),
    "draft_max": ("llm.load.llama.speculativeDecoding.draftMaxTokens", "int", False),
    "draft_min": ("llm.load.llama.speculativeDecoding.draftMinTokens", "int", False),
    "draft_prob": ("llm.load.llama.speculativeDecoding.draftMinContinueProbability", "float", False),
    "kcache": ("llm.load.llama.kCacheQuantizationType", "str", True),
    "vcache": ("llm.load.llama.vCacheQuantizationType", "str", True),
    "pbatch": ("llm.load.llama.physicalBatchSize", "int", False),
    "keep_in_ram": ("llm.load.llama.keepModelInMemory", "bool", False),
    "mmap": ("llm.load.llama.tryMmap", "bool", False),
}


def cfg_path(gguf_rel: str) -> str:
    return os.path.join(CFG_ROOT, gguf_rel + ".json")


def coerce(t: str, raw: str):
    if t == "int":
        return int(raw)
    if t == "float":
        return float(raw)
    if t == "bool":
        return raw.lower() in ("1", "true", "yes", "on")
    return raw


def load_cfg(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"preset": "", "operation": {"fields": []}, "load": {"fields": []}}


def set_field(cfg: dict, full_key: str, value, wrapped: bool):
    fields = cfg.setdefault("load", {}).setdefault("fields", [])
    payload = {"checked": (value not in ("off", "none")), "value": value} if wrapped else value
    for f in fields:
        if f.get("key") == full_key:
            f["value"] = payload
            return
    fields.append({"key": full_key, "value": payload})


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True,
                   help="model gguf path relative to the LM Studio models dir")
    p.add_argument("--set", action="append", default=[], dest="sets",
                   help="short_key=value (repeatable)")
    p.add_argument("--show", action="store_true", help="print current config and exit")
    p.add_argument("--print", action="store_true", dest="do_print",
                   help="print resulting config after edits")
    a = p.parse_args()

    path = cfg_path(a.gguf)
    cfg = load_cfg(path)

    if a.show:
        print(json.dumps(cfg, indent=2))
        return 0

    for item in a.sets:
        if "=" not in item:
            print(f"bad --set '{item}', expected key=value", file=sys.stderr)
            return 2
        short, raw = item.split("=", 1)
        if short not in KEYMAP:
            print(f"unknown key '{short}'. known: {', '.join(KEYMAP)}", file=sys.stderr)
            return 2
        full, typ, wrapped = KEYMAP[short]
        set_field(cfg, full, coerce(typ, raw), wrapped)
        print(f"[apply] {short} -> {full} = {raw}", file=sys.stderr)

    if not os.path.exists(path + ".bak") and os.path.exists(path):
        with open(path) as src, open(path + ".bak", "w") as dst:
            dst.write(src.read())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[apply] wrote {path}", file=sys.stderr)
    if a.do_print:
        print(json.dumps(cfg, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
