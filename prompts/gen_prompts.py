#!/usr/bin/env python3
"""Generate deterministic fixed-size prefill prompts for the benchmark.

Prefill prompts are used to measure prompt-processing (prefill) tok/s: a large
fixed context followed by an instruction to reply with a single token, so
time_to_first_token ~= time to process the whole prompt.

Output byte-identical files so every run processes the same prompt. Run:
    python3 prompts/gen_prompts.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# A fixed base paragraph (deterministic, no randomness) repeated to reach a size.
BASE = (
    "Memory bandwidth is the primary constraint on decode throughput for large "
    "language models running on commodity CPUs. Each generated token requires "
    "streaming the active weights through the cache hierarchy from main memory, "
    "so a quad-channel DDR4 configuration running at a fixed transfer rate sets a "
    "hard ceiling on tokens per second. Mixture-of-experts architectures reduce "
    "the active parameter count per token, which shifts the balance between "
    "compute and bandwidth and changes how offloading layers to a GPU helps. "
)

# Rough heuristic: ~4 characters per token for English prose.
TARGETS = {"prefill_2k.txt": 2000, "prefill_16k.txt": 16000}
INSTRUCTION = (
    "\n\nBased only on the passage above, reply with exactly one word: OK"
)


def build(target_tokens: int) -> str:
    target_chars = target_tokens * 4
    body = (BASE * (target_chars // len(BASE) + 1))[:target_chars]
    return body + INSTRUCTION


def main() -> None:
    for name, toks in TARGETS.items():
        text = build(toks)
        path = os.path.join(HERE, name)
        with open(path, "w") as f:
            f.write(text)
        print(f"wrote {path} ({len(text)} chars, ~{toks} tokens target)")


if __name__ == "__main__":
    main()
