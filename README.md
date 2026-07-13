# llm-bench

A small, dependency-free harness to measure **tokens/sec** of local models served by
[LM Studio](https://lmstudio.ai), and to sweep load/runtime configs to find the fastest one.
Built to optimize `qwen3.6-35b-a3b-mtp` on a bandwidth-constrained box, but **model-agnostic** —
add a block to `models.json` and it benchmarks anything LM Studio can serve.

## Why this exists

Decode speed on CPU-heavy setups is dominated by **memory bandwidth**: every generated token
streams the active weights through the cache hierarchy from RAM. For a Mixture-of-Experts model
(35B total / ~3B active) only the active experts are read per token, so the wins come from:
shrinking bytes-read-per-token (quantization), right-sizing the KV cache, and offloading the
compute-heavy attention to the GPU while experts stream from CPU RAM. This harness measures the
effect of each change instead of guessing.

## Target machine

| | |
|---|---|
| CPU | Intel i7-5960X, 8C/16T (Haswell-E), OC ~4.3 GHz |
| RAM | 64 GB DDR4 @ 2400 MT/s, quad-channel (~77 GB/s ceiling; not OC'd) |
| GPU | NVIDIA GTX 1070, 8 GB, compute 6.1 (Pascal) |
| Runtime | LM Studio `llama.cpp nvidia-cuda-avx2 v2.24.0` |

Baseline before tuning: **~15.9 tok/s decode** at Q8_K_XL / 256k ctx / parallel 4, with MTP
speculative decoding already active (~100% draft acceptance on repetitive prompts).

## How it measures

Hits LM Studio's **native** endpoint `POST /api/v0/chat/completions`, which returns per-request
`stats.tokens_per_second`, `stats.time_to_first_token`, and `usage` token counts (incl. MTP draft
acceptance). No OpenAI/LM Studio SDK needed — stdlib `urllib` only.

- **decode** mode → `tg_tok_s = stats.tokens_per_second` over a long generation. This is the number
  that matters for interactive chat.
- **prefill** mode → `pp_tok_s = prompt_tokens / time_to_first_token` over a large fixed prompt
  (`prompts/prefill_16k.txt`). Measures prompt-processing throughput.

Each measurement runs `--warmup` discarded requests then `--runs` measured, reporting the **median**.
Every run appends a row to `results/results.csv` tagged with the full config so nothing is anecdotal.

## Files

| File | Purpose |
|---|---|
| `bench.py` | Single-stream decode/prefill benchmark → `results/results.csv` |
| `bench_parallel.py` | Concurrent throughput benchmark → `results/throughput.csv` |
| `sweep.sh` | Driver: loops configs, `lms load`/`unload`, calls the benchmarks |
| `models.json` | Model matrix (add a block to benchmark a new model) |
| `prompts/` | Fixed prompts; `gen_prompts.py` regenerates the prefill files |

## Usage

```bash
# One measurement against an already-loaded model:
python3 bench.py --model qwen3.6-35b-a3b-mtp --mode decode \
    --prompt prompts/decode.txt --max-tokens 256 --runs 3 --warmup 1 \
    --quant q8_k_xl --ctx 131072 --parallel 1 --gpu max --mtp on

# Full config sweep (unloads/loads per row in sweep.sh CONFIGS):
FLASH=on KV_QUANT=q8_0 THREADS=8 ./sweep.sh qwen3.6-35b-a3b-mtp q8_k_xl

# Add throughput passes:
THROUGHPUT=1 ./sweep.sh qwen3.6-35b-a3b-mtp q8_k_xl
```

### CLI vs GUI knobs

`sweep.sh` sets these via `lms load`: `--gpu`, `-c/--context-length`, `--parallel`,
`--speculative-draft-mtp`. These are **not** exposed by `lms load` and must be set in the LM Studio
UI (then exported so they land in the CSV): **flash attention**, **KV-cache quantization**,
**force MoE expert weights to CPU / offload KV to GPU**, **CPU thread count**. Sweep GUI knobs in
batches: set the toggle, run `sweep.sh` with `FLASH=/KV_QUANT=/THREADS=` matching what you set.

### Benchmarking a specific quant

To compare quants, download them and load the variant you want:
```bash
lms get unsloth/Qwen3.6-35B-A3B-MTP-GGUF@q4_k_xl
```
Pass the resulting model key to `sweep.sh` with a matching `<quant-label>` so the CSV records it.

## Results (Qwen3.6-35B-A3B, this machine)

Full data in `results/results.csv` + `results/throughput.csv`. Decode = generation tok/s (what you
feel in chat); prefill = prompt ingest tok/s. All single-stream unless noted.

| Config | Decode tok/s | vs baseline | Notes |
|---|---:|---:|---|
| **Q8_K_XL, 256k ctx, parallel 4 (as-found baseline)** | 15.7 | — | MTP on, KV q8_0, flash on |
| Q8_K_XL, 128k, parallel 1 | 16.3 | +4% | freeing KV barely helps — Q8 is VRAM-capped |
| Q4_K_XL, 128k, MTP draft 2 | 19.2 | +22% | quant is the big lever |
| Q4_K_XL, 128k, MTP **draft 4** | 20.6 | +31% | deeper MTP drafting |
| **Q4_K_XL, 128k, draft 4, experts 0.70, threads 8** | **20.8** | **+33%** | max speed |
| **Q5_K_XL, 128k, draft 4, threads 8** | **20.2** | **+29%** | **best balance — ~Q4 speed, higher precision + 95% MTP accept** |
| Q6_K_XL, 128k, draft 4, threads 8 | 19.0 | +21% | slower, diminishing returns |

**Quant curve (all at the tuned 128k config):** the decode rate barely moves across Q4→Q5→Q6 (20.8 /
20.2 / 19.0) even though file size grows 23→27→33 GB. Decode is bound by *active* expert bytes/token,
and the UD quants' active-tensor delta between Q4 and Q5 is small — so **Q5 buys precision for ~free**.
Only Q8 (15.7) falls off, because at 40.9 GB far fewer expert layers fit on the GPU.

Prefill is not a bottleneck: **16k tokens ingest in ~3 s (≈3700 tok/s)** even at 128k context, so
large context is cheap on the prompt side — the cost of big context is memory, not prefill time.

### What moved the needle (and what didn't)

1. **Quantization Q8→Q4 (biggest lever, +~1.5× raw decode).** Decode here is bound by streaming the
   ~3B active expert weights from RAM (quad-channel DDR4-2400 ≈ 77 GB/s). Q4 ~halves bytes/token.
   Q4 output quality verified coherent (correct arithmetic, clean reasoning) — UD dynamic quants hold
   up well on MoE. **This is the single most important change.**
2. **MTP draft depth 2→4 (+~7%).** The model ships a multi-token-prediction head; drafting 4 tokens
   with ~92% acceptance beats the default 2. Past 4 gives nothing (accept rate falls). `draft_max=4`.
3. **Threads = 8 (physical cores).** 8 → 20.8, 6 → 19.3, **16 → 16.3**. Hyperthreading badly hurts this
   bandwidth-bound workload — never use all 16 threads.
4. **Disable the vision projector for text.** The repo ships `mmproj-F32.gguf` (1.8 GB F32) which LM
   Studio auto-loads onto the GPU. At Q4 (more layers offloaded) it overflows the 8 GB card and the
   server **SIGABRTs at 128k** (`cudaMalloc failed` inside `clip_init`). Renaming it `.disabled` frees
   ~1.8 GB and is what lets Q4 load at 128k. See `disable_vision.sh` / `restore_vision.sh`.
5. **Marginal / no effect:** `numCpuExpertLayersRatio` (VRAM is saturated ~7 GB either way, so pushing
   more experts to GPU doesn't fit — 0.70 ≈ 0.60), KV-cache→GPU (no room), context 128k vs 256k.

### Throughput / concurrency — no gain on this box

Aggregate tok/s is **flat at ~20 regardless of concurrency** (1/2/4 streams → 19.6 / 20.4 / 19.7),
per-stream just divides (20 → 10.4 → 5.3). The shared expert-weight reads saturate memory bandwidth,
so batching users does **not** raise total throughput here — unlike a GPU-bound setup. **Use
parallel=1**; only raise it if you specifically need concurrent sessions and accept proportionally
slower each.

### Full context (256k) — works, but with a catch

The model's max context is 262144 (256k). It **loads and runs at ~21 tok/s** at the fast config, but
the GPU attention/compute buffer for 256k nearly fills the 8 GB card, so a non-trivial prompt OOMs and
**crashes the server**. Making 256k stable requires pushing more experts to CPU
(`cpu_experts ≈ 0.78–0.82`) for VRAM headroom — which is fine for decode but **collapses prefill**:

| Context | Empty decode | 16k-prompt prefill | Notes |
|---|---:|---:|---|
| **128k, experts 0.70** | 20.8 | **~3 s (3700 tok/s)** | stable, recommended |
| 256k, experts 0.70 | 20.9 | **crashes (OOM)** | not usable with real prompts |
| 256k, experts 0.82 | 21.2 | **~55 s (290 tok/s)** | stable but prefill 18× slower |

So 256k is usable only if the context fills slowly via generation; for ingesting large prompts it's
impractical on 8 GB VRAM. **128k is the sweet spot** — full speed, fast prefill, stable headroom.
(A `cpu_experts=0.82` 256k preset works if you truly need the depth and accept slow prompt ingest.)

### The hard ceiling

CPU-side decode is capped by RAM bandwidth. DDR4-2400 quad-channel ≈ 77 GB/s; at ~3.3 GB/token (Q4)
that's a theoretical ~23 tok/s, and we reach ~21. The only way past it is faster RAM (BIOS XMP to
2666–2933 ≈ +10–20%) — the CPU is overclocked but the RAM is not. That's a BIOS change, out of scope
here, but it's the one lever left.

## Recommended settings

Both are single-stream, 128k context, vision projector disabled. **Pick Q5 unless you want the last
~3%.** Presets in `presets/`.

**Q5_K_XL — best balance (default, loaded):** `presets/qwen3.6-q5-balanced.json` — ctx 131072, flash on,
KV q8_0, MTP draft 4, threads 8, experts 0.80. → **~20.2 tok/s, higher precision, 95% MTP accept.**

**Q4_K_XL — max speed:** `presets/qwen3.6-q4-single-stream.json` — same but experts 0.70. → **~20.8 tok/s.**

```bash
./disable_vision.sh                       # one-time; frees 1.8 GB VRAM for text use
GGUF=unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf   # or ...Q4...
python3 apply_config.py --gguf "$GGUF" \
  --set ctx=131072 --set flash=true --set kcache=q8_0 --set vcache=q8_0 \
  --set mtp=true --set draft_max=4 --set threads=8 --set cpu_experts=0.80 --set kv_to_gpu=false
lms load qwen3.6-35b-a3b-mtp@q5_k_xl --identifier bench --parallel 1 -y
```

Q6_K_XL (19.0 tok/s) offers no advantage over Q5 here — same fidelity ballpark, slower. Q8_K_XL only
if you need maximum fidelity and accept 15.7 tok/s.
