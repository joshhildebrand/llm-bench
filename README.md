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

## Results

See `results/results.csv` (single-stream) and `results/throughput.csv` (concurrent). Findings and the
recommended single-stream / throughput presets are written up here as the sweep progresses.

_TBD — populated by the optimization sweep._
