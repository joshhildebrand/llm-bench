# llm-bench

A small, dependency-free harness to measure **tokens/sec** of local models served by
[LM Studio](https://lmstudio.ai), and to sweep load/runtime configs to find the fastest one
**on whatever box you run it on**. Model-agnostic (add a block to `models.json`) and
machine-agnostic (each machine tags its own results and gets its own results page).

## Why this exists

The fastest config for a local LLM depends on where the bottleneck is, and that moves with
your hardware. Two regimes dominate for a Mixture-of-Experts model (e.g. 35B total / ~3B
active, where only the active experts are read per token):

- **Fits-in-VRAM (GPU-bound).** If the weights + KV cache fit in your GPU(s), decode is bound
  by GPU memory bandwidth — very fast. The lever is: pick the largest quant that still fits.
- **Hybrid (RAM-bandwidth-bound).** If it doesn't fit, the win is to keep every layer's
  attention/router/shared-expert on the GPU and stream the bulk experts from system RAM
  (`n-cpu-moe` / `numCpuExpertLayersRatio`). Decode is then bound by RAM bandwidth and the
  bytes-read-per-token (i.e. quantization).

This harness *measures* the effect of each change instead of guessing, and records every run
so nothing is anecdotal.

## How it measures

Hits LM Studio's **native** endpoint `POST /api/v0/chat/completions`, which returns per-request
`stats.tokens_per_second`, `stats.time_to_first_token`, and `usage` token counts (incl. MTP draft
acceptance). No OpenAI/LM Studio SDK needed — stdlib `urllib` only.

- **decode** mode → `tg_tok_s = stats.tokens_per_second` over a long generation. This is the number
  that matters for interactive chat.
- **prefill** mode → `pp_tok_s = prompt_tokens / time_to_first_token` over a large fixed prompt
  (`prompts/prefill_16k.txt`). Measures prompt-processing throughput.

**Concurrency is first-class.** `bench.py --concurrency N` fires N simultaneous streams and records,
on one row: `concurrency`, the per-stream decode rate (`tg_tok_s`), and the aggregate decode rate
across all streams (`tg_tok_s_agg = tg_tok_s × concurrency`). Single-stream is just `concurrency=1`
(the two decode columns are then equal), so single- and multi-stream results are directly comparable
in one table — no separate throughput file.

Each measurement runs `--warmup` discarded requests then `--runs` measured, reporting the **median**.
Every run appends a row to `results/results.csv` tagged with the full config **and the machine id**.

## Files

| File | Purpose |
|---|---|
| `bench.py` | decode/prefill benchmark, any concurrency → `results/results.csv` |
| `sweep.sh` | Driver: loops configs (and concurrency), `lms load`/`unload`, calls `bench.py` |
| `apply_config.py` | Writes advanced load params LM Studio doesn't expose via `lms load` |
| `machine.py` | Detects hardware specs, mints/records this machine's id (see below) |
| `ramguard.ps1` | Watchdog for risky (near-RAM-size) loads: auto-unloads on low commit headroom |
| `report.py` | Builds a machine's results page `machines/<id>.md` from the CSV |
| `models.json` | Model matrix (add a block to benchmark a new model) |
| `machines/` | One `<id>.json` (specs) + `<id>.md` (results) per contributing machine |
| `prompts/` | Fixed prompts; `gen_prompts.py` regenerates the prefill files |

## Multi-machine tracking

Results from many boxes share one CSV and one repo. Each row carries a non-identifying
`machine_id` (a spec-derived slug + random suffix, e.g. `7-9800x3d-2x5060ti16g-64g-f732`), and
each machine gets its own page under [`machines/`](machines/). Identity is automatic:

```bash
python3 machine.py --ensure     # detect specs, mint id, write machines/<id>.json
# ... run a sweep (below); every result row is stamped with this machine's id ...
python3 report.py               # (re)build machines/<id>.md for this machine
```

The id lives in `.machine_id` (gitignored), so a fresh clone always mints its **own** id
instead of inheriting one. Only hardware **specs** are recorded — never hostname, username, or
serials — so the pages are safe to publish. See [`machines/README.md`](machines/README.md).

## Usage

```bash
# One measurement against an already-loaded model (single stream):
python3 bench.py --model <identifier> --mode decode \
    --prompt prompts/decode.txt --max-tokens 256 --runs 3 --warmup 1 \
    --quant q4_k_xl --ctx 131072 --parallel 1 --gpu max --mtp on

# Concurrency: N simultaneous streams (load the model with --parallel >= N):
python3 bench.py --model <identifier> --mode decode --concurrency 4 \
    --prompt prompts/decode.txt --quant q4_k_xl --ctx 32768 --parallel 4

# Full config sweep (unloads/loads per row in sweep.sh CONFIGS):
./sweep.sh <model-key> <quant-tag> <gguf-rel-path>

# Sweep across concurrency levels too (load must allow it: PARALLEL >= max CONC):
CONC="1 2 4 8" PARALLEL=8 ./sweep.sh <model-key> <quant-tag> <gguf-rel-path>
```

On Windows, run `sweep.sh` from Git Bash (the `lms` CLI, `curl`, and `python3` are all on PATH);
`bench.py` / `machine.py` / `report.py` are pure-stdlib Python and run anywhere.

### CLI vs GUI knobs

`sweep.sh` sets these via `lms load`: `--gpu`, `-c/--context-length`, `--parallel`,
`--speculative-draft-mtp`. The **advanced** knobs — flash attention, KV-cache quantization,
`numCpuExpertLayersRatio` (force MoE experts to CPU), CPU thread count, MTP draft depth — are not
exposed by `lms load`; `apply_config.py` writes them into the concrete per-model config the server
reads on load.

### Benchmarking a specific quant

Download the quant and load the variant you want:
```bash
lms get <hf_repo>@<quant>     # e.g. unsloth/Qwen3.6-35B-A3B-MTP-GGUF@q4_k_xl
```
Pass the resulting model key to `sweep.sh` with a matching `<quant-tag>` so the CSV records it.

## What tends to move the needle

Empirically, across machines (see each `machines/<id>.md` for the numbers on that box):

1. **Fit the whole model in VRAM if you can (biggest lever when you have the VRAM).** A quant
   that fully fits GPU memory at your target context decodes far faster than any hybrid, because
   nothing streams from RAM. On a multi-GPU box this also sidesteps a slow PCIe link between cards:
   with layer-split, only a tiny per-token hidden state crosses the link, not the weights.
2. **Quantization is the other big lever.** Fewer bytes-read-per-token → faster decode when you're
   RAM- or VRAM-bandwidth-bound, and it's what *lets* a model fit. UD dynamic quants hold quality
   well on MoE; verify coherence at the quant you pick.
3. **When it doesn't fit, use MoE-aware hybrid offload.** Keep attention/router/shared-expert on
   GPU (`n-gpu-layers=max`), force bulk experts to CPU (`numCpuExpertLayersRatio`), then lower that
   ratio a few layers at a time to fill spare VRAM — stop one step before it overfills and regresses.
4. **MTP / speculative draft depth.** If the model ships a multi-token-prediction head, drafting
   ~2–4 tokens per step is a free decode speedup (acceptance-dependent; higher on code/structured
   output). `draft_max=4` is a good default; past that, acceptance falls.
5. **Threads = physical cores.** Hyperthreading hurts bandwidth-bound MoE decode — never use all
   logical threads.
6. **Right-size context and parallelism.** KV cache is pre-allocated for your full context and
   competes with weights for VRAM; big context or `parallel>1` can force a spill. Use `parallel=1`
   unless you specifically need concurrent sessions.
7. **Watch auto-loaded vision projectors.** For multimodal GGUFs, LM Studio auto-loads the mmproj
   onto the GPU; for text-only use it just wastes VRAM (and can OOM at high context). Disable it
   (`disable_vision.sh` / `restore_vision.sh`).

**The best quant depends on your regime — that's why each machine has its own page.** On a
RAM-bandwidth-bound box (model doesn't fit VRAM) the speed curve across Q4→Q5→Q6 is nearly *flat*,
so you buy precision almost for free — pick the higher quant. On a VRAM-rich, GPU-bound box (model
fits) the curve is *steep* and smaller quants win, so the "best-quality Q4" can actually be ~20%
slower than a leaner Q4. Same model, opposite advice — compare the pages under `machines/`. A related
warning: at very high context the GPU attention buffer can force enough expert offload that **prefill
collapses** even while decode stays fast, so validate prompt-ingest time, not just tok/s.

## Recommended workflow

1. `python3 machine.py --ensure` to register the box.
2. Find the largest quant that **fully fits** your VRAM at your target context (start there — it's
   usually fastest). If none fit, drop to hybrid offload and tune `numCpuExpertLayersRatio`.
3. Flash attention on; KV cache `q8_0`; MTP on with `draft_max=4`; threads = physical cores;
   `parallel=1`.
4. `python3 report.py` to write your machine's results page, then commit `machines/` + `results/`.
