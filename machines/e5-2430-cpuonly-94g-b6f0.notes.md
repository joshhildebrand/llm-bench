**A pre-AVX2 box, benchmarked as a deliberate counterexample.** Xeon E5-2430 (Sandy Bridge-EN,
6c/12t), **no AVX2, no FMA, no F16C**, 6× 16 GB DDR3 in **3 channels** clocked to 1333 MT/s.
CPU-only, llama.cpp, same model/quant as `e5-2680-cpuonly-63g-b105` so the two are comparable.

### ⭐ Decode degrades gracefully; prefill falls off a cliff

| | this box | e5-2680 v4 (AVX2) | ratio |
|---|---:|---:|---:|
| STREAM Triad | 20.1 GB/s | 48.0 GB/s | 2.39× |
| **decode** | **5.58 tok/s** | 12.95 | **2.32×** |
| **prefill (2k)** | **8.34 tok/s** | ~65 | **7.8×** |
| **TTFT, 2k prompt** | **169.7 s** | ~21 s | — |

Decode tracks memory bandwidth almost exactly — the missing ISA barely matters, because decode
is memory-bound. **Prefill collapses 7.8×**, far past the bandwidth gap, because prefill is
compute-bound and that is exactly what AVX2/FMA accelerate. Cleanest separation of the two
regimes I have measured anywhere.

Practical consequence: at 8.34 tok/s prefill, **2k ≈ 2.8 min to first token, 8k ≈ 16 min,
32k ≈ 64 min.** Long-context, RAG and agentic use are out on TTFT alone.

### ⛔ MTP is a NET LOSS here

| depth | tok/s | vs off | acceptance |
|---:|---:|---:|---:|
| off | **5.58** | — | — |
| 1 | 5.19 | −7.1% | 0.801 |
| 2 | 4.64 | −14.8% | 0.706 |
| 4 | 3.88 | −28.8% | 0.509 |

Acceptance is healthy (0.801 vs the Broadwell box's 0.848), so this is not a prediction-quality
failure. Speculative decoding **trades compute for bandwidth**, and this CPU has no spare
compute to trade. Note the inversion: the *more* bandwidth-starved box is the one where MTP
does **not** pay — the compute-to-bandwidth **ratio** is the predictor, not absolute scarcity.

### Other results

- **Hyperthreading is neutral** (5.58 → 5.53 at 6→12 threads, within ±1%) versus a clear −11%
  on the AVX2 box. SMT hides latency when the core issues many simple non-vector ops. STREAM
  still regresses with HT (19.8 → 18.9), so it is specifically compute-starved decode that is
  indifferent.
- **Bandwidth saturates at 3 threads = 3 channels** (12.3 / 20.1 / 19.8 / 18.9 at 1/3/6/12).
- **0.628 of theoretical** (20.1 of 32.0) — versus **0.625** on the DDR4/Broadwell box. Two
  memory generations, same ratio to within 0.3%; that factor looks generation-independent.
- **~1.72× more instructions retired** for identical work (1.079×10¹² vs 6.28×10¹¹). Its
  *higher* IPC (1.35 vs 0.82) is a symptom, not a virtue — **do not compare platforms by IPC.**

### Caveats

Six DIMMs, but **two are DDR3-1333 and four are DDR3-1600**, so all six clock at 1333 — the
slowest module sets the bus. Matched 1600 sticks would give 38.4 GB/s theoretical (+20%).
This was a temporary host measured while idle between jobs; the configuration no longer exists,
so these numbers are comparable but not re-runnable.
