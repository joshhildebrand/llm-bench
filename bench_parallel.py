#!/usr/bin/env python3
"""Multi-stream throughput benchmark against LM Studio's native REST API.

Fires K concurrent decode requests and measures AGGREGATE tokens/sec (sum of
completion tokens across all streams / wall-clock) plus mean per-stream tok/s.
Use to evaluate `--parallel N` load configs. The model must already be loaded
with parallel >= concurrency.

Example:
  python3 bench_parallel.py --model qwen3.6-35b-a3b-mtp --concurrency 4 \
      --prompt prompts/decode.txt --max-tokens 256 \
      --quant q8_k_xl --ctx 131072 --parallel 4 --out results/throughput.csv
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.request

HOST = os.environ.get("LMS_HOST", "http://localhost:1234")
ENDPOINT = HOST + "/api/v0/chat/completions"

CSV_COLUMNS = [
    "timestamp", "model", "label", "quant", "ctx", "parallel", "concurrency",
    "gpu_ratio", "flash", "kv_quant", "threads", "mtp",
    "agg_tok_s", "mean_stream_tok_s", "total_completion_tokens", "wall_s",
]


def one_request(prompt, max_tokens, timeout):
    payload = {
        "model": ARGS.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    if ARGS.seed is not None:
        payload["seed"] = ARGS.seed
    data = json.dumps(payload).encode()
    req = urllib.request.Request(ENDPOINT, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    return {
        "completion_tokens": body.get("usage", {}).get("completion_tokens", 0),
        "stream_tok_s": body.get("stats", {}).get("tokens_per_second"),
    }


def main():
    with open(ARGS.prompt) as f:
        prompt = f.read()
    if ARGS.thinking == "no_think":
        prompt = prompt.rstrip() + "\n/no_think"

    print(f"[bench_parallel] {ARGS.model} concurrency={ARGS.concurrency} "
          f"max_tokens={ARGS.max_tokens}", file=sys.stderr)

    # One warmup round to load caches / spin up slots.
    with cf.ThreadPoolExecutor(max_workers=ARGS.concurrency) as ex:
        list(ex.map(lambda _: one_request(prompt, 8, ARGS.timeout),
                    range(ARGS.concurrency)))

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=ARGS.concurrency) as ex:
        results = list(ex.map(
            lambda _: one_request(prompt, ARGS.max_tokens, ARGS.timeout),
            range(ARGS.concurrency)))
    wall = time.time() - t0

    total_tokens = sum(r["completion_tokens"] for r in results)
    stream_rates = [r["stream_tok_s"] for r in results if r["stream_tok_s"]]
    agg = round(total_tokens / wall, 3) if wall else None
    mean_stream = round(sum(stream_rates) / len(stream_rates), 3) if stream_rates else None

    row = {
        "timestamp": int(time.time()), "model": ARGS.model, "label": ARGS.label,
        "quant": ARGS.quant, "ctx": ARGS.ctx, "parallel": ARGS.parallel,
        "concurrency": ARGS.concurrency, "gpu_ratio": ARGS.gpu, "flash": ARGS.flash,
        "kv_quant": ARGS.kv_quant, "threads": ARGS.threads, "mtp": ARGS.mtp,
        "agg_tok_s": agg, "mean_stream_tok_s": mean_stream,
        "total_completion_tokens": total_tokens, "wall_s": round(wall, 3),
    }

    write_header = not os.path.exists(ARGS.out) or os.path.getsize(ARGS.out) == 0
    os.makedirs(os.path.dirname(ARGS.out) or ".", exist_ok=True)
    with open(ARGS.out, "a") as f:
        if write_header:
            f.write(",".join(CSV_COLUMNS) + "\n")
        f.write(",".join(str(row[c]) for c in CSV_COLUMNS) + "\n")

    print(f"[bench_parallel] AGGREGATE {agg} tok/s across {ARGS.concurrency} streams "
          f"(mean per-stream {mean_stream} tok/s, wall {row['wall_s']}s) -> {ARGS.out}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=256, dest="max_tokens")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--thinking", choices=["allow", "no_think"], default="allow")
    p.add_argument("--out", default="results/throughput.csv")
    p.add_argument("--label", default="")
    p.add_argument("--quant", default="")
    p.add_argument("--ctx", default="")
    p.add_argument("--parallel", default="")
    p.add_argument("--gpu", default="")
    p.add_argument("--flash", default="")
    p.add_argument("--kv-quant", default="", dest="kv_quant")
    p.add_argument("--threads", default="")
    p.add_argument("--mtp", default="")
    ARGS = p.parse_args()
    sys.exit(main())
