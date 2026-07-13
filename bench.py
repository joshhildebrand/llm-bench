#!/usr/bin/env python3
"""Single-stream tokens/sec benchmark against LM Studio's native REST API.

Hits POST /api/v0/chat/completions, which returns per-request timing in
`stats` (tokens_per_second, time_to_first_token, generation_time) and token
counts in `usage` (including MTP draft acceptance). No third-party deps.

Two modes:
  decode  -> pure generation rate. Long output, tg_tok_s = stats.tokens_per_second.
  prefill -> prompt-processing rate. 1-token output over a big prompt,
             pp_tok_s = prompt_tokens / time_to_first_token.

Runs W warmup requests (discarded) then N measured; reports the MEDIAN and
appends one row to the results CSV. The model must already be loaded (sweep.sh
loads it via `lms load`).

Example:
  python3 bench.py --model qwen3.6-35b-a3b-mtp --mode decode \
      --prompt prompts/decode.txt --max-tokens 256 --runs 3 --warmup 1 \
      --quant q8_k_xl --ctx 262144 --parallel 4 --gpu max --mtp on \
      --out results/results.csv
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

HOST = os.environ.get("LMS_HOST", "http://localhost:1234")
ENDPOINT = HOST + "/api/v0/chat/completions"

CSV_COLUMNS = [
    "timestamp", "model", "label", "quant", "ctx", "parallel", "gpu_ratio",
    "flash", "kv_quant", "threads", "mtp", "mode", "prompt_tokens",
    "completion_tokens", "pp_tok_s", "tg_tok_s", "ttft_s", "accept_rate", "runs",
]


def post(payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        ENDPOINT, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def one_request(prompt: str, max_tokens: int, timeout: float) -> dict:
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    if args.seed is not None:
        payload["seed"] = args.seed
    body = post(payload, timeout)
    stats = body.get("stats", {})
    usage = body.get("usage", {})
    ttft = stats.get("time_to_first_token")
    ptoks = usage.get("prompt_tokens", 0)
    total_draft = usage.get("total_draft_tokens_count") or 0
    accepted = usage.get("accepted_draft_tokens_count") or 0
    return {
        "tg_tok_s": stats.get("tokens_per_second"),
        "ttft_s": ttft,
        "pp_tok_s": (ptoks / ttft) if (ttft and ptoks) else None,
        "prompt_tokens": ptoks,
        "completion_tokens": usage.get("completion_tokens", 0),
        "accept_rate": (accepted / total_draft) if total_draft else None,
    }


def median(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.median(vals), 3) if vals else None


def main() -> int:
    with open(args.prompt) as f:
        prompt = f.read()
    if args.thinking == "no_think":
        prompt = prompt.rstrip() + "\n/no_think"

    max_tokens = 1 if args.mode == "prefill" else args.max_tokens
    print(f"[bench] {args.model} mode={args.mode} prompt={os.path.basename(args.prompt)} "
          f"max_tokens={max_tokens} warmup={args.warmup} runs={args.runs}", file=sys.stderr)

    for i in range(args.warmup):
        try:
            one_request(prompt, max_tokens, args.timeout)
            print(f"[bench]   warmup {i+1}/{args.warmup} done", file=sys.stderr)
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"[bench] ERROR during warmup: {e}", file=sys.stderr)
            return 1

    samples = []
    for i in range(args.runs):
        try:
            r = one_request(prompt, max_tokens, args.timeout)
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"[bench] ERROR during run {i+1}: {e}", file=sys.stderr)
            return 1
        samples.append(r)
        primary = r["pp_tok_s"] if args.mode == "prefill" else r["tg_tok_s"]
        print(f"[bench]   run {i+1}/{args.runs}: "
              f"{'pp' if args.mode=='prefill' else 'tg'}={primary} tok/s "
              f"ttft={r['ttft_s']}s accept={r['accept_rate']}", file=sys.stderr)

    row = {
        "timestamp": int(time.time()),
        "model": args.model,
        "label": args.label,
        "quant": args.quant,
        "ctx": args.ctx,
        "parallel": args.parallel,
        "gpu_ratio": args.gpu,
        "flash": args.flash,
        "kv_quant": args.kv_quant,
        "threads": args.threads,
        "mtp": args.mtp,
        "mode": args.mode,
        "prompt_tokens": median([s["prompt_tokens"] for s in samples]),
        "completion_tokens": median([s["completion_tokens"] for s in samples]),
        "pp_tok_s": median([s["pp_tok_s"] for s in samples]),
        "tg_tok_s": median([s["tg_tok_s"] for s in samples]),
        "ttft_s": median([s["ttft_s"] for s in samples]),
        "accept_rate": median([s["accept_rate"] for s in samples]),
        "runs": args.runs,
    }

    write_header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a") as f:
        if write_header:
            f.write(",".join(CSV_COLUMNS) + "\n")
        f.write(",".join(str(row[c]) for c in CSV_COLUMNS) + "\n")

    metric = "pp_tok_s" if args.mode == "prefill" else "tg_tok_s"
    print(f"[bench] MEDIAN {metric} = {row[metric]} tok/s  (ttft {row['ttft_s']}s, "
          f"accept {row['accept_rate']})  -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--mode", choices=["decode", "prefill"], default="decode")
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-tokens", type=int, default=256, dest="max_tokens")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--thinking", choices=["allow", "no_think"], default="allow")
    p.add_argument("--out", default="results/results.csv")
    # Config metadata columns (what sweep.sh set; recorded for comparison).
    p.add_argument("--label", default="")
    p.add_argument("--quant", default="")
    p.add_argument("--ctx", default="")
    p.add_argument("--parallel", default="")
    p.add_argument("--gpu", default="")
    p.add_argument("--flash", default="")
    p.add_argument("--kv-quant", default="", dest="kv_quant")
    p.add_argument("--threads", default="")
    p.add_argument("--mtp", default="")
    args = p.parse_args()
    sys.exit(main())
