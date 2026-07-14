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
import concurrent.futures as cf
import json
import os
import statistics
import sys
import time
import urllib.error
import uuid
import urllib.request

import machine

HOST = os.environ.get("LMS_HOST", "http://localhost:1234")
ENDPOINT = HOST + "/api/v0/chat/completions"

# One unified schema. Concurrency is first-class: every row records how many
# streams ran (1 = single-stream), the per-stream decode rate (tg_tok_s), and the
# aggregate decode rate across all streams (tg_tok_s_agg = tg_tok_s x concurrency).
# For a single stream the two decode columns are equal. Both are decode tok/s, so
# single-stream and concurrent rows are directly comparable.
CSV_COLUMNS = [
    "timestamp", "machine_id", "model", "label", "quant", "ctx", "parallel",
    "concurrency", "gpu_ratio", "flash", "kv_quant", "threads", "mtp", "mode",
    "prompt_tokens", "completion_tokens", "pp_tok_s", "tg_tok_s", "tg_tok_s_agg",
    "ttft_s", "accept_rate", "runs",
]


def post(payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        ENDPOINT, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def one_request(prompt: str, max_tokens: int, timeout: float) -> dict:
    # Prefill measures prompt processing, so every request must miss the server's
    # prompt cache. With parallel=1 all requests share one slot and an identical
    # prompt gets ttft~0 from the cache; a unique prefix defeats that.
    if args.mode == "prefill":
        prompt = f"[req {uuid.uuid4().hex[:12]}]\n{prompt}"
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
    # Some model/server combos (e.g. gpt-oss) report ttft=0 on non-streaming
    # requests and carry the prefill time in generation_time instead. For a
    # 1-token prefill probe, generation_time IS the prompt-processing time.
    if not ttft and args.mode == "prefill":
        ttft = stats.get("generation_time")
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


def run_batch(prompt: str, max_tokens: int, timeout: float, concurrency: int) -> list:
    """Fire `concurrency` requests at once; return the per-stream result dicts."""
    if concurrency <= 1:
        return [one_request(prompt, max_tokens, timeout)]
    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(one_request, prompt, max_tokens, timeout)
                for _ in range(concurrency)]
        return [f.result() for f in futs]


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def median(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.median(vals), 3) if vals else None


def main() -> int:
    with open(args.prompt) as f:
        prompt = f.read()
    if args.thinking == "no_think":
        prompt = prompt.rstrip() + "\n/no_think"

    max_tokens = 1 if args.mode == "prefill" else args.max_tokens
    conc = 1 if args.mode == "prefill" else max(1, args.concurrency)
    print(f"[bench] {args.model} mode={args.mode} concurrency={conc} "
          f"prompt={os.path.basename(args.prompt)} max_tokens={max_tokens} "
          f"warmup={args.warmup} runs={args.runs}", file=sys.stderr)

    for i in range(args.warmup):
        try:
            run_batch(prompt, max_tokens, args.timeout, conc)
            print(f"[bench]   warmup {i+1}/{args.warmup} done", file=sys.stderr)
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"[bench] ERROR during warmup: {e}", file=sys.stderr)
            return 1

    # Each run fires `conc` concurrent streams; we track the per-stream decode
    # rate (mean across streams) and the aggregate (per-stream x conc).
    per_run, agg_run, ttft_run, acc_run, ptok_run, ctok_run, pp_run = ([] for _ in range(7))
    for i in range(args.runs):
        try:
            streams = run_batch(prompt, max_tokens, args.timeout, conc)
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"[bench] ERROR during run {i+1}: {e}", file=sys.stderr)
            return 1
        per = mean([s["tg_tok_s"] for s in streams])
        agg = (per * conc) if per is not None else None
        per_run.append(per); agg_run.append(agg)
        ttft_run.append(mean([s["ttft_s"] for s in streams]))
        acc_run.append(mean([s["accept_rate"] for s in streams]))
        ptok_run.append(mean([s["prompt_tokens"] for s in streams]))
        ctok_run.append(mean([s["completion_tokens"] for s in streams]))
        pp_run.append(mean([s["pp_tok_s"] for s in streams]))
        if args.mode == "prefill":
            print(f"[bench]   run {i+1}/{args.runs}: pp={pp_run[-1]} tok/s "
                  f"ttft={ttft_run[-1]}s", file=sys.stderr)
        else:
            print(f"[bench]   run {i+1}/{args.runs}: {conc} stream(s) "
                  f"per-stream={per} agg={agg} tok/s accept={acc_run[-1]}", file=sys.stderr)

    row = {
        "timestamp": int(time.time()),
        "machine_id": args.machine_id or machine.ensure_registered(),
        "model": args.model_name or args.model,
        "label": args.label,
        "quant": args.quant,
        "ctx": args.ctx,
        "parallel": args.parallel,
        "concurrency": conc,
        "gpu_ratio": args.gpu,
        "flash": args.flash,
        "kv_quant": args.kv_quant,
        "threads": args.threads,
        "mtp": args.mtp,
        "mode": args.mode,
        "prompt_tokens": median(ptok_run),
        "completion_tokens": median(ctok_run),
        "pp_tok_s": median(pp_run),
        "tg_tok_s": median(per_run),
        "tg_tok_s_agg": median(agg_run),
        "ttft_s": median(ttft_run),
        "accept_rate": median(acc_run),
        "runs": args.runs,
    }

    write_header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a") as f:
        if write_header:
            f.write(",".join(CSV_COLUMNS) + "\n")
        f.write(",".join(str(row[c]) for c in CSV_COLUMNS) + "\n")

    if args.mode == "prefill":
        print(f"[bench] MEDIAN pp_tok_s = {row['pp_tok_s']} tok/s  (ttft {row['ttft_s']}s)"
              f"  -> {args.out}", file=sys.stderr)
    else:
        print(f"[bench] MEDIAN decode = {row['tg_tok_s']} tok/s per stream, "
              f"{row['tg_tok_s_agg']} tok/s aggregate across {conc} stream(s)  "
              f"(accept {row['accept_rate']})  -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="API model id to send requests to (the loaded identifier)")
    p.add_argument("--model-name", default=None, dest="model_name",
                   help="model name recorded in the CSV (default: --model). Use when "
                        "loading under a stable identifier so rows show the real model.")
    p.add_argument("--mode", choices=["decode", "prefill"], default="decode")
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-tokens", type=int, default=256, dest="max_tokens")
    p.add_argument("--concurrency", type=int, default=1,
                   help="simultaneous streams per run (1=single-stream; decode only). "
                        "The loaded model must have --parallel >= this.")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--thinking", choices=["allow", "no_think"], default="allow")
    p.add_argument("--out", default="results/results.csv")
    p.add_argument("--machine-id", default=None, dest="machine_id",
                   help="override machine id (default: auto-detect via machine.py)")
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
