#!/usr/bin/env python3
"""Decode-speed benchmark for the Modal-hosted vLLM endpoint (about 8 requests total).

Usage:
  python tests/bench.py [BASE_URL]

BASE_URL is the bare Modal URL printed by `modal deploy`, e.g.
https://<workspace>--qwen-abliterated-serve.<region>.modal.run ("/v1" is appended).
Resolution order: CLI argument, then MODAL_BASE_URL in the environment, then ./.env.
The API key is read from VLLM_API_KEY in the environment or ./.env. It is never printed.

Measures, with stream_options include_usage for true token counts:
  (i)   1 stream, thinking OFF, repeated SINGLE_RUNS times -> median TTFT and decode tok/s
  (ii)  CONCURRENCY streams at once, thinking OFF -> per-stream and aggregate tok/s
  (iii) 1 stream, thinking ON -> TTFT, decode tok/s

Decode tok/s = (completion_tokens - 1) / (t_last_token - t_first_token).
TTFT = time from request start to the first streamed token (reasoning or content).
Aggregate tok/s = sum(completion_tokens) / (last token time - first request start).
Run 1 of (i) includes the cold start if the container was idle, so its TTFT is inflated;
the median over SINGLE_RUNS drops it.
"""

import statistics
import sys
import threading
import time
from pathlib import Path

from openai import OpenAI

MODEL = "qwen-abliterated"
ROOT = Path(__file__).resolve().parent.parent
FIRST_CALL_TIMEOUT = 900.0  # only the very first call may hit a cold start (image + weights + compile)
CALL_TIMEOUT = 300.0
PROMPT = "Write a detailed explanation of how a hash map works, at least 600 words."
MAX_TOKENS = 512
SINGLE_RUNS = 3
CONCURRENCY = 4
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}
THINKING_ON = {"chat_template_kwargs": {"enable_thinking": True}}


def load_dotenv(path):
    values = {}
    if not path.is_file():
        return values
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_config():
    env_file = load_dotenv(ROOT / ".env")
    base = (
        (sys.argv[1] if len(sys.argv) > 1 else "")
        or env_file.get("MODAL_BASE_URL", "")
    ).strip().rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    key = (env_file.get("VLLM_API_KEY", "") or "").strip()
    if not base:
        sys.exit("No base URL: pass it as an argument, or set MODAL_BASE_URL in .env")
    if not key:
        sys.exit("No API key: set VLLM_API_KEY in .env")
    return base, key


def _field(obj, name):
    """Read a delta field that the openai SDK may keep as an extra (reasoning vs reasoning_content)."""
    value = getattr(obj, name, None)
    if value is None and getattr(obj, "model_extra", None):
        value = obj.model_extra.get(name)
    return value


def stream_once(client, extra):
    t0 = time.time()
    t_first = None
    t_last = None
    usage = None
    finish = None
    chunks_with_tokens = 0
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=MAX_TOKENS,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
        extra_body=extra,
    )
    for chunk in stream:
        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage
        for choice in chunk.choices or []:
            if choice.finish_reason:
                finish = choice.finish_reason
            delta = choice.delta
            if delta is None:
                continue
            if _field(delta, "content") or _field(delta, "reasoning_content") or _field(delta, "reasoning"):
                now = time.time()
                if t_first is None:
                    t_first = now
                t_last = now
                chunks_with_tokens += 1
    t_end = t_last if t_last is not None else time.time()
    # Fall back to counting streamed chunks if the server sent no usage block.
    completion = usage.completion_tokens if usage is not None else chunks_with_tokens
    ttft = (t_first - t0) if t_first is not None else None
    decode = None
    if t_first is not None and t_last is not None and t_last > t_first and completion > 1:
        decode = (completion - 1) / (t_last - t_first)
    return {
        "start": t0,
        "end": t_end,
        "ttft": ttft,
        "decode": decode,
        "completion": completion,
        "finish": finish,
        "usage_ok": usage is not None,
    }


def fmt(value, spec=".2f"):
    return "-" if value is None else format(value, spec)


def main():
    base, key = get_config()
    print(f"Endpoint: {base}/v1  model={MODEL}  max_tokens={MAX_TOKENS}  temperature=0")
    first_client = OpenAI(base_url=f"{base}/v1", api_key=key, timeout=FIRST_CALL_TIMEOUT, max_retries=0)
    client = OpenAI(base_url=f"{base}/v1", api_key=key, timeout=CALL_TIMEOUT, max_retries=0)
    errors = []

    # (i) single stream, thinking OFF, repeated
    single = []
    for i in range(SINGLE_RUNS):
        try:
            r = stream_once(first_client if i == 0 else client, NO_THINKING)
            single.append(r)
            print(f"[single {i + 1}/{SINGLE_RUNS}] ttft={fmt(r['ttft'])}s decode={fmt(r['decode'], '.1f')} tok/s "
                  f"tokens={r['completion']} finish={r['finish']} usage={r['usage_ok']}")
        except Exception as exc:
            errors.append(f"single {i + 1}: {type(exc).__name__}: {str(exc)[:160]}")
            print(f"[single {i + 1}/{SINGLE_RUNS}] ERROR {errors[-1]}")

    # (ii) concurrent streams, thinking OFF
    conc_results = [None] * CONCURRENCY

    def worker(idx):
        try:
            conc_results[idx] = stream_once(client, NO_THINKING)
        except Exception as exc:
            conc_results[idx] = {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(CONCURRENCY)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    conc_ok = [r for r in conc_results if r and "error" not in r]
    for r in conc_results:
        if r and "error" in r:
            errors.append(f"concurrent: {r['error']}")
    print(f"[concurrent x{CONCURRENCY}] ok={len(conc_ok)}/{CONCURRENCY}")

    # (iii) single stream, thinking ON
    think = None
    try:
        think = stream_once(client, THINKING_ON)
        print(f"[thinking on] ttft={fmt(think['ttft'])}s decode={fmt(think['decode'], '.1f')} tok/s "
              f"tokens={think['completion']} finish={think['finish']}")
    except Exception as exc:
        errors.append(f"thinking: {type(exc).__name__}: {str(exc)[:160]}")
        print(f"[thinking on] ERROR {errors[-1]}")

    # Table
    print("\n| Test | Median TTFT (s) | Median decode tok/s | Per-run / detail |")
    print("|---|---|---|---|")
    singles_ok = [r for r in single if r["decode"] is not None]
    if singles_ok:
        ttfts = [r["ttft"] for r in single if r["ttft"] is not None]
        decs = [r["decode"] for r in singles_ok]
        detail = ", ".join(f"{r['decode']:.1f}" for r in singles_ok) + f" (tokens {single[0]['completion']})"
        print(f"| (i) 1 stream, thinking off, n={len(singles_ok)} | {fmt(statistics.median(ttfts))} | "
              f"{fmt(statistics.median(decs), '.1f')} | {detail} |")
    else:
        print("| (i) 1 stream, thinking off | - | - | no successful runs |")
    if conc_ok:
        wall = max(r["end"] for r in conc_ok) - min(r["start"] for r in conc_ok)
        total_tokens = sum(r["completion"] for r in conc_ok)
        per_stream = [r["decode"] for r in conc_ok if r["decode"] is not None]
        conc_ttfts = [r["ttft"] for r in conc_ok if r["ttft"] is not None]
        per_detail = (f"per-stream decode median {fmt(statistics.median(per_stream), '.1f')} "
                      f"(min {fmt(min(per_stream), '.1f')}, max {fmt(max(per_stream), '.1f')}); "
                      f"aggregate {fmt(total_tokens / wall, '.1f')} tok/s over {wall:.1f}s")
        print(f"| (ii) {len(conc_ok)} concurrent streams, thinking off | "
              f"{fmt(statistics.median(conc_ttfts)) if conc_ttfts else '-'} | "
              f"{fmt(statistics.median(per_stream), '.1f') if per_stream else '-'} | {per_detail} |")
    else:
        print(f"| (ii) {CONCURRENCY} concurrent streams | - | - | no successful streams |")
    if think is not None:
        print(f"| (iii) 1 stream, thinking on | {fmt(think['ttft'])} | {fmt(think['decode'], '.1f')} | "
              f"tokens={think['completion']} finish={think['finish']} |")
    else:
        print("| (iii) 1 stream, thinking on | - | - | failed |")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"- {e}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
