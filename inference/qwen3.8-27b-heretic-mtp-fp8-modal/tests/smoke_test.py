#!/usr/bin/env python3
"""Smoke test for the Modal-hosted vLLM endpoint (4 requests total).

Usage:
  python tests/smoke_test.py [BASE_URL]

BASE_URL is the bare Modal URL printed by `modal deploy`, e.g.
https://<workspace>--qwen-abliterated-serve.<region>.modal.run ("/v1" is appended).
Resolution order: CLI argument, then MODAL_BASE_URL in the environment, then ./.env.
The API key is read from VLLM_API_KEY in the environment or ./.env. It is never printed.
"""

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

MODEL = "qwen-abliterated"
ROOT = Path(__file__).resolve().parent.parent
FIRST_CALL_TIMEOUT = 900.0  # cold start (image + 30GB weights + compile) can take ~10 min
CALL_TIMEOUT = 180.0
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    }
]

results = []  # (check, passed, latency_s, tokens_per_s, detail)


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
        or os.environ.get("MODAL_BASE_URL", "")
        or env_file.get("MODAL_BASE_URL", "")
    ).strip().rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    key = (os.environ.get("VLLM_API_KEY") or env_file.get("VLLM_API_KEY", "")).strip()
    if not base:
        sys.exit("No base URL: pass it as an argument, or set MODAL_BASE_URL in .env")
    if not key:
        sys.exit("No API key: set VLLM_API_KEY in .env or the environment")
    return base, key


def record(check, passed, latency, tok_s, detail):
    results.append((check, passed, latency, tok_s, detail))
    status = "PASS" if passed else "FAIL"
    lat = f"{latency:.2f}s" if latency is not None else "-"
    rate = f" {tok_s:.1f} tok/s" if tok_s is not None else ""
    print(f"[{status}] {check} ({lat}{rate}): {detail}")


def run_check(check, fn):
    try:
        fn()
    except Exception as exc:  # report and continue to the next check
        record(check, False, None, None, f"{type(exc).__name__}: {str(exc)[:200]}")


def main():
    base_url, api_key = get_config()
    print(f"Endpoint: {base_url}/v1  model={MODEL}")

    # Only the first call gets the long timeout (cold start). No retries anywhere.
    first_client = OpenAI(base_url=f"{base_url}/v1", api_key=api_key, timeout=FIRST_CALL_TIMEOUT, max_retries=0)
    client = OpenAI(base_url=f"{base_url}/v1", api_key=api_key, timeout=CALL_TIMEOUT, max_retries=0)

    def check_models():
        t0 = time.time()
        ids = [m.id for m in first_client.models.list().data]
        dt = time.time() - t0
        record("(a) GET /v1/models", MODEL in ids, dt, None, f"ids={ids}")

    def check_chat():
        t0 = time.time()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Reply with exactly one word: pong"}],
            max_tokens=32,
            temperature=0,
            extra_body=NO_THINKING,
        )
        dt = time.time() - t0
        text = (resp.choices[0].message.content or "").strip()
        tok_s = resp.usage.completion_tokens / dt if dt > 0 else None
        record("(b) short chat, thinking off", bool(text), dt, tok_s, f"reply={text[:60]!r}")

    def check_tool():
        t0 = time.time()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "What is the weather in Paris? Use the get_weather tool."}],
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=256,
            temperature=0,
            extra_body=NO_THINKING,
        )
        dt = time.time() - t0
        calls = resp.choices[0].message.tool_calls or []
        if not calls:
            record("(c) tool call", False, dt, None, f"no tool_calls; content={(resp.choices[0].message.content or '')[:80]!r}")
            return
        args = calls[0].function.arguments
        parsed = json.loads(args)  # raises if not valid JSON -> FAIL via run_check
        tok_s = resp.usage.completion_tokens / dt if dt > 0 else None
        record("(c) tool call", isinstance(parsed, dict), dt, tok_s, f"name={calls[0].function.name} args={args[:80]}")

    def check_long_prompt():
        filler = "The quick brown fox jumps over the lazy dog. " * 850  # ~8.5k tokens
        t0 = time.time()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": filler + "\n\nIgnore the text above. Reply with the single word DONE."}],
            max_tokens=64,
            temperature=0,
            extra_body=NO_THINKING,
        )
        dt = time.time() - t0
        prompt_tokens = resp.usage.prompt_tokens
        text = (resp.choices[0].message.content or "").strip()
        tok_s = resp.usage.completion_tokens / dt if dt > 0 else None
        passed = prompt_tokens >= 7500 and bool(text)
        record("(d) ~8k-token prompt", passed, dt, tok_s, f"prompt_tokens={prompt_tokens} reply={text[:40]!r}")

    run_check("(a) GET /v1/models", check_models)
    run_check("(b) short chat, thinking off", check_chat)
    run_check("(c) tool call", check_tool)
    run_check("(d) ~8k-token prompt", check_long_prompt)

    print("\n| Check | Result | Latency | Completion tok/s | Detail |")
    print("|---|---|---|---|---|")
    for check, passed, latency, tok_s, detail in results:
        lat = f"{latency:.2f}s" if latency is not None else "-"
        rate = f"{tok_s:.1f}" if tok_s is not None else "-"
        print(f"| {check} | {'PASS' if passed else 'FAIL'} | {lat} | {rate} | {detail} |")

    failed = [r for r in results if not r[1]]
    total = len(results)
    print(f"\n{total - len(failed)}/{total} checks passed")
    sys.exit(1 if failed or total < 4 else 0)


if __name__ == "__main__":
    main()
