# Qwen3.8-27B heretic MTP FP8 on Modal (vLLM)

Serves `OptimizeLLM/Qwen3.8-27B-heretic-MTP-FP8` (FP8, about 31 GB on the Hub) with vLLM on one Modal GPU. The default GPU is an L40S, and an H100 option is documented. The endpoint is OpenAI-compatible, requires an API key, and scales to zero when idle.

Failure notes and measured numbers are in [FAILURES.md](FAILURES.md).

## Files

- `modal_app.py`: the vLLM server (Modal app `qwen-abliterated`). GPU, context, concurrency, MTP, timeouts and cache paths come from env vars (see the table below).
- `download.py`: CPU-only Modal job that downloads the weights into the `qwen-weights` Volume.
- `deploy.sh`: the canonical deploy line for the long-task config. Deploy through it so the env and the image stay identical.
- `tests/smoke_test.py`: four checks against the live endpoint.
- `tests/bench.py`: decode tok/s benchmark (one stream, four concurrent streams, thinking on).
- `scripts/watchdog.py` and `wd`: read-only cost and safety checks (stdlib only).
- `.env` (not committed): `VLLM_API_KEY` and `MODAL_BASE_URL` for the test scripts.

## Prerequisites

- A Modal account and CLI (`pip install modal`, then `modal token new`).
- The model repo is public and not gated on Hugging Face.

## Setup

Create the Modal Secret that holds the API key. Generate the key yourself, for example with `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`:

```bash
modal secret create qwen-api-key VLLM_API_KEY=<key> --force
```

For the test scripts, put the same key and the endpoint URL in a local `.env` next to `modal_app.py`:

```
VLLM_API_KEY=<key>
MODAL_BASE_URL=https://<workspace>--qwen-abliterated-serve.<region>.modal.run
```

## Commands

```bash
# 1. Download the weights into the Volume (CPU only, about 31 GB, one time)
modal run download.py

# 2. Deploy the default server (L40S, 32k context, 4 seqs, no MTP). No GPU runs until a request arrives.
modal deploy modal_app.py
#    Copy the printed https://...modal.run URL into MODAL_BASE_URL in .env

#    Long-task config (98k context, 2 seqs, 3 MTP drafts, 3 h container cap). Always use the script,
#    so the env and image stay identical:
./deploy.sh
#    Then run the watchdog with the matching threshold: WD_MAX_AGE_MIN=185 ./wd --watch 300

#    H100 + MTP (env vars are read on the deploy machine and baked into the image):
QWEN_GPU='H100!' QWEN_MTP=1 QWEN_MAX_LEN=65536 QWEN_MAX_SEQS=32 QWEN_SCALEDOWN=60 QWEN_STARTUP=1500 modal deploy modal_app.py
#    "H100!" pins H100 (no automatic H200 upgrade). MAX_SEQS=32 is the value the model card uses.
#    QWEN_STARTUP=1500 covers the first H100 boot (DeepGEMM JIT warmup, about 10 min). Keep it on later deploys too.
#    Deploy with the same values each time: the container reads them from the image, not from your shell.

# 3. Smoke test (needs the openai SDK locally)
python3 -m venv .venv && . .venv/bin/activate && pip install openai
python tests/smoke_test.py            # or: python tests/smoke_test.py <URL>

# 3b. Decode-speed bench (about 8 requests: thinking off x3, 4 concurrent, thinking on)
python tests/bench.py                 # or: python tests/bench.py <URL>

# 4. Stop the app (stops the GPU; the Volumes with the weights and caches remain)
modal app stop qwen-abliterated
```

## Configuration

Values are read when `modal_app.py` is imported. Set them on the deploy command line, and the deploy-time values are baked into the image (`CONTAINER_ENV`) so the container sees the same config.

| Variable | Default | Meaning |
|---|---|---|
| `QWEN_GPU` | `L40S` | GPU type. `H100!` pins H100 and blocks the automatic H200 upgrade. |
| `QWEN_MAX_LEN` | `32768` | `--max-model-len`. Set the client context window to this value. |
| `QWEN_MAX_SEQS` | `4` | `--max-num-seqs`. |
| `QWEN_MTP` | `0` | `1` adds MTP speculative decoding. |
| `QWEN_SPEC_TOKENS` | `4` | `num_speculative_tokens` for MTP. The model card value is 4. Use 3 for agent traffic with low draft acceptance. |
| `QWEN_SCALEDOWN` | `300` | `scaledown_window` in seconds (idle time before a container stops). |
| `QWEN_TIMEOUT` | `1800` | Function timeout in seconds: the hard cap per container (1 to 86400). |
| `QWEN_STARTUP` | `600` | Web server `startup_timeout`. The readiness deadline is this value minus 30. |
| `QWEN_DG_WARMUP` | empty | `VLLM_DEEP_GEMM_WARMUP`. Empty uses the vLLM default. `skip` is an H100 fallback only. |
| `QWEN_GPU_UTIL` | `0.90` | `--gpu-memory-utilization`. Read the KV math below before raising it. |
| `QWEN_KV_DTYPE` | auto | `--kv-cache-dtype`, passed only when set. Unverified for this model. |
| `QWEN_DEFAULT_THINK` | `1` | `1` keeps thinking on by default. `0` makes `enable_thinking=false` the server default. |

Flags that `modal_app.py` always passes to `vllm serve`: `--served-model-name qwen-abliterated`, `--enable-auto-tool-choice`, `--tool-call-parser qwen3_coder`, `--reasoning-parser qwen3`, `--reasoning-config {}`, `--enable-prefix-caching`, `--language-model-only`. vLLM is pinned to `v0.33.0`.

## Using it from an OpenAI-compatible client

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<workspace>--qwen-abliterated-serve.<region>.modal.run/v1",
    api_key="<VLLM_API_KEY>",
)
resp = client.chat.completions.create(
    model="qwen-abliterated",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},  # omit to keep thinking on
)
print(resp.choices[0].message.content)
```

- Thinking is on by default (the chat template defaults to a high reasoning effort). Set `enable_thinking: False` per request for latency-sensitive calls, or deploy with `QWEN_DEFAULT_THINK=0`. A request with a non-null `reasoning_effort` (other than `none`) turns thinking on even when the default is off (vLLM `build_chat_params`).
- Set the client context window to `QWEN_MAX_LEN` (98304 or 65536 in the configs below), not the model's 262K. Otherwise the client under-counts, and requests return 400 past the server limit.
- Tool calling is enabled (`qwen3_coder` parser). Pass `tools=` as usual.
- Image input is disabled (`--language-model-only`).
- Only `/v1` routes check the API key. Send `Authorization: Bearer <key>`.

## Operational notes

- Cold start loads the weights from the Volume and runs torch compile. The first request can wait several minutes.
- Compile and JIT caches live on the `qwen-vllm-cache` Volume (mounted at `/cache`): vLLM torch.compile (`VLLM_CACHE_ROOT`), DeepGEMM kernels (`DG_JIT_CACHE_DIR`), and Triton (`TRITON_CACHE_DIR`). The container commits this Volume once the server is healthy.
- DeepGEMM runs only on Hopper and Blackwell (sm90 and sm100). L40S (Ada, sm89) skips it, so an L40S cold start has no DeepGEMM warmup. On H100 the warmup is the long step on a first boot.
- The startup log has one `config:` line with the GPU, `max_len`, seqs, MTP, `spec_tokens`, timeout, budget and warmup mode. Check it to confirm which config ran (`modal container logs <container-id> --all --timestamps`).
- Each container stops after `QWEN_TIMEOUT` s, and an idle one stops after `QWEN_SCALEDOWN` s. The next request then cold-starts a new one. The function timeout bounds each container's runtime, which bounds its spend.
- `QWEN_SPEC_TOKENS`: on real agent traffic, MTP draft acceptance measured 24 to 35% at 4 drafts, which is why 3 is the long-task setting. Check the `SpecDecoding metrics` lines in the logs before keeping it.
- Long-context candidate (commented in `deploy.sh`): `QWEN_MAX_LEN=98304 QWEN_MAX_SEQS=1 QWEN_DEFAULT_THINK=0`. KV math (vLLM v0.33.0 `kv_cache_planning.py` checks that one full-length request fits): at util 0.90 the log reports about 109k KV tokens in 8.45 GiB. Full-attention KV from the config (16 layers plus 1 MTP layer, 4 KV heads, head_dim 256, bf16) is 69,632 bytes per token, so 98,304 tokens need about 6.4 GiB, plus about 0.15 GiB of GDN state per sequence. That fits with margin. 131072 would need `QWEN_GPU_UTIL=0.94` (about 132k tokens, under 1% margin), which is not recommended.
- `QWEN_GPU_UTIL`: each +0.01 adds about 0.44 GiB of KV on the L40S (about 5.7k tokens).
- `QWEN_KV_DTYPE`: fp8 KV is unverified for this model. In v0.33.0 the FlashAttention backend accepts fp8 KV only on SM90 (FA3) or SM100 (FA4). On L40S (sm89) the selector falls back to another backend. The checkpoint has no `kv_cache_scheme`, so k/v scales default to 1.0. Do not use it without a run on the endpoint and a quality check.
- To change the model or vLLM version, edit `VLLM_TAG` or `MODEL_DIR` in `modal_app.py`, and keep `MODEL_DIR` in `download.py` in sync.

## Watchdog

`./wd` (wrapper for `scripts/watchdog.py`, stdlib only) checks the deployment for cost and safety problems. It runs only read-only `modal ... list --json` commands and reads local files. It never calls the served endpoint.

```bash
./wd                  # one pass: apps, containers, spend, resources, secrets, config
./wd --json           # same checks, machine-readable
./wd --fix --dry-run  # list what --fix would stop (stops nothing)
./wd --fix            # stop flagged apps/containers, one y/N prompt each (needs a terminal)
./wd --watch 300      # re-check every 300 s (min 60): prints changes only, exits 2 on first ALERT
./wd --notify         # also run notify-send on ALERT, if installed
WD_MAX_AGE_MIN=185 ./wd --watch 300   # long-task config (QWEN_TIMEOUT=10800): runaway threshold 185 min
```

- `WD_MAX_AGE_MIN` (default 35) is the runaway threshold in minutes. Set it to `QWEN_TIMEOUT/60 + 5`.
- Each live container gets an `est. cost so far` line (uptime x the GPU rate, GPU only).
- The spend check is optional. If a `COST.md` file sits next to `modal_app.py` and has a `Running total` column, `./wd` reads its last value. Without that file the check is skipped with a message. Set `WD_BUDGET` (dollars) to get WARN at 40% and ALERT at 75% of it.
- Exit codes: 0 all OK, 1 WARN, 2 ALERT, 64 usage error. `--fix` only stops unexpected apps, containers from other apps, containers older than `WD_MAX_AGE_MIN`, and extra containers. It never stops the `qwen-abliterated` app itself.
- Limits: the CLI cannot see per-request cost or your real bill. Check the Modal dashboard for the true total, and set a workspace spending limit there yourself, since the watchdog cannot set one.

## Pricing used for estimates

Public Modal list prices (see https://modal.com/pricing). GPU only, CPU and memory are billed on top:

- L40S: $0.000542 per second (about $1.95 per hour while a container runs).
- H100 SXM5: $0.001097 per second (about $3.95 per hour). An unpinned `H100` request may land on H200 at a different rate, so pin with `QWEN_GPU='H100!'`.
