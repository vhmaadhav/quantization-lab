# Failure notes: Qwen3.8-27B heretic MTP FP8 on Modal

What went wrong while setting up and tuning this vLLM endpoint, how it was fixed, and what to check first next time. Env var and flag names refer to `modal_app.py`. No prompts, outputs or account-specific figures are recorded here. Prices are the public Modal list rates from the README.

## Index

| # | Incident | Lesson |
|---|---|---|
| 1 | Deploy died mid image build | Check app state before redoing a deploy |
| 2 | Deploy-time env vars missing in the container | Bake config into the image and log it at startup |
| 3 | H100 DeepGEMM warmup exceeded the readiness deadline | Persist caches; size the startup budget for the slowest first boot |
| 4 | `hf_transfer` extra removed | Re-check extras names when a library major version changes |
| 5 | `modal app logs` shows only the last 100 lines | Read full container logs with `--all` |
| 6 | Request wedged at the proxy while vLLM was idle | Health 200 is not proof; use timeouts and stop the container |
| 7 | A container ran about 71 minutes under active use | Cap runtime with the function timeout; watch container age |
| 8 | Context overflow and HTTP 400s | Client context must match the server limit; KV math decides it |
| 9 | Thinking mode used up the output budget | Set thinking and max output per request |
| 10 | Cold start and MTP tuning | Persist the compile cache; tune draft count on real traffic |
| 11 | FP8 on L40S is weight-only | Expect slower prefill than on H100 |
| 12 | Stopping versus deleting | Stop to save money; never delete without an explicit go-ahead |

## Incidents

### 1. Deploy killed during the image build

- **Symptom:** a deploy attempt stopped partway through the image build when the working session was killed.
- **Cause:** the local client process died. It was not clear from the terminal whether the app had been created or what state it was in.
- **Fix:** check the app and container state before running the deploy again (`modal app list`, then the container list). Then redeploy. Image layers that finished are cached, so the rerun is quick.
- **Lesson:** before repeating a long-running command, check what it left behind.

### 2. Deploy-time env vars did not reach the container

- **Symptom:** a benchmark ran with a different config from the one that was deployed (wrong context, MTP state or GPU).
- **Cause:** Modal evaluates `modal_app.py` on the local machine at deploy time, but the container re-imports the module without the shell's `QWEN_*` variables. The `os.environ.get` calls inside the container fell back to their defaults.
- **Fix:** the deploy-time values are baked into the image with `.env(CONTAINER_ENV)`. The server also prints one `config:` line at startup with the effective values, and that line is the first thing to check.
- **Lesson:** config read at import time must live in the image, not only in the deploying shell. Confirm the `config:` line before trusting a benchmark.

### 3. H100 DeepGEMM warmup exceeded the readiness deadline

- **Symptom:** on the first H100 boot the server was not ready in time. The readiness deadline is `QWEN_STARTUP - 30` (570 s at the default 600 s), and the web server `startup_timeout` is 600 s.
- **Cause:** on Hopper, vLLM runs a DeepGEMM kernel JIT warmup of about 1,500 items. On an empty cache it takes 9 to 10 minutes on top of the normal startup, which is more than the default budget.
- **Fix:**
  - Persist the compile and JIT caches on the `qwen-vllm-cache` Volume (`VLLM_CACHE_ROOT`, `DG_JIT_CACHE_DIR`, `TRITON_CACHE_DIR`), and commit the Volume once the server is healthy.
  - Use `QWEN_STARTUP=1500` for H100 deploys. It also covers the first boot.
  - `QWEN_DG_WARMUP=skip` exists only as an H100 fallback.
- **Note:** L40S (sm89) never runs DeepGEMM, so its cold start has no warmup step.
- **Lesson:** size the startup budget for the slowest first boot, and keep anything expensive to rebuild on a Volume.

### 4. `hf_transfer` extra removed

- **Symptom:** the download image pinned `huggingface_hub[hf_transfer]`, but that extra no longer exists in huggingface_hub 2.x.
- **Cause:** huggingface_hub 2.x dropped the `hf_transfer` extra. `hf_xet` is the accelerated backend now.
- **Fix:** pin `huggingface_hub[hf_xet]==2.4.0` in `download.py`.
- **Lesson:** when a library changes major version, check extra names and backends, not only the version number.

### 5. `modal app logs` shows only the last 100 lines

- **Symptom:** the logs looked stale, and startup lines were missing.
- **Cause:** `modal app logs` returns only the most recent 100 lines.
- **Fix:** read the full log for one container with `modal container logs <container-id> --all --timestamps`. Find the container ID with `modal container list`.
- **Lesson:** for any startup investigation, read the full container log.

### 6. Request wedged at the proxy while vLLM was idle

- **Symptom:** one request stayed "Running" for a long time. The `/health` endpoint returned 200, and vLLM logged no engine activity.
- **Cause:** not reproduced. The request appears to have stalled at Modal's proxy while the engine was idle.
- **Fix:** stop the stuck container with `modal container stop <container-id> --yes`. The next request cold-starts a new one. The test scripts use explicit timeouts with `max_retries=0`, so a stall is reported instead of retried silently.
- **Note:** a call still marked "Running" keeps its container alive and billing. The watchdog flags containers older than `WD_MAX_AGE_MIN`, and `QWEN_TIMEOUT` caps each container regardless.
- **Lesson:** a 200 from `/health` does not prove the request path works. Set client timeouts and watch container age.

### 7. A container stayed up about 71 minutes under active use

- **Symptom:** one container ran about 71 minutes, far longer than the idle window suggested.
- **Cause:** continuous requests kept the container busy. The scale-down window starts only after the last request, so a steady stream keeps the container up. Estimates made from memory drifted from the real uptime.
- **Fix:** use the function timeout (`QWEN_TIMEOUT`) as the hard cap, set the watchdog threshold to match it, and estimate cost from measured container uptime (`est. cost so far` in `./wd`).
- **Lesson:** treat the timeout as the real ceiling, and estimate from measured uptime rather than from memory.

### 8. Context overflow and HTTP 400s

- **Symptom:** long sessions began getting HTTP 400 responses once the prompt plus output passed the server limit.
- **Cause:** the server ran with `--max-model-len 32768`, while the client assumed the model's 262K window. The client never compacted its context, so requests went past the server limit.
- **Fix:**
  - Raise the server context in steps: 32768, then 65536, then 98304 (`QWEN_MAX_LEN`).
  - Set the client context window to the same value as `QWEN_MAX_LEN`.
  - KV math on the 48 GB L40S: at `QWEN_GPU_UTIL=0.90` the log reports about 109k KV tokens (8.45 GiB), and usage logs showed about 111k. One full 98,304-token request fits. Two full-length requests do not, so vLLM queues or preempts them. This is why the long-task config runs `QWEN_MAX_SEQS=2`.
  - fp8 KV cache was not used. On sm89 the FlashAttention backend does not accept fp8 KV (FA3 needs SM90, FA4 needs SM100), so vLLM falls back to another backend, and the checkpoint has no KV scales. Left unset.
- **Lesson:** match the client's context window to the server, and do the KV arithmetic before raising the context.

### 9. Thinking mode used up the output budget

- **Symptom:** replies ended with an "output token limit" condition, and clients looped on retries.
- **Cause:** thinking is on by default (the chat template defaults to a high reasoning effort). The reasoning used up the max output tokens before the answer was written.
- **Fix, in order of preference:** raise the per-request max output tokens; compact the context; turn thinking off per request with `chat_template_kwargs: {"enable_thinking": false}`; or deploy with `QWEN_DEFAULT_THINK=0` to make thinking off the server default. Note that a non-null `reasoning_effort` (other than `none`) turns thinking on again.
- **Lesson:** thinking changes both latency and token use, so set it per request and budget output tokens with it.

### 10. Cold starts and MTP tuning

- **Symptom:** the first cold starts took about 285 to 290 s to become ready.
- **Measured after the fix:** 78 to 111 s to first token once the compile cache was persisted on the `qwen-vllm-cache` Volume.
- **MTP limits found:** with this hybrid model, MTP worked with `--max-num-seqs` at or below 32. The H100 run used 32, the value the model card gives.
- **Draft count:** the model card recommends `num_speculative_tokens` 4. Measured acceptance on real agent traffic was 24 to 35% at 4 drafts. Acceptance drops further on code-heavy and agent-style traffic, so the long-task config uses 3. Check the `SpecDecoding metrics` lines before keeping that value.
- **Lesson:** persist the compile cache early, and tune draft tokens on the traffic you actually serve, not on a short benchmark.

### 11. FP8 on L40S uses the weight-only path

- **Symptom:** prefill ran slower than expected.
- **Cause:** L40S (Ada, sm89) has no native FP8 compute path for this model. vLLM uses the Marlin weight-only FP8 path, which dequantizes the weights and computes in 16-bit.
- **Lesson:** expect slower prefill on L40S than on H100, and compare configurations on the same hardware.

### 12. Stopping versus deleting

- **Stop** (`modal app stop qwen-abliterated`): halts GPU billing. The Volumes with the weights and caches are kept.
- **Delete** (Volumes or secrets): irreversible. A deleted weights Volume means a fresh download of about 31 GB, and a deleted cache Volume means a fresh compile.
- **Rules we use:** run only stop commands during routine cost control. Require an explicit go-ahead before any delete. Check app state before redeploying. Set a spending limit in the Modal dashboard as well.

## Benchmark (L40S, MTP, 32k context, 4 seqs, 4 draft tokens)

`tests/bench.py`: 512 max tokens, temperature 0.

| Measure | Result |
|---|---|
| Single stream, thinking off | about 53 tok/s (median of 3; TTFT about 1.5 s when warm) |
| 4 concurrent streams, thinking off | about 45 tok/s per stream, about 162 tok/s aggregate |
| Single stream, thinking on | about 50 tok/s |
| Real agent traffic (estimate) | about 27 to 37 tok/s, with lower MTP acceptance |
| Cold start, first L40S boots | about 285 to 290 s |
| Cold start, compile cache on Volume | about 78 to 111 s |

## Recommended config (long-task)

`deploy.sh` runs:

```
QWEN_MTP=1 QWEN_STARTUP=600 QWEN_MAX_LEN=98304 QWEN_MAX_SEQS=2 QWEN_SPEC_TOKENS=3 QWEN_TIMEOUT=10800 QWEN_SCALEDOWN=300 modal deploy modal_app.py
```

- Client context window: 98304.
- Thinking stays on by default (`QWEN_DEFAULT_THINK=1`). Set `enable_thinking: false` per request for latency-sensitive calls.
- Watchdog: `WD_MAX_AGE_MIN=185 ./wd --watch 300`, which matches the 3 h timeout.
- Short-context fallback (32k, 4 seqs, 4 drafts) is kept as a commented rollback line in `deploy.sh`.
- Long-context candidate for thinking-off traffic (98k, 1 seq, `QWEN_DEFAULT_THINK=0`) is commented in `deploy.sh` and is not the default.

## What we would do differently

- Bake the deploy-time config into the image and print it at startup from the first deploy (incident 2).
- Put the compile and JIT caches on a Volume from the first boot, and size the startup budget for the H100 first boot separately (incidents 3 and 10).
- Read full container logs with `--all` as the first diagnostic (incident 5).
- Set the client context window to the server limit before the first long session, and do the KV arithmetic first (incident 8).
- Set client timeouts and the watchdog age threshold before first use, not after the first stuck request (incidents 6 and 7).
- Benchmark thinking on and off separately, and tune draft tokens on real traffic (incidents 9 and 10).
- Stop rather than delete, and require an explicit go-ahead for any destructive command (incident 12).
