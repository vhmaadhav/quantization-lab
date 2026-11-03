"""Serve OptimizeLLM/Qwen3.8-27B-heretic-MTP-FP8 with vLLM on one Modal GPU.

Prerequisites:
  - weights downloaded into the "qwen-weights" Volume (run download.py first)
  - Modal Secret "qwen-api-key" holding VLLM_API_KEY

Config is read from env vars at import time. Modal evaluates decorators on the local
machine at deploy time, so set them on the deploy command line. Defaults = tested L40S config.
The deploy-time values are also baked into the image (CONTAINER_ENV) so the container,
which re-imports this module, sees the same config.

  QWEN_GPU        GPU type            (default L40S; use "H100!" to pin H100 and block auto-upgrade to H200)
  QWEN_MAX_LEN    --max-model-len     (default 32768)
  QWEN_MAX_SEQS   --max-num-seqs      (default 4)
  QWEN_MTP        "1" adds MTP speculative decoding (default "0")
  QWEN_SCALEDOWN  scaledown_window s  (default 300)
  QWEN_SPEC_TOKENS  num_speculative_tokens for MTP (default 4; model card value)
  QWEN_TIMEOUT    @app.function timeout s = hard cap per container, 1..86400 (default 1800)
  QWEN_STARTUP    web server startup_timeout s; readiness deadline = QWEN_STARTUP - 30 (default 600)
  QWEN_DG_WARMUP  VLLM_DEEP_GEMM_WARMUP: "" = vLLM default (relax), or "skip" (H100 fallback only)
  QWEN_GPU_UTIL   --gpu-memory-utilization (default "0.90"; KV math in README before raising)
  QWEN_KV_DTYPE   --kv-cache-dtype, passed only when set (default: vLLM auto). fp8 is unverified here, see README
  QWEN_DEFAULT_THINK  "1" = thinking on (default, same as before); "0" = server default enable_thinking=false

  H100 + MTP, first boot (empty compile/JIT cache, about 15 min):
    QWEN_GPU='H100!' QWEN_MTP=1 QWEN_MAX_LEN=65536 QWEN_MAX_SEQS=32 QWEN_SCALEDOWN=60 QWEN_STARTUP=1500 modal deploy modal_app.py
  H100 + MTP, later boots: same command (expected to be much faster: cached kernels, see FAILURES.md).
  L40S default: modal deploy modal_app.py

  stop: modal app stop qwen-abliterated
"""

import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request

import modal

APP_NAME = "qwen-abliterated"
GPU = os.environ.get("QWEN_GPU", "L40S")
MAX_LEN = os.environ.get("QWEN_MAX_LEN", "32768")
MAX_SEQS = os.environ.get("QWEN_MAX_SEQS", "4")
MTP = os.environ.get("QWEN_MTP", "0") == "1"
SCALEDOWN = int(os.environ.get("QWEN_SCALEDOWN", "300"))
STARTUP = int(os.environ.get("QWEN_STARTUP", "600"))
DG_WARMUP = os.environ.get("QWEN_DG_WARMUP", "")
SPEC_TOKENS = int(os.environ.get("QWEN_SPEC_TOKENS", "4"))
TIMEOUT = int(os.environ.get("QWEN_TIMEOUT", "1800"))
GPU_UTIL = os.environ.get("QWEN_GPU_UTIL", "0.90")
KV_DTYPE = os.environ.get("QWEN_KV_DTYPE", "")
DEFAULT_THINK = os.environ.get("QWEN_DEFAULT_THINK", "1")
if SPEC_TOKENS < 1:
    raise ValueError("QWEN_SPEC_TOKENS must be >= 1")
if not 1 <= TIMEOUT <= 86400:  # Modal max function timeout is 24 h
    raise ValueError("QWEN_TIMEOUT must be between 1 and 86400 seconds")
if not 0.0 < float(GPU_UTIL) <= 1.0:
    raise ValueError("QWEN_GPU_UTIL must be in (0, 1]")
if DEFAULT_THINK not in ("0", "1"):
    raise ValueError('QWEN_DEFAULT_THINK must be "0" or "1"')
VLLM_TAG = "v0.33.0"  # newest stable vllm/vllm-openai tag on Docker Hub (2026-11-02)
MODEL_DIR = "/models/qwen3.8-27b-heretic-mtp-fp8"  # written by download.py
SERVED_MODEL_NAME = "qwen-abliterated"
PORT = 8000
CACHE_DIR = "/cache"  # mount point of the qwen-vllm-cache Volume

# Model card (huggingface.co/OptimizeLLM/Qwen3.8-27B-heretic-MTP-FP8) recommends 4 (the default).
# QWEN_SPEC_TOKENS=3 is for agent traffic with low draft acceptance (see README).
MTP_SPEC_CONFIG = f'{{"method":"mtp","num_speculative_tokens":{SPEC_TOKENS}}}'

app = modal.App(APP_NAME)

weights = modal.Volume.from_name("qwen-weights", create_if_missing=True)
cache_vol = modal.Volume.from_name("qwen-vllm-cache", create_if_missing=True)
api_secret = modal.Secret.from_name("qwen-api-key", required_keys=["VLLM_API_KEY"])

# Deploy-time config must reach the container (the module is re-imported there without
# the QWEN_* shell vars). Compile and JIT caches go to the cache Volume so they survive
# cold starts: vLLM torch.compile (VLLM_CACHE_ROOT/torch_compile_cache/<hash>),
# DeepGEMM kernels (DG_JIT_CACHE_DIR, vLLM default is under VLLM_CACHE_ROOT), Triton.
CONTAINER_ENV = {
    "HF_HUB_OFFLINE": "1",  # the weights are already on the Volume
    "QWEN_GPU": GPU,
    "QWEN_MAX_LEN": MAX_LEN,
    "QWEN_MAX_SEQS": MAX_SEQS,
    "QWEN_MTP": "1" if MTP else "0",
    "QWEN_STARTUP": str(STARTUP),
    "VLLM_CACHE_ROOT": f"{CACHE_DIR}/vllm",
    "DG_JIT_CACHE_DIR": f"{CACHE_DIR}/deep_gemm",
    "TRITON_CACHE_DIR": f"{CACHE_DIR}/triton",
}
if DG_WARMUP:
    CONTAINER_ENV["VLLM_DEEP_GEMM_WARMUP"] = DG_WARMUP
# Baked only when set, so a default deploy produces the same image as before (defaults match).
if "QWEN_SPEC_TOKENS" in os.environ:
    CONTAINER_ENV["QWEN_SPEC_TOKENS"] = str(SPEC_TOKENS)
if "QWEN_TIMEOUT" in os.environ:
    CONTAINER_ENV["QWEN_TIMEOUT"] = str(TIMEOUT)
if "QWEN_GPU_UTIL" in os.environ:
    CONTAINER_ENV["QWEN_GPU_UTIL"] = GPU_UTIL
if "QWEN_KV_DTYPE" in os.environ:
    CONTAINER_ENV["QWEN_KV_DTYPE"] = KV_DTYPE
if "QWEN_DEFAULT_THINK" in os.environ:
    CONTAINER_ENV["QWEN_DEFAULT_THINK"] = DEFAULT_THINK

# The vllm-openai image's ENTRYPOINT is `vllm serve`. Clear it so Modal can run its
# own runtime; serve() launches vLLM itself.
vllm_image = (
    modal.Image.from_registry(f"vllm/vllm-openai:{VLLM_TAG}", add_python="3.12")
    .entrypoint([])
    .env(CONTAINER_ENV)
)

_vllm_proc = None  # keep a handle so the child process is not garbage-collected


@app.function(
    image=vllm_image,
    gpu=GPU,
    volumes={"/models": weights, CACHE_DIR: cache_vol},
    secrets=[api_secret],
    timeout=TIMEOUT,  # hard cap per container (QWEN_TIMEOUT, default 1800 s); next request cold-starts a new one
    min_containers=0,
    max_containers=1,
    scaledown_window=SCALEDOWN,
)
@modal.concurrent(max_inputs=16)  # vLLM batches internally (--max-num-seqs from QWEN_MAX_SEQS); allow queueing at the proxy
@modal.web_server(PORT, startup_timeout=STARTUP)
def serve():
    global _vllm_proc

    api_key = os.environ.get("VLLM_API_KEY")
    if not api_key:
        raise RuntimeError("VLLM_API_KEY is not set: check Modal secret 'qwen-api-key'")
    if not os.path.isdir(MODEL_DIR):
        raise RuntimeError(f"{MODEL_DIR} not found: run `modal run download.py` first")

    weights.reload()
    vllm_bin = shutil.which("vllm") or "vllm"
    cmd = [
        vllm_bin,
        "serve",
        MODEL_DIR,
        "--served-model-name", SERVED_MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--max-model-len", MAX_LEN,
        "--max-num-seqs", MAX_SEQS,
        "--gpu-memory-utilization", GPU_UTIL,
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
        "--reasoning-parser", "qwen3",
        "--reasoning-config", "{}",
        "--enable-prefix-caching",
        "--language-model-only",
    ]  # vLLM reads VLLM_API_KEY from the env when --api-key is absent
    if MTP:
        cmd += ["--speculative-config", MTP_SPEC_CONFIG]
    if KV_DTYPE:
        cmd += ["--kv-cache-dtype", KV_DTYPE]
    if DEFAULT_THINK == "0":
        # Request-level chat_template_kwargs still override this default (vLLM renderers/params.py merge_kwargs).
        cmd += ["--default-chat-template-kwargs", '{"enable_thinking": false}']
    print(
        f"config: gpu={GPU} max_len={MAX_LEN} max_seqs={MAX_SEQS} mtp={MTP} spec_tokens={SPEC_TOKENS} "
        f"timeout={TIMEOUT}s startup={STARTUP}s "
        f"dg_warmup={os.environ.get('VLLM_DEEP_GEMM_WARMUP', 'vllm-default')} cache={CACHE_DIR} "
        f"gpu_util={GPU_UTIL} kv_dtype={KV_DTYPE or 'auto'} default_think={DEFAULT_THINK}",
        flush=True,
    )
    print(f"vLLM cmd: {' '.join(cmd)}", flush=True)
    _vllm_proc = subprocess.Popen(cmd, env=os.environ.copy())

    # Fail fast if vLLM dies during startup instead of burning GPU time until the startup timeout.
    budget = STARTUP - 30  # keep 30 s of margin under web_server startup_timeout
    deadline = time.monotonic() + budget
    while True:
        if _vllm_proc.poll() is not None:
            raise RuntimeError(f"vLLM exited during startup with code {_vllm_proc.returncode}: see container logs")
        if time.monotonic() > deadline:
            raise RuntimeError(f"vLLM did not become healthy within {budget}s")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as resp:
                if resp.status == 200:
                    break
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)

    # Server is healthy, so warmup and compile caches are complete. Persist them now.
    try:
        cache_vol.commit()
        print("cache volume committed", flush=True)
    except Exception as exc:  # a failed commit must not kill a healthy server
        print(f"cache volume commit failed: {exc}", flush=True)
