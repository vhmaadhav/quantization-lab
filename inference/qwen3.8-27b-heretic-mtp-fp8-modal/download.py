"""CPU-only Modal job: download OptimizeLLM/Qwen3.8-27B-heretic-MTP-FP8 into the "qwen-weights" Volume.

  run: modal run download.py
"""

import os

import modal

APP_NAME = "qwen-abliterated-download"
REPO_ID = "OptimizeLLM/Qwen3.8-27B-heretic-MTP-FP8"  # public, not gated (checked 2026-11-03)
MODEL_DIR = "/models/qwen3.8-27b-heretic-mtp-fp8"  # must match modal_app.py

app = modal.App(APP_NAME)
weights = modal.Volume.from_name("qwen-weights", create_if_missing=True)

# huggingface_hub 2.x dropped the hf_transfer extra; hf_xet is the accelerated backend now.
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "huggingface_hub[hf_xet]==2.4.0"
)


@app.function(
    image=image,
    volumes={"/models": weights},
    timeout=3600,
)
def download():
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=MODEL_DIR,
        ignore_patterns=["reproduce/*"],  # HERETIC recipe files, not needed for serving
    )
    weights.commit()

    files = []
    total = 0
    for root, _, names in os.walk(path):
        for name in names:
            full = os.path.join(root, name)
            size = os.path.getsize(full)
            total += size
            files.append((os.path.relpath(full, path), size))

    print(f"Model dir: {path}")
    print(f"Total size on disk: {total / 1e9:.2f} GB ({total / 2**30:.2f} GiB) in {len(files)} files")
    for rel, size in sorted(files):
        print(f"  {size / 1e6:10.1f} MB  {rel}")
    return total


@app.local_entrypoint()
def main():
    download.remote()
