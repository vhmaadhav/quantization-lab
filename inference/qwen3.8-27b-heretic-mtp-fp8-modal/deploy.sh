#!/usr/bin/env bash
# Canonical deploy for the long-task config. Always deploy through this script so the env
# (and the image layer it bakes) stays identical. Deploying starts no GPU, but the next request does, so review the values first.
# Then set WD_MAX_AGE_MIN=185 for ./wd (see README, Watchdog).
set -euo pipefail
cd "$(dirname "$0")"

QWEN_MTP=1 QWEN_STARTUP=600 QWEN_MAX_LEN=98304 QWEN_MAX_SEQS=2 QWEN_SPEC_TOKENS=3 QWEN_TIMEOUT=10800 QWEN_SCALEDOWN=300 modal deploy modal_app.py

# Rollback (previous long-task config, 65k context):
# QWEN_MTP=1 QWEN_STARTUP=600 QWEN_MAX_LEN=65536 QWEN_MAX_SEQS=2 QWEN_SPEC_TOKENS=3 QWEN_TIMEOUT=10800 QWEN_SCALEDOWN=300 modal deploy modal_app.py

# Short config (explicit copy of the defaults), for rollback. Keep commented:
# QWEN_MTP=1 QWEN_STARTUP=600 QWEN_MAX_LEN=32768 QWEN_MAX_SEQS=4 QWEN_SPEC_TOKENS=4 QWEN_TIMEOUT=1800 QWEN_SCALEDOWN=300 modal deploy modal_app.py

# Long-context + thinking-off candidate (not the default). The client context window must be set to 98304, not the model's 262K.
# QWEN_MTP=1 QWEN_STARTUP=600 QWEN_MAX_LEN=98304 QWEN_MAX_SEQS=1 QWEN_SPEC_TOKENS=3 QWEN_TIMEOUT=10800 QWEN_SCALEDOWN=300 QWEN_DEFAULT_THINK=0 modal deploy modal_app.py
