"""
Shared configuration for all modules.

Three run modes:
  - Local CPU (default):  small models (distilgpt2 / gpt2), no GPU.
  - Colab:                gated Llama + 4-bit quant on a T4. Set USE_COLAB = True below.
  - Cluster (H100):       full-precision Llama on GPU. Driven by env vars (see below)
                          so the same code runs without editing this file:
                              USE_CLUSTER=1
                              DRAFT_MODEL=...  TARGET_MODEL=...   (optional overrides)

Env vars always win over the in-file flags, which keeps the cluster sbatch scripts
self-contained and avoids editing source between runs.
"""

import os


def _truthy(val):
    return str(val).strip().lower() in ("1", "true", "yes", "on")


# In-file flag preserved for the documented Colab workflow ("flip this to True").
USE_COLAB = False

# Env overrides (used on the cluster; harmless locally).
if os.environ.get("USE_COLAB") is not None:
    USE_COLAB = _truthy(os.environ.get("USE_COLAB"))
USE_CLUSTER = _truthy(os.environ.get("USE_CLUSTER", ""))

if USE_COLAB:
    from transformers import BitsAndBytesConfig
    DRAFT_MODEL_NAME = os.environ.get("DRAFT_MODEL", "meta-llama/Llama-3.2-1B")
    TARGET_MODEL_NAME = os.environ.get("TARGET_MODEL", "meta-llama/Llama-3.2-3B")
    QUANTIZATION_CONFIG = BitsAndBytesConfig(load_in_4bit=True)
    DEVICE_MAP = "auto"
elif USE_CLUSTER:
    # Full precision on H100 (80GB) — no quantization needed.
    DRAFT_MODEL_NAME = os.environ.get("DRAFT_MODEL", "meta-llama/Llama-3.2-1B")
    TARGET_MODEL_NAME = os.environ.get("TARGET_MODEL", "meta-llama/Llama-3.2-3B")
    QUANTIZATION_CONFIG = None
    DEVICE_MAP = "auto"
else:
    DRAFT_MODEL_NAME = os.environ.get("DRAFT_MODEL", "distilgpt2")
    TARGET_MODEL_NAME = os.environ.get("TARGET_MODEL", "gpt2")
    QUANTIZATION_CONFIG = None
    DEVICE_MAP = None

SEED = 42
K_VALUES = [1, 2, 4, 6, 8]
MAX_NEW_TOKENS = 50
RESULTS_PATH = "results.json"
