"""
Shared model loading. Every module calls load_models() instead of
duplicating loading logic. Behavior is controlled entirely by config.py
(which itself reads env vars on the cluster).
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from config import (
    DRAFT_MODEL_NAME, TARGET_MODEL_NAME, QUANTIZATION_CONFIG, DEVICE_MAP,
)


def _load_one(name):
    kwargs = {}
    if QUANTIZATION_CONFIG is not None:
        kwargs["quantization_config"] = QUANTIZATION_CONFIG
    if DEVICE_MAP is not None:
        kwargs["device_map"] = DEVICE_MAP
    return AutoModelForCausalLM.from_pretrained(name, **kwargs)


def load_models():
    """Load draft model, target model, and tokenizer based on config.py settings."""
    tokenizer = AutoTokenizer.from_pretrained(DRAFT_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    draft_model = _load_one(DRAFT_MODEL_NAME)
    target_model = _load_one(TARGET_MODEL_NAME)

    draft_model.eval()
    target_model.eval()
    return draft_model, target_model, tokenizer
