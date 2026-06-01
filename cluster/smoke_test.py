"""
Cheap GPU smoke test. Validates the full harness on the H100 using the
ungated gpt2 pair (no HF token needed) before spending budget on Llama.
Run via smoke.sbatch with USE_CLUSTER=1 DRAFT_MODEL=distilgpt2 TARGET_MODEL=gpt2.
"""

import sys
import torch
import transformers

print(f"transformers {transformers.__version__}")

from config import DRAFT_MODEL_NAME, TARGET_MODEL_NAME, DEVICE_MAP
from models import load_models
from speculative_decode import speculative_decode
from baseline import baseline_autoregressive

print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()} "
      f"| device count: {torch.cuda.device_count()}")
if not torch.cuda.is_available():
    print("SMOKE FAIL: CUDA not available in container")
    sys.exit(1)
print(f"gpu0: {torch.cuda.get_device_name(0)} | device_map={DEVICE_MAP}")

print(f"loading models: {DRAFT_MODEL_NAME} (draft) / {TARGET_MODEL_NAME} (target)")
draft, target, tok = load_models()
dd = next(draft.parameters()).device
td = next(target.parameters()).device
print(f"draft on {dd} | target on {td}")
if dd.type != "cuda" or td.type != "cuda":
    print("SMOKE FAIL: models not on GPU")
    sys.exit(1)

prompt = "The quick brown fox"
ids = tok.encode(prompt, return_tensors="pt").to(dd)
gen, acc, prop = speculative_decode(draft, target, tok, ids, K=4, max_new_tokens=20)
print(f"spec output: {tok.decode(gen[0], skip_special_tokens=True)!r}")
print(f"accepted {acc}/{prop}")

ids2 = tok.encode(prompt, return_tensors="pt").to(td)
_, elapsed, tps = baseline_autoregressive(target, tok, ids2, max_new_tokens=20)
print(f"baseline: {tps:.2f} tok/s in {elapsed:.2f}s")

print("SMOKE OK")
