"""
KV-Cache comparison + correctness check.

1. Equivalence: with a fixed seed, the cached decoders must produce token-for-token
   identical output to the non-cached versions (the cache changes *how* logits are
   computed, not their values). This is the losslessness guarantee.
2. Speedup: wall-clock of cached vs non-cached for baseline and speculative (K=4).

Run via cluster/kv.sbatch. Works for any model pair from config.py (smoke with
gpt2 by setting DRAFT_MODEL/TARGET_MODEL; real run with USE_CLUSTER=1 -> Llama).
"""

import json
import time
import torch
import numpy as np

from config import DRAFT_MODEL_NAME, TARGET_MODEL_NAME, SEED, MAX_NEW_TOKENS
from models import load_models
from baseline import baseline_autoregressive
from speculative_decode import speculative_decode
from kv_cache import baseline_autoregressive_kv, speculative_decode_kv

K = 4
PROMPTS = [
    "Once upon a time in a distant kingdom, there lived a",
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    "To prove that the square root of 2 is irrational, assume for contradiction that",
]


def _seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED)


def main():
    print(f"models: {DRAFT_MODEL_NAME} (draft) / {TARGET_MODEL_NAME} (target)")
    draft, target, tok = load_models()
    ddev = next(draft.parameters()).device
    tdev = next(target.parameters()).device
    print(f"draft on {ddev} | target on {tdev}")

    # ---------- 1. Correctness: cached == non-cached (token-identical) ----------
    print("\n=== equivalence check ===")
    all_ok = True
    for p in PROMPTS:
        ids = tok.encode(p, return_tensors="pt")

        _seed()
        g_base, _, _ = baseline_autoregressive(target, tok, ids.to(tdev), max_new_tokens=20)
        _seed()
        g_base_kv, _, _ = baseline_autoregressive_kv(target, tok, ids.to(tdev), max_new_tokens=20)
        base_ok = torch.equal(g_base.cpu(), g_base_kv.cpu())

        _seed()
        g_spec, _, _ = speculative_decode(draft, target, tok, ids.to(ddev), K=K, max_new_tokens=20)
        _seed()
        g_spec_kv, _, _ = speculative_decode_kv(draft, target, tok, ids.to(ddev), K=K, max_new_tokens=20)
        spec_ok = torch.equal(g_spec.cpu(), g_spec_kv.cpu())

        all_ok = all_ok and base_ok and spec_ok
        print(f"  [{ 'OK' if base_ok else 'MISMATCH' }] baseline   "
              f"[{ 'OK' if spec_ok else 'MISMATCH' }] speculative   prompt={p[:30]!r}")

    print(f"LOSSLESS: {'PASS' if all_ok else 'FAIL'}")

    # ---------- 2. Speedup: cached vs non-cached ----------
    print("\n=== speedup (cached vs non-cached) ===")
    def bench(fn):
        ts = []
        for p in PROMPTS:
            ids = tok.encode(p, return_tensors="pt")
            _seed()
            t0 = time.perf_counter()
            fn(ids)
            ts.append(time.perf_counter() - t0)
        return float(np.mean(ts))

    base_nc = bench(lambda ids: baseline_autoregressive(target, tok, ids.to(tdev), max_new_tokens=MAX_NEW_TOKENS))
    base_kv = bench(lambda ids: baseline_autoregressive_kv(target, tok, ids.to(tdev), max_new_tokens=MAX_NEW_TOKENS))
    spec_nc = bench(lambda ids: speculative_decode(draft, target, tok, ids.to(ddev), K=K, max_new_tokens=MAX_NEW_TOKENS))
    spec_kv = bench(lambda ids: speculative_decode_kv(draft, target, tok, ids.to(ddev), K=K, max_new_tokens=MAX_NEW_TOKENS))

    results = {
        "draft": DRAFT_MODEL_NAME,
        "target": TARGET_MODEL_NAME,
        "K": K,
        "max_new_tokens": MAX_NEW_TOKENS,
        "lossless": all_ok,
        "baseline_no_cache_s": round(base_nc, 4),
        "baseline_kv_s": round(base_kv, 4),
        "baseline_kv_speedup": round(base_nc / base_kv, 3) if base_kv else None,
        "speculative_no_cache_s": round(spec_nc, 4),
        "speculative_kv_s": round(spec_kv, 4),
        "speculative_kv_speedup": round(spec_nc / spec_kv, 3) if spec_kv else None,
    }
    for k, v in results.items():
        print(f"  {k}: {v}")

    with open("kv_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved kv_results.json")


if __name__ == "__main__":
    main()
