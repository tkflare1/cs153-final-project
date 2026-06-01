"""
Draft Model Size Sweep
----------------------
Speculative decoding requires the draft and target models to share a tokenizer
(the accept/reject test compares per-token probabilities over the same vocab).
Llama 3.2 only ships 1B and 3B, so this sweep uses the Qwen2.5 family, whose
0.5B / 1.5B / 3B / 7B checkpoints all share one tokenizer and are ungated.

Fixed target (Qwen2.5-7B), varying draft size, to find the draft-to-target
size ratio where speculative decoding stops paying off. Models load in the same
precision as the main cluster run (fp32, device_map="auto").

Outputs: draft_sweep.json + draft_sweep.png  (written incrementally so a failure
on one draft still leaves partial results).
"""

import json
import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from speculative_decode import speculative_decode
from baseline import baseline_autoregressive

SEED = 42
TARGET = "Qwen/Qwen2.5-7B"
TARGET_SIZE_B = 7.0
DRAFTS = [
    ("Qwen/Qwen2.5-0.5B", 0.5),
    ("Qwen/Qwen2.5-1.5B", 1.5),
    ("Qwen/Qwen2.5-3B", 3.0),
]
K = 4
MAX_NEW_TOKENS = 50
PROMPTS = [
    ("prose", "Once upon a time in a distant kingdom, there lived a"),
    ("code", "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n"),
    ("math", "To prove that the square root of 2 is irrational, assume for contradiction that"),
]


def load(name):
    model = AutoModelForCausalLM.from_pretrained(name, device_map="auto")
    model.eval()
    return model


def main():
    print(f"loading shared tokenizer + target: {TARGET}")
    tok = AutoTokenizer.from_pretrained(TARGET)
    tok.pad_token = tok.eos_token
    target = load(TARGET)
    tdev = next(target.parameters()).device
    print(f"target on {tdev}")

    # Baseline throughput of the target.
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    btps = []
    for _, p in PROMPTS:
        ids = tok.encode(p, return_tensors="pt").to(tdev)
        _, _, tps = baseline_autoregressive(target, tok, ids, max_new_tokens=MAX_NEW_TOKENS)
        btps.append(tps)
    baseline_tps = float(np.mean(btps))
    print(f"baseline (target {TARGET}): {baseline_tps:.2f} tok/s")

    sweep = []
    for name, size in DRAFTS:
        print(f"\n=== draft {name} ({size}B), ratio {TARGET_SIZE_B/size:.1f}:1 ===")
        draft = load(name)
        ddev = next(draft.parameters()).device
        accs, tpss = [], []
        for domain, p in PROMPTS:
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            ids = tok.encode(p, return_tensors="pt").to(ddev)
            t0 = time.perf_counter()
            gen, acc, prop = speculative_decode(
                draft, target, tok, ids, K=K, max_new_tokens=MAX_NEW_TOKENS
            )
            el = time.perf_counter() - t0
            n_new = gen.shape[1] - ids.shape[1]
            tpss.append(n_new / el if el > 0 else 0.0)
            accs.append(acc / prop if prop > 0 else 0.0)
        row = {
            "draft": name,
            "draft_size_b": size,
            "target_size_b": TARGET_SIZE_B,
            "ratio": round(TARGET_SIZE_B / size, 2),
            "acceptance_rate": round(float(np.mean(accs)), 4),
            "tokens_per_sec": round(float(np.mean(tpss)), 2),
            "speedup": round(float(np.mean(tpss)) / baseline_tps, 4),
        }
        sweep.append(row)
        print(row)

        # Incremental save so a later failure doesn't lose earlier rows.
        with open("draft_sweep.json", "w") as f:
            json.dump({"target": TARGET, "K": K,
                       "baseline_tps": round(baseline_tps, 2), "sweep": sweep}, f, indent=2)

        del draft
        torch.cuda.empty_cache()

    print("\nsaved draft_sweep.json")
    _plot(sweep)


def _plot(sweep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = [r["draft_size_b"] for r in sweep]
    speed = [r["speedup"] for r in sweep]
    acc = [r["acceptance_rate"] for r in sweep]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(sizes, speed, "o-", color="tab:blue", linewidth=2, markersize=8, label="Speedup")
    ax1.set_xlabel("Draft model size (B params)")
    ax1.set_ylabel("Speedup vs baseline", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.axhline(1.0, ls="--", color="gray", alpha=0.7)

    ax2 = ax1.twinx()
    ax2.plot(sizes, acc, "s-", color="tab:red", linewidth=2, markersize=8, label="Acceptance rate")
    ax2.set_ylabel("Acceptance rate", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title(f"Draft-size sweep: drafts vs {TARGET} (K={K})")
    fig.tight_layout()
    plt.savefig("draft_sweep.png", dpi=120)
    print("saved draft_sweep.png")


if __name__ == "__main__":
    main()
