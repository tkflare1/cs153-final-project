"""
Adaptive-K + Long-Context Experiments

Runs two extensions:
1) Long-context benchmark: fixed-K sweep at larger max_new_tokens.
2) Adaptive-K benchmark: per-prompt K update driven by acceptance rate.

Outputs:
  - long_context_results.json
  - adaptive_k_results.json
  - long_context_speedup.png
  - adaptive_k_usage.png
"""

import argparse
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import SEED, K_VALUES, MAX_NEW_TOKENS
from models import load_models
from benchmark import (
    PROMPTS,
    run_full_benchmark,
    run_baseline_benchmark,
    run_adaptive_speculative_benchmark,
)


def _round(v):
    return round(float(v), 4)


def _plot_long_context(rows, out_path="long_context_speedup.png"):
    tokens = [r["max_new_tokens"] for r in rows]
    best = [r["best_speedup"] for r in rows]
    base = [r["baseline_avg_tokens_per_sec"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(tokens, best, "o-", linewidth=2, color="tab:blue", label="Best speedup")
    ax1.axhline(1.0, linestyle="--", color="gray", alpha=0.7)
    ax1.set_xlabel("max_new_tokens")
    ax1.set_ylabel("Best speculative speedup", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(tokens, base, "s-", linewidth=2, color="tab:red", label="Baseline tok/s")
    ax2.set_ylabel("Baseline tok/s", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("Long-context trend")
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _plot_adaptive_k(k_trace, out_path="adaptive_k_usage.png"):
    if not k_trace:
        return
    x = list(range(1, len(k_trace) + 1))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, k_trace, "o-", linewidth=2, markersize=6, color="#7c3aed")
    ax.set_xlabel("Prompt index (stream order)")
    ax.set_ylabel("K used")
    ax.set_title("Adaptive-K trajectory across prompts")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def run_long_context(draft, target, tok, token_counts):
    rows = []
    for n in token_counts:
        print("=" * 60)
        print(f"Long-context sweep: max_new_tokens={n}")
        results = run_full_benchmark(
            draft,
            target,
            tok,
            prompts_dict=PROMPTS,
            k_values=K_VALUES,
            max_new_tokens=n,
        )
        best = max(results["k_summary"], key=lambda r: r["speedup"])
        rows.append({
            "max_new_tokens": n,
            "baseline_avg_tokens_per_sec": _round(results["baseline_avg_tokens_per_sec"]),
            "best_K": int(best["K"]),
            "best_speedup": _round(best["speedup"]),
            "k_summary": results["k_summary"],
        })
    payload = {"token_counts": token_counts, "results": rows}
    with open("long_context_results.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("Saved: long_context_results.json")
    _plot_long_context(rows)
    return payload


def run_adaptive_k(draft, target, tok, max_new_tokens, k_min, k_max, k_init, hi, lo):
    print("=" * 60)
    print("Adaptive-K run")
    baseline_rows = run_baseline_benchmark(target, tok, PROMPTS, max_new_tokens=max_new_tokens)
    baseline_tps = np.mean([r["tokens_per_sec"] for r in baseline_rows])

    adaptive_rows = run_adaptive_speculative_benchmark(
        draft,
        target,
        tok,
        PROMPTS,
        k_min=k_min,
        k_max=k_max,
        k_init=k_init,
        hi=hi,
        lo=lo,
        max_new_tokens=max_new_tokens,
    )
    adaptive_tps = np.mean([r["tokens_per_sec"] for r in adaptive_rows])
    adaptive_acc = np.mean([r["acceptance_rate"] for r in adaptive_rows])
    k_trace = [int(r["K_used"]) for r in adaptive_rows]

    payload = {
        "policy": {
            "k_min": k_min,
            "k_max": k_max,
            "k_init": k_init,
            "hi": hi,
            "lo": lo,
            "max_new_tokens": max_new_tokens,
        },
        "baseline_avg_tokens_per_sec": _round(baseline_tps),
        "adaptive_avg_tokens_per_sec": _round(adaptive_tps),
        "adaptive_speedup_vs_baseline": _round(adaptive_tps / baseline_tps if baseline_tps > 0 else 0),
        "adaptive_avg_acceptance_rate": _round(adaptive_acc),
        "k_trace": k_trace,
        "adaptive_rows": adaptive_rows,
    }
    with open("adaptive_k_results.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("Saved: adaptive_k_results.json")
    _plot_adaptive_k(k_trace)
    return payload


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--long-context", action="store_true", help="run long-context sweep")
    p.add_argument("--adaptive-k", action="store_true", help="run adaptive-K benchmark")
    p.add_argument("--token-counts", type=int, nargs="+", default=[MAX_NEW_TOKENS, 200])
    p.add_argument("--adaptive-max-new", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--k-min", type=int, default=min(K_VALUES))
    p.add_argument("--k-max", type=int, default=max(K_VALUES))
    p.add_argument("--k-init", type=int, default=2)
    p.add_argument("--hi", type=float, default=0.80)
    p.add_argument("--lo", type=float, default=0.60)
    return p.parse_args()


def main():
    args = parse_args()
    if not args.long_context and not args.adaptive_k:
        args.long_context = True
        args.adaptive_k = True

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading models...")
    draft, target, tok = load_models()

    if args.long_context:
        run_long_context(draft, target, tok, args.token_counts)

    if args.adaptive_k:
        run_adaptive_k(
            draft,
            target,
            tok,
            max_new_tokens=args.adaptive_max_new,
            k_min=args.k_min,
            k_max=args.k_max,
            k_init=args.k_init,
            hi=args.hi,
            lo=args.lo,
        )


if __name__ == "__main__":
    main()
