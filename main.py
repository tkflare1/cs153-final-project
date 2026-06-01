"""
Main Entry Point — Speculative Decoding Research Project
Runs the full benchmark pipeline end to end: loads models, sweeps K values
across all prompt domains, saves results.json, and generates all plots.

Stanford CS153 — Speculative Decoding
"""

import json
import torch
import numpy as np

from config import (
    DRAFT_MODEL_NAME, TARGET_MODEL_NAME, USE_COLAB,
    SEED, K_VALUES, MAX_NEW_TOKENS, RESULTS_PATH,
)
from models import load_models
from benchmark import run_full_benchmark, print_results_tables, PROMPTS
from plot import generate_all_plots
from visualize import run_latency_sweep, plot_latency_breakdown, plot_latency_percentage, generate_token_html


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("  Speculative Decoding — Full Benchmark")
    print(f"  Draft model:  {DRAFT_MODEL_NAME}")
    print(f"  Target model: {TARGET_MODEL_NAME}")
    print(f"  GPU mode:     {USE_COLAB}")
    print(f"  K values:     {K_VALUES}")
    print(f"  Domains:      {list(PROMPTS.keys())}")
    print("=" * 60)

    # ---- Load models ----
    print("\nLoading tokenizer and models...")
    draft_model, target_model, tokenizer = load_models()
    print("Models loaded.\n")

    # ---- Run benchmark ----
    results = run_full_benchmark(draft_model, target_model, tokenizer)

    # ---- Print results ----
    print_results_tables(results)

    # ---- Save results ----
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # ---- Generate speedup and acceptance rate plots ----
    print("\nGenerating plots...")
    generate_all_plots(RESULTS_PATH)

    # ---- Latency breakdown and token visualization ----
    print("\n" + "=" * 60)
    print("  Running Latency Breakdown & Token Visualization")
    print("=" * 60)
    latency_prompt = PROMPTS["prose"][0]
    latency_by_k, all_traces = run_latency_sweep(
        draft_model, target_model, tokenizer, latency_prompt, K_VALUES,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    plot_latency_breakdown(latency_by_k)
    plot_latency_percentage(latency_by_k)

    with open("latency_breakdown.json", "w") as f:
        json.dump(latency_by_k, f, indent=2)
    print("Saved: latency_breakdown.json")

    k4_trace = all_traces.get(4)
    if k4_trace:
        generate_token_html(
            k4_trace["trace"], latency_prompt, k4_trace["output_text"],
            K=4, acceptance_rate=k4_trace["acceptance_rate"],
        )

    # ---- Sample comparison printout (K=4) ----
    print("\n" + "=" * 60)
    print("  Sample Comparison: Baseline vs Speculative (K=4)")
    print("=" * 60)
    baseline_tps = results["baseline_avg_tokens_per_sec"]
    k4_row = next((r for r in results["k_summary"] if r["K"] == 4), None)
    if k4_row:
        print(f"  Baseline:     {baseline_tps:.2f} tokens/sec")
        print(f"  Speculative:  {k4_row['tokens_per_sec']:.2f} tokens/sec (K=4)")
        print(f"  Speedup:      {k4_row['speedup']:.4f}x")
        print(f"  Accept rate:  {k4_row['acceptance_rate']:.4f}")
    else:
        print("  K=4 results not found.")

    print("\nDone. All results saved and plots generated.")


if __name__ == "__main__":
    main()
