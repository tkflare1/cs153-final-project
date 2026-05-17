"""
Main Entry Point — Speculative Decoding Research Project
Runs the full benchmark pipeline end to end: loads models, sweeps K values
across all prompt domains, saves results.json, and generates plots.

Stanford CS153 — Speculative Decoding
"""

import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# MODEL CONFIGURATION — flip USE_COLAB to True when running on Colab with GPU
# ============================================================================
USE_COLAB = False

if USE_COLAB:
    # --- Google Colab (T4 GPU, Llama 3.2 with 4-bit quantization) ---
    from transformers import BitsAndBytesConfig
    DRAFT_MODEL_NAME = "meta-llama/Llama-3.2-1B"
    TARGET_MODEL_NAME = "meta-llama/Llama-3.2-3B"
    QUANTIZATION_CONFIG = BitsAndBytesConfig(load_in_4bit=True)
else:
    # --- Local development (CPU, small models) ---
    DRAFT_MODEL_NAME = "distilgpt2"       # 82M params
    TARGET_MODEL_NAME = "gpt2"            # 124M params
    QUANTIZATION_CONFIG = None
# ============================================================================

from benchmark import run_full_benchmark, print_results_tables, PROMPTS, K_VALUES
from plot import generate_all_plots

SEED = 42
RESULTS_PATH = "results.json"


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
    tokenizer = AutoTokenizer.from_pretrained(DRAFT_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    if USE_COLAB:
        draft_model = AutoModelForCausalLM.from_pretrained(
            DRAFT_MODEL_NAME,
            quantization_config=QUANTIZATION_CONFIG,
            device_map="auto",
        )
        target_model = AutoModelForCausalLM.from_pretrained(
            TARGET_MODEL_NAME,
            quantization_config=QUANTIZATION_CONFIG,
            device_map="auto",
        )
    else:
        draft_model = AutoModelForCausalLM.from_pretrained(DRAFT_MODEL_NAME)
        target_model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL_NAME)

    draft_model.eval()
    target_model.eval()
    print("Models loaded.\n")

    # ---- Run benchmark ----
    results = run_full_benchmark(draft_model, target_model, tokenizer)

    # ---- Print results ----
    print_results_tables(results)

    # ---- Save results ----
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # ---- Generate plots ----
    print("\nGenerating plots...")
    generate_all_plots(RESULTS_PATH)

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
