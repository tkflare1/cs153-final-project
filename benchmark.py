"""
Benchmarking Harness
Runs both baseline and speculative decoding across multiple prompts, domains,
and speculation lengths (K). Collects tokens/sec, acceptance rate, wall-clock
time, and effective speedup ratio.
"""

import json
import time
import torch
import numpy as np
from tqdm import tqdm

from speculative_decode import speculative_decode
from baseline import baseline_autoregressive
from config import SEED, K_VALUES, MAX_NEW_TOKENS

torch.manual_seed(SEED)
np.random.seed(SEED)


def _model_device(model):
    """Get the device a model's parameters live on (works with device_map='auto')."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")

PROMPTS = {
    "prose": [
        "Once upon a time in a distant kingdom, there lived a",
        "The sun set over the ocean, painting the sky in shades of",
        "She opened the old leather journal and began to read the first entry,",
    ],
    "code": [
        "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
        "class LinkedList:\n    def __init__(self):\n",
        "import numpy as np\n\ndef matrix_multiply(A, B):\n",
    ],
    "math": [
        "To prove that the square root of 2 is irrational, assume for contradiction that",
        "The derivative of f(x) = x^3 + 2x^2 - 5x + 1 is",
        "Using the binomial theorem, expand (x + y)^4 to get",
    ],
}


def run_baseline_benchmark(model, tokenizer, prompts_dict):
    """Run baseline autoregressive decoding on all prompts and collect metrics."""
    results = []
    for domain, prompts in prompts_dict.items():
        for prompt in prompts:
            torch.manual_seed(SEED)
            np.random.seed(SEED)

            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            input_ids = input_ids.to(_model_device(model))
            _, elapsed, tps = baseline_autoregressive(
                model, tokenizer, input_ids, max_new_tokens=MAX_NEW_TOKENS
            )
            results.append({
                "domain": domain,
                "prompt": prompt[:50],
                "tokens_per_sec": tps,
                "elapsed": elapsed,
            })
    return results


def run_speculative_benchmark(draft_model, target_model, tokenizer, prompts_dict, K):
    """Run speculative decoding for a given K on all prompts and collect metrics."""
    results = []
    for domain, prompts in prompts_dict.items():
        for prompt in prompts:
            torch.manual_seed(SEED)
            np.random.seed(SEED)

            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            input_ids = input_ids.to(_model_device(draft_model))

            start_time = time.perf_counter()
            generated, accepted, proposed = speculative_decode(
                draft_model, target_model, tokenizer, input_ids,
                K=K, max_new_tokens=MAX_NEW_TOKENS,
            )
            elapsed = time.perf_counter() - start_time

            num_new = generated.shape[1] - input_ids.shape[1]
            tps = num_new / elapsed if elapsed > 0 else 0
            acc_rate = accepted / proposed if proposed > 0 else 0.0

            results.append({
                "domain": domain,
                "prompt": prompt[:50],
                "K": K,
                "tokens_per_sec": tps,
                "elapsed": elapsed,
                "acceptance_rate": acc_rate,
                "tokens_generated": num_new,
                "tokens_accepted": accepted,
                "tokens_proposed": proposed,
            })
    return results


def run_full_benchmark(draft_model, target_model, tokenizer, prompts_dict=None, k_values=None):
    """
    Run the complete benchmark sweep.

    Returns a dict with:
      - baseline_results: list of per-prompt baseline metrics
      - speculative_results: list of per-prompt speculative metrics for each K
      - k_summary: aggregate metrics per K value
      - domain_summary: aggregate metrics per domain
    """
    if prompts_dict is None:
        prompts_dict = PROMPTS
    if k_values is None:
        k_values = K_VALUES

    print("=" * 60)
    print("Running Baseline Benchmark")
    print("=" * 60)
    baseline_results = run_baseline_benchmark(target_model, tokenizer, prompts_dict)
    baseline_avg_tps = np.mean([r["tokens_per_sec"] for r in baseline_results])
    print(f"Baseline average: {baseline_avg_tps:.2f} tokens/sec\n")

    all_spec_results = []
    for K in tqdm(k_values, desc="Sweeping K values"):
        print(f"\n--- Speculative Decoding K={K} ---")
        spec_results = run_speculative_benchmark(
            draft_model, target_model, tokenizer, prompts_dict, K
        )
        all_spec_results.extend(spec_results)

    k_summary = []
    for K in k_values:
        k_results = [r for r in all_spec_results if r["K"] == K]
        avg_tps = np.mean([r["tokens_per_sec"] for r in k_results])
        avg_acc = np.mean([r["acceptance_rate"] for r in k_results])
        avg_elapsed = np.mean([r["elapsed"] for r in k_results])
        speedup = avg_tps / baseline_avg_tps if baseline_avg_tps > 0 else 0
        k_summary.append({
            "K": K,
            "acceptance_rate": round(avg_acc, 4),
            "tokens_per_sec": round(avg_tps, 2),
            "speedup": round(speedup, 4),
            "avg_elapsed": round(avg_elapsed, 2),
        })

    domains = list(prompts_dict.keys())
    domain_summary = []
    for domain in domains:
        domain_results = [r for r in all_spec_results if r["domain"] == domain]
        avg_acc = np.mean([r["acceptance_rate"] for r in domain_results])
        avg_tps = np.mean([r["tokens_per_sec"] for r in domain_results])
        speedup = avg_tps / baseline_avg_tps if baseline_avg_tps > 0 else 0
        domain_summary.append({
            "domain": domain,
            "acceptance_rate": round(avg_acc, 4),
            "avg_tokens_per_sec": round(avg_tps, 2),
            "speedup": round(speedup, 4),
        })

    results = {
        "baseline_avg_tokens_per_sec": round(baseline_avg_tps, 2),
        "baseline_results": baseline_results,
        "speculative_results": all_spec_results,
        "k_summary": k_summary,
        "domain_summary": domain_summary,
    }

    return results


def print_results_tables(results):
    """Pretty-print the K-summary and domain-summary tables."""
    print("\n" + "=" * 60)
    print("K vs Acceptance Rate vs Tokens/sec vs Speedup")
    print("=" * 60)
    print(f"{'K':>4}  {'Acc. Rate':>10}  {'Tok/sec':>10}  {'Speedup':>10}")
    print("-" * 40)
    for row in results["k_summary"]:
        print(f"{row['K']:>4}  {row['acceptance_rate']:>10.4f}  "
              f"{row['tokens_per_sec']:>10.2f}  {row['speedup']:>10.4f}")

    print(f"\nBaseline: {results['baseline_avg_tokens_per_sec']:.2f} tokens/sec")

    print("\n" + "=" * 60)
    print("Domain vs Acceptance Rate vs Speedup")
    print("=" * 60)
    print(f"{'Domain':>10}  {'Acc. Rate':>10}  {'Tok/sec':>10}  {'Speedup':>10}")
    print("-" * 45)
    for row in results["domain_summary"]:
        print(f"{row['domain']:>10}  {row['acceptance_rate']:>10.4f}  "
              f"{row['avg_tokens_per_sec']:>10.2f}  {row['speedup']:>10.4f}")


if __name__ == "__main__":
    from models import load_models
    print("Loading models for benchmark verification...")
    draft_model, target_model, tokenizer = load_models()

    test_prompts = {"prose": [PROMPTS["prose"][0]]}
    results = run_full_benchmark(
        draft_model, target_model, tokenizer,
        prompts_dict=test_prompts, k_values=[4]
    )
    print_results_tables(results)
    print("\nbenchmark.py verification PASSED.")
