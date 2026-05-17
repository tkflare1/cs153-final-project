"""
Plotting Module
Generates publication-quality figures from results.json:
  - Plot 1: K (x-axis) vs Speedup (y-axis)
  - Plot 2: K (x-axis) vs Acceptance Rate (y-axis)
"""

import json
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt


def load_results(path="results.json"):
    """Load benchmark results from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def plot_speedup(k_summary, save_path="speedup_vs_k.png"):
    """Plot K vs effective speedup ratio."""
    ks = [row["K"] for row in k_summary]
    speedups = [row["speedup"] for row in k_summary]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, speedups, "o-", color="#2563eb", linewidth=2, markersize=8)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.6, label="Baseline (1.0x)")
    ax.set_xlabel("Speculation Length (K)", fontsize=13)
    ax.set_ylabel("Speedup (speculative / baseline)", fontsize=13)
    ax.set_title("Speculative Decoding: Speedup vs K", fontsize=15)
    ax.set_xticks(ks)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_acceptance_rate(k_summary, save_path="acceptance_rate_vs_k.png"):
    """Plot K vs average acceptance rate."""
    ks = [row["K"] for row in k_summary]
    acc_rates = [row["acceptance_rate"] for row in k_summary]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, acc_rates, "s-", color="#dc2626", linewidth=2, markersize=8)
    ax.set_xlabel("Speculation Length (K)", fontsize=13)
    ax.set_ylabel("Acceptance Rate", fontsize=13)
    ax.set_title("Speculative Decoding: Acceptance Rate vs K", fontsize=15)
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def generate_all_plots(results_path="results.json"):
    """Load results and generate both plots."""
    results = load_results(results_path)
    k_summary = results["k_summary"]
    plot_speedup(k_summary)
    plot_acceptance_rate(k_summary)
    print("All plots generated successfully.")


# ---------------------------------------------------------------------------
# Standalone verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Create minimal test data for verification if results.json doesn't exist
    import os
    if not os.path.exists("results.json"):
        test_results = {
            "baseline_avg_tokens_per_sec": 15.0,
            "k_summary": [
                {"K": 1, "acceptance_rate": 0.80, "tokens_per_sec": 12.0, "speedup": 0.80},
                {"K": 2, "acceptance_rate": 0.70, "tokens_per_sec": 14.0, "speedup": 0.93},
                {"K": 4, "acceptance_rate": 0.55, "tokens_per_sec": 16.0, "speedup": 1.07},
                {"K": 6, "acceptance_rate": 0.45, "tokens_per_sec": 15.0, "speedup": 1.00},
                {"K": 8, "acceptance_rate": 0.35, "tokens_per_sec": 13.0, "speedup": 0.87},
            ],
            "domain_summary": [],
        }
        with open("results.json", "w") as f:
            json.dump(test_results, f, indent=2)
        print("Created test results.json for verification.")

    generate_all_plots()
    print("plot.py verification PASSED.")
