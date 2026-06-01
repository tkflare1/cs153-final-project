"""
Visualization Module
Generates latency breakdown charts and token-level HTML visualizations
from instrumented speculative decoding runs.
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# =========================================================================
# 1. Latency Breakdown — Stacked Bar Chart per K
# =========================================================================

def plot_latency_breakdown(latency_by_k, save_path="latency_breakdown.png"):
    """
    Stacked bar chart showing draft/verify/accept time per K value.

    Args:
        latency_by_k: list of dicts with keys: K, draft_time, verify_time, accept_time
    """
    ks = [r["K"] for r in latency_by_k]
    draft = [r["draft_time"] for r in latency_by_k]
    verify = [r["verify_time"] for r in latency_by_k]
    accept = [r["accept_time"] for r in latency_by_k]

    x = np.arange(len(ks))
    width = 0.5

    fig, ax = plt.subplots(figsize=(9, 5))
    p1 = ax.bar(x, draft, width, label="Draft generation", color="#3b82f6")
    p2 = ax.bar(x, verify, width, bottom=draft, label="Target verification", color="#ef4444")
    p3 = ax.bar(x, accept, width,
                bottom=[d + v for d, v in zip(draft, verify)],
                label="Accept/reject logic", color="#22c55e")

    ax.set_xlabel("Speculation Length (K)", fontsize=13)
    ax.set_ylabel("Time (seconds)", fontsize=13)
    ax.set_title("Latency Breakdown by Phase", fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Percentage labels inside bars
    for i in range(len(ks)):
        total = draft[i] + verify[i] + accept[i]
        if total > 0:
            ax.text(x[i], draft[i] / 2, f"{draft[i]/total*100:.0f}%",
                    ha="center", va="center", fontsize=10, fontweight="bold", color="white")
            ax.text(x[i], draft[i] + verify[i] / 2, f"{verify[i]/total*100:.0f}%",
                    ha="center", va="center", fontsize=10, fontweight="bold", color="white")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_latency_percentage(latency_by_k, save_path="latency_percentage.png"):
    """100% stacked bar showing relative time proportions per K."""
    ks = [r["K"] for r in latency_by_k]
    totals = [r["draft_time"] + r["verify_time"] + r["accept_time"] for r in latency_by_k]
    draft_pct = [r["draft_time"] / t * 100 if t > 0 else 0 for r, t in zip(latency_by_k, totals)]
    verify_pct = [r["verify_time"] / t * 100 if t > 0 else 0 for r, t in zip(latency_by_k, totals)]
    accept_pct = [r["accept_time"] / t * 100 if t > 0 else 0 for r, t in zip(latency_by_k, totals)]

    x = np.arange(len(ks))
    width = 0.5

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, draft_pct, width, label="Draft generation", color="#3b82f6")
    ax.bar(x, verify_pct, width, bottom=draft_pct, label="Target verification", color="#ef4444")
    ax.bar(x, accept_pct, width,
           bottom=[d + v for d, v in zip(draft_pct, verify_pct)],
           label="Accept/reject logic", color="#22c55e")

    ax.set_xlabel("Speculation Length (K)", fontsize=13)
    ax.set_ylabel("Percentage of Total Time", fontsize=13)
    ax.set_title("Latency Breakdown (Relative)", fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


# =========================================================================
# 2. Token-Level HTML Visualization
# =========================================================================

def generate_token_html(trace, prompt_text, output_text, K, acceptance_rate,
                        save_path="token_trace.html"):
    """
    Generate an HTML file showing each speculative round with color-coded tokens.

    Green = accepted, Red = rejected (with resampled replacement shown),
    Blue = bonus token from target model.
    """
    css = """
    <style>
        body { font-family: 'Segoe UI', system-ui, sans-serif; max-width: 900px;
               margin: 40px auto; padding: 0 20px; background: #fafafa; color: #1a1a1a; }
        h1 { font-size: 22px; border-bottom: 2px solid #e5e5e5; padding-bottom: 10px; }
        h2 { font-size: 16px; color: #555; margin-top: 30px; }
        .meta { background: #f0f0f0; padding: 12px 16px; border-radius: 8px;
                font-size: 14px; margin-bottom: 20px; }
        .round { background: white; border: 1px solid #e0e0e0; border-radius: 8px;
                 padding: 14px 18px; margin-bottom: 12px; }
        .round-header { font-weight: 600; font-size: 14px; color: #444; margin-bottom: 8px; }
        .token { display: inline-block; padding: 3px 6px; margin: 2px; border-radius: 4px;
                 font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px; }
        .accepted { background: #dcfce7; border: 1px solid #86efac; color: #166534; }
        .rejected { background: #fee2e2; border: 1px solid #fca5a5; color: #991b1b;
                    text-decoration: line-through; }
        .resampled { background: #fef3c7; border: 1px solid #fcd34d; color: #92400e; }
        .bonus { background: #dbeafe; border: 1px solid #93c5fd; color: #1e40af; }
        .prob { font-size: 11px; color: #888; margin-left: 2px; }
        .legend { display: flex; gap: 16px; margin-bottom: 16px; font-size: 13px; }
        .legend span { padding: 2px 8px; border-radius: 4px; }
        .arrow { color: #999; margin: 0 2px; }
    </style>
    """

    legend = """
    <div class="legend">
        <span class="token accepted">Accepted</span>
        <span class="token rejected">Rejected</span>
        <span class="token resampled">Resampled</span>
        <span class="token bonus">Bonus</span>
    </div>
    """

    rounds_html = ""
    for r in trace:
        tokens_html = ""
        for t in r["tokens"]:
            if t["accepted"]:
                tokens_html += (
                    f'<span class="token accepted">'
                    f'{_esc(t["token_text"])}'
                    f'<span class="prob"> {t["ratio"]:.2f}</span></span> '
                )
            else:
                tokens_html += (
                    f'<span class="token rejected">'
                    f'{_esc(t["token_text"])}'
                    f'<span class="prob"> {t["ratio"]:.2f}</span></span>'
                )
                if "resampled_text" in t:
                    tokens_html += (
                        f'<span class="arrow">&rarr;</span>'
                        f'<span class="token resampled">'
                        f'{_esc(t["resampled_text"])}</span> '
                    )

        if r.get("bonus_token"):
            tokens_html += (
                f'<span class="token bonus">'
                f'{_esc(r["bonus_token"]["token_text"])} (bonus)</span>'
            )

        n_acc = sum(1 for t in r["tokens"] if t["accepted"])
        n_tok = len(r["tokens"])
        rounds_html += f"""
        <div class="round">
            <div class="round-header">Round {r['round']}
                &mdash; {n_acc}/{n_tok} accepted</div>
            {tokens_html}
        </div>
        """

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Speculative Decoding Token Trace</title>
{css}</head><body>
<h1>Speculative Decoding: Token-Level Trace</h1>
<div class="meta">
    <strong>Prompt:</strong> {_esc(prompt_text)}<br>
    <strong>K:</strong> {K} &nbsp;|&nbsp;
    <strong>Acceptance Rate:</strong> {acceptance_rate:.1%} &nbsp;|&nbsp;
    <strong>Rounds:</strong> {len(trace)}
</div>
{legend}
<h2>Generation Rounds</h2>
{rounds_html}
<h2>Full Output</h2>
<div class="round" style="white-space:pre-wrap; font-family:monospace; font-size:13px;">
{_esc(output_text)}</div>
</body></html>"""

    with open(save_path, "w") as f:
        f.write(html)
    print(f"Saved: {save_path}")


def _esc(text):
    """Escape HTML special characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# =========================================================================
# 3. Run full instrumented benchmark across K values
# =========================================================================

def run_latency_sweep(draft_model, target_model, tokenizer, prompt, k_values, max_new_tokens=50):
    """
    Run instrumented speculative decoding for each K value on a single prompt.
    Returns latency_by_k list and per-K traces.
    """
    import torch
    import numpy as np
    from config import SEED
    from speculative_decode_instrumented import speculative_decode_instrumented

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    device = next(draft_model.parameters()).device
    input_ids = input_ids.to(device)

    latency_by_k = []
    all_traces = {}

    for K in k_values:
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        generated, accepted, proposed, timing, trace = speculative_decode_instrumented(
            draft_model, target_model, tokenizer, input_ids,
            K=K, max_new_tokens=max_new_tokens,
        )

        output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        acc_rate = accepted / proposed if proposed > 0 else 0.0

        latency_by_k.append({
            "K": K,
            "draft_time": round(timing["draft_time"], 4),
            "verify_time": round(timing["verify_time"], 4),
            "accept_time": round(timing["accept_time"], 4),
            "acceptance_rate": round(acc_rate, 4),
        })

        all_traces[K] = {
            "trace": trace,
            "output_text": output_text,
            "acceptance_rate": acc_rate,
        }

    return latency_by_k, all_traces


# ---------------------------------------------------------------------------
# Standalone verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from models import load_models

    print("Loading models for visualization verification...")
    draft_model, target_model, tokenizer = load_models()

    prompt = "Once upon a time in a distant kingdom, there lived a"
    k_values = [1, 2, 4, 6, 8]

    print(f"Prompt: {prompt}")
    print(f"Sweeping K = {k_values} ...")

    latency_by_k, all_traces = run_latency_sweep(
        draft_model, target_model, tokenizer, prompt, k_values, max_new_tokens=30
    )

    # Print latency table
    print("\n--- Latency Breakdown ---")
    print(f"{'K':>4}  {'Draft':>8}  {'Verify':>8}  {'Accept':>8}  {'Total':>8}  {'Draft%':>7}  {'Verify%':>7}")
    print("-" * 60)
    for r in latency_by_k:
        total = r["draft_time"] + r["verify_time"] + r["accept_time"]
        d_pct = r["draft_time"] / total * 100 if total > 0 else 0
        v_pct = r["verify_time"] / total * 100 if total > 0 else 0
        print(f"{r['K']:>4}  {r['draft_time']:>8.3f}  {r['verify_time']:>8.3f}  "
              f"{r['accept_time']:>8.3f}  {total:>8.3f}  {d_pct:>6.1f}%  {v_pct:>6.1f}%")

    # Generate plots
    plot_latency_breakdown(latency_by_k)
    plot_latency_percentage(latency_by_k)

    # Generate HTML for K=4
    k4 = all_traces[4]
    generate_token_html(
        k4["trace"], prompt, k4["output_text"],
        K=4, acceptance_rate=k4["acceptance_rate"],
    )

    # Save latency data
    with open("latency_breakdown.json", "w") as f:
        json.dump(latency_by_k, f, indent=2)
    print("Saved: latency_breakdown.json")

    print("\nvisualize.py verification PASSED.")
