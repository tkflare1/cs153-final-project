# Speculative Decoding: Benchmarking Draft Model Efficiency and Acceptance Rate Tradeoffs

Stanford CS153 — Spring 2026

## Overview

Speculative decoding accelerates LLM inference by using a small draft model to propose tokens that a larger target model verifies in one forward pass. The algorithm itself is well established (Chen et al. 2023, Leviathan et al. 2023). This project is not a reimplementation of the paper. The contribution is an empirical study of questions the original papers did not address:

1. **How does prompt domain affect acceptance rate?** Code, prose, and math have different token predictability. I measure whether structured domains like code give the draft model a systematic advantage.

2. **What is the optimal speculation length in practice?** The papers describe K as a hyperparameter but do not provide guidance on choosing it for real model pairings. I sweep K and show the tradeoff between acceptance rate and amortized verification cost.

3. **Where does time actually go?** I instrument the decode loop and break down latency into draft generation, target verification, and acceptance overhead to understand the bottleneck at each K value.

4. **How does the draft to target size ratio affect speedup?** I test multiple draft model sizes against the same target to find the ratio where speculative decoding stops being worth it.

## Results (Llama 3.2 1B / 3B, 4-bit, T4 GPU)

### K-value Sweep

Baseline autoregressive throughput: **3.71 tokens/sec**

| K | Acceptance Rate | Tokens/sec | Speedup |
|---|----------------|------------|---------|
| 1 | 75.2% | 4.54 | 1.22x |
| **2** | **67.7%** | **4.86** | **1.31x** |
| 4 | 58.3% | 4.78 | 1.29x |
| 6 | 44.0% | 4.01 | 1.08x |
| 8 | 43.2% | 4.02 | 1.08x |

### Domain Analysis

| Domain | Acceptance Rate | Tokens/sec | Speedup |
|--------|----------------|------------|---------|
| Code | 67.0% | 5.15 | 1.39x |
| Math | 56.4% | 4.11 | 1.11x |
| Prose | 49.6% | 4.06 | 1.09x |

### Latency Breakdown

Per-phase timing for the prose prompt ("Once upon a time..."), showing where wall-clock time is spent at each K:

| K | Draft (s) | Verify (s) | Accept (s) | Draft % | Verify % |
|---|-----------|------------|------------|---------|----------|
| 1 | 3.03 | 4.63 | 3.48 | 27% | 42% |
| 2 | 5.13 | 4.06 | 3.06 | 42% | 33% |
| 4 | 7.32 | 2.94 | 2.23 | 59% | 24% |
| 6 | 8.62 | 2.88 | 1.59 | 66% | 22% |
| 8 | 10.49 | 1.90 | 1.45 | 76% | 14% |

At low K, verification dominates. By K=8, drafting consumes 76% of total time on tokens that mostly get rejected.

### Key Findings

- **K=2 is optimal** for this model pairing (1.31x speedup). Higher K wastes draft compute because the 1B model is only 3x smaller than the 3B target.
- **Code gets 1.39x speedup** vs 1.09x for prose — structured syntax is more predictable for the draft model.
- **Draft time grows linearly with K** while verification time shrinks, explaining why there's a crossover point after which larger K hurts.
- The **latency breakdown** reveals the mechanism behind the optimal K: the crossover where drafting becomes the dominant cost falls between K=2 and K=4, matching the speedup peak.

### Plots

All plots and the interactive token trace are in `colab_results/`:

| File | Description |
|------|-------------|
| `speedup_vs_k.png` | Speedup curve showing the inverted-U shape with peak at K=2 |
| `acceptance_rate_vs_k.png` | Monotonic decline in acceptance rate from 75% to 43% |
| `latency_breakdown.png` | Stacked bar chart: absolute time per phase at each K |
| `latency_percentage.png` | Stacked bar chart: percentage breakdown per phase |
| `token_trace.html` | Interactive round-by-round visualization of accept/reject decisions |

## Results (Llama 3.2 1B / 3B, full precision, H100 GPU)

I re-ran the full benchmark in **full precision** (no quantization) on a Stanford H100. This is a different hardware/precision regime from the 4-bit T4 run above, and the results shift in an instructive way.

Baseline autoregressive throughput: **49.21 tokens/sec** (vs 3.71 on the 4-bit T4).

| K | Acceptance Rate | Tokens/sec | Speedup |
|---|----------------|------------|---------|
| **1** | **84.8%** | **57.54** | **1.17x** |
| 2 | 75.4% | 56.99 | 1.16x |
| 4 | 70.1% | 56.35 | 1.15x |
| 6 | 60.0% | 50.75 | 1.03x |
| 8 | 53.1% | 46.01 | 0.94x |

| Domain | Acceptance Rate | Tokens/sec | Speedup |
|--------|----------------|------------|---------|
| Code | 77.2% | 58.57 | 1.19x |
| Prose | 65.4% | 51.85 | 1.05x |
| Math | 63.3% | 50.16 | 1.02x |

**Cross-regime finding:** on the fast H100 in full precision, the optimal K drops to **K=1** and peak speedup is more modest (1.17x vs 1.31x on the T4), and K=8 is actually *slower* than baseline. When the target forward pass is cheap (fast GPU, no dequantization), per-call overhead dominates and speculating far ahead stops paying off. Code remains the best domain in both regimes. Full H100 outputs are in `h100_results/`.

## Completed Extensions

### KV Cache Integration
Added KV caching to both the baseline and speculative loops (`kv_cache.py`). The non-trivial part is **cache rollback on rejection**: the target verifies K speculative tokens but only the accepted prefix is committed, so the draft and target caches (which advance at different rates — the draft only forwards K−1 of its K proposals) are each cropped back to exactly what they processed, and the freshly resampled/bonus token is fed as the next round's input.

Correctness is verified by an **equivalence check** (`kv_compare.py`): with a fixed seed, the cached decoders produce **token-for-token identical** output to the non-cached versions, confirming the cache is lossless.

| Setting | Baseline KV speedup | Speculative KV speedup | Lossless |
|---------|--------------------|------------------------|----------|
| gpt2 / distilgpt2, CPU | 2.08x | 1.75x | PASS |
| Llama 3.2 1B/3B, H100 | 1.11x | 1.00x | PASS |

The cache helps far more on CPU (where recomputing the prefix is relatively expensive) and at longer sequences. At 50 tokens on an H100 the recomputed prefix is short, so the gain is small — and speculative decoding already does few forward passes, so it benefits least.

### Draft Model Size Sweep
Swept draft model size against a fixed target (`draft_sweep.py`). Speculative decoding requires the draft and target to share a tokenizer, so this uses the **Qwen2.5 family** (0.5B / 1.5B / 3B drafts → 7B target), all of which share one tokenizer. Run at K=4 on the H100; baseline 42.4 tok/s.

| Draft | Draft:Target ratio | Acceptance Rate | Speedup |
|-------|-------------------|----------------|---------|
| 0.5B | 14:1 | 64.2% | 0.94x |
| 1.5B | 4.7:1 | 59.1% | 0.80x |
| **3B** | **2.3:1** | **84.8%** | **1.17x** |

**Finding:** the relationship is **not** "smaller draft = better." The 3B draft (closest in size to the 7B target) wins decisively because its acceptance rate jumps to 85%. At batch-1 on an H100, the per-token latency of a 0.5B and a 3B model are similar (both overhead-bound, not FLOP-bound), so the cheaper draft saves little while accepting far less. The size ratio that maximizes speedup here is the *large* draft — the opposite of the naive intuition. Plot in `h100_results/draft_sweep.png`.

### Latency Breakdown Instrumentation
Instrumented the speculative decode loop to separately measure draft forward pass time, target verification time, and accept/reject overhead. Produces stacked bar charts (absolute and percentage) showing where time is spent at each K value. This reveals that at K=8, 76% of wall-clock time is spent on draft generation — most of which is wasted on rejected tokens.

### Token-Level Visualization
Built an interactive HTML visualization that shows each speculative round step by step: accepted tokens (green), rejected tokens (red with strikethrough), resampled tokens (yellow), and bonus tokens (blue). Each token displays the probability ratio used for the accept/reject decision. Makes the algorithm tangible beyond aggregate metrics.

## Project Structure

```
config.py                    Shared configuration (run mode, model names, K values)
models.py                    Shared model loading logic
speculative_decode.py        Core speculative sampling algorithm
speculative_decode_instrumented.py   Instrumented version with per-phase timing
baseline.py                  Manual autoregressive decoding loop
kv_cache.py                  KV-cached baseline + speculative decoders (with rollback)
kv_compare.py                KV losslessness check + cached-vs-uncached speedup
draft_sweep.py               Draft model size sweep (Qwen2.5 family)
benchmark.py                 Benchmarking harness, K sweeps, domain sweeps
plot.py                      Generates speedup and acceptance rate plots
visualize.py                 Latency breakdown plots and token trace HTML
main.py                      End to end runner
cluster/                     SLURM sbatch scripts for the H100 cluster
  smoke.sbatch                 GPU environment smoke test
  bench.sbatch                 Full benchmark (main.py)
  sweep.sbatch                 Draft-size sweep
  kv.sbatch                    KV-cache comparison
requirements.txt             Dependencies
writeup.md                   Full analysis and writeup
colab_results/               Results and plots from the 4-bit Colab T4 run
h100_results/                Results and plots from the full-precision H100 run
```

## Setup

### Local (CPU, small models)
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

### Colab (GPU, Llama models, 4-bit)
1. Upload all `.py` files and `requirements.txt`
2. Open `config.py`, change `USE_COLAB = False` to `USE_COLAB = True`
3. Run `!python main.py`

### Cluster (H100, full precision)
The `cluster/*.sbatch` scripts run inside the NGC PyTorch container and drive the
same code via env vars (no source edits). From the login pod:
```
sbatch cluster/bench.sbatch    # full benchmark  -> results.json + plots
sbatch cluster/sweep.sbatch    # draft-size sweep -> draft_sweep.json + png
sbatch cluster/kv.sbatch       # KV-cache study   -> kv_results.json
```
`USE_CLUSTER=1` selects full-precision Llama 3.2 1B/3B; `DRAFT_MODEL`/`TARGET_MODEL`
override the pair (the sweep uses Qwen2.5). See `CLAUDE.md` for the verified
container workflow (pinned `transformers==4.46.3`, HF token handling, etc.).

## References

- Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" https://arxiv.org/abs/2302.01318
- Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" https://arxiv.org/abs/2211.17192
