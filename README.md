# Speculative Decoding: Benchmarking Draft Model Efficiency and Acceptance Rate Tradeoffs

Stanford CS153 — Spring 2026

## Overview

Speculative decoding accelerates LLM inference by using a small draft model to propose tokens that a larger target model verifies in one forward pass. The algorithm itself is well established (Chen et al. 2023, Leviathan et al. 2023). This project is not a reimplementation of the paper. The contribution is an empirical study of questions the original papers did not address:

1. **How does prompt domain affect acceptance rate?** Code, prose, and math have different token predictability. I measure whether structured domains like code give the draft model a systematic advantage.

2. **What is the optimal speculation length in practice?** The papers describe K as a hyperparameter but do not provide guidance on choosing it for real model pairings. I sweep K and show the tradeoff between acceptance rate and amortized verification cost.

3. **Where does time actually go?** I break down latency into draft generation, target verification, and acceptance overhead to understand the bottleneck at each K value.

4. **How does the draft to target size ratio affect speedup?** I test multiple draft model sizes against the same target to find the ratio where speculative decoding stops being worth it.

## Current Results

Using Llama 3.2 1B (draft) and Llama 3.2 3B (target), both 4 bit quantized, on a T4 GPU:

| K | Acceptance Rate | Tokens/sec | Speedup |
|---|----------------|------------|---------|
| 1 | 75.2% | 4.35 | 1.23x |
| 2 | 67.7% | 4.58 | 1.30x |
| 4 | 58.3% | 4.43 | 1.26x |
| 6 | 44.0% | 3.78 | 1.07x |
| 8 | 43.2% | 3.81 | 1.08x |

| Domain | Acceptance Rate | Speedup |
|--------|----------------|---------|
| Code | 67.0% | 1.37x |
| Math | 56.4% | 1.10x |
| Prose | 49.6% | 1.09x |

Key findings so far:
- K=2 is optimal for this model pairing. Higher K wastes draft compute because the 1B model is only 3x smaller than the 3B target.
- Code has significantly higher acceptance than prose (67% vs 50%) because token sequences are more syntactically constrained.
- The speedup curve peaks early and declines, suggesting the optimal K depends heavily on the draft to target cost ratio.

## Planned Extensions

### KV Cache Integration
Add KV caching to both baseline and speculative loops. The nontrivial part is cache rollback on rejection: accepted prefix caches must be preserved while rejected suffix caches are discarded. Compare latency with and without caching.

### Latency Breakdown
Instrument the decode loop to separately measure draft forward pass time, target verification time, and acceptance overhead. Produce stacked bar charts showing where time goes at each K value. Measure GPU utilization during each phase.

### Token Level Visualization
Visualize each speculative round step by step. Show accepted tokens, rejected tokens, probability ratios at each position, and residual resampling events. Makes the algorithm tangible beyond aggregate metrics.

### Draft Model Size Sweep
Test multiple draft sizes against the same target to find the optimal draft to target size ratio. Produce tradeoff curves showing when speculative decoding stops being worth it.

## Project Structure

```
speculative_decode.py    Core speculative sampling algorithm
baseline.py              Manual autoregressive decoding loop
benchmark.py             Benchmarking harness, K sweeps, domain sweeps
plot.py                  Generates figures from results.json
main.py                  End to end runner (USE_COLAB flag for GPU mode)
requirements.txt         Dependencies
writeup.md               Full analysis and writeup
colab_results/           Results from Colab T4 GPU runs
```

## Setup

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

For Colab set USE_COLAB = True in main.py and authenticate with your HuggingFace token.

## References

- Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" https://arxiv.org/abs/2302.01318
- Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" https://arxiv.org/abs/2211.17192
