# Speculative Decoding: Benchmarking Draft Model Efficiency and Acceptance Rate Tradeoffs

Stanford CS153 — Spring 2026

## Overview

A from scratch implementation of speculative decoding for LLM inference based on Chen et al. 2023 and Leviathan et al. 2023. A small draft model proposes candidate token sequences. A larger target model verifies them in a single forward pass using rejection sampling. The output distribution is mathematically identical to sampling from the target model alone.

## What Has Been Implemented

### Core Algorithm
- Manual autoregressive baseline decoding (no model.generate)
- Full speculative sampling engine: draft generation, batched target verification, token acceptance/rejection via rejection sampling, residual distribution resampling, bonus token on full acceptance
- Seeded randomness for reproducibility

### Benchmarking Harness
- Sweep over speculation lengths K = 1, 2, 4, 6, 8
- 9 prompts across 3 domains: prose, code, math (3 each)
- Metrics: tokens per second, acceptance rate, wall clock time, effective speedup ratio
- Per domain summary with average acceptance rate and speedup

### Models Tested
- Local CPU: distilgpt2 (82M draft) / gpt2 (124M target)
- Colab T4 GPU: Llama 3.2 1B (draft) / Llama 3.2 3B (target), both 4 bit quantized via BitsAndBytes

### Results
- Best speedup: 1.30x at K=2 with 67.7% acceptance rate (Llama 1B/3B on T4)
- Code prompts had highest acceptance rate (67%) and speedup (1.37x)
- Prose had lowest acceptance rate (50%) due to less predictable token sequences
- Acceptance rate decreases monotonically with K as expected

### Outputs
- Results tables (K vs metrics, domain vs metrics)
- Two matplotlib plots: speedup vs K, acceptance rate vs K
- Full results saved to results.json

## Planned Extensions

### KV Cache Integration
Add KV caching to both the baseline and speculative decoding loops. Measure latency with and without caching. The nontrivial part is cache rollback on token rejection since accepted prefix caches must be preserved while rejected suffix caches are discarded.

### Latency Breakdown Analysis
Instrument the speculative decode loop to separately measure draft forward pass time, target verification time, and acceptance logic overhead. Produce stacked bar charts showing where time is spent for each K value. Measure GPU utilization during each phase.

### Token Level Visualization
Build a visualization that shows each speculative round step by step. Color accepted tokens green and rejected tokens red. Display the probability ratios (p_target / p_draft) at each position. Show when residual resampling occurs and what the bonus token is.

### Draft Model Size Sweep
Test multiple draft model sizes against the same target to find the optimal draft to target size ratio. For example Llama 3.2 1B and Llama 3.2 3B as drafts against a larger target. Produce tradeoff curves of speedup vs draft model size.

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
