# Speculative Decoding: Implementation and Empirical Analysis

**Stanford CS153 — Spring 2026**

---

## 1. Introduction

Speculative decoding (Chen et al., 2023) is a technique for accelerating autoregressive language model inference without changing the model's output distribution. The core insight is that a smaller, faster *draft model* can propose multiple candidate tokens cheaply, and a larger *target model* can verify all of them in a single forward pass — exploiting the parallelism of the attention mechanism during verification while bypassing the sequential bottleneck of token-by-token generation.

This project implements the full speculative decoding algorithm, benchmarks it against standard autoregressive decoding, and analyzes how performance varies across speculation lengths (K) and prompt domains.

## 2. Method

### 2.1 Baseline: Autoregressive Decoding

Standard autoregressive generation produces one token per forward pass. At each step:

1. Run the full model on the current sequence
2. Sample from the output distribution at the last position
3. Append the token and repeat

This is simple but inefficient: the target model's compute is fully utilized for just one token per pass.

### 2.2 Speculative Decoding Algorithm

Speculative decoding amortizes the cost of target model calls across multiple tokens:

1. **Draft phase**: The draft model autoregressively generates K candidate tokens, storing the probability distribution at each step.

2. **Verification phase**: All K candidates are fed to the target model in a single forward pass, yielding target-model probabilities for every position.

3. **Accept/reject phase**: For each candidate token i (left to right):
   - Compute the acceptance probability: `r = min(1, p_target[i] / p_draft[i])`
   - Accept with probability r; on acceptance, keep the token and move to i+1
   - On rejection, resample from the *residual distribution* `max(0, p_target - p_draft)`, discard all tokens after position i, and restart

4. **Bonus token**: If all K tokens are accepted, sample one additional token from the target model's distribution at position K+1 (which is available for free from the verification pass).

This procedure is *provably lossless* — the output distribution is identical to sampling directly from the target model — because the residual resampling exactly corrects for the mismatch between draft and target distributions.

### 2.3 Models

| | Model | Parameters | Quantization |
|---|---|---|---|
| Draft | Llama 3.2 1B | 1.24B | 4-bit (BitsAndBytes) |
| Target | Llama 3.2 3B | 3.21B | 4-bit (BitsAndBytes) |

Both models were run on a Google Colab T4 GPU (16 GB VRAM). The 4-bit quantization via BitsAndBytes allowed both models to fit comfortably in GPU memory.

## 3. Experimental Setup

### 3.1 Prompts

We tested on 9 prompts across 3 domains (3 prompts each):

- **Prose**: Creative/narrative writing (e.g., "Once upon a time in a distant kingdom...")
- **Code**: Python programming (e.g., "def fibonacci(n):...")
- **Math**: Mathematical reasoning (e.g., "To prove that the square root of 2 is irrational...")

### 3.2 Parameters

- **Speculation length (K)**: Swept over {1, 2, 4, 6, 8}
- **Max new tokens**: 50 per generation
- **Temperature**: 1.0
- **Random seed**: 42 (fixed for reproducibility)

### 3.3 Metrics

- **Tokens per second**: Wall-clock throughput (new tokens / elapsed time)
- **Acceptance rate**: Fraction of draft tokens accepted by the target model (accepted / proposed)
- **Speedup**: Ratio of speculative tok/sec to baseline tok/sec
- **Wall-clock time**: Total generation time per prompt

## 4. Results

### 4.1 Baseline Performance

The target model (Llama 3.2 3B, 4-bit) achieved an average baseline throughput of **3.53 tokens/sec** across all 9 prompts on the T4 GPU.

### 4.2 Speculation Length (K) Sweep

| K | Acceptance Rate | Tokens/sec | Speedup | Avg Time (s) |
|---|----------------|------------|---------|---------------|
| 1 | 0.7519 | 4.35 | 1.23x | 11.69 |
| **2** | **0.6773** | **4.58** | **1.30x** | **11.13** |
| 4 | 0.5828 | 4.43 | 1.26x | 11.80 |
| 6 | 0.4400 | 3.78 | 1.07x | 14.07 |
| 8 | 0.4318 | 3.81 | 1.08x | 13.94 |

**K=2 is the optimal speculation length**, achieving 1.30x speedup with a 67.7% acceptance rate.

### 4.3 Domain Analysis

| Domain | Acceptance Rate | Avg Tokens/sec | Speedup |
|--------|----------------|----------------|---------|
| Prose | 0.4959 | 3.86 | 1.09x |
| **Code** | **0.6701** | **4.85** | **1.37x** |
| Math | 0.5643 | 3.87 | 1.10x |

**Code prompts yield the highest speedup (1.37x)** with a 67% acceptance rate.

## 5. Analysis

### 5.1 Why Does Acceptance Rate Decrease with K?

The acceptance rate drops monotonically from 75.2% (K=1) to 43.2% (K=8). This is expected: the acceptance rate reported is the *average per-token* acceptance rate across all proposed tokens. With larger K, the draft model must predict further into the future, and later tokens in the K-sequence are conditioned on earlier draft tokens that may themselves diverge from what the target model would have produced. Even if the first few tokens are likely correct, the last few are increasingly speculative, dragging down the per-token average.

### 5.2 Why Does Speedup Peak at K=2?

Speedup is determined by the balance of two opposing forces:

- **Benefit of K**: More tokens accepted per verification pass → fewer expensive target model calls
- **Cost of K**: More draft model forward passes per round, and higher rejection probability means more wasted draft computation

At K=1, each round costs 1 draft pass + 1 target pass, and we accept ~75% of proposals. The speedup is 1.23x.

At K=2, each round costs 2 draft passes + 1 target pass. We accept ~68% per token, meaning on average we get ~1.36 tokens per target call (plus the bonus token when both accepted). This is the sweet spot where the extra draft call pays for itself.

At K=4+, the draft model forward passes start to dominate the cost. The Llama 1B model is only ~3x smaller than the 3B target, so each draft forward pass is a significant fraction (~1/3) of a target pass. By K=6, we're spending 6 draft passes (≈ 2 target-equivalent passes) to get only ~2.2 accepted tokens per round, which is worse than K=2.

**Key insight**: The optimal K depends critically on the *cost ratio* between draft and target models. With a much cheaper draft (e.g., a 100M model drafting for a 70B target), higher K values would be optimal because each draft pass is nearly free relative to verification.

### 5.3 Why Does Code Have the Highest Acceptance Rate?

Code prompts achieved 67% acceptance vs. 50% for prose and 56% for math. This reflects the predictability structure of each domain:

- **Code** is highly structured with predictable syntax patterns. Indentation, closing brackets, common idioms (`return`, `self.`, `for i in range`), and API patterns are well-learned even by the smaller 1B model. When the next tokens are syntactically constrained, the 1B and 3B models are likely to agree.

- **Math** falls in between — it has structured notation (equations, symbols) but also requires reasoning about proof strategies where the models may diverge.

- **Prose** is the least predictable. Creative writing involves subjective word choices where even a small distributional difference between models leads to frequent rejections. The 1B model may favor different adjectives, sentence structures, or plot directions than the 3B model.

### 5.4 Cost-Benefit Perspective

On the T4 with Llama 1B/3B:
- Baseline: 3.53 tok/sec
- Best speculative: 4.58 tok/sec (K=2)
- **Net speedup: 1.30x (30% faster)**

This translates to generating 50 tokens in **11.1s** instead of **14.2s** — a saving of ~3 seconds per generation. For batch workloads processing thousands of prompts, this compounds to significant wall-clock savings.

### 5.5 Comparison: GPU vs CPU Results

We also ran the same benchmark locally on CPU with smaller models (distilgpt2 82M / gpt2 124M):

| Setting | Best K | Best Speedup | Best Acceptance Rate |
|---------|--------|-------------|---------------------|
| CPU (distilgpt2/gpt2) | 1 | 1.08x | 0.72 |
| GPU (Llama 1B/3B) | 2 | 1.30x | 0.68 |

On CPU, speculative decoding barely helps because:
1. The draft model (82M) is only ~1.5x smaller than the target (124M), so draft passes are almost as expensive as target passes
2. CPU execution doesn't benefit as much from the batched verification pass (less parallelism)

On GPU, the speedup is substantially better because the T4's parallelism makes the single-pass verification of K tokens nearly as fast as a single-token forward pass, amplifying the benefit of each accepted token.

## 6. Limitations and Future Work

1. **KV-cache optimization**: Our implementation recomputes attention over the full sequence at each step. A production implementation would use KV-caching to avoid redundant computation, which would increase both baseline and speculative throughput but likely improve speculative decoding's relative advantage.

2. **Adaptive K**: Rather than a fixed K, one could dynamically adjust the speculation length based on the running acceptance rate — speculate more aggressively when the draft model is performing well and pull back when rejections are frequent.

3. **Larger model gaps**: The speedup benefits would be more pronounced with a greater size gap between draft and target (e.g., 1B drafting for 70B). Our 1B/3B pair has a modest 3x size ratio, limiting the achievable speedup.

4. **Tree-structured speculation**: Instead of a single draft sequence, the draft model could explore a tree of possible continuations, allowing the target model to verify multiple alternative paths in one pass (Medusa, SpecInfer).

5. **Batch inference**: Our benchmark tests single-sequence generation. In high-throughput serving scenarios, the interplay between speculative decoding and continuous batching introduces additional trade-offs.

## 7. Conclusion

We implemented and benchmarked speculative decoding using Llama 3.2 1B (draft) and 3B (target) on a T4 GPU. The optimal configuration (K=2) achieved a **1.30x speedup** over standard autoregressive decoding with a **67.7% acceptance rate**, while producing output from an identical distribution. Code generation benefited most (1.37x speedup, 67% acceptance), consistent with the higher predictability of structured programming text.

The results confirm the theoretical prediction: speculative decoding provides a genuine free lunch — faster inference with no quality degradation — when the draft-to-target cost ratio is favorable and the draft model is a reasonable approximation of the target.

## References

- Chen, C., Borgeaud, S., Irving, G., Lespiau, J.-B., Sifre, L., & Kramber, J. (2023). *Accelerating Large Language Model Decoding with Speculative Sampling*. arXiv:2302.01318.
- Leviathan, Y., Kalman, M., & Matias, Y. (2023). *Fast Inference from Transformers via Speculative Decoding*. ICML 2023.
