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

I tested on 9 prompts across 3 domains (3 prompts each):

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

The target model (Llama 3.2 3B, 4-bit) achieved an average baseline throughput of **3.71 tokens/sec** across all 9 prompts on the T4 GPU.

### 4.2 Speculation Length (K) Sweep

| K | Acceptance Rate | Tokens/sec | Speedup | Avg Time (s) |
|---|----------------|------------|---------|---------------|
| 1 | 0.7519 | 4.54 | 1.22x | 11.20 |
| **2** | **0.6773** | **4.86** | **1.31x** | **10.48** |
| 4 | 0.5828 | 4.78 | 1.29x | 10.88 |
| 6 | 0.4400 | 4.01 | 1.08x | 13.28 |
| 8 | 0.4318 | 4.02 | 1.08x | 13.21 |

**K=2 is the optimal speculation length**, achieving 1.31x speedup with a 67.7% acceptance rate.

### 4.3 Domain Analysis

| Domain | Acceptance Rate | Avg Tokens/sec | Speedup |
|--------|----------------|----------------|---------|
| Prose | 0.4959 | 4.06 | 1.09x |
| **Code** | **0.6701** | **5.15** | **1.39x** |
| Math | 0.5643 | 4.11 | 1.11x |

**Code prompts yield the highest speedup (1.39x)** with a 67% acceptance rate.

### 4.4 Latency Breakdown by Phase

I instrumented the speculative decode loop to separately measure time spent in each phase. The following table shows absolute wall-clock time for each phase on the prose prompt ("Once upon a time..."):

| K | Draft (s) | Verify (s) | Accept (s) | Total (s) | Draft % | Verify % |
|---|-----------|------------|------------|-----------|---------|----------|
| 1 | 3.03 | 4.63 | 3.48 | 11.13 | 27% | 42% |
| 2 | 5.13 | 4.06 | 3.06 | 12.25 | 42% | 33% |
| 4 | 7.32 | 2.94 | 2.23 | 12.49 | 59% | 24% |
| 6 | 8.62 | 2.88 | 1.59 | 13.09 | 66% | 22% |
| 8 | 10.49 | 1.90 | 1.45 | 13.84 | 76% | 14% |

At K=1, verification (the target model forward pass) dominates at 42% of total time. As K increases, draft generation grows linearly — each round runs K sequential draft passes — while verification time shrinks because fewer rounds are needed. By K=8, drafting consumes 76% of wall-clock time, and most of those draft tokens are rejected (only 41% acceptance). This directly explains the speedup peak: the crossover point where drafting becomes the dominant cost falls between K=2 and K=4, precisely where speedup is maximized.

### 4.5 Token-Level Trace (K=4)

An interactive visualization (`colab_results/token_trace.html`) shows the round-by-round accept/reject decisions for the K=4 prose generation. Key observations:

- **Full acceptance rounds** (4/4 tokens accepted) occurred on predictable phrases: "of the", "He was a", "for he was", "the tongue and the sword". These are syntactically constrained continuations where the 1B draft model matches the 3B target.
- **Immediate rejection rounds** (0/1 accepted) occurred when the draft model chose a plausible but wrong word (e.g., "whom" vs "the", "sword" vs "line"). The target model rejected at position 0 and resampled.
- **Partial acceptance rounds** (e.g., 2/3) show the cascade effect: once a rejection occurs, all subsequent tokens in the draft are discarded regardless of quality.
- **Probability ratios** beside each token quantify the mismatch: ratios near 1.0 indicate strong agreement (e.g., "prince" at 1.00), while low ratios indicate divergence (e.g., "law" at 0.05).

### 4.6 Full-Precision H100 Run (Cross-Regime Comparison)

The results above use 4-bit quantization on a T4. I re-ran the identical benchmark in **full precision (no quantization)** on a Stanford H100 to test how the conclusions hold in a different hardware/precision regime. Baseline throughput rose to **49.21 tokens/sec** (vs 3.71 on the 4-bit T4).

| K | Acceptance Rate | Tokens/sec | Speedup |
|---|----------------|------------|---------|
| **1** | **0.848** | **57.54** | **1.17x** |
| 2 | 0.754 | 56.99 | 1.16x |
| 4 | 0.701 | 56.35 | 1.15x |
| 6 | 0.600 | 50.75 | 1.03x |
| 8 | 0.531 | 46.01 | 0.94x |

| Domain | Acceptance Rate | Tokens/sec | Speedup |
|--------|----------------|------------|---------|
| **Code** | **0.772** | **58.57** | **1.19x** |
| Prose | 0.654 | 51.85 | 1.05x |
| Math | 0.633 | 50.16 | 1.02x |

Two things change and one stays the same:

- **The optimal K drops to 1** (from 2 on the T4), and the peak speedup is smaller (1.17x vs 1.31x). K=8 is now *slower than baseline* (0.94x). In full precision on a fast GPU, the target forward pass is cheap and the fixed per-call overhead of drafting dominates sooner, so speculating far ahead is counterproductive.
- **Acceptance rates are higher across the board** (e.g., 84.8% at K=1 vs 75.2% on the T4). 4-bit quantization perturbs the draft and target distributions unequally, lowering agreement; full precision restores it.
- **Code is still the most favorable domain** in both regimes (1.19x here, 1.39x on the T4), confirming the domain finding is robust to precision.

The key takeaway is that the *optimal speculation length is hardware- and precision-dependent*, not an intrinsic property of the model pair. This is exactly the kind of practical guidance the original papers leave open.

### 4.7 Extension: KV-Cache Integration

The baseline and speculative loops in Sections 2.1–2.2 recompute attention over the full sequence every step. I added KV-cached variants of both (`kv_cache.py`). The interesting part is **cache rollback on rejection**: the target verifies K speculative tokens in one pass and extends its cache by K, but only the accepted prefix is committed. Worse, the draft and target caches advance at *different* rates — the draft only forwards K−1 of its K proposals (the last proposed token's logits are never needed by the draft itself). I therefore track the two cache lengths separately and crop each back to exactly the positions it processed, feeding the freshly resampled/bonus token (which neither model has seen) as the next round's input.

**Correctness.** Because the cache changes only *how* logits are computed, not their values, a cached run with a fixed seed must reproduce the non-cached output exactly. `kv_compare.py` checks this token-for-token and confirms **losslessness (PASS)** for both decoders on both gpt2 (CPU) and Llama 3.2 (H100).

| Setting | Baseline KV speedup | Speculative KV speedup | Lossless |
|---------|--------------------|------------------------|----------|
| gpt2 / distilgpt2, CPU | 2.08x | 1.75x | PASS |
| Llama 3.2 1B/3B, H100 | 1.11x | 1.00x | PASS |

The cache helps dramatically on CPU, where recomputing the prefix is relatively expensive, but only modestly on the H100 at 50 tokens — the recomputed prefix is short, and speculative decoding already issues few forward passes, so it has the least to gain. The benefit would grow with longer generations.

### 4.8 Extension: Draft Model Size Sweep

To find the draft-to-target size ratio where speculation stops paying off, I swept draft size against a fixed target (`draft_sweep.py`). Since speculative decoding requires a shared tokenizer, I used the **Qwen2.5 family** (0.5B / 1.5B / 3B drafts against a 7B target — all share one tokenizer), at K=4 on the H100. Baseline (Qwen2.5-7B): 42.4 tok/s.

| Draft | Draft:Target ratio | Acceptance Rate | Speedup |
|-------|-------------------|----------------|---------|
| 0.5B | 14:1 | 0.642 | 0.94x |
| 1.5B | 4.7:1 | 0.591 | 0.80x |
| **3B** | **2.3:1** | **0.848** | **1.17x** |

The result is counter-intuitive: **the largest draft wins**, not the smallest. The naive expectation is that a tiny draft is "nearly free" and should maximize speedup, but only the 3B draft beats baseline. Two effects combine: (1) the 3B draft's acceptance rate is far higher (85% vs 64%), and (2) at batch size 1 on an H100, per-token latency is dominated by fixed kernel-launch and memory overhead rather than FLOPs, so a 0.5B model is *not* meaningfully cheaper per token than a 3B one. The cheap draft therefore saves almost nothing while accepting much less. The optimal ratio is regime-dependent: in a FLOP-bound setting (huge target, large batch) the tiny draft would win, but in this latency-bound batch-1 setting the larger, more accurate draft is better.

### 4.9 Extension: Long-Context Benchmark

The standard benchmark uses `max_new_tokens=50`. I re-ran the full K sweep at **50 and 200** tokens on the H100 (`adaptive_long_experiments.py`) to test whether speculative gains grow with generation length.

| max_new_tokens | Baseline (tok/s) | Best K | Best speedup |
|----------------|------------------|--------|--------------|
| 50 | 48.8 | 1 | 1.19x |
| 200 | 51.3 | 2 | **1.50x** |

At 200 tokens, even K=1 reaches **1.37x**, and K=2 peaks at **1.50x** (77.0 tok/s). This reconciles with the KV-cache finding: when the sequence is longer, amortizing target verification across more accepted tokens pays off more, and the optimal fixed K moves back toward 2 — closer to the T4 regime than the short 50-token H100 run where K=1 won.

### 4.10 Extension: Adaptive-K Policy

Instead of a fixed K, I implemented a simple online controller (`run_adaptive_speculative_benchmark` in `benchmark.py`): start at K=2; after each prompt, if acceptance > 0.80 increase K by 1, if < 0.60 decrease by 1; clamp to [1, 8]. Evaluated at `max_new_tokens=200` on the H100.

| Policy | Avg tok/s | Speedup vs baseline |
|--------|-----------|---------------------|
| Baseline | 72.3 | 1.00x |
| Adaptive-K | 77.8 | 1.08x |

The K trace was `[2, 2, 2, 2, 2, 3, 4, 4, 4]` — the controller correctly ramped speculation on prompts where the draft stayed aligned with the target. Adaptive-K beat baseline but did not match the best **fixed** K=2 at 200 tokens (1.50x). The gap suggests room for smoother control (e.g., exponential moving average of acceptance, per-domain K, or caps tied to draft/target latency ratio).

## 5. Analysis

### 5.1 Why Does Acceptance Rate Decrease with K?

The acceptance rate drops monotonically from 75.2% (K=1) to 43.2% (K=8). This is expected: the acceptance rate reported is the *average per-token* acceptance rate across all proposed tokens. With larger K, the draft model must predict further into the future, and later tokens in the K-sequence are conditioned on earlier draft tokens that may themselves diverge from what the target model would have produced. Even if the first few tokens are likely correct, the last few are increasingly speculative, dragging down the per-token average.

### 5.2 Why Does Speedup Peak at K=2?

Speedup is determined by the balance of two opposing forces:

- **Benefit of K**: More tokens accepted per verification pass → fewer expensive target model calls
- **Cost of K**: More draft model forward passes per round, and higher rejection probability means more wasted draft computation

At K=1, each round costs 1 draft pass + 1 target pass, and we accept ~75% of proposals. The speedup is 1.23x.

At K=2, each round costs 2 draft passes + 1 target pass. The system accepts ~68% per token, meaning on average ~1.36 tokens per target call (plus the bonus token when both are accepted). This is the sweet spot where the extra draft call pays for itself.

At K=4+, the draft model forward passes start to dominate the cost. The Llama 1B model is only ~3x smaller than the 3B target, so each draft forward pass is a significant fraction (~1/3) of a target pass. By K=6, we're spending 6 draft passes (≈ 2 target-equivalent passes) to get only ~2.2 accepted tokens per round, which is worse than K=2.

**Key insight**: The optimal K depends critically on the *cost ratio* between draft and target models. With a much cheaper draft (e.g., a 100M model drafting for a 70B target), higher K values would be optimal because each draft pass is nearly free relative to verification.

### 5.3 Why Does Code Have the Highest Acceptance Rate?

Code prompts achieved 67% acceptance vs. 50% for prose and 56% for math. This reflects the predictability structure of each domain:

- **Code** is highly structured with predictable syntax patterns. Indentation, closing brackets, common idioms (`return`, `self.`, `for i in range`), and API patterns are well-learned even by the smaller 1B model. When the next tokens are syntactically constrained, the 1B and 3B models are likely to agree.

- **Math** falls in between — it has structured notation (equations, symbols) but also requires reasoning about proof strategies where the models may diverge.

- **Prose** is the least predictable. Creative writing involves subjective word choices where even a small distributional difference between models leads to frequent rejections. The 1B model may favor different adjectives, sentence structures, or plot directions than the 3B model.

### 5.4 Cost-Benefit Perspective

On the T4 with Llama 1B/3B:
- Baseline: 3.71 tok/sec
- Best speculative: 4.86 tok/sec (K=2)
- **Net speedup: 1.31x (31% faster)**

This translates to generating 50 tokens in **10.5s** instead of **13.5s** — a saving of ~3 seconds per generation. For batch workloads processing thousands of prompts, this compounds to significant wall-clock savings.

### 5.5 Comparison: GPU vs CPU Results

I also ran the same benchmark locally on CPU with smaller models (distilgpt2 82M / gpt2 124M):

| Setting | Best K | Best Speedup | Best Acceptance Rate |
|---------|--------|-------------|---------------------|
| CPU (distilgpt2/gpt2) | 1 | 1.08x | 0.72 |
| GPU (Llama 1B/3B) | 2 | 1.30x | 0.68 |

On CPU, speculative decoding barely helps because:
1. The draft model (82M) is only ~1.5x smaller than the target (124M), so draft passes are almost as expensive as target passes
2. CPU execution doesn't benefit as much from the batched verification pass (less parallelism)

On GPU, the speedup is substantially better because the T4's parallelism makes the single-pass verification of K tokens nearly as fast as a single-token forward pass, amplifying the benefit of each accepted token.

### 5.6 Latency Breakdown: Explaining the Mechanism

The instrumented latency data (Section 4.4) provides the mechanistic explanation for the speedup curve:

1. **Draft cost is linear in K.** Each round runs K sequential autoregressive passes through the 1B model. Going from K=1 (3.03s draft) to K=8 (10.49s draft) is a ~3.5x increase, consistent with the per-token draft cost being roughly constant.

2. **Verification cost is inversely proportional to K.** Fewer rounds are needed at higher K, and each round still requires exactly one target forward pass. Verify time drops from 4.63s (K=1) to 1.90s (K=8).

3. **The crossover determines the optimal K.** At K=2, draft and verification are roughly balanced (42% vs 33%). Beyond K=4, drafting dominates — and since most drafted tokens are rejected, this time is largely wasted.

This breakdown is not present in the original Chen et al. paper, which analyzes speculative decoding theoretically but does not report empirical phase-level timing.

## 6. Limitations and Future Work

Two items originally listed as future work were implemented and are reported above: **KV-cache integration** (Section 4.7, verified lossless) and the **draft model size sweep** (Section 4.8). The draft-size sweep also partially addresses the "larger model gaps" question — though it revealed that at batch-1 on a fast GPU the relationship is dominated by latency overhead rather than the FLOP ratio. Remaining directions:

1. **Stronger adaptive K**: I implemented a basic adaptive-K policy (Section 4.10); it helped (+8% over baseline at 200 tokens) but did not beat the best fixed K=2. Next steps: smoother acceptance estimates, per-domain K, or hardware-aware caps. The cross-regime result (optimal K=1 on short H100 runs vs K=2 on T4 and long H100 runs) confirms K should adapt to both hardware and generation length.

2. **Much larger size gaps in a FLOP-bound setting**: The size sweep here was latency-bound at batch 1. With a very large target (e.g., 70B) or large batches, draft FLOPs would matter more and a tiny draft could win — the opposite of what was observed here.

3. **Tree-structured speculation**: Instead of a single draft sequence, the draft model could explore a tree of possible continuations, allowing the target model to verify multiple alternative paths in one pass (Medusa, SpecInfer).

4. **Batch inference**: The benchmark tests single-sequence generation. In high-throughput serving scenarios, the interplay between speculative decoding and continuous batching introduces additional trade-offs — and would shift the size-sweep conclusion toward FLOP-bound behavior.

## 7. Conclusion

I implemented and benchmarked speculative decoding using Llama 3.2 1B (draft) and 3B (target) on a T4 GPU. The optimal configuration (K=2) achieved a **1.31x speedup** over standard autoregressive decoding with a **67.7% acceptance rate**, while producing output from an identical distribution. Code generation benefited most (1.39x speedup, 67% acceptance), consistent with the higher predictability of structured programming text.

I additionally validated the same pipeline in **full precision on an H100**, where the optimal K shifts to 1 and peak speedup is 1.17x — showing that the best speculation length is hardware- and precision-dependent rather than intrinsic to the model pair.

Beyond reproducing the core algorithm, this project contributes:
- **Per-domain analysis** showing that code, math, and prose have meaningfully different acceptance rates and speedups — a dimension not explored in the original papers, and one that holds across both the T4 and H100 regimes.
- **Instrumented latency breakdown** revealing that draft generation becomes the bottleneck at higher K, and that the crossover between draft and verification cost precisely explains the optimal K.
- **Token-level visualization** making the accept/reject dynamics visible at the individual token level, including probability ratios and resampling events.
- **A cross-regime study** (4-bit T4 vs full-precision H100) demonstrating that the optimal speculation length and achievable speedup depend on hardware and precision.
- **A lossless KV-cache implementation** with non-trivial dual-cache rollback on rejection, verified token-for-token against the non-cached decoders.
- **A draft-size sweep** showing the counter-intuitive result that at batch-1 on a fast GPU the *larger*, higher-acceptance draft maximizes speedup, because small-model latency is overhead-bound rather than FLOP-bound.
- **Long-context and adaptive-K studies** showing that at 200 tokens fixed K=2 reaches **1.50x** on the H100, while a simple adaptive controller modestly improves over baseline but does not yet match the best fixed policy.

The results confirm the theoretical prediction — speculative decoding is a genuine free lunch when the draft-to-target cost ratio is favorable — but also sharpen it: whether that ratio is favorable depends as much on the hardware, precision, and batch regime as on the model sizes themselves.

## References

- Chen, C., Borgeaud, S., Irving, G., Lespiau, J.-B., Sifre, L., & Kramber, J. (2023). *Accelerating Large Language Model Decoding with Speculative Sampling*. arXiv:2302.01318.
- Leviathan, Y., Kalman, M., & Matias, Y. (2023). *Fast Inference from Transformers via Speculative Decoding*. ICML 2023.
