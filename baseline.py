"""
Baseline Autoregressive Generation
Standard greedy autoregressive decoding implemented manually (no model.generate()).
Used as the performance baseline for comparison with speculative decoding.
"""

import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def baseline_autoregressive(
    model,
    tokenizer,
    input_ids,
    max_new_tokens=50,
    temperature=1.0,
):
    """
    Standard greedy autoregressive decoding loop.

    At each step, run a full forward pass through the model, take the argmax
    (or sample) of the last position's logits, and append it to the sequence.

    Args:
        model: the target language model
        tokenizer: tokenizer (used for EOS detection)
        input_ids: tensor of shape (1, seq_len) with prompt token ids
        max_new_tokens: how many tokens to generate
        temperature: sampling temperature

    Returns:
        generated_ids: tensor with prompt + generated tokens
        elapsed: wall-clock generation time in seconds
        tokens_per_sec: throughput measurement
    """
    device = input_ids.device
    generated = input_ids.clone()

    start_time = time.perf_counter()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(generated)
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = max_new_tokens / elapsed

    return generated, elapsed, tokens_per_sec


# ---------------------------------------------------------------------------
# Standalone verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading target model for baseline verification...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()

    prompt = "The quick brown fox"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    print(f"Prompt: {prompt}")
    print("Running baseline autoregressive decoding, max_new_tokens=20 ...")

    generated, elapsed, tps = baseline_autoregressive(
        model, tokenizer, input_ids, max_new_tokens=20
    )

    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"Output: {output_text}")
    print(f"Time: {elapsed:.2f}s | Tokens/sec: {tps:.2f}")
    print("baseline.py verification PASSED.")
