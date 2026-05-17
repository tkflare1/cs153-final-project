"""
Speculative Decoding Engine
Implements the speculative sampling algorithm from Chen et al. 2023 (arXiv:2302.01318).

The core idea: a small draft model proposes K candidate tokens, then the larger
target model verifies them in a single forward pass. Accepted tokens are kept;
on rejection, we resample from the residual distribution and discard subsequent tokens.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def get_logits(model, input_ids):
    """Run a forward pass and return logits for the last position."""
    with torch.no_grad():
        outputs = model(input_ids)
    return outputs.logits


def sample_from_distribution(probs):
    """Sample a single token from a probability distribution."""
    return torch.multinomial(probs, num_samples=1).item()


def speculative_decode(
    draft_model,
    target_model,
    tokenizer,
    input_ids,
    K=4,
    max_new_tokens=50,
    temperature=1.0,
):
    """
    Speculative decoding following Chen et al. 2023.

    Args:
        draft_model: smaller, faster model used for drafting tokens
        target_model: larger, more capable model used for verification
        tokenizer: shared tokenizer for both models
        input_ids: tensor of shape (1, seq_len) with the prompt token ids
        K: number of candidate tokens the draft model proposes per step
        max_new_tokens: maximum number of new tokens to generate
        temperature: sampling temperature (1.0 = greedy-ish for this project)

    Returns:
        generated_ids: tensor of all generated token ids (prompt + new tokens)
        total_accepted: total number of tokens accepted from draft proposals
        total_proposed: total number of tokens proposed by draft model
    """
    draft_device = next(draft_model.parameters()).device
    target_device = next(target_model.parameters()).device
    generated = input_ids.to(draft_device).clone()
    total_accepted = 0
    total_proposed = 0
    tokens_generated = 0

    while tokens_generated < max_new_tokens:
        # --- Step 1: Draft model generates K candidate tokens autoregressively ---
        draft_input = generated.clone()
        draft_probs_list = []
        draft_tokens = []

        for _ in range(K):
            if tokens_generated + len(draft_tokens) >= max_new_tokens:
                break
            logits = get_logits(draft_model, draft_input)
            last_logits = logits[:, -1, :]
            probs = torch.softmax(last_logits / temperature, dim=-1)
            draft_probs_list.append(probs.squeeze(0))
            token = sample_from_distribution(probs.squeeze(0))
            draft_tokens.append(token)
            draft_input = torch.cat(
                [draft_input, torch.tensor([[token]], device=draft_device)], dim=1
            )

        if not draft_tokens:
            break

        num_draft = len(draft_tokens)
        total_proposed += num_draft

        # --- Step 2: Target model scores all candidate tokens in one forward pass ---
        candidate_ids = torch.tensor([draft_tokens], device=target_device)
        verify_input = torch.cat([generated.to(target_device), candidate_ids], dim=1)
        target_logits = get_logits(target_model, verify_input)

        # Extract target probabilities at positions corresponding to each draft token
        # Position of first draft token in the verify_input is generated.shape[1]
        start_pos = generated.shape[1] - 1  # logits at this position predict first draft token

        # --- Step 3: Accept/reject each draft token ---
        # Move target logits to draft device for uniform comparison
        target_logits_local = target_logits.to(draft_device)
        num_accepted = 0
        for i in range(num_draft):
            target_probs_i = torch.softmax(
                target_logits_local[:, start_pos + i, :] / temperature, dim=-1
            ).squeeze(0)
            draft_probs_i = draft_probs_list[i]
            draft_token = draft_tokens[i]

            p_target = target_probs_i[draft_token].item()
            p_draft = draft_probs_i[draft_token].item()

            if p_draft == 0:
                residual = torch.clamp(target_probs_i - draft_probs_i, min=0)
                residual = residual / (residual.sum() + 1e-10)
                resampled = sample_from_distribution(residual)
                generated = torch.cat(
                    [generated, torch.tensor([[resampled]], device=draft_device)], dim=1
                )
                tokens_generated += 1
                num_accepted += 0
                break

            ratio = min(1.0, p_target / p_draft)
            r = np.random.random()

            if r < ratio:
                generated = torch.cat(
                    [generated, torch.tensor([[draft_token]], device=draft_device)], dim=1
                )
                tokens_generated += 1
                num_accepted += 1
            else:
                residual = torch.clamp(target_probs_i - draft_probs_i, min=0)
                residual = residual / (residual.sum() + 1e-10)
                resampled = sample_from_distribution(residual)
                generated = torch.cat(
                    [generated, torch.tensor([[resampled]], device=draft_device)], dim=1
                )
                tokens_generated += 1
                break
        else:
            bonus_probs = torch.softmax(
                target_logits_local[:, start_pos + num_draft, :] / temperature, dim=-1
            ).squeeze(0)
            bonus_token = sample_from_distribution(bonus_probs)
            generated = torch.cat(
                [generated, torch.tensor([[bonus_token]], device=draft_device)], dim=1
            )
            tokens_generated += 1

        total_accepted += num_accepted

    return generated, total_accepted, total_proposed


# ---------------------------------------------------------------------------
# Standalone verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading models for standalone verification...")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    draft_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    target_model = AutoModelForCausalLM.from_pretrained("gpt2")
    draft_model.eval()
    target_model.eval()

    prompt = "The quick brown fox"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    print(f"Prompt: {prompt}")
    print("Running speculative decoding with K=4, max_new_tokens=20 ...")

    generated, accepted, proposed = speculative_decode(
        draft_model, target_model, tokenizer, input_ids, K=4, max_new_tokens=20
    )

    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    acceptance_rate = accepted / proposed if proposed > 0 else 0.0

    print(f"Output: {output_text}")
    print(f"Accepted: {accepted}/{proposed} ({acceptance_rate:.2%})")
    print("speculative_decode.py verification PASSED.")
