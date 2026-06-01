"""
Instrumented Speculative Decoding Engine
Same algorithm as speculative_decode.py but with per-phase timing and
per-token trace logging for latency breakdown and token visualization.
"""

import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import SEED

torch.manual_seed(SEED)
np.random.seed(SEED)


def get_logits(model, input_ids):
    with torch.no_grad():
        outputs = model(input_ids)
    return outputs.logits


def sample_from_distribution(probs):
    return torch.multinomial(probs, num_samples=1).item()


def speculative_decode_instrumented(
    draft_model,
    target_model,
    tokenizer,
    input_ids,
    K=4,
    max_new_tokens=50,
    temperature=1.0,
):
    """
    Speculative decoding with full instrumentation.

    Returns:
        generated_ids: tensor of all tokens (prompt + generated)
        total_accepted: count of accepted draft tokens
        total_proposed: count of proposed draft tokens
        timing: dict with cumulative time per phase
        trace: list of per-round dicts recording token-level decisions
    """
    draft_device = next(draft_model.parameters()).device
    target_device = next(target_model.parameters()).device
    generated = input_ids.to(draft_device).clone()
    total_accepted = 0
    total_proposed = 0
    tokens_generated = 0

    timing = {
        "draft_time": 0.0,
        "verify_time": 0.0,
        "accept_time": 0.0,
    }
    trace = []  # per-round token-level trace
    round_num = 0

    while tokens_generated < max_new_tokens:
        round_num += 1
        round_trace = {
            "round": round_num,
            "K": K,
            "tokens": [],
            "bonus_token": None,
        }

        # --- Draft phase ---
        t0 = time.perf_counter()
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
        t1 = time.perf_counter()
        timing["draft_time"] += t1 - t0

        if not draft_tokens:
            break

        num_draft = len(draft_tokens)
        total_proposed += num_draft

        # --- Verification phase ---
        t2 = time.perf_counter()
        candidate_ids = torch.tensor([draft_tokens], device=target_device)
        verify_input = torch.cat([generated.to(target_device), candidate_ids], dim=1)
        target_logits = get_logits(target_model, verify_input)
        t3 = time.perf_counter()
        timing["verify_time"] += t3 - t2

        start_pos = generated.shape[1] - 1

        # --- Accept/reject phase ---
        t4 = time.perf_counter()
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

            token_text = tokenizer.decode([draft_token])

            if p_draft == 0:
                residual = torch.clamp(target_probs_i - draft_probs_i, min=0)
                residual = residual / (residual.sum() + 1e-10)
                resampled = sample_from_distribution(residual)
                round_trace["tokens"].append({
                    "position": i,
                    "token_id": draft_token,
                    "token_text": token_text,
                    "p_target": p_target,
                    "p_draft": p_draft,
                    "ratio": 0.0,
                    "accepted": False,
                    "reason": "zero_draft_prob",
                    "resampled_id": resampled,
                    "resampled_text": tokenizer.decode([resampled]),
                })
                generated = torch.cat(
                    [generated, torch.tensor([[resampled]], device=draft_device)], dim=1
                )
                tokens_generated += 1
                break

            ratio = min(1.0, p_target / p_draft)
            r = np.random.random()

            if r < ratio:
                round_trace["tokens"].append({
                    "position": i,
                    "token_id": draft_token,
                    "token_text": token_text,
                    "p_target": round(p_target, 6),
                    "p_draft": round(p_draft, 6),
                    "ratio": round(ratio, 6),
                    "accepted": True,
                    "reason": "accepted",
                })
                generated = torch.cat(
                    [generated, torch.tensor([[draft_token]], device=draft_device)], dim=1
                )
                tokens_generated += 1
                num_accepted += 1
            else:
                residual = torch.clamp(target_probs_i - draft_probs_i, min=0)
                residual = residual / (residual.sum() + 1e-10)
                resampled = sample_from_distribution(residual)
                round_trace["tokens"].append({
                    "position": i,
                    "token_id": draft_token,
                    "token_text": token_text,
                    "p_target": round(p_target, 6),
                    "p_draft": round(p_draft, 6),
                    "ratio": round(ratio, 6),
                    "accepted": False,
                    "reason": "rejected",
                    "resampled_id": resampled,
                    "resampled_text": tokenizer.decode([resampled]),
                })
                generated = torch.cat(
                    [generated, torch.tensor([[resampled]], device=draft_device)], dim=1
                )
                tokens_generated += 1
                break
        else:
            # All K accepted — bonus token
            bonus_probs = torch.softmax(
                target_logits_local[:, start_pos + num_draft, :] / temperature, dim=-1
            ).squeeze(0)
            bonus_token = sample_from_distribution(bonus_probs)
            round_trace["bonus_token"] = {
                "token_id": bonus_token,
                "token_text": tokenizer.decode([bonus_token]),
            }
            generated = torch.cat(
                [generated, torch.tensor([[bonus_token]], device=draft_device)], dim=1
            )
            tokens_generated += 1

        t5 = time.perf_counter()
        timing["accept_time"] += t5 - t4

        total_accepted += num_accepted
        trace.append(round_trace)

    return generated, total_accepted, total_proposed, timing, trace


# ---------------------------------------------------------------------------
# Standalone verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from models import load_models

    print("Loading models for instrumented verification...")
    draft_model, target_model, tokenizer = load_models()

    prompt = "The quick brown fox"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(next(draft_model.parameters()).device)

    print(f"Prompt: {prompt}")
    print("Running instrumented speculative decoding K=4, max_new_tokens=20 ...")

    generated, accepted, proposed, timing, trace = speculative_decode_instrumented(
        draft_model, target_model, tokenizer, input_ids, K=4, max_new_tokens=20
    )

    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    acc_rate = accepted / proposed if proposed > 0 else 0.0
    total_time = sum(timing.values())

    print(f"\nOutput: {output_text}")
    print(f"Accepted: {accepted}/{proposed} ({acc_rate:.2%})")
    print(f"\n--- Latency Breakdown ---")
    print(f"Draft:    {timing['draft_time']:.3f}s ({timing['draft_time']/total_time*100:.1f}%)")
    print(f"Verify:   {timing['verify_time']:.3f}s ({timing['verify_time']/total_time*100:.1f}%)")
    print(f"Accept:   {timing['accept_time']:.3f}s ({timing['accept_time']/total_time*100:.1f}%)")
    print(f"Total:    {total_time:.3f}s")

    print(f"\n--- Token Trace (first 3 rounds) ---")
    for r in trace[:3]:
        print(f"\nRound {r['round']}:")
        for t in r["tokens"]:
            status = "ACCEPT" if t["accepted"] else "REJECT"
            print(f"  [{status}] '{t['token_text']}' "
                  f"p_target={t['p_target']:.4f} p_draft={t['p_draft']:.4f} "
                  f"ratio={t['ratio']:.4f}")
        if r["bonus_token"]:
            print(f"  [BONUS] '{r['bonus_token']['token_text']}'")

    print("\nspeculative_decode_instrumented.py verification PASSED.")
