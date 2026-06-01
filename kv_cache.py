"""
KV-Cache Integration
--------------------
Cached variants of baseline autoregressive decoding and speculative decoding.

Both mirror the exact sampling order of the non-cached versions in
baseline.py / speculative_decode.py (same multinomial / np.random calls in the
same sequence), so with a fixed seed they produce *identical* output. That
equivalence is the correctness/losslessness check for the cache logic.

The non-trivial part is cache rollback on rejection: after the target verifies
K speculative tokens, only the accepted prefix is committed, so both caches are
cropped back to the committed length (excluding the freshly resampled/bonus
token, which neither model has processed yet and which is fed as the 1-token
"tail" at the start of the next round).
"""

import time
import torch
import numpy as np

SEED = 42


def crop_cache(past, length):
    """Truncate a KV cache to `length` positions along the sequence axis."""
    if past is None:
        return None
    if hasattr(past, "crop"):
        past.crop(length)
        return past
    # Legacy tuple-of-tuples cache: (key, value) per layer, [B, H, S, D].
    return tuple((k[:, :, :length, :], v[:, :, :length, :]) for k, v in past)


def baseline_autoregressive_kv(model, tokenizer, input_ids, max_new_tokens=50, temperature=1.0):
    """Autoregressive decoding with a KV cache (feeds one new token per step)."""
    device = input_ids.device
    generated = input_ids.clone()

    start_time = time.perf_counter()
    past = None
    cur = generated
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(cur, past_key_values=past, use_cache=True)
            past = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            cur = next_token

    elapsed = time.perf_counter() - start_time
    return generated, elapsed, max_new_tokens / elapsed


def speculative_decode_kv(draft_model, target_model, tokenizer, input_ids,
                          K=4, max_new_tokens=50, temperature=1.0):
    """Speculative decoding with KV caching on both models + rollback on rejection."""
    draft_device = next(draft_model.parameters()).device
    target_device = next(target_model.parameters()).device
    generated = input_ids.to(draft_device).clone()
    total_accepted = 0
    total_proposed = 0
    tokens_generated = 0

    draft_past = None
    target_past = None
    # The draft only forwards K-1 of its K proposed tokens, while the target
    # forwards all K, so the two caches advance at different rates. Track them
    # separately and feed each model its own uncached committed "tail".
    draft_cache_len = 0
    target_cache_len = 0

    with torch.no_grad():
        while tokens_generated < max_new_tokens:
            gen_len = generated.shape[1]
            draft_tail = generated[:, draft_cache_len:]
            target_tail = generated[:, target_cache_len:]

            # --- Step 1: draft proposes K tokens, each via a 1-token cached step ---
            draft_tokens = []
            draft_probs_list = []
            draft_cur = draft_tail.to(draft_device)
            for _ in range(K):
                if tokens_generated + len(draft_tokens) >= max_new_tokens:
                    break
                out = draft_model(draft_cur, past_key_values=draft_past, use_cache=True)
                draft_past = out.past_key_values
                last_logits = out.logits[:, -1, :]
                probs = torch.softmax(last_logits / temperature, dim=-1).squeeze(0)
                draft_probs_list.append(probs)
                token = torch.multinomial(probs, num_samples=1).item()
                draft_tokens.append(token)
                draft_cur = torch.tensor([[token]], device=draft_device)

            if not draft_tokens:
                break
            num_draft = len(draft_tokens)
            total_proposed += num_draft

            # --- Step 2: target verifies tail + all K candidates in one cached pass ---
            candidate_ids = torch.tensor([draft_tokens], device=target_device)
            verify_cur = torch.cat([target_tail.to(target_device), candidate_ids], dim=1)
            out = target_model(verify_cur, past_key_values=target_past, use_cache=True)
            target_past = out.past_key_values
            target_logits = out.logits.to(draft_device)
            m = target_tail.shape[1]  # offset: logit at m-1 predicts first draft token

            # --- Step 3: accept / reject (identical RNG order to the non-cached version) ---
            num_accepted = 0
            committed_this_round = 0
            for i in range(num_draft):
                target_probs_i = torch.softmax(
                    target_logits[:, m - 1 + i, :] / temperature, dim=-1
                ).squeeze(0)
                draft_probs_i = draft_probs_list[i]
                draft_token = draft_tokens[i]

                p_target = target_probs_i[draft_token].item()
                p_draft = draft_probs_i[draft_token].item()

                _v = min(target_probs_i.shape[0], draft_probs_i.shape[0])

                if p_draft == 0:
                    residual = torch.clamp(target_probs_i[:_v] - draft_probs_i[:_v], min=0)
                    residual = residual / (residual.sum() + 1e-10)
                    resampled = torch.multinomial(residual, num_samples=1).item()
                    generated = torch.cat(
                        [generated, torch.tensor([[resampled]], device=draft_device)], dim=1)
                    tokens_generated += 1
                    committed_this_round += 1
                    break

                ratio = min(1.0, p_target / p_draft)
                r = np.random.random()
                if r < ratio:
                    generated = torch.cat(
                        [generated, torch.tensor([[draft_token]], device=draft_device)], dim=1)
                    tokens_generated += 1
                    num_accepted += 1
                    committed_this_round += 1
                else:
                    residual = torch.clamp(target_probs_i[:_v] - draft_probs_i[:_v], min=0)
                    residual = residual / (residual.sum() + 1e-10)
                    resampled = torch.multinomial(residual, num_samples=1).item()
                    generated = torch.cat(
                        [generated, torch.tensor([[resampled]], device=draft_device)], dim=1)
                    tokens_generated += 1
                    committed_this_round += 1
                    break
            else:
                # all proposed tokens accepted -> sample a bonus token from target
                bonus_probs = torch.softmax(
                    target_logits[:, m - 1 + num_draft, :] / temperature, dim=-1
                ).squeeze(0)
                bonus_token = torch.multinomial(bonus_probs, num_samples=1).item()
                generated = torch.cat(
                    [generated, torch.tensor([[bonus_token]], device=draft_device)], dim=1)
                tokens_generated += 1
                committed_this_round += 1

            total_accepted += num_accepted

            # --- Step 4: roll back each cache to what it actually processed ---
            # Target forwarded all proposed tokens, so it holds the committed
            # accepted ones (num_accepted of them). The draft only forwarded its
            # first num_draft-1 proposals, so in the all-accepted case it is one
            # short. The resampled/bonus token was fed to neither and becomes the
            # next round's tail.
            target_cache_len = gen_len + num_accepted
            draft_cache_len = gen_len + min(num_accepted, num_draft - 1)
            target_past = crop_cache(target_past, target_cache_len)
            draft_past = crop_cache(draft_past, draft_cache_len)

    return generated, total_accepted, total_proposed
