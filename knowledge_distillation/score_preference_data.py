#!/usr/bin/env python3
"""
Score preference datasets with vLLM using correct chosen-token log-likelihoods.

Per response (chosen & rejected) we compute:
  (1) Perplexity
  (2) Total (summed) log-likelihood
  (3) Mean token log-likelihood
  (4) Min token log-likelihood

Key correctness points:
- We request prompt logprobs for the *chosen* tokens (prompt_logprobs=1 by default).
- We slice only the response span within the prompt tokens.
- We generate 1 token (required by vLLM) but IGNORE it for metrics.
- We never substitute another token's logprob for the chosen token.
- If a chosen token logprob is missing (rare), behavior is controlled by --missing-policy.

Usage example:
  python score_pref_vllm.py \
    --model meta-llama/Llama-3-8b-instruct \
    --dataset allenai/llama-3.1-tulu-3-8b-preference-mixture \
    --split train \
    --output tulu_scores.csv
"""

from __future__ import annotations
import argparse
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams


# ---------------- Chat templating helpers ----------------

def has_chat_template(tok) -> bool:
    return hasattr(tok, "apply_chat_template")

def apply_chat_build_ids(tok, messages_prefix: List[Dict[str, str]], response_text: str) -> Tuple[str, List[int], List[int]]:
    """Render with chat template and return (full_text, prefix_ids, full_ids)."""
    prefix_text = tok.apply_chat_template(
        messages_prefix,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
        thinking=False
    )
    prefix_ids = tok(prefix_text, add_special_tokens=False).input_ids

    full_text = tok.apply_chat_template(
        messages_prefix + [{"role": "assistant", "content": response_text}],
        add_generation_prompt=False,
        tokenize=False,
        enable_thinking=False,
        thinking=False
    )
    full_ids = tok(full_text, add_special_tokens=False).input_ids
    return full_text, prefix_ids, full_ids

def raw_concat_build_ids(tok, prompt_text: str, response_text: str) -> Tuple[str, List[int], List[int]]:
    """Fallback when no chat template is available or --no-chat is set."""
    full_text = (prompt_text or "") + (response_text or "")
    prefix_ids = tok(prompt_text or "", add_special_tokens=False).input_ids
    full_ids = tok(full_text, add_special_tokens=False).input_ids
    return full_text, prefix_ids, full_ids


# ---------------- Dataset normalization ----------------

def extract_pref_row(row: Dict[str, Any]) -> Tuple[str, List[Dict[str, str]], str, List[Dict[str, str]], str, Optional[str]]:
    """
    Normalize a row into:
      prompt_text_for_csv,
      chosen_prefix_msgs, chosen_response,
      rejected_prefix_msgs, rejected_response,
      sample_id (or None)
    """
    sample_id = None
    for key in ("id", "sample_id", "example_id"):
        if key in row and row[key] is not None:
            sample_id = str(row[key])
            break

    # string-style
    if isinstance(row.get("chosen"), str) and isinstance(row.get("rejected"), str):
        prm = str(row.get("prompt", ""))
        return (prm,
                [{"role": "user", "content": prm}], str(row["chosen"]),
                [{"role": "user", "content": prm}], str(row["rejected"]),
                sample_id)

    # chat-style
    def split_messages(msgs: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], str]:
        if not msgs: return [], ""
        idx = len(msgs) - 1
        if msgs[-1].get("role") != "assistant":
            for j in range(len(msgs) - 1, -1, -1):
                if msgs[j].get("role") == "assistant":
                    idx = j
                    break
        prefix = msgs[:idx]
        resp = msgs[idx].get("content", "")
        return prefix, resp

    c, r = row.get("chosen"), row.get("rejected")
    if isinstance(c, list) and isinstance(r, list):
        ch_prefix, ch_resp = split_messages(c)
        rj_prefix, rj_resp = split_messages(r)

        def latest_user(prefix: List[Dict[str, str]]) -> str:
            for m in reversed(prefix):
                if m.get("role") == "user":
                    return str(m.get("content", ""))
            return str(row.get("prompt", "")) if row.get("prompt") else ""

        prm = latest_user(ch_prefix)
        return prm, ch_prefix, ch_resp, rj_prefix, rj_resp, sample_id

    # fallback
    return "", [{"role": "user", "content": ""}], str(row.get("chosen", "")), \
           [{"role": "user", "content": ""}], str(row.get("rejected", "")), sample_id


# ---------------- Metrics ----------------

def metrics_from_logprobs(seq: List[Optional[float]]) -> Tuple[float, float, float, float, int]:
    xs = [x for x in seq if x is not None and not (isinstance(x, float) and math.isnan(x))]
    N = len(xs)
    if N == 0:
        return float("inf"), float("-inf"), float("-inf"), float("-inf"), 0
    total = sum(xs)
    mean = total / N
    min_lp = min(xs)
    ppl = math.exp(-mean)
    return ppl, total, mean, min_lp, N


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--dataset", required=True, help="HF dataset name or path.")
    ap.add_argument("--split", default="train", help="Dataset split.")
    ap.add_argument("--id-column", default="id", help="Preferred ID column if present.")
    ap.add_argument("--output", required=True, help="Output CSV path.")
    # Model
    ap.add_argument("--model", required=True, help="HF model id or local path (causal LM).")
    ap.add_argument("--system", default=None, help="Optional system prompt for chat templates.")
    ap.add_argument("--no-chat", action="store_true", help="Disable chat template even if available.")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    # Logprob controls
    ap.add_argument("--prompt-topk", type=int, default=1,
                    help="K alternatives for prompt_logprobs. Use 1 to request the chosen-token logprob only.")
    ap.add_argument("--missing-policy", choices=["error", "skip", "floor"], default="error",
                    help="What to do if a chosen prompt token's logprob is missing when prompt-topk>1.")
    ap.add_argument("--missing-floor", type=float, default=-100.0,
                    help="Logprob value used when --missing-policy=floor.")
    args = ap.parse_args()

    # Load dataset
    ds = load_dataset(args.dataset, split=args.split).select(range(10000))

    # Init vLLM
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="auto",
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_batched_tokens=32768,
        max_model_len=2*8192,
        gpu_memory_utilization=0.85,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
    )
    tok = llm.get_tokenizer()
    use_chat = has_chat_template(tok) and not args.no_chat
    eos_id = getattr(tok, "eos_token_id", None)
    
    # Get max model length
    max_model_len = getattr(llm.llm_engine.model_config, "max_model_len", 2*8192)

    # Build prompts & spans
    prompts_full: List[str] = []
    spans: List[Tuple[int, int]] = []
    meta: List[Dict[str, Any]] = []
    skipped_count = 0

    for idx, row in enumerate(tqdm(ds)):
        prm_text, ch_prefix, ch_resp, rj_prefix, rj_resp, sample_id = extract_pref_row(row)
        if args.id_column in row and row[args.id_column] is not None:
            sample_id = str(row[args.id_column])

        def with_system(prefix_msgs: List[Dict[str, str]]) -> List[Dict[str, str]]:
            if not use_chat:
                return prefix_msgs
            return ([{"role": "system", "content": args.system}] + prefix_msgs) if args.system else prefix_msgs

        for is_chosen, prefix_msgs, resp_text in (
            (True, ch_prefix, ch_resp),
            (False, rj_prefix, rj_resp),
        ):
            if use_chat:
                full_text, prefix_ids, full_ids = apply_chat_build_ids(tok, with_system(prefix_msgs), resp_text)
            else:
                # raw fallback: use last user content as "prompt" string
                if prefix_msgs:
                    last_user = ""
                    for m in reversed(prefix_msgs):
                        if m.get("role") == "user":
                            last_user = m.get("content", "")
                            break
                    raw_prompt = last_user or "\n".join(f"[{m.get('role','')}] {m.get('content','')}" for m in prefix_msgs)
                else:
                    raw_prompt = prm_text or ""
                full_text, prefix_ids, full_ids = raw_concat_build_ids(tok, raw_prompt, resp_text)

            # Skip if prompt is too long
            if len(full_ids) > max_model_len:
                skipped_count += 1
                continue

            start = len(prefix_ids)
            end = len(full_ids)
            if eos_id is not None and end > start and full_ids[end - 1] == eos_id:
                end -= 1

            prompts_full.append(full_text)
            spans.append((start, end))
            meta.append({
                "sample_index": idx,
                "sample_id": sample_id if sample_id is not None else None,
                "prompt": prm_text,
                "is_chosen": is_chosen,
            })

    # Request prompt logprobs for chosen tokens; gen 1 token (ignored later)
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=args.prompt_topk if args.prompt_topk is not None else 1,
        logprobs=0,
    )
    outputs = llm.generate(prompts_full, sampling)

    # Collect metrics with rigorous chosen-token handling
    rows = []
    total_resp_tokens = 0
    missing_count = 0

    for out, (start, end), m in zip(outputs, spans, meta):
        v_ids = getattr(out, "prompt_token_ids", None)
        lp_dicts = getattr(out, "prompt_logprobs", None)
        chosen_lp_list = getattr(out, "prompt_token_logprobs", None)  # may exist in newer vLLM

        if v_ids is None or lp_dicts is None:
            raise RuntimeError(
                "Prompt logprobs not returned. Ensure vLLM is recent and prompt_logprobs>0."
            )

        # Realized (chosen) token logprob per prompt position
        realized: List[Optional[float]] = [None] * len(v_ids)

        if chosen_lp_list is not None and len(chosen_lp_list) == len(v_ids):
            # Direct chosen-token logprobs provided (preferred)
            realized = chosen_lp_list
        else:
            # Extract from dictionaries; enforce chosen-token lookup
            for pos, (tid, d) in enumerate(zip(v_ids, lp_dicts)):
                if d is None:
                    realized[pos] = None
                else:
                    entry = d.get(tid, None)
                    if entry is not None:
                        realized[pos] = entry.logprob
                    else:
                        # Chosen token missing from returned set (possible if prompt_topk>1 and too small)
                        if args.prompt_topk == 1:
                            # With topk=1, we expect the chosen token; treat as missing.
                            missing_count += 1
                            if args.missing_policy == "error":
                                raise RuntimeError(
                                    "Chosen token logprob missing with prompt_topk=1; "
                                    "try increasing --prompt-topk."
                                )
                            elif args.missing_policy == "skip":
                                realized[pos] = None
                            else:  # floor
                                realized[pos] = args.missing_floor
                        else:
                            # K>1 but chosen not included
                            missing_count += 1
                            if args.missing_policy == "error":
                                raise RuntimeError(
                                    f"Chosen token not in top-{args.prompt_topk} at position {pos}. "
                                    "Increase --prompt-topk or change --missing-policy."
                                )
                            elif args.missing_policy == "skip":
                                realized[pos] = None
                            else:
                                realized[pos] = args.missing_floor

        # Slice to response span and compute metrics
        start = max(0, min(start, len(realized)))
        end = max(start, min(end, len(realized)))
        resp_lps = realized[start:end]
        total_resp_tokens += len([x for x in resp_lps if x is not None])

        ppl, total_ll, mean_ll, min_ll, n_tokens = metrics_from_logprobs(resp_lps)

        rows.append({
            "sample_index": m["sample_index"],
            "sample_id": m["sample_id"],
            "is_chosen": m["is_chosen"],
            "perplexity": ppl,
            "total_loglik": total_ll,
            "mean_loglik": mean_ll,
            "min_loglik": min_ll,
            "response_token_count": n_tokens,
        })

    # Save
    out_df = pd.DataFrame(rows)
    out_df.sort_values(["sample_index", "is_chosen"], inplace=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_csv(args.output, index=False)

    # Coverage summary
    missing_note = ""
    if args.missing_policy == "skip":
        # Count Nones across response tokens
        missing_tokens = sum(
            1 for r in rows
            for _ in range(r["response_token_count"])  # approximate; detailed accounting omitted for brevity
        )  # not exact; we already reported missing_count events above
        missing_note = f"(missing events: {missing_count}; skipped tokens not counted in metrics)"
    elif args.missing_policy == "floor":
        missing_note = f"(missing events: {missing_count}; used floor={args.missing_floor})"
    else:
        missing_note = f"(missing events: {missing_count})"

    print(f"✅ Wrote {len(out_df)} rows to: {args.output}")
    print(f"Skipped {skipped_count} prompts that exceeded max model length of {max_model_len} tokens")
    print("Columns: sample_index, sample_id, prompt, is_chosen, perplexity, total_loglik, mean_loglik, min_loglik, response_token_count")
    print(f"Chosen-token coverage note: {missing_note}")

if __name__ == "__main__":
    main()
