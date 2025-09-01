#!/usr/bin/env python3
"""
Score preference datasets with vLLM using a strict YES/NO continuation rater.

For each (prompt, continuation) pair (chosen & rejected independently) we ask the
model: "Is the continuation a GOOD answer to the prompt?" and force a single-token
answer: yes/no. We then extract the answer and compute a calibrated probability
p(yes) by summing softmax mass over tokenizer variants of the "yes" and "no"
first-token candidates.

Outputs a CSV with, per example-side:
  - sample_index, sample_id, is_chosen
  - answer ("yes"|"no")
  - p_yes, p_no (binary-normalized)
  - gen_token (first generated token, for debugging)

Usage example:
  python score_pref_yesno_vllm.py \
    --model meta-llama/Llama-3-8b-instruct \
    --dataset allenai/llama-3.1-tulu-3-8b-preference-mixture \
    --split train \
    --output tulu_yesno.csv

Compared to a perplexity scorer, this script:
  - Does NOT request prompt logprobs.
  - Generates exactly 1 token and reads its logprob distribution.
  - Calibrates p(yes) over the union of (yes ∪ no) candidate tokens.

Notes:
  - If either side (yes/no) is missing from top-K, we apply a tiny epsilon floor
    (configurable) or advise increasing --logprobs-k.
  - Keep temperature=0 for deterministic argmax answers; probability still comes
    from the returned logprobs.
"""

from __future__ import annotations
import argparse
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Set

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams


# ---------------- System prompt & user template ----------------

SYSTEM_PROMPT = (
    "You are a strict binary rater for assistant responses. "
    "Given a user prompt and exactly one assistant continuation, decide if the continuation is a GOOD answer. "
    "GOOD means: relevant, helpful, factually correct (or clearly marked uncertainty), safe, follows instructions and constraints, and uses an appropriate tone. "
    "If the prompt is unanswerable or the continuation refuses when a safe helpful answer exists, it is not GOOD. "
    "Output exactly one token: yes or no. Do not add any other text."
)

USER_TEMPLATE = (
    "[BEGIN PROMPT]\n{prompt}\n[END PROMPT]\n\n"
    "[BEGIN CONTINUATION]\n{continuation}\n[END CONTINUATION]\n\n"
    "Is the continuation a GOOD answer to the prompt?\n"
    "Answer with exactly one token: yes or no."
)


# ---------------- Chat templating helpers ----------------

def has_chat_template(tok) -> bool:
    return hasattr(tok, "apply_chat_template")


def render_for_model(tok, system_prompt: Optional[str], user_text: str, use_chat: bool) -> str:
    """Render a single-turn chat or raw text for generation."""
    if use_chat and has_chat_template(tok):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_text})
        return tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
            thinking=False
        )
    else:
        # Raw fallback: prepend system prompt if given.
        prefix = (system_prompt + "\n\n") if system_prompt else ""
        return prefix + user_text


# ---------------- Dataset normalization (same spirit as original) ----------------

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
        return (
            prm,
            [{"role": "user", "content": prm}], str(row["chosen"]),
            [{"role": "user", "content": prm}], str(row["rejected"]),
            sample_id,
        )

    # chat-style
    def split_messages(msgs: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], str]:
        if not msgs:
            return [], ""
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
    return (
        "",
        [{"role": "user", "content": ""}], str(row.get("chosen", "")),
        [{"role": "user", "content": ""}], str(row.get("rejected", "")),
        sample_id,
    )


# ---------------- YES/NO token handling ----------------

def yes_no_token_sets(tok) -> Tuple[Set[int], Set[int]]:
    """Collect plausible single-token ids for yes/no, including space/case variants."""
    variants = [
        "yes", "Yes", "YES", " yes", " Yes",
        "no",  "No",  "NO",  " no",  " No",
    ]
    enc = {v: tok(v, add_special_tokens=False).input_ids for v in variants}
    y_ids = {ids[0] for v, ids in enc.items() if v.strip().lower() == "yes" and len(ids) == 1}
    n_ids = {ids[0] for v, ids in enc.items() if v.strip().lower() == "no"  and len(ids) == 1}
    # Fallback if empty (rare): take first token of plain form
    if not y_ids:
        ids = tok("yes", add_special_tokens=False).input_ids
        if ids:
            y_ids = {ids[0]}
    if not n_ids:
        ids = tok("no", add_special_tokens=False).input_ids
        if ids:
            n_ids = {ids[0]}
    return y_ids, n_ids


# ---------------- Scoring core ----------------

def compute_binary_mass(lp_dict: Dict[int, Any], yes_ids: Set[int], no_ids: Set[int], eps: float = 1e-9) -> Tuple[float, float, float]:
    """Return (p_yes, y_mass, n_mass) normalized over yes∪no using top-k logprobs dict."""
    y_mass = sum(math.exp(obj.logprob) for tid, obj in lp_dict.items() if tid in yes_ids)
    n_mass = sum(math.exp(obj.logprob) for tid, obj in lp_dict.items() if tid in no_ids)
    if y_mass == 0.0:
        y_mass = eps
    if n_mass == 0.0:
        n_mass = eps
    p_yes = y_mass / (y_mass + n_mass)
    return p_yes, y_mass, n_mass


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
    ap.add_argument("--system", default=None, help="Override system prompt (uses default if omitted).")
    ap.add_argument("--no-chat", action="store_true", help="Disable chat template even if available.")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    # Inference controls
    ap.add_argument("--logprobs-k", type=int, default=5, help="Top-K token logprobs to return for first token.")
    ap.add_argument("--epsilon-floor", type=float, default=1e-9, help="Floor mass if yes/no absent from top-K.")
    ap.add_argument("--max-examples", type=int, default=None, help="Optional cap for debugging.")

    args = ap.parse_args()

    # Load dataset
    ds = load_dataset(args.dataset, split=args.split).select(range(10000))

    # Init vLLM
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="auto",
        tensor_parallel_size=args.tensor_parallel_size,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        max_num_batched_tokens=32768,
        max_model_len=2*8192,
        gpu_memory_utilization=0.85,
        
    )
    tok = llm.get_tokenizer()
    use_chat = has_chat_template(tok) and not args.no_chat

    # Build requests
    prompts: List[str] = []
    meta: List[Dict[str, Any]] = []

    n_rows = len(ds) if args.max_examples is None else min(args.max_examples, len(ds))

    for idx, row in enumerate(tqdm(ds, total=n_rows)):
        if args.max_examples is not None and idx >= args.max_examples:
            break
        prm_text, ch_prefix, ch_resp, rj_prefix, rj_resp, sample_id = extract_pref_row(row)
        if args.id_column in row and row[args.id_column] is not None:
            sample_id = str(row[args.id_column])

        def latest_user(prefix_msgs: List[Dict[str, str]]) -> str:
            for m in reversed(prefix_msgs):
                if m.get("role") == "user":
                    return str(m.get("content", ""))
            return prm_text or ""

        # Choose what to present as {prompt}: latest user content (common in pref datasets)
        prompt_render = latest_user(ch_prefix)

        for is_chosen, resp_text in ((True, ch_resp), (False, rj_resp)):
            user_text = USER_TEMPLATE.format(prompt=prompt_render or "", continuation=resp_text or "")
            rendered = render_for_model(tok, args.system or SYSTEM_PROMPT, user_text, use_chat)
            prompts.append(rendered)
            meta.append({
                "sample_index": idx,
                "sample_id": sample_id if sample_id is not None else None,
                "prompt": prompt_render,
                "is_chosen": is_chosen,
            })

    # Prepare generation params
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=args.logprobs_k,
    )

    # Generate in one go (vLLM batches internally)
    outputs = llm.generate(prompts, sampling)

    # YES/NO token sets once
    yes_ids, no_ids = yes_no_token_sets(tok)

    # Collect results
    rows: List[Dict[str, Any]] = []

    for out, m in zip(outputs, meta):
        if not out.outputs:
            # No token generated (should not happen with max_tokens=1); mark unknown
            rows.append({
                "sample_index": m["sample_index"],
                "sample_id": m["sample_id"],
                "is_chosen": m["is_chosen"],
                "answer": "no",
                "p_yes": 0.5,
                "p_no": 0.5,
                "gen_token": "",
            })
            continue

        first = out.outputs[0]
        gen_tid = first.token_ids[0]
        gen_tok = tok.convert_ids_to_tokens([gen_tid])[0] if hasattr(tok, "convert_ids_to_tokens") else tok.decode([gen_tid])
        lp_dict = first.logprobs[0]  # dict: token_id -> LogProb

        p_yes, y_mass, n_mass = compute_binary_mass(lp_dict, yes_ids, no_ids, eps=args.epsilon_floor)

        # Discrete answer from generated token; fallback to prob if token is neither
        if gen_tid in yes_ids:
            ans = "yes"
        elif gen_tid in no_ids:
            ans = "no"
        else:
            # Normalize token string for a last-ditch heuristic, then fall back to p_yes
            gen_text = first.text.strip().lower() if hasattr(first, "text") else tok.decode([gen_tid]).strip().lower()
            if gen_text.startswith("yes"):
                ans = "yes"
            elif gen_text.startswith("no"):
                ans = "no"
            else:
                ans = "yes" if p_yes >= 0.5 else "no"

        rows.append({
            "sample_index": m["sample_index"],
            "sample_id": m["sample_id"],
            "is_chosen": m["is_chosen"],
            "answer": ans,
            "p_yes": float(p_yes),
            "p_no": float(1.0 - p_yes),
            "gen_token": gen_tok,
        })

    # Save
    out_df = pd.DataFrame(rows)
    out_df.sort_values(["sample_index", "is_chosen"], inplace=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_csv(args.output, index=False)

    print(f"✅ Wrote {len(out_df)} rows to: {args.output}")
    print("Columns: sample_index, sample_id, is_chosen, answer, p_yes, p_no, gen_token")
    print(f"Chat template used: {use_chat}; logprobs_k={args.logprobs_k}; yes_ids={sorted(list(yes_ids))[:5]}… no_ids={sorted(list(no_ids))[:5]}…")


if __name__ == "__main__":
    main()
