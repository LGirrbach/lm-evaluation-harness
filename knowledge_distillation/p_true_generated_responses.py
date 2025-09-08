#!/usr/bin/env python3
"""
Score (prompt, response) pairs saved by generate_responses_vllm.py
using a strict YES/NO continuation rater with vLLM.

Input  (JSON list): [{ "prompt": str, "response": str, "id": Optional[str] }, ...]
Output (CSV):       row_index, sample_id, answer, p_yes, p_no, gen_token

Method:
- For each pair, ask the model:
    "Is the continuation a GOOD answer to the prompt?"
  and force a single-token answer (max_tokens=1) with logprobs.
- Calibrate p(yes) over the union of tokenizer variants for "yes"/"no" within
  the returned top-K logprobs of the first generated token.

Example:
  python score_generated_responses_yesno_vllm.py \
    --model meta-llama/Llama-3-8b-instruct \
    --input outputs/tulu3_responses.json \
    --output outputs/tulu3_responses_scored.csv
"""

from __future__ import annotations
import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm
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
            # important for models exposing these flags
            enable_thinking=False,
            thinking=False,
        )
    # Raw fallback: prepend system prompt if provided.
    prefix = (system_prompt + "\n\n") if system_prompt else ""
    return prefix + user_text


# ---------------- YES/NO token handling ----------------

def yes_no_token_sets(tok) -> Tuple[Set[int], Set[int]]:
    """
    Collect plausible single-token ids for yes/no, including space+case variants.
    Only keep forms that are exactly one token under the tokenizer.
    """
    variants = [
        "yes", "Yes", "YES", " yes", " Yes",
        "no",  "No",  "NO",  " no",  " No",
    ]
    enc = {v: tok(v, add_special_tokens=False).input_ids for v in variants}
    y_ids = {ids[0] for v, ids in enc.items() if v.strip().lower() == "yes" and len(ids) == 1}
    n_ids = {ids[0] for v, ids in enc.items() if v.strip().lower() == "no"  and len(ids) == 1}

    # Fallbacks (rare) if no single-token forms exist
    if not y_ids:
        ids = tok("yes", add_special_tokens=False).input_ids
        if ids:
            y_ids = {ids[0]}
    if not n_ids:
        ids = tok("no", add_special_tokens=False).input_ids
        if ids:
            n_ids = {ids[0]}
    return y_ids, n_ids


def compute_binary_mass(lp_dict: Dict[int, Any],
                        yes_ids: Set[int],
                        no_ids: Set[int],
                        eps: float = 1e-9) -> Tuple[float, float, float]:
    """
    Return (p_yes, y_mass, n_mass) normalized over yes∪no using top-k logprobs dict.
    """
    y_mass = sum(math.exp(obj.logprob) for tid, obj in lp_dict.items() if tid in yes_ids)
    n_mass = sum(math.exp(obj.logprob) for tid, obj in lp_dict.items() if tid in no_ids)
    if y_mass == 0.0:
        y_mass = eps
    if n_mass == 0.0:
        n_mass = eps
    p_yes = y_mass / (y_mass + n_mass)
    return p_yes, y_mass, n_mass


# ---------------- I/O helpers ----------------

def read_response_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of {prompt, response, id?} objects.")
    # Normalize keys to expected types
    norm: List[Dict[str, Any]] = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            continue
        prompt = str(row.get("prompt", "") or "")
        response = str(row.get("response", "") or "")
        sample_id = row.get("id")
        sample_id = None if sample_id is None else str(sample_id)
        norm.append({"prompt": prompt, "response": response, "id": sample_id})
    return norm


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--input", required=True, help="Path to JSON from generate_responses_vllm.py.")
    ap.add_argument("--output", required=True, help="Output CSV path.")
    # Model
    ap.add_argument("--model", required=True, help="HF model id or local path (causal LM).")
    ap.add_argument("--system", default=None, help="Override system prompt (defaults to strict rater).")
    ap.add_argument("--no-chat", action="store_true", help="Disable chat template even if available.")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--pipeline-parallel-size", type=int, default=1)
    ap.add_argument("--kv-cache-dtype", type=str, default=None, help="KV cache dtype: fp8, fp8_e4m3, fp8_e5m2, fp16/bf16")
    # Inference controls
    ap.add_argument("--logprobs-k", type=int, default=5, help="Top-K token logprobs to return for first token.")
    ap.add_argument("--epsilon-floor", type=float, default=1e-9, help="Floor mass if yes/no absent from top-K.")
    ap.add_argument("--max-examples", type=int, default=None, help="Optional cap for debugging.")
    args = ap.parse_args()

    # Load the generated (prompt, response) pairs
    examples = read_response_json(args.input)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    # Init vLLM
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="auto",
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        max_num_batched_tokens=8192,
        max_model_len=8192,
        gpu_memory_utilization=0.6,
        kv_cache_dtype=args.kv_cache_dtype,
    )
    tok = llm.get_tokenizer()
    use_chat = has_chat_template(tok) and not args.no_chat

    # Build rating prompts
    prompts: List[str] = []
    meta: List[Dict[str, Any]] = []
    for idx, ex in enumerate(tqdm(examples, desc="Preparing rating prompts")):
        user_text = USER_TEMPLATE.format(prompt=ex["prompt"], continuation=ex["response"])
        rendered = render_for_model(tok, args.system or SYSTEM_PROMPT, user_text, use_chat)
        prompts.append(rendered)
        meta.append({
            "row_index": idx,
            "sample_id": ex.get("id"),
        })

    # Sampling params: one token + logprobs
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=args.logprobs_k,
    )

    # Run generation in batch
    outputs = llm.generate(prompts, sampling)

    # Prepare yes/no token sets
    yes_ids, no_ids = yes_no_token_sets(tok)

    # Collect results
    rows: List[Dict[str, Any]] = []
    for out, m in zip(outputs, meta):
        if not out.outputs:
            rows.append({
                "row_index": m["row_index"],
                "sample_id": m["sample_id"],
                "answer": "no",
                "p_yes": 0.5,
                "p_no": 0.5,
                "gen_token": "",
            })
            continue

        first = out.outputs[0]
        gen_tid = first.token_ids[0]
        gen_tok = tok.convert_ids_to_tokens([gen_tid])[0] if hasattr(tok, "convert_ids_to_tokens") else tok.decode([gen_tid])
        lp_dict = first.logprobs[0]  # token_id -> LogProb

        p_yes, _, _ = compute_binary_mass(lp_dict, yes_ids, no_ids, eps=args.epsilon_floor)

        # Discrete answer from generated token (fallbacks mirror the reference script)
        if gen_tid in yes_ids:
            ans = "yes"
        elif gen_tid in no_ids:
            ans = "no"
        else:
            gen_text = first.text.strip().lower() if hasattr(first, "text") else tok.decode([gen_tid]).strip().lower()
            if gen_text.startswith("yes"):
                ans = "yes"
            elif gen_text.startswith("no"):
                ans = "no"
            else:
                ans = "yes" if p_yes >= 0.5 else "no"

        rows.append({
            "row_index": m["row_index"],
            "sample_id": m["sample_id"],
            "answer": ans,
            "p_yes": float(p_yes),
            "p_no": float(1.0 - p_yes),
            "gen_token": gen_tok,
        })

    # Save CSV
    out_df = pd.DataFrame(rows)
    out_df.sort_values(["row_index"], inplace=True)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_df.to_csv(args.output, index=False)

    print(f"✅ Scored {len(out_df)} responses to: {args.output}")
    print("Columns: row_index, sample_id, answer, p_yes, p_no, gen_token")
    print(f"Chat template used: {use_chat}; logprobs_k={args.logprobs_k}")
    print(f"yes_ids (sample): {sorted(list(yes_ids))[:5]} … | no_ids (sample): {sorted(list(no_ids))[:5]} …")


if __name__ == "__main__":
    main()
