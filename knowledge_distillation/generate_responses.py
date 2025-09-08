#!/usr/bin/env python3
"""
Generate model responses for prompts in a preference-style dataset using vLLM.

- Uses a neutral system prompt: "You are a helpful assistant."
- The user message is ONLY the dataset's prompt text (no extra instructions).
- Disables "thinking" in chat templates.
- Optimizes vLLM engine flags for efficient offline/batch inference.
- Saves a JSON list with: { "prompt": ..., "response": ..., "id": ... (if available) }.

Example:
  python generate_responses_vllm.py \
    --model meta-llama/Llama-3-8b-instruct \
    --dataset allenai/llama-3.1-tulu-3-8b-preference-mixture \
    --split train \
    --output outputs/tulu3_responses.json

Optionally restrict to one prompt id:
  python generate_responses_vllm.py \
    --model meta-llama/Llama-3-8b-instruct \
    --dataset your/dataset \
    --id 12345 \
    --output one_example.json
"""

from __future__ import annotations
import argparse
import json
import os
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams


NEUTRAL_SYSTEM_PROMPT = "You are a helpful assistant."


# ---------- Helpers for rendering with/without chat template ----------

def has_chat_template(tok) -> bool:
    return hasattr(tok, "apply_chat_template")


def render_chat(tok, system_prompt: Optional[str], user_text: str, max_model_len: int, model_name: str) -> str:
    """
    Render a single-turn chat prompt, explicitly disabling any 'thinking' features
    some chat templates expose.
    """
    messages = []
    if system_prompt:
        if model_name == "google/gemma-2-9b-it":
            pass
        else:
            messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    chat = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        # important for models whose templates expose these switches
        enable_thinking=False,
        thinking=False,
    )

    # Check tokenized length of chat
    chat_length_in_tokens = len(tok(chat, add_special_tokens=False).input_ids)
    if chat_length_in_tokens > max_model_len:
        raise ValueError(f"Chat length in tokens is greater than the model limit: {chat_length_in_tokens} > {max_model_len}")
    return chat


def render_fallback_plain(system_prompt: Optional[str], user_text: str) -> str:
    """
    Fallback if tokenizer has no chat template: prepend system prompt, then user text.
    We keep the user text untouched, as requested.
    """
    prefix = (system_prompt + "\n\n") if system_prompt else ""
    return prefix + user_text


# ---------- Prompt extraction from generic preference-like rows ----------

def extract_prompt_and_id(row: Dict[str, Any]) -> (str, Optional[str]):
    """
    Try common fields used in preference datasets. If a row has chat-style structures,
    we use the latest user message as the prompt. Otherwise fall back to 'prompt'.
    """
    # Preferred id-like columns
    sample_id = None
    for k in ("id", "sample_id", "example_id", "prompt_id"):
        if k in row and row[k] is not None:
            sample_id = str(row[k])
            break

    # If 'prompt' exists as a plain string, use it
    if isinstance(row.get("prompt"), str) and row["prompt"].strip():
        return row["prompt"], sample_id

    # Some datasets have chat histories under 'chosen'/'rejected' (lists of messages)
    def latest_user(msgs: List[Dict[str, str]]) -> Optional[str]:
        for m in reversed(msgs):
            if m.get("role") == "user":
                return str(m.get("content", ""))
        return None

    for key in ("chosen", "rejected"):
        val = row.get(key)
        if isinstance(val, list) and val:
            p = latest_user(val)
            if p:
                return p, sample_id

    # Fallback: empty prompt
    return "", sample_id


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--dataset", required=True, help="HF dataset name or local path.")
    ap.add_argument("--split", default="train", help="Dataset split (default: train).")
    ap.add_argument("--id", dest="only_id", default=None, help="Optionally select a single prompt by id.")
    ap.add_argument("--max-examples", type=int, default=None, help="Optional cap on number of prompts.")
    ap.add_argument("--output", required=True, help="Path to output JSON file.")

    # Model / Inference
    ap.add_argument("--model", required=True, help="HF model id or local path (causal LM).")
    ap.add_argument("--max-model-len", type=int, default=16384, help="Max prompt length in tokens.")
    ap.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallelism.")
    ap.add_argument("--max-tokens", type=int, default=512, help="Max new tokens to generate per prompt.")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default 0 for deterministic).")
    ap.add_argument("--top_p", type=float, default=1.0, help="Top-p nucleus sampling.")

    args = ap.parse_args()

    # Load dataset
    ds = load_dataset(args.dataset, split=args.split)
    # Filter out prompts with more than 7000 tokens
    ds = ds.filter(lambda x: len(x["prompt"].split()) < 7000)

    # Optionally filter to a single id
    if args.only_id is not None:
        # we try to match across common id columns
        id_cols = ["id", "sample_id", "example_id", "prompt_id"]
        def matches(row):
            for c in id_cols:
                if c in row and row[c] is not None and str(row[c]) == str(args.only_id):
                    return True
            return False
        ds = ds.filter(matches)

    # Optionally limit examples
    if args.max_examples is not None:
        ds = ds.select(range(min(args.max_examples, len(ds))))

    # Initialize vLLM (optimized for offline/batch throughput)
    max_model_len = args.max_model_len

    llm = LLM(
        model=args.model,
        trust_remote_code=True,          # needed by many HF chat models
        dtype="auto",                    # auto-selects the best precision for the hardware
        tensor_parallel_size=args.tensor_parallel_size,
        enable_chunked_prefill=True,     # throughput for long prompts
        enable_prefix_caching=True,      # helpful when many prompts share prefixes/system
        max_num_batched_tokens=32768,    # allow large batch token budgeting
        max_model_len=max_model_len,          # safe default; vLLM will cap to model limit
        gpu_memory_utilization=0.85,     # high utilization for offline inference
    )
    tok = llm.get_tokenizer()
    use_chat = has_chat_template(tok)

    # Prepare prompts
    prompts: List[str] = []
    metas: List[Dict[str, Any]] = []

    system_prompt = NEUTRAL_SYSTEM_PROMPT
    if args.model == "google/gemma-2-9b-it":
        system_prompt = None

    for row in tqdm(ds, desc="Preparing prompts"):
        prompt_text, sample_id = extract_prompt_and_id(row)
        if use_chat:
            try:
                rendered = render_chat(tok, system_prompt, prompt_text or "", max_model_len - args.max_tokens, args.model)
            except ValueError as e:
                continue
        else:
            rendered = render_fallback_plain(system_prompt, prompt_text or "", max_model_len - args.max_tokens, args.model)

        prompts.append(rendered)
        metas.append({
            "prompt": prompt_text or "",
            "id": sample_id
        })

    # Sampling parameters
    sampling = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Generate (batched internally by vLLM)
    outputs = llm.generate(prompts, sampling)

    # Collect results
    results: List[Dict[str, Any]] = []
    for out, meta in zip(outputs, metas):
        if not out.outputs:
            resp_text = ""
        else:
            resp_text = out.outputs[0].text

        rec = {
            "prompt": meta["prompt"],
            "response": resp_text,
        }
        if meta["id"] is not None:
            rec["id"] = meta["id"]
        results.append(rec)

    # Save JSON
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(results)} responses to: {args.output}")


if __name__ == "__main__":
    main()
