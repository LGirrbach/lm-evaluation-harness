#!/usr/bin/env python3
"""
Score preference datasets with vLLM using one of three judge modes:
  - binary (yes/no)
  - rubrics (4 aspects, each 1-5)
  - holistic (single 0-10)

This script generates exactly 1 token and reads the top-K logprob distribution
for that first token. It aggregates *raw* probability masses (i.e., sums of
exp(logprob) for target token sets) WITHOUT re-normalizing across classes.

CSV outputs:

binary:
  sample_index, sample_id, is_chosen, answer, p_yes, p_no, gen_token

rubrics:
  sample_index, sample_id, is_chosen,
  instr_p1..instr_p5, info_p1..info_p5, honesty_p1..honesty_p5, truth_p1..truth_p5

holistic:
  sample_index, sample_id, is_chosen, holistic_p0..holistic_p10
"""

from __future__ import annotations
import argparse
import math
import os
import json
import ast
from typing import Any, Dict, List, Optional, Tuple, Set

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams


# ---------------- System prompts ----------------

SYSTEM_PROMPT_BINARY = (
    "You are a strict binary rater for assistant responses. "
    "Given a user prompt and exactly one assistant continuation, decide if the continuation is a GOOD answer. "
    "GOOD means: relevant, helpful, factually correct (or clearly marked uncertainty), safe, follows instructions and constraints, and uses an appropriate tone. "
    "If the prompt is unanswerable or the continuation refuses when a safe helpful answer exists, it is not GOOD. "
    "Output exactly one token: yes or no. Do not add any other text."
)

SYSTEM_PROMPTS_RUBRICS = {
    "instr": (
        "You are a strict rater for assistant responses. "
        "Given a user prompt and exactly one assistant continuation, evaluate INSTRUCTION FOLLOWING. "
        "Instruction Following means: alignment between output and intent, assessing understanding of the task goal and any restrictions (styles, formats, designated methods). "
        "Rate from 1 to 5: "
        "1 = Irrelevant: No alignment. "
        "2 = Partial Focus: Addresses one aspect poorly. "
        "3 = Partial Compliance: Meets either goal or restrictions, or acknowledges both with slight deviations. "
        "4 = Almost There: Near alignment, minor deviations. "
        "5 = Comprehensive Compliance: Fully aligns, meets all requirements. "
        "Output exactly one integer 1-5. Do not add any other text."
    ),
    "info": (
        "You are a strict rater for assistant responses. "
        "Given a user prompt and exactly one assistant continuation, evaluate INFORMATIVENESS and HELPFULNESS. "
        "Emphasize correctness and informativeness. "
        "Rate from 1 to 5: "
        "1 = Severely Incorrect: significant inaccuracies or fabrication. "
        "2 = Partially Incorrect: errors that may cause confusion. "
        "3 = Correct: accurate and useful, meets the task. "
        "4 = Highly Informative: accurate, extensive, valuable insights. "
        "5 = Outstandingly Helpful: accurate, in-depth, comprehensive. "
        "Output exactly one integer 1-5. Do not add any other text."
    ),
    "honesty": (
        "You are a strict rater for assistant responses. "
        "Given a user prompt and exactly one assistant continuation, evaluate HONESTY and UNCERTAINTY EXPRESSION. "
        "Assess whether confidence aligns with correctness and whether uncertainty is properly expressed. "
        "Rate from 1 to 5: "
        "1 = Confidently Incorrect. "
        "2 = Confident with major mistakes OR unconfident and wrong. "
        "3 = Uncertain or minor errors; or refusal without sufficient reason. "
        "4 = Correct but uncertain; or subtle mistakes with expressed doubt. "
        "5 = Correct and confident; or precisely acknowledges minor uncertainty. "
        "Output exactly one integer 1-5. Do not add any other text."
    ),
    "truth": (
        "You are a strict rater for assistant responses. "
        "Given a user prompt and exactly one assistant continuation, evaluate TRUTHFULNESS and HALLUCINATION. "
        "Check for factual accuracy and absence of fabricated or misleading details. "
        "Rate from 1 to 5: "
        "1 = Completely Hallucinated. "
        "2 = Severe Hallucination: large parts incorrect or misleading. "
        "3 = Partial Hallucination / Misunderstanding. "
        "4 = Insignificant Hallucination: largely truthful with minor issues. "
        "5 = No Hallucination: entirely truthful and accurate. "
        "Output exactly one integer 1-5. Do not add any other text."
    ),
}

SYSTEM_PROMPT_HOLISTIC = (
    "You are a strict evaluator of assistant responses. "
    "Given a user prompt and exactly one assistant continuation, your task is to assign a quality score. "
    "Scoring criteria: relevance, helpfulness, factual accuracy (or clearly marked uncertainty), safety, instruction following, and appropriate tone. "
    "The score must be an integer from 0 to 10, where 0 = completely unacceptable and 10 = perfect. "
    "Output exactly one integer in this range, nothing else. "
    "Do not add explanations, words, or formatting."
)

USER_TEMPLATE_BINARY = (
    "[BEGIN PROMPT]\n{prompt}\n[END PROMPT]\n\n"
    "[BEGIN CONTINUATION]\n{continuation}\n[END CONTINUATION]\n\n"
    "Is the continuation a GOOD answer to the prompt?\n"
    "Answer with exactly one token: yes or no."
)

USER_TEMPLATES_RUBRICS = {
    "instr": (
        "[BEGIN PROMPT]\n{prompt}\n[END PROMPT]\n\n"
        "[BEGIN CONTINUATION]\n{continuation}\n[END CONTINUATION]\n\n"
        "Rate INSTRUCTION FOLLOWING from 1-5. Output exactly one integer 1-5."
    ),
    "info": (
        "[BEGIN PROMPT]\n{prompt}\n[END PROMPT]\n\n"
        "[BEGIN CONTINUATION]\n{continuation}\n[END CONTINUATION]\n\n"
        "Rate INFORMATIVENESS / HELPFULNESS from 1-5. Output exactly one integer 1-5."
    ),
    "honesty": (
        "[BEGIN PROMPT]\n{prompt}\n[END PROMPT]\n\n"
        "[BEGIN CONTINUATION]\n{continuation}\n[END CONTINUATION]\n\n"
        "Rate HONESTY / UNCERTAINTY EXPRESSION from 1-5. Output exactly one integer 1-5."
    ),
    "truth": (
        "[BEGIN PROMPT]\n{prompt}\n[END PROMPT]\n\n"
        "[BEGIN CONTINUATION]\n{continuation}\n[END CONTINUATION]\n\n"
        "Rate TRUTHFULNESS / HALLUCINATION from 1-5. Output exactly one integer 1-5."
    ),
}

USER_TEMPLATE_HOLISTIC = (
    "[BEGIN PROMPT]\n{prompt}\n[END PROMPT]\n\n"
    "[BEGIN CONTINUATION]\n{continuation}\n[END CONTINUATION]\n\n"
    "How good is the continuation as an answer to the prompt?\n"
    "Answer with exactly one integer from 0 to 10."
)


# ---------------- Chat templating helpers ----------------

def has_chat_template(tok) -> bool:
    return hasattr(tok, "apply_chat_template")

def render_for_model(tok, system_prompt: Optional[str], user_text: str, use_chat: bool) -> str:
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
        prefix = (system_prompt + "\n\n") if system_prompt else ""
        return prefix + user_text


# ---------------- Dataset normalization ----------------

from typing import Tuple  # ensure available in some environments


# ---------------- Token sets ----------------

def _single_token_ids(tok, variants: List[str]) -> Dict[str, List[int]]:
    return {v: tok(v, add_special_tokens=False).input_ids for v in variants}

def yes_no_token_sets(tok) -> Tuple[Set[int], Set[int]]:
    variants = ["yes", "Yes", "YES", " yes", " Yes", "no", "No", "NO", " no", " No"]
    enc = _single_token_ids(tok, variants)
    y_ids = {ids[0] for v, ids in enc.items() if v.strip().lower() == "yes" and len(ids) == 1}
    n_ids = {ids[0] for v, ids in enc.items() if v.strip().lower() == "no"  and len(ids) == 1}
    if not y_ids:
        ids = tok("yes", add_special_tokens=False).input_ids
        if ids: y_ids = {ids[0]}
    if not n_ids:
        ids = tok("no", add_special_tokens=False).input_ids
        if ids: n_ids = {ids[0]}
    return y_ids, n_ids

def one_to_five_token_sets(tok) -> Dict[str, Set[int]]:
    variants = []
    for d in ["1","2","3","4","5"]:
        variants += [d, f" {d}"]
    enc = _single_token_ids(tok, variants)
    cls: Dict[str, Set[int]] = {str(i): set() for i in range(1,6)}
    for v, ids in enc.items():
        s = v.strip()
        if s in cls and len(ids) == 1:
            cls[s].add(ids[0])
    for d in ["1","2","3","4","5"]:
        if not cls[d]:
            ids = tok(d, add_special_tokens=False).input_ids
            if ids: cls[d].add(ids[0])
    return cls

def zero_to_ten_token_sets(tok) -> Dict[str, Set[int]]:
    """
    Collect single-token ids for '0'..'10'. Only includes '10' if it is a SINGLE token.
    Space-prefixed variants are also considered.
    """
    digits = [str(i) for i in range(0, 11)]
    variants = []
    for d in digits:
        variants += [d, f" {d}"]
    enc = _single_token_ids(tok, variants)

    cls: Dict[str, Set[int]] = {d: set() for d in digits}
    for v, ids in enc.items():
        s = v.strip()
        if s in cls and len(ids) == 1:
            cls[s].add(ids[0])

    # Fallbacks: try bare form for any missing class; only accept if single token
    for d in digits:
        if not cls[d]:
            ids = tok(d, add_special_tokens=False).input_ids
            if len(ids) == 1:
                cls[d].add(ids[0])
            # If still empty (e.g., "10" split into two tokens), we'll leave it empty;
            # its mass will come solely from epsilon at scoring time.
    return cls


# ---------------- Scoring (RAW masses) ----------------

def compute_binary_mass(lp_dict: Dict[int, Any], yes_ids: Set[int], no_ids: Set[int], eps: float = 1e-9) -> Tuple[float, float]:
    y_mass = sum(math.exp(obj.logprob) for tid, obj in lp_dict.items() if tid in yes_ids)
    n_mass = sum(math.exp(obj.logprob) for tid, obj in lp_dict.items() if tid in no_ids)
    if y_mass == 0.0: y_mass = eps
    if n_mass == 0.0: n_mass = eps
    return y_mass, n_mass

def compute_multiclass_mass(lp_dict: Dict[int, Any], class_token_sets: Dict[str, Set[int]], eps: float = 1e-9) -> Dict[str, float]:
    """
    Return RAW masses (sum of exp(logprob)) for each class key using top-k logprobs dict.
    Does NOT renormalize; total mass across classes <= 1 (plus eps where missing).
    """
    masses: Dict[str, float] = {}
    for k, idset in class_token_sets.items():
        m = sum(math.exp(obj.logprob) for tid, obj in lp_dict.items() if tid in idset)
        masses[k] = m if m > 0.0 else eps
    # Keep raw masses; caller may aggregate/compare as needed.
    return masses


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--response-json", required=True, help="Path to JSON from generate_responses.py.")
    ap.add_argument("--output", required=True, help="Output CSV path.")
    # Model
    ap.add_argument("--model", required=True, help="HF model id or local path (causal LM).")
    ap.add_argument("--system", default=None, help="Override system prompt for BINARY mode only.")
    ap.add_argument("--no-chat", action="store_true", help="Disable chat template even if available.")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    # Inference controls
    ap.add_argument("--logprobs-k", type=int, default=5, help="Top-K token logprobs to return for first token.")
    ap.add_argument("--epsilon-floor", type=float, default=1e-9, help="Floor mass if targets absent from top-K.")
    ap.add_argument("--max-examples", type=int, default=None, help="Optional cap for debugging.")
    # Judge mode
    ap.add_argument("--judge-mode", choices=["binary", "rubrics", "holistic"], default="binary",
                    help="binary=yes/no; rubrics=4 aspects (1-5); holistic=single 0-10 score.")

    args = ap.parse_args()

    # Load reso
    with open(args.response_json, "r") as f:
        ds = json.load(f)

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

    n_responses_to_rate = len(ds) if args.max_examples is None else min(args.max_examples, len(ds))

    # ---------------- Build prompt list ----------------
    for idx, response_to_rate in enumerate(tqdm(ds, total=n_responses_to_rate)):
        if args.max_examples is not None and idx >= args.max_examples:
            break

        sample_id = response_to_rate["id"]
        prompt = response_to_rate["prompt"]
        response = response_to_rate["response"]

        if args.judge_mode == "binary":
            user_text = USER_TEMPLATE_BINARY.format(prompt=prompt or "", continuation=response or "")
            rendered = render_for_model(tok, args.system or SYSTEM_PROMPT_BINARY, user_text, use_chat)
            prompts.append(rendered)
            meta.append({
                "sample_index": idx,
                "sample_id": sample_id if sample_id is not None else None,
                "prompt": prompt,
                "mode": "binary",
            })

        elif args.judge_mode == "rubrics":
            for rubric_key in ["instr", "info", "honesty", "truth"]:
                sys_p = SYSTEM_PROMPTS_RUBRICS[rubric_key]
                usr_t = USER_TEMPLATES_RUBRICS[rubric_key].format(
                    prompt=prompt or "", continuation=response or ""
                )
                rendered = render_for_model(tok, sys_p, usr_t, use_chat)
                prompts.append(rendered)
                meta.append({
                    "sample_index": idx,
                    "sample_id": sample_id if sample_id is not None else None,
                    "prompt": prompt,
                    "mode": "rubrics",
                    "rubric": rubric_key,
                })

        else:  # holistic
            usr_t = USER_TEMPLATE_HOLISTIC.format(
                prompt=prompt or "", continuation=response or ""
            )
            rendered = render_for_model(tok, SYSTEM_PROMPT_HOLISTIC, usr_t, use_chat)
            prompts.append(rendered)
            meta.append({
                "sample_index": idx,
                "sample_id": sample_id if sample_id is not None else None,
                "prompt": prompt,
                "mode": "holistic",
            })

    # Generation params
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=args.logprobs_k,
    )

    # Generate
    outputs = llm.generate(prompts, sampling)

    rows: List[Dict[str, Any]] = []

    if args.judge_mode == "binary":
        yes_ids, no_ids = yes_no_token_sets(tok)
        for out, m in zip(outputs, meta):
            if not out.outputs:
                rows.append({
                    "sample_index": m["sample_index"],
                    "sample_id": m["sample_id"],
                    "answer": "no",
                    "p_yes": float(args.epsilon_floor),
                    "p_no": float(args.epsilon_floor),
                    "gen_token": "",
                })
                continue

            first = out.outputs[0]
            gen_tid = first.token_ids[0]
            gen_tok = tok.convert_ids_to_tokens([gen_tid])[0] if hasattr(tok, "convert_ids_to_tokens") else tok.decode([gen_tid])
            lp_dict = first.logprobs[0]

            p_yes, p_no = compute_binary_mass(lp_dict, yes_ids, no_ids, eps=args.epsilon_floor)

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
                    ans = "yes" if p_yes >= p_no else "no"

            rows.append({
                "sample_index": m["sample_index"],
                "sample_id": m["sample_id"],
                "answer": ans,
                "p_yes": float(p_yes),
                "p_no": float(p_no),
                "gen_token": gen_tok,
            })

        out_df = pd.DataFrame(rows)
        out_df.sort_values(["sample_index"], inplace=True)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        out_df.to_csv(args.output, index=False)

        print(f"✅ Wrote {len(out_df)} rows to: {args.output}")
        print("Columns: sample_index, sample_id, answer, p_yes, p_no, gen_token")

    elif args.judge_mode == "rubrics":
        digit_ids = one_to_five_token_sets(tok)
        agg: Dict[Tuple[int, bool], Dict[str, Any]] = {}

        for out, m in zip(outputs, meta):
            key = m["sample_index"]
            if key not in agg:
                agg[key] = {
                    "sample_index": m["sample_index"],
                    "sample_id": m["sample_id"],
                    **{f"instr_p{i}": None for i in range(1,6)},
                    **{f"info_p{i}": None for i in range(1,6)},
                    **{f"honesty_p{i}": None for i in range(1,6)},
                    **{f"truth_p{i}": None for i in range(1,6)},
                }

            if not out.outputs:
                probs = {str(i): float(args.epsilon_floor) for i in range(1,6)}
            else:
                first = out.outputs[0]
                lp_dict = first.logprobs[0]
                probs = compute_multiclass_mass(lp_dict, digit_ids, eps=args.epsilon_floor)

            rub = m.get("rubric")
            for i in range(1, 6):
                agg[key][f"{rub}_p{i}"] = float(probs[str(i)])

        rows = list(agg.values())
        out_df = pd.DataFrame(rows)
        out_df.sort_values(["sample_index"], inplace=True)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        out_df.to_csv(args.output, index=False)

        print(f"✅ Wrote {len(out_df)} rows to: {args.output}")
        print("Columns: sample_index, sample_id, instr_p1..p5, info_p1..p5, honesty_p1..p5, truth_p1..p5")

    else:  # holistic
        class_ids = zero_to_ten_token_sets(tok)
        agg: Dict[Tuple[int, bool], Dict[str, Any]] = {}

        for out, m in zip(outputs, meta):
            key = m["sample_index"]
            if key not in agg:
                base = {
                    "sample_index": m["sample_index"],
                    "sample_id": m["sample_id"],
                }
                for i in range(0, 11):
                    base[f"holistic_p{i}"] = None
                agg[key] = base

            if not out.outputs:
                probs = {str(i): float(args.epsilon_floor) for i in range(0,11)}
            else:
                first = out.outputs[0]
                lp_dict = first.logprobs[0]
                probs = compute_multiclass_mass(lp_dict, class_ids, eps=args.epsilon_floor)

            for i in range(0, 11):
                agg[key][f"holistic_p{i}"] = float(probs[str(i)])

        rows = list(agg.values())
        out_df = pd.DataFrame(rows)
        out_df.sort_values(["sample_index"], inplace=True)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        out_df.to_csv(args.output, index=False)

        print(f"✅ Wrote {len(out_df)} rows to: {args.output}")
        print("Columns: sample_index, sample_id, holistic_p0..holistic_p10")


if __name__ == "__main__":
    main()
