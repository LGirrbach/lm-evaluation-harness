#!/usr/bin/env python3
"""
Score preference datasets with vLLM using one of three judge modes:
  - binary (yes/no)
  - rubrics (4 aspects, each 1–5)
  - holistic (single 0–10)

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
        "Output exactly one integer 1–5. Do not add any other text."
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
        "Output exactly one integer 1–5. Do not add any other text."
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
        "Output exactly one integer 1–5. Do not add any other text."
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
        "Output exactly one integer 1–5. Do not add any other text."
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
        "Rate INSTRUCTION FOLLOWING from 1–5. Output exactly one integer 1–5."
    ),
    "info": (
        "[BEGIN PROMPT]\n{prompt}\n[END PROMPT]\n\n"
        "[BEGIN CONTINUATION]\n{continuation}\n[END CONTINUATION]\n\n"
        "Rate INFORMATIVENESS / HELPFULNESS from 1–5. Output exactly one integer 1–5."
    ),
    "honesty": (
        "[BEGIN PROMPT]\n{prompt}\n[END PROMPT]\n\n"
        "[BEGIN CONTINUATION]\n{continuation}\n[END CONTINUATION]\n\n"
        "Rate HONESTY / UNCERTAINTY EXPRESSION from 1–5. Output exactly one integer 1–5."
    ),
    "truth": (
        "[BEGIN PROMPT]\n{prompt}\n[END PROMPT]\n\n"
        "[BEGIN CONTINUATION]\n{continuation}\n[END CONTINUATION]\n\n"
        "Rate TRUTHFULNESS / HALLUCINATION from 1–5. Output exactly one integer 1–5."
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

def extract_pref_row(row: Dict[str, Any]) -> Tuple[str, List[Dict[str, str]], str, List[Dict[str, str]], str, Optional[str]]:
    """Original generic extractor (kept for backwards compatibility)."""
    sample_id = None
    for key in ("id", "sample_id", "example_id"):
        if key in row and row[key] is not None:
            sample_id = str(row[key])
            break

    if isinstance(row.get("chosen"), str) and isinstance(row.get("rejected"), str):
        prm = str(row.get("prompt", ""))
        return (
            prm,
            [{"role": "user", "content": prm}], str(row["chosen"]),
            [{"role": "user", "content": prm}], str(row["rejected"]),
            sample_id,
        )

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

    return (
        "",
        [{"role": "user", "content": ""}], str(row.get("chosen", "")),
        [{"role": "user", "content": ""}], str(row.get("rejected", "")),
        sample_id,
    )


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


# ---------------- Utilities for ARENA-SCHEMA ----------------

def _parse_jsonish_list(x) -> List[str]:
    """Parse strings like '["...", "..."]' into Python lists; pass through if already list/None."""
    if x is None:
        return []
    if isinstance(x, list):
        return [str(y) for y in x]
    if isinstance(x, str):
        x = x.strip()
        # Try JSON, then literal_eval
        try:
            val = json.loads(x)
            if isinstance(val, list):
                return [str(y) for y in val]
        except Exception:
            try:
                val = ast.literal_eval(x)
                if isinstance(val, list):
                    return [str(y) for y in val]
            except Exception:
                # Fallback: treat the whole string as a single entry
                return [x]
    # Anything else: stringified
    return [str(x)]

def _detect_arena_schema(example: Dict[str, Any]) -> bool:
    needed = {"prompt", "response_a", "response_b", "winner_model_a", "winner_model_b", "winner_tie"}
    return all(k in example for k in needed)

def _arena_extract_last_turn(row: Dict[str, Any]) -> Tuple[str, str, str]:
    """Return (prompt_last, resp_a_last, resp_b_last) from arena-schema row."""
    prompts = _parse_jsonish_list(row.get("prompt"))
    ra = _parse_jsonish_list(row.get("response_a"))
    rb = _parse_jsonish_list(row.get("response_b"))
    # Use last available; if lengths mismatch, still take the tail
    prm = prompts[-1] if prompts else str(row.get("prompt", "")) or ""
    ra_last = ra[-1] if ra else str(row.get("response_a", "")) or ""
    rb_last = rb[-1] if rb else str(row.get("response_b", "")) or ""
    return prm, ra_last, rb_last


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
    ap.add_argument("--system", default=None, help="Override system prompt for BINARY mode only.")
    ap.add_argument("--no-chat", action="store_true", help="Disable chat template even if available.")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    # Inference controls
    ap.add_argument("--logprobs-k", type=int, default=5, help="Top-K token logprobs to return for first token.")
    ap.add_argument("--epsilon-floor", type=float, default=1e-9, help="Floor mass if targets absent from top-K.")
    ap.add_argument("--max-examples", type=int, default=None, help="Optional cap for debugging.")
    # Judge mode
    ap.add_argument("--judge-mode", choices=["binary", "rubrics", "holistic"], default="binary",
                    help="binary=yes/no; rubrics=4 aspects (1–5); holistic=single 0–10 score.")
    # ARENA-SCHEMA behavior
    ap.add_argument("--include-ties", action="store_true",
                    help="For arena-style datasets, include rows with winner_tie==1 (labels are not true prefs).")

    args = ap.parse_args()

    # Load dataset
    ds = load_dataset(args.dataset, split=args.split).select(range(args.max_examples))

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

    # Detect schema on first row (safe: iterate until we find a row)
    example0 = ds[0] if len(ds) > 0 else {}
    arena_schema = _detect_arena_schema(example0)

    # ---------------- Build prompt list ----------------
    for idx, row in enumerate(tqdm(ds, total=n_rows)):
        if args.max_examples is not None and idx >= args.max_examples:
            break

        # ---------- ARENA-SCHEMA path ----------
        if arena_schema:
            # Skip ties unless requested
            tie = int(row.get("winner_tie", 0) or 0) == 1
            if tie and not args.include_ties:
                continue

            # Select last turn prompt and last responses
            prompt_render, resp_a, resp_b = _arena_extract_last_turn(row)

            # Determine chosen/rejected (if tie and included, we'll still make a pair but mark is_chosen normally—note: labels not true prefs)
            w_a = int(row.get("winner_model_a", 0) or 0) == 1
            w_b = int(row.get("winner_model_b", 0) or 0) == 1

            if w_a and not w_b:
                chosen_text, rejected_text = resp_a, resp_b
            elif w_b and not w_a:
                chosen_text, rejected_text = resp_b, resp_a
            else:
                # tie or malformed flags: keep deterministic order
                chosen_text, rejected_text = resp_a, resp_b

            sample_id = str(row.get(args.id_column)) if row.get(args.id_column) is not None else None
            pairs = [(True if not tie else None, chosen_text), (False if not tie else None, rejected_text)]

            if args.judge_mode == "binary":
                for is_chosen, resp_text in pairs:
                    user_text = USER_TEMPLATE_BINARY.format(prompt=prompt_render or "", continuation=resp_text or "")
                    rendered = render_for_model(tok, args.system or SYSTEM_PROMPT_BINARY, user_text, use_chat)
                    prompts.append(rendered)
                    meta.append({
                        "sample_index": idx,
                        "sample_id": sample_id,
                        "prompt": prompt_render,
                        "is_chosen": is_chosen,
                        "mode": "binary",
                    })

            elif args.judge_mode == "rubrics":
                for is_chosen, resp_text in pairs:
                    for rubric_key in ["instr", "info", "honesty", "truth"]:
                        sys_p = SYSTEM_PROMPTS_RUBRICS[rubric_key]
                        usr_t = USER_TEMPLATES_RUBRICS[rubric_key].format(
                            prompt=prompt_render or "", continuation=resp_text or ""
                        )
                        rendered = render_for_model(tok, sys_p, usr_t, use_chat)
                        prompts.append(rendered)
                        meta.append({
                            "sample_index": idx,
                            "sample_id": sample_id,
                            "prompt": prompt_render,
                            "is_chosen": is_chosen,
                            "mode": "rubrics",
                            "rubric": rubric_key,
                        })

            else:  # holistic
                for is_chosen, resp_text in pairs:
                    usr_t = USER_TEMPLATE_HOLISTIC.format(
                        prompt=prompt_render or "", continuation=resp_text or ""
                    )
                    rendered = render_for_model(tok, SYSTEM_PROMPT_HOLISTIC, usr_t, use_chat)
                    prompts.append(rendered)
                    meta.append({
                        "sample_index": idx,
                        "sample_id": sample_id,
                        "prompt": prompt_render,
                        "is_chosen": is_chosen,
                        "mode": "holistic",
                    })

            continue  # done with this row

        # ---------- ORIGINAL SCHEMAS path ----------
        prm_text, ch_prefix, ch_resp, rj_prefix, rj_resp, sample_id = extract_pref_row(row)
        if args.id_column in row and row[args.id_column] is not None:
            sample_id = str(row[args.id_column])

        def latest_user(prefix_msgs: List[Dict[str, str]]) -> str:
            for m in reversed(prefix_msgs):
                if m.get("role") == "user":
                    return str(m.get("content", ""))
            return prm_text or ""

        prompt_render = latest_user(ch_prefix)

        pairs = [(True, ch_resp), (False, rj_resp)]

        if args.judge_mode == "binary":
            for is_chosen, resp_text in pairs:
                user_text = USER_TEMPLATE_BINARY.format(prompt=prompt_render or "", continuation=resp_text or "")
                rendered = render_for_model(tok, args.system or SYSTEM_PROMPT_BINARY, user_text, use_chat)
                prompts.append(rendered)
                meta.append({
                    "sample_index": idx,
                    "sample_id": sample_id if sample_id is not None else None,
                    "prompt": prompt_render,
                    "is_chosen": is_chosen,
                    "mode": "binary",
                })

        elif args.judge_mode == "rubrics":
            for is_chosen, resp_text in pairs:
                for rubric_key in ["instr", "info", "honesty", "truth"]:
                    sys_p = SYSTEM_PROMPTS_RUBRICS[rubric_key]
                    usr_t = USER_TEMPLATES_RUBRICS[rubric_key].format(
                        prompt=prompt_render or "", continuation=resp_text or ""
                    )
                    rendered = render_for_model(tok, sys_p, usr_t, use_chat)
                    prompts.append(rendered)
                    meta.append({
                        "sample_index": idx,
                        "sample_id": sample_id if sample_id is not None else None,
                        "prompt": prompt_render,
                        "is_chosen": is_chosen,
                        "mode": "rubrics",
                        "rubric": rubric_key,
                    })

        else:  # holistic
            for is_chosen, resp_text in pairs:
                usr_t = USER_TEMPLATE_HOLISTIC.format(
                    prompt=prompt_render or "", continuation=resp_text or ""
                )
                rendered = render_for_model(tok, SYSTEM_PROMPT_HOLISTIC, usr_t, use_chat)
                prompts.append(rendered)
                meta.append({
                    "sample_index": idx,
                    "sample_id": sample_id if sample_id is not None else None,
                    "prompt": prompt_render,
                    "is_chosen": is_chosen,
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
                    "is_chosen": m["is_chosen"],
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
                "is_chosen": m["is_chosen"],
                "answer": ans,
                "p_yes": float(p_yes),
                "p_no": float(p_no),
                "gen_token": gen_tok,
            })

        out_df = pd.DataFrame(rows)
        out_df.sort_values(["sample_index", "is_chosen"], inplace=True)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        out_df.to_csv(args.output, index=False)

        print(f"✅ Wrote {len(out_df)} rows to: {args.output}")
        print("Columns: sample_index, sample_id, is_chosen, answer, p_yes, p_no, gen_token")

    elif args.judge_mode == "rubrics":
        digit_ids = one_to_five_token_sets(tok)
        agg: Dict[Tuple[int, bool], Dict[str, Any]] = {}

        for out, m in zip(outputs, meta):
            key = (m["sample_index"], m["is_chosen"])
            if key not in agg:
                agg[key] = {
                    "sample_index": m["sample_index"],
                    "sample_id": m["sample_id"],
                    "is_chosen": m["is_chosen"],
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
        out_df.sort_values(["sample_index", "is_chosen"], inplace=True)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        out_df.to_csv(args.output, index=False)

        print(f"✅ Wrote {len(out_df)} rows to: {args.output}")
        print("Columns: sample_index, sample_id, is_chosen, instr_p1..p5, info_p1..p5, honesty_p1..p5, truth_p1..p5")

    else:  # holistic
        class_ids = zero_to_ten_token_sets(tok)
        agg: Dict[Tuple[int, bool], Dict[str, Any]] = {}

        for out, m in zip(outputs, meta):
            key = (m["sample_index"], m["is_chosen"])
            if key not in agg:
                base = {
                    "sample_index": m["sample_index"],
                    "sample_id": m["sample_id"],
                    "is_chosen": m["is_chosen"],
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
        out_df.sort_values(["sample_index", "is_chosen"], inplace=True)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        out_df.to_csv(args.output, index=False)

        print(f"✅ Wrote {len(out_df)} rows to: {args.output}")
        print("Columns: sample_index, sample_id, is_chosen, holistic_p0..holistic_p10")


if __name__ == "__main__":
    main()
