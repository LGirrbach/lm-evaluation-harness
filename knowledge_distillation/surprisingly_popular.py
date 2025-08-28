#!/usr/bin/env python3
import os
import re
import json
import argparse
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams

from dataset_loader import load_dataset, MCQItem  # and optionally load_mmlu_dataset


# =========================
# Shared prompts
# =========================

SYSTEM_PROMPT = """
You are a careful, concise, and literal assistant for answering and analyzing multiple-choice questions about factual claims.

Core rules you MUST follow:
1) When asked to ANSWER a multiple-choice question, you will output EXACTLY one option string, copied VERBATIM from the provided options. Output nothing else.
2) When asked to PREDICT what most people would answer, you will either:
   (a) output EXACTLY one option string (if asked for a single choice), or
   (b) output a FULL probability distribution over ALL options as a JSON object whose keys are the option strings copied VERBATIM and whose values are probabilities in [0,1] that sum to 1.0.
3) Never invent new options or paraphrase any option text.
4) Be meticulous and avoid any extra commentary or formatting.
""".strip()

# Round 1: model's own direct answer
DIRECT_PROMPT_TEMPLATE = """
You will answer a multiple-choice question by selecting exactly one option.

Question:
{question}

Options (choose ONE by copying the option text verbatim):
{options_block}

Output: reply with EXACTLY one of the option strings above. Do NOT add any extra text.
""".strip()

# Round 2 (mode A): "what would most people answer?" — SINGLE wrong option
CROWD_PROMPT_TEMPLATE_SINGLE_WRONG = """
Which WRONG option do you predict MOST PEOPLE would most likely choose as their answer to the same question?

Question:
{question}

Options (choose ONE by copying the option text verbatim):
{options_block}

Output: reply with EXACTLY one of the option strings above. Do NOT add any extra text.
""".strip()

# Round 2 (mode B): Predict FULL distribution over ALL options as JSON
CROWD_PROMPT_TEMPLATE_FULL_DIST = """
You will predict how a broad, non-expert crowd would distribute their answers across the options.

Question:
{question}

Options (predict a probability for EACH option; keys must match the option strings verbatim):
{options_block}

Output: Return ONLY a JSON object mapping each option string to a probability in [0,1] that sums to 1.0.
For example: {{"- A": 0.62, "- B": 0.18, "- C": 0.12, "- D": 0.08}}
Do NOT include any text before or after the JSON.
""".strip()


# =========================
# Utilities
# =========================

def build_options_block(options: List[str]) -> str:
    return "\n".join([f"- {opt}" for opt in options])

def clean_choice(text: str) -> str:
    return text.strip().strip("`").strip('"').strip("'")

def mode_choice(choices: List[str]) -> str:
    c = Counter(choices)
    if not c:
        return ""
    max_count = max(c.values())
    candidates = [k for k, v in c.items() if v == max_count]
    return sorted(candidates)[0]  # tie-breaker

def pairwise_agreement_rate(choices: List[str]) -> float:
    n = len(choices)
    if n <= 1:
        return 1.0
    agree = 0
    total = n * (n - 1) // 2
    for i in range(n):
        for j in range(i + 1, n):
            if choices[i] == choices[j]:
                agree += 1
    return agree / total if total > 0 else 1.0

def majority_share(choices: List[str]) -> float:
    if not choices:
        return 0.0
    c = Counter(choices)
    return max(c.values()) / len(choices)

def compute_accuracy(preds: List[str], golds: List[str]) -> float:
    return np.mean([p == g for p, g in zip(preds, golds)]) if golds else 0.0

def soft_hist_distribution(samples: List[str], options: List[str]) -> Dict[str, float]:
    """Convert categorical samples into a normalized distribution over options."""
    if not samples:
        return {o: 0.0 for o in options}
    c = Counter(samples)
    total = sum(c.values())
    return {o: (c[o] / total) if o in c else 0.0 for o in options}

def try_parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Parse the first JSON object found in text. Returns None if not found/invalid."""
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        try:
            return json.loads(t)
        except Exception:
            pass
    try:
        start = t.index("{")
        end = len(t) - t[::-1].index("}")
        snippet = t[start:end]
        return json.loads(snippet)
    except Exception:
        return None

def normalize_distribution(d: Dict[str, float], options: List[str]) -> Optional[Dict[str, float]]:
    """Keep only known options, clamp negatives to 0, renormalize; return None if empty."""
    if not isinstance(d, dict):
        return None
    filt = {k: float(v) for k, v in d.items() if k in options and np.isfinite(v)}
    if not filt:
        return None
    for k in list(filt.keys()):
        if filt[k] < 0:
            filt[k] = 0.0
    s = sum(filt.values())
    if s <= 0:
        # make uniform over keys present
        n = len(filt)
        if n == 0:
            return None
        u = 1.0 / n
        return {k: u for k in filt}
    return {k: v / s for k, v in filt.items()}


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(description="MCQ two-round sampling with optional full crowd distribution (SP-ready)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to MCQ JSON file (or 'mmlu' for MMLU from Hugging Face).")
    parser.add_argument("--format", type=str, choices=["auto", "truthfulqa", "standard", "mmlu"], default="auto",
                        help="Dataset format.")
    parser.add_argument("--model", type=str, default="microsoft/phi-4",
                        help="HF model name.")
    parser.add_argument("--num-samples", type=int, default=20,
                        help="Number of samples per prompt for each round.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="Max new tokens for each generation.")
    parser.add_argument("--output_dir", type=str, default="results/mcq_two_rounds/",
                        help="Directory to save outputs.")
    parser.add_argument("--crowd-mode", type=str, choices=["single_wrong", "full_dist"], default="single_wrong",
                        help="Round-2 mode: 'single_wrong' (original prompt) or 'full_dist' (JSON distribution).")
    parser.add_argument("--r2-history", choices=["none", "with_r1_answer"], default="none", help="Whether Round-2 should see the Round-1 exchange. 'with_r1_answer' reproduces the original behavior.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    format_type = None if args.format == "auto" else args.format
    if args.dataset.lower() == "mmlu" or format_type == "mmlu":
        print("Loading MMLU dataset from Hugging Face...")
        try:
            from dataset_loader import load_mmlu_dataset
            items: List[MCQItem] = load_mmlu_dataset()
            dataset_name = "mmlu"
            print(f"Loaded {len(items)} questions successfully.")
        except Exception as e:
            print(f"Error loading MMLU dataset: {e}")
            return 1
    else:
        print(f"Loading dataset from: {args.dataset}")
        try:
            items = load_dataset(args.dataset, format_type)
            dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
            print(f"Loaded {len(items)} questions successfully.")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return 1

    # Initialize vLLM
    print(f"Loading model: {args.model} ...")
    quantization_llms = ["zai-org/GLM-4-32B-0414", "LGAI-EXAONE/EXAONE-4.0.1-32B"]
    llm = LLM(
        model=args.model,
        enable_chunked_prefill=True,
        max_num_batched_tokens=32768,
        max_model_len=4096,
        gpu_memory_utilization=0.95,
        trust_remote_code=True,
        quantization='bitsandbytes' if args.model in quantization_llms else None
    )
    tokenizer = llm.get_tokenizer()

    # Round 1 prompts
    print("Preparing Round 1 prompts (direct answers)...")
    round1_prompts = []
    for it in items:
        options_block = build_options_block(it.options)
        user_prompt = DIRECT_PROMPT_TEMPLATE.format(
            question=it.question,
            options_block=options_block
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompt_str = tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            thinking=False
        )
        round1_prompts.append(prompt_str)

    sampling_params_r1 = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=args.num_samples,
        logprobs=0,
    )

    print("Running Round 1 (direct answers) batch inference...")
    outputs1 = llm.generate(round1_prompts, sampling_params_r1)
    print("Round 1 complete.")

    # Collect Round 1 results
    r1_samples_per_q: List[List[str]] = []
    r1_mode_answers: List[str] = []
    for it, out in zip(items, outputs1):
        options = it.options
        samples = []
        for cand in out.outputs:
            ans = clean_choice(cand.text)
            if ans in options:
                samples.append(ans)
        if not samples:
            # don't inject first option silently; mark as empty sample set
            samples = []
        r1_samples_per_q.append(samples)
        r1_mode_answers.append(mode_choice(samples) if samples else (options[0] if options else ""))

    # Round 2 prompts based on mode
    print(f"Preparing Round 2 prompts (mode: {args.crowd_mode}, r2-history: {args.r2_history})...")
    round2_prompts = []
    for idx, it in enumerate(items):
        options_block = build_options_block(it.options)

        if args.crowd_mode == "single_wrong":
            round2_user = CROWD_PROMPT_TEMPLATE_SINGLE_WRONG.format(
                question=it.question, options_block=options_block
            )
        else:
            round2_user = CROWD_PROMPT_TEMPLATE_FULL_DIST.format(
                question=it.question, options_block=options_block
            )

        if args.r2_history == "with_r1_answer":
            # Recreate the R1 user turn and inject the model's R1 mode answer as assistant.
            r1_user = DIRECT_PROMPT_TEMPLATE.format(
                question=it.question, options_block=options_block
            )
            r1_ans = r1_mode_answers[idx] if idx < len(r1_mode_answers) else ""

            # If there was no valid R1 answer, just omit the assistant turn to avoid injecting a fake value.
            if r1_ans:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": r1_user},
                   {"role": "assistant", "content": r1_ans},
                    {"role": "user", "content": round2_user},
                ]
            else:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": round2_user},
                ]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": round2_user},
            ]

        prompt_str = tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            thinking=False
        )
        round2_prompts.append(prompt_str)

    sampling_params_r2 = SamplingParams(
        temperature=args.temperature,
        max_tokens=max(args.max_tokens, 128) if args.crowd_mode == "full_dist" else args.max_tokens,
        n=args.num_samples,
        logprobs=0,
    )

    print("Running Round 2 batch inference...")
    outputs2 = llm.generate(round2_prompts, sampling_params_r2)
    print("Round 2 complete.")

    # Containers shared across modes
    rows = []
    gold = [it.correct for it in items]

    if args.crowd_mode == "single_wrong":
        # === Original behavior: Round-2 returns a single WRONG option per sample ===
        r2_samples_per_q: List[List[str]] = []
        r2_mode_answers: List[str] = []

        for it, out in zip(items, outputs2):
            options = it.options
            samples = []
            for cand in out.outputs:
                ans = clean_choice(cand.text)
                if ans in options:
                    samples.append(ans)
            if not samples:
                samples = []  # keep empty; we'll fall back when saving
            r2_samples_per_q.append(samples)
            r2_mode_answers.append(mode_choice(samples) if samples else (options[0] if options else ""))

        # Save per-question rows (compatible with your original CSV, plus keep fields the same)
        for it, d_mode, c_mode, d_samp, c_samp in zip(
            items, r1_mode_answers, r2_mode_answers, r1_samples_per_q, r2_samples_per_q
        ):
            rows.append({
                "qid": it.qid,
                "question": it.question,
                "correct": it.correct,
                "direct_answer_mode": d_mode,
                "crowd_answer_mode": c_mode,
                "direct_samples_json": json.dumps(d_samp, ensure_ascii=False),
                "crowd_samples_json": json.dumps(c_samp, ensure_ascii=False),
            })

        df = pd.DataFrame(rows)
        true_model_name = args.model.split("/")[-1]
        out_file = os.path.join(args.output_dir, true_model_name, f"{dataset_name}_mcq_two_rounds.csv")
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        df.to_csv(out_file, index=False, encoding="utf-8")
        print(f"Saved per-question results to: {out_file}")

        # Descriptive stats (same style as before)
        direct_mode_acc = compute_accuracy(r1_mode_answers, gold)
        per_q_sample_acc = []
        for it, samples in zip(items, r1_samples_per_q):
            per_q_sample_acc.append(np.mean([s == it.correct for s in samples]) if samples else 0.0)
        avg_sample_acc = float(np.mean(per_q_sample_acc)) if per_q_sample_acc else 0.0

        r1_majority_share = np.mean([majority_share(s) for s in r1_samples_per_q]) if r1_samples_per_q else 0.0
        r2_majority_share = np.mean([majority_share(s) for s in r2_samples_per_q]) if r2_samples_per_q else 0.0
        r1_unanimous_rate = np.mean([len(set(s)) == 1 for s in r1_samples_per_q]) if r1_samples_per_q else 0.0
        r2_unanimous_rate = np.mean([len(set(s)) == 1 for s in r2_samples_per_q]) if r2_samples_per_q else 0.0
        r1_pair_agree = np.mean([pairwise_agreement_rate(s) for s in r1_samples_per_q]) if r1_samples_per_q else 0.0
        r2_pair_agree = np.mean([pairwise_agreement_rate(s) for s in r2_samples_per_q]) if r2_samples_per_q else 0.0
        mode_agree_r1_r2 = np.mean([a == b for a, b in zip(r1_mode_answers, r2_mode_answers)]) if r2_mode_answers else 0.0

        print("\n=== Descriptive stats (single_wrong mode) ===")
        print(f"Direct accuracy (mode answer):            {direct_mode_acc*100:.2f}%")
        print(f"Average direct accuracy over samples:     {avg_sample_acc*100:.2f}%")
        print(f"Round 1 agreement: majority share         {r1_majority_share*100:.2f}%")
        print(f"Round 1 agreement: unanimous rate         {r1_unanimous_rate*100:.2f}%")
        print(f"Round 1 agreement: pairwise agreement     {r1_pair_agree*100:.2f}%")
        print(f"Round 2 agreement: majority share         {r2_majority_share*100:.2f}%")
        print(f"Round 2 agreement: unanimous rate         {r2_unanimous_rate*100:.2f}%")
        print(f"Round 2 agreement: pairwise agreement     {r2_pair_agree*100:.2f}%")
        print(f"Mode agreement between R1 and R2:         {mode_agree_r1_r2*100:.2f}%")
        print("\nDone.")

    else:
        # === Full distribution mode: Round-2 returns JSON distributions; compute SP ===
        # Build empirical popularity from Round-1 samples and predicted popularity from Round-2 distributions.
        p_hat_list: List[Dict[str, float]] = []
        q_hat_list: List[Optional[Dict[str, float]]] = []
        sp_scores_list: List[Optional[Dict[str, float]]] = []
        sp_choice_list: List[str] = []
        direct_mode_list: List[str] = []

        # Round-2: parse and average per question across n samples
        r2_raw_jsons_per_q: List[List[str]] = []
        r2_avg_dist_per_q: List[Optional[Dict[str, float]]] = []

        for it, out in zip(items, outputs2):
            options = it.options
            raw_jsons = []
            parsed = []
            for cand in out.outputs:
                txt = cand.text.strip()
                raw_jsons.append(txt)
                obj = try_parse_json_object(txt)
                if obj is None:
                    continue
                norm = normalize_distribution(obj, options)
                if norm is not None:
                    parsed.append(norm)
            r2_raw_jsons_per_q.append(raw_jsons)
            if parsed:
                # average across samples for this question
                acc = defaultdict(float)
                for d in parsed:
                    for o in options:
                        acc[o] += d.get(o, 0.0)
                avg = {o: acc[o] / len(parsed) for o in options}
                # normalize again to be safe
                r2_avg_dist_per_q.append(normalize_distribution(avg, options))
            else:
                r2_avg_dist_per_q.append(None)

        # Compute p_hat (empirical from R1 samples), direct mode, SP pieces
        for it, samples, qdist in zip(items, r1_samples_per_q, r2_avg_dist_per_q):
            options = it.options
            p_hat = soft_hist_distribution(samples, options)
            p_hat_list.append(p_hat)

            direct_mode_list.append(mode_choice(samples) if samples else (options[0] if options else ""))

            q_hat_list.append(qdist)
            if qdist is not None:
                scores = {o: p_hat.get(o, 0.0) - qdist.get(o, 0.0) for o in options}
                sp_scores_list.append(scores)
                sp_choice_list.append(max(scores.items(), key=lambda kv: kv[1])[0])
            else:
                sp_scores_list.append(None)
                sp_choice_list.append(direct_mode_list[-1])  # fallback to direct mode if no qdist

        # Save per-question CSV (augmented)
        for it, d_mode, sp_choice, p_hat, q_hat, sp_scores, d_samples, raw_jsons in zip(
            items, direct_mode_list, sp_choice_list, p_hat_list, q_hat_list, sp_scores_list, r1_samples_per_q, r2_raw_jsons_per_q
        ):
            rows.append({
                "qid": it.qid,
                "question": it.question,
                "correct": it.correct,
                "direct_answer_mode": d_mode,
                "sp_choice": sp_choice,
                "direct_samples_json": json.dumps(d_samples, ensure_ascii=False),
                "p_hat_json": json.dumps(p_hat, ensure_ascii=False) if p_hat is not None else None,
                "q_hat_json": json.dumps(q_hat, ensure_ascii=False) if q_hat is not None else None,
                "sp_scores_json": json.dumps(sp_scores, ensure_ascii=False) if sp_scores is not None else None,
                "crowd_raw_json_samples": json.dumps(raw_jsons, ensure_ascii=False),
            })

        df = pd.DataFrame(rows)
        true_model_name = args.model.split("/")[-1]
        out_dir = os.path.join(args.output_dir, true_model_name)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{dataset_name}_mcq_two_rounds_full_dist.csv")
        df.to_csv(out_file, index=False, encoding="utf-8")
        print(f"Saved per-question (full_dist) results to: {out_file}")

        # Metrics
        direct_acc = compute_accuracy(direct_mode_list, gold)
        sp_acc = compute_accuracy(sp_choice_list, gold)

        print("\n=== Descriptive stats (full_dist mode) ===")
        print(f"Direct accuracy (mode of R1 samples):     {direct_acc*100:.2f}%")
        print(f"SP choice accuracy:                        {sp_acc*100:.2f}%")

        print("\nDone.")


if __name__ == "__main__":
    main()
