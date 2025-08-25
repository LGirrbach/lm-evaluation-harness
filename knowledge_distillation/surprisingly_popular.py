import os
import json
import argparse
from collections import Counter
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams

from dataset_loader import load_dataset, MCQItem


# =========================
# Shared prompts
# =========================

SYSTEM_PROMPT = """
You are a careful, concise, and literal assistant for answering and analyzing multiple-choice questions about factual claims.

Core rules you MUST follow:
1) When asked to ANSWER a multiple-choice question, you will output EXACTLY one option string, copied VERBATIM from the provided options. Output nothing else.
2) When asked to PREDICT what most people would answer, you will output EXACTLY one option string, copied VERBATIM from the provided options. Output nothing else.
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

# Round 2: "what would most people answer?"
# Includes the Round-1 exchange in the chat history (assistant's direct answer).
CROWD_PROMPT_TEMPLATE = """
Ignoring correctness, which ONE option do you predict MOST PEOPLE would choose as their answer to the same question?

Question:
{question}

Options (choose ONE by copying the option text verbatim):
{options_block}

Output: reply with EXACTLY one of the option strings above. Do NOT add any extra text.
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
    return sorted(candidates)[0]  # tie-breaker for reproducibility

def pairwise_agreement_rate(choices: List[str]) -> float:
    """
    Proportion of agreeing pairs among all pairs.
    """
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


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(description="Multiple-choice questions two-round sampling with vLLM")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to multiple-choice questions JSON file (or 'mmlu' for MMLU dataset from Hugging Face).")
    parser.add_argument("--format", type=str, choices=["auto", "truthfulqa", "standard", "mmlu"], default="auto",
                        help="Dataset format: 'auto' for auto-detection, 'truthfulqa' for original TruthfulQA format, 'standard' for new standardized format, 'mmlu' for MMLU from Hugging Face.")
    parser.add_argument("--model", type=str, default="microsoft/phi-4",
                        help="HF model name.")
    parser.add_argument("--num-samples", type=int, default=20,
                        help="Number of samples per prompt for each round.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=32,
                        help="Max new tokens for each generation.")
    parser.add_argument("--output_dir", type=str, default="results/mcq_two_rounds/",
                        help="Directory to save CSV outputs.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset using unified loader
    format_type = None if args.format == "auto" else args.format
    
    # Special handling for MMLU dataset
    if args.dataset.lower() == "mmlu" or format_type == "mmlu":
        print("Loading MMLU dataset from Hugging Face...")
        try:
            from dataset_loader import load_mmlu_dataset
            items = load_mmlu_dataset()
            print(f"Loaded {len(items)} questions successfully.")
        except Exception as e:
            print(f"Error loading MMLU dataset: {e}")
            return 1
    else:
        # Load from file
        print(f"Loading dataset from: {args.dataset}")
        if format_type:
            print(f"Using format: {format_type}")
        else:
            print("Auto-detecting format...")
        
        try:
            items = load_dataset(args.dataset, format_type)
            print(f"Loaded {len(items)} questions successfully.")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return 1

    # Initialize vLLM
    print(f"Loading model: {args.model} ...")
    llm = LLM(
        model=args.model,
        enable_chunked_prefill=True,
        max_num_batched_tokens=32768,
        max_model_len=3000,
        gpu_memory_utilization=0.95,
        trust_remote_code=True
    )
    tokenizer = llm.get_tokenizer()

    # ================
    # Round 1: Direct answers (sample n times)
    # ================
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
            add_generation_prompt=True
        )
        round1_prompts.append(prompt_str)

    sampling_params_r1 = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=args.num_samples,
    )

    print("Running Round 1 (direct answers) batch inference...")
    outputs1 = llm.generate(round1_prompts, sampling_params_r1)
    print("Round 1 complete.")

    r1_samples_per_q: List[List[str]] = []
    r1_mode_answers: List[str] = []
    for it, out in zip(items, outputs1):
        options = it.options
        samples = []
        for cand in out.outputs:
            ans = clean_choice(cand.text)
            if ans in options:
                samples.append(ans)
        # fallback: if nothing valid, pick first option
        if not samples:
            samples = [options[0]]
        r1_samples_per_q.append(samples)
        r1_mode_answers.append(mode_choice(samples))

    # ================
    # Round 2: Predicted-popular (sample n times)
    # Include Round 1 Q&A in chat history
    # ================
    print("Preparing Round 2 prompts (predicted popular answer)...")
    round2_prompts = []
    for it, direct_mode in zip(items, r1_mode_answers):
        options_block = build_options_block(it.options)
        round1_user = DIRECT_PROMPT_TEMPLATE.format(
            question=it.question,
            options_block=options_block
        )
        round2_user = CROWD_PROMPT_TEMPLATE.format(
            question=it.question,
            options_block=options_block
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": round1_user},
            {"role": "assistant", "content": direct_mode},
            {"role": "user", "content": round2_user},
        ]
        prompt_str = tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True
        )
        round2_prompts.append(prompt_str)

    sampling_params_r2 = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=args.num_samples,
    )

    print("Running Round 2 (predicted popular answers) batch inference...")
    outputs2 = llm.generate(round2_prompts, sampling_params_r2)
    print("Round 2 complete.")

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
            samples = [options[0]]
        r2_samples_per_q.append(samples)
        r2_mode_answers.append(mode_choice(samples))

    # ================
    # Save results (per question)
    # ================
    rows = []
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
    
    # Extract dataset name for the save path
    if args.dataset.lower() == "mmlu" or args.format == "mmlu":
        dataset_name = "mmlu"
    else:
        dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    
    out_file = os.path.join(args.output_dir, true_model_name, f"{dataset_name}_mcq_two_rounds.csv")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    df.to_csv(out_file, index=False, encoding="utf-8")
    print(f"Saved per-question results to: {out_file}")

    # ================
    # Descriptive stats (over samples) — no cross-model evaluation
    # ================
    gold = [it.correct for it in items]

    # Direct accuracy of the per-question MODE
    direct_mode_acc = compute_accuracy(r1_mode_answers, gold)

    # Average direct accuracy across samples (treat each sample as a candidate)
    # i.e., for each question, accuracy is (correct count / n)
    per_q_sample_acc = []
    for it, samples in zip(items, r1_samples_per_q):
        per_q_sample_acc.append(np.mean([s == it.correct for s in samples]))
    avg_sample_acc = float(np.mean(per_q_sample_acc))

    # Agreement/consistency within each round
    r1_majority_share = np.mean([majority_share(s) for s in r1_samples_per_q])
    r2_majority_share = np.mean([majority_share(s) for s in r2_samples_per_q])
    r1_unanimous_rate = np.mean([len(set(s)) == 1 for s in r1_samples_per_q])
    r2_unanimous_rate = np.mean([len(set(s)) == 1 for s in r2_samples_per_q])
    r1_pair_agree = np.mean([pairwise_agreement_rate(s) for s in r1_samples_per_q])
    r2_pair_agree = np.mean([pairwise_agreement_rate(s) for s in r2_samples_per_q])

    # Agreement between Round-1 mode and Round-2 mode
    mode_agree_r1_r2 = np.mean([a == b for a, b in zip(r1_mode_answers, r2_mode_answers)])

    print("\n=== Descriptive stats (within this run) ===")
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


if __name__ == "__main__":
    main()
