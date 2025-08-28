import os
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

from dataset_loader import load_dataset, load_mmlu_dataset, MCQItem


# =========================
# Prompts (engineered for strict Yes/No)
# =========================

SYSTEM_PROMPT = """
You are a careful, literal grader for multiple-choice questions.
Your job is to decide, for ONE proposed option at a time, whether it is a TRUE and FACTUALLY CORRECT answer to the question.

Hard rules you MUST follow:
1) Reply with EXACTLY one word: "Yes" or "No".
2) No punctuation, no extra words, no explanations, no JSON, no labels.
3) If the option is even partially incorrect or not fully answering the question, reply "No".
4) Do NOT try to pick the best overall option — judge ONLY the single option shown.
""".strip()

YESNO_PROMPT_TEMPLATE = """
Question:
{question}

Candidate option (judge this option only):
{option}

Task: Is the candidate option a fully correct answer to the question?

Output: Reply with EXACTLY one word: Yes or No
""".strip()


# =========================
# Utilities
# =========================

def build_prompt(question: str, option: str) -> str:
    """Render the user message for one (question, option) pair."""
    return YESNO_PROMPT_TEMPLATE.format(question=question, option=option)

def clean_yesno(text: str) -> Optional[str]:
    """
    Normalize and validate the model output as 'Yes' or 'No'.
    Returns 'Yes', 'No', or None if unparseable.
    """
    t = text.strip().strip('`"\'').split()[0] if text.strip() else ""
    t_low = t.lower()
    if t_low.startswith("yes"):
        return "Yes"
    if t_low.startswith("no"):
        return "No"
    return None

def extract_logprob_chain(cand) -> Tuple[float, float]:
    """
    From a vLLM candidate, collect the per-token logprobs (for the actually generated tokens)
    and return (avg_logprob, sum_logprob). We set logprobs=1 so each step returns only
    the generated token with its logprob.
    """
    token_logprobs: List[float] = []
    if getattr(cand, "logprobs", None):
        for step_dict in cand.logprobs:
            # With logprobs=1, we get a dict with a single entry: {generated_token: LogprobInfo}
            if step_dict and isinstance(step_dict, dict):
                lp = list(step_dict.values())[0].logprob
                token_logprobs.append(lp)
    if not token_logprobs:
        return 0.0, 0.0
    return float(np.mean(token_logprobs)), float(np.sum(token_logprobs))

def prob_from_sum_logprob(sum_logprob: float) -> float:
    """
    Convert (approximate) sequence logprob to probability. This is the probability of the *exact*
    generated string under the model’s tokenization, not a calibrated Yes-vs-No comparison.
    """
    try:
        p = float(np.exp(sum_logprob))
        # Avoid underflow/overflow surprises in JSON by clamping to [0,1].
        if p < 0.0: p = 0.0
        if p > 1.0: p = 1.0
        return p
    except Exception:
        return 0.0


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(description="Per-option Yes/No grading for MCQs with vLLM")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to MCQ JSON (TruthfulQA or Standard) or 'mmlu' to load from HF.")
    parser.add_argument("--format", type=str, choices=["auto", "truthfulqa", "standard", "mmlu"], default="auto",
                        help="Dataset format: 'auto' for auto-detect, 'truthfulqa' for original TruthfulQA format, 'standard' for new standardized format, 'mmlu' for MMLU from HF.")
    parser.add_argument("--model", type=str, default="microsoft/phi-4",
                        help="HF model name.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0.0 recommended for deterministic Yes/No).")
    parser.add_argument("--max-tokens", type=int, default=3,
                        help="Max new tokens for each generation (Yes/No fits in 1-2 tokens).")
    parser.add_argument("--output_dir", type=str, default="results/mcq_yesno/",
                        help="Directory to save JSON outputs.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    format_type = None if args.format == "auto" else args.format
    if args.dataset.lower() == "mmlu" or format_type == "mmlu":
        print("Loading MMLU dataset from Hugging Face...")
        try:
            items = load_mmlu_dataset()
            print(f"Loaded {len(items)} questions successfully.")
            dataset_name = "mmlu"
        except Exception as e:
            print(f"Error loading MMLU dataset: {e}")
            return 1
    else:
        print(f"Loading dataset from: {args.dataset}")
        try:
            items = load_dataset(args.dataset, format_type)
            print(f"Loaded {len(items)} questions successfully.")
            dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
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
        max_model_len=3000,
        gpu_memory_utilization=0.95,
        trust_remote_code=True,
        quantization='bitsandbytes' if args.model in quantization_llms else None
    )
    tokenizer = llm.get_tokenizer()

    # Build a single batch over ALL (question, option) pairs
    print("Preparing prompts...")
    prompts: List[str] = []
    index_map: List[Tuple[int, int]] = []  # (question_idx, option_idx)
    for qi, it in enumerate(items):
        for oi, opt in enumerate(it.options):
            user_msg = build_prompt(it.question, opt)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            prompt_str = tokenizer.apply_chat_template(
                conversation=messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
                thinking=False
            )
            prompts.append(prompt_str)
            index_map.append((qi, oi))

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=1,                # one sample per (question, option)
        logprobs=1,         # we only need the generated token's logprob at each step
        stop=None
    )

    # Run one round of generation
    print("Running batch inference (one pass)...")
    outputs = llm.generate(prompts, sampling_params)
    print("Inference complete.")

    # Collect results per question/option
    per_q_results: List[Dict[str, Any]] = [
        {
            "qid": it.qid,
            "question": it.question,
            "correct": it.correct,  # gold
            "options": [
                {
                    "text": opt,
                    "model_answer": None,        # "Yes" | "No" | None
                    "probability": 0.0,          # exp(sum_logprob) of generated answer
                    "avg_logprob": 0.0,
                    "sum_logprob": 0.0
                } for opt in it.options
            ],
        }
        for it in items
    ]

    # Parse model outputs back into per_q_results
    for (qi, oi), out in zip(index_map, outputs):
        # out.outputs is a list of candidates (n=1 here)
        if not out.outputs:
            continue
        cand = out.outputs[0]
        raw_text = cand.text
        yesno = clean_yesno(raw_text)  # "Yes" / "No" / None
        avg_lp, sum_lp = extract_logprob_chain(cand)
        prob = prob_from_sum_logprob(sum_lp)

        per_q_results[qi]["options"][oi]["model_answer"] = yesno
        per_q_results[qi]["options"][oi]["avg_logprob"] = avg_lp
        per_q_results[qi]["options"][oi]["sum_logprob"] = sum_lp
        per_q_results[qi]["options"][oi]["probability"] = prob

    # Optional: derive a predicted answer from the Yes/No signals
    # Strategy: choose the option with the highest probability among those labeled "Yes".
    # If none labeled "Yes", fall back to the option with the highest probability (rare).
    for qres in per_q_results:
        yes_candidates = [
            (i, opt["probability"])
            for i, opt in enumerate(qres["options"])
            if opt["model_answer"] == "Yes"
        ]
        if yes_candidates:
            pred_idx = max(yes_candidates, key=lambda x: x[1])[0]
        else:
            pred_idx = int(np.argmin([opt["probability"] for opt in qres["options"]]))
        qres["predicted_answer"] = qres["options"][pred_idx]["text"]

    # Summary metrics (purely descriptive, optional)
    golds = [it.correct for it in items]
    preds = [q["predicted_answer"] for q in per_q_results]
    accuracy = float(np.mean([p == g for p, g in zip(preds, golds)])) if golds else 0.0
    print(f"Predicted-answer accuracy (derived from Yes/No): {accuracy*100:.2f}%")

    # Save JSON
    true_model_name = args.model.split("/")[-1]
    out_file = os.path.join(args.output_dir, true_model_name, f"{dataset_name}_per_option_yesno.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    payload = {
        "dataset_name": dataset_name,
        "model": args.model,
        "format_version": "1.0",
        "num_questions": len(items),
        "notes": "Per-option Yes/No grading with one pass. 'probability' is exp(sum_logprob) of the generated Yes/No string.",
        "summary": {
            "predicted_answer_accuracy": accuracy
        },
        "items": per_q_results
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved results to: {out_file}")
    print("Done.")


if __name__ == "__main__":
    main()
