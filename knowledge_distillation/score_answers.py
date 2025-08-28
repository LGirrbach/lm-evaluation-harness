import os
import json
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams

from dataset_loader import load_dataset, load_mmlu_dataset, MCQItem


# =========================
# Prompts
# =========================

SYSTEM_PROMPT = """
You are a careful, concise, and literal assistant for answering and analyzing multiple-choice questions about factual claims.

Core rules you MUST follow:
1) When asked to ANSWER a multiple-choice question, you will output EXACTLY one option string, copied VERBATIM from the provided options. Output nothing else.
2) Never invent new options or paraphrase any option text.
3) Be meticulous and avoid any extra commentary or formatting.
""".strip()

# We will *score* each option via teacher-forcing by appending an assistant
# message that contains exactly the option. The user turn sets the context.
SCORING_PROMPT_TEMPLATE = """
You will answer a multiple-choice question by selecting exactly one option.

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

def apply_chat(tokenizer, system: str, user: str, assistant: str = None) -> str:
    """
    Render a chat transcript into a single prompt string using the model's chat template.
    """
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    if assistant is not None:
        messages.append({"role": "assistant", "content": assistant})
    prompt = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=False,
        add_generation_prompt=False,  # we are not asking the model to continue
        enable_thinking=False,
        thinking=False
    )
    return prompt

def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text).input_ids)

def safe_mean(x: List[float]) -> float:
    return float(np.mean(x)) if len(x) > 0 else 0.0


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(description="Per-option log-likelihood scoring for MCQs with vLLM (single pass)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to multiple-choice questions JSON file (or 'mmlu' for MMLU dataset from Hugging Face).")
    parser.add_argument("--format", type=str, choices=["auto", "truthfulqa", "standard", "mmlu"], default="auto",
                        help="Dataset format: 'auto' for auto-detection, 'truthfulqa' for original TruthfulQA format, 'standard' for new standardized format, 'mmlu' for MMLU from Hugging Face.")
    parser.add_argument("--model", type=str, default="microsoft/phi-4",
                        help="HF model name.")
    parser.add_argument("--output_dir", type=str, default="results/mcq_option_ll/",
                        help="Directory to save outputs.")
    parser.add_argument("--max-model-len", type=int, default=3000,
                        help="Max model context length to configure for vLLM engine.")
    parser.add_argument("--gpu-mem-frac", type=float, default=0.95,
                        help="GPU memory utilization fraction for vLLM.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------
    # Load dataset
    # -----------------
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
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return 1
        dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]

    # -----------------
    # Initialize vLLM
    # -----------------
    print(f"Loading model: {args.model} ...")
    quantization_llms = ["zai-org/GLM-4-32B-0414", "LGAI-EXAONE/EXAONE-4.0.1-32B"]
    llm = LLM(
        model=args.model,
        enable_chunked_prefill=True,
        max_num_batched_tokens=32768,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_frac,
        trust_remote_code=True,
        quantization='bitsandbytes' if args.model in quantization_llms else None
    )
    tokenizer = llm.get_tokenizer()

    # -----------------
    # Build prompts for ALL (question, option) pairs
    #   - For each option we will create:
    #       * base_prompt: system + user (for token boundary locating)
    #       * scored_prompt: system + user + assistant(option)
    #   - We will feed ONLY the scored_prompts to vLLM with prompt_logprobs=True
    #     and max_tokens=1 (generate 1 token to avoid vLLM error).
    #   - Then we slice the returned prompt_logprobs to the assistant segment.
    # -----------------
    all_scored_prompts: List[str] = []
    # For later slicing
    assistant_token_lengths: List[int] = []
    base_token_lengths: List[int] = []
    triplets_index: List[Tuple[int, int]] = []  # (item_idx, option_idx)

    print("Preparing prompts for scoring...")
    for i, it in enumerate(items):
        options_block = build_options_block(it.options)
        user_prompt = SCORING_PROMPT_TEMPLATE.format(
            question=it.question,
            options_block=options_block
        )

        # Base prompt without assistant (for token boundary)
        base_prompt = apply_chat(tokenizer, SYSTEM_PROMPT, user_prompt, assistant=None)
        base_len = count_tokens(tokenizer, base_prompt)

        for j, opt in enumerate(it.options):
            # Scored prompt includes assistant content = exact option
            scored_prompt = apply_chat(tokenizer, SYSTEM_PROMPT, user_prompt, assistant=opt)
            scored_len = count_tokens(tokenizer, scored_prompt)
            asst_len = scored_len - base_len
            if asst_len <= 0:
                # Fallback guard (shouldn't happen)
                asst_len = max(1, asst_len)

            all_scored_prompts.append(scored_prompt)
            base_token_lengths.append(base_len)
            assistant_token_lengths.append(asst_len)
            triplets_index.append((i, j))

    print(f"Total (question, option) pairs to score: {len(all_scored_prompts)}")

    # -----------------
    # Run a SINGLE inference pass over all scored prompts
    # -----------------
    sampling = SamplingParams(
        max_tokens=1,            # generate 1 token to avoid vLLM error
        prompt_logprobs=True,    # request logprobs over prompt tokens
        logprobs=0               # we do not need top-k for generated tokens
    )

    print("Scoring options with prompt_logprobs...")
    outputs = llm.generate(all_scored_prompts, sampling)

    # -----------------
    # Extract per-option log-likelihoods
    # -----------------
    # We will gather rows: one row per (qid, option)
    rows = []
    per_q_best_option = {}  # qid -> (best_option, sum_logprob)

    for (item_idx, opt_idx), out, base_len, asst_len in zip(
        triplets_index, outputs, base_token_lengths, assistant_token_lengths
    ):
        it: MCQItem = items[item_idx]
        opt_text: str = it.options[opt_idx]

        # vLLM returns out.prompt_logprobs as a list equal to #prompt_tokens
        # Each element is either None (for the very first token) or a dict-like
        # mapping of candidate tokens to objects with .logprob. We want the
        # logprob of the *actual* prompt token at that position.
        prompt_lp = out.prompt_logprobs or []

        # Identify slice for assistant segment: last `asst_len` tokens
        # (but exclude the generated token, so we only get the prompt portion)
        start = max(0, len(prompt_lp) - asst_len)
        asst_logprobs: List[float] = []

        # Extract the taken-token logprob for each assistant token.
        # We iterate only through the assistant tokens in the prompt, ignoring the generated token.
        for entry in prompt_lp[start:start+asst_len]:
            if not entry:
                continue
            try:
                # Typical structure: dict[token_str] -> Logprob(token, logprob)
                logprob = list(entry.values())[0].logprob
                asst_logprobs.append(float(logprob))
            except Exception:
                # Fallback if structure differs
                try:
                    # Some versions expose 'chosen_logprob'
                    if hasattr(entry, "chosen_logprob"):
                        asst_logprobs.append(float(entry.chosen_logprob))
                    else:
                        asst_logprobs.append(0.0)
                except Exception:
                    asst_logprobs.append(0.0)

        sum_ll = float(np.sum(asst_logprobs)) if asst_logprobs else 0.0
        avg_ll = float(np.mean(asst_logprobs)) if asst_logprobs else 0.0
        n_tok = int(len(asst_logprobs))

        rows.append({
            "qid": it.qid,
            "question": it.question,
            "option": opt_text,
            "is_correct": (opt_text == it.correct),
            "sum_logprob": sum_ll,
            "avg_logprob": avg_ll,
            "n_tokens": n_tok,
        })

        # Track best option by sum_logprob
        key = it.qid
        if key not in per_q_best_option or sum_ll > per_q_best_option[key][1]:
            per_q_best_option[key] = (opt_text, sum_ll)

    df = pd.DataFrame(rows)

    # Convenience: per-question summary (argmax by summed LL)
    best_rows = []
    for it in items:
        best_opt, best_ll = per_q_best_option[it.qid]
        best_rows.append({
            "qid": it.qid,
            "question": it.question,
            "predicted_by_sum_ll": best_opt,
            "correct": it.correct,
            "is_correct_prediction": best_opt == it.correct
        })
    df_best = pd.DataFrame(best_rows)
    acc = float(np.mean(df_best["is_correct_prediction"])) if len(df_best) else 0.0
    print(f"\n=== Summary ===")
    print(f"Argmax-by-sum-logprob accuracy: {acc*100:.2f}% on {len(items)} questions")

    # -----------------
    # Save outputs
    # -----------------
    true_model_name = args.model.split("/")[-1]
    base_dir = os.path.join(args.output_dir, true_model_name)
    os.makedirs(base_dir, exist_ok=True)

    per_option_file = os.path.join(base_dir, f"{dataset_name}_per_option_ll.csv")
    per_question_file = os.path.join(base_dir, f"{dataset_name}_per_question_summary.csv")

    df.to_csv(per_option_file, index=False, encoding="utf-8")
    df_best.to_csv(per_question_file, index=False, encoding="utf-8")

    print(f"Saved per-option scores to: {per_option_file}")
    print(f"Saved per-question summary to: {per_question_file}")
    print("\nDone.")


if __name__ == "__main__":
    main()
