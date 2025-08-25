#!/usr/bin/env python3
"""
Evaluate TruthfulQA MC1 results aggregated across models.

Input layout (as produced by your generation script):
<inputs_dir>/
  <modelA>/
    truthfulqa_mc1_two_rounds.csv
  <modelB>/
    truthfulqa_mc1_two_rounds.csv
  ...

Each CSV has columns:
- qid, question, correct, direct_answer_mode, crowd_answer_mode,
  direct_samples_json, crowd_samples_json

This script:
1) Aggregates per question across models.
2) Computes:
   - Mode-of-Modes (direct majority across models) and its accuracy.
   - Surprisingly Popular (SP) answer using:
       Actual%   = freq across models of direct_answer_mode
       Predicted% = freq across models of crowd_answer_mode
     SP picks argmax of (Actual% - Predicted%).
3) Saves a per-question CSV with diagnostics.
4) Prints summary metrics.

Usage:
python evaluate_truthfulqa_across_models.py \
  --inputs_dir results/truthfulqa_two_rounds \
  --file_name truthfulqa_mc1_two_rounds.csv \
  --out_dir results/truthfulqa_eval

"""

import os
import argparse
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def list_model_dirs(inputs_dir: str) -> List[str]:
    return sorted(
        d for d in (os.path.join(inputs_dir, x) for x in os.listdir(inputs_dir))
        if os.path.isdir(d)
    )


def read_model_csv(model_dir: str, file_name: str) -> pd.DataFrame:
    path = os.path.join(model_dir, file_name)
    if not os.path.exists(path):
        print(f"[WARN] Missing file for model dir: {model_dir}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["model_name"] = os.path.basename(model_dir)
    return df


def tie_break_lex(keys: List[str]) -> str:
    """Deterministic tie-breaker: lexicographically earliest non-empty string."""
    keys = [k for k in keys if isinstance(k, str)]
    return sorted(keys)[0] if keys else ""


def mode_from_counts(counts: Dict[str, int]) -> str:
    if not counts:
        return ""
    max_c = max(counts.values())
    tops = [k for k, v in counts.items() if v == max_c]
    return tie_break_lex(tops)


def normalize_counts_to_pct(counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        return {k: 0.0 for k in counts.keys()}
    return {k: (v * 100.0 / total) for k, v in counts.items()}


def surprise_scores(actual_pct: Dict[str, float],
                    predicted_pct: Dict[str, float],
                    option_universe: List[str]) -> Dict[str, float]:
    # Ensure all options present (fill with 0 if absent)
    out = {}
    for opt in option_universe:
        a = actual_pct.get(opt, 0.0)
        p = predicted_pct.get(opt, 0.0)
        out[opt] = a - p
    return out


def pairwise_agreement_rate(labels: List[str]) -> float:
    n = len(labels)
    if n <= 1:
        return 1.0
    agree = 0
    total = n * (n - 1) // 2
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                agree += 1
    return agree / total if total > 0 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs_dir", type=str, required=True,
                    help="Directory containing model subfolders with per-model CSVs.")
    ap.add_argument("--file_name", type=str, default="truthfulqa_mc1_two_rounds.csv",
                    help="CSV file name inside each model folder.")
    ap.add_argument("--out_dir", type=str, default="results/truthfulqa_eval",
                    help="Where to write aggregated outputs.")
    ap.add_argument("--min_models_per_q", type=int, default=1,
                    help="Drop questions with fewer than this many models contributing.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load all model CSVs
    model_dirs = list_model_dirs(args.inputs_dir)
    all_dfs = []
    for md in model_dirs:
        df = read_model_csv(md, args.file_name)
        if not df.empty:
            # Keep only necessary columns
            cols = ["qid", "question", "correct", "direct_answer_mode", "crowd_answer_mode", "model_name"]
            missing = [c for c in cols if c not in df.columns]
            if missing:
                print(f"[WARN] Skipping {md}: missing columns {missing}")
                continue
            all_dfs.append(df[cols].copy())

    if not all_dfs:
        raise SystemExit("No valid per-model CSVs found.")

    big = pd.concat(all_dfs, ignore_index=True)
    n_models = big["model_name"].nunique()
    print(f"Loaded {n_models} models, {len(big)} rows.")

    # Sanity: ensure 'correct' is consistent per qid
    corr_check = big.groupby("qid")["correct"].nunique()
    inconsistent = corr_check[corr_check > 1]
    if len(inconsistent) > 0:
        print(f"[WARN] Found {len(inconsistent)} qids with inconsistent 'correct' labels across models. "
              f"Taking the most frequent 'correct' per qid.")
        # Fix by majority label per qid
        def majority_correct(sub: pd.Series) -> str:
            counts = Counter(sub.tolist())
            top = mode_from_counts(counts)
            return top
        correct_map = big.groupby("qid")["correct"].apply(majority_correct).to_dict()
        big["correct"] = big["qid"].map(correct_map)

    # Filter by min_models_per_q
    counts_per_q = big.groupby("qid")["model_name"].nunique()
    keep_qids = counts_per_q[counts_per_q >= args.min_models_per_q].index
    big = big[big["qid"].isin(keep_qids)].copy()

    # Aggregate
    rows = []
    r1_pair_agreements = []
    r2_pair_agreements = []

    for qid, g in big.groupby("qid"):
        correct = g["correct"].iloc[0]
        question = g["question"].iloc[0]

        # Votes across models
        direct_labels = g["direct_answer_mode"].tolist()
        crowd_labels = g["crowd_answer_mode"].tolist()

        direct_counts = Counter(direct_labels)
        crowd_counts = Counter(crowd_labels)

        # Deterministic option universe
        option_universe = sorted(set(list(direct_counts.keys()) + list(crowd_counts.keys()) + [correct]))

        # Mode of modes (direct)
        direct_majority_answer = mode_from_counts(direct_counts)

        # Predicted-majority (from "what most people would answer")
        predicted_majority_answer = mode_from_counts(crowd_counts)

        # Percentages for SP
        actual_pct = normalize_counts_to_pct({k: direct_counts.get(k, 0) for k in option_universe})
        predicted_pct = normalize_counts_to_pct({k: crowd_counts.get(k, 0) for k in option_universe})

        # Surprise scores and SP answer
        surprise = surprise_scores(actual_pct, predicted_pct, option_universe)
        sp_answer = mode_from_counts(surprise) if isinstance(surprise, Counter) else \
                    sorted([k for k, v in surprise.items() if v == max(surprise.values())])[0]

        # Agreement diagnostics across models
        r1_pair_agreements.append(pairwise_agreement_rate(direct_labels))
        r2_pair_agreements.append(pairwise_agreement_rate(crowd_labels))
        r1_majority_share = max(direct_counts.values()) / len(direct_labels) if direct_labels else 0.0
        r2_majority_share = max(crowd_counts.values()) / len(crowd_labels) if crowd_labels else 0.0

        rows.append({
            "qid": qid,
            "question": question,
            "correct": correct,
            "n_models_for_q": g["model_name"].nunique(),
            "direct_majority_answer": direct_majority_answer,
            "predicted_majority_answer": predicted_majority_answer,
            "sp_answer": sp_answer,
            "direct_majority_is_correct": int(direct_majority_answer == correct),
            "sp_is_correct": int(sp_answer == correct),
            "direct_counts_json": str(dict(direct_counts)),
            "crowd_counts_json": str(dict(crowd_counts)),
            "actual_pct_json": str(actual_pct),
            "predicted_pct_json": str(predicted_pct),
            "surprise_scores_json": str(surprise),
            "r1_majority_share": r1_majority_share,
            "r2_majority_share": r2_majority_share,
        })

    agg_df = pd.DataFrame(rows).sort_values("qid").reset_index(drop=True)

    # Save per-question aggregation
    out_csv = os.path.join(args.out_dir, "truthfulqa_mc1_across_models_eval.csv")
    agg_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved per-question aggregation to: {out_csv}")

    # -------- Summary metrics --------
    direct_mode_of_modes_acc = np.mean(agg_df["direct_majority_is_correct"])
    sp_acc = np.mean(agg_df["sp_is_correct"])

    diff_rate_sp_vs_direct = np.mean(agg_df["sp_answer"] != agg_df["direct_majority_answer"])
    agree_r1_pair = float(np.mean(r1_pair_agreements)) if r1_pair_agreements else 1.0
    agree_r2_pair = float(np.mean(r2_pair_agreements)) if r2_pair_agreements else 1.0
    avg_r1_majority_share = float(np.mean(agg_df["r1_majority_share"])) if len(agg_df) else 0.0
    avg_r2_majority_share = float(np.mean(agg_df["r2_majority_share"])) if len(agg_df) else 0.0

    print("\n=== Summary (across models) ===")
    print(f"Models included:                         {n_models}")
    print(f"Questions aggregated:                    {len(agg_df)}")
    print(f"Direct accuracy (mode of modes):         {direct_mode_of_modes_acc*100:.2f}%")
    print(f"SP accuracy (Actual% - Predicted%):      {sp_acc*100:.2f}%")
    print(f"SP vs Direct — differ rate:              {diff_rate_sp_vs_direct*100:.2f}%")
    print(f"Inter-model pairwise agreement (R1):     {agree_r1_pair*100:.2f}%")
    print(f"Inter-model pairwise agreement (R2):     {agree_r2_pair*100:.2f}%")
    print(f"Avg majority share across models (R1):   {avg_r1_majority_share*100:.2f}%")
    print(f"Avg majority share across models (R2):   {avg_r2_majority_share*100:.2f}%")
    print("Done.")


if __name__ == "__main__":
    main()
