#!/usr/bin/env python3
import os
import glob
import math
import argparse
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


def _find_file(model_dir: str, pattern: str) -> str | None:
    hits = glob.glob(os.path.join(model_dir, pattern))
    return hits[0] if hits else None


def _load_per_question(model_dir: str) -> pd.DataFrame | None:
    f = _find_file(model_dir, "*per_question_summary.csv")
    if not f:
        return None
    df = pd.read_csv(f)
    # Expect columns: qid, question, predicted_by_sum_ll, correct, is_correct_prediction
    needed = {"qid", "predicted_by_sum_ll", "correct"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Missing columns in {f}: need {needed}, have {df.columns}")
    return df[["qid", "predicted_by_sum_ll", "correct"]].copy()


def _load_per_option(model_dir: str) -> pd.DataFrame | None:
    f = _find_file(model_dir, "*per_option_ll.csv")
    if not f:
        return None
    df = pd.read_csv(f)
    # Expect columns: qid, option, is_correct, sum_logprob, avg_logprob, n_tokens
    needed = {"qid", "option", "is_correct", "avg_logprob", "n_tokens"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Missing columns in {f}: need {needed}, have {df.columns}")
    return df[["qid", "option", "is_correct", "avg_logprob", "n_tokens"]].copy()


def _perplexity(row) -> float:
    # Use token-average logprob; guard n_tokens==0
    if pd.isna(row["avg_logprob"]) or (row.get("n_tokens", 1) == 0):
        return float("inf")
    return math.exp(-float(row["avg_logprob"]))


def majority_ensemble_accuracy(dataset_dir: str) -> float:
    """
    (1) Ensemble majority vote over per-question predictions across models.
    Tie-breaker: among tied options, select the one with the lowest mean perplexity
    (computed from models that have per-option files). If still tied, pick
    the lexicographically smallest option string to be deterministic.
    """
    model_dirs = [os.path.join(dataset_dir, d) for d in os.listdir(dataset_dir)
                  if os.path.isdir(os.path.join(dataset_dir, d))]
    per_q = []
    per_opt_by_model = {}  # model_name -> per-option DF (for tie-break)
    for mdir in model_dirs:
        dfq = _load_per_question(mdir)
        if dfq is not None:
            per_q.append((os.path.basename(mdir), dfq))
        dfo = _load_per_option(mdir)
        if dfo is not None:
            dfo = dfo.copy()
            dfo["perplexity"] = dfo.apply(_perplexity, axis=1)
            per_opt_by_model[os.path.basename(mdir)] = dfo

    if not per_q:
        return 0.0

    # Build set of qids common across models that provided per-question summaries
    common_qids = None
    for _, dfq in per_q:
        qset = set(dfq["qid"])
        common_qids = qset if common_qids is None else (common_qids & qset)
    if not common_qids:
        return 0.0

    # For correctness reference, use the first df
    ref_df = per_q[0][1].set_index("qid")
    correct_map = ref_df["correct"].to_dict()

    n_correct = 0
    n_total = 0

    # Pre-build a map of ppx per (qid, option) across models for tie-breaks
    ppx_by_qopt: dict[tuple, list] = defaultdict(list)
    for mname, dfo in per_opt_by_model.items():
        sub = dfo[dfo["qid"].isin(common_qids)]
        for (qid, option), grp in sub.groupby(["qid", "option"]):
            ppx_vals = grp["perplexity"].dropna().tolist()
            if ppx_vals:
                # There should be exactly one row per (qid, option) per model, but just in case
                ppx_by_qopt[(qid, option)].append(float(np.mean(ppx_vals)))

    # Iterate questions
    for qid in sorted(common_qids):
        votes = []
        for _, dfq in per_q:
            row = dfq[dfq["qid"] == qid]
            if not row.empty:
                votes.append(row.iloc[0]["predicted_by_sum_ll"])

        if not votes:
            continue

        counts = Counter(votes)
        max_ct = max(counts.values())
        tied = sorted([str(opt) for opt, c in counts.items() if c == max_ct])  # sorted for determinism

        if len(tied) == 1:
            pred = tied[0]
        else:
            # tie-break 1: lowest mean perplexity across models
            means = []
            for opt in tied:
                ppx_list = ppx_by_qopt.get((qid, opt), [])
                mean_ppx = float(np.mean(ppx_list)) if ppx_list else float("inf")
                means.append((mean_ppx, opt))
            means.sort(key=lambda x: (x[0], x[1]))  # lowest ppx, then lexicographic
            pred = means[0][1]

        corr = correct_map.get(qid)
        if corr is not None:
            n_total += 1
            if pred == corr:
                n_correct += 1

    return (n_correct / n_total) if n_total else 0.0


def ranking_ensemble_accuracy(dataset_dir: str) -> float:
    """
    (2) Ranking ensemble:
        For each model, rank options for each question by perplexity (asc).
        Average ranks across models, pick option with lowest average rank.
    """
    model_dirs = [os.path.join(dataset_dir, d) for d in os.listdir(dataset_dir)
                  if os.path.isdir(os.path.join(dataset_dir, d))]

    per_opts = []
    for mdir in model_dirs:
        dfo = _load_per_option(mdir)
        if dfo is None:
            continue
        dfo = dfo.copy()
        dfo["perplexity"] = dfo.apply(_perplexity, axis=1)
        per_opts.append((os.path.basename(mdir), dfo))

    if not per_opts:
        return 0.0

    # Intersection of qids that appear in all models with per-option data
    common_qids = None
    for _, dfo in per_opts:
        qset = set(dfo["qid"].unique())
        common_qids = qset if common_qids is None else (common_qids & qset)
    if not common_qids:
        return 0.0

    # For correctness, use the first model's per-option data (is_correct)
    ref = per_opts[0][1]
    correct_map = {qid: grp.loc[grp["is_correct"] == True, "option"].iloc[0]
                   for qid, grp in ref.groupby("qid") if (grp["is_correct"] == True).any()}

    n_correct = 0
    n_total = 0

    for qid in sorted(common_qids):
        # Collect ranks per model
        ranks_by_option: dict[str, list[float]] = defaultdict(list)
        for _, dfo in per_opts:
            sub = dfo[dfo["qid"] == qid].copy()
            if sub.empty:
                continue
            # Rank by perplexity (ascending); stable, with average method for ties
            sub["rank"] = sub["perplexity"].rank(method="average", ascending=True)
            for _, row in sub.iterrows():
                ranks_by_option[row["option"]].append(float(row["rank"]))

        if not ranks_by_option:
            continue

        # Average rank across models (options missing from some models simply have fewer entries)
        avg_ranks = [(np.mean(ranks), opt) for opt, ranks in ranks_by_option.items() if ranks]
        if not avg_ranks:
            continue
        avg_ranks.sort(key=lambda x: (x[0], x[1]))  # lowest avg rank wins
        pred = avg_ranks[0][1]

        corr = correct_map.get(qid)
        if corr is not None:
            n_total += 1
            if pred == corr:
                n_correct += 1

    return (n_correct / n_total) if n_total else 0.0


def mean_perplexity_accuracy(dataset_dir: str) -> float:
    """
    (3) Mean perplexity:
        For each question & option, compute mean perplexity across models,
        pick option with lowest mean perplexity.
    """
    model_dirs = [os.path.join(dataset_dir, d) for d in os.listdir(dataset_dir)
                  if os.path.isdir(os.path.join(dataset_dir, d))]

    per_opts = []
    for mdir in model_dirs:
        dfo = _load_per_option(mdir)
        if dfo is None:
            continue
        dfo = dfo.copy()
        dfo["perplexity"] = dfo.apply(_perplexity, axis=1)
        per_opts.append(dfo)

    if not per_opts:
        return 0.0

    # Intersection of qids
    common_qids = set(per_opts[0]["qid"].unique())
    for dfo in per_opts[1:]:
        common_qids &= set(dfo["qid"].unique())
    if not common_qids:
        return 0.0

    # Correct map from first df
    ref = per_opts[0]
    correct_map = {qid: grp.loc[grp["is_correct"] == True, "option"].iloc[0]
                   for qid, grp in ref.groupby("qid") if (grp["is_correct"] == True).any()}

    # Concatenate and keep only common qids
    all_df = pd.concat(per_opts, ignore_index=True)
    all_df = all_df[all_df["qid"].isin(common_qids)]

    # Mean perplexity per (qid, option) across models
    mean_ppx = (all_df
                .groupby(["qid", "option"], as_index=False)["perplexity"]
                .mean()
                .rename(columns={"perplexity": "mean_perplexity"}))

    # Pick lowest mean perplexity per qid
    preds = (mean_ppx.sort_values(["qid", "mean_perplexity", "option"])
             .groupby("qid", as_index=False).first())

    # Compute accuracy
    merged = preds[["qid", "option"]].rename(columns={"option": "pred"})
    merged["correct"] = merged["qid"].map(correct_map)
    merged = merged.dropna(subset=["correct"])
    acc = (merged["pred"] == merged["correct"]).mean() if len(merged) else 0.0
    return float(acc)


def main():
    ap = argparse.ArgumentParser(description="Compute ensemble accuracies across models per dataset.")
    ap.add_argument("--input_root", type=str, default="results/score_answers",
                    help="Root folder: results/score_answers/<dataset>/<model>/*.csv")
    ap.add_argument("--output_csv", type=str, default="results/score_answers_ensembles.csv",
                    help="Path to write dataset,metric,accuracy CSV.")
    args = ap.parse_args()

    if not os.path.isdir(args.input_root):
        raise SystemExit(f"Input root not found: {args.input_root}")

    datasets = [d for d in os.listdir(args.input_root)
                if os.path.isdir(os.path.join(args.input_root, d))]

    records = []
    for ds in sorted(datasets):
        ds_dir = os.path.join(args.input_root, ds)
        acc_majority = majority_ensemble_accuracy(ds_dir)
        acc_rank = ranking_ensemble_accuracy(ds_dir)
        acc_mean_ppx = mean_perplexity_accuracy(ds_dir)

        records.append({"dataset": ds, "metric": "ensemble_majority", "accuracy": acc_majority})
        records.append({"dataset": ds, "metric": "ranking_ensemble", "accuracy": acc_rank})
        records.append({"dataset": ds, "metric": "mean_perplexity", "accuracy": acc_mean_ppx})

        print(f"[{ds}] ensemble_majority={acc_majority:.4f} | ranking_ensemble={acc_rank:.4f} | mean_perplexity={acc_mean_ppx:.4f}")

    out_df = pd.DataFrame.from_records(records)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    main()
