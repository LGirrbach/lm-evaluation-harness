#!/usr/bin/env python3
import os
import glob
import json
import argparse
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


# ---------- Discovery for your folder layout ----------
# results/p_true/<DATASET>/<MODEL>/*_per_option_yesno.json

def _iter_dataset_dirs(root: str) -> list[str]:
    return [os.path.join(root, d) for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))]

def _iter_model_dirs(dataset_dir: str) -> list[str]:
    return [os.path.join(dataset_dir, d) for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d))]

def _iter_model_jsons(model_dir: str) -> list[str]:
    # Be permissive: handle *_per_option_yesno.json (with or without extra tokens like _standard_)
    return sorted(glob.glob(os.path.join(model_dir, "*_per_option_yesno.json")))

def _group_by_dataset_dir(input_root: str) -> dict[str, list[tuple[str, dict]]]:
    """
    Returns: { <dataset_dir_name>: [(model_name, payload_dict), ...], ... }
    """
    grouped: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for ds_dir in _iter_dataset_dirs(input_root):
        ds_name = os.path.basename(ds_dir)
        for mdir in _iter_model_dirs(ds_dir):
            mname = os.path.basename(mdir)
            jsons = _iter_model_jsons(mdir)
            if not jsons:
                continue
            # If multiple JSONs exist, load all (usually there's one)
            for jf in jsons:
                try:
                    with open(jf, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    grouped[ds_name].append((mname, payload))
                except Exception:
                    # Skip unreadable files
                    continue
    return grouped


# ---------- Data shaping ----------

def _per_question_df(payload: dict) -> pd.DataFrame:
    rows = []
    for it in payload.get("items", []):
        rows.append({
            "qid": it.get("qid"),
            "predicted_answer": it.get("predicted_answer"),
            "correct": it.get("correct"),
        })
    return pd.DataFrame(rows)

def _per_option_df(payload: dict) -> pd.DataFrame:
    """
    yes_score = prob if model_answer == "Yes" else 1 - prob  (None -> 0.0)
    """
    rows = []
    for it in payload.get("items", []):
        qid = it.get("qid")
        for opt in it.get("options", []):
            text = opt.get("text")
            ans = opt.get("model_answer")
            prob = float(opt.get("probability", 0.0) or 0.0)
            if ans == "Yes":
                yes_score = prob
            elif ans == "No":
                yes_score = 1.0 - prob
            else:
                yes_score = 0.0
            rows.append({
                "qid": qid,
                "option": text,
                "model_answer": ans,
                "probability": prob,   # prob of the generated answer string
                "yes_score": yes_score # calibrated yes confidence
            })
    return pd.DataFrame(rows)


# ---------- Metrics (same behavior, using yes_score) ----------

def majority_ensemble_accuracy(group: list[tuple[str, dict]]) -> float:
    """
    Majority vote over per-question 'predicted_answer' across models.
    Tie-breaker: highest mean yes_score across models; then lexicographic.
    """
    per_q = []
    per_opt_by_model = {}
    for mname, payload in group:
        dfq = _per_question_df(payload)
        if not dfq.empty:
            per_q.append((mname, dfq))
        dfo = _per_option_df(payload)
        if not dfo.empty:
            per_opt_by_model[mname] = dfo

    if not per_q:
        return 0.0

    # Common qids across models that provided per-question predictions
    common_qids = None
    for _, dfq in per_q:
        qset = set(dfq["qid"])
        common_qids = qset if common_qids is None else (common_qids & qset)
    if not common_qids:
        return 0.0

    # Reference correct answers from the first df (all models should agree on 'correct')
    ref_df = per_q[0][1].set_index("qid")
    correct_map = ref_df["correct"].to_dict()

    # Mean yes_score per (qid, option) from any models that have per-option data
    yes_score_by_qopt: dict[tuple, list] = defaultdict(list)
    for _, dfo in per_opt_by_model.items():
        sub = dfo[dfo["qid"].isin(common_qids)]
        for (qid, option), grp in sub.groupby(["qid", "option"]):
            vals = grp["yes_score"].dropna().tolist()
            if vals:
                yes_score_by_qopt[(qid, option)].append(float(np.mean(vals)))

    n_correct = 0
    n_total = 0
    for qid in sorted(common_qids):
        votes = []
        for _, dfq in per_q:
            row = dfq[dfq["qid"] == qid]
            if not row.empty:
                votes.append(str(row.iloc[0]["predicted_answer"]))

        if not votes:
            continue

        counts = Counter(votes)
        max_ct = max(counts.values())
        tied = sorted([opt for opt, c in counts.items() if c == max_ct])

        if len(tied) == 1:
            pred = tied[0]
        else:
            # tie-break: highest mean yes_score; then lexicographic
            means = []
            for opt in tied:
                ys_list = yes_score_by_qopt.get((qid, opt), [])
                mean_ys = float(np.mean(ys_list)) if ys_list else float("-inf")
                means.append((mean_ys, opt))
            means.sort(key=lambda x: (-x[0], x[1]))
            pred = means[0][1]

        corr = correct_map.get(qid)
        if corr is not None:
            n_total += 1
            if pred == corr:
                n_correct += 1

    return (n_correct / n_total) if n_total else 0.0


def ranking_ensemble_accuracy(group: list[tuple[str, dict]]) -> float:
    """
    For each model, rank options per question by yes_score (desc),
    average ranks across models, pick the lowest avg rank.
    """
    per_opts = []
    for _, payload in group:
        dfo = _per_option_df(payload)
        if not dfo.empty:
            per_opts.append(dfo)

    if not per_opts:
        return 0.0

    # qids common to all models with per-option data
    common_qids = set(per_opts[0]["qid"].unique())
    for dfo in per_opts[1:]:
        common_qids &= set(dfo["qid"].unique())
    if not common_qids:
        return 0.0

    # Correct map from any payload
    correct_map = {}
    for _, payload in group:
        for it in payload.get("items", []):
            qid = it.get("qid")
            corr = it.get("correct")
            if qid is not None and corr:
                correct_map[qid] = corr

    n_correct = 0
    n_total = 0
    for qid in sorted(common_qids):
        ranks_by_option: dict[str, list[float]] = defaultdict(list)
        for dfo in per_opts:
            sub = dfo[dfo["qid"] == qid].copy()
            if sub.empty:
                continue
            sub["rank"] = sub["yes_score"].rank(method="average", ascending=False)
            for _, row in sub.iterrows():
                ranks_by_option[str(row["option"])].append(float(row["rank"]))

        if not ranks_by_option:
            continue

        avg_ranks = [(np.mean(ranks), opt) for opt, ranks in ranks_by_option.items() if ranks]
        avg_ranks.sort(key=lambda x: (x[0], x[1]))  # best = lowest avg rank
        pred = avg_ranks[0][1]

        corr = correct_map.get(qid)
        if corr is not None:
            n_total += 1
            if pred == corr:
                n_correct += 1

    return (n_correct / n_total) if n_total else 0.0


def mean_yesscore_accuracy(group: list[tuple[str, dict]]) -> float:
    """
    For each (qid, option), compute mean yes_score across models,
    pick option with highest mean yes_score.
    """
    per_opts = []
    for _, payload in group:
        dfo = _per_option_df(payload)
        if not dfo.empty:
            per_opts.append(dfo)

    if not per_opts:
        return 0.0

    common_qids = set(per_opts[0]["qid"].unique())
    for dfo in per_opts[1:]:
        common_qids &= set(dfo["qid"].unique())
    if not common_qids:
        return 0.0

    # Correct map from any payload
    correct_map = {}
    for _, payload in group:
        for it in payload.get("items", []):
            qid = it.get("qid")
            corr = it.get("correct")
            if qid is not None and corr:
                correct_map[qid] = corr

    all_df = pd.concat(per_opts, ignore_index=True)
    all_df = all_df[all_df["qid"].isin(common_qids)]

    mean_ys = (all_df.groupby(["qid", "option"], as_index=False)["yes_score"]
               .mean().rename(columns={"yes_score": "mean_yes_score"}))

    preds = (mean_ys.sort_values(["qid", "mean_yes_score", "option"],
                                 ascending=[True, False, True])
             .groupby("qid", as_index=False).first())

    merged = preds[["qid", "option"]].rename(columns={"option": "pred"})
    merged["correct"] = merged["qid"].map(correct_map)
    merged = merged.dropna(subset=["correct"])
    acc = (merged["pred"] == merged["correct"]).mean() if len(merged) else 0.0
    return float(acc)


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Ensemble accuracies for S1 JSON outputs (folder layout: results/p_true/<dataset>/<model>/*.json)"
    )
    ap.add_argument("--input_root", type=str, default="results/p_true",
                    help="Root folder with dataset/model subfolders.")
    ap.add_argument("--output_csv", type=str, default="results/p_true_ensembles.csv",
                    help="Where to write dataset,metric,accuracy CSV.")
    args = ap.parse_args()

    if not os.path.isdir(args.input_root):
        raise SystemExit(f"Input root not found: {args.input_root}")

    grouped = _group_by_dataset_dir(args.input_root)
    if not grouped:
        raise SystemExit(f"No *_per_option_yesno.json files found under: {args.input_root}")

    records = []
    for ds in sorted(grouped.keys()):
        group = grouped[ds]
        if not group:
            continue

        acc_majority = majority_ensemble_accuracy(group)
        acc_rank = ranking_ensemble_accuracy(group)
        acc_mean_yes = mean_yesscore_accuracy(group)

        records.append({"dataset": ds, "metric": "ensemble_majority", "accuracy": acc_majority})
        records.append({"dataset": ds, "metric": "ranking_ensemble", "accuracy": acc_rank})
        records.append({"dataset": ds, "metric": "mean_yesscore", "accuracy": acc_mean_yes})

        print(f"[{ds}] ensemble_majority={acc_majority:.4f} | ranking_ensemble={acc_rank:.4f} | mean_yesscore={acc_mean_yes:.4f}")

    out_df = pd.DataFrame.from_records(records)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    main()
