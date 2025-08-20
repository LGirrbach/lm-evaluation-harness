#!/usr/bin/env python3
"""
Parse evaluation result JSONL files for multiple LLMs into CSVs.

This version adds per-candidate *confidence features* based on token-level
log-probabilities (and optional token entropies) using the provided formulas.

New per-row columns (pipe-separated, index-aligned with all_answer_ids):
  - all_softmin_tau
  - all_cvar_alpha
  - all_entropy_weighted_mean
  - all_last_k_mean
  - all_typicality_mad
  - all_typicality_frac_above

We keep the previously added columns:
  model, task, doc_id,
  predicted_* (IDs and exact scores), ground_truth_id,
  n_options, all_answer_ids, all_total_loglikelihoods, all_avg_token_loglikelihoods

Optional CLI hyperparameters to control the new scores:
  --tau  (softmin temperature, default 0.1)
  --alpha (CVaR tail fraction, default 0.1)
  --k-last (last-k mean window, default 20)
  --beta (entropy-weighted mean exponent, default 1.0)
  --kappa (typicality threshold in nats, default 1.0)

Usage:
  python parse_llm_jsonl_to_csv_extras.py /path/to/ROOT \
    --outdir /path/to/OUTPUT_ROOT --overwrite \
    --tau 0.1 --alpha 0.1 --k-last 20 --beta 1.0 --kappa 1.0
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ----------------------------- helpers ---------------------------------

NON_ALNUM = re.compile(r"[^\w]+", flags=re.UNICODE)


def norm_text(s: str) -> str:
    """Normalize short answer text for loose matching."""
    return NON_ALNUM.sub(" ", s).strip().lower()


def sorted_gen_args(arguments: Dict[str, Any]) -> List[str]:
    """Return argument keys (gen_args_*) sorted by their numeric suffix."""
    keys = [k for k in (arguments or {}).keys() if k.startswith("gen_args_")]
    def key_idx(k: str) -> int:
        try:
            return int(k.split("_")[-1])
        except Exception:
            return 10**9
    return sorted(keys, key=key_idx)


def extract_options(record: Dict[str, Any]) -> List[str]:
    """Extract candidate answer texts from arguments.gen_args_*.arg_1.

    We return the texts (for matching), but we do NOT write texts to CSV.
    """
    arguments = record.get("arguments") or {}
    opts: List[str] = []
    for k in sorted_gen_args(arguments):
        v = arguments.get(k) or {}
        ans = v.get("arg_1")
        if isinstance(ans, str):
            opts.append(ans.strip())
        else:
            opts.append(str(ans))
    return opts


def parse_resp_obj(obj: Any) -> Optional[Dict[str, Any]]:
    """Parse a single response entry into a dict with total_loglikelihood and logprobs.

    The JSON often stores Python-literal strings like: "{'total_loglikelihood': -0.5, ...}".
    We use ast.literal_eval for safety and fall back to json.loads.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        obj = obj[0] if obj else None
        if obj is None:
            return None
    if isinstance(obj, str):
        s = obj.strip()
        try:
            return ast.literal_eval(s)
        except Exception:
            try:
                return json.loads(s)
            except Exception:
                return None
    return None

# ---------- Confidence feature functions (as provided) ------------------

def softmin_score(log_probs, entropies=None, tau=0.1):
    """
    Soft-min aggregation of token log-probabilities.
    As tau -> 0, approaches min(log_probs). Higher is better.
    """
    lp = list(log_probs)
    if not lp: raise ValueError("log_probs is empty")
    if tau <= 0: raise ValueError("tau must be > 0")
    a = [(-x)/tau for x in lp]                 # a_i = -logp_i / tau
    a_max = max(a)                             # for numerical stability
    return -tau * (a_max + math.log(sum(math.exp(ai - a_max) for ai in a)))

def cvar_score(log_probs, entropies=None, alpha=0.1):
    """
    CVaR over the worst alpha-fraction of tokens (tail-focused mean).
    Example: alpha=0.1 averages the bottom 10% lowest log-probs.
    Higher is better, but it focuses on the brittle tail.
    """
    lp = sorted(list(log_probs))               # ascending; worst first
    if not lp: raise ValueError("log_probs is empty")
    if not (0 < alpha <= 1): raise ValueError("alpha must be in (0,1]")
    n = len(lp)
    k = max(1, math.ceil(alpha * n))
    tail = lp[:k]
    return sum(tail) / len(tail)

def entropy_weighted_mean_logprob(log_probs, entropies=None, beta=1.0, eps=1e-12):
    """
    Entropy-weighted mean of log-probs: weights w_t ∝ (H_t)^beta.
    Set beta>0 to emphasize uncertain positions; beta=0 reduces to a plain mean.
    If `entropies` is None, returns the plain mean.
    """
    lp = list(log_probs)
    if not lp: raise ValueError("log_probs is empty")
    if entropies is None or beta == 0:
        return sum(lp) / len(lp)
    H = list(entropies)
    if len(H) != len(lp): raise ValueError("entropies must match length of log_probs")
    weights = [(max(h, 0.0) + eps) ** beta for h in H]
    Z = sum(weights)
    return sum(w * x for w, x in zip(weights, lp)) / Z

def last_k_mean_logprob(log_probs, entropies=None, k=20):
    """
    Mean log-probability over the last k tokens (or all if shorter).
    Useful to focus on the final commitment span.
    """
    lp = list(log_probs)
    if not lp: raise ValueError("log_probs is empty")
    if k is None or k <= 0: raise ValueError("k must be positive")
    segment = lp[-k:] if k < len(lp) else lp
    return sum(segment) / len(segment)

def typicality_mad(log_probs, entropies=None):
    """
    Mean absolute deviation of surprisal from a reference entropy.
    surprisal s_t = -log p_t
      - If entropies provided: mean |s_t - H_t| (typical-sampling style)
      - Else: mean |s_t - mean(s)| (sequence-relative typicality)
    Lower is better (more typical).
    """
    lp = list(log_probs)
    if not lp: raise ValueError("log_probs is empty")
    s = [-x for x in lp]
    if entropies is not None:
        H = list(entropies)
        if len(H) != len(lp): raise ValueError("entropies must match length of log_probs")
        devs = [abs(si - hi) for si, hi in zip(s, H)]
    else:
        mu = sum(s) / len(s)
        devs = [abs(si - mu) for si in s]
    return sum(devs) / len(devs)

def typicality_fraction_above(log_probs, entropies=None, kappa=1.0):
    """
    Fraction of tokens whose surprisal deviates from the reference entropy
    by more than kappa (in nats).
      - If entropies provided: uses per-step H_t
      - Else: uses sequence mean surprisal
    Lower is better.
    """
    lp = list(log_probs)
    if not lp: raise ValueError("log_probs is empty")
    s = [-x for x in lp]
    if entropies is not None:
        H = list(entropies)
        if len(H) != len(lp): raise ValueError("entropies must match length of log_probs")
        devs = [abs(si - hi) for si, hi in zip(s, H)]
    else:
        mu = sum(s) / len(s)
        devs = [abs(si - mu) for si in s]
    return sum(d > kappa for d in devs) / len(devs)

def compute_confidence_features(
    log_probs,
    entropies=None,
    *,
    tau=0.1,
    alpha=0.1,
    k_last=20,
    beta=1.0,
    kappa=1.0
):
    """
    Convenience wrapper: returns all the above scores in a dict.
    """
    return {
        "softmin_tau": softmin_score(log_probs, entropies, tau=tau),
        "cvar_alpha": cvar_score(log_probs, entropies, alpha=alpha),
        "entropy_weighted_mean": entropy_weighted_mean_logprob(log_probs, entropies, beta=beta),
        "last_k_mean": last_k_mean_logprob(log_probs, entropies, k=k_last),
        "typicality_mad": typicality_mad(log_probs, entropies),
        "typicality_frac_above": typicality_fraction_above(log_probs, entropies, kappa=kappa),
    }

# --------------------- extraction & aggregation utils -------------------

def coerce_num_list(x: Any) -> Optional[List[float]]:
    if isinstance(x, (list, tuple)):
        out: List[float] = []
        for v in x:
            try:
                out.append(float(v))
            except Exception:
                return None
        return out
    return None


def extract_candidate_series(record: Dict[str, Any], num_options: int) -> List[Tuple[Optional[List[float]], Optional[List[float]]]]:
    """Return list of (log_probs, entropies) per candidate if available, else (None, None)."""
    series: List[Tuple[Optional[List[float]], Optional[List[float]]]] = []

    src = record.get("filtered_resps")
    if not src:
        src = record.get("resps")

    if not isinstance(src, Sequence):
        src = []

    for i in range(num_options):
        entry = None
        if i < len(src):
            entry = src[i]
        resp = parse_resp_obj(entry)
        if resp is None:
            series.append((None, None))
            continue
        lp = coerce_num_list(resp.get("logprobs"))
        # entropies sometimes under different keys
        ent = resp.get("entropies")
        if ent is None:
            ent = resp.get("token_entropies")
        ent = coerce_num_list(ent)
        if lp is None:
            series.append((None, None))
        else:
            # if entropies exist but wrong length, drop them
            if ent is not None and len(ent) != len(lp):
                ent = None
            series.append((lp, ent))
    return series


def extract_candidate_scores(record: Dict[str, Any], num_options: int) -> List[Tuple[float, int]]:
    """Return list of (total_loglikelihood, token_count) per candidate.

    Prefers filtered_resps; falls back to resps.
    Missing/invalid entries get (-inf, 0).
    """
    scores: List[Tuple[float, int]] = []

    src = record.get("filtered_resps")
    if not src:
        src = record.get("resps")

    if not isinstance(src, Sequence):
        src = []

    for i in range(num_options):
        entry = None
        if i < len(src):
            entry = src[i]
        resp = parse_resp_obj(entry)
        if resp is None:
            scores.append((float("-inf"), 0))
            continue
        total = resp.get("total_loglikelihood")
        try:
            total_f = float(total)
        except Exception:
            total_f = float("-inf")
        logprobs = resp.get("logprobs")
        tok_count = len(logprobs) if isinstance(logprobs, (list, tuple)) else 1
        scores.append((total_f, tok_count))
    return scores


def choose_indices(scores: List[Tuple[float, int]]) -> Tuple[int, int]:
    """Return indices for (max total, max average per token)."""
    if not scores:
        return -1, -1
    best_total = max(range(len(scores)), key=lambda i: scores[i][0])
    best_avg = max(range(len(scores)), key=lambda i: (scores[i][0] / max(1, scores[i][1])))
    return best_total, best_avg


def ground_truth_id(record: Dict[str, Any], options: List[str]) -> str:
    """Derive the ground-truth identifier (candidate index as string)."""
    target = record.get("target")

    # 1) index-like targets (e.g., "3", 0)
    if isinstance(target, (int, float)):
        idx = int(target)
        if 0 <= idx < len(options):
            return str(idx)
    elif isinstance(target, str):
        t = target.strip()
        if t.isdigit():
            idx = int(t)
            if 0 <= idx < len(options):
                return str(idx)

    # 2) textual target matching
    if isinstance(target, str):
        t_norm = norm_text(target)
        for i, opt in enumerate(options):
            if norm_text(opt) == t_norm:
                return str(i)
        # 3) A/B/C/D mapping
        if len(t_norm) == 1 and t_norm.isalpha():
            letter = t_norm.upper()
            pos = ord(letter) - ord('A')
            if 0 <= pos < len(options):
                return str(pos)

    # 4) unknown
    return ""


def derive_model_and_task(root: Path, file_path: Path) -> Tuple[str, str]:
    """Model is the first directory under ROOT; task is the filename stem."""
    rel = file_path.relative_to(root)
    parts = rel.parts
    model = parts[0] if parts else "unknown_model"
    task = file_path.stem
    return model, task


# ----------------------------- core ------------------------------------

def process_jsonl_file(
    root: Path,
    file_path: Path,
    outdir: Optional[Path],
    *,
    overwrite: bool = False,
    tau: float = 0.1,
    alpha: float = 0.1,
    k_last: int = 20,
    beta: float = 1.0,
    kappa: float = 1.0,
) -> Optional[Path]:
    model, task = derive_model_and_task(root, file_path)

    # Output path
    if outdir:
        out_path = outdir / file_path.relative_to(root)
        out_path = out_path.with_suffix(".csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = file_path.with_suffix(".csv")

    if out_path.exists() and not overwrite:
        print(f"[skip] {out_path} exists. Use --overwrite to replace.")
        return out_path

    rows: List[List[Any]] = []

    with file_path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                try:
                    record = ast.literal_eval(line)
                except Exception as e:
                    print(f"[warn] {file_path}:{ln}: cannot parse JSON line ({e}).")
                    continue

            options = extract_options(record)
            n_opts = len(options)

            # core scores and token series
            scores = extract_candidate_scores(record, n_opts)
            series = extract_candidate_series(record, n_opts)
            idx_total, idx_avg = choose_indices(scores)

            # identifiers
            pred_id_total = str(idx_total) if idx_total >= 0 else ""
            pred_id_avg = str(idx_avg) if idx_avg >= 0 else ""
            gt_id = ground_truth_id(record, options)

            # scores for the chosen ones
            total_ll = scores[idx_total][0] if 0 <= idx_total < len(scores) else ""
            avg_ll = (
                (scores[idx_avg][0] / max(1, scores[idx_avg][1]))
                if 0 <= idx_avg < len(scores) else ""
            )

            # doc_id (best-effort)
            doc_id = record.get("doc_id")
            if doc_id is None:
                doc = record.get("doc") or {}
                doc_id = doc.get("doc_id") or doc.get("id")

            # all-answers columns (pipe-separated)
            ids = [str(i) for i in range(n_opts)]
            totals: List[str] = []
            avgs: List[str] = []
            softmins: List[str] = []
            cvars: List[str] = []
            ewm_means: List[str] = []
            lastk_means: List[str] = []
            typ_mads: List[str] = []
            typ_fracs: List[str] = []

            for (total, tok_cnt), (lp, ent) in zip(scores, series):
                if total is None or (isinstance(total, float) and math.isinf(total)):
                    totals.append("")
                    avgs.append("")
                else:
                    totals.append(str(total))
                    avgs.append(str(total / max(1, tok_cnt)))

                # Confidence features
                if lp is None or len(lp) == 0:
                    softmins.append("")
                    cvars.append("")
                    ewm_means.append("")
                    lastk_means.append("")
                    typ_mads.append("")
                    typ_fracs.append("")
                else:
                    try:
                        feats = compute_confidence_features(
                            lp, ent,
                            tau=tau, alpha=alpha, k_last=k_last, beta=beta, kappa=kappa
                        )
                        softmins.append(str(feats["softmin_tau"]))
                        cvars.append(str(feats["cvar_alpha"]))
                        ewm_means.append(str(feats["entropy_weighted_mean"]))
                        lastk_means.append(str(feats["last_k_mean"]))
                        typ_mads.append(str(feats["typicality_mad"]))
                        typ_fracs.append(str(feats["typicality_frac_above"]))
                    except Exception:
                        softmins.append("")
                        cvars.append("")
                        ewm_means.append("")
                        lastk_means.append("")
                        typ_mads.append("")
                        typ_fracs.append("")

            rows.append([
                model,
                task,
                doc_id if doc_id is not None else "",
                pred_id_total,
                total_ll,
                pred_id_avg,
                avg_ll,
                gt_id,
                n_opts,
                "|".join(ids),
                "|".join(totals),
                "|".join(avgs),
                "|".join(softmins),
                "|".join(cvars),
                "|".join(ewm_means),
                "|".join(lastk_means),
                "|".join(typ_mads),
                "|".join(typ_fracs),
            ])

    # Write CSV
    header = [
        "model",
        "task",
        "doc_id",
        "predicted_answer_id_total_loglikelihood",
        "predicted_total_loglikelihood",
        "predicted_answer_id_avg_token_likelihood",
        "predicted_avg_token_loglikelihood",
        "ground_truth_id",
        "n_options",
        "all_answer_ids",
        "all_total_loglikelihoods",
        "all_avg_token_loglikelihoods",
        "all_softmin_tau",
        "all_cvar_alpha",
        "all_entropy_weighted_mean",
        "all_last_k_mean",
        "all_typicality_mad",
        "all_typicality_frac_above",
    ]

    if out_path.exists() and not overwrite:
        pass  # already handled above; kept for clarity

    with out_path.open("w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(header)
        w.writerows(rows)

    print(f"[ok] Wrote {out_path} ({len(rows)} rows)")
    return out_path


# ----------------------------- cli -------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Parse LLM evaluation JSONL files into CSVs (IDs + scores + confidence features).")
    ap.add_argument("root", type=Path, help="Root directory containing per-model subfolders with JSONL files.")
    ap.add_argument("--outdir", type=Path, default=None, help="Optional output root directory to mirror structure into.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing CSVs.")

    # hyperparameters for confidence features
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--k-last", type=int, default=20)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--kappa", type=float, default=1.0)

    args = ap.parse_args()

    root: Path = args.root.resolve()
    outdir: Optional[Path] = args.outdir.resolve() if args.outdir else None

    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Root path not found or not a directory: {root}")

    files = sorted(root.rglob("*.jsonl"))
    if not files:
        print(f"[info] No JSONL files found under {root}")
        return

    for fp in files:
        try:
            process_jsonl_file(
                root, fp, outdir,
                overwrite=args.overwrite,
                tau=args.tau,
                alpha=args.alpha,
                k_last=args.k_last,
                beta=args.beta,
                kappa=args.kappa,
            )
        except Exception as e:
            print(f"[error] Failed to process {fp}: {e}")


if __name__ == "__main__":
    main()