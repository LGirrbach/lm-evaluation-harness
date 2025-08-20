#!/usr/bin/env python3
"""
Parse evaluation result JSONL files for multiple LLMs into CSVs.

Changes vs v1:
- Store IDENTIFIERS (candidate indices) instead of full answer texts.
- Add exact score columns for the selected predictions:
    * predicted_total_loglikelihood
    * predicted_avg_token_loglikelihood

Folder layout (assumed):
ROOT/
  model_A/
    task1.jsonl
    task2.jsonl
  model_B/
    task1.jsonl
    other_task.jsonl
  ...

For each *.jsonl file, this script writes a sibling .csv (or mirrors to --outdir)
with columns:
  model,
  task,
  predicted_answer_id_total_loglikelihood,
  predicted_total_loglikelihood,
  predicted_answer_id_avg_token_likelihood,
  predicted_avg_token_loglikelihood,
  ground_truth_id

Prediction selection rules:
- Each JSONL line encodes multiple candidate answers via arguments.gen_args_0..N
  (their texts in arg_1) and their scores in filtered_resps (preferred) or resps.
- "predicted_answer_id_total_loglikelihood" is the index of the candidate with
  the highest total_loglikelihood.
- "predicted_answer_id_avg_token_likelihood" is the index of the candidate with
  the highest average log-likelihood per token (total / number_of_tokens).

Ground truth rules:
- If top-level "target" is an int or numeric string, it is treated as the
  index into the candidate list (0-based) and saved directly.
- Otherwise we try to match the textual target (e.g., "yes" or option text) to
  one of the candidates (case/punct-insensitive). If a match is found, we save
  that index; else we save an empty value.

Robustness:
- Handles both "filtered_resps" (list[str]) and "resps" (list[list[str]]).
- Safely parses Python-literal-looking dict strings using ast.literal_eval.
- Ignores malformed lines but logs them.

Usage:
  python parse_llm_jsonl_to_csv_ids_scores.py /path/to/ROOT
  python parse_llm_jsonl_to_csv_ids_scores.py /path/to/ROOT --outdir /path/to/OUTPUT_ROOT
  python parse_llm_jsonl_to_csv_ids_scores.py /path/to/ROOT --pattern "**/*.jsonl" --overwrite

"""
from __future__ import annotations

import argparse
import ast
import csv
import json
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

    Falls back to empty list if not present.
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
    We use ast.literal_eval for safety.
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


def extract_candidate_scores(record: Dict[str, Any], num_options: int) -> List[Tuple[float, int]]:
    """Return list of (total_loglikelihood, token_count) per candidate.

    Prefers filtered_resps; falls back to resps.
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
        tok_count = len(logprobs) if isinstance(logprobs, Sequence) else 1
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
    """Derive the ground-truth identifier (candidate index as string).

    Priority:
      1) If target is an int (or numeric string), return it if in-range.
      2) Else, try to match textual target to an option ignoring case/punct and return that index.
      3) Else, try single-letter label mapping (A-D -> 0-3).
      4) Else, return empty string.
    """
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

def process_jsonl_file(root: Path, file_path: Path, outdir: Optional[Path], overwrite: bool = False) -> Optional[Path]:
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
            scores = extract_candidate_scores(record, len(options))
            idx_total, idx_avg = choose_indices(scores)

            # identifiers
            pred_id_total = str(idx_total) if idx_total >= 0 else ""
            pred_id_avg = str(idx_avg) if idx_avg >= 0 else ""
            gt_id = ground_truth_id(record, options)

            # scores
            total_ll = scores[idx_total][0] if 0 <= idx_total < len(scores) else ""
            avg_ll = (
                (scores[idx_avg][0] / max(1, scores[idx_avg][1]))
                if 0 <= idx_avg < len(scores) else ""
            )

            rows.append([
                model,
                task,
                pred_id_total,
                total_ll,
                pred_id_avg,
                avg_ll,
                gt_id,
            ])

    # Write CSV
    header = [
        "model",
        "task",
        "predicted_answer_id_total_loglikelihood",
        "predicted_total_loglikelihood",
        "predicted_answer_id_avg_token_likelihood",
        "predicted_avg_token_loglikelihood",
        "ground_truth_id",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(header)
        w.writerows(rows)

    print(f"[ok] Wrote {out_path} ({len(rows)} rows)")
    return out_path


# ----------------------------- cli -------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Parse LLM evaluation JSONL files into CSVs (IDs + scores).")
    ap.add_argument("root", type=Path, help="Root directory containing per-model subfolders with JSONL files.")
    ap.add_argument("--outdir", type=Path, default=None, help="Optional output root directory to mirror structure into.")
    ap.add_argument("--pattern", type=str, default="**/*.jsonl", help="Glob pattern (relative to each model folder) to find files. Default: **/*.jsonl")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing CSVs.")

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
            process_jsonl_file(root, fp, outdir=outdir, overwrite=args.overwrite)
        except Exception as e:
            print(f"[error] Failed to process {fp}: {e}")


if __name__ == "__main__":
    main()
