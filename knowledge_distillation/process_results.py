#!/usr/bin/env python3
"""
Parse evaluation result JSONL files for multiple LLMs into CSVs.

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
  model, task, predicted_answer_total_loglikelihood,
  predicted_answer_avg_token_likelihood, ground_truth_answer

Prediction selection rules:
- Each JSONL line encodes multiple candidate answers via arguments.gen_args_0..N
  and their scores in filtered_resps (preferred) or resps.
- "predicted_answer_total_loglikelihood" selects the candidate with the highest
  total_loglikelihood.
- "predicted_answer_avg_token_likelihood" selects the candidate with the highest
  average log-likelihood per token (total / number_of_tokens).

Ground truth rules:
- If top-level "target" is an integer or numeric string, it is treated as the
  index into the candidate list (0-based).
- Otherwise we try to match the textual target (e.g., "yes") to one of the
  candidate texts (case-insensitive, punctuation-insensitive). If no match is
  found, we keep the raw target value.

Robustness:
- Handles both "filtered_resps" (list[str]) and "resps" (list[list[str]]).
- Safely parses Python-literal-looking dict strings using ast.literal_eval.
- Ignores malformed lines but logs them.

Usage:
  python parse_llm_jsonl_to_csv.py /path/to/ROOT
  python parse_llm_jsonl_to_csv.py /path/to/ROOT --outdir /path/to/OUTPUT_ROOT
  python parse_llm_jsonl_to_csv.py /path/to/ROOT --pattern "**/*.jsonl" --overwrite

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
            # Sometimes answers might be encoded differently; try to stringify
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
        # If it's JSON-looking with single quotes, literal_eval works well.
        try:
            return ast.literal_eval(s)
        except Exception:
            # Try JSON loads as a fallback (if it uses double quotes)
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

    # Ensure length matches options length when possible
    for i in range(num_options):
        entry = None
        if i < len(src):
            entry = src[i]
        resp = parse_resp_obj(entry)
        if resp is None:
            scores.append((float("-inf"), 0))
            continue
        total = resp.get("total_loglikelihood")
        if total is None:
            # Some variants might use another key; treat as missing
            total = float("-inf")
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


def ground_truth_text(record: Dict[str, Any], options: List[str]) -> str:
    """Derive the ground-truth answer text.

    Priority:
      1) If target is an int (or numeric string), map to options[index] when possible.
      2) Else, try to match textual target to an option ignoring case/punct.
      3) Else, return the raw target as a string.
    """
    target = record.get("target")

    # 1) index-like targets (e.g., "3", 0)
    idx: Optional[int] = None
    if isinstance(target, (int, float)):
        idx = int(target)
    elif isinstance(target, str):
        t = target.strip()
        if t.isdigit():
            idx = int(t)
    if idx is not None and 0 <= idx < len(options):
        return options[idx]

    # 2) textual target matching (yes/no or labels like A/B/C/D)
    if isinstance(target, str):
        t_norm = norm_text(target)
        # Exact option text match
        for opt in options:
            if norm_text(opt) == t_norm:
                return opt
        # Single-letter label mapping (A-D etc.)
        if len(t_norm) == 1 and t_norm.isalpha():
            letter = t_norm.upper()
            pos = ord(letter) - ord('A')
            if 0 <= pos < len(options):
                return options[pos]
        return target

    # Fallback: stringify whatever it is
    return str(target)


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

    rows: List[List[str]] = []

    with file_path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Some logs may contain single quotes at the top level; try literal_eval
                try:
                    record = ast.literal_eval(line)
                except Exception as e:
                    print(f"[warn] {file_path}:{ln}: cannot parse JSON line ({e}).")
                    continue

            options = extract_options(record)
            scores = extract_candidate_scores(record, len(options))
            idx_total, idx_avg = choose_indices(scores)

            pred_total = options[idx_total] if 0 <= idx_total < len(options) else ""
            pred_avg = options[idx_avg] if 0 <= idx_avg < len(options) else ""
            gt = ground_truth_text(record, options)

            rows.append([
                model,
                task,
                pred_total,
                pred_avg,
                gt,
            ])

    # Write CSV
    header = [
        "model",
        "task",
        "predicted_answer_total_loglikelihood",
        "predicted_answer_avg_token_likelihood",
        "ground_truth_answer",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(header)
        w.writerows(rows)

    print(f"[ok] Wrote {out_path} ({len(rows)} rows)")
    return out_path


# ----------------------------- cli -------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Parse LLM evaluation JSONL files into CSVs.")
    ap.add_argument("root", type=Path, help="Root directory containing per-model subfolders with JSONL files.")
    ap.add_argument("--outdir", type=Path, default=None, help="Optional output root directory to mirror structure into.")
    ap.add_argument("--pattern", type=str, default="**/*.jsonl", help="Glob pattern (relative to each model folder) to find files. Default: **/*.jsonl")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing CSVs.")

    args = ap.parse_args()

    root: Path = args.root.resolve()
    outdir: Optional[Path] = args.outdir.resolve() if args.outdir else None

    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Root path not found or not a directory: {root}")

    # Find all jsonl files under root
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
