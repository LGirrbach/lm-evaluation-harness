import os
import json
import pandas as pd

from tqdm import tqdm
from ast import literal_eval
from typing import Any, Dict, List, Optional


LETTER_TO_IDX = {chr(ord('A') + i): i for i in range(26)}


def extract_task_name(filename: str) -> str:
    base = os.path.splitext(filename)[0]
    rest = base[len("samples_"):] if base.startswith("samples_") else base
    if "_" in rest:
        rest = rest.rsplit("_", 1)[0]
    return rest


def normalize_text(s: str) -> str:
    return " ".join(s.strip().split()).casefold()


def _entropy_weighted_avg_logprobs(logprobs: List[float], entropies: List[float]):
    """Compute sum(H_i * lp_i) / sum(H_i). If sum(H_i)==0, fall back to mean logprobs.
    Returns pandas.NA if inputs are empty.
    """
    if not logprobs:
        return pd.NA
    if not entropies or len(entropies) != len(logprobs):
        return sum(logprobs) / len(logprobs)
    denom = sum(entropies)
    if denom == 0:
        return sum(logprobs) / len(logprobs)
    num = sum(h * lp for h, lp in zip(entropies, logprobs))
    return num / denom


def get_candidate_resps(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return candidate response dicts from filtered_resps or resps."""
    if record.get("filtered_resps"):
        entries = record["filtered_resps"]
    elif record.get("resps"):
        raw = record["resps"]
        entries = [r[0] if isinstance(r, list) and r else r for r in raw]
    else:
        return []

    out: List[Dict[str, Any]] = []
    for e in entries:
        if isinstance(e, dict):
            out.append(e)
        elif isinstance(e, str):
            out.append(literal_eval(e))
    return out


def resolve_correct_index(record: Dict[str, Any], num_candidates: int) -> Optional[int]:
    """Resolve the ground-truth index from several possible fields and formats."""
    choices = record.get("doc", {}).get("choices")

    def try_coerce(val: Any) -> Optional[int]:
        if isinstance(val, int):
            return val if 0 <= val < num_candidates else None
        if isinstance(val, float) and float(val).is_integer():
            ival = int(val)
            return ival if 0 <= ival < num_candidates else None
        if isinstance(val, str):
            s = val.strip()
            if s.isdigit():
                ival = int(s)
                return ival if 0 <= ival < num_candidates else None
            lower = s.lower()
            if lower.startswith("answer:"):
                s = s.split(":", 1)[1].strip()
            s = s.strip().rstrip(".")
            if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
                s = s[1:-1].strip()
            if len(s) == 1 and s.upper() in LETTER_TO_IDX:
                idx = LETTER_TO_IDX[s.upper()]
                return idx if 0 <= idx < num_candidates else None
            for tok in s.replace("(", " ").replace(")", " ").split()[::-1]:
                up = tok.upper()
                if len(up) == 1 and up in LETTER_TO_IDX:
                    idx = LETTER_TO_IDX[up]
                    return idx if 0 <= idx < num_candidates else None
            if choices and isinstance(choices, list):
                s_norm = normalize_text(s)
                for i, ch in enumerate(choices):
                    if normalize_text(str(ch)) == s_norm:
                        return i if 0 <= i < num_candidates else None
        return None

    for cand in [record.get("target"), record.get("doc", {}).get("answer"), record.get("doc", {}).get("target")]:
        idx = try_coerce(cand)
        if idx is not None:
            return idx
    return None


def best_pred_index(resp_dicts: List[Dict[str, Any]]) -> Optional[int]:
    if not resp_dicts:
        return None
    best_i = 0
    best_val = float("-inf")
    for i, d in enumerate(resp_dicts):
        val = d.get("total_loglikelihood", float("-inf"))
        if val > best_val:
            best_val = val
            best_i = i
    return best_i


def process_results(base_dir: str, drop_unresolved: bool = False):
    """Return {model: {task: DataFrame}} where each DF has two rows per doc: role in {gt, pred}.
    Columns: doc_id, role, avg_logprobs, avg_token_entropy, entropy_weighted_avg_logprobs,
             total_loglikelihood, num_tokens, correct.
    """
    model_data: Dict[str, Dict[str, pd.DataFrame]] = {}
    progress_bar = tqdm(total=len(os.listdir(base_dir)))

    for model_name in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_name)
        if not os.path.isdir(model_path):
            continue

        for file in os.listdir(model_path):
            if not (file.endswith(".jsonl") and file.startswith("samples_")):
                continue

            progress_bar.set_description(f"Processing {model_name}/{file}")

            task_name = extract_task_name(file)
            file_path = os.path.join(model_path, file)

            rows: List[Dict[str, Any]] = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    doc_id = record.get("doc_id")
                    resp_dicts = get_candidate_resps(record)
                    if not resp_dicts:
                        if not drop_unresolved:
                            for role in ("gt", "pred"):
                                rows.append({
                                    "doc_id": doc_id,
                                    "role": role,
                                    "avg_logprobs": pd.NA,
                                    "avg_token_entropy": pd.NA,
                                    "entropy_weighted_avg_logprobs": pd.NA,
                                    "total_loglikelihood": pd.NA,
                                    "num_tokens": pd.NA,
                                    "correct": pd.NA,
                                })
                        continue

                    gt_idx = resolve_correct_index(record, len(resp_dicts))
                    pred_idx = best_pred_index(resp_dicts)

                    correct_flag = (
                        None if gt_idx is None or pred_idx is None else int(pred_idx == gt_idx)
                    )

                    def stats_for(idx: Optional[int]) -> Dict[str, Any]:
                        if idx is None:
                            return {
                                "avg_logprobs": pd.NA,
                                "avg_token_entropy": pd.NA,
                                "entropy_weighted_avg_logprobs": pd.NA,
                                "total_loglikelihood": pd.NA,
                                "num_tokens": pd.NA,
                            }
                        resp = resp_dicts[idx]
                        lp = resp.get("logprobs", []) or []
                        ent = resp.get("entropies", []) or []
                        total_ll = resp.get("total_loglikelihood", pd.NA)
                        return {
                            "avg_logprobs": (sum(lp) / len(lp)) if lp else pd.NA,
                            "avg_token_entropy": (sum(ent) / len(ent)) if ent else pd.NA,
                            "entropy_weighted_avg_logprobs": _entropy_weighted_avg_logprobs(lp, ent),
                            "total_loglikelihood": total_ll,
                            "num_tokens": len(lp),
                        }

                    # GT row
                    rows.append({
                        "doc_id": doc_id,
                        "role": "gt",
                        **stats_for(gt_idx),
                        "correct": pd.NA if correct_flag is None else correct_flag,
                    })

                    # Pred row
                    rows.append({
                        "doc_id": doc_id,
                        "role": "pred",
                        **stats_for(pred_idx),
                        "correct": pd.NA if correct_flag is None else correct_flag,
                    })

            df = pd.DataFrame(rows)
            model_data.setdefault(model_name, {})[task_name] = df

        progress_bar.update(1)

    return model_data


if __name__ == "__main__":
    base_dir = "results_scratch"  # replace with your top-level directory path
    results = process_results(base_dir)

    for model, tasks in results.items():
        for task, df in tasks.items():
            out_dir = os.path.join("processed_results", model)
            os.makedirs(out_dir, exist_ok=True)
            df.to_csv(os.path.join(out_dir, f"{task}.csv"), index=False)
