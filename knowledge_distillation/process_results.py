import os
import re
import json
import pandas as pd
from ast import literal_eval
from typing import Any, Dict, List


def _extract_task_name(filename: str) -> str:
    """
    Extract task name from files like 'samples_<task>_<timestamp>.jsonl'.
    Keeps full task prefix (e.g., 'mmlu_world_religions').
    """
    base = filename[:-6] if filename.endswith(".jsonl") else filename
    if base.startswith("samples_"):
        base = base[len("samples_"):]
    # Drop the trailing timestamp chunk after the last underscore
    return base.rsplit("_", 1)[0] if "_" in base else base


def _to_resp_dict(x: Any) -> Dict[str, Any]:
    """
    Convert a mixed response entry (str, dict, list[str]) into a dict.
    Handles:
      - dict
      - "{"total_loglikelihood": ...}" (as str)
      - ["{"total_loglikelihood": ...}"] (from resps)
    """
    if isinstance(x, dict):
        return x
    if isinstance(x, list) and x:
        x = x[0]
    if isinstance(x, str):
        return literal_eval(x)
    raise ValueError(f"Unsupported response entry type: {type(x)}")


def _safe_mean(arr: List[float]) -> float | None:
    try:
        return (sum(arr) / len(arr)) if arr else None
    except Exception:
        return None


def _total_ll(d: Dict[str, Any]) -> float:
    """
    Get a comparable total log-likelihood for an option.
    Prefer 'total_loglikelihood'; else fall back to sum(logprobs).
    """
    if "total_loglikelihood" in d and d["total_loglikelihood"] is not None:
        return float(d["total_loglikelihood"])
    lps = d.get("logprobs")
    if isinstance(lps, list) and lps:
        try:
            return float(sum(lps))
        except Exception:
            pass
    return float("-inf")  # so it never wins argmax


def _target_to_index(record: Dict[str, Any]) -> int:
    """
    Convert various target formats to a 0-based index.
    Accepts integers, numeric strings ("0"), letter labels ("A", "(C)"),
    or falls back to record['doc']['answer'] if needed.
    """
    def _letter_to_idx(s: str) -> int | None:
        m = re.search(r"([A-Za-z])", s)
        if not m:
            return None
        return ord(m.group(1).upper()) - ord("A")

    # Primary: record['target']
    tgt = record.get("target")
    if isinstance(tgt, int):
        return tgt
    if isinstance(tgt, str):
        s = tgt.strip()
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        idx = _letter_to_idx(s)
        if idx is not None:
            return idx

    # Fallback: record['doc']['answer']
    doc_ans = record.get("doc", {}).get("answer")
    if isinstance(doc_ans, int):
        return doc_ans
    if isinstance(doc_ans, str):
        s = doc_ans.strip()
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        idx = _letter_to_idx(s)
        if idx is not None:
            return idx

    raise ValueError("Could not determine target index from record")


def process_results(base_dir: str):
    """
    Walk the model directories under base_dir, read all .jsonl result files,
    and return a nested dict: model -> task -> DataFrame with columns:
      doc_id, avg_logprobs, avg_token_entropy, correct
    """
    model_data: Dict[str, Dict[str, pd.DataFrame]] = {}

    for model_name in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_name)
        if not os.path.isdir(model_path):
            continue

        for file in os.listdir(model_path):
            if not file.endswith(".jsonl"):
                continue

            task_name = _extract_task_name(file)
            file_path = os.path.join(model_path, file)

            with open(file_path, "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]

            rows = []
            for record in records:
                doc_id = record.get("doc_id")

                # Prefer filtered_resps; fallback to resps
                raw_frs = record.get("filtered_resps")
                if raw_frs is None and "resps" in record:
                    raw_frs = record["resps"]
                if raw_frs is None:
                    continue  # skip if responses are missing

                try:
                    filtered_resps = [_to_resp_dict(x) for x in raw_frs]
                except Exception:
                    continue  # skip malformed responses

                # Determine correct target index (supports ints and letters)
                try:
                    target_idx = _target_to_index(record)
                except Exception:
                    continue

                if not (0 <= target_idx < len(filtered_resps)):
                    continue  # out-of-range target

                correct_resp = filtered_resps[target_idx]

                # Averages
                avg_logprobs = _safe_mean(correct_resp.get("logprobs", []))
                avg_entropy = _safe_mean(correct_resp.get("entropies", []))

                # Predicted = argmax of total log-likelihood (higher is better)
                totals = [_total_ll(r) for r in filtered_resps]
                try:
                    best_idx = max(range(len(totals)), key=lambda i: totals[i])
                except ValueError:
                    continue  # empty totals

                correct = 1 if best_idx == target_idx else 0

                rows.append({
                    "doc_id": doc_id,
                    "avg_logprobs": avg_logprobs,
                    "avg_token_entropy": avg_entropy,
                    "correct": correct
                })

            df = pd.DataFrame(rows)
            model_data.setdefault(model_name, {})[task_name] = df

    return model_data


if __name__ == "__main__":
    base_dir = "results_scratch"  # <-- set to your top-level results directory
    results = process_results(base_dir)

    # Save per-model, per-task dataframes
    for model, tasks in results.items():
        for task, df in tasks.items():
            out_dir = os.path.join("processed_results", model)
            os.makedirs(out_dir, exist_ok=True)
            df.to_csv(os.path.join(out_dir, f"{task}.csv"), index=False)
