import os
import json
import pandas as pd
from ast import literal_eval


def process_results(base_dir):
    model_data = {}

    for model_name in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_name)
        if not os.path.isdir(model_path):
            continue

        for file in os.listdir(model_path):
            if not file.endswith(".jsonl"):
                continue

            # Extract task name
            parts = file.split("samples_")[-1].split("_")
            task_name = "_".join(parts[1:-1])  # remove leading 'mmlu' and timestamp

            file_path = os.path.join(model_path, file)
            with open(file_path, "r") as f:
                records = [json.loads(line) for line in f]

            rows = []
            for record in records:
                doc_id = record["doc_id"]
                target = int(record["target"])
                filtered_resps = [literal_eval(r) for r in record["filtered_resps"]]

                # Get values for correct answer
                correct_resp = filtered_resps[target]
                avg_logprobs = sum(correct_resp["logprobs"]) / len(correct_resp["logprobs"])
                avg_entropy = sum(correct_resp["entropies"]) / len(correct_resp["entropies"])

                # Check if model prediction was correct
                best_idx = min(range(len(filtered_resps)), key=lambda i: filtered_resps[i]["total_loglikelihood"])
                correct = 1 if best_idx == target else 0

                rows.append({
                    "doc_id": doc_id,
                    "avg_logprobs": avg_logprobs,
                    "avg_token_entropy": avg_entropy,
                    "correct": correct
                })

            df = pd.DataFrame(rows)

            if model_name not in model_data:
                model_data[model_name] = {}

            model_data[model_name][task_name] = df

    return model_data


if __name__ == "__main__":
    base_dir = "results_scratch"
    results = process_results(base_dir)

    # Example: Save per-model, per-task dataframes
    for model, tasks in results.items():
        for task, df in tasks.items():
            out_dir = os.path.join("processed_results", model)
            os.makedirs(out_dir, exist_ok=True)
            df.to_csv(os.path.join(out_dir, f"{task}.csv"), index=False)
