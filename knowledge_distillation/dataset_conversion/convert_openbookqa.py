#!/usr/bin/env python3
"""
Conversion script to transform OpenBookQA (allenai/openbookqa)
from Hugging Face into the standardized multiple-choice question format.

Usage:
    python convert_openbookqa.py --output openbookqa_standard.json
"""

import argparse
import json
from datasets import load_dataset
from dataset_loader import save_standard_format, MCQItem


def load_openbookqa_dataset():
    """Load OpenBookQA dataset from Hugging Face and combine splits."""
    print("Loading OpenBookQA dataset from Hugging Face...")
    dataset = load_dataset("allenai/openbookqa")

    # Choose which splits to include. Adjust as needed.
    include_splits = ["train"]

    all_items = []
    for split_name, split_data in dataset.items():
        if split_name not in include_splits:
            continue
        print(f"Processing {split_name} split with {len(split_data)} items...")
        for row in split_data:
            row["_split_name"] = split_name
            all_items.append(row)

    print(f"Total items loaded: {len(all_items)}")
    return all_items


def _gather_choices_and_labels(item):
    """
    Collect options and labels from OpenBookQA in canonical order.
    item['choices'] is a dict with keys 'text' and 'label', e.g.:
      {'text': ['disease','fecal matter','fuel','fertilizer'],
       'label': ['A','B','C','D']}
    We return (choices, labels) keeping their given order.
    """
    choices_dict = item.get("choices") or {}
    texts = choices_dict.get("text") or []
    labels = choices_dict.get("label") or []

    # Normalize: ensure strings, strip whitespace
    choices = [(t or "").strip() if isinstance(t, str) else "" for t in texts]
    labels = [(l or "").strip() if isinstance(l, str) else "" for l in labels]

    # If labels are missing, synthesize A/B/C/D based on length to keep indices stable.
    if not labels or len(labels) != len(choices):
        fallback = ["A", "B", "C", "D", "E", "F"]
        labels = fallback[: len(choices)]

    return choices, labels


def convert_openbookqa_to_standard(obqa_items):
    """Convert OpenBookQA items to standardized MCQItem list."""
    print("Converting OpenBookQA items to standardized format...")

    standard_items = []

    for idx, item in enumerate(obqa_items):
        qid = str(item.get("id", idx))
        # OBQA uses 'question_stem' for the prompt
        question = (item.get("question_stem") or "").strip()
        choices, labels = _gather_choices_and_labels(item)

        # Answer in OBQA is a single letter in 'answerKey'
        answer_label = (item.get("answerKey") or "").strip()

        # Basic validations
        if not question:
            print(f"Warning: Empty question for item {idx} (qid={qid}), skipping...")
            continue

        if not choices:
            print(f"Warning: No choices for item {idx} (qid={qid}), skipping...")
            continue

        # Determine the answer index. Prefer the provided labels list to stay robust.
        try:
            answer_idx = labels.index(answer_label)
        except ValueError:
            # Fallback to classic A=0, B=1,...
            label_to_idx = {chr(ord("A") + i): i for i in range(len(choices))}
            answer_idx = label_to_idx.get(answer_label, None)

        if not isinstance(answer_idx, int) or not (0 <= answer_idx < len(choices)):
            print(
                f"Warning: Invalid answer label '{answer_label}' for item {idx} "
                f"(qid={qid}), skipping..."
            )
            continue

        correct_answer = choices[answer_idx]

        # OpenBookQA doesn't ship subject categories; keep 'unknown' to match your schema.
        category = "unknown"

        mcq_item = MCQItem(
            qid=qid,
            question=question,
            options=choices,
            correct=correct_answer,
            metadata={
                "category": category,
                "original_answer_label": answer_label,
                "original_answer_index": answer_idx,
                "original_labels": labels,
                "source": "openbookqa",
                "split": item.get("_split_name", "unknown"),
            },
        )

        standard_items.append(mcq_item)

    print(f"Successfully converted {len(standard_items)} items")
    return standard_items


def analyze_categories(items):
    """Analyze the distribution of categories (kept as 'unknown' for OBQA)."""
    counts = {}
    for it in items:
        cat = it.metadata.get("category", "unknown")
        counts[cat] = counts.get(cat, 0) + 1

    print("\nCategory distribution:")
    for cat in sorted(counts):
        print(f"  {cat}: {counts[cat]} questions")

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Convert OpenBookQA dataset to standardized format"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="openbookqa_standard.json",
        help="Path to output standardized JSON file (default: openbookqa_standard.json)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="OpenBookQA",
        help="Name for the dataset (default: OpenBookQA)",
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default="1.0",
        help="Version for the dataset (default: 1.0)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the converted dataset after conversion",
    )

    args = parser.parse_args()

    try:
        # 1) Load dataset
        obqa_items = load_openbookqa_dataset()

        # 2) Convert
        standard_items = convert_openbookqa_to_standard(obqa_items)
        if not standard_items:
            print("Error: No items were successfully converted.")
            return 1

        # 3) Analyze (trivial for OBQA; still prints a summary to mirror MedMCQA flow)
        _ = analyze_categories(standard_items)

        # 4) Save
        print(f"\nSaving standardized format to: {args.output}")
        save_standard_format(
            items=standard_items,
            output_path=args.output,
            dataset_name=args.dataset_name,
            dataset_version=args.dataset_version,
        )
        print("Conversion completed successfully!")

        # 5) Optional validation
        if args.validate:
            print("\nValidating converted dataset...")
            try:
                with open(args.output, "r", encoding="utf-8") as f:
                    converted = json.load(f)

                print("Converted dataset info:")
                print(f"  - Dataset name: {converted.get('dataset_name', 'N/A')}")
                print(f"  - Dataset version: {converted.get('dataset_version', 'N/A')}")
                print(f"  - Format version: {converted.get('format_version', 'N/A')}")
                print(f"  - Number of items: {converted.get('num_items', 'N/A')}")

                if (
                    "items" in converted
                    and isinstance(converted["items"], list)
                    and converted["items"]
                ):
                    print("  - Items structure: ✓ Valid")
                    first = converted["items"][0]
                    required = ["qid", "question", "options", "correct"]
                    missing = [k for k in required if k not in first]
                    if not missing:
                        print("  - Item structure: ✓ Valid")
                        print(f"  - First question: {first['question'][:100]}...")
                        print(f"  - Number of options: {len(first['options'])}")
                        meta = first.get("metadata", {}) or {}
                        print(f"  - Category: {meta.get('category', 'N/A')}")
                    else:
                        print(f"  - Item structure: ✗ Missing fields: {missing}")
                else:
                    print("  - Items structure: ✗ Invalid or empty")

            except Exception as e:
                print(f"Error validating converted dataset: {e}")
                return 1

        return 0

    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
