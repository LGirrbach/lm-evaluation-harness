#!/usr/bin/env python3
"""
Conversion script to transform MedMCQA (openlifescienceai/medmcqa)
from Hugging Face into the standardized multiple-choice question format.

Usage:
    python convert_medmcqa.py --output medmcqa_standard.json
"""

import argparse
import json
from datasets import load_dataset
from dataset_loader import save_standard_format, MCQItem


def load_medmcqa_dataset():
    """Load MedMCQA dataset from Hugging Face and combine splits."""
    print("Loading MedMCQA dataset from Hugging Face...")
    dataset = load_dataset("openlifescienceai/medmcqa")

    # Choose which splits to include. Adjust as needed.
    include_splits = ["validation"]

    all_items = []
    for split_name, split_data in dataset.items():
        if split_name not in include_splits:
            continue
        print(f"Processing {split_name} split with {len(split_data)} items...")
        # Attach split name to each row for metadata later
        for row in split_data:
            row["_split_name"] = split_name
            all_items.append(row)

    print(f"Total items loaded: {len(all_items)}")
    return all_items


def _gather_choices(item):
    """Collect options from MedMCQA fields in canonical order."""
    # Some rows can have None/empty strings; filter them out while preserving order.
    raw_choices = [item.get("opa"), item.get("opb"), item.get("opc"), item.get("opd")]
    choices = [c.strip() if isinstance(c, str) else "" for c in raw_choices]
    # Keep even empty strings to preserve indexing, but we'll validate the answer index later.
    return choices


def convert_medmcqa_to_standard(med_items):
    """Convert MedMCQA items to standardized MCQItem list."""
    print("Converting MedMCQA items to standardized format...")

    standard_items = []

    for idx, item in enumerate(med_items):
        qid = str(item.get("id", idx))
        question = (item.get("question") or "").strip()
        choices = _gather_choices(item)

        # Answer index in MedMCQA: 'cop' (0-based index pointing into [opa, opb, opc, opd])
        answer_idx = item.get("cop")

        # Basic validations
        if not question:
            print(f"Warning: Empty question for item {idx} (qid={qid}), skipping...")
            continue

        if not isinstance(answer_idx, int) or not (0 <= answer_idx < len(choices)):
            print(f"Warning: Invalid answer index {answer_idx} for item {idx} (qid={qid}), skipping...")
            continue

        correct_answer = choices[answer_idx]

        # Category comes from 'subject_name'; save as 'category' in metadata
        category = item.get("subject_name") or "unknown"

        mcq_item = MCQItem(
            qid=qid,
            question=question,
            options=choices,
            correct=correct_answer,
            metadata={
                "category": category,
                "original_answer_index": answer_idx,
                "source": "medmcqa",
                "split": item.get("_split_name", "unknown"),
                # Optional extra fields that might be useful downstream:
                "choice_type": item.get("choice_type"),
                "topic_name": item.get("topic_name"),
            },
        )

        standard_items.append(mcq_item)

    print(f"Successfully converted {len(standard_items)} items")
    return standard_items


def analyze_categories(items):
    """Analyze the distribution of categories (subjects) in the dataset."""
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
        description="Convert MedMCQA dataset to standardized format"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="medmcqa_standard.json",
        help="Path to output standardized JSON file (default: medmcqa_standard.json)"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="MedMCQA",
        help="Name for the dataset (default: MedMCQA)"
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default="1.0",
        help="Version for the dataset (default: 1.0)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the converted dataset after conversion"
    )

    args = parser.parse_args()

    try:
        # 1) Load dataset
        med_items = load_medmcqa_dataset()

        # 2) Convert
        standard_items = convert_medmcqa_to_standard(med_items)
        if not standard_items:
            print("Error: No items were successfully converted.")
            return 1

        # 3) Analyze categories
        _ = analyze_categories(standard_items)

        # 4) Save
        print(f"\nSaving standardized format to: {args.output}")
        save_standard_format(
            items=standard_items,
            output_path=args.output,
            dataset_name=args.dataset_name,
            dataset_version=args.dataset_version
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

                if "items" in converted and isinstance(converted["items"], list) and converted["items"]:
                    print("  - Items structure: ✓ Valid")
                    first = converted["items"][0]
                    required = ["qid", "question", "options", "correct"]
                    missing = [k for k in required if k not in first]
                    if not missing:
                        print("  - Item structure: ✓ Valid")
                        print(f"  - First question: {first['question'][:100]}...")
                        print(f"  - Number of options: {len(first['options'])}")
                        # category lives in metadata
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
