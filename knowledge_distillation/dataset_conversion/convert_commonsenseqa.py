#!/usr/bin/env python3
"""
Conversion script to transform CommonsenseQA (tau/commonsense_qa)
from Hugging Face into the standardized multiple-choice question format.

Usage:
    python convert_commonsenseqa.py --output commonsenseqa_standard.json
"""

import argparse
import json
from datasets import load_dataset
from dataset_loader import save_standard_format, MCQItem


def load_commonsenseqa_dataset():
    """Load CommonsenseQA dataset from Hugging Face and combine splits."""
    print("Loading CommonsenseQA dataset from Hugging Face...")
    dataset = load_dataset("tau/commonsense_qa")

    # Choose which splits to include. Adjust as needed.
    # Note: the test split may not include answers; validation/train do.
    include_splits = ["validation"]

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


def _gather_choices(item):
    """
    Collect options from CommonsenseQA fields in canonical order.
    CSQA provides:
      - item['choices']['label'] e.g., ['A','B','C','D','E']
      - item['choices']['text']  e.g., ['optA','optB','optC','optD','optE']
    We'll keep options ordered as given by 'label' to ensure 'answerKey' aligns.
    """
    choices_obj = item.get("choices") or {}
    labels = choices_obj.get("label") or []
    texts = choices_obj.get("text") or []

    # Build a label->text mapping, then order by the label sequence
    # while being robust to length mismatches.
    label_to_text = {}
    for i in range(min(len(labels), len(texts))):
        lbl = labels[i]
        txt = texts[i] if isinstance(texts[i], str) else ""
        label_to_text[lbl] = (txt.strip() if txt else "")

    ordered_choices = [label_to_text.get(lbl, "") for lbl in labels]
    return labels, ordered_choices


def convert_commonsenseqa_to_standard(csqa_items):
    """Convert CommonsenseQA items to standardized MCQItem list."""
    print("Converting CommonsenseQA items to standardized format...")

    standard_items = []

    for idx, item in enumerate(csqa_items):
        qid = str(item.get("id", idx))
        question = (item.get("question") or "").strip()

        labels, choices = _gather_choices(item)
        answer_key = (item.get("answerKey") or "").strip()

        # Basic validations
        if not question:
            print(f"Warning: Empty question for item {idx} (qid={qid}), skipping...")
            continue

        if not labels or not choices or len(labels) != len(choices):
            print(f"Warning: Invalid/mismatched choices for item {idx} (qid={qid}), skipping...")
            continue

        # Map answerKey (e.g., 'A') to index using provided labels order
        try:
            answer_idx = labels.index(answer_key)
        except ValueError:
            print(f"Warning: answerKey '{answer_key}' not found among labels {labels} "
                  f"for item {idx} (qid={qid}), skipping...")
            continue

        if not (0 <= answer_idx < len(choices)):
            print(f"Warning: Computed answer index {answer_idx} out of range for item {idx} (qid={qid}), skipping...")
            continue

        correct_answer = choices[answer_idx]

        # Use question_concept as a lightweight "category"
        category = item.get("question_concept") or "unknown"

        mcq_item = MCQItem(
            qid=qid,
            question=question,
            options=choices,
            correct=correct_answer,
            metadata={
                "category": category,
                "original_answer_label": answer_key,
                "original_labels_order": labels,
                "original_answer_index": answer_idx,
                "source": "commonsense_qa",
                "split": item.get("_split_name", "unknown"),
                # Optional extra fields that might be useful downstream:
                "question_concept": item.get("question_concept"),
            },
        )

        standard_items.append(mcq_item)

    print(f"Successfully converted {len(standard_items)} items")
    return standard_items


def analyze_categories(items):
    """Analyze the distribution of categories (question_concept) in the dataset."""
    counts = {}
    for it in items:
        cat = it.metadata.get("category", "unknown")
        counts[cat] = counts.get(cat, 0) + 1

    print("\nCategory distribution (top 25 shown):")
    # Print the top 25 by frequency to avoid flooding the console
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:25]
    for cat, n in top:
        print(f"  {cat}: {n} questions")
    if len(counts) > 25:
        print(f"  ... (+{len(counts) - 25} more concepts)")
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Convert CommonsenseQA dataset to standardized format"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="commonsenseqa_standard.json",
        help="Path to output standardized JSON file (default: commonsenseqa_standard.json)"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="CommonsenseQA",
        help="Name for the dataset (default: CommonsenseQA)"
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
        csqa_items = load_commonsenseqa_dataset()

        # 2) Convert
        standard_items = convert_commonsenseqa_to_standard(csqa_items)
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
