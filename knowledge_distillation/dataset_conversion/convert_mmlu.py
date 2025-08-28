#!/usr/bin/env python3
"""
Conversion script to transform MMLU dataset from Hugging Face to the 
new standardized multiple-choice question format.

Usage:
    python convert_mmlu.py --output mmlu_standard.json
"""

import argparse
import json
import os
from datasets import load_dataset
from dataset_loader import save_standard_format, MCQItem


def load_mmlu_dataset():
    """Load MMLU dataset from Hugging Face."""
    print("Loading MMLU dataset from Hugging Face...")
    dataset = load_dataset("cais/mmlu", "all")
    
    # Combine all splits (train, validation, test)
    all_items = []
    
    for split_name, split_data in dataset.items():
        if split_name != "test":
            continue
        print(f"Processing {split_name} split with {len(split_data)} items...")
        all_items.extend(split_data)
    
    print(f"Total items loaded: {len(all_items)}")
    return all_items


def convert_mmlu_to_standard(mmlu_items):
    """Convert MMLU items to standardized MCQItem format."""
    print("Converting MMLU items to standardized format...")
    
    standard_items = []
    
    for idx, item in enumerate(mmlu_items):
        # Extract fields from MMLU format
        question = item["question"]
        choices = item["choices"]
        answer_idx = item["answer"]
        subject = item.get("subject", "unknown")
        
        # Validate answer index
        if not (0 <= answer_idx < len(choices)):
            print(f"Warning: Invalid answer index {answer_idx} for item {idx}, skipping...")
            continue
        
        # Get correct answer from choices
        correct_answer = choices[answer_idx]
        
        # Create MCQItem with metadata
        mcq_item = MCQItem(
            qid=str(idx),
            question=question,
            options=choices,
            correct=correct_answer,
            metadata={
                "subject": subject,
                "original_answer_index": answer_idx,
                "source": "mmlu",
                "split": "combined"  # Since we're combining all splits
            }
        )
        
        standard_items.append(mcq_item)
    
    print(f"Successfully converted {len(standard_items)} items")
    return standard_items


def analyze_subjects(items):
    """Analyze the distribution of subjects in the dataset."""
    subject_counts = {}
    for item in items:
        subject = item.metadata.get("subject", "unknown")
        subject_counts[subject] = subject_counts.get(subject, 0) + 1
    
    print("\nSubject distribution:")
    for subject, count in sorted(subject_counts.items()):
        print(f"  {subject}: {count} questions")
    
    return subject_counts


def main():
    parser = argparse.ArgumentParser(
        description="Convert MMLU dataset to standardized format"
    )
    parser.add_argument(
        "--output", "-o", 
        type=str, 
        default="mmlu_standard.json",
        help="Path to output standardized JSON file (default: mmlu_standard.json)"
    )
    parser.add_argument(
        "--dataset-name", 
        type=str, 
        default="MMLU",
        help="Name for the dataset (default: MMLU)"
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
        # Load MMLU dataset
        mmlu_items = load_mmlu_dataset()
        
        # Convert to standard format
        standard_items = convert_mmlu_to_standard(mmlu_items)
        
        if not standard_items:
            print("Error: No items were successfully converted.")
            return 1
        
        # Analyze subjects
        subject_counts = analyze_subjects(standard_items)
        
        # Save in standard format
        print(f"\nSaving standardized format to: {args.output}")
        save_standard_format(
            items=standard_items,
            output_path=args.output,
            dataset_name=args.dataset_name,
            dataset_version=args.dataset_version
        )
        
        print("Conversion completed successfully!")
        
        # Optional validation of output
        if args.validate:
            print("\nValidating converted dataset...")
            try:
                with open(args.output, "r", encoding="utf-8") as f:
                    converted_data = json.load(f)
                
                print(f"Converted dataset info:")
                print(f"  - Dataset name: {converted_data.get('dataset_name', 'N/A')}")
                print(f"  - Dataset version: {converted_data.get('dataset_version', 'N/A')}")
                print(f"  - Format version: {converted_data.get('format_version', 'N/A')}")
                print(f"  - Number of items: {converted_data.get('num_items', 'N/A')}")
                
                # Verify structure
                if "items" in converted_data and isinstance(converted_data["items"], list):
                    print(f"  - Items structure: ✓ Valid")
                    
                    # Check first few items
                    first_item = converted_data["items"][0]
                    required_fields = ["qid", "question", "options", "correct"]
                    missing_fields = [f for f in required_fields if f not in first_item]
                    
                    if not missing_fields:
                        print(f"  - Item structure: ✓ Valid")
                        print(f"  - First question: {first_item['question'][:100]}...")
                        print(f"  - Number of options: {len(first_item['options'])}")
                        print(f"  - Subject metadata: {first_item.get('subject', 'N/A')}")
                    else:
                        print(f"  - Item structure: ✗ Missing fields: {missing_fields}")
                else:
                    print(f"  - Items structure: ✗ Invalid")
                    
            except Exception as e:
                print(f"Error validating converted dataset: {e}")
                return 1
        
        return 0
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
