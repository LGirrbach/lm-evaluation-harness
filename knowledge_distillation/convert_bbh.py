#!/usr/bin/env python3
"""
Conversion script to transform BBH dataset from Hugging Face to the 
new standardized multiple-choice question format.

Usage:
    python convert_bbh.py --output bbh_standard.json
"""

import argparse
import json
import os
from datasets import load_dataset
from dataset_loader import save_standard_format, MCQItem


def load_bbh_dataset():
    """Load BBH dataset from Hugging Face."""
    print("Loading BBH dataset from Hugging Face...")
    
    # BBH categories as specified
    bbh_categories = [
        'causal_judgement', 'date_understanding', 'disambiguation_qa', 
        'dyck_languages', 'geometric_shapes', 'logical_deduction_five_objects', 
        'logical_deduction_seven_objects', 'logical_deduction_three_objects', 
        'movie_recommendation', 'navigate', 'reasoning_about_colored_objects', 
        'ruin_names', 'salient_translation_error_detection', 'snarks', 
        'sports_understanding', 'temporal_sequences', 
        'tracking_shuffled_objects_five_objects', 
        'tracking_shuffled_objects_seven_objects', 
        'tracking_shuffled_objects_three_objects'
    ]
    
    all_items = []
    
    for category in bbh_categories:
        try:
            print(f"Loading category: {category}")
            dataset = load_dataset("lighteval/bbh", category)
            
            # Process train split for each category
            if "train" in dataset:
                train_data = dataset["train"]
                print(f"  - {category}: {len(train_data)} items")
                
                # Add category information to each item
                for item in train_data:
                    item = dict(item)
                    item["category"] = category
                
                all_items.extend(train_data)
            else:
                print(f"  - {category}: No train split found")
                
        except Exception as e:
            print(f"  - {category}: Error loading - {e}")
            continue
    
    print(f"Total items loaded: {len(all_items)}")
    return all_items


def convert_bbh_to_standard(bbh_items):
    """Convert BBH items to standardized MCQItem format."""
    print("Converting BBH items to standardized format...")
    
    standard_items = []
    
    for idx, item in enumerate(bbh_items):
        # Extract fields from BBH format
        input_text = item["input"]
        task_prefix = item.get("task_prefix", "")
        choices = item["choices"]
        target_idx = item["target_idx"]
        category = item.get("category", "unknown")
        
        # Combine task prefix and input into a single question
        if task_prefix and task_prefix.strip():
            # Remove trailing newlines and combine
            task_prefix = task_prefix.rstrip('\n')
            question = f"{task_prefix}\n\n{input_text}"
        else:
            question = input_text
        
        # Validate target index
        if not (0 <= target_idx < len(choices)):
            print(f"Warning: Invalid target index {target_idx} for item {idx}, skipping...")
            continue
        
        # Get correct answer from choices
        correct_answer = choices[target_idx]
        
        # Create MCQItem with metadata
        mcq_item = MCQItem(
            qid=str(idx),
            question=question,
            options=choices,
            correct=correct_answer,
            metadata={
                "category": category,
                "original_target_index": target_idx,
                "source": "bbh",
                "original_id": item.get("id", str(idx)),
                "choice_prefix": item.get("choice_prefix"),
                "append_choices": item.get("append_choices", False),
                "example_input_prefix": item.get("example_input_prefix"),
                "example_output_prefix": item.get("example_output_prefix")
            }
        )
        
        standard_items.append(mcq_item)
    
    print(f"Successfully converted {len(standard_items)} items")
    return standard_items


def analyze_categories(items):
    """Analyze the distribution of categories in the dataset."""
    category_counts = {}
    for item in items:
        category = item.metadata.get("category", "unknown")
        category_counts[category] = category_counts.get(category, 0) + 1
    
    print("\nCategory distribution:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count} questions")
    
    return category_counts


def main():
    parser = argparse.ArgumentParser(
        description="Convert BBH dataset to standardized format"
    )
    parser.add_argument(
        "--output", "-o", 
        type=str, 
        default="bbh_standard.json",
        help="Path to output standardized JSON file (default: bbh_standard.json)"
    )
    parser.add_argument(
        "--dataset-name", 
        type=str, 
        default="BBH",
        help="Name for the dataset (default: BBH)"
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
        # Load BBH dataset
        bbh_items = load_bbh_dataset()
        
        # Convert to standard format
        standard_items = convert_bbh_to_standard(bbh_items)
        
        if not standard_items:
            print("Error: No items were successfully converted.")
            return 1
        
        # Analyze categories
        category_counts = analyze_categories(standard_items)
        
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
                        print(f"  - Category metadata: {first_item.get('category', 'N/A')}")
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
