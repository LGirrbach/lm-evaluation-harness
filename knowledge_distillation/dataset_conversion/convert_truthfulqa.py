#!/usr/bin/env python3
"""
Conversion script to transform TruthfulQA dataset from original mc1_targets format
to the new standardized multiple-choice question format.

Usage:
    python convert_truthfulqa.py --input truthfulqa_questions.json --output truthfulqa_standard.json
"""

import argparse
import json
import os
from dataset_loader import TruthfulQALoader, save_standard_format


def main():
    parser = argparse.ArgumentParser(
        description="Convert TruthfulQA dataset to standardized format"
    )
    parser.add_argument(
        "--input", "-i", 
        type=str, 
        required=True,
        help="Path to input TruthfulQA JSON file (original mc1_targets format)"
    )
    parser.add_argument(
        "--output", "-o", 
        type=str, 
        required=True,
        help="Path to output standardized JSON file"
    )
    parser.add_argument(
        "--dataset-name", 
        type=str, 
        default="TruthfulQA",
        help="Name for the dataset (default: TruthfulQA)"
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
    
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        return 1
    
    try:
        print(f"Loading TruthfulQA dataset from: {args.input}")
        
        # Load using TruthfulQA loader
        loader = TruthfulQALoader()
        items = loader.load(args.input)
        
        print(f"Loaded {len(items)} questions successfully.")
        
        # Validate the loaded data
        if not loader.validate(items):
            print("Warning: Dataset validation failed!")
            return 1
        
        print("Dataset validation passed.")
        
        # Convert to standard format
        print(f"Saving standardized format to: {args.output}")
        save_standard_format(
            items=items,
            output_path=args.output,
            dataset_name=args.dataset_name,
            dataset_version=args.dataset_version
        )
        
        print("Conversion completed successfully!")
        
        # Optional validation of output
        if args.validate:
            print("Validating converted dataset...")
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
