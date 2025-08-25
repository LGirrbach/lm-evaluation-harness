"""
Unified dataset loader interface for multiple-choice questions.

This module provides a standardized way to load and process multiple-choice question datasets
from various sources, making it easy to extend the surprisingly popular evaluation to new datasets.
"""

import json
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class MCQItem:
    """Standardized representation of a multiple-choice question item."""
    qid: str  # Question ID (can be string or int)
    question: str  # The question text
    options: List[str]  # List of answer options
    correct: str  # The correct answer (must match one of the options exactly)
    metadata: Optional[Dict[str, Any]] = None  # Additional dataset-specific metadata


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""
    
    @abstractmethod
    def load(self, file_path: str) -> List[MCQItem]:
        """Load dataset from file and return standardized MCQItem objects."""
        pass
    
    @abstractmethod
    def validate(self, items: List[MCQItem]) -> bool:
        """Validate that loaded items meet the requirements."""
        pass


class TruthfulQALoader(DatasetLoader):
    """Loader for TruthfulQA dataset in the original mc1_targets format."""
    
    def load(self, file_path: str) -> List[MCQItem]:
        """Load TruthfulQA dataset from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("TruthfulQA JSON must be a list of items.")
        
        items = []
        for qid, item in enumerate(data):
            if "question" not in item or "mc1_targets" not in item:
                raise ValueError(f"Item {qid} missing required fields: question, mc1_targets")
            
            question = item["question"]
            mc1_targets = item["mc1_targets"]
            
            # Extract options and correct answer
            options = list(mc1_targets.keys())
            correct = None
            for opt, val in mc1_targets.items():
                if int(val) == 1:
                    correct = opt
                    break
            
            if correct is None:
                raise ValueError(f"No correct option found in item {qid}")
            
            # Store original metadata
            metadata = {k: v for k, v in item.items() if k not in ["question", "mc1_targets"]}
            
            items.append(MCQItem(
                qid=str(qid),
                question=question,
                options=options,
                correct=correct,
                metadata=metadata
            ))
        
        return items
    
    def validate(self, items: List[MCQItem]) -> bool:
        """Validate TruthfulQA items."""
        for item in items:
            if not item.question.strip():
                return False
            if len(item.options) < 2:
                return False
            if item.correct not in item.options:
                return False
        return True


class StandardMCQLoader(DatasetLoader):
    """Loader for datasets in the new standardized format."""
    
    def load(self, file_path: str) -> List[MCQItem]:
        """Load dataset from standardized JSON format."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not isinstance(data, dict) or "items" not in data:
            raise ValueError("Standard JSON must have 'items' field containing the question list.")
        
        items_data = data["items"]
        if not isinstance(items_data, list):
            raise ValueError("'items' field must be a list.")
        
        items = []
        for item_data in items_data:
            required_fields = ["qid", "question", "options", "correct"]
            for field in required_fields:
                if field not in item_data:
                    raise ValueError(f"Item missing required field: {field}")
            
            # Store additional metadata
            metadata = {k: v for k, v in item_data.items() if k not in required_fields}
            
            items.append(MCQItem(
                qid=str(item_data["qid"]),
                question=item_data["question"],
                options=item_data["options"],
                correct=item_data["correct"],
                metadata=metadata
            ))
        
        return items
    
    def validate(self, items: List[MCQItem]) -> bool:
        """Validate standardized MCQ items."""
        for item in items:
            if not item.question.strip():
                return False
            if len(item.options) < 2:
                return False
            if item.correct not in item.options:
                return False
            if not item.qid:
                return False
        return True


class MMLULoader(DatasetLoader):
    """Loader for MMLU dataset from Hugging Face."""
    
    def __init__(self, split: str = "all"):
        """
        Initialize MMLU loader.
        
        Args:
            split: Dataset split to load ('train', 'validation', 'test', or 'all')
        """
        self.split = split
    
    def load(self, file_path: str = None) -> List[MCQItem]:
        """
        Load MMLU dataset from Hugging Face.
        
        Note: file_path is ignored for MMLU as it loads directly from HF.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("MMLU loader requires 'datasets' package. Install with: pip install datasets")
        
        print(f"Loading MMLU dataset (split: {self.split}) from Hugging Face...")
        dataset = load_dataset("cais/mmlu", self.split)
        
        items = []
        
        if self.split == "all":
            # Combine all splits
            for split_name, split_data in dataset.items():
                print(f"Processing {split_name} split with {len(split_data)} items...")
                items.extend(self._convert_split(split_data, split_name))
        else:
            # Single split
            items = self._convert_split(dataset, self.split)
        
        print(f"Total MMLU items loaded: {len(items)}")
        return items
    
    def _convert_split(self, split_data, split_name: str) -> List[MCQItem]:
        """Convert a single split of MMLU data to MCQItems."""
        converted_items = []
        
        for idx, item in enumerate(split_data):
            question = item["question"]
            choices = item["choices"]
            answer_idx = item["answer"]
            subject = item.get("subject", "unknown")
            
            # Validate answer index
            if not (0 <= answer_idx < len(choices)):
                print(f"Warning: Invalid answer index {answer_idx} for item {idx} in {split_name}, skipping...")
                continue
            
            # Get correct answer from choices
            correct_answer = choices[answer_idx]
            
            # Create MCQItem with metadata
            mcq_item = MCQItem(
                qid=f"{split_name}_{idx}",
                question=question,
                options=choices,
                correct=correct_answer,
                metadata={
                    "subject": subject,
                    "original_answer_index": answer_idx,
                    "source": "mmlu",
                    "split": split_name
                }
            )
            
            converted_items.append(mcq_item)
        
        return converted_items
    
    def validate(self, items: List[MCQItem]) -> bool:
        """Validate MMLU items."""
        for item in items:
            if not item.question.strip():
                return False
            if len(item.options) < 2:
                return False
            if item.correct not in item.options:
                return False
            if not item.qid:
                return False
            if "subject" not in item.metadata:
                return False
        return True


class DatasetLoaderFactory:
    """Factory for creating appropriate dataset loaders."""
    
    @staticmethod
    def create_loader(file_path: str, format_type: Optional[str] = None) -> DatasetLoader:
        """
        Create appropriate dataset loader based on file content or specified format.
        
        Args:
            file_path: Path to the dataset file (ignored for 'mmlu' format)
            format_type: Optional format specification ('truthfulqa', 'standard', 'mmlu', or None for auto-detect)
        
        Returns:
            Appropriate DatasetLoader instance
        """
        if format_type == "truthfulqa":
            return TruthfulQALoader()
        elif format_type == "standard":
            return StandardMCQLoader()
        elif format_type == "mmlu":
            return MMLULoader()
        elif format_type is None:
            # Auto-detect format
            return DatasetLoaderFactory._auto_detect_loader(file_path)
        else:
            raise ValueError(f"Unknown format type: {format_type}")
    
    @staticmethod
    def _auto_detect_loader(file_path: str) -> DatasetLoader:
        """Auto-detect dataset format by examining file content."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Check if it's the new standard format
            if isinstance(data, dict) and "items" in data:
                return StandardMCQLoader()
            
            # Check if it's TruthfulQA format
            if isinstance(data, list) and len(data) > 0:
                first_item = data[0]
                if "question" in first_item and "mc1_targets" in first_item:
                    return TruthfulQALoader()
            
            # Default to standard format
            return StandardMCQLoader()
            
        except Exception as e:
            raise ValueError(f"Could not auto-detect format: {e}")


def load_dataset(file_path: str, format_type: Optional[str] = None) -> List[MCQItem]:
    """
    Convenience function to load a dataset with automatic format detection.
    
    Args:
        file_path: Path to the dataset file (ignored for 'mmlu' format)
        format_type: Optional format specification
    
    Returns:
        List of standardized MCQItem objects
    """
    loader = DatasetLoaderFactory.create_loader(file_path, format_type)
    
    # For MMLU, we don't need a file path
    if format_type == "mmlu":
        items = loader.load()
    else:
        items = loader.load(file_path)
    
    if not loader.validate(items):
        raise ValueError("Dataset validation failed")
    
    return items


def load_mmlu_dataset(split: str = "all") -> List[MCQItem]:
    """
    Convenience function to load MMLU dataset directly from Hugging Face.
    
    Args:
        split: Dataset split to load ('train', 'validation', 'test', or 'all')
    
    Returns:
        List of standardized MCQItem objects
    """
    loader = MMLULoader(split)
    items = loader.load()
    
    if not loader.validate(items):
        raise ValueError("MMLU dataset validation failed")
    
    return items


def save_standard_format(items: List[MCQItem], output_path: str, 
                        dataset_name: str = "dataset", 
                        dataset_version: str = "1.0") -> None:
    """
    Save MCQItems in the new standardized format.
    
    Args:
        items: List of MCQItem objects
        output_path: Path to save the standardized JSON
        dataset_name: Name of the dataset
        dataset_version: Version of the dataset
    """
    standard_data = {
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "format_version": "1.0",
        "num_items": len(items),
        "items": [
            {
                "qid": item.qid,
                "question": item.question,
                "options": item.options,
                "correct": item.correct,
                **(item.metadata or {})
            }
            for item in items
        ]
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(standard_data, f, indent=2, ensure_ascii=False)
