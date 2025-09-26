"""Data processing utilities."""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split
from .logger import logger

class DataProcessor:
    """Handles data loading and preprocessing."""
    
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from JSONL file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        logger.info(f"Loaded {len(data)} examples from {file_path}")
        return data
    
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} examples from {file_path}")
        return df
    
    def create_prompt(self, instruction: str, input_text: str = "", output: str = "") -> str:
        """
        Create a formatted prompt for fine-tuning.
        
        Args:
            instruction: Task instruction
            input_text: Optional input context
            output: Expected output (for training)
            
        Returns:
            Formatted prompt string
        """
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        return prompt
    
    def tokenize_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Tokenize examples for training."""
        model_inputs = self.tokenizer(
            examples["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None
        )
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs
    
    def prepare_dataset(
        self,
        data: List[Dict[str, Any]],
        test_size: float = 0.1,
        seed: int = 42
    ) -> DatasetDict:
        """
        Prepare dataset for training and evaluation.
        
        Args:
            data: List of data dictionaries
            test_size: Fraction for test split
            seed: Random seed
            
        Returns:
            DatasetDict with train and test splits
        """
        # Create formatted prompts
        formatted_data = []
        for item in data:
            text = self.create_prompt(
                instruction=item.get("instruction", ""),
                input_text=item.get("input", ""),
                output=item.get("output", "")
            )
            formatted_data.append({"text": text})
        
        # Split data
        train_data, test_data = train_test_split(
            formatted_data, test_size=test_size, random_state=seed
        )
        
        # Create datasets
        train_dataset = Dataset.from_list(train_data)
        test_dataset = Dataset.from_list(test_data)
        
        # Tokenize
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        test_dataset = test_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })
        
        logger.info(f"Dataset prepared - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
        return dataset_dict
