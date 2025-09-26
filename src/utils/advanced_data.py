"""Advanced data processing with streaming and augmentation."""
from datasets import IterableDataset
import torch
from typing import Iterator, Dict, List
from transformers import PreTrainedTokenizer
import random

class StreamingDataProcessor:
    """Handle large datasets with streaming."""
    
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        augment: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        
        # Augmentation strategies
        self.augmentations = [
            self.paraphrase_instruction,
            self.add_noise,
            self.synonym_replacement
        ]
    
    def create_streaming_dataset(
        self, 
        file_path: str,
        buffer_size: int = 10000
    ) -> IterableDataset:
        """Create streaming dataset for large files."""
        def data_generator():
            buffer = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    
                    # Apply augmentation
                    if self.augment and random.random() < 0.3:
                        data = self.apply_augmentation(data)
                    
                    buffer.append(data)
                    
                    if len(buffer) >= buffer_size:
                        # Shuffle buffer
                        random.shuffle(buffer)
                        for item in buffer:
                            yield self.process_item(item)
                        buffer = []
                
                # Process remaining items
                for item in buffer:
                    yield self.process_item(item)
        
        return IterableDataset.from_generator(data_generator)
    
    def paraphrase_instruction(self, data: Dict) -> Dict:
        """Augment by paraphrasing instruction."""
        paraphrase_templates = [
            "Please {instruction}",
            "Could you {instruction}",
            "I need you to {instruction}",
            "{instruction}. Can you help?",
        ]
        
        template = random.choice(paraphrase_templates)
        data["instruction"] = template.format(
            instruction=data["instruction"].lower()
        )
        return data
    
    def add_noise(self, data: Dict) -> Dict:
        """Add minor noise to inputs."""
        if data.get("input"):
            words = data["input"].split()
            # Randomly swap adjacent words
            if len(words) > 2 and random.random() < 0.1:
                idx = random.randint(0, len(words) - 2)
                words[idx], words[idx + 1] = words[idx + 1], words[idx]
            data["input"] = " ".join(words)
        return data
    
    def synonym_replacement(self, data: Dict) -> Dict:
        """Replace words with synonyms."""
        # Simple implementation - in production, use WordNet
        simple_synonyms = {
            "create": "make",
            "build": "construct",
            "show": "display",
            "write": "compose"
        }
        
        instruction_words = data["instruction"].split()
        for i, word in enumerate(instruction_words):
            if word.lower() in simple_synonyms and random.random() < 0.3:
                instruction_words[i] = simple_synonyms[word.lower()]
        
        data["instruction"] = " ".join(instruction_words)
        return data
