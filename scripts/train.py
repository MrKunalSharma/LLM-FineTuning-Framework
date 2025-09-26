"""Main training script."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
from src.training.trainer import LLMTrainer
from src.utils.data_utils import DataProcessor
from src.utils.logger import logger
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--run_name", type=str, help="Name for this training run")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases")
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = LLMTrainer(args.config)
        
        # Load tokenizer for data processing
        tokenizer = AutoTokenizer.from_pretrained(
            trainer.model_config.name,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Process data
        processor = DataProcessor(tokenizer, trainer.model_config.max_length)
        
        # Load data (supports JSONL)
        data = processor.load_jsonl(args.data_path)
        dataset = processor.prepare_dataset(data)
        
        # Train
        trainer.train(
            dataset=dataset,
            run_name=args.run_name,
            use_wandb=not args.no_wandb
        )
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
