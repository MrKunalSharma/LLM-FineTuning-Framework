"""Evaluation script for fine-tuned models."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from src.evaluation.metrics import ModelEvaluator
from src.utils.logger import logger
from src.utils.data_utils import DataProcessor

def load_model_for_evaluation(model_path: str, device: str = "cuda"):
    """Load fine-tuned model for evaluation."""
    logger.info(f"Loading model from {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Check if it's a PEFT model
    peft_config_path = Path(model_path) / "adapter_config.json"
    
    if peft_config_path.exists():
        # Load PEFT model
        with open(peft_config_path, 'r') as f:
            peft_config = json.load(f)
        
        base_model_name = peft_config.get("base_model_name_or_path")
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        
        # Load PEFT model
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
    else:
        # Load regular model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned LLM")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data (JSONL)")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", help="Output file for results")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to evaluate")
    
    args = parser.parse_args()
    
    try:
        # Load model
        model, tokenizer = load_model_for_evaluation(args.model_path, args.device)
        
        # Load test data
        processor = DataProcessor(tokenizer)
        test_data = processor.load_jsonl(args.test_data)
        
        # Limit samples if specified
        if args.max_samples:
            test_data = test_data[:args.max_samples]
        
        # Initialize evaluator
        evaluator = ModelEvaluator(model, tokenizer)
        
        # Run evaluation
        results = evaluator.comprehensive_evaluate(test_data)
        
        # Save results
        with open(args.output_file, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        
        logger.info(f"Evaluation results saved to {args.output_file}")
        
        # Print summary
        print("\n=== Evaluation Results ===")
        print(f"Perplexity: {results.perplexity:.2f}")
        print(f"BLEU Score: {results.bleu_score:.4f}")
        print(f"ROUGE-L: {results.rouge_scores['rougeL']:.4f}")
        print(f"BERTScore F1: {results.bert_score['f1']:.4f}")
        print(f"Factuality: {results.factuality_score:.4f}")
        print(f"Coherence: {results.coherence_score:.4f}")
        print(f"Safety Score: {results.toxicity_score:.4f}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()

