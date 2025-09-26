"""Evaluation metrics for LLM fine-tuning."""
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from transformers import PreTrainedModel, PreTrainedTokenizer
import evaluate
from bert_score import score as bert_score
from rouge_score import rouge_scorer
import re
from ..utils.logger import logger

@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    perplexity: float
    bleu_score: float
    rouge_scores: Dict[str, float]
    bert_score: Dict[str, float]
    factuality_score: Optional[float] = None
    coherence_score: Optional[float] = None
    toxicity_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "perplexity": self.perplexity,
            "bleu_score": self.bleu_score,
            "rouge_scores": self.rouge_scores,
            "bert_score": self.bert_score,
            "factuality_score": self.factuality_score,
            "coherence_score": self.coherence_score,
            "toxicity_score": self.toxicity_score
        }

class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Initialize metrics
        self.bleu = evaluate.load("bleu")
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )
        
    def calculate_perplexity(self, texts: List[str]) -> float:
        """Calculate perplexity on a list of texts."""
        total_loss = 0
        total_tokens = 0
        
        self.model.eval()
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=self.tokenizer.model_max_length
                ).to(self.device)
                
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)
        
        perplexity = np.exp(total_loss / total_tokens)
        return perplexity
    
    def calculate_bleu(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> float:
        """Calculate BLEU score."""
        # Tokenize for BLEU calculation
        predictions_tokens = [pred.split() for pred in predictions]
        references_tokens = [[ref.split()] for ref in references]
        
        results = self.bleu.compute(
            predictions=predictions_tokens,
            references=references_tokens
        )
        return results["bleu"]
    
    def calculate_rouge(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            for key in rouge_scores:
                rouge_scores[key].append(scores[key].fmeasure)
        
        # Average scores
        return {key: np.mean(scores) for key, scores in rouge_scores.items()}
    
    def calculate_bert_score(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """Calculate BERTScore."""
        P, R, F1 = bert_score(
            predictions, 
            references, 
            lang="en", 
            rescale_with_baseline=True,
            device=self.device
        )
        
        return {
            "precision": P.mean().item(),
            "recall": R.mean().item(),
            "f1": F1.mean().item()
        }
    
    def evaluate_factuality(
        self, 
        predictions: List[str], 
        source_texts: List[str]
    ) -> float:
        """
        Evaluate factual consistency between predictions and source.
        This is a simplified version - you can integrate more sophisticated methods.
        """
        scores = []
        for pred, source in zip(predictions, source_texts):
            # Extract entities/facts from source
            source_facts = set(re.findall(r'\b[A-Z][a-z]+\b', source))
            pred_facts = set(re.findall(r'\b[A-Z][a-z]+\b', pred))
            
            # Calculate overlap
            if source_facts:
                score = len(source_facts.intersection(pred_facts)) / len(source_facts)
            else:
                score = 1.0
            scores.append(score)
        
        return np.mean(scores)
    
    def evaluate_coherence(self, texts: List[str]) -> float:
        """
        Evaluate text coherence using sentence similarity.
        """
        coherence_scores = []
        
        for text in texts:
            sentences = text.split('.')
            if len(sentences) < 2:
                coherence_scores.append(1.0)
                continue
            
            # Simple coherence: check if sentences are related
            # In production, use sentence embeddings
            word_overlap_scores = []
            for i in range(len(sentences) - 1):
                words1 = set(sentences[i].lower().split())
                words2 = set(sentences[i + 1].lower().split())
                if words1 and words2:
                    overlap = len(words1.intersection(words2)) / max(len(words1), len(words2))
                    word_overlap_scores.append(overlap)
            
            if word_overlap_scores:
                coherence_scores.append(np.mean(word_overlap_scores))
            else:
                coherence_scores.append(1.0)
        
        return np.mean(coherence_scores)
    
    def evaluate_toxicity(self, texts: List[str]) -> float:
        """
        Evaluate toxicity in generated texts.
        In production, use Perspective API or similar.
        """
        # Simplified toxicity check - look for problematic words
        toxic_words = {'hate', 'kill', 'stupid', 'idiot', 'damn'}  # Simplified list
        
        toxicity_scores = []
        for text in texts:
            words = set(text.lower().split())
            toxic_count = len(words.intersection(toxic_words))
            toxicity_score = toxic_count / max(len(words), 1)
            toxicity_scores.append(toxicity_score)
        
        # Return inverse (1 - toxicity) so higher is better
        return 1.0 - np.mean(toxicity_scores)
    
    def comprehensive_evaluate(
        self,
        test_data: List[Dict[str, str]],
        generate_kwargs: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Run comprehensive evaluation on test data.
        
        Args:
            test_data: List of dicts with 'instruction', 'input', 'output' keys
            generate_kwargs: Generation parameters
            
        Returns:
            EvaluationResult object
        """
        if generate_kwargs is None:
            generate_kwargs = {
                "max_new_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            }
        
        logger.info("Starting comprehensive evaluation...")
        
        # Generate predictions
        predictions = []
        references = []
        source_texts = []
        
        self.model.eval()
        for item in test_data:
            # Format input
            if item.get("input"):
                prompt = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{item['instruction']}\n\n### Response:\n"
            
            # Generate
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generate_kwargs,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated[len(prompt):].strip()
            
            predictions.append(response)
            references.append(item['output'])
            source_texts.append(item.get('input', item['instruction']))
        
        # Calculate metrics
        results = EvaluationResult(
            perplexity=self.calculate_perplexity([ref[:512] for ref in references]),
            bleu_score=self.calculate_bleu(predictions, references),
            rouge_scores=self.calculate_rouge(predictions, references),
            bert_score=self.calculate_bert_score(predictions, references),
            factuality_score=self.evaluate_factuality(predictions, source_texts),
            coherence_score=self.evaluate_coherence(predictions),
            toxicity_score=self.evaluate_toxicity(predictions)
        )
        
        logger.info(f"Evaluation completed: {results.to_dict()}")
        return results
