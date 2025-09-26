"""Model management for fine-tuning."""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training
)
from typing import Tuple, Optional
from ..utils.logger import logger
from ..utils.config import ModelConfig, LoRAConfig

class ModelManager:
    """Manages model loading and configuration."""
    
    def __init__(self, model_config: ModelConfig, lora_config: LoRAConfig):
        self.model_config = model_config
        self.lora_config = lora_config
        self.device = self._get_device()
        
    def _get_device(self) -> str:
        """Determine the device to use."""
        if torch.cuda.is_available() and self.model_config.device == "cuda":
            return "cuda"
        return "cpu"
    
    def load_model_and_tokenizer(
        self,
        use_8bit: bool = False,
        use_4bit: bool = True
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load model and tokenizer with quantization options.
        
        Args:
            use_8bit: Use 8-bit quantization
            use_4bit: Use 4-bit quantization (QLoRA)
            
        Returns:
            Model and tokenizer tuple
        """
        logger.info(f"Loading model: {self.model_config.name}")
        
        # Configure quantization
        bnb_config = None
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif use_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
            )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.name,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Add padding token if not exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_config.name,
            quantization_config=bnb_config,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        )
        
        # Prepare model for k-bit training
        if use_4bit or use_8bit:
            model = prepare_model_for_kbit_training(model)
        
        # Apply LoRA
        model = self._apply_lora(model)
        
        logger.info("Model and tokenizer loaded successfully")
        return model, tokenizer
    
    def _apply_lora(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply LoRA configuration to the model."""
        logger.info("Applying LoRA configuration")
        
        lora_config = LoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            target_modules=self.lora_config.target_modules,
            lora_dropout=self.lora_config.lora_dropout,
            bias=self.lora_config.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
    
    def save_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        output_dir: str
    ):
        """Save the fine-tuned model and tokenizer."""
        logger.info(f"Saving model to {output_dir}")
        
        # Save model
        model.save_pretrained(output_dir)
        
        # Save tokenizer
        tokenizer.save_pretrained(output_dir)
        
        logger.info("Model saved successfully")
