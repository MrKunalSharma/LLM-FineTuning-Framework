"""Training pipeline for LLM fine-tuning."""
import os
from pathlib import Path
from typing import Optional, Dict, Any
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    TrainerCallback
)
from datasets import DatasetDict
import wandb
from ..utils.logger import logger
from ..utils.config import TrainingConfig, ConfigManager
from .model_manager import ModelManager

class CustomCallback(TrainerCallback):
    """Custom callback for additional logging."""
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Log epoch metrics."""
        if state.is_local_process_zero:
            logger.info(f"Epoch {state.epoch} completed")
            if wandb.run is not None:
                wandb.log({
                    "epoch": state.epoch,
                    "learning_rate": state.log_history[-1].get("learning_rate", 0),
                    "loss": state.log_history[-1].get("loss", 0)
                })

class LLMTrainer:
    """Manages the training pipeline."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_manager = ConfigManager(config_path)
        self.model_config = self.config_manager.get_model_config()
        self.training_config = self.config_manager.get_training_config()
        self.lora_config = self.config_manager.get_lora_config()
        self.wandb_config = self.config_manager.get_wandb_config()
        
        # Initialize model manager
        self.model_manager = ModelManager(self.model_config, self.lora_config)
        
    def setup_wandb(self, run_name: Optional[str] = None):
        """Initialize Weights & Biases tracking."""
        if self.wandb_config.get('api_key'):
            wandb.login(key=self.wandb_config['api_key'])
            wandb.init(
                project=self.wandb_config['project'],
                entity=self.wandb_config.get('entity'),
                name=run_name,
                tags=self.wandb_config.get('tags', []),
                config={
                    "model": self.model_config.__dict__,
                    "training": self.training_config.__dict__,
                    "lora": self.lora_config.__dict__
                }
            )
            logger.info("Weights & Biases initialized")
    
    def train(
        self,
        dataset: DatasetDict,
        run_name: Optional[str] = None,
        use_wandb: bool = True,
        resume_from_checkpoint: Optional[str] = None
    ):
        """
        Train the model.
        
        Args:
            dataset: Dataset dictionary with train/test splits
            run_name: Name for the training run
            use_wandb: Enable Weights & Biases tracking
            resume_from_checkpoint: Path to resume training
        """
        # Setup wandb
        if use_wandb:
            self.setup_wandb(run_name)
        
        # Load model and tokenizer
        model, tokenizer = self.model_manager.load_model_and_tokenizer()
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.training_config.output_dir,
            num_train_epochs=self.training_config.num_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            warmup_steps=self.training_config.warmup_steps,
            weight_decay=self.training_config.weight_decay,
            logging_dir=self.training_config.logging_dir,
            logging_steps=10,
            save_steps=self.training_config.save_steps,
            eval_steps=self.training_config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb" if use_wandb else "none",
            run_name=run_name,
            learning_rate=self.training_config.learning_rate,
            fp16=self.model_config.device == "cuda",
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            remove_unused_columns=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[
                CustomCallback(),
                EarlyStoppingCallback(early_stopping_patience=3)
            ],
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save final model
        output_dir = Path(self.training_config.output_dir) / "final"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model_manager.save_model(model, tokenizer, str(output_dir))
        
        # Finish wandb run
        if use_wandb and wandb.run is not None:
            wandb.finish()
        
        logger.info("Training completed!")
        return trainer
