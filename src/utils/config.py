"""Configuration management utilities."""
import yaml
import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ModelConfig:
    """Model configuration settings."""
    name: str
    max_length: int
    temperature: float
    top_p: float
    device: str

@dataclass
class TrainingConfig:
    """Training configuration settings."""
    output_dir: str
    num_epochs: int
    learning_rate: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    warmup_steps: int
    weight_decay: float
    logging_dir: str
    save_steps: int
    eval_steps: int

@dataclass
class LoRAConfig:
    """LoRA configuration settings."""
    r: int
    lora_alpha: int
    target_modules: list
    lora_dropout: float
    bias: str
    task_type: str

class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        return ModelConfig(**self.config['model'])
    
    def get_training_config(self) -> TrainingConfig:
        """Get training configuration."""
        return TrainingConfig(**self.config['training'])
    
    def get_lora_config(self) -> LoRAConfig:
        """Get LoRA configuration."""
        return LoRAConfig(**self.config['lora'])
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration."""
        return self.config['api']
    
    def get_wandb_config(self) -> Dict[str, Any]:
        """Get Weights & Biases configuration."""
        wandb_config = self.config['wandb'].copy()
        wandb_config['api_key'] = os.getenv('WANDB_API_KEY')
        return wandb_config
