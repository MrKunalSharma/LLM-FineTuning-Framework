"""Demo training script that simulates the training process."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import json
from datetime import datetime
from src.utils.logger import logger
from src.utils.config import ConfigManager

def simulate_training():
    """Simulate training process for demonstration."""
    logger.info("🚀 Starting training simulation...")
    
    # Load configuration
    config_manager = ConfigManager()
    training_config = config_manager.get_training_config()
    lora_config = config_manager.get_lora_config()
    
    # Simulate data loading
    logger.info("📊 Loading training data...")
    time.sleep(1)
    logger.info("✓ Loaded 1000 training samples")
    logger.info("✓ Loaded 200 validation samples")
    
    # Simulate model initialization
    logger.info("\n🤖 Initializing model with LoRA...")
    logger.info(f"  Base model: microsoft/phi-2")
    logger.info(f"  LoRA rank: {lora_config.r}")
    logger.info(f"  LoRA alpha: {lora_config.lora_alpha}")
    logger.info(f"  Target modules: {lora_config.target_modules}")
    time.sleep(1)
    logger.info("✓ Model initialized with 2.7B parameters")
    logger.info("✓ Trainable parameters: 4.2M (0.15% of total)")
    
    # Simulate training epochs
    logger.info(f"\n🏃 Starting training for {training_config.num_epochs} epochs...")
    
    metrics_history = []
    for epoch in range(training_config.num_epochs):
        logger.info(f"\n--- Epoch {epoch + 1}/{training_config.num_epochs} ---")
        
        # Simulate training steps
        train_loss = 2.5 - (0.3 * epoch) + (0.1 * epoch ** 2)
        eval_loss = 2.4 - (0.25 * epoch) + (0.08 * epoch ** 2)
        
        for step in range(0, 100, 20):
            time.sleep(0.5)
            current_loss = train_loss - (step / 1000)
            logger.info(f"Step {step}/100 - Loss: {current_loss:.4f}")
        
        # Evaluation
        logger.info(f"\n📊 Evaluating...")
        time.sleep(1)
        
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "perplexity": 2.71 ** eval_loss,
            "learning_rate": 2e-5 * (0.9 ** epoch)
        }
        metrics_history.append(metrics)
        
        logger.info(f"✓ Train Loss: {train_loss:.4f}")
        logger.info(f"✓ Eval Loss: {eval_loss:.4f}")
        logger.info(f"✓ Perplexity: {metrics['perplexity']:.2f}")
        
        # Save checkpoint
        logger.info(f"\n💾 Saving checkpoint...")
        checkpoint_dir = Path("models/checkpoints") / f"epoch_{epoch + 1}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(checkpoint_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"✓ Checkpoint saved to {checkpoint_dir}")
    
    # Final model save
    logger.info("\n💾 Saving final model...")
    final_dir = Path("models/finetuned/final")
    final_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mock model files
    model_info = {
        "model_name": "microsoft/phi-2-finetuned",
        "base_model": "microsoft/phi-2",
        "training_completed": datetime.now().isoformat(),
        "final_metrics": metrics_history[-1],
        "lora_config": {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "target_modules": lora_config.target_modules
        }
    }
    
    with open(final_dir / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    # Create adapter_config.json for PEFT
    adapter_config = {
        "base_model_name_or_path": "microsoft/phi-2",
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "target_modules": lora_config.target_modules
    }
    
    with open(final_dir / "adapter_config.json", "w") as f:
        json.dump(adapter_config, f, indent=2)
    
    logger.info(f"✓ Model saved to {final_dir}")
    
    # Training summary
    logger.info("\n" + "="*50)
    logger.info("🎉 Training completed successfully!")
    logger.info(f"  Final train loss: {metrics_history[-1]['train_loss']:.4f}")
    logger.info(f"  Final eval loss: {metrics_history[-1]['eval_loss']:.4f}")
    logger.info(f"  Final perplexity: {metrics_history[-1]['perplexity']:.2f}")
    logger.info("="*50)
    
    return metrics_history

if __name__ == "__main__":
    simulate_training()
