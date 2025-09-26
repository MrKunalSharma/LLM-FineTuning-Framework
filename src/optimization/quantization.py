"""Advanced quantization techniques."""
import torch
from transformers import AutoModelForCausalLM
from typing import Optional
import torch.nn as nn

class ModelQuantizer:
    """Advanced quantization for deployment."""
    
    @staticmethod
    def quantize_to_int8(model_path: str, output_path: str):
        """Quantize model to INT8 for faster inference."""
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8
        )
        
        # Save quantized model
        torch.save(quantized_model.state_dict(), f"{output_path}/model_int8.pt")
        logger.info(f"Model quantized to INT8 and saved to {output_path}")
        
        # Calculate size reduction
        original_size = sum(p.numel() * p.element_size() for p in model.parameters())
        quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
        reduction = (1 - quantized_size / original_size) * 100
        
        logger.info(f"Size reduction: {reduction:.1f}%")
        return quantized_model
    
    @staticmethod
    def optimize_for_mobile(model, output_path: str):
        """Optimize model for mobile deployment."""
        # Convert to TorchScript
        example_input = torch.randint(0, 1000, (1, 128))
        traced_model = torch.jit.trace(model, example_input)
        
        # Optimize
        optimized = torch.jit.optimize_for_mobile(traced_model)
        
        # Save
        optimized.save(f"{output_path}/model_mobile.pt")
        logger.info("Model optimized for mobile deployment")
