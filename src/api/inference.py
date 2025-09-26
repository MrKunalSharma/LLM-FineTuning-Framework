"""Inference service for the API."""
import time
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import json

# Import utilities first
from ..utils.logger import logger
from ..utils.config import ConfigManager
from .models import InferenceRequest, InferenceResponse

# Try to import ML libraries, fall back to None if not available
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from peft import PeftModel
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    GenerationConfig = None
    PeftModel = None

class InferenceService:
    """Handles model loading and inference."""
    
    def __init__(self, model_path: str, config_path: str = "configs/config.yaml"):
        self.model_path = Path(model_path)
        self.config_manager = ConfigManager(config_path)
        self.model_config = self.config_manager.get_model_config()
        
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        self.model_info = {}
        
    def _get_device(self) -> str:
        """Determine the device to use."""
        if ML_AVAILABLE and torch and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def load_model(self):
        """Load the fine-tuned model."""
        if not ML_AVAILABLE:
            logger.warning("ML libraries not available, using demo mode")
            self.model = "demo"  # Set to non-None to indicate loaded
            return
            
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Check if it's a PEFT model
            peft_config_path = self.model_path / "adapter_config.json"
            
            if peft_config_path.exists():
                # Load PEFT model
                with open(peft_config_path, 'r') as f:
                    peft_config = json.load(f)
                
                base_model_name = peft_config.get("base_model_name_or_path")
                
                # Load base model
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
                
                # Load PEFT adapters
                self.model = PeftModel.from_pretrained(base_model, self.model_path)
                self.model_info = {
                    "type": "peft",
                    "base_model": base_model_name,
                    "lora_config": peft_config
                }
            else:
                # Load regular model
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
                self.model_info = {"type": "full"}
            
            self.model.eval()
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.warning("Falling back to demo mode")
            self.model = "demo"
    
    def create_prompt(self, instruction: str, input_text: Optional[str] = None) -> str:
        """Create a formatted prompt."""
        if input_text:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        return f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    def generate(
        self,
        request: InferenceRequest
    ) -> Tuple[str, int, float]:
        """
        Generate response for a single request.
        
        Returns:
            Tuple of (response, tokens_generated, generation_time)
        """
        if not ML_AVAILABLE or self.model == "demo" or self.model is None:
            # Fallback to demo response
            return self._generate_demo_response(request)
        
        # Create prompt
        prompt = self.create_prompt(request.instruction, request.input_text)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.model_config.max_length
        ).to(self.device)
        
        # Generation config
        generation_config = GenerationConfig(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            do_sample=request.do_sample,
            repetition_penalty=request.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config
            )
        generation_time = time.time() - start_time
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        
        # Count tokens
        tokens_generated = len(outputs[0]) - len(inputs['input_ids'][0])
        
        return response, tokens_generated, generation_time
    
    def _generate_demo_response(self, request: InferenceRequest) -> Tuple[str, int, float]:
        """Generate demo response when ML libs not available."""
        import random
        
        start_time = time.time()
        time.sleep(random.uniform(0.5, 1.5))  # Simulate processing
        
        instruction_lower = request.instruction.lower()
        
        # Generate contextual demo responses
        if "translate" in instruction_lower:
            if "french" in instruction_lower:
                response = "Bonjour! Ceci est une réponse de démonstration générée par l'API."
            elif "spanish" in instruction_lower:
                response = "¡Hola! Esta es una respuesta de demostración generada por la API."
            else:
                response = "This is a translated response (demo mode)."
        elif "summarize" in instruction_lower:
            response = "Summary: The text discusses key concepts and presents main ideas in a concise format."
        elif any(word in instruction_lower for word in ["code", "python", "function"]):
            response = """def demo_function(input_data):
    '''AI-generated function demonstration'''
    processed = process_input(input_data)
    return f"Processed: {processed}\""""
        elif "explain" in instruction_lower:
            response = "Explanation: This concept involves understanding fundamental principles and their practical applications in real-world scenarios."
        else:
            responses = [
                "Based on your instruction, here's the AI-generated response.",
                "The model has processed your request and generated this output.",
                "Here's the synthesized response for your query."
            ]
            response = random.choice(responses)
            
        if request.input_text:
            response += f"\n\nContext considered: '{request.input_text[:100]}...'"
            
        generation_time = time.time() - start_time
        tokens = len(response.split())
        
        return response, tokens, generation_time
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if not ML_AVAILABLE or self.model == "demo":
            return {
                "model_name": "llm-finetuned-demo",
                "model_type": "demo",
                "parameters": {"total": 2700000, "trainable": 4200},
                "device": "cpu",
                "quantization": None,
                "status": "demo_mode"
            }
            
        if self.model is None:
            return {"status": "not_loaded"}
        
        param_count = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_name": str(self.model_path),
            "model_type": self.model_info.get("type", "unknown"),
            "parameters": {
                "total": param_count,
                "trainable": trainable_params
            },
            "lora_config": self.model_info.get("lora_config"),
            "device": self.device,
            "quantization": "fp16" if self.device == "cuda" else None
        }
