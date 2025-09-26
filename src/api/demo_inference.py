"""Demo inference service for cloud deployment."""
import time
import random
from typing import Tuple
from .models import InferenceRequest

class DemoInferenceService:
    """Demo service that simulates model inference."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = "demo-model"
        self.device = "cpu"
        
    def load_model(self):
        """Simulate model loading."""
        time.sleep(1)
        print("Demo model loaded")
    
    def generate(self, request: InferenceRequest) -> Tuple[str, int, float]:
        """Generate demo responses."""
        start_time = time.time()
        
        # Simulate processing
        time.sleep(random.uniform(0.5, 1.5))
        
        # Generate contextual responses
        instruction_lower = request.instruction.lower()
        
        if "translate" in instruction_lower and "french" in instruction_lower:
            response = "Bonjour, ceci est une réponse générée."
        elif "summarize" in instruction_lower:
            response = "This text presents key insights about the topic discussed."
        elif any(word in instruction_lower for word in ["code", "function", "python"]):
            response = """def generated_function(param):
    # AI-generated implementation
    result = process_data(param)
    return result"""
        elif "explain" in instruction_lower:
            response = "This concept involves understanding the fundamental principles and their applications in real-world scenarios."
        else:
            responses = [
                "Based on my analysis, here's the generated response.",
                "The model has processed your request and generated this output.",
                "Here's what the AI model generated for your instruction."
            ]
            response = random.choice(responses)
        
        # Add context if provided
        if request.input_text:
            response = f"{response}\n\nContext considered: {request.input_text[:50]}..."
        
        tokens = len(response.split())
        generation_time = time.time() - start_time
        
        return response, tokens, generation_time
    
    def get_model_info(self):
        """Get demo model info."""
        return {
            "model_name": "demo-llm-finetuned",
            "model_type": "demo",
            "parameters": {"total": 1000000, "trainable": 150000},
            "device": "cpu",
            "quantization": None
        }
