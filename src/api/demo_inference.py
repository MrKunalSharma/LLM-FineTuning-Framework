"""Demo inference service for cloud deployment."""
import time
import random
from typing import Tuple
from pathlib import Path

class DemoInferenceService:
    """Demo service that simulates model inference."""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = "demo-model"
        self.device = "cpu"
        self.tokenizer = None
    
    def load_model(self):
        """Simulate model loading."""
        time.sleep(1)
        print("Demo model loaded")
    
    def create_prompt(self, instruction: str, input_text: str = None) -> str:
        """Create a formatted prompt."""
        if input_text:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        return f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    def generate(self, request) -> Tuple[str, int, float]:
        """Generate demo responses."""
        start_time = time.time()
        
        # Simulate processing time
        time.sleep(random.uniform(0.5, 1.5))
        
        instruction_lower = request.instruction.lower()
        
        # Generate contextual responses
        if "translate" in instruction_lower and "french" in instruction_lower:
            response = "Bonjour, ceci est une réponse générée par l'IA."
        elif "summarize" in instruction_lower:
            response = "This text presents key insights about the main topic, highlighting the most important aspects in a concise manner."
        elif any(word in instruction_lower for word in ["code", "function", "python"]):
            if "factorial" in instruction_lower:
                response = """def factorial(n):
    '''Calculate factorial recursively'''
    if n <= 1:
        return 1
    return n * factorial(n - 1)"""
            else:
                response = """def ai_generated_function(param):
    # AI-generated implementation
    result = process_data(param)
    return result"""
        elif "explain" in instruction_lower:
            response = "Based on the analysis, this concept involves understanding the fundamental principles and their practical applications in real-world scenarios."
        else:
            responses = [
                "Based on the input provided, here's the AI-generated response addressing your request.",
                "The model has analyzed your instruction and generated this contextually relevant output.",
                "Here's the synthesized response based on the given parameters and instruction."
            ]
            response = random.choice(responses)
        
        # Add context consideration
        if request.input_text:
            response = f"{response}\n\nContext considered: '{request.input_text[:100]}...'"
        
        tokens = len(response.split())
        generation_time = time.time() - start_time
        
        return response, tokens, generation_time
    
    def get_model_info(self):
        """Get demo model info."""
        return {
            "model_name": "llm-finetuned-demo",
            "model_type": "demo",
            "parameters": {"total": 2700000, "trainable": 4200},
            "device": "cpu",
            "quantization": None
        }
