"""Demo API simulation."""
import json
from datetime import datetime
import time

def simulate_api():
    """Simulate API responses."""
    print("\n🌐 API Simulation Demo")
    print("="*50)
    
    # Simulate server startup
    print("Starting FastAPI server...")
    time.sleep(1)
    print("✓ Server started on http://localhost:8000")
    
    # Show available endpoints
    print("\n📋 Available Endpoints:")
    endpoints = [
        ("GET", "/", "Root endpoint"),
        ("GET", "/health", "Health check"),
        ("GET", "/docs", "Interactive API documentation"),
        ("POST", "/generate", "Generate text (requires auth)"),
        ("POST", "/batch", "Batch inference (requires auth)"),
        ("GET", "/model/info", "Model information (requires auth)")
    ]
    
    for method, path, desc in endpoints:
        print(f"  {method:6} {path:15} - {desc}")
    
    # Simulate requests
    print("\n🧪 Simulating API Requests:")
    
    # Health check
    print("\n1. Health Check:")
    health_response = {
        "status": "healthy",
        "model_loaded": True,
        "model_name": "phi-2-finetuned",
        "device": "cuda" if False else "cpu",
        "timestamp": datetime.utcnow().isoformat()
    }
    print(f"   Response: {json.dumps(health_response, indent=2)}")
    
    # Generate request
    print("\n2. Text Generation:")
    gen_request = {
        "instruction": "Translate to Spanish",
        "input_text": "Hello, how are you?",
        "temperature": 0.7
    }
    print(f"   Request: {json.dumps(gen_request, indent=2)}")
    
    time.sleep(1)
    
    gen_response = {
        "response": "Hola, ¿cómo estás?",
        "instruction": gen_request["instruction"],
        "input_text": gen_request["input_text"],
        "generation_time": 0.523,
        "tokens_generated": 8,
        "model_name": "phi-2-finetuned",
        "timestamp": datetime.utcnow().isoformat()
    }
    print(f"   Response: {json.dumps(gen_response, indent=2)}")
    
    print("\n✅ API simulation complete!")
    print("\nTo run the actual API:")
    print("  python scripts/run_api.py --model_path models/finetuned/final")

if __name__ == "__main__":
    simulate_api()
