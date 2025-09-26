"""Download a model for testing."""
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def download_model():
    model_name = "microsoft/phi-2"
    print(f"📥 Downloading {model_name}...")
    print("This may take a few minutes on first run...\n")
    
    try:
        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir="./models/cache"
        )
        print("✓ Tokenizer downloaded")
        
        # Download model (smaller precision for testing)
        print("\nDownloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            cache_dir="./models/cache"
        )
        print("✓ Model downloaded")
        
        # Quick test
        print("\n🧪 Testing model...")
        inputs = tokenizer("Hello, I am", return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Test output: {response}")
        
        print("\n✅ Model ready for fine-tuning!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\nTip: If you're behind a firewall, you may need to:")
        print("1. Set proxy environment variables")
        print("2. Use a VPN")
        print("3. Download models manually")

if __name__ == "__main__":
    download_model()
