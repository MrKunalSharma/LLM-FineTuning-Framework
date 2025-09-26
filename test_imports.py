"""Test if all imports work correctly."""
print("Testing imports...")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"✗ PyTorch: {e}")

try:
    import transformers
    print(f"✓ Transformers {transformers.__version__}")
except ImportError as e:
    print(f"✗ Transformers: {e}")

try:
    import peft
    print(f"✓ PEFT {peft.__version__}")
except ImportError as e:
    print(f"✗ PEFT: {e}")

try:
    import fastapi
    print(f"✓ FastAPI {fastapi.__version__}")
except ImportError as e:
    print(f"✗ FastAPI: {e}")

try:
    import wandb
    print(f"✓ Weights & Biases {wandb.__version__}")
except ImportError as e:
    print(f"✗ Wandb: {e}")

try:
    import bert_score
    print(f"✓ BERTScore")
except ImportError as e:
    print(f"✗ BERTScore: {e}")

print("\n✅ All imports successful!" if all else "❌ Some imports failed")
