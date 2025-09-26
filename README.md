# LLM Fine-Tuning & Evaluation Framework

A production-ready framework for fine-tuning Large Language Models (LLMs) using LoRA/QLoRA, with comprehensive evaluation metrics and a scalable inference API.

## 🚀 Features

- **Efficient Fine-tuning**: Support for LoRA and QLoRA techniques
- **Comprehensive Evaluation**: Multiple metrics including BLEU, ROUGE, BERTScore, and custom metrics
- **Production API**: FastAPI-based inference service with authentication
- **Experiment Tracking**: Integration with Weights & Biases
- **Docker Support**: Containerized deployment
- **Modular Architecture**: Clean, extensible codebase

## 📋 Prerequisites

- Python 3.11
- CUDA-capable GPU (optional, but recommended)
- Docker (for containerized deployment)
## 🎨 Streamlit Interface

Access the web interface at http://localhost:8501

Features:
- Interactive chat interface
- Real-time analytics dashboard
- Batch testing capabilities
- Example use cases

## 🚀 Quick Start with Docker

```bash
docker-compose up -d


## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd LLM-FineTuning-Framework


                
Create virtual environment

          

bash


python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate


                
Install dependencies

          

bash


pip install -r requirements.txt


                
Set up environment variables

          

bash


cp .env.example .env
# Edit .env with your API keys


                
📊 Project Structure



LLM-FineTuning-Framework/
├── src/
│   ├── training/       # Training modules
│   ├── evaluation/     # Evaluation metrics
│   ├── api/           # FastAPI inference service
│   └── utils/         # Utility functions
├── configs/           # Configuration files
├── data/              # Dataset storage
├── models/            # Model checkpoints
├── scripts/           # Executable scripts
├── tests/             # Unit tests
├── docker/            # Docker configurations
└── notebooks/         # Jupyter notebooks


          
🏃 Quick Start
1. Prepare Your Data
Create a JSONL file with your training data:


          

json


{"instruction": "Translate to French", "input": "Hello", "output": "Bonjour"}
{"instruction": "Summarize", "input": "Long text...", "output": "Summary..."}


                
2. Fine-tune a Model

          

bash


python scripts/train.py --data_path data/train.jsonl --run_name my_experiment


                
3. Evaluate the Model

          

bash


python scripts/evaluate.py --model_path models/finetuned/final --test_data data/test.jsonl


                
4. Start the Inference API

          

bash


python scripts/run_api.py --model_path models/finetuned/final


                
🐳 Docker Deployment
Build and Run with Docker Compose

          

bash


docker-compose up -d


                
Build Docker Image Manually

          

bash


docker build -f docker/Dockerfile -t llm-finetuning-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models llm-finetuning-api


                
📡 API Usage
Authentication
All API endpoints require a Bearer token:


          

bash


curl -H "Authorization: Bearer your-api-key" http://localhost:8000/health


                
Generate Text

          

bash


curl -X POST http://localhost:8000/generate \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Translate to Spanish",
    "input_text": "Hello, how are you?",
    "temperature": 0.7
  }'


                
Batch Inference

          

bash


curl -X POST http://localhost:8000/batch \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"instruction": "Translate to French", "input_text": "Hello"},
      {"instruction": "Summarize", "input_text": "Long text..."}
    ]
  }'


                
📈 Configuration
Edit configs/config.yaml to customize:

Model selection
Training hyperparameters
LoRA configuration
API settings
🧪 Running Tests

          

bash


# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html


                
📊 Experiment Tracking
The framework integrates with Weights & Biases for experiment tracking:

Set your W&B API key in .env
Training runs will automatically log to W&B
View results at wandb.ai
🏗️ Architecture
Training Pipeline
Supports multiple model architectures (GPT, LLaMA, Phi, etc.)
Efficient parameter-efficient fine-tuning with LoRA
Automatic mixed precision training
Gradient accumulation for larger effective batch sizes
Evaluation Framework
Automatic Metrics: Perplexity, BLEU, ROUGE, BERTScore
Quality Metrics: Factuality, Coherence, Toxicity detection
Custom Metrics: Extensible evaluation system
Inference API
FastAPI-based REST API
JWT authentication
Request batching
Health checks and monitoring
Prometheus metrics export
🤝 Contributing
Fork the repository
Create a feature branch
Commit your changes
Push to the branch
Open a Pull Request
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Hugging Face Transformers
PEFT library for LoRA implementation
FastAPI framework
Weights & Biases for experiment tracking