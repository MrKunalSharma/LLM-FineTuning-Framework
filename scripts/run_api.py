"""Script to start the inference API."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import os
from dotenv import load_dotenv
from src.api.app import start_server
from src.utils.logger import logger

def main():
    parser = argparse.ArgumentParser(description="Start LLM Inference API")
    parser.add_argument("--model_path", type=str, help="Path to fine-tuned model")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Set model path in environment if provided
    if args.model_path:
        os.environ["MODEL_PATH"] = args.model_path
    
    # Ensure model path is set
    if not os.environ.get("MODEL_PATH"):
        logger.error("MODEL_PATH not set. Use --model_path or set MODEL_PATH env variable")
        sys.exit(1)
    
    logger.info(f"Starting API server on {args.host}:{args.port}")
    logger.info(f"Model path: {os.environ['MODEL_PATH']}")
    
    try:
        # Start server
        import uvicorn
        uvicorn.run(
            "src.api.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise

if __name__ == "__main__":
    main()
