"""FastAPI application for model inference."""
from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import time
from datetime import datetime
from typing import Optional, Dict
from .demo_inference import DemoInferenceService
import uvicorn

from .models import (
    InferenceRequest, InferenceResponse,
    BatchInferenceRequest, BatchInferenceResponse,
    HealthResponse, ModelInfoResponse, ErrorResponse
)
from .inference import InferenceService
from ..utils.logger import logger
from .metrics import (
    track_metrics, track_model_inference, metrics_endpoint,
    set_model_info, batch_size, requests_by_user, track_cache_operation
)

# Security
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API token."""
    token = credentials.credentials
    expected_token = os.getenv("API_KEY", "default-api-key")
    
    if token != expected_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return token

# Global inference service
inference_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_service
    
    logger.info("Starting API server...")
    model_path = os.getenv("MODEL_PATH", "models/finetuned/final")
    
    # Use demo mode if on cloud without GPU
    use_demo = os.getenv("USE_DEMO_MODE", "false").lower() == "true"
    
    try:
        if use_demo:
            inference_service = DemoInferenceService(model_path)
        else:
            inference_service = InferenceService(model_path)
        
        inference_service.load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        # Fallback to demo mode
        logger.warning(f"Failed to load real model: {e}. Using demo mode.")
        inference_service = DemoInferenceService(model_path)
        inference_service.load_model()
    
    yield
    logger.info("Shutting down API server...")

# Create FastAPI app
app = FastAPI(
    title="LLM Fine-Tuning Inference API",
    description="Production-ready API for fine-tuned LLM inference",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "LLM Fine-Tuning Inference API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if inference_service and inference_service.model else "unhealthy",
        model_loaded=inference_service is not None and inference_service.model is not None,
        model_name=inference_service.model_path.name if inference_service else None,
        device=inference_service.device if inference_service else "unknown",
        timestamp=datetime.utcnow()
    )

@app.get("/model/info", response_model=ModelInfoResponse, dependencies=[Depends(verify_token)])
async def model_info():
    """Get model information."""
    if not inference_service or not inference_service.model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    info = inference_service.get_model_info()
    return ModelInfoResponse(**info)

@app.get("/metrics", include_in_schema=False)
async def get_metrics():
    """Prometheus metrics endpoint."""
    return await metrics_endpoint()

# Cache stub (implement if needed)
async def check_cache(request: InferenceRequest):
    """Check cache for existing response."""
    # TODO: Implement cache check
    return None

@track_model_inference("phi-2-finetuned")
async def generate_with_metrics(request: InferenceRequest):
    """Generate with model metrics tracking."""
    return inference_service.generate(request)

@app.post("/generate", response_model=InferenceResponse, dependencies=[Depends(verify_token)])
@track_metrics("generate", "POST")
async def generate(
    request: InferenceRequest,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """Generate text based on instruction."""
    # Track user
    requests_by_user.labels(user_id=credentials.credentials[:8]).inc()
    
    try:
        if not inference_service or not inference_service.model:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        # Check cache first
        cached_result = await check_cache(request)
        if cached_result:
            return cached_result
        
        # Generate response
        response, tokens, time_taken = inference_service.generate(request)
        
        return InferenceResponse(
            response=response,
            instruction=request.instruction,
            input_text=request.input_text,
            generation_time=time_taken,
            tokens_generated=tokens,
            model_name=inference_service.model_path.name,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise

@app.post("/batch", response_model=BatchInferenceResponse, dependencies=[Depends(verify_token)])
@track_metrics("batch", "POST")
async def batch_generate(
    batch_request: BatchInferenceRequest,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """Generate text for multiple requests."""
    # Track batch size
    batch_size.labels(endpoint="batch").observe(len(batch_request.requests))
    requests_by_user.labels(user_id=credentials.credentials[:8]).inc()
    
    try:
        if not inference_service or not inference_service.model:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        responses = []
        total_start = time.time()
        
        for request in batch_request.requests:
            response, tokens, time_taken = inference_service.generate(request)
            responses.append(InferenceResponse(
                response=response,
                instruction=request.instruction,
                input_text=request.input_text,
                generation_time=time_taken,
                tokens_generated=tokens,
                model_name=inference_service.model_path.name,
                timestamp=datetime.utcnow()
            ))
        
        total_time = time.time() - total_start
        
        return BatchInferenceResponse(
            responses=responses,
            total_time=total_time,
            batch_size=len(responses)
        )
        
    except Exception as e:
        logger.error(f"Batch generation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return ErrorResponse(
        error=exc.detail,
        detail=str(exc),
        timestamp=datetime.utcnow()
    )

def start_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False
):
    """Start the API server."""
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=reload
    )
