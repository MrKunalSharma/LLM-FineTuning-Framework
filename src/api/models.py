"""Pydantic models for API requests and responses."""
from pydantic import BaseModel, Field, validator, ConfigDict  # Add ConfigDict to imports
from typing import Optional, List, Dict, Any
from datetime import datetime

class InferenceRequest(BaseModel):
    """Request model for inference endpoint."""
    instruction: str = Field(..., description="The instruction or prompt")
    input_text: Optional[str] = Field(None, description="Optional input context")
    max_new_tokens: Optional[int] = Field(256, ge=1, le=1024)
    temperature: Optional[float] = Field(0.7, ge=0.1, le=2.0)
    top_p: Optional[float] = Field(0.9, ge=0.1, le=1.0)
    top_k: Optional[int] = Field(50, ge=1, le=100)
    do_sample: Optional[bool] = Field(True)
    repetition_penalty: Optional[float] = Field(1.1, ge=1.0, le=2.0)
    
    @validator('instruction')
    def instruction_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Instruction cannot be empty')
        return v

class InferenceResponse(BaseModel):
    """Response model for inference endpoint."""
    response: str
    instruction: str
    input_text: Optional[str]
    generation_time: float
    tokens_generated: int
    model_name: str
    timestamp: datetime

class BatchInferenceRequest(BaseModel):
    """Request model for batch inference."""
    requests: List[InferenceRequest]
    
    @validator('requests')
    def validate_batch_size(cls, v):
        if len(v) > 100:  # Max batch size
            raise ValueError('Batch size cannot exceed 100')
        return v

class BatchInferenceResponse(BaseModel):
    """Response model for batch inference."""
    responses: List[InferenceResponse]
    total_time: float
    batch_size: int

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_name: Optional[str]
    device: str
    timestamp: datetime

class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_config = ConfigDict(protected_namespaces=())  # This prevents the warning
    
    model_name: str
    model_type: str
    parameters: Dict[str, Any]
    lora_config: Optional[Dict[str, Any]]
    device: str
    quantization: Optional[str]

class EvaluationRequest(BaseModel):
    """Request for model evaluation."""
    test_samples: List[Dict[str, str]]
    metrics: Optional[List[str]] = Field(
        ["bleu", "rouge", "bert_score"],
        description="Metrics to calculate"
    )

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str]
    timestamp: datetime
