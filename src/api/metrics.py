"""Prometheus metrics for monitoring."""
from prometheus_client import Counter, Histogram, Gauge, Info, Summary, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client.core import CollectorRegistry
from fastapi import Response
import time
from functools import wraps
from typing import Callable
import psutil
import torch

# Create custom registry
registry = CollectorRegistry()

# Define metrics
request_count = Counter(
    'llm_api_requests_total',
    'Total API requests',
    ['endpoint', 'method', 'status'],
    registry=registry
)

request_duration = Histogram(
    'llm_api_request_duration_seconds',
    'Request duration in seconds',
    ['endpoint', 'method'],
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0),
    registry=registry
)

model_inference_duration = Histogram(
    'llm_inference_duration_seconds',
    'Model inference duration',
    ['model_name', 'model_version'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
    registry=registry
)

active_requests = Gauge(
    'llm_api_active_requests',
    'Number of active requests',
    ['endpoint'],
    registry=registry
)

model_info = Info(
    'llm_model',
    'Information about loaded model',
    registry=registry
)

tokens_generated = Counter(
    'llm_tokens_generated_total',
    'Total tokens generated',
    ['model_name'],
    registry=registry
)

cache_metrics = Counter(
    'llm_cache_requests_total',
    'Cache hit/miss statistics',
    ['operation', 'status'],
    registry=registry
)

# System metrics
gpu_memory_usage = Gauge(
    'llm_gpu_memory_usage_bytes',
    'GPU memory usage in bytes',
    ['device_id'],
    registry=registry
)

gpu_utilization = Gauge(
    'llm_gpu_utilization_percent',
    'GPU utilization percentage',
    ['device_id'],
    registry=registry
)

cpu_usage = Gauge(
    'llm_cpu_usage_percent',
    'CPU usage percentage',
    registry=registry
)

memory_usage = Gauge(
    'llm_memory_usage_bytes',
    'Memory usage in bytes',
    registry=registry
)

# Model-specific metrics
model_load_time = Histogram(
    'llm_model_load_duration_seconds',
    'Time taken to load model',
    ['model_name'],
    registry=registry
)

batch_size = Histogram(
    'llm_batch_size',
    'Batch size distribution',
    ['endpoint'],
    buckets=(1, 2, 4, 8, 16, 32, 64, 128),
    registry=registry
)

response_token_count = Histogram(
    'llm_response_token_count',
    'Number of tokens in responses',
    ['model_name'],
    buckets=(10, 50, 100, 200, 500, 1000, 2000),
    registry=registry
)

# Error metrics
error_count = Counter(
    'llm_errors_total',
    'Total errors by type',
    ['error_type', 'endpoint'],
    registry=registry
)

# Business metrics
requests_by_user = Counter(
    'llm_requests_by_user_total',
    'Requests per user/API key',
    ['user_id'],
    registry=registry
)

def update_system_metrics():
    """Update system resource metrics."""
    # CPU and Memory
    cpu_usage.set(psutil.cpu_percent(interval=1))
    memory = psutil.virtual_memory()
    memory_usage.set(memory.used)
    
    # GPU metrics (if available)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_memory_usage.labels(device_id=str(i)).set(
                torch.cuda.memory_allocated(i)
            )
            # Note: GPU utilization requires nvidia-ml-py
            try:
                import nvidia_ml_py as nvml
                nvml.nvmlInit()
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization.labels(device_id=str(i)).set(util.gpu)
            except:
                pass

def track_metrics(endpoint: str, method: str = "POST"):
    """Decorator to track API metrics."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            active_requests.labels(endpoint=endpoint).inc()
            
            try:
                result = await func(*args, **kwargs)
                request_count.labels(
                    endpoint=endpoint, 
                    method=method, 
                    status='success'
                ).inc()
                return result
            except Exception as e:
                request_count.labels(
                    endpoint=endpoint, 
                    method=method, 
                    status='error'
                ).inc()
                error_count.labels(
                    error_type=type(e).__name__,
                    endpoint=endpoint
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                request_duration.labels(
                    endpoint=endpoint,
                    method=method
                ).observe(duration)
                active_requests.labels(endpoint=endpoint).dec()
                
                # Update system metrics periodically
                update_system_metrics()
        
        return wrapper
    return decorator

def track_model_inference(model_name: str, model_version: str = "latest"):
    """Track model inference metrics."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Track tokens if available
                if hasattr(result, 'tokens_generated'):
                    tokens_generated.labels(model_name=model_name).inc(
                        result.tokens_generated
                    )
                    response_token_count.labels(model_name=model_name).observe(
                        result.tokens_generated
                    )
                
                return result
            finally:
                duration = time.time() - start_time
                model_inference_duration.labels(
                    model_name=model_name,
                    model_version=model_version
                ).observe(duration)
        
        return wrapper
    return decorator

def track_cache_operation(operation: str):
    """Track cache operations."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                status = 'hit' if result is not None else 'miss'
                cache_metrics.labels(
                    operation=operation,
                    status=status
                ).inc()
                return result
            except Exception as e:
                cache_metrics.labels(
                    operation=operation,
                    status='error'
                ).inc()
                raise
        return wrapper
    return decorator

# Metrics endpoint handler
async def metrics_endpoint():
    """Prometheus metrics endpoint."""
    return Response(
        generate_latest(registry), 
        media_type=CONTENT_TYPE_LATEST
    )

# Initialize model info
def set_model_info(name: str, version: str, parameters: dict):
    """Set model information for metrics."""
    model_info.info({
        'name': name,
        'version': version,
        'parameters': str(parameters.get('total', 'unknown')),
        'quantization': parameters.get('quantization', 'none'),
        'device': parameters.get('device', 'cpu')
    })
