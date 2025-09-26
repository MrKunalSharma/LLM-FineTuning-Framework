"""Streaming response support for real-time generation."""
from typing import AsyncGenerator
import asyncio
from fastapi import StreamingResponse
from fastapi.responses import StreamingResponse
import json
from ..utils.logger import logger

class StreamingInference:
    """Handle streaming text generation."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    async def stream_generate(
        self, 
        prompt: str, 
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        """Stream tokens as they're generated."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Simulate streaming (in production, use TextIteratorStreamer)
        generated_text = ""
        for i in range(max_tokens):
            # In real implementation, generate token by token
            token = f"token_{i} "
            generated_text += token
            
            # Stream as SSE (Server-Sent Events)
            yield f"data: {json.dumps({'token': token, 'text': generated_text})}\n\n"
            await asyncio.sleep(0.05)  # Simulate generation time
        
        yield f"data: {json.dumps({'finished': True})}\n\n"

# Add to API
@app.post("/stream", dependencies=[Depends(verify_token)])
async def stream_generate(request: InferenceRequest):
    """Stream text generation in real-time."""
    streamer = StreamingInference(inference_service.model, inference_service.tokenizer)
    return StreamingResponse(
        streamer.stream_generate(
            request.instruction,
            request.max_new_tokens,
            request.temperature
        ),
        media_type="text/event-stream"
    )
