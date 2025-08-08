#!/usr/bin/env python3
"""
Mock Server for Creativity Environment Testing

Provides a FastAPI-based mock server that simulates OpenAI-compatible API
for testing creativity environment without actual model deployment.
"""

import json
import time
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import asyncio
import uuid

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    BaseModel = object
    Field = lambda **kwargs: None

from mock_inference import MockCreativityInference

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Message role (system, user, assistant)")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    """Chat completion request model."""
    model: str = Field(default="mock-creativity-model", description="Model name")
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    max_tokens: Optional[int] = Field(default=300, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=0.8, description="Generation temperature")
    top_p: Optional[float] = Field(default=0.9, description="Top-p sampling")
    stream: Optional[bool] = Field(default=False, description="Stream response")
    creativity_mode: Optional[str] = Field(default="high", description="Creativity level")


class ChatCompletionChoice(BaseModel):
    """Chat completion choice model."""
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    """Chat completion response model."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int]
    creativity_metadata: Optional[Dict[str, Any]] = None


class MockCreativityServer:
    """
    Mock server for creativity inference testing.
    
    Provides OpenAI-compatible API endpoints for testing creativity
    environment integration.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize mock server.
        
        Args:
            config_path: Path to configuration file
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI and uvicorn are required for mock server")
        
        self.mock_inference = MockCreativityInference(config_path)
        self.app = FastAPI(
            title="Mock Creativity Inference API",
            description="Mock API for creativity environment testing",
            version="1.0.0"
        )
        
        self.request_count = 0
        self.active_streams = {}
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up FastAPI routes."""
        
        @self.app.get("/")
        async def root():
            return {
                "message": "Mock Creativity Inference API", 
                "version": "1.0.0",
                "status": "running"
            }
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "requests_processed": self.request_count,
                "mock_system_stats": self.mock_inference.get_statistics()
            }
        
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            """Handle chat completion requests."""
            self.request_count += 1
            
            try:
                # Extract user message for creativity generation
                user_messages = [msg for msg in request.messages if msg.role == "user"]
                if not user_messages:
                    raise HTTPException(status_code=400, detail="No user messages found")
                
                prompt = user_messages[-1].content
                
                # Generate mock response
                mock_response = self.mock_inference.generate_response(
                    prompt=prompt,
                    max_tokens=request.max_tokens or 300,
                    temperature=request.temperature or 0.8
                )
                
                # Create response
                response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                created_time = int(time.time())
                
                choice = ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=mock_response.content),
                    finish_reason="stop"
                )
                
                # Calculate token usage (rough estimate)
                prompt_tokens = len(prompt.split()) * 1.3  # Rough conversion
                completion_tokens = len(mock_response.content.split()) * 1.3
                total_tokens = int(prompt_tokens + completion_tokens)
                
                usage = {
                    "prompt_tokens": int(prompt_tokens),
                    "completion_tokens": int(completion_tokens),
                    "total_tokens": total_tokens
                }
                
                response = ChatCompletionResponse(
                    id=response_id,
                    created=created_time,
                    model=request.model,
                    choices=[choice],
                    usage=usage,
                    creativity_metadata={
                        "category": mock_response.metadata["category"],
                        "creativity_score": mock_response.metadata["total_creativity_score"],
                        "simulated_metrics": mock_response.simulated_metrics,
                        "generation_info": mock_response.generation_info
                    }
                )
                
                if request.stream:
                    return StreamingResponse(
                        self._stream_response(response),
                        media_type="text/plain"
                    )
                else:
                    return response
            
            except Exception as e:
                logger.error(f"Error processing chat completion: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/v1/completions")
        async def completions(request: Request):
            """Handle legacy completion requests."""
            try:
                request_data = await request.json()
                
                prompt = request_data.get("prompt", "")
                max_tokens = request_data.get("max_tokens", 300)
                temperature = request_data.get("temperature", 0.8)
                
                # Generate mock response
                mock_response = self.mock_inference.generate_response(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                # Create legacy completion response
                response = {
                    "id": f"cmpl-{uuid.uuid4().hex[:8]}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": request_data.get("model", "mock-creativity-model"),
                    "choices": [{
                        "text": mock_response.content,
                        "index": 0,
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": len(prompt.split()) * 1.3,
                        "completion_tokens": len(mock_response.content.split()) * 1.3,
                        "total_tokens": (len(prompt.split()) + len(mock_response.content.split())) * 1.3
                    },
                    "creativity_metadata": {
                        "category": mock_response.metadata["category"],
                        "creativity_score": mock_response.metadata["total_creativity_score"],
                        "simulated_metrics": mock_response.simulated_metrics
                    }
                }
                
                return JSONResponse(content=response)
            
            except Exception as e:
                logger.error(f"Error processing completion: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/v1/models")
        async def list_models():
            """List available models."""
            return {
                "object": "list",
                "data": [
                    {
                        "id": "mock-creativity-model",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "mock-org",
                        "permission": [],
                        "root": "mock-creativity-model",
                        "parent": None,
                        "description": "Mock model for creativity testing"
                    }
                ]
            }
        
        @self.app.get("/stats")
        async def get_statistics():
            """Get server and inference statistics."""
            return {
                "server": {
                    "requests_processed": self.request_count,
                    "active_streams": len(self.active_streams)
                },
                "mock_inference": self.mock_inference.get_statistics()
            }
        
        @self.app.post("/v1/creativity/evaluate")
        async def evaluate_creativity(request: Request):
            """Custom endpoint for creativity evaluation."""
            try:
                request_data = await request.json()
                text = request_data.get("text", "")
                
                if not text:
                    raise HTTPException(status_code=400, detail="Text is required")
                
                # Use actual reward function for evaluation
                from reward import reward_function
                
                # Calculate actual creativity score
                actual_score = reward_function(text)
                
                # Also get simulated metrics for comparison
                mock_response = self.mock_inference.generate_response(text, max_tokens=50)
                simulated_metrics = mock_response.simulated_metrics
                
                return {
                    "text": text,
                    "actual_creativity_score": actual_score,
                    "simulated_metrics": simulated_metrics,
                    "evaluation_timestamp": time.time()
                }
            
            except Exception as e:
                logger.error(f"Error evaluating creativity: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _stream_response(self, response: ChatCompletionResponse):
        """Stream response for chat completions."""
        # Convert to streaming format
        chunks = []
        
        # Split content into chunks
        content = response.choices[0].message.content
        words = content.split()
        
        for i, word in enumerate(words):
            chunk_data = {
                "id": response.id,
                "object": "chat.completion.chunk",
                "created": response.created,
                "model": response.model,
                "choices": [{
                    "delta": {"content": word + " "} if i < len(words) - 1 else {"content": word},
                    "index": 0,
                    "finish_reason": None if i < len(words) - 1 else "stop"
                }]
            }
            
            chunk_line = f"data: {json.dumps(chunk_data)}\n\n"
            yield chunk_line
            
            # Small delay to simulate streaming
            await asyncio.sleep(0.05)
        
        # Final chunk
        final_chunk = {
            "id": response.id,
            "object": "chat.completion.chunk", 
            "created": response.created,
            "model": response.model,
            "choices": [{
                "delta": {},
                "index": 0,
                "finish_reason": "stop"
            }]
        }
        
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    
    def run(self, host: str = "localhost", port: int = 8888, **kwargs):
        """
        Run the mock server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            **kwargs: Additional uvicorn arguments
        """
        logger.info(f"Starting Mock Creativity Server on {host}:{port}")
        
        uvicorn_config = {
            "host": host,
            "port": port,
            "log_level": "info",
            **kwargs
        }
        
        uvicorn.run(self.app, **uvicorn_config)


def create_test_client():
    """Create test client for the mock server."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required for test client")
    
    from fastapi.testclient import TestClient
    
    server = MockCreativityServer()
    return TestClient(server.app)


def main():
    """Run the mock server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mock Creativity Inference Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8888, help="Port to bind to")
    parser.add_argument("--config", type=Path, help="Path to configuration file")
    
    args = parser.parse_args()
    
    if not FASTAPI_AVAILABLE:
        print("Error: FastAPI and uvicorn are required to run the mock server")
        print("Install with: pip install fastapi uvicorn")
        return 1
    
    try:
        server = MockCreativityServer(config_path=args.config)
        server.run(host=args.host, port=args.port)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())