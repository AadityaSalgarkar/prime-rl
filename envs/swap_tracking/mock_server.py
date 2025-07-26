#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["fastapi", "uvicorn"]
# requires-python = ">=3.8"
# ///

"""
Mock inference server for swap tracking environment.
Provides OpenAI-compatible API for testing training pipelines without GPU.
"""

import json
import sys
import time
from typing import Dict, List, Optional
from pathlib import Path

# Add environment directory to path
env_dir = Path(__file__).parent
sys.path.insert(0, str(env_dir))

from mock_inference import MockInferenceClient

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("Missing dependencies. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn"])
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn


# Request/Response models
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.0
    stream: Optional[bool] = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


# FastAPI app
app = FastAPI(title="Mock Swap Tracking Inference Server")


# Global mock client
mock_client = None


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "running", "service": "mock-swap-tracking-inference"}


@app.get("/v1/models")
async def list_models():
    """List available mock models."""
    return {
        "object": "list",
        "data": [
            {
                "id": "mock-identity",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mock-provider"
            },
            {
                "id": "mock-simple",
                "object": "model", 
                "created": int(time.time()),
                "owned_by": "mock-provider"
            },
            {
                "id": "mock-swap-aware",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mock-provider"
            },
            {
                "id": "mock-random",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mock-provider"
            }
        ]
    }


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create chat completion using mock model."""
    global mock_client
    
    try:
        # Initialize mock client if needed
        if mock_client is None or mock_client.model_name != request.model:
            mock_client = MockInferenceClient(request.model, n_boxes=10)
        
        # Convert Pydantic messages to dict format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Get mock response
        mock_response = mock_client.create_completion(messages)
        
        # Convert to API response format
        response = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=mock_response["choices"][0]["message"]["content"]
                    ),
                    finish_reason="stop"
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=mock_response["usage"]["prompt_tokens"],
                completion_tokens=mock_response["usage"]["completion_tokens"],
                total_tokens=mock_response["usage"]["total_tokens"]
            )
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mock inference error: {str(e)}")


def start_server(host: str = "0.0.0.0", port: int = 8888):
    """Start the mock inference server."""
    print(f"Starting mock inference server on {host}:{port}")
    print("Available models:")
    print("  - mock-identity: Returns original order [1,2,3,...]")
    print("  - mock-simple: Basic completion")
    print("  - mock-swap-aware: Attempts swap tracking (with errors)")
    print("  - mock-random: Random arrangements")
    print("")
    print("Example usage:")
    print("  curl -X POST http://localhost:8888/v1/chat/completions \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"model\": \"mock-identity\", \"messages\": [{\"role\": \"user\", \"content\": \"test\"}]}'")
    print("")
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Mock Swap Tracking Inference Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8888, help="Port to bind to")
    
    args = parser.parse_args()
    start_server(args.host, args.port)