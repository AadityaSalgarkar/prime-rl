#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["fastapi", "uvicorn"]
# requires-python = ">=3.8"
# ///
"""
Mock inference server for testing training pipeline.
Provides OpenAI-compatible API endpoints with mock responses.

Run with: ./mock_server.py
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
    from pydantic import BaseModel
except ImportError:
    print("❌ FastAPI not available. Install with: uv add fastapi uvicorn")
    exit(1)

# Import our mock inference engine
import sys
sys.path.insert(0, str(Path(__file__).parent))
from mock_inference import MockInferenceEngine


# Pydantic models for API requests/responses
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 128
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict]
    usage: Dict


# Global mock engine
mock_engine = None


app = FastAPI(title="Mock Inference Server", version="1.0.0")


@app.on_event("startup")
async def startup_event():
    global mock_engine
    mock_engine = MockInferenceEngine(mode="thesaurus_aware")
    print("🚀 Mock Inference Server started")
    print("📝 Using thesaurus-aware mock mode")


@app.get("/")
async def root():
    return {"message": "Mock Inference Server", "status": "running"}


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
                "owned_by": "mock"
            },
            {
                "id": "mock-thesaurus",
                "object": "model", 
                "created": int(time.time()),
                "owned_by": "mock"
            }
        ]
    }


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion using mock inference."""
    
    if not mock_engine:
        raise HTTPException(status_code=500, detail="Mock engine not initialized")
    
    # Extract user message
    user_content = ""
    for message in request.messages:
        if message.role == "user":
            user_content = message.content
            break
    
    if not user_content:
        raise HTTPException(status_code=400, detail="No user message found")
    
    # Generate mock completion
    try:
        completion = mock_engine.complete(user_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mock inference failed: {str(e)}")
    
    # Create response
    response = ChatCompletionResponse(
        id=f"chatcmpl-mock-{int(time.time())}",
        created=int(time.time()),
        model=request.model,
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": completion
                },
                "finish_reason": "stop"
            }
        ],
        usage={
            "prompt_tokens": len(user_content.split()),
            "completion_tokens": len(completion.split()),
            "total_tokens": len(user_content.split()) + len(completion.split())
        }
    )
    
    return response


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/v1/engines/{engine_id}/completions")
async def create_completion(engine_id: str, request: dict):
    """Legacy completion endpoint for compatibility."""
    
    # Convert to chat format
    prompt = request.get("prompt", "")
    chat_request = ChatCompletionRequest(
        model=engine_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=request.get("temperature", 0.0),
        max_tokens=request.get("max_tokens", 128)
    )
    
    return await create_chat_completion(chat_request)


def main():
    """Run the mock server."""
    print("🧪 Starting Mock Inference Server for Testing")
    print("=" * 50)
    print("📍 Server will run on: http://localhost:8888")
    print("📋 Endpoints:")
    print("   • GET  /v1/models")
    print("   • POST /v1/chat/completions")
    print("   • GET  /health")
    print("=" * 50)
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8888,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Mock server stopped")


if __name__ == "__main__":
    main()