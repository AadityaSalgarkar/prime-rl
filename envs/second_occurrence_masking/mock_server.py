#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "fastapi",
#   "uvicorn",
#   "pydantic",
# ]
# requires-python = ">=3.8"
# ///
"""
Mock inference server for Second Occurrence Masking environment.

This server provides an OpenAI-compatible API for testing training pipelines
without requiring external inference services or GPU resources.

Features:
- OpenAI-compatible /v1/chat/completions endpoint
- Multiple mock model modes (identity, simple_completion, masking_aware)
- Configurable accuracy simulation
- Request/response logging
- CORS support for web clients

Usage:
    ./mock_server.py                    # Default: identity mode on port 8888
    ./mock_server.py --port 8000 --mode masking_aware --accuracy 0.7
    
API Usage:
    curl -X POST http://localhost:8888/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "mock-identity", "messages": [{"role": "user", "content": "Hello"}]}'
"""

import argparse
import json
import time
import uuid
from typing import List, Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our mock inference models
from mock_inference import create_mock_model, MockInferenceModel


# Pydantic models for API requests/responses
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


class MockInferenceServer:
    """Mock inference server with OpenAI-compatible API."""
    
    def __init__(self, mode: str = "identity", accuracy: float = 1.0, seed: int = 42):
        self.mode = mode
        self.accuracy = accuracy
        self.seed = seed
        self.model = create_mock_model(mode, accuracy, seed)
        self.request_count = 0
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Mock Inference Server",
            description="OpenAI-compatible API for Second Occurrence Masking testing",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True, 
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register API routes."""
        
        @self.app.get("/")
        async def root():
            return {
                "message": "Mock Inference Server for Second Occurrence Masking",
                "mode": self.mode,
                "accuracy": self.accuracy,
                "requests_served": self.request_count
            }
        
        @self.app.get("/v1/models")
        async def list_models():
            """List available models (OpenAI compatible)."""
            return {
                "object": "list",
                "data": [
                    {
                        "id": f"mock-{self.mode}",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "mock-org"
                    }
                ]
            }
        
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            """Handle chat completions (OpenAI compatible)."""
            try:
                self.request_count += 1
                
                # Log request
                print(f"📨 Request #{self.request_count}: {request.model}")
                for msg in request.messages:
                    print(f"   {msg.role}: {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}")
                
                # Convert to format expected by mock model
                messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
                
                # Generate response using mock model
                mock_response = self.model.complete(messages)
                
                # Calculate token usage (simplified)
                prompt_text = " ".join(msg.content for msg in request.messages)
                prompt_tokens = len(prompt_text.split())
                completion_tokens = len(mock_response.content.split())
                total_tokens = prompt_tokens + completion_tokens
                
                # Create response
                response = ChatCompletionResponse(
                    id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    created=int(time.time()),
                    model=request.model,
                    choices=[
                        Choice(
                            index=0,
                            message=Message(role="assistant", content=mock_response.content),
                            finish_reason=mock_response.finish_reason
                        )
                    ],
                    usage=Usage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens
                    )
                )
                
                # Log response
                print(f"🤖 Response: {mock_response.content[:100]}{'...' if len(mock_response.content) > 100 else ''}")
                print()
                
                return response
                
            except Exception as e:
                print(f"❌ Error processing request: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "mode": self.mode,
                "accuracy": self.accuracy,
                "requests_served": self.request_count
            }


def test_server_locally():
    """Test the server with some sample requests."""
    import requests
    
    base_url = "http://localhost:8888"
    
    print("🧪 Testing mock server locally...")
    
    # Test basic endpoint
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Root endpoint working")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False
    
    # Test models endpoint
    try:
        response = requests.get(f"{base_url}/v1/models")
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Models endpoint: {len(models['data'])} models")
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Models endpoint error: {e}")
    
    # Test chat completions
    try:
        payload = {
            "model": "mock-identity",
            "messages": [
                {"role": "user", "content": "Fill in the [MASK] tokens: The cat chased the [MASK]."}
            ],
            "max_tokens": 50
        }
        
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            print(f"✅ Chat completion: {answer}")
        else:
            print(f"❌ Chat completion failed: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Chat completion error: {e}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Mock inference server")
    parser.add_argument("--port", type=int, default=8888, 
                       help="Port to run server on")
    parser.add_argument("--host", default="0.0.0.0", 
                       help="Host to bind server to")
    parser.add_argument("--mode", choices=["identity", "simple_completion", "masking_aware"],
                       default="identity", help="Mock model mode")
    parser.add_argument("--accuracy", type=float, default=1.0,
                       help="Simulated accuracy (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--test", action="store_true",
                       help="Test server after starting")
    
    args = parser.parse_args()
    
    # Create server
    server = MockInferenceServer(args.mode, args.accuracy, args.seed)
    
    print(f"🚀 Starting Mock Inference Server")
    print(f"   Mode: {args.mode}")
    print(f"   Accuracy: {args.accuracy}")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   API: http://{args.host}:{args.port}/v1/chat/completions")
    print()
    
    # Run server
    uvicorn.run(
        server.app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()