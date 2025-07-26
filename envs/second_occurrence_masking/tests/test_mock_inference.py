#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "requests",
#   "openai",
# ]
# requires-python = ">=3.8"
# ///
"""
Mock inference testing for Second Occurrence Masking environment.

This test suite validates:
- Mock inference model functionality in different modes
- Mock server API compatibility with OpenAI format
- Integration with environment reward calculation
- Training pipeline compatibility without external dependencies

Usage:
    ./tests/test_mock_inference.py
"""

import sys
import time
import subprocess
import signal
import requests
from pathlib import Path
from openai import OpenAI


# Add environment directory to path
env_dir = Path(__file__).parent.parent
sys.path.insert(0, str(env_dir))

from mock_inference import create_mock_model, IdentityModel, SimpleCompletionModel, MaskingAwareModel


def test_mock_models():
    """Test different mock model modes."""
    print("🔍 Testing mock model modes...")
    
    test_messages = [
        [{"role": "user", "content": "Hello! Please respond with 'Hello back!'"}],
        [{"role": "user", "content": "Fill in the [MASK] tokens with the original words: The cat chased the [MASK]."}],
    ]
    
    modes = ["identity", "simple_completion", "masking_aware"]
    
    for mode in modes:
        print(f"   Testing {mode} mode...")
        model = create_mock_model(mode, accuracy=0.8)
        
        # Test basic functionality
        for messages in test_messages:
            response = model.complete(messages)
            assert hasattr(response, 'content')
            assert hasattr(response, 'finish_reason')
            assert len(response.content) > 0
        
        print(f"   ✅ {mode} mode working")
    
    print()


def test_identity_model_accuracy():
    """Test identity model provides correct answers for known cases."""
    print("🔍 Testing identity model accuracy...")
    
    model = IdentityModel()
    
    test_cases = [
        {
            "messages": [{"role": "user", "content": "Fill in the [MASK] tokens with the original words: The cat chased the [MASK]."}],
            "expected": "cat"
        },
        {
            "messages": [{"role": "user", "content": "Fill in the [MASK] tokens with the original words: She opened the door and walked through the [MASK]."}],
            "expected": "door"
        }
    ]
    
    correct = 0
    for test_case in test_cases:
        response = model.complete(test_case["messages"])
        if test_case["expected"].lower() in response.content.lower():
            correct += 1
    
    accuracy = correct / len(test_cases)
    print(f"   ✅ Identity model accuracy: {accuracy:.1%}")
    assert accuracy >= 0.5  # Should get at least half right
    print()


def test_masking_aware_model():
    """Test masking-aware model behavior."""
    print("🔍 Testing masking-aware model...")
    
    # Test high accuracy mode
    model_high = MaskingAwareModel(accuracy=0.9)
    
    # Test low accuracy mode  
    model_low = MaskingAwareModel(accuracy=0.2)
    
    test_message = [{"role": "user", "content": "Fill in the [MASK] tokens with the original words: The cat chased the [MASK]."}]
    
    # Run multiple tests to check accuracy simulation
    high_correct = 0
    low_correct = 0
    num_tests = 20
    
    for _ in range(num_tests):
        response_high = model_high.complete(test_message)
        response_low = model_low.complete(test_message)
        
        if "cat" in response_high.content.lower():
            high_correct += 1
        if "cat" in response_low.content.lower():
            low_correct += 1
    
    high_accuracy = high_correct / num_tests
    low_accuracy = low_correct / num_tests
    
    print(f"   ✅ High accuracy model: {high_accuracy:.1%}")
    print(f"   ✅ Low accuracy model: {low_accuracy:.1%}")
    
    # High accuracy should be better than low accuracy
    assert high_accuracy > low_accuracy
    print()


def start_mock_server(port=8888, mode="identity"):
    """Start the mock server for testing."""
    print(f"🚀 Starting mock server on port {port}...")
    
    server_script = env_dir / "mock_server.py"
    process = subprocess.Popen([
        str(server_script), 
        "--port", str(port),
        "--mode", mode,
        "--host", "127.0.0.1"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    for _ in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=1)
            if response.status_code == 200:
                print(f"   ✅ Mock server started on port {port}")
                return process
        except:
            time.sleep(1)
    
    # If we get here, server didn't start
    process.terminate()
    print(f"   ❌ Failed to start mock server on port {port}")
    return None


def stop_mock_server(process):
    """Stop the mock server."""
    if process:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        print("   ✅ Mock server stopped")


def test_mock_server_api():
    """Test the mock server API endpoints."""
    print("🔍 Testing mock server API...")
    
    # Start server
    server_process = start_mock_server(port=8887, mode="masking_aware")
    if not server_process:
        print("   ⚠️  Skipping server tests - could not start server")
        return False
    
    try:
        base_url = "http://localhost:8887"
        
        # Test root endpoint
        response = requests.get(f"{base_url}/")
        assert response.status_code == 200
        data = response.json()
        assert "mode" in data
        print(f"   ✅ Root endpoint: {data['mode']} mode")
        
        # Test models endpoint
        response = requests.get(f"{base_url}/v1/models")
        assert response.status_code == 200
        models = response.json()
        assert "data" in models
        assert len(models["data"]) > 0
        print(f"   ✅ Models endpoint: {len(models['data'])} models")
        
        # Test chat completions
        payload = {
            "model": "mock-masking_aware",
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
        
        assert response.status_code == 200
        result = response.json()
        assert "choices" in result
        assert len(result["choices"]) > 0
        answer = result["choices"][0]["message"]["content"]
        print(f"   ✅ Chat completion: '{answer}'")
        
        return True
        
    finally:
        stop_mock_server(server_process)
    
    print()


def test_openai_client_compatibility():
    """Test compatibility with OpenAI client library."""
    print("🔍 Testing OpenAI client compatibility...")
    
    # Start server
    server_process = start_mock_server(port=8886, mode="identity")
    if not server_process:
        print("   ⚠️  Skipping OpenAI client tests - could not start server")
        return False
    
    try:
        # Create OpenAI client pointing to our mock server
        client = OpenAI(
            base_url="http://localhost:8886/v1",
            api_key="mock"
        )
        
        # Test basic completion
        response = client.chat.completions.create(
            model="mock-identity",
            messages=[
                {"role": "user", "content": "Fill in the [MASK] tokens: The cat chased the [MASK]."}
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        assert response.choices[0].message.content
        answer = response.choices[0].message.content
        print(f"   ✅ OpenAI client compatible: '{answer}'")
        
        return True
        
    except Exception as e:
        print(f"   ❌ OpenAI client test failed: {e}")
        return False
        
    finally:
        stop_mock_server(server_process)
    
    print()


def test_environment_integration():
    """Test mock inference with environment reward calculation."""
    print("🔍 Testing environment integration...")
    
    try:
        from second_occurrence_loader import SecondOccurrenceMaskingLoader
        
        # Create test case
        loader = SecondOccurrenceMaskingLoader(seed=42)
        text = "The cat chased the cat and the dog."
        result = loader.mask_text(text)
        
        if not result.target_words:
            print("   ⚠️  No masks created, skipping integration test")
            return True
        
        print(f"   📝 Test case: {result.masked_text}")
        print(f"   🎯 Expected: {result.target_words}")
        
        # Test with different mock models
        models = {
            "identity": create_mock_model("identity"),
            "masking_aware": create_mock_model("masking_aware", accuracy=0.8)
        }
        
        for mode, model in models.items():
            messages = [
                {"role": "user", "content": f"Fill in the [MASK] tokens with the original words: {result.masked_text}"}
            ]
            
            response = model.complete(messages)
            reward = loader.calculate_reward(
                result.original_text,
                response.content,
                result.mask_positions,
                result.target_words
            )
            
            print(f"   🤖 {mode}: '{response.content}' -> reward: {reward:.2f}")
        
        print("   ✅ Environment integration successful")
        return True
        
    except Exception as e:
        print(f"   ❌ Environment integration failed: {e}")
        return False
    
    print()


def run_all_tests():
    """Run all mock inference tests."""
    print("🧪 Running Mock Inference Tests for Second Occurrence Masking\n")
    
    test_functions = [
        test_mock_models,
        test_identity_model_accuracy,
        test_masking_aware_model,
        test_mock_server_api,
        test_openai_client_compatibility,
        test_environment_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            result = test_func()
            if result is not False:  # None or True are both considered passing
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"   ❌ {test_func.__name__} failed with exception: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All mock inference tests passed!")
        print("\n💡 To use mock inference with training:")
        print("   ./mock_server.py --mode masking_aware --accuracy 0.8 &")
        print("   cp inference_config_mock.toml inference_config.toml")
        print("   # Then run your training with mock inference")
        return True
    else:
        print(f"💥 {failed} tests failed!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)