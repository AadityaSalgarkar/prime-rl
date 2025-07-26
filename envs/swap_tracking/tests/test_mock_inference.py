#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["requests", "openai"]
# requires-python = ">=3.8"
# ///

"""
Mock inference tests for the swap tracking environment.
Tests mock inference system and validates training pipeline compatibility.
"""

import requests
import sys
import time
import subprocess
import threading
from pathlib import Path

# Add environment directory to path
env_dir = Path(__file__).parent.parent
sys.path.insert(0, str(env_dir))

from swap_tracking_loader import SwapTrackingLoader


def start_mock_server_background(port=8888):
    """Start mock server in background."""
    print(f"Starting mock server on port {port}...")
    mock_server_path = env_dir / "mock_server.py"
    
    if not mock_server_path.exists():
        print(f"❌ Mock server not found at {mock_server_path}")
        return None
    
    try:
        # Start server in background
        process = subprocess.Popen([
            str(mock_server_path), 
            "--host", "localhost", 
            "--port", str(port)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Check if server is responding
        try:
            response = requests.get(f"http://localhost:{port}/", timeout=5)
            if response.status_code == 200:
                print(f"✓ Mock server started on port {port}")
                return process
            else:
                print(f"❌ Mock server not responding properly")
                process.terminate()
                return None
        except requests.exceptions.RequestException:
            print(f"❌ Mock server not reachable")
            process.terminate()
            return None
            
    except Exception as e:
        print(f"❌ Failed to start mock server: {e}")
        return None


def test_mock_server_health():
    """Test mock server health endpoint."""
    print("Testing mock server health...")
    try:
        response = requests.get("http://localhost:8888/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_mock_models_list():
    """Test listing available mock models."""
    print("Testing mock models list...")
    try:
        response = requests.get("http://localhost:8888/v1/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model["id"] for model in data["data"]]
            print(f"✓ Available models: {models}")
            
            expected_models = ["mock-identity", "mock-simple", "mock-swap-aware", "mock-random"]
            missing = [m for m in expected_models if m not in models]
            if missing:
                print(f"⚠ Missing expected models: {missing}")
            
            return True
        else:
            print(f"❌ Models list failed with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Models list failed: {e}")
        return False


def test_mock_inference_direct():
    """Test mock inference with direct API calls."""
    print("Testing mock inference with direct API calls...")
    
    # Test data
    test_instruction = (
        "Boxes are arranged from 1 to n=5. "
        "Then the box at location one is swapped with the box at location five. "
        "What are the final contents of all 5 boxes?"
    )
    
    models_to_test = [
        ("mock-identity", "[1, 2, 3, 4, 5]"),  # Should return original order
        ("mock-simple", None),  # Just check it responds
        ("mock-swap-aware", None),  # Should attempt swap tracking
        ("mock-random", None),  # Should return random arrangement
    ]
    
    all_passed = True
    
    for model_name, expected in models_to_test:
        print(f"\nTesting model: {model_name}")
        
        try:
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": test_instruction}
                ],
                "max_tokens": 100,
                "temperature": 0.0
            }
            
            response = requests.post(
                "http://localhost:8888/v1/chat/completions",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                result = data["choices"][0]["message"]["content"]
                print(f"✓ Response: {result}")
                
                # Check expected result for identity model
                if expected and result.strip() == expected:
                    print(f"✓ Response matches expected: {expected}")
                elif model_name == "mock-identity":
                    print(f"⚠ Expected {expected}, got {result}")
                    all_passed = False
                
            else:
                print(f"❌ API call failed with status {response.status_code}")
                all_passed = False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ API call failed: {e}")
            all_passed = False
    
    return all_passed


def test_mock_inference_with_openai_client():
    """Test mock inference using OpenAI client."""
    print("Testing mock inference with OpenAI client...")
    
    try:
        from openai import OpenAI
        
        # Configure client for mock server
        client = OpenAI(
            base_url="http://localhost:8888/v1",
            api_key="mock"
        )
        
        # Generate a swap tracking task
        loader = SwapTrackingLoader(n_boxes=5, n_swaps=3)
        instruction, swaps, final_state = loader.generate_swap_task(seed=42)
        question = loader.format_question(instruction)
        
        print(f"Test question: {question[:80]}...")
        print(f"Expected answer: {final_state}")
        
        # Test with identity model (should be deterministic)
        response = client.chat.completions.create(
            model="mock-identity",
            messages=[{"role": "user", "content": question}],
            max_tokens=100,
            temperature=0.0
        )
        
        result = response.choices[0].message.content
        print(f"Identity model response: {result}")
        
        # Test with swap-aware model
        response = client.chat.completions.create(
            model="mock-swap-aware", 
            messages=[{"role": "user", "content": question}],
            max_tokens=100,
            temperature=0.0
        )
        
        result = response.choices[0].message.content
        print(f"Swap-aware model response: {result}")
        
        # Calculate rewards
        identity_reward = loader.calculate_reward(response.choices[0].message.content, final_state)
        print(f"Identity model reward: {identity_reward}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAI client test failed: {e}")
        return False


def test_training_pipeline_compatibility():
    """Test compatibility with training pipeline format."""
    print("Testing training pipeline compatibility...")
    
    try:
        from openai import OpenAI
        
        client = OpenAI(
            base_url="http://localhost:8888/v1",
            api_key="mock"
        )
        
        # Generate examples in training format
        loader = SwapTrackingLoader(n_boxes=10, n_swaps=20)
        examples = loader.generate_training_examples(3, seed=42)
        
        successful_tests = 0
        
        for i, example in enumerate(examples):
            try:
                question = example["question"]
                expected_answer = example["answer"]
                final_state = example["info"]["final_state"]
                
                # Test with identity model (most predictable)
                response = client.chat.completions.create(
                    model="mock-identity",
                    messages=[{"role": "user", "content": question}],
                    max_tokens=128,
                    temperature=0.0
                )
                
                result = response.choices[0].message.content
                reward = loader.calculate_reward(result, final_state)
                
                print(f"Example {i+1}: Reward = {reward}")
                successful_tests += 1
                
            except Exception as e:
                print(f"Example {i+1}: Failed - {e}")
        
        if successful_tests == len(examples):
            print(f"✓ All {successful_tests} training examples processed successfully")
            return True
        else:
            print(f"⚠ Only {successful_tests}/{len(examples)} examples successful")
            return False
            
    except Exception as e:
        print(f"❌ Training pipeline compatibility test failed: {e}")
        return False


def run_all_tests():
    """Run all mock inference tests."""
    print("=" * 60)
    print("SWAP TRACKING ENVIRONMENT - MOCK INFERENCE TESTS")
    print("=" * 60)
    
    # Start mock server
    server_process = start_mock_server_background()
    if not server_process:
        print("❌ Failed to start mock server")
        return False
    
    try:
        tests = [
            (test_mock_server_health, "Mock Server Health"),
            (test_mock_models_list, "Mock Models List"),
            (test_mock_inference_direct, "Direct API Calls"),
            (test_mock_inference_with_openai_client, "OpenAI Client"),
            (test_training_pipeline_compatibility, "Training Pipeline Compatibility"),
        ]
        
        passed = 0
        failed = 0
        
        for test_func, test_name in tests:
            print(f"\nRunning: {test_name}")
            try:
                if test_func():
                    passed += 1
                    print(f"✓ {test_name} PASSED")
                else:
                    failed += 1
                    print(f"❌ {test_name} FAILED")
            except Exception as e:
                failed += 1
                print(f"❌ {test_name} FAILED: {e}")
        
        print("\n" + "=" * 60)
        print(f"RESULTS: {passed} PASSED, {failed} FAILED")
        print("=" * 60)
        
        if failed == 0:
            print("🎉 All mock inference tests passed!")
            print("")
            print("USAGE INSTRUCTIONS:")
            print("1. Start mock server: ./mock_server.py")
            print("2. Copy mock config: cp inference_config_mock.toml inference_config.toml")
            print("3. Run training with mock inference (no GPU required)")
            return True
        else:
            print("❌ Some mock inference tests failed!")
            return False
    
    finally:
        # Clean up server
        if server_process:
            print("\nShutting down mock server...")
            server_process.terminate()
            server_process.wait()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)