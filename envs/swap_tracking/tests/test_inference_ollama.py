#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["requests", "openai"]
# requires-python = ">=3.8"
# ///

"""
Ollama integration tests for the swap tracking environment.
Tests local inference using Ollama models and validates task performance.
"""

import requests
import sys
import time
from pathlib import Path

# Add environment directory to path
env_dir = Path(__file__).parent.parent
sys.path.insert(0, str(env_dir))

from swap_tracking_loader import SwapTrackingLoader


def check_ollama_connection():
    """Check if Ollama is running and accessible."""
    print("Checking Ollama connection...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            print(f"✓ Ollama is running with {len(models)} models: {model_names}")
            return True, model_names
        else:
            print(f"❌ Ollama responded with status {response.status_code}")
            return False, []
    except requests.exceptions.RequestException as e:
        print(f"❌ Could not connect to Ollama: {e}")
        return False, []


def check_model_availability(model_name):
    """Check if specific model is available in Ollama."""
    print(f"Checking if model '{model_name}' is available...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            available_models = [model["name"] for model in models]
            if model_name in available_models:
                print(f"✓ Model '{model_name}' is available")
                return True
            else:
                print(f"❌ Model '{model_name}' not found. Available models: {available_models}")
                print(f"💡 To install: ollama pull {model_name}")
                return False
        else:
            print(f"❌ Failed to check models, status: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Error checking model availability: {e}")
        return False


def test_ollama_inference(model_name="qwen2.5:0.5b-instruct"):
    """Test basic inference with Ollama."""
    print(f"Testing Ollama inference with model '{model_name}'...")
    
    try:
        from openai import OpenAI
        
        # Configure OpenAI client for Ollama
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"  # Ollama doesn't use API keys
        )
        
        # Simple test
        messages = [
            {"role": "user", "content": "What is 2+2? Respond with just the number."}
        ]
        
        print("Sending test request...")
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=10,
            temperature=0.0
        )
        
        result = response.choices[0].message.content
        print(f"✓ Test response: '{result}'")
        return True
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False


def test_swap_tracking_task(model_name="qwen2.5:0.5b-instruct"):
    """Test swap tracking task with Ollama."""
    print(f"Testing swap tracking task with model '{model_name}'...")
    
    try:
        from openai import OpenAI
        
        # Configure OpenAI client for Ollama
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"  # Ollama doesn't use API keys
        )
        
        # Generate a simple swap tracking task
        loader = SwapTrackingLoader(n_boxes=5, n_swaps=3)  # Smaller task for testing
        instruction, swaps, final_state = loader.generate_swap_task(seed=42)
        question = loader.format_question(instruction)
        
        print(f"Generated question: {question[:100]}...")
        print(f"Expected answer: {final_state}")
        
        messages = [
            {"role": "user", "content": question}
        ]
        
        print("Sending swap tracking request...")
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=100,
            temperature=0.0
        )
        
        result = response.choices[0].message.content
        print(f"Model response: '{result}'")
        
        # Calculate reward
        reward = loader.calculate_reward(result, final_state)
        print(f"Reward: {reward}")
        
        # Consider test successful if we get some non-zero reward or reasonable response
        if reward > 0:
            print("✓ Model gave partially correct answer!")
        elif any(str(num) in result for num in final_state):
            print("✓ Model response contains expected numbers")
        else:
            print("⚠ Model response doesn't match expected format, but inference worked")
        
        return True
        
    except Exception as e:
        print(f"❌ Swap tracking test failed: {e}")
        return False


def test_performance_benchmark(model_name="qwen2.5:0.5b-instruct", num_tasks=3):
    """Run performance benchmark with multiple tasks."""
    print(f"Running performance benchmark with {num_tasks} tasks...")
    
    try:
        from openai import OpenAI
        
        # Configure OpenAI client for Ollama
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )
        
        loader = SwapTrackingLoader(n_boxes=6, n_swaps=5)  # Medium difficulty
        total_reward = 0
        successful_requests = 0
        
        for i in range(num_tasks):
            try:
                # Generate task
                instruction, swaps, final_state = loader.generate_swap_task(seed=42 + i)
                question = loader.format_question(instruction)
                
                # Get model response
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": question}],
                    max_tokens=100,
                    temperature=0.0
                )
                
                result = response.choices[0].message.content
                reward = loader.calculate_reward(result, final_state)
                total_reward += reward
                successful_requests += 1
                
                print(f"Task {i+1}: Reward = {reward:.2f}")
                
            except Exception as e:
                print(f"Task {i+1}: Failed - {e}")
        
        if successful_requests > 0:
            avg_reward = total_reward / successful_requests
            print(f"✓ Average reward: {avg_reward:.3f} ({successful_requests}/{num_tasks} successful)")
            return True
        else:
            print("❌ No successful requests in benchmark")
            return False
            
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False


def test_configuration_template():
    """Test creation of Ollama configuration template."""
    print("Testing Ollama configuration template...")
    
    config_content = """[model]
name = "qwen2.5:0.5b-instruct"
base_url = "http://localhost:11434/v1"
api_key = "ollama"

[generation]
max_tokens = 128
temperature = 0.0
"""
    
    config_path = env_dir / "inference_config_ollama.toml"
    
    try:
        with open(config_path, 'w') as f:
            f.write(config_content)
        print(f"✓ Created Ollama config template: {config_path}")
        
        # Verify file exists and has content
        if config_path.exists() and config_path.stat().st_size > 0:
            print("✓ Configuration file created successfully")
            return True
        else:
            print("❌ Configuration file not created properly")
            return False
            
    except Exception as e:
        print(f"❌ Failed to create configuration template: {e}")
        return False


def run_all_tests():
    """Run all Ollama integration tests."""
    print("=" * 60)
    print("SWAP TRACKING ENVIRONMENT - OLLAMA INTEGRATION TESTS")
    print("=" * 60)
    
    # Check Ollama connection first
    ollama_available, models = check_ollama_connection()
    if not ollama_available:
        print("❌ Ollama is not running. Please start Ollama and try again.")
        print("💡 Installation: https://ollama.ai/")
        return False
    
    # Determine which model to use
    preferred_models = ["qwen2.5:0.5b-instruct", "qwen2.5:1.5b-instruct", "qwen2.5:3b-instruct"]
    model_to_use = None
    
    for model in preferred_models:
        if model in models:
            model_to_use = model
            break
    
    if not model_to_use:
        # Try the first available model that looks like qwen
        for model in models:
            if "qwen" in model.lower():
                model_to_use = model
                break
    
    if not model_to_use:
        print("❌ No suitable model found. Please install a compatible model:")
        print("💡 ollama pull qwen2.5:0.5b-instruct")
        return False
    
    print(f"Using model: {model_to_use}")
    print("")
    
    tests = [
        (lambda: test_ollama_inference(model_to_use), "Basic Ollama Inference"),
        (lambda: test_swap_tracking_task(model_to_use), "Swap Tracking Task"),
        (lambda: test_performance_benchmark(model_to_use, 3), "Performance Benchmark"),
        (test_configuration_template, "Configuration Template"),
    ]
    
    passed = 0
    failed = 0
    
    for test_func, test_name in tests:
        print(f"Running: {test_name}")
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
        print("")
    
    print("=" * 60)
    print(f"RESULTS: {passed} PASSED, {failed} FAILED")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 All Ollama integration tests passed!")
        print("")
        print("SETUP INSTRUCTIONS:")
        print("1. Copy the Ollama config template:")
        print("   cp inference_config_ollama.toml inference_config.toml")
        print("")
        print("2. Run training with Ollama:")
        print("   uv run rl \\")
        print("     --trainer @ trainer_config.toml \\")
        print("     --orchestrator @ orchestrator_config.toml \\")
        print("     --inference @ inference_config.toml")
        return True
    else:
        print("❌ Some Ollama integration tests failed!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)