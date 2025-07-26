#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "requests",
#   "openai",
# ]
# requires-python = ">=3.8"
# ///
"""
Ollama inference testing for Second Occurrence Masking environment.

This test suite validates:
- Ollama API connection and model availability
- Basic inference functionality with local models
- Environment-specific mask filling task testing
- Configuration template validation

Prerequisites:
    1. Install Ollama: https://ollama.ai
    2. Pull model: ollama pull qwen2.5:0.5b-instruct
    3. Ensure Ollama is running: ollama serve

Usage:
    ./tests/test_inference_ollama.py
"""

import requests
import time
import sys
from pathlib import Path
from openai import OpenAI


def check_ollama_connection():
    """Check if Ollama is running and accessible."""
    print("🔍 Checking Ollama connection...")
    
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            version_info = response.json()
            print(f"   ✅ Ollama is running (version: {version_info.get('version', 'unknown')})")
            return True
        else:
            print(f"   ❌ Ollama responded with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Cannot connect to Ollama: {e}")
        print("   💡 Make sure Ollama is installed and running:")
        print("      - Install: https://ollama.ai")
        print("      - Start: ollama serve")
        return False


def check_model_availability():
    """Check if the required model is available in Ollama."""
    print("🔍 Checking model availability...")
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models_info = response.json()
            model_names = [model['name'] for model in models_info.get('models', [])]
            
            required_model = "qwen2.5:0.5b-instruct"
            if required_model in model_names:
                print(f"   ✅ Model {required_model} is available")
                return True
            else:
                print(f"   ❌ Model {required_model} not found")
                print(f"   📋 Available models: {model_names}")
                print(f"   💡 Pull the model with: ollama pull {required_model}")
                return False
        else:
            print(f"   ❌ Failed to list models: status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Error checking models: {e}")
        return False


def test_basic_inference():
    """Test basic inference with Ollama."""
    print("🔍 Testing basic inference...")
    
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"  # Ollama doesn't require a real API key
    )
    
    try:
        response = client.chat.completions.create(
            model="qwen2.5:0.5b-instruct",
            messages=[
                {"role": "user", "content": "Hello! Please respond with 'Hello back!'"}
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        answer = response.choices[0].message.content.strip()
        print(f"   ✅ Basic inference successful")
        print(f"   📝 Response: {answer}")
        return True
        
    except Exception as e:
        print(f"   ❌ Basic inference failed: {e}")
        return False


def test_mask_filling_task():
    """Test the model's ability to handle mask filling tasks."""
    print("🔍 Testing mask filling task...")
    
    client = OpenAI(
        base_url="http://localhost:11434/v1", 
        api_key="ollama"
    )
    
    # Test cases with known answers
    test_cases = [
        {
            "masked": "The cat chased the [MASK].",
            "expected": "cat",
            "description": "Simple repetition"
        },
        {
            "masked": "She opened the door and walked through the [MASK].",
            "expected": "door", 
            "description": "Object repetition"
        },
        {
            "masked": "The student studied hard. [MASK] student passed the test.",
            "expected": "The",
            "description": "Article repetition"
        }
    ]
    
    system_prompt = """You are tasked with filling in [MASK] tokens with the original words that were replaced. 

Look at the context carefully and determine what words were originally in the masked positions. Each [MASK] represents a single word that appeared earlier in the text.

Provide your answer as a space-separated list of the words that should replace each [MASK] in order."""
    
    correct_predictions = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"   📝 Test {i}: {test_case['description']}")
        print(f"      Input: {test_case['masked']}")
        
        try:
            response = client.chat.completions.create(
                model="qwen2.5:0.5b-instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Fill in the [MASK] tokens with the original words: {test_case['masked']}"}
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content.strip().lower()
            expected = test_case['expected'].lower()
            
            print(f"      Response: {answer}")
            print(f"      Expected: {expected}")
            
            # Check if expected word is in the response
            if expected in answer:
                print(f"      ✅ Correct!")
                correct_predictions += 1
            else:
                print(f"      ❌ Incorrect")
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
        
        print()
    
    accuracy = correct_predictions / total_tests
    print(f"   📊 Accuracy: {correct_predictions}/{total_tests} ({accuracy:.1%})")
    
    return accuracy > 0.3  # Pass if > 30% accuracy (reasonable for small model)


def test_config_compatibility():
    """Test configuration file compatibility."""
    print("🔍 Testing configuration compatibility...")
    
    # Check if config files exist
    env_dir = Path(__file__).parent.parent
    config_files = [
        "inference_config_ollama.toml",
        "trainer_config.toml", 
        "orchestrator_config.toml"
    ]
    
    missing_files = []
    for config_file in config_files:
        config_path = env_dir / config_file
        if not config_path.exists():
            missing_files.append(config_file)
    
    if missing_files:
        print(f"   ❌ Missing config files: {missing_files}")
        return False
    
    # Validate Ollama config format
    ollama_config_path = env_dir / "inference_config_ollama.toml"
    try:
        with open(ollama_config_path, 'r') as f:
            content = f.read()
            
        required_fields = [
            'name = "qwen2.5:0.5b-instruct"',
            'base_url = "http://localhost:11434/v1"',
            'api_key = "ollama"'
        ]
        
        for field in required_fields:
            if field not in content:
                print(f"   ❌ Missing required field in Ollama config: {field}")
                return False
        
        print(f"   ✅ All configuration files present and valid")
        return True
        
    except Exception as e:
        print(f"   ❌ Error reading Ollama config: {e}")
        return False


def test_environment_with_ollama():
    """Test the full environment with Ollama inference."""
    print("🔍 Testing environment integration with Ollama...")
    
    try:
        # Add environment directory to path
        env_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(env_dir))
        
        from second_occurrence_loader import SecondOccurrenceMaskingLoader
        
        # Create a simple test case
        loader = SecondOccurrenceMaskingLoader(seed=42)
        text = "The cat chased the cat and the dog."
        result = loader.mask_text(text)
        
        if not result.target_words:
            print("   ⚠️  No masks created, skipping Ollama integration test")
            return True
        
        print(f"   📝 Test case: {result.masked_text}")
        print(f"   🎯 Expected: {result.target_words}")
        
        # Test with Ollama
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )
        
        system_prompt = """You are tasked with filling in [MASK] tokens with the original words that were replaced. 

Look at the context carefully and determine what words were originally in the masked positions. Each [MASK] represents a single word that appeared earlier in the text.

Provide your answer as a space-separated list of the words that should replace each [MASK] in order."""
        
        response = client.chat.completions.create(
            model="qwen2.5:0.5b-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Fill in the [MASK] tokens with the original words: {result.masked_text}"}
            ],
            max_tokens=100,
            temperature=0.1
        )
        
        answer = response.choices[0].message.content.strip()
        print(f"   🤖 Ollama response: {answer}")
        
        # Calculate reward using environment logic
        reward = loader.calculate_reward(
            result.original_text,
            answer, 
            result.mask_positions,
            result.target_words
        )
        
        print(f"   🏆 Reward: {reward:.2f}")
        print(f"   ✅ Environment integration successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Environment integration failed: {e}")
        return False


def run_all_tests():
    """Run all Ollama inference tests."""
    print("🧪 Running Ollama Inference Tests for Second Occurrence Masking\n")
    
    # Check prerequisites first
    if not check_ollama_connection():
        print("💥 Ollama connection failed. Skipping all tests.")
        return False
    
    if not check_model_availability():
        print("💥 Required model not available. Skipping inference tests.")
        return False
    
    test_functions = [
        test_basic_inference,
        test_mask_filling_task,
        test_config_compatibility,
        test_environment_with_ollama,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
            print()
        except Exception as e:
            print(f"   ❌ {test_func.__name__} failed with exception: {e}")
            failed += 1
            print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All Ollama tests passed!")
        print("\n💡 To use Ollama with training:")
        print("   cp inference_config_ollama.toml inference_config.toml")
        print("   # Then run your training with local inference")
        return True
    else:
        print(f"💥 {failed} tests failed!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)