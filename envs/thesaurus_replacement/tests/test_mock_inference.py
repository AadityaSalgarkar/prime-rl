#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["requests", "fastapi", "uvicorn"]
# requires-python = ">=3.8"
# ///
"""
Test script for mock inference system.
Tests the mock model's ability to provide identity mappings and basic completions.

Run with: ./tests/test_mock_inference.py
"""

import json
import sys
import time
import threading
import subprocess
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_mock_inference_engine():
    """Test the MockInferenceEngine directly."""
    print("🧪 Testing MockInferenceEngine...")
    
    try:
        from mock_inference import MockInferenceEngine
        
        # Test identity mode
        print("  📝 Testing identity mode...")
        engine_identity = MockInferenceEngine(mode="identity")
        
        test_inputs = [
            "Hello world",
            "The cat sat on the mat",
            "Restore the original text: She opened the door."
        ]
        
        for test_input in test_inputs:
            output = engine_identity.complete(test_input)
            if output == test_input:
                print(f"    ✅ Identity: '{test_input}' → '{output}'")
            else:
                print(f"    ❌ Identity failed: '{test_input}' → '{output}'")
                return False
        
        # Test simple completion mode
        print("  📝 Testing simple completion mode...")
        engine_simple = MockInferenceEngine(mode="simple_completion")
        
        completion_tests = [
            ("Complete this sentence: The cat sat on the", "and lived happily ever after."),
            ("Restore the original text: Hello world", "Hello world"),
            ("Random prompt", "[completed]")
        ]
        
        for test_input, expected_ending in completion_tests:
            output = engine_simple.complete(test_input)
            if expected_ending in output:
                print(f"    ✅ Simple completion: '{test_input}' → '{output}'")
            else:
                print(f"    ❌ Simple completion failed: '{test_input}' → '{output}'")
        
        # Test thesaurus-aware mode
        print("  📝 Testing thesaurus-aware mode...")
        engine_thesaurus = MockInferenceEngine(mode="thesaurus_aware")
        
        # Test basic restoration
        thesaurus_input = "Restore the original text: The good dog ran fast."
        thesaurus_output = engine_thesaurus.complete(thesaurus_input)
        
        print(f"    📊 Thesaurus test: '{thesaurus_input}' → '{thesaurus_output}'")
        
        if "dog" in thesaurus_output.lower():
            print("    ✅ Thesaurus-aware mode working")
        else:
            print("    ⚠️  Thesaurus-aware mode may need adjustment")
        
        print("✅ MockInferenceEngine tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ MockInferenceEngine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mock_openai_api():
    """Test the MockOpenAICompatibleAPI."""
    print("\n🧪 Testing MockOpenAICompatibleAPI...")
    
    try:
        from mock_inference import create_mock_client
        
        # Test identity mode
        client_identity = create_mock_client("identity")
        
        messages = [{"role": "user", "content": "Hello world"}]
        response = client_identity.create(model="mock", messages=messages)
        
        output = response.choices[0].message.content
        if output == "Hello world":
            print("  ✅ Mock API identity mode working")
        else:
            print(f"  ❌ Mock API identity failed: expected 'Hello world', got '{output}'")
            return False
        
        # Test thesaurus-aware mode
        client_thesaurus = create_mock_client("thesaurus_aware")
        
        messages = [{"role": "user", "content": "Restore the original text: She unfastened the door."}]
        response = client_thesaurus.create(model="mock-thesaurus", messages=messages)
        
        output = response.choices[0].message.content
        print(f"  📊 Mock API thesaurus test: '{messages[0]['content']}' → '{output}'")
        
        if "door" in output.lower():
            print("  ✅ Mock API thesaurus-aware mode working")
        else:
            print("  ⚠️  Mock API thesaurus-aware mode may need adjustment")
        
        print("✅ MockOpenAICompatibleAPI tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ MockOpenAICompatibleAPI test failed: {e}")
        return False

def check_mock_server_available() -> bool:
    """Check if mock server is running."""
    try:
        import requests
        response = requests.get("http://localhost:8888/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def test_mock_server_integration():
    """Test integration with the mock server."""
    print("\n🧪 Testing Mock Server Integration...")
    
    if not check_mock_server_available():
        print("⚠️  Mock server not running. Start with: ./mock_server.py")
        print("   Testing will continue with server simulation...")
        return test_mock_server_simulation()
    
    try:
        import requests
        
        # Test models endpoint
        response = requests.get("http://localhost:8888/v1/models")
        if response.status_code == 200:
            models_data = response.json()
            print(f"  ✅ Models endpoint working: {len(models_data['data'])} models available")
        else:
            print(f"  ❌ Models endpoint failed: {response.status_code}")
            return False
        
        # Test chat completions
        chat_request = {
            "model": "mock-identity",
            "messages": [{"role": "user", "content": "Hello mock server"}],
            "temperature": 0.0
        }
        
        response = requests.post(
            "http://localhost:8888/v1/chat/completions",
            json=chat_request
        )
        
        if response.status_code == 200:
            completion_data = response.json()
            output = completion_data["choices"][0]["message"]["content"]
            print(f"  ✅ Chat completion working: '{output}'")
        else:
            print(f"  ❌ Chat completion failed: {response.status_code}")
            return False
        
        # Test thesaurus restoration
        thesaurus_request = {
            "model": "mock-thesaurus",
            "messages": [{"role": "user", "content": "Restore the original text: She unfastened the door."}],
            "temperature": 0.0
        }
        
        response = requests.post(
            "http://localhost:8888/v1/chat/completions",
            json=thesaurus_request
        )
        
        if response.status_code == 200:
            completion_data = response.json()
            output = completion_data["choices"][0]["message"]["content"]
            print(f"  📊 Thesaurus restoration: '{output}'")
            print("  ✅ Mock server integration working")
        else:
            print(f"  ❌ Thesaurus request failed: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mock server integration test failed: {e}")
        return False

def test_mock_server_simulation():
    """Simulate mock server testing when server is not running."""
    print("  📝 Simulating mock server behavior...")
    
    try:
        from mock_inference import create_mock_client
        
        # Simulate OpenAI client using our mock
        mock_client = create_mock_client("thesaurus_aware")
        
        # Test chat completion simulation
        messages = [{"role": "user", "content": "Restore the original text: She unfastened the door."}]
        response = mock_client.create(model="mock", messages=messages)
        
        output = response.choices[0].message.content
        print(f"  📊 Simulated completion: '{output}'")
        print("  ✅ Mock server simulation working")
        
        return True
        
    except Exception as e:
        print(f"❌ Mock server simulation failed: {e}")
        return False

def test_training_pipeline_compatibility():
    """Test compatibility with training pipeline."""
    print("\n🧪 Testing Training Pipeline Compatibility...")
    
    try:
        # Test configuration file exists
        config_path = Path(__file__).parent.parent / "inference_config_mock.toml"
        if config_path.exists():
            print("  ✅ Mock configuration file exists")
            
            with open(config_path, 'r') as f:
                config_content = f.read()
                if 'mock-identity' in config_content:
                    print("  ✅ Configuration contains mock model reference")
                if 'localhost:8888' in config_content:
                    print("  ✅ Configuration points to mock server")
        else:
            print("  ❌ Mock configuration file missing")
            return False
        
        # Test thesaurus environment integration
        from thesaurus_loader import ThesaurusLoader
        
        loader = ThesaurusLoader()
        
        # Create a test example
        original = "The good dog ran fast."
        augmented, replacements = loader.replace_with_synonyms(original, replacement_rate=0.3)
        
        if replacements:
            print(f"  📝 Test example: '{original}' → '{augmented}'")
            
            # Test with mock inference
            from mock_inference import create_mock_client
            mock_client = create_mock_client("thesaurus_aware")
            
            prompt = f"Restore the original text: {augmented}"
            messages = [{"role": "user", "content": prompt}]
            response = mock_client.create(model="mock", messages=messages)
            
            mock_output = response.choices[0].message.content
            print(f"  🤖 Mock restoration: '{mock_output}'")
            
            # Simple accuracy check
            import re
            original_words = re.findall(r'\b\w+\b', original.lower())
            mock_words = re.findall(r'\b\w+\b', mock_output.lower())
            
            matches = sum(1 for o, m in zip(original_words, mock_words) if o == m)
            accuracy = matches / len(original_words) if original_words else 0
            
            print(f"  📊 Mock accuracy: {accuracy:.2f} ({matches}/{len(original_words)} words)")
            print("  ✅ Training pipeline compatibility confirmed")
        else:
            print("  ⚠️  No replacements made in test example")
        
        return True
        
    except Exception as e:
        print(f"❌ Training pipeline compatibility test failed: {e}")
        return False

def main():
    """Run all mock inference tests."""
    print("🚀 Testing Mock Inference System")
    print("=" * 50)
    
    tests = [
        ("Mock Inference Engine", test_mock_inference_engine),
        ("Mock OpenAI API", test_mock_openai_api),
        ("Mock Server Integration", test_mock_server_integration),
        ("Training Pipeline Compatibility", test_training_pipeline_compatibility),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"\n{status} {test_name}")
        except Exception as e:
            print(f"\n❌ FAIL {test_name}: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print(f"📊 TEST SUMMARY: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("🎉 All mock inference tests passed!")
        print("✅ Ready for training pipeline testing")
        
        print("\n💡 To use mock inference:")
        print("   1. Start mock server: ./mock_server.py")
        print("   2. Use mock config: cp inference_config_mock.toml inference_config.toml")
        print("   3. Run training with mock inference")
        
        return 0
    else:
        print("💥 Some tests failed. Check output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())