#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["requests", "openai"]
# requires-python = ">=3.8"
# ///
"""
Test script for thesaurus replacement environment with Ollama API inference.
Tests the environment's ability to work with local Ollama models.

Run with: ./tests/test_inference_ollama.py
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_ollama_available() -> bool:
    """Check if Ollama is running and available."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def get_available_models() -> list:
    """Get list of available Ollama models."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        return []
    except Exception:
        return []

def test_ollama_connection():
    """Test basic connection to Ollama API."""
    print("🧪 Testing Ollama API connection...")
    
    if not check_ollama_available():
        print("❌ Ollama not available at localhost:11434")
        print("   Please start Ollama with: ollama serve")
        return False
    
    print("✅ Ollama is running")
    
    models = get_available_models()
    if not models:
        print("❌ No models found in Ollama")
        print("   Please install a model with: ollama pull qwen2.5:0.5b-instruct")
        return False
    
    print(f"✅ Found {len(models)} models: {models[:3]}{'...' if len(models) > 3 else ''}")
    return True

def test_ollama_inference():
    """Test basic inference with Ollama."""
    print("\n🧪 Testing Ollama inference...")
    
    try:
        from openai import OpenAI
        
        # Create OpenAI client pointing to Ollama
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"  # Ollama doesn't require real API key
        )
        
        models = get_available_models()
        if not models:
            print("❌ No models available for testing")
            return False
        
        test_model = models[0]
        print(f"✅ Using model: {test_model}")
        
        # Test simple completion
        messages = [
            {"role": "user", "content": "Complete this sentence: The cat sat on the"}
        ]
        
        print("  🔄 Testing completion...")
        response = client.chat.completions.create(
            model=test_model,
            messages=messages,
            max_tokens=10,
            temperature=0.1
        )
        
        completion = response.choices[0].message.content
        print(f"  ✅ Got response: '{completion.strip()}'")
        return True
        
    except Exception as e:
        print(f"❌ Ollama inference test failed: {e}")
        return False

def test_thesaurus_environment_with_ollama():
    """Test thesaurus replacement environment with Ollama API."""
    print("\n🧪 Testing thesaurus environment with Ollama...")
    
    try:
        from thesaurus_loader import ThesaurusLoader
        from openai import OpenAI
        
        # Setup
        loader = ThesaurusLoader()
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )
        
        models = get_available_models()
        if not models:
            print("❌ No models available")
            return False
        
        test_model = models[0]
        
        # Create test example
        original_sentence = "The good dog ran fast through the big park."
        augmented_sentence, replacements = loader.replace_with_synonyms(
            original_sentence, replacement_rate=0.5
        )
        
        if not replacements:
            print("⚠️  No replacements made, using manual example")
            original_sentence = "She opened the ancient door."
            augmented_sentence = "She unfastened the antique door."
        
        print(f"  📝 Original: {original_sentence}")
        print(f"  📝 Augmented: {augmented_sentence}")
        
        # Test with Ollama
        prompt = f"Restore the original text: {augmented_sentence}"
        messages = [{"role": "user", "content": prompt}]
        
        print("  🔄 Querying Ollama...")
        response = client.chat.completions.create(
            model=test_model,
            messages=messages,
            max_tokens=50,
            temperature=0.1
        )
        
        model_response = response.choices[0].message.content.strip()
        print(f"  🤖 Model response: {model_response}")
        
        # Simple evaluation
        import re
        original_words = re.findall(r'\b\w+\b', original_sentence.lower())
        response_words = re.findall(r'\b\w+\b', model_response.lower())
        
        matches = 0
        for i in range(min(len(original_words), len(response_words))):
            if original_words[i] == response_words[i]:
                matches += 1
        
        accuracy = matches / len(original_words) if original_words else 0
        print(f"  📊 Word accuracy: {accuracy:.2f} ({matches}/{len(original_words)} words)")
        
        if accuracy > 0.5:
            print("  ✅ Reasonable performance detected")
        else:
            print("  ⚠️  Low accuracy (expected for simple models)")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment test with Ollama failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_config_compatibility():
    """Test that environment configs can work with Ollama."""
    print("\n🧪 Testing config compatibility with Ollama...")
    
    try:
        # Check if config files exist
        config_files = {
            "inference_config.toml": "inference",
            "orchestrator_config.toml": "orchestrator", 
            "trainer_config.toml": "trainer"
        }
        
        for config_file, config_type in config_files.items():
            config_path = Path(__file__).parent.parent / config_file
            if config_path.exists():
                with open(config_path, 'r') as f:
                    content = f.read()
                    print(f"  ✅ {config_type} config exists")
                    
                    # Check if it contains model reference
                    if 'Qwen' in content or 'model' in content:
                        print(f"    📝 Contains model configuration")
            else:
                print(f"  ❌ {config_type} config missing")
                return False
        
        print("  💡 To use with Ollama, update inference config:")
        print("     [model]")
        print("     name = \"qwen2.5:0.5b-instruct\"  # matches training config")
        print("     base_url = \"http://localhost:11434/v1\"")
        print("     api_key = \"ollama\"")
        
        return True
        
    except Exception as e:
        print(f"❌ Config compatibility test failed: {e}")
        return False

def test_ollama_model_recommendations():
    """Provide model recommendations for thesaurus replacement task."""
    print("\n🧪 Testing Ollama model recommendations...")
    
    available_models = get_available_models()
    
    # Recommended models for text reconstruction tasks  
    recommended_models = [
        "qwen2.5", "llama3", "mistral", "phi", "qwen", "codellama"
    ]
    
    found_recommended = []
    for model in available_models:
        for rec in recommended_models:
            if rec in model.lower():
                found_recommended.append(model)
                break
    
    if found_recommended:
        print(f"  ✅ Found recommended models: {found_recommended}")
        print(f"  💡 Best for text tasks: {found_recommended[0]}")
    else:
        print(f"  ⚠️  Available models: {available_models}")
        print(f"  💡 Consider installing: ollama pull qwen2.5:0.5b-instruct")
    
    # Performance expectations
    print("  📊 Expected performance for thesaurus replacement:")
    print("     • qwen2.5:0.5b-instruct: Matches training config (recommended)")
    print("     • llama3: Good instruction following")
    print("     • mistral: Fast and accurate")
    print("     • qwen: Strong multilingual support")
    print("     • phi: Lightweight but capable")
    
    return True

def main():
    """Run all Ollama integration tests."""
    print("🚀 Testing Thesaurus Replacement Environment with Ollama")
    print("=" * 60)
    
    tests = [
        ("Ollama Connection", test_ollama_connection),
        ("Basic Inference", test_ollama_inference),
        ("Environment Integration", test_thesaurus_environment_with_ollama),
        ("Config Compatibility", test_environment_config_compatibility),
        ("Model Recommendations", test_ollama_model_recommendations),
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
    
    print("\n" + "=" * 60)
    print(f"📊 TEST SUMMARY: {sum(results)}/{len(results)} tests passed")
    
    if not check_ollama_available():
        print("\n💡 To use Ollama with this environment:")
        print("   1. Install Ollama: https://ollama.ai")
        print("   2. Start server: ollama serve")
        print("   3. Install model: ollama pull qwen2.5:0.5b-instruct")
        print("   4. Update inference_config.toml with Ollama settings")
    
    if all(results):
        print("🎉 All tests passed! Environment works with Ollama.")
        return 0
    else:
        print("💥 Some tests failed. Check Ollama setup.")
        return 1

if __name__ == "__main__":
    sys.exit(main())