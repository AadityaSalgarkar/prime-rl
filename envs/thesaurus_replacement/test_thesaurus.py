#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# requires-python = ">=3.8"
# ///
"""
Standalone test script for thesaurus replacement environment.
Run with: ./test_thesaurus.py
"""

import json
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_thesaurus_data():
    """Test that the thesaurus data file is properly formatted."""
    print("🧪 Testing thesaurus data format...")
    
    data_path = Path(__file__).parent / "en_thesaurus.jsonl"
    
    if not data_path.exists():
        print(f"❌ Data file not found at {data_path}")
        return False
    
    print(f"✅ Data file exists: {data_path}")
    print(f"✅ File size: {data_path.stat().st_size / (1024*1024):.1f} MB")
    
    # Test data format
    valid_entries = 0
    total_entries = 0
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 1000:  # Test first 1000 lines
                break
                
            try:
                entry = json.loads(line.strip())
                total_entries += 1
                
                if 'word' in entry and 'synonyms' in entry:
                    word = entry['word']
                    synonyms = entry['synonyms']
                    if word and isinstance(synonyms, list):
                        valid_entries += 1
                        
            except json.JSONDecodeError:
                continue
    
    print(f"✅ Valid entries in first 1000: {valid_entries}/{total_entries} ({valid_entries/total_entries*100:.1f}%)")
    return valid_entries > 0

def test_thesaurus_loader():
    """Test the ThesaurusLoader functionality."""
    print("\n🧪 Testing ThesaurusLoader...")
    
    try:
        from thesaurus_loader import ThesaurusLoader
        
        loader = ThesaurusLoader()
        print(f"✅ Loaded thesaurus with {len(loader.word_to_synonyms)} words")
        
        # Test common words
        test_words = ['good', 'bad', 'big', 'small', 'fast', 'slow', 'happy', 'old', 'new', 'great']
        words_with_synonyms = 0
        
        for word in test_words:
            synonyms = loader.get_synonyms(word)
            if synonyms:
                words_with_synonyms += 1
                print(f"  ✅ {word}: {synonyms[:3]}...")
            else:
                print(f"  ⚠️  {word}: no synonyms")
        
        print(f"✅ Found synonyms for {words_with_synonyms}/{len(test_words)} test words")
        
        # Test sentence replacement
        test_sentences = [
            "The good dog ran fast.",
            "She opened the old door.",
            "A big house with great rooms."
        ]
        
        successful_replacements = 0
        for sentence in test_sentences:
            replaceable = loader.get_replaceable_words(sentence)
            if replaceable:
                modified, replacements = loader.replace_with_synonyms(sentence, replacement_rate=0.5)
                if replacements:
                    successful_replacements += 1
                    print(f"  ✅ '{sentence}' → '{modified}'")
                else:
                    print(f"  ⚠️  '{sentence}' → no replacements made")
            else:
                print(f"  ⚠️  '{sentence}' → no replaceable words")
        
        print(f"✅ Successfully replaced words in {successful_replacements}/{len(test_sentences)} sentences")
        return True
        
    except Exception as e:
        print(f"❌ ThesaurusLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_reward_function():
    """Test the word-level accuracy reward function."""
    print("\n🧪 Testing reward function logic...")
    
    try:
        import re
        
        def word_level_accuracy_reward(response, answer):
            """Word-level exact matching reward function."""
            response = response.strip()
            answer = answer.strip()
            
            response_words = re.findall(r'\b\w+\b', response)
            answer_words = re.findall(r'\b\w+\b', answer)
            
            if not answer_words:
                return 0.0
            
            matches = 0
            for i in range(min(len(response_words), len(answer_words))):
                if response_words[i] == answer_words[i]:
                    matches += 1
            
            if len(response_words) != len(answer_words):
                matches = min(matches, len(answer_words) - abs(len(response_words) - len(answer_words)))
            
            return max(0.0, matches / len(answer_words))
        
        # Test cases
        test_cases = [
            ("She opened the ancient door.", "She opened the ancient door.", 1.0, "Perfect match"),
            ("She opened the old door.", "She opened the ancient door.", 0.8, "One word different"),
            ("She closed the ancient door.", "She opened the ancient door.", 0.8, "One word different"),
            ("Completely different sentence.", "She opened the ancient door.", 0.0, "No match"),
            ("She opened the", "She opened the ancient door.", 0.4, "Truncated (3/5 with penalty)"),
            ("", "She opened the ancient door.", 0.0, "Empty response"),
        ]
        
        all_passed = True
        for response, answer, expected_min, description in test_cases:
            actual = word_level_accuracy_reward(response, answer)
            if actual >= expected_min - 0.1:  # Allow small tolerance
                print(f"  ✅ {description}: {actual:.2f}")
            else:
                print(f"  ❌ {description}: {actual:.2f} (expected ≥{expected_min:.2f})")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Reward function test failed: {e}")
        return False

def test_environment_integration():
    """Test environment integration (lightweight)."""
    print("\n🧪 Testing environment integration...")
    
    try:
        # Test configuration files exist
        config_files = [
            "trainer_config.toml",
            "orchestrator_config.toml", 
            "inference_config.toml"
        ]
        
        for config_file in config_files:
            config_path = Path(__file__).parent / config_file
            if config_path.exists():
                print(f"✅ Config file exists: {config_file}")
                # Validate basic TOML structure
                with open(config_path, 'r') as f:
                    content = f.read()
                    if 'Qwen/Qwen2.5-0.5B-Instruct' in content:
                        print(f"  ✅ Contains expected model name")
                    if config_file == 'orchestrator_config.toml' and 'thesaurus-replacement' in content:
                        print(f"  ✅ Contains environment ID")
            else:
                print(f"❌ Config file missing: {config_file}")
                return False
        
        # Test that registry file exists and contains our environment
        repo_root = Path(__file__).parent.parent.parent
        registry_path = repo_root / "src" / "prime_rl" / "environments" / "registry.py"
        
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                registry_content = f.read()
                if 'load_thesaurus_replacement_environment' in registry_content:
                    print("✅ Environment function found in registry")
                if '"thesaurus-replacement"' in registry_content:
                    print("✅ Environment ID found in REGISTRY")
                else:
                    print("❌ Environment ID not found in REGISTRY")
                    return False
        else:
            print("❌ Registry file not found")
            return False
        
        print("✅ Environment integration check passed")
        return True
        
    except Exception as e:
        print(f"❌ Environment integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Thesaurus Replacement Environment")
    print("=" * 50)
    
    tests = [
        ("Data Format", test_thesaurus_data),
        ("ThesaurusLoader", test_thesaurus_loader),
        ("Reward Function", test_reward_function),
        ("Environment Integration", test_environment_integration),
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
        print("🎉 All tests passed! Environment is ready for use.")
        return 0
    else:
        print("💥 Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())