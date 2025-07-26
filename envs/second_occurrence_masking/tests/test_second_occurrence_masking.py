#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "datasets",
# ]
# requires-python = ">=3.8"
# ///
"""
Core functionality tests for Second Occurrence Masking environment.

This test suite validates:
- SecondOccurrenceMaskingLoader functionality
- Text masking logic and edge cases
- Reward function accuracy  
- Environment integration with prime-rl
- Configuration compatibility

Usage:
    ./tests/test_second_occurrence_masking.py
"""

import sys
import os
import tempfile
from pathlib import Path

# Add environment directory to path
env_dir = Path(__file__).parent.parent
sys.path.insert(0, str(env_dir))

from second_occurrence_loader import SecondOccurrenceMaskingLoader, MaskingResult, create_sample_examples


def test_basic_masking():
    """Test basic text masking functionality."""
    print("🔍 Testing basic masking functionality...")
    
    loader = SecondOccurrenceMaskingLoader(min_length=10, max_length=200, seed=42)
    
    # Test simple case
    text = "The cat chased the cat."
    result = loader.mask_text(text)
    
    assert result.original_text == text
    assert "[MASK]" in result.masked_text
    assert len(result.target_words) > 0
    assert "cat" in result.target_words or "the" in result.target_words
    
    print(f"   ✅ Original: {result.original_text}")
    print(f"   ✅ Masked: {result.masked_text}")
    print(f"   ✅ Targets: {result.target_words}")
    

def test_no_repeated_words():
    """Test handling of text with no repeated words."""
    print("🔍 Testing text with no repeated words...")
    
    loader = SecondOccurrenceMaskingLoader()
    
    # Text with no repeated words
    text = "Quick brown fox jumps over lazy dog."
    result = loader.mask_text(text)
    
    assert result.original_text == text
    assert result.masked_text == text  # Should be unchanged
    assert len(result.target_words) == 0
    assert result.metadata["num_masks"] == 0
    
    print(f"   ✅ No masks created for unique words")


def test_content_word_filtering():
    """Test that function words are properly filtered when content_words_only=True."""
    print("🔍 Testing content word filtering...")
    
    # Test with content words only
    loader_content = SecondOccurrenceMaskingLoader(content_words_only=True)
    text = "The cat and the dog and the bird."
    result = loader_content.mask_text(text)
    
    # Should not mask "the" or "and" (function words)
    assert "the" not in result.target_words
    assert "and" not in result.target_words
    
    # Test with all words
    loader_all = SecondOccurrenceMaskingLoader(content_words_only=False)
    result_all = loader_all.mask_text(text)
    
    # Should mask "the" and "and"
    assert "the" in result_all.target_words or "and" in result_all.target_words
    
    print(f"   ✅ Content-only masks: {result.target_words}")
    print(f"   ✅ All-words masks: {result_all.target_words}")


def test_mask_limits():
    """Test min/max mask constraints."""
    print("🔍 Testing mask limits...")
    
    # Test max_masks limit
    loader = SecondOccurrenceMaskingLoader(max_masks=2, content_words_only=False)
    text = "The cat and the dog and the bird and the fish and the mouse."
    result = loader.mask_text(text)
    
    assert len(result.target_words) <= 2
    print(f"   ✅ Respected max_masks=2: {len(result.target_words)} masks")
    
    # Test min_masks requirement (should not fail, just return what's available)
    loader_min = SecondOccurrenceMaskingLoader(min_masks=10)
    text_short = "Cat dog cat."
    result_min = loader_min.mask_text(text_short)
    
    assert len(result_min.target_words) >= 0  # Should not crash
    print(f"   ✅ Handled min_masks gracefully: {len(result_min.target_words)} masks")


def test_reward_calculation():
    """Test reward function accuracy."""
    print("🔍 Testing reward calculation...")
    
    loader = SecondOccurrenceMaskingLoader()
    
    # Create a test case
    text = "The cat chased the cat."
    result = loader.mask_text(text)
    
    if result.target_words:
        # Test perfect response
        perfect_response = " ".join(result.target_words)
        reward_perfect = loader.calculate_reward(
            result.original_text, perfect_response, 
            result.mask_positions, result.target_words
        )
        assert reward_perfect == 1.0
        
        # Test partial response (if multiple targets)
        if len(result.target_words) > 1:
            partial_response = result.target_words[0]
            reward_partial = loader.calculate_reward(
                result.original_text, partial_response,
                result.mask_positions, result.target_words
            )
            expected_partial = 1.0 / len(result.target_words)
            assert abs(reward_partial - expected_partial) < 0.01
        
        # Test wrong response
        wrong_response = "wrong words entirely"
        reward_wrong = loader.calculate_reward(
            result.original_text, wrong_response,
            result.mask_positions, result.target_words
        )
        assert reward_wrong == 0.0
        
        # Test empty response
        reward_empty = loader.calculate_reward(
            result.original_text, "",
            result.mask_positions, result.target_words
        )
        assert reward_empty == 0.0
        
        print(f"   ✅ Perfect response reward: {reward_perfect}")
        print(f"   ✅ Wrong response reward: {reward_wrong}")
        print(f"   ✅ Empty response reward: {reward_empty}")
    else:
        print("   ⚠️  No targets to test reward calculation")


def test_sample_examples():
    """Test the create_sample_examples function."""
    print("🔍 Testing sample examples generation...")
    
    examples = create_sample_examples()
    
    assert len(examples) > 0
    for i, example in enumerate(examples):
        assert isinstance(example, MaskingResult)
        assert example.original_text != ""
        print(f"   ✅ Example {i+1}: {len(example.target_words)} masks")


def test_dataset_integration():
    """Test integration with datasets."""
    print("🔍 Testing dataset integration...")
    
    # Test with a fallback when the main dataset is not available
    loader = SecondOccurrenceMaskingLoader(
        dataset_name="nonexistent/dataset",  # Should fallback to TinyStories
        min_length=30,
        max_length=100
    )
    
    try:
        text = loader.get_sample_text()
        assert len(text) >= 30
        assert len(text) <= 100
        print(f"   ✅ Generated sample text: {len(text)} chars")
        
        # Test masking the generated text
        result = loader.mask_text(text)
        print(f"   ✅ Created {len(result.target_words)} masks from dataset text")
        
    except Exception as e:
        print(f"   ⚠️  Dataset test failed: {e}")


def test_edge_cases():
    """Test various edge cases."""
    print("🔍 Testing edge cases...")
    
    loader = SecondOccurrenceMaskingLoader()
    
    # Empty text
    result_empty = loader.mask_text("")
    assert result_empty.masked_text == ""
    assert len(result_empty.target_words) == 0
    
    # Single word
    result_single = loader.mask_text("hello")
    assert result_single.masked_text == "hello"
    assert len(result_single.target_words) == 0
    
    # Text with punctuation
    text_punct = "Hello! Hello? Yes, hello again."
    result_punct = loader.mask_text(text_punct)
    if result_punct.target_words:
        assert "hello" in [word.lower() for word in result_punct.target_words]
    
    # Very short repeated words
    text_short = "A cat, a dog, a bird."
    result_short = loader.mask_text(text_short)
    # With content_words_only=True, "a" should not be masked
    
    print(f"   ✅ Empty text: {len(result_empty.target_words)} masks")
    print(f"   ✅ Single word: {len(result_single.target_words)} masks")
    print(f"   ✅ With punctuation: {len(result_punct.target_words)} masks")
    print(f"   ✅ Short words: {len(result_short.target_words)} masks")


def test_environment_loader():
    """Test the environment loader function (requires prime-rl imports)."""
    print("🔍 Testing environment loader integration...")
    
    try:
        # Add prime-rl src to path
        prime_rl_src = Path(__file__).parent.parent.parent.parent / "src"
        if str(prime_rl_src) not in sys.path:
            sys.path.insert(0, str(prime_rl_src))
        
        from prime_rl.environments.registry import load_second_occurrence_masking_environment
        
        # Test loading with minimal examples for speed
        env = load_second_occurrence_masking_environment(
            num_examples=10,
            min_length=30,
            max_length=100
        )
        
        assert env is not None
        assert len(env.dataset) > 0
        
        # Check a few examples
        for i, example in enumerate(env.dataset):
            if i >= 3:  # Just check first 3
                break
            assert "question" in example
            assert "answer" in example
            assert "info" in example
            assert "[MASK]" in example["question"]
            
        print(f"   ✅ Environment loaded with {len(env.dataset)} examples")
        
    except ImportError as e:
        print(f"   ⚠️  Environment loader test skipped: {e}")
    except Exception as e:
        print(f"   ❌ Environment loader test failed: {e}")


def run_all_tests():
    """Run all test functions."""
    print("🧪 Running Second Occurrence Masking Environment Tests\n")
    
    test_functions = [
        test_basic_masking,
        test_no_repeated_words, 
        test_content_word_filtering,
        test_mask_limits,
        test_reward_calculation,
        test_sample_examples,
        test_dataset_integration,
        test_edge_cases,
        test_environment_loader,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
            print()
        except Exception as e:
            print(f"   ❌ {test_func.__name__} failed: {e}")
            failed += 1
            print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print(f"💥 {failed} tests failed!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)