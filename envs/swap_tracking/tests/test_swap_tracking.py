#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# requires-python = ">=3.8"
# ///

"""
Core functionality tests for the swap tracking environment.
Tests data loading, environment logic, reward functions, and basic integration.
"""

import sys
from pathlib import Path

# Add environment directory to path
env_dir = Path(__file__).parent.parent
sys.path.insert(0, str(env_dir))

from swap_tracking_loader import SwapTrackingLoader, generate_swap_task


def test_swap_task_generation():
    """Test that swap task generation works correctly."""
    print("Testing swap task generation...")
    
    # Test basic generation
    instruction, swaps, final_state = generate_swap_task(n=10, m=20, seed=42)
    
    assert isinstance(instruction, str), "Instruction should be a string"
    assert isinstance(swaps, list), "Swaps should be a list"
    assert isinstance(final_state, list), "Final state should be a list"
    assert len(swaps) == 20, f"Expected 20 swaps, got {len(swaps)}"
    assert len(final_state) == 10, f"Expected 10 boxes, got {len(final_state)}"
    assert set(final_state) == set(range(1, 11)), f"Final state should contain boxes 1-10, got {final_state}"
    
    print(f"✓ Generated instruction: {instruction[:100]}...")
    print(f"✓ Generated {len(swaps)} swaps")
    print(f"✓ Final state: {final_state}")
    
    # Test reproducibility
    instruction2, swaps2, final_state2 = generate_swap_task(n=10, m=20, seed=42)
    assert instruction == instruction2, "Same seed should produce same instruction"
    assert swaps == swaps2, "Same seed should produce same swaps"
    assert final_state == final_state2, "Same seed should produce same final state"
    
    print("✓ Reproducibility test passed")


def test_loader_initialization():
    """Test SwapTrackingLoader initialization."""
    print("Testing loader initialization...")
    
    # Default initialization
    loader = SwapTrackingLoader()
    assert loader.n_boxes == 10, f"Expected default n_boxes=10, got {loader.n_boxes}"
    assert loader.n_swaps == 20, f"Expected default n_swaps=20, got {loader.n_swaps}"
    
    # Custom initialization
    loader = SwapTrackingLoader(n_boxes=5, n_swaps=10)
    assert loader.n_boxes == 5, f"Expected n_boxes=5, got {loader.n_boxes}"
    assert loader.n_swaps == 10, f"Expected n_swaps=10, got {loader.n_swaps}"
    
    print("✓ Loader initialization tests passed")


def test_question_and_answer_formatting():
    """Test question and answer formatting."""
    print("Testing question and answer formatting...")
    
    loader = SwapTrackingLoader()
    instruction, swaps, final_state = loader.generate_swap_task(seed=42)
    
    # Test question formatting
    question = loader.format_question(instruction)
    assert isinstance(question, str), "Question should be a string"
    assert "What are the final contents" in question, "Question should ask for final contents"
    assert str(loader.n_boxes) in question, "Question should mention number of boxes"
    
    # Test answer formatting
    answer = loader.format_answer(final_state)
    assert isinstance(answer, str), "Answer should be a string"
    assert answer == str(final_state), f"Answer should be string representation of final_state"
    
    print(f"✓ Question: {question[:80]}...")
    print(f"✓ Answer: {answer}")


def test_reward_calculation():
    """Test reward calculation logic."""
    print("Testing reward calculation...")
    
    loader = SwapTrackingLoader()
    final_state = [10, 2, 7, 4, 5, 6, 3, 8, 9, 1]
    
    # Test perfect prediction
    perfect_prediction = str(final_state)
    reward = loader.calculate_reward(perfect_prediction, final_state)
    assert reward == 1.0, f"Perfect prediction should give reward 1.0, got {reward}"
    
    # Test completely wrong prediction
    wrong_prediction = "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"
    reward = loader.calculate_reward(wrong_prediction, final_state)
    print(f"Original order reward: {reward}")
    
    # Test partial prediction (first half correct)
    partial_prediction = "[10, 2, 7, 4, 5, 1, 2, 3, 4, 5]"
    reward = loader.calculate_reward(partial_prediction, final_state)
    expected_reward = 5 / 10  # 5 correct positions out of 10
    assert reward == expected_reward, f"Expected reward {expected_reward}, got {reward}"
    
    # Test invalid prediction
    invalid_prediction = "not a valid list"
    reward = loader.calculate_reward(invalid_prediction, final_state)
    assert reward == 0.0, f"Invalid prediction should give reward 0.0, got {reward}"
    
    # Test wrong length prediction
    wrong_length = "[1, 2, 3]"
    reward = loader.calculate_reward(wrong_length, final_state)
    assert reward == 0.0, f"Wrong length prediction should give reward 0.0, got {reward}"
    
    print("✓ Reward calculation tests passed")


def test_training_examples_generation():
    """Test generation of multiple training examples."""
    print("Testing training examples generation...")
    
    loader = SwapTrackingLoader()
    
    # Generate examples
    examples = loader.generate_training_examples(5, seed=42)
    
    assert len(examples) == 5, f"Expected 5 examples, got {len(examples)}"
    
    for i, example in enumerate(examples):
        assert "question" in example, f"Example {i} missing 'question'"
        assert "answer" in example, f"Example {i} missing 'answer'"
        assert "info" in example, f"Example {i} missing 'info'"
        assert "task" in example, f"Example {i} missing 'task'"
        assert example["task"] == "swap-tracking", f"Example {i} has wrong task name"
        
        # Check info structure
        info = example["info"]
        assert "instruction_text" in info, f"Example {i} info missing 'instruction_text'"
        assert "swaps" in info, f"Example {i} info missing 'swaps'"
        assert "final_state" in info, f"Example {i} info missing 'final_state'"
        assert "n_boxes" in info, f"Example {i} info missing 'n_boxes'"
        assert "n_swaps" in info, f"Example {i} info missing 'n_swaps'"
        
        assert len(info["swaps"]) == loader.n_swaps, f"Example {i} has wrong number of swaps"
        assert len(info["final_state"]) == loader.n_boxes, f"Example {i} has wrong number of boxes"
    
    print(f"✓ Generated {len(examples)} training examples")
    print(f"✓ Example structure validation passed")


def test_environment_integration():
    """Test basic integration with verifiers library concepts."""
    print("Testing environment integration...")
    
    loader = SwapTrackingLoader()
    examples = loader.generate_training_examples(3, seed=42)
    
    # Test reward calculation with actual examples
    for i, example in enumerate(examples):
        final_state = example["info"]["final_state"]
        answer = example["answer"]
        
        # Test reward with the correct answer
        reward = loader.calculate_reward(answer, final_state)
        assert reward == 1.0, f"Example {i}: correct answer should give reward 1.0, got {reward}"
        
        # Test reward with incorrect answer
        wrong_answer = "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"
        reward = loader.calculate_reward(wrong_answer, final_state)
        print(f"Example {i}: wrong answer reward = {reward}")
    
    print("✓ Environment integration tests passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("SWAP TRACKING ENVIRONMENT - CORE FUNCTIONALITY TESTS")
    print("=" * 60)
    
    tests = [
        test_swap_task_generation,
        test_loader_initialization,
        test_question_and_answer_formatting,
        test_reward_calculation,
        test_training_examples_generation,
        test_environment_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
            print("")
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            failed += 1
            print("")
    
    print("=" * 60)
    print(f"RESULTS: {passed} PASSED, {failed} FAILED")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)