#!/usr/bin/env python3
"""
Integration test for creativity environment with prime-rl system.

This test validates the complete integration between the creativity environment
and the main RLVR training system.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_environment_loading():
    """Test that the environment loads correctly through the registry."""
    try:
        # Test import from registry
        repo_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(repo_root / "src"))
        
        from prime_rl.environments.registry import load_environment
        
        # Load creativity environment through registry
        env = load_environment("creativity", {
            "num_train_samples": 10,
            "num_eval_samples": 5
        })
        
        logger.info("✓ Environment loading through registry successful")
        
        # Check environment structure
        assert hasattr(env, 'dataset'), "Environment missing dataset"
        assert hasattr(env, 'eval_dataset'), "Environment missing eval_dataset"
        assert hasattr(env, 'system_prompt'), "Environment missing system_prompt"
        assert hasattr(env, 'rubric'), "Environment missing rubric"
        
        logger.info("✓ Environment structure validation successful")
        
        # Test basic environment properties
        assert len(env.dataset) >= 10, f"Expected at least 10 training samples, got {len(env.dataset)}"
        assert len(env.eval_dataset) >= 5, f"Expected at least 5 eval samples, got {len(env.eval_dataset)}"
        
        logger.info(f"✓ Environment has {len(env.dataset)} training and {len(env.eval_dataset)} evaluation samples")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Environment loading failed: {e}")
        return False

def test_reward_calculation():
    """Test reward calculation functionality."""
    try:
        from reward import reward_function
        
        # Test with various text samples
        test_samples = [
            "The quick brown fox jumps over the lazy dog.",  # Basic text
            "In the ephemeral dance of gossamer dreams, crystalline thoughts cascade through labyrinthine corridors of imagination, where whispered secrets paint iridescent tapestries of possibility.",  # Creative text
            "Test. Test. Test.",  # Repetitive text
        ]
        
        expected_patterns = [
            "should be moderate",  # Basic text should get moderate score
            "should be high",      # Creative text should get high score  
            "should be low"        # Repetitive text should get low score
        ]
        
        scores = []
        for i, text in enumerate(test_samples):
            score = reward_function(text)
            scores.append(score)
            logger.info(f"Sample {i+1} score: {score:.3f} ({expected_patterns[i]})")
        
        # Validate score patterns
        assert scores[1] > scores[0], "Creative text should score higher than basic text"
        assert scores[0] > scores[2], "Basic text should score higher than repetitive text"
        
        logger.info("✓ Reward calculation patterns are correct")
        return True
        
    except Exception as e:
        logger.error(f"✗ Reward calculation test failed: {e}")
        return False

def test_data_generation():
    """Test data generation and augmentation."""
    try:
        from data_utils import CreativeTextGenerator, DataAugmentation
        
        # Test text generator
        generator = CreativeTextGenerator(seed=42)
        sample_text = generator.generate_creative_sample(
            "Write about the nature of time",
            length="medium",
            technique="synesthesia"
        )
        
        assert len(sample_text) > 50, "Generated text too short"
        assert len(sample_text.split()) > 10, "Generated text has too few words"
        
        logger.info(f"✓ Generated sample text ({len(sample_text)} chars): {sample_text[:100]}...")
        
        # Test data augmentation
        base_prompts = ["Write a creative story", "Describe a magical place"]
        augmented = DataAugmentation.augment_prompt_variations(base_prompts)
        
        assert len(augmented) > len(base_prompts), "Augmentation should increase prompt count"
        
        logger.info(f"✓ Data augmentation created {len(augmented)} prompts from {len(base_prompts)} base prompts")
        return True
        
    except Exception as e:
        logger.error(f"✗ Data generation test failed: {e}")
        return False

def test_evaluation_system():
    """Test evaluation and analysis systems."""
    try:
        from evaluation_rubrics import CreativityRubric, CreativityTracker
        
        # Test rubric evaluation
        rubric = CreativityRubric()
        
        test_text = "The kaleidoscopic symphony of dreams whispered through gossamer threads of consciousness, painting ethereal tapestries across the canvas of imagination."
        
        analysis = rubric.evaluate_text(test_text, "Write creatively about dreams", "experimental")
        
        assert hasattr(analysis, 'total_score'), "Analysis missing total_score"
        assert hasattr(analysis, 'component_scores'), "Analysis missing component_scores"
        assert hasattr(analysis, 'text_stats'), "Analysis missing text_stats"
        assert analysis.total_score > 0, "Total score should be positive"
        
        grade, description = rubric.grade_creativity_level(analysis.total_score)
        logger.info(f"✓ Text analysis: score={analysis.total_score:.3f}, grade={grade}")
        
        # Test tracker
        tracker = CreativityTracker()
        tracker.add_analysis(analysis, iteration=1)
        
        progress = tracker.get_progress_summary()
        assert progress['total_samples'] == 1, "Tracker should show 1 sample"
        
        logger.info("✓ Evaluation system working correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ Evaluation system test failed: {e}")
        return False

def test_configuration_validation():
    """Test configuration file validation."""
    try:
        from utils import ConfigValidator
        
        validator = ConfigValidator()
        env_dir = Path(__file__).parent
        
        # Validate all configuration files
        results = validator.validate_all_configs(env_dir)
        
        config_files = ['trainer_config.toml', 'orchestrator_config.toml', 'inference_config.toml']
        
        for config_file in config_files:
            if config_file in results:
                if results[config_file]['valid']:
                    logger.info(f"✓ {config_file} is valid")
                else:
                    logger.warning(f"⚠ {config_file} has validation issues: {results[config_file]['errors']}")
        
        logger.info(f"✓ Configuration validation completed. Overall valid: {results.get('overall_valid', False)}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration validation test failed: {e}")
        return False

def test_complete_pipeline():
    """Test a complete pipeline from environment creation to reward calculation."""
    try:
        from creativity_loader import load_creativity_environment
        
        # Create small test environment
        env = load_creativity_environment(
            num_train_samples=5,
            num_eval_samples=2,
            reward_weights={
                'w_entropy': 1.0,
                'w_distinct': 1.0,
                'w_uncommon': 0.8,
                'w_bigrams': 1.2,
                'w_sentence_len_var': 0.6,
                'w_word_len_var': 0.4,
                'w_sentence_end_var': 0.5,
            }
        )
        
        # Get a sample from the dataset
        sample = env.dataset[0]
        
        assert 'question' in sample, "Sample missing question field"
        assert 'task' in sample, "Sample missing task field"
        
        # Test reward calculation on mock completion
        mock_completion = "The ethereal whispers of creativity dance through labyrinthine corridors of imagination, painting gossamer dreams across the canvas of possibility. Each word becomes a crystalline fragment of thought, refracting meaning through prismatic layers of understanding."
        
        # Get rubric from environment
        rubric = env.rubric
        
        # Test reward calculation (simulated)
        reward = rubric.funcs[0](mock_completion, sample.get('info', {}))
        
        assert isinstance(reward, (int, float)), "Reward should be numeric"
        assert reward >= 0, "Reward should be non-negative"
        
        logger.info(f"✓ Complete pipeline test successful. Sample reward: {reward:.3f}")
        
        # Test format reward
        format_reward = rubric.funcs[1](mock_completion)
        assert isinstance(format_reward, (int, float)), "Format reward should be numeric"
        
        logger.info(f"✓ Format reward: {format_reward:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Complete pipeline test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    logger.info("Starting creativity environment integration tests...")
    
    tests = [
        ("Environment Loading", test_environment_loading),
        ("Reward Calculation", test_reward_calculation),
        ("Data Generation", test_data_generation),
        ("Evaluation System", test_evaluation_system),
        ("Configuration Validation", test_configuration_validation),
        ("Complete Pipeline", test_complete_pipeline),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("INTEGRATION TEST SUMMARY")
    logger.info("="*50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "PASSED" if passed_test else "FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All integration tests PASSED! Environment is ready for RLVR training.")
        return True
    else:
        logger.warning(f"⚠️  {total - passed} test(s) FAILED. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)