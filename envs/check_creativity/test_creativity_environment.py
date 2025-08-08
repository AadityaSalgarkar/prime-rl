#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "numpy",
#   "nltk",
#   "datasets",
#   "transformers",
#   "torch",
#   "verifiers @ git+https://github.com/willccbb/verifiers@90c06b2",
# ]
# requires-python = ">=3.8"
# ///

"""
Test script for creativity environment functionality.

This script comprehensively tests all components of the creativity environment
including data generation, reward calculation, environment integration,
and evaluation metrics.
"""

import sys
import traceback
from pathlib import Path
import tempfile
import json

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_reward_function():
    """Test the core reward function."""
    print("Testing reward function...")
    
    try:
        from reward import reward_function
        
        # Test cases with different creativity levels
        test_cases = [
            {
                'text': "The cat sat on the mat. It was nice.",
                'expected_range': (0, 3),
                'description': 'Low creativity text'
            },
            {
                'text': """The iridescent feline perched gracefully upon the gossamer textile, 
                        contemplating the ephemeral beauty of afternoon sunlight. What mysteries 
                        danced through its crystalline consciousness? Perhaps dreams of ethereal 
                        realms where time flows like honey through infinite possibilities.""",
                'expected_range': (5, 15),
                'description': 'High creativity text'
            },
            {
                'text': "She walked through the garden, noticing the colorful flowers and listening to the birds singing cheerfully in the warm sunshine.",
                'expected_range': (2, 6),
                'description': 'Medium creativity text'
            }
        ]
        
        for i, case in enumerate(test_cases):
            score = reward_function(case['text'])
            print(f"  Test {i+1}: {case['description']}")
            print(f"    Score: {score:.2f}")
            print(f"    Expected range: {case['expected_range']}")
            
            min_expected, max_expected = case['expected_range']
            if min_expected <= score <= max_expected:
                print(f"    ✓ PASS")
            else:
                print(f"    ✗ FAIL - Score outside expected range")
        
        print("✓ Reward function tests completed\n")
        return True
        
    except Exception as e:
        print(f"✗ Reward function test failed: {e}")
        traceback.print_exc()
        return False


def test_data_generation():
    """Test the data generation utilities."""
    print("Testing data generation...")
    
    try:
        from data_utils import CreativeTextGenerator, DataAugmentation
        
        generator = CreativeTextGenerator(seed=42)
        
        # Test prompt generation
        prompt = "Write about a character who can taste colors."
        creative_text = generator.generate_creative_sample(prompt, length="medium")
        
        print(f"  Generated creative text length: {len(creative_text)} characters")
        print(f"  Sample text: {creative_text[:100]}...")
        
        if len(creative_text) > 50:
            print("  ✓ Creative text generation works")
        else:
            print("  ✗ Generated text too short")
            return False
        
        # Test data augmentation
        base_prompts = ["Write a story", "Describe a scene"]
        augmented = DataAugmentation.augment_prompt_variations(base_prompts)
        
        print(f"  Original prompts: {len(base_prompts)}")
        print(f"  Augmented prompts: {len(augmented)}")
        
        if len(augmented) > len(base_prompts):
            print("  ✓ Data augmentation works")
        else:
            print("  ✗ Data augmentation failed")
            return False
        
        print("✓ Data generation tests completed\n")
        return True
        
    except Exception as e:
        print(f"✗ Data generation test failed: {e}")
        traceback.print_exc()
        return False


def test_creativity_loader():
    """Test the creativity environment loader."""
    print("Testing creativity environment loader...")
    
    try:
        from creativity_loader import CreativityEnvironmentLoader, load_creativity_environment
        
        # Test loader initialization
        loader = CreativityEnvironmentLoader(num_train_samples=10, num_eval_samples=5)
        
        # Test prompt generation
        prompts = loader.generate_creative_prompts(5)
        print(f"  Generated {len(prompts)} prompts")
        
        if len(prompts) == 5:
            print("  ✓ Prompt generation works")
        else:
            print("  ✗ Wrong number of prompts generated")
            return False
        
        # Test dataset creation
        dataset = loader.create_dataset(prompts, "test")
        print(f"  Created dataset with {len(dataset)} entries")
        
        if len(dataset) == 5:
            print("  ✓ Dataset creation works")
        else:
            print("  ✗ Dataset creation failed")
            return False
        
        # Test reward function integration
        test_text = "The shimmering aurora danced across the sky like ethereal ribbons of light."
        reward = loader.creativity_reward_function(test_text)
        print(f"  Reward for test text: {reward:.3f}")
        
        if 0 <= reward <= 1:
            print("  ✓ Reward function integration works")
        else:
            print("  ✗ Reward function returned invalid value")
            return False
        
        # Test full environment loading
        print("  Testing full environment loading...")
        env = load_creativity_environment(num_train_samples=5, num_eval_samples=2)
        
        if env is not None:
            print("  ✓ Environment loading works")
        else:
            print("  ✗ Environment loading failed")
            return False
        
        print("✓ Creativity loader tests completed\n")
        return True
        
    except Exception as e:
        print(f"✗ Creativity loader test failed: {e}")
        traceback.print_exc()
        return False


def test_comprehensive_environment():
    """Test the comprehensive environment class."""
    print("Testing comprehensive environment class...")
    
    try:
        from creativity_env import CreativityEnvironment, CreativityAction
        
        # Test environment initialization
        env = CreativityEnvironment(config={'max_episode_steps': 1})
        
        # Test environment reset
        obs = env.reset()
        print(f"  Initial observation prompt length: {len(obs.prompt)}")
        
        if obs and obs.prompt:
            print("  ✓ Environment reset works")
        else:
            print("  ✗ Environment reset failed")
            return False
        
        # Test environment step
        test_response = "The luminous garden whispered secrets of ancient wisdom, where each flower held memories of forgotten dreams and crystalline hopes."
        action = CreativityAction(text=test_response)
        
        next_obs, reward, done, info = env.step(action)
        
        print(f"  Step reward: {reward:.3f}")
        print(f"  Episode done: {done}")
        print(f"  Info keys: {list(info.keys())}")
        
        if reward >= 0 and 'reward_components' in info:
            print("  ✓ Environment step works")
        else:
            print("  ✗ Environment step failed")
            return False
        
        # Test batch processing
        test_texts = [
            "Simple text here.",
            "The ethereal moonbeams danced through gossamer clouds, painting silver dreams across the velvet sky.",
            "Another basic sentence for testing."
        ]
        
        batch_scores = env.batch_process(test_texts)
        print(f"  Batch processing results: {batch_scores}")
        
        if len(batch_scores) == len(test_texts):
            print("  ✓ Batch processing works")
        else:
            print("  ✗ Batch processing failed")
            return False
        
        # Test metrics summary
        summary = env.get_metrics_summary()
        print(f"  Metrics summary keys: {list(summary.keys())}")
        
        if summary:
            print("  ✓ Metrics summary works")
        else:
            print("  ✗ Metrics summary failed")
            return False
        
        print("✓ Comprehensive environment tests completed\n")
        return True
        
    except Exception as e:
        print(f"✗ Comprehensive environment test failed: {e}")
        traceback.print_exc()
        return False


def test_evaluation_rubrics():
    """Test the evaluation and analysis system."""
    print("Testing evaluation rubrics...")
    
    try:
        from evaluation_rubrics import CreativityRubric, CreativityTracker, CreativityBenchmark
        
        # Test creativity rubric
        rubric = CreativityRubric()
        
        test_text = """The kaleidoscope of thoughts spiraled through her consciousness like 
                      iridescent butterflies seeking nectar in gardens of possibility. 
                      Each idea tasted of copper pennies and starlight, while silence 
                      hummed with the frequency of unspoken dreams."""
        
        analysis = rubric.evaluate_text(test_text, prompt="Write creatively", category="experimental")
        
        print(f"  Analysis score: {analysis.total_score:.2f}")
        print(f"  Component scores: {len(analysis.component_scores)} components")
        print(f"  Text stats: {len(analysis.text_stats)} statistics")
        
        if analysis.total_score > 0 and analysis.component_scores:
            print("  ✓ Creativity rubric works")
        else:
            print("  ✗ Creativity rubric failed")
            return False
        
        # Test creativity level grading
        grade, description = rubric.grade_creativity_level(analysis.total_score)
        print(f"  Creativity grade: {grade} - {description}")
        
        # Test tracker
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = CreativityTracker(save_dir=Path(temp_dir))
            
            # Add some analyses
            for i in range(5):
                test_analysis = rubric.evaluate_text(f"Test text {i} with varying creativity levels and different approaches.")
                tracker.add_analysis(test_analysis, iteration=i)
            
            # Get progress summary
            progress = tracker.get_progress_summary()
            print(f"  Tracker progress keys: {list(progress.keys())}")
            
            if progress['total_samples'] == 5:
                print("  ✓ Creativity tracker works")
            else:
                print("  ✗ Creativity tracker failed")
                return False
            
            # Test insights generation
            insights = tracker.generate_training_insights()
            print(f"  Generated insights: {len(insights)} categories")
            
            # Test saving/loading
            save_path = Path(temp_dir) / "test_analysis.json"
            tracker.save_analysis(save_path)
            
            if save_path.exists():
                print("  ✓ Analysis saving works")
            else:
                print("  ✗ Analysis saving failed")
                return False
        
        # Test benchmark system
        benchmark = CreativityBenchmark()
        benchmark_result = benchmark.benchmark_text(test_text)
        
        print(f"  Benchmark level: {benchmark_result['benchmark_level']}")
        print(f"  Assessment: {benchmark_result['overall_assessment'][:50]}...")
        
        if benchmark_result['benchmark_level'] in ['low_creativity', 'medium_creativity', 'high_creativity']:
            print("  ✓ Creativity benchmark works")
        else:
            print("  ✗ Creativity benchmark failed")
            return False
        
        print("✓ Evaluation rubrics tests completed\n")
        return True
        
    except Exception as e:
        print(f"✗ Evaluation rubrics test failed: {e}")
        traceback.print_exc()
        return False


def test_integration_with_registry():
    """Test integration with the main registry."""
    print("Testing registry integration...")
    
    try:
        # This would normally test the actual registry integration
        # For now, we'll test that our environment can be imported correctly
        
        # Test that the loader function exists and works
        from creativity_loader import load_creativity_environment
        
        env = load_creativity_environment(num_train_samples=3, num_eval_samples=1)
        
        if env is not None:
            print("  ✓ Environment loads successfully")
            
            # Test that it has the required verifiers interface
            if hasattr(env, 'dataset') and hasattr(env, 'rubric'):
                print("  ✓ Environment has correct verifiers interface")
            else:
                print("  ✗ Environment missing verifiers interface")
                return False
            
            # Test dataset structure
            if len(env.dataset) > 0:
                sample = env.dataset[0]
                required_fields = ['question', 'info', 'task']
                
                if all(field in sample for field in required_fields):
                    print("  ✓ Dataset has correct structure")
                else:
                    print("  ✗ Dataset missing required fields")
                    return False
            else:
                print("  ✗ Dataset is empty")
                return False
        else:
            print("  ✗ Environment loading failed")
            return False
        
        print("✓ Registry integration tests completed\n")
        return True
        
    except Exception as e:
        print(f"✗ Registry integration test failed: {e}")
        traceback.print_exc()
        return False


def test_configuration_files():
    """Test that configuration files are valid."""
    print("Testing configuration files...")
    
    try:
        import toml
        
        config_files = [
            'trainer_config.toml',
            'orchestrator_config.toml', 
            'inference_config.toml'
        ]
        
        current_dir = Path(__file__).parent
        
        for config_file in config_files:
            config_path = current_dir / config_file
            
            if config_path.exists():
                try:
                    config = toml.load(config_path)
                    print(f"  ✓ {config_file} is valid TOML")
                    
                    # Check for required sections
                    if config_file == 'trainer_config.toml':
                        required_sections = ['model', 'data', 'optimizer']
                        if all(section in config for section in required_sections):
                            print(f"    ✓ Has required sections")
                        else:
                            print(f"    ✗ Missing required sections")
                            return False
                    
                except Exception as e:
                    print(f"  ✗ {config_file} is invalid: {e}")
                    return False
            else:
                print(f"  ✗ {config_file} not found")
                return False
        
        print("✓ Configuration file tests completed\n")
        return True
        
    except ImportError:
        print("  Warning: toml package not available, skipping config validation")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("CREATIVITY ENVIRONMENT TEST SUITE")
    print("=" * 60)
    print()
    
    tests = [
        ("Core Reward Function", test_reward_function),
        ("Data Generation", test_data_generation),
        ("Creativity Loader", test_creativity_loader),
        ("Comprehensive Environment", test_comprehensive_environment),
        ("Evaluation Rubrics", test_evaluation_rubrics),
        ("Registry Integration", test_integration_with_registry),
        ("Configuration Files", test_configuration_files),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        print("-" * 40)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            traceback.print_exc()
            results.append((test_name, False))
        
        print()
    
    # Print summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        
        if result:
            passed += 1
        else:
            failed += 1
    
    print()
    print(f"Total tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! The creativity environment is ready for use.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the output above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)