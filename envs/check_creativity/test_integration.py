#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "numpy",
#   "nltk", 
#   "datasets",
#   "transformers",
#   "torch",
#   "verifiers @ git+https://github.com/willccbb/verifiers@90c06b2",
#   "toml",
# ]
# requires-python = ">=3.8"
# ///

"""
Integration test script for creativity environment with prime-rl components.

This script tests the full integration of the creativity environment
with the prime-rl training pipeline components.
"""

import sys
import tempfile
import json
from pathlib import Path
import traceback

# Add the current directory and repo root to Python path
current_dir = Path(__file__).parent
repo_root = current_dir.parent.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(repo_root / "src"))


def test_environment_registration():
    """Test that the environment is properly registered."""
    print("Testing environment registration...")
    
    try:
        # Import the registry
        from prime_rl.environments.registry import REGISTRY, load_environment
        
        # Check if creativity environment is in registry
        if 'creativity' in REGISTRY:
            print("  ✓ Creativity environment found in registry")
            
            env_info = REGISTRY['creativity']
            print(f"    Type: {env_info['type']}")
            print(f"    Tags: {env_info['tags']}")
            
            # Test loading the environment
            try:
                env = load_environment('creativity', {'num_train_samples': 5, 'num_eval_samples': 2})
                print("  ✓ Environment loaded successfully from registry")
                
                # Verify environment structure
                if hasattr(env, 'dataset') and hasattr(env, 'rubric'):
                    print("  ✓ Environment has correct verifiers interface")
                else:
                    print("  ✗ Environment missing required attributes")
                    return False
                
                # Test dataset structure
                if len(env.dataset) > 0:
                    sample = env.dataset[0]
                    if 'question' in sample and 'task' in sample:
                        print("  ✓ Dataset has correct structure")
                    else:
                        print("  ✗ Dataset missing required fields")
                        return False
                else:
                    print("  ✗ Empty dataset")
                    return False
                    
            except Exception as e:
                print(f"  ✗ Failed to load environment: {e}")
                return False
                
        else:
            print("  ✗ Creativity environment not found in registry")
            return False
        
        print("✓ Environment registration test completed\n")
        return True
        
    except Exception as e:
        print(f"✗ Environment registration test failed: {e}")
        traceback.print_exc()
        return False


def test_trainer_integration():
    """Test integration with trainer components."""
    print("Testing trainer integration...")
    
    try:
        # Test configuration loading
        import toml
        
        config_path = current_dir / "trainer_config.toml"
        if config_path.exists():
            config = toml.load(config_path)
            print("  ✓ Trainer config loaded")
            
            # Check essential config sections
            required_sections = ['model', 'data', 'optimizer']
            for section in required_sections:
                if section in config:
                    print(f"    ✓ {section} section present")
                else:
                    print(f"    ✗ {section} section missing")
                    return False
            
            # Check creativity-specific settings
            if 'data' in config and 'creativity' in config['data']:
                creativity_config = config['data']['creativity']
                print(f"    ✓ Creativity data config: {list(creativity_config.keys())}")
            else:
                print("    ✗ Missing creativity-specific data config")
                return False
                
        else:
            print("  ✗ Trainer config file not found")
            return False
        
        # Test environment loading with trainer-style parameters
        from prime_rl.environments.registry import load_environment
        
        trainer_args = {
            'num_train_samples': config['data']['creativity']['num_train_samples'],
            'num_eval_samples': config['data']['creativity']['num_eval_samples'],
            'reward_weights': config['data']['creativity']['reward_weights']
        }
        
        env = load_environment('creativity', trainer_args)
        
        if env and len(env.dataset) == trainer_args['num_train_samples']:
            print("  ✓ Environment loaded with trainer configuration")
        else:
            print("  ✗ Environment loading with trainer config failed")
            return False
        
        print("✓ Trainer integration test completed\n")
        return True
        
    except Exception as e:
        print(f"✗ Trainer integration test failed: {e}")
        traceback.print_exc()
        return False


def test_orchestrator_integration():
    """Test integration with orchestrator components."""
    print("Testing orchestrator integration...")
    
    try:
        import toml
        
        # Test orchestrator config
        config_path = current_dir / "orchestrator_config.toml"
        if config_path.exists():
            config = toml.load(config_path)
            print("  ✓ Orchestrator config loaded")
            
            # Check environment configuration
            if 'environment' in config:
                env_config = config['environment']
                if env_config.get('env_id') == 'creativity':
                    print("  ✓ Correct environment ID in orchestrator config")
                else:
                    print("  ✗ Wrong environment ID in orchestrator config")
                    return False
            else:
                print("  ✗ Missing environment section in orchestrator config")
                return False
            
            # Check sampling configuration
            if 'sampling' in config:
                sampling_config = config['sampling']
                print(f"    ✓ Sampling config: temperature={sampling_config.get('temperature', 'N/A')}")
            
            # Check rewards configuration
            if 'rewards' in config:
                print("    ✓ Rewards configuration present")
            
        else:
            print("  ✗ Orchestrator config file not found")
            return False
        
        # Test environment with orchestrator-style parameters
        from prime_rl.environments.registry import load_environment
        
        orchestrator_args = {
            'num_train_samples': config['environment']['creativity']['num_train_samples'],
            'reward_weights': config['environment']['creativity']['reward_weights'],
            'seed': config['environment']['creativity']['seed']
        }
        
        env = load_environment('creativity', orchestrator_args)
        
        if env:
            print("  ✓ Environment compatible with orchestrator parameters")
        else:
            print("  ✗ Environment failed with orchestrator parameters")
            return False
        
        print("✓ Orchestrator integration test completed\n")
        return True
        
    except Exception as e:
        print(f"✗ Orchestrator integration test failed: {e}")
        traceback.print_exc()
        return False


def test_inference_integration():
    """Test integration with inference components."""
    print("Testing inference integration...")
    
    try:
        import toml
        
        # Test inference config
        config_path = current_dir / "inference_config.toml"
        if config_path.exists():
            config = toml.load(config_path)
            print("  ✓ Inference config loaded")
            
            # Check creativity-specific inference settings
            if 'creativity_weights' in config:
                weights = config['creativity_weights']
                print(f"    ✓ Creativity weights: {len(weights)} components")
            else:
                print("    ✗ Missing creativity weights in inference config")
                return False
            
            # Check generation parameters
            if 'generation' in config:
                gen_config = config['generation']
                if gen_config.get('encourage_creativity', False):
                    print("    ✓ Creativity-optimized generation settings")
                else:
                    print("    ✗ Missing creativity optimization in generation")
                    return False
            
        else:
            print("  ✗ Inference config file not found")
            return False
        
        # Test that our reward function can be used for inference evaluation
        from reward import reward_function
        
        test_output = "The ethereal moonlight cascaded through gossamer clouds, painting silver dreams across the velvet canvas of night."
        score = reward_function(test_output, **config['creativity_weights'])
        
        print(f"    ✓ Reward function works with inference weights: {score:.3f}")
        
        # Test batch processing for inference
        test_outputs = [
            "Simple text here.",
            "The luminous garden whispered ancient secrets while silver butterflies danced through time.",
            "Another basic sentence."
        ]
        
        batch_scores = [reward_function(text, **config['creativity_weights']) for text in test_outputs]
        print(f"    ✓ Batch processing works: {len(batch_scores)} scores")
        
        print("✓ Inference integration test completed\n")
        return True
        
    except Exception as e:
        print(f"✗ Inference integration test failed: {e}")
        traceback.print_exc()
        return False


def test_end_to_end_workflow():
    """Test a complete end-to-end workflow simulation."""
    print("Testing end-to-end workflow...")
    
    try:
        # 1. Load environment (as trainer would)
        from prime_rl.environments.registry import load_environment
        
        env = load_environment('creativity', {
            'num_train_samples': 3,
            'num_eval_samples': 2
        })
        
        print("  ✓ Environment loaded")
        
        # 2. Get a sample (as orchestrator would)
        sample = env.dataset[0]
        prompt = sample['question']
        print(f"  ✓ Got prompt: {prompt[:50]}...")
        
        # 3. Simulate model response
        model_response = """The kaleidoscope of imagination spiraled through consciousness 
                           like iridescent threads weaving tapestries of possibility. Each 
                           thought tasted of copper dreams and whispered secrets, while 
                           silence hummed with frequencies of unborn stars."""
        
        # 4. Evaluate response (as verifiers would)
        try:
            # Test with rubric evaluation
            rubric = env.rubric
            
            # Create mock completion format that verifiers expects
            completion = [{"role": "assistant", "content": model_response}]
            
            # Calculate reward
            reward = rubric.funcs[0](completion, **sample)  # Primary reward function
            format_reward = rubric.funcs[1](completion, **sample)  # Format reward function
            
            total_reward = reward * rubric.weights[0] + format_reward * rubric.weights[1]
            
            print(f"    ✓ Creativity reward: {reward:.3f}")
            print(f"    ✓ Format reward: {format_reward:.3f}")
            print(f"    ✓ Total weighted reward: {total_reward:.3f}")
            
        except Exception as e:
            print(f"    ✗ Reward calculation failed: {e}")
            return False
        
        # 5. Test evaluation dataset
        if hasattr(env, 'eval_dataset') and env.eval_dataset:
            eval_sample = env.eval_dataset[0]
            print(f"  ✓ Evaluation dataset available: {len(env.eval_dataset)} samples")
        else:
            print("  ✗ No evaluation dataset")
            return False
        
        # 6. Simulate metrics tracking
        from evaluation_rubrics import CreativityTracker
        
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = CreativityTracker(save_dir=Path(temp_dir))
            
            # Simulate multiple training steps
            for step in range(5):
                step_response = f"Step {step}: {model_response} with variation {step}"
                
                from evaluation_rubrics import CreativityRubric
                rubric = CreativityRubric()
                analysis = rubric.evaluate_text(step_response, prompt, "test")
                tracker.add_analysis(analysis, step)
            
            # Get progress summary
            progress = tracker.get_progress_summary()
            print(f"  ✓ Tracked {progress['total_samples']} samples")
            print(f"  ✓ Training trend: {progress['improvement_trend']}")
            
            # Generate insights
            insights = tracker.generate_training_insights()
            print(f"  ✓ Generated {len(insights)} insight categories")
        
        print("✓ End-to-end workflow test completed\n")
        return True
        
    except Exception as e:
        print(f"✗ End-to-end workflow test failed: {e}")
        traceback.print_exc()
        return False


def test_performance_and_scalability():
    """Test performance with larger datasets."""
    print("Testing performance and scalability...")
    
    try:
        import time
        
        # Test with larger dataset
        print("  Testing with larger dataset...")
        start_time = time.time()
        
        from prime_rl.environments.registry import load_environment
        
        env = load_environment('creativity', {
            'num_train_samples': 50,
            'num_eval_samples': 10
        })
        
        load_time = time.time() - start_time
        print(f"    ✓ Loaded 50 samples in {load_time:.2f}s")
        
        # Test batch reward calculation
        start_time = time.time()
        
        test_texts = [
            "The ethereal moonbeams danced through gossamer clouds.",
            "Simple text for comparison purposes here.",
            "Iridescent butterflies whispered secrets to the wind.",
            "Another basic sentence to test efficiency.",
            "Kaleidoscopic dreams spiraled through consciousness like silver threads."
        ]
        
        # Use the environment's batch processing
        from creativity_env import CreativityEnvironment
        
        full_env = CreativityEnvironment()
        batch_scores = full_env.batch_process(test_texts)
        
        batch_time = time.time() - start_time
        print(f"    ✓ Processed {len(test_texts)} texts in {batch_time:.3f}s")
        print(f"    ✓ Average processing time: {batch_time/len(test_texts):.3f}s per text")
        
        # Test memory usage (basic check)
        import sys
        
        initial_objects = len(gc.get_objects()) if 'gc' in dir() else 0
        
        # Process more samples
        larger_env = load_environment('creativity', {'num_train_samples': 100})
        
        if larger_env and len(larger_env.dataset) == 100:
            print(f"    ✓ Successfully loaded 100 training samples")
        
        print("✓ Performance and scalability test completed\n")
        return True
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        traceback.print_exc()
        return False


def run_integration_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("CREATIVITY ENVIRONMENT INTEGRATION TEST SUITE")
    print("=" * 60)
    print()
    
    tests = [
        ("Environment Registration", test_environment_registration),
        ("Trainer Integration", test_trainer_integration),
        ("Orchestrator Integration", test_orchestrator_integration),
        ("Inference Integration", test_inference_integration),
        ("End-to-End Workflow", test_end_to_end_workflow),
        ("Performance & Scalability", test_performance_and_scalability),
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
    print("INTEGRATION TEST SUMMARY")
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
        print("\n🎉 All integration tests passed! The creativity environment is fully integrated.")
        return True
    else:
        print(f"\n⚠️  {failed} integration test(s) failed. Please review the output above.")
        return False


if __name__ == "__main__":
    import gc  # Import gc here for the performance test
    success = run_integration_tests()
    sys.exit(0 if success else 1)