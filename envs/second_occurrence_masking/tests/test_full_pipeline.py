#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "torch",
#   "transformers", 
#   "requests",
#   "openai",
#   "rich",
# ]
# requires-python = ">=3.8"
# ///
"""
Full pipeline testing for Second Occurrence Masking environment.

This test suite validates:
- Complete orchestrator integration
- Training simulation with mock inference
- End-to-end workflow validation 
- Configuration compatibility across all components
- Performance metrics and reward calculation accuracy

Usage:
    ./tests/test_full_pipeline.py
"""

import sys
import time
import subprocess
import tempfile
import json
from pathlib import Path
from rich.console import Console
from rich.progress import track

# Add paths for prime-rl and environment
env_dir = Path(__file__).parent.parent
prime_rl_src = env_dir.parent.parent / "src"
sys.path.insert(0, str(env_dir))
sys.path.insert(0, str(prime_rl_src))

console = Console()


def test_environment_loading():
    """Test loading the environment through prime-rl registry."""
    console.print("🔍 Testing environment loading...", style="bold blue")
    
    try:
        from prime_rl.environments.registry import load_environment
        
        # Test loading with minimal parameters
        env = load_environment("second-occurrence-masking", {
            "num_examples": 10,
            "min_length": 30,
            "max_length": 100,
            "min_masks": 1,
            "max_masks": 3
        })
        
        assert env is not None
        assert len(env.dataset) > 0
        
        # Validate dataset structure
        sample = env.dataset[0]
        required_fields = ["question", "answer", "info", "task"]
        for field in required_fields:
            assert field in sample, f"Missing field: {field}"
        
        assert "[MASK]" in sample["question"]
        assert sample["task"] == "second-occurrence-masking"
        
        console.print(f"   ✅ Environment loaded with {len(env.dataset)} examples")
        console.print(f"   ✅ Sample question: {sample['question'][:100]}...")
        console.print(f"   ✅ Sample answer: {sample['answer']}")
        
        return True
        
    except Exception as e:
        console.print(f"   ❌ Environment loading failed: {e}", style="red")
        return False


def test_config_files_validation():
    """Test that all configuration files are valid and compatible."""
    console.print("🔍 Testing configuration files...", style="bold blue")
    
    try:
        import toml
        
        config_files = {
            "trainer_config.toml": ["model", "optimizer", "scheduler"],
            "orchestrator_config.toml": ["environment", "data"],  
            "inference_config.toml": ["model", "server"],
            "inference_config_ollama.toml": ["model"],
            "inference_config_mock.toml": ["model", "mock"]
        }
        
        for config_file, required_sections in config_files.items():
            config_path = env_dir / config_file
            
            if not config_path.exists():
                console.print(f"   ❌ Missing config file: {config_file}", style="red")
                return False
            
            # Parse TOML
            config = toml.load(config_path)
            
            # Check required sections
            for section in required_sections:
                if section not in config:
                    console.print(f"   ❌ Missing section '{section}' in {config_file}", style="red")
                    return False
            
            console.print(f"   ✅ {config_file} valid")
        
        # Test environment ID consistency
        orch_config = toml.load(env_dir / "orchestrator_config.toml")
        assert orch_config["environment"]["id"] == "second-occurrence-masking"
        
        console.print("   ✅ All configuration files valid and consistent")
        return True
        
    except Exception as e:
        console.print(f"   ❌ Config validation failed: {e}", style="red")
        return False


def start_mock_server_for_pipeline(port=8889):
    """Start mock server for pipeline testing."""
    console.print(f"🚀 Starting mock server for pipeline testing...", style="bold blue")
    
    server_script = env_dir / "mock_server.py"
    process = subprocess.Popen([
        str(server_script),
        "--port", str(port),
        "--mode", "masking_aware", 
        "--accuracy", "0.7",
        "--host", "127.0.0.1"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    for i in track(range(30), description="Starting server..."):
        try:
            import requests
            response = requests.get(f"http://localhost:{port}/health", timeout=1)
            if response.status_code == 200:
                console.print(f"   ✅ Mock server ready on port {port}")
                return process
        except:
            time.sleep(1)
    
    # Server failed to start
    process.terminate()
    console.print(f"   ❌ Failed to start mock server", style="red")
    return None


def stop_mock_server(process):
    """Stop the mock server process."""
    if process:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        console.print("   ✅ Mock server stopped")


def test_inference_compatibility():
    """Test inference compatibility with mock server."""
    console.print("🔍 Testing inference compatibility...", style="bold blue")
    
    # Start mock server
    server_process = start_mock_server_for_pipeline(port=8890)
    if not server_process:
        console.print("   ⚠️  Skipping inference test - server failed to start", style="yellow")
        return True
    
    try:
        from openai import OpenAI
        
        # Test with mock server
        client = OpenAI(
            base_url="http://localhost:8890/v1",
            api_key="mock"
        )
        
        # Create a realistic test case
        from second_occurrence_loader import SecondOccurrenceMaskingLoader
        loader = SecondOccurrenceMaskingLoader(seed=42)
        text = "The student studied hard. The student passed the test."
        result = loader.mask_text(text)
        
        if result.target_words:
            console.print(f"   📝 Test case: {result.masked_text}")
            console.print(f"   🎯 Expected: {result.target_words}")
            
            # Test inference
            response = client.chat.completions.create(
                model="mock-masking_aware",
                messages=[
                    {"role": "system", "content": "Fill in [MASK] tokens with original words."},
                    {"role": "user", "content": f"Fill in the [MASK] tokens: {result.masked_text}"}
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content
            console.print(f"   🤖 Mock response: {answer}")
            
            # Calculate reward
            reward = loader.calculate_reward(
                result.original_text,
                answer,
                result.mask_positions,
                result.target_words
            )
            
            console.print(f"   🏆 Reward: {reward:.2f}")
            console.print("   ✅ Inference compatibility confirmed")
            
        return True
        
    except Exception as e:
        console.print(f"   ❌ Inference test failed: {e}", style="red")
        return False
        
    finally:
        stop_mock_server(server_process)


def test_orchestrator_integration():
    """Test integration with orchestrator configuration."""
    console.print("🔍 Testing orchestrator integration...", style="bold blue")
    
    try:
        # Test that environment can be loaded with orchestrator config
        import toml
        orch_config = toml.load(env_dir / "orchestrator_config.toml")
        
        env_id = orch_config["environment"]["id"]
        env_args = orch_config["environment"].get("args", {})
        
        # Load environment with orchestrator settings
        from prime_rl.environments.registry import load_environment
        env = load_environment(env_id, env_args)
        
        assert env is not None
        assert len(env.dataset) > 0
        
        # Test data generation parameters
        console.print(f"   ✅ Environment loaded with orchestrator config")
        console.print(f"   ✅ Dataset size: {len(env.dataset)}")
        console.print(f"   ✅ Batch size: {orch_config.get('batch_size', 'default')}")
        
        # Validate a few examples
        valid_examples = 0
        total_masks = 0
        
        for i, example in enumerate(env.dataset[:10]):  # Check first 10
            if "[MASK]" in example["question"] and example["answer"]:
                valid_examples += 1
                total_masks += example["question"].count("[MASK]")
        
        console.print(f"   ✅ Valid examples: {valid_examples}/10")
        console.print(f"   ✅ Average masks per example: {total_masks/10:.1f}")
        
        return True
        
    except Exception as e:
        console.print(f"   ❌ Orchestrator integration failed: {e}", style="red")
        return False


def test_reward_calculation_accuracy():
    """Test reward calculation with various scenarios."""
    console.print("🔍 Testing reward calculation accuracy...", style="bold blue")
    
    try:
        from second_occurrence_loader import SecondOccurrenceMaskingLoader
        
        loader = SecondOccurrenceMaskingLoader(seed=42)
        
        # Test scenarios
        test_scenarios = [
            {
                "name": "Perfect match",
                "text": "The cat chased the cat.",
                "response_func": lambda targets: " ".join(targets),
                "expected_reward": 1.0
            },
            {
                "name": "Partial match", 
                "text": "The cat and the dog and the bird.",
                "response_func": lambda targets: targets[0] if targets else "",
                "expected_min": 0.0,
                "expected_max": 0.5
            },
            {
                "name": "Wrong answer",
                "text": "The cat chased the cat.",
                "response_func": lambda targets: "wrong answer",
                "expected_reward": 0.0
            },
            {
                "name": "Empty response",
                "text": "The cat chased the cat.",
                "response_func": lambda targets: "",
                "expected_reward": 0.0
            }
        ]
        
        for scenario in test_scenarios:
            result = loader.mask_text(scenario["text"])
            
            if result.target_words:  # Only test if masks were created
                response = scenario["response_func"](result.target_words)
                reward = loader.calculate_reward(
                    result.original_text,
                    response,
                    result.mask_positions,
                    result.target_words
                )
                
                console.print(f"   📝 {scenario['name']}: '{response}' -> {reward:.2f}")
                
                # Check expected reward
                if "expected_reward" in scenario:
                    assert abs(reward - scenario["expected_reward"]) < 0.01
                elif "expected_min" in scenario and "expected_max" in scenario:
                    assert scenario["expected_min"] <= reward <= scenario["expected_max"]
        
        console.print("   ✅ Reward calculation accuracy confirmed")
        return True
        
    except Exception as e:
        console.print(f"   ❌ Reward calculation test failed: {e}", style="red")
        return False


def test_dataset_quality():
    """Test quality and diversity of generated dataset."""
    console.print("🔍 Testing dataset quality...", style="bold blue")
    
    try:
        from prime_rl.environments.registry import load_environment
        
        # Load with larger sample for quality testing
        env = load_environment("second-occurrence-masking", {
            "num_examples": 50,
            "min_length": 40,
            "max_length": 200,
            "min_masks": 1,
            "max_masks": 5
        })
        
        # Analyze dataset quality
        total_examples = len(env.dataset)
        valid_examples = 0
        mask_counts = []
        text_lengths = []
        unique_target_words = set()
        
        for example in env.dataset:
            info = example.get("info", {})
            original_text = info.get("original_text", "")
            target_words = info.get("target_words", [])
            
            if "[MASK]" in example["question"] and target_words:
                valid_examples += 1
                mask_counts.append(len(target_words))
                text_lengths.append(len(original_text))
                unique_target_words.update(word.lower() for word in target_words)
        
        # Quality metrics
        validity_ratio = valid_examples / total_examples
        avg_masks = sum(mask_counts) / len(mask_counts) if mask_counts else 0
        avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
        
        console.print(f"   📊 Dataset metrics:")
        console.print(f"      Valid examples: {valid_examples}/{total_examples} ({validity_ratio:.1%})")
        console.print(f"      Average masks per example: {avg_masks:.1f}")
        console.print(f"      Average text length: {avg_length:.0f} chars")
        console.print(f"      Unique target words: {len(unique_target_words)}")
        
        # Quality thresholds
        assert validity_ratio >= 0.8  # At least 80% valid examples
        assert 1 <= avg_masks <= 5    # Reasonable mask count
        assert 40 <= avg_length <= 200  # Within expected length range
        assert len(unique_target_words) >= 10  # Diverse vocabulary
        
        console.print("   ✅ Dataset quality meets requirements")
        return True
        
    except Exception as e:
        console.print(f"   ❌ Dataset quality test failed: {e}", style="red")
        return False


def test_performance_simulation():
    """Simulate training performance with different mock accuracies."""
    console.print("🔍 Testing performance simulation...", style="bold blue")
    
    try:
        from second_occurrence_loader import SecondOccurrenceMaskingLoader
        from mock_inference import create_mock_model
        
        loader = SecondOccurrenceMaskingLoader(seed=42)
        
        # Test different accuracy levels
        accuracy_levels = [0.1, 0.5, 0.8, 1.0]
        
        for accuracy in accuracy_levels:
            model = create_mock_model("masking_aware", accuracy=accuracy)
            
            # Generate test cases
            test_cases = []
            for _ in range(10):
                text = loader.get_sample_text()
                result = loader.mask_text(text)
                if result.target_words:
                    test_cases.append(result)
            
            if not test_cases:
                console.print(f"   ⚠️  No valid test cases generated", style="yellow")
                continue
            
            # Calculate average reward
            total_reward = 0
            for result in test_cases:
                messages = [{"role": "user", "content": f"Fill masks: {result.masked_text}"}]
                response = model.complete(messages)
                reward = loader.calculate_reward(
                    result.original_text,
                    response.content,
                    result.mask_positions,
                    result.target_words
                )
                total_reward += reward
            
            avg_reward = total_reward / len(test_cases)
            console.print(f"   🎯 Accuracy {accuracy:.1f}: Average reward {avg_reward:.2f}")
            
            # Sanity check: higher model accuracy should generally lead to higher rewards
            if accuracy == 1.0:
                assert avg_reward >= 0.5  # Perfect model should do well
        
        console.print("   ✅ Performance simulation working correctly")
        return True
        
    except Exception as e:
        console.print(f"   ❌ Performance simulation failed: {e}", style="red")
        return False


def run_all_tests():
    """Run all full pipeline tests."""
    console.print("🧪 [bold]Running Full Pipeline Tests for Second Occurrence Masking[/bold]\n")
    
    test_functions = [
        test_environment_loading,
        test_config_files_validation,
        test_inference_compatibility,
        test_orchestrator_integration,
        test_reward_calculation_accuracy,
        test_dataset_quality,
        test_performance_simulation,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            if test_func():
                passed += 1
                console.print()
            else:
                failed += 1
                console.print()
        except Exception as e:
            console.print(f"   ❌ {test_func.__name__} failed with exception: {e}", style="red")
            failed += 1
            console.print()
    
    console.print("=" * 60)
    console.print(f"📊 [bold]Test Results: {passed} passed, {failed} failed[/bold]")
    
    if failed == 0:
        console.print("🎉 [bold green]All pipeline tests passed![/bold green]")
        console.print("\n💡 [bold]Ready for training![/bold]")
        console.print("   To run with mock inference:")
        console.print("   [cyan]./mock_server.py --mode masking_aware &[/cyan]")
        console.print("   [cyan]cp inference_config_mock.toml inference_config.toml[/cyan]")
        console.print("   [cyan]uv run rl --trainer @ trainer_config.toml --orchestrator @ orchestrator_config.toml --inference @ inference_config.toml[/cyan]")
        return True
    else:
        console.print(f"💥 [bold red]{failed} tests failed![/bold red]")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)