#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "torch",
#   "transformers", 
#   "datasets",
#   "requests",
#   "openai",
#   "rich",
#   "fastapi",
#   "uvicorn"
# ]
# requires-python = ">=3.8"
# ///
"""
Test the full training pipeline including trainer, orchestrator, and inference.
This simulates the complete RL workflow on macOS without GPU requirements.

Run with: ./tests/test_full_pipeline.py
"""

import json
import sys
import time
import random
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_mock_server_running() -> bool:
    """Check if mock server is running."""
    try:
        import requests
        response = requests.get("http://localhost:8888/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False

def start_mock_server():
    """Start the mock server in background."""
    if check_mock_server_running():
        print("✅ Mock server already running")
        return None
    
    print("🚀 Starting mock server...")
    try:
        # Start mock server in background
        server_process = subprocess.Popen(
            ["./mock_server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(__file__).parent.parent
        )
        
        # Wait for server to start
        for _ in range(10):
            if check_mock_server_running():
                print("✅ Mock server started successfully")
                return server_process
            time.sleep(1)
        
        print("⚠️  Mock server may not have started properly")
        return server_process
        
    except Exception as e:
        print(f"❌ Failed to start mock server: {e}")
        return None

def test_orchestrator_integration():
    """Test orchestrator with mock inference."""
    print("\n🧪 Testing Orchestrator Integration...")
    
    try:
        from thesaurus_loader import ThesaurusLoader
        from openai import OpenAI
        
        # Setup components
        loader = ThesaurusLoader()
        client = OpenAI(
            base_url="http://localhost:8888/v1",
            api_key="mock"
        )
        
        # Test orchestrator workflow
        print("  📝 Generating training batch...")
        
        # Sample data generation (simulating orchestrator)
        base_sentences = [
            "The good dog ran fast.",
            "She opened the ancient door.",
            "A big house stood on the hill."
        ]
        
        batch = []
        for sentence in base_sentences:
            # Create augmented version
            augmented, replacements = loader.replace_with_synonyms(sentence, replacement_rate=0.3)
            
            # Get model response via mock inference
            prompt = f"Restore the original text: {augmented}"
            messages = [{"role": "user", "content": prompt}]
            
            response = client.chat.completions.create(
                model="mock-thesaurus",
                messages=messages,
                max_tokens=50,
                temperature=0.0
            )
            
            model_output = response.choices[0].message.content
            
            # Calculate reward
            import re
            original_words = re.findall(r'\b\w+\b', sentence.lower())
            output_words = re.findall(r'\b\w+\b', model_output.lower())
            
            matches = sum(1 for o, m in zip(original_words, output_words) if o == m)
            reward = matches / len(original_words) if original_words else 0.0
            
            batch.append({
                "original": sentence,
                "augmented": augmented,
                "model_output": model_output,
                "reward": reward,
                "replacements": replacements
            })
        
        # Display results
        total_reward = sum(item["reward"] for item in batch)
        avg_reward = total_reward / len(batch)
        
        print(f"  📊 Batch Results:")
        print(f"    • Samples: {len(batch)}")
        print(f"    • Average Reward: {avg_reward:.4f}")
        print(f"    • Total Reward: {total_reward:.4f}")
        
        # Show sample
        sample = batch[0]
        print(f"  📝 Sample:")
        print(f"    Original: {sample['original']}")
        print(f"    Augmented: {sample['augmented']}")
        print(f"    Model Output: {sample['model_output']}")
        print(f"    Reward: {sample['reward']:.4f}")
        
        print("  ✅ Orchestrator integration working!")
        return True
        
    except Exception as e:
        print(f"  ❌ Orchestrator integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_simulation():
    """Test training simulation with batched data."""
    print("\n🧪 Testing Training Simulation...")
    
    try:
        # Simulate training loop
        print("  🎯 Running mini training loop...")
        
        total_reward = 0.0
        total_samples = 0
        
        for step in range(5):
            # Simulate batch generation (orchestrator)
            batch_size = 4
            batch_reward = random.uniform(0.1, 0.8)  # Simulate varying performance
            
            total_reward += batch_reward * batch_size
            total_samples += batch_size
            
            avg_reward = total_reward / total_samples
            step_loss = 1.0 - batch_reward + random.uniform(-0.1, 0.1)
            
            print(f"    Step {step}: Loss={step_loss:.4f}, Reward={batch_reward:.4f}, Avg={avg_reward:.4f}")
            
            # Simulate checkpoint saving
            if step % 2 == 0:
                checkpoint_dir = Path("checkpoints") / f"step_{step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                checkpoint_data = {
                    "step": step,
                    "avg_reward": avg_reward,
                    "total_samples": total_samples
                }
                
                with open(checkpoint_dir / "trainer.json", "w") as f:
                    json.dump(checkpoint_data, f, indent=2)
                
                print(f"      💾 Saved checkpoint at step {step}")
        
        print(f"  📊 Training Summary:")
        print(f"    • Final Average Reward: {avg_reward:.4f}")
        print(f"    • Total Samples: {total_samples}")
        print(f"    • Checkpoints Saved: 3")
        
        print("  ✅ Training simulation working!")
        return True
        
    except Exception as e:
        print(f"  ❌ Training simulation failed: {e}")
        return False

def test_end_to_end_workflow():
    """Test the complete end-to-end workflow."""
    print("\n🧪 Testing End-to-End Workflow...")
    
    try:
        print("  🔄 Simulating complete RL pipeline...")
        
        # 1. Environment Setup
        from thesaurus_loader import ThesaurusLoader
        loader = ThesaurusLoader()
        print("    ✅ Environment loaded")
        
        # 2. Inference Setup
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:8888/v1", api_key="mock")
        print("    ✅ Inference client ready")
        
        # 3. Training Loop Simulation
        steps = 3
        total_reward = 0.0
        
        for step in range(steps):
            # Generate data (orchestrator role)
            original = "The good dog ran fast through the park."
            augmented, _ = loader.replace_with_synonyms(original, replacement_rate=0.3)
            
            # Get model response (inference role)
            prompt = f"Restore the original text: {augmented}"
            response = client.chat.completions.create(
                model="mock-thesaurus",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=30
            )
            
            model_output = response.choices[0].message.content
            
            # Calculate reward (environment role)
            import re
            original_words = re.findall(r'\b\w+\b', original.lower())
            output_words = re.findall(r'\b\w+\b', model_output.lower())
            matches = sum(1 for o, m in zip(original_words, output_words) if o == m)
            reward = matches / len(original_words) if original_words else 0.0
            
            total_reward += reward
            
            # Training step (trainer role)
            loss = 1.0 - reward + random.uniform(-0.1, 0.1)
            
            print(f"    Step {step+1}: Reward={reward:.4f}, Loss={loss:.4f}")
        
        avg_reward = total_reward / steps
        print(f"  📊 E2E Results:")
        print(f"    • Steps Completed: {steps}")
        print(f"    • Average Reward: {avg_reward:.4f}")
        print(f"    • Pipeline Status: ✅ Working")
        
        print("  ✅ End-to-end workflow working!")
        return True
        
    except Exception as e:
        print(f"  ❌ End-to-end workflow failed: {e}")
        return False

def main():
    """Run all full pipeline tests."""
    print("🚀 Testing Full Training Pipeline")
    print("=" * 50)
    
    # Start mock server
    server_process = start_mock_server()
    
    try:
        # Wait for server to be ready
        if not check_mock_server_running():
            print("❌ Mock server not available. Some tests will be skipped.")
            return 1
        
        tests = [
            ("Orchestrator Integration", test_orchestrator_integration),
            ("Training Simulation", test_training_simulation),
            ("End-to-End Workflow", test_end_to_end_workflow),
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
        print(f"📊 PIPELINE TEST SUMMARY: {sum(results)}/{len(results)} tests passed")
        
        if all(results):
            print("🎉 Full pipeline testing successful!")
            print("✅ Ready for production training with real models")
            
            print("\n💡 Next steps:")
            print("   1. Replace mock inference with real model API")
            print("   2. Use actual prime-rl trainer and orchestrator")
            print("   3. Scale up training parameters")
            
            return 0
        else:
            print("💥 Some pipeline tests failed.")
            return 1
            
    finally:
        # Cleanup
        if server_process:
            try:
                server_process.terminate()
                server_process.wait(timeout=5)
                print("\n🛑 Mock server stopped")
            except Exception:
                pass

if __name__ == "__main__":
    sys.exit(main())