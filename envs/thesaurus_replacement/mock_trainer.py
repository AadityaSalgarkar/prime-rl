#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "torch>=2.0.0",
#   "transformers>=4.30.0", 
#   "datasets>=2.10.0",
#   "requests>=2.31.0",
#   "openai>=1.0.0",
#   "rich>=13.0.0",
#   "omegaconf>=2.3.0"
# ]
# requires-python = ">=3.8"
# ///
"""
Mock trainer for testing the thesaurus replacement environment on macOS.
Simulates the training workflow without requiring CUDA or full prime-rl framework.

Run with: ./mock_trainer.py
"""

import json
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

@dataclass
class MockTrainingConfig:
    """Configuration for mock training."""
    max_steps: int = 30
    batch_size: int = 8
    learning_rate: float = 3e-6
    log_interval: int = 5
    save_interval: int = 10
    model_name: str = "mock-qwen2.5-0.5b"
    environment_id: str = "thesaurus-replacement"
    
class MockOrchestrator:
    """Mock orchestrator for generating training data."""
    
    def __init__(self, config: MockTrainingConfig):
        self.config = config
        self.step = 0
        
        # Load thesaurus data
        from thesaurus_loader import ThesaurusLoader
        self.thesaurus = ThesaurusLoader()
        
        # Mock inference client
        from mock_inference import create_mock_client
        self.inference_client = create_mock_client("thesaurus_aware")
        
    def generate_rollout_batch(self) -> List[Dict]:
        """Generate a batch of rollouts for training."""
        batch = []
        
        # Sample sentences from a small dataset
        base_sentences = [
            "The good dog ran fast through the park.",
            "She opened the ancient door carefully.",
            "A big house stood on the hill.",
            "The small cat jumped over the fence.",
            "He found the old book in the library.",
            "The bright sun shone in the sky.",
            "They walked down the long road.",
            "The young girl sang a beautiful song.",
        ]
        
        for _ in range(self.config.batch_size):
            # Select random sentence
            original = random.choice(base_sentences)
            
            # Create augmented version
            augmented, replacements = self.thesaurus.replace_with_synonyms(
                original, replacement_rate=0.3
            )
            
            # Get mock model response
            prompt = f"Restore the original text: {augmented}"
            messages = [{"role": "user", "content": prompt}]
            response = self.inference_client.create(model="mock", messages=messages)
            model_output = response.choices[0].message.content
            
            # Calculate reward (word-level accuracy)
            import re
            original_words = re.findall(r'\b\w+\b', original.lower())
            output_words = re.findall(r'\b\w+\b', model_output.lower())
            
            matches = sum(1 for o, m in zip(original_words, output_words) if o == m)
            reward = matches / len(original_words) if original_words else 0.0
            
            batch.append({
                "prompt": prompt,
                "response": model_output,
                "reward": reward,
                "original": original,
                "augmented": augmented,
                "replacements": replacements,
                "step": self.step
            })
        
        return batch

class MockTrainer:
    """Mock trainer that simulates model training."""
    
    def __init__(self, config: MockTrainingConfig):
        self.config = config
        self.step = 0
        self.total_reward = 0.0
        self.total_samples = 0
        
        # Initialize mock model parameters
        self.model_parameters = {
            "embedding_dim": 512,
            "hidden_dim": 1024,
            "num_layers": 12,
            "vocab_size": 32000
        }
        
    def train_step(self, batch: List[Dict]) -> Dict:
        """Simulate a training step."""
        step_reward = sum(item["reward"] for item in batch)
        step_samples = len(batch)
        
        self.total_reward += step_reward
        self.total_samples += step_samples
        
        # Simulate training metrics
        avg_reward = step_reward / step_samples
        cumulative_avg_reward = self.total_reward / self.total_samples
        
        # Simulate loss (inversely related to reward)
        loss = 1.0 - avg_reward + random.uniform(-0.1, 0.1)
        
        # Simulate gradient norm
        grad_norm = random.uniform(0.1, 2.0)
        
        metrics = {
            "step": self.step,
            "loss": loss,
            "reward": avg_reward,
            "cumulative_reward": cumulative_avg_reward,
            "grad_norm": grad_norm,
            "batch_size": step_samples,
            "learning_rate": self.config.learning_rate
        }
        
        return metrics
    
    def save_checkpoint(self):
        """Simulate saving a checkpoint."""
        checkpoint_dir = Path("checkpoints") / f"step_{self.step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_data = {
            "step": self.step,
            "model_parameters": self.model_parameters,
            "config": self.config.__dict__,
            "total_reward": self.total_reward,
            "total_samples": self.total_samples
        }
        
        with open(checkpoint_dir / "trainer.json", "w") as f:
            json.dump(checkpoint_data, f, indent=2)
        
        print(f"💾 Saved checkpoint at step {self.step}")

def run_mock_training():
    """Run the mock training loop."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.progress import Progress, SpinnerColumn, TextColumn
        console = Console()
    except ImportError:
        console = None
    
    config = MockTrainingConfig()
    orchestrator = MockOrchestrator(config)
    trainer = MockTrainer(config)
    
    if console:
        console.print("🚀 Starting Mock Training for Thesaurus Replacement Environment", style="bold green")
        console.print(f"📋 Config: {config.max_steps} steps, batch size {config.batch_size}")
        console.print()
    else:
        print("🚀 Starting Mock Training for Thesaurus Replacement Environment")
        print(f"📋 Config: {config.max_steps} steps, batch size {config.batch_size}")
        print()
    
    start_time = time.time()
    
    for step in range(config.max_steps):
        trainer.step = step
        orchestrator.step = step
        
        # Generate training data
        batch = orchestrator.generate_rollout_batch()
        
        # Training step
        metrics = trainer.train_step(batch)
        
        # Logging
        if step % config.log_interval == 0 or step == config.max_steps - 1:
            elapsed = time.time() - start_time
            
            if console:
                table = Table(title=f"Training Step {step}")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="magenta")
                
                table.add_row("Loss", f"{metrics['loss']:.4f}")
                table.add_row("Reward", f"{metrics['reward']:.4f}")
                table.add_row("Cumulative Reward", f"{metrics['cumulative_reward']:.4f}")
                table.add_row("Grad Norm", f"{metrics['grad_norm']:.4f}")
                table.add_row("Elapsed Time", f"{elapsed:.1f}s")
                
                console.print(table)
                
                # Show sample data
                sample = batch[0]
                console.print(f"\n📝 Sample:")
                console.print(f"  Original: {sample['original']}")
                console.print(f"  Augmented: {sample['augmented']}")
                console.print(f"  Model Output: {sample['response']}")
                console.print(f"  Reward: {sample['reward']:.4f}")
                console.print()
            else:
                print(f"Step {step}: Loss={metrics['loss']:.4f}, Reward={metrics['reward']:.4f}, "
                      f"Cumulative={metrics['cumulative_reward']:.4f}, Time={elapsed:.1f}s")
        
        # Save checkpoint
        if step % config.save_interval == 0 or step == config.max_steps - 1:
            trainer.save_checkpoint()
        
        # Small delay to simulate training time
        time.sleep(0.1)
    
    total_time = time.time() - start_time
    
    if console:
        console.print("✅ Training Completed!", style="bold green")
        console.print(f"📊 Final Metrics:")
        console.print(f"  • Total Steps: {config.max_steps}")
        console.print(f"  • Final Reward: {metrics['cumulative_reward']:.4f}")
        console.print(f"  • Total Time: {total_time:.1f}s")
        console.print(f"  • Samples Processed: {trainer.total_samples}")
    else:
        print("✅ Training Completed!")
        print(f"📊 Final Metrics: {config.max_steps} steps, "
              f"Final Reward: {metrics['cumulative_reward']:.4f}, "
              f"Time: {total_time:.1f}s")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mock trainer for thesaurus replacement environment")
    parser.add_argument("--steps", type=int, default=30, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-6, help="Learning rate")
    
    args = parser.parse_args()
    
    # Update config with command line args
    config = MockTrainingConfig(
        max_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    # Check if required files exist
    required_files = ["thesaurus_loader.py", "mock_inference.py", "en_thesaurus.jsonl"]
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("   Make sure you're running from the thesaurus_replacement directory")
        return 1
    
    try:
        run_mock_training()
        return 0
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())