#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "rich",
#   "requests",
#   "openai",
# ]
# requires-python = ">=3.8"
# ///
"""
macOS-compatible training simulator for Second Occurrence Masking environment.

This script simulates the complete RL training pipeline without requiring GPU
resources or external dependencies. It's designed for development and testing
on macOS systems.

Features:
- Complete training simulation with realistic metrics
- Rich console output with progress tracking
- Mock orchestrator + trainer + inference in one script
- Configurable parameters for different testing scenarios
- Checkpointing simulation
- Performance analysis and reporting

Usage:
    ./mock_trainer.py                           # Default: 20 steps
    ./mock_trainer.py --steps 50 --batch-size 16 --accuracy 0.8
    ./mock_trainer.py --mode fast               # Quick test mode
"""

import argparse
import time
import random
import json
import subprocess
import signal
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich import box

# Add environment to path
env_dir = Path(__file__).parent
sys.path.insert(0, str(env_dir))

console = Console()


@dataclass
class TrainingMetrics:
    """Training metrics for simulation."""
    step: int
    loss: float
    reward: float
    accuracy: float
    examples_per_sec: float
    memory_usage: float


@dataclass
class TrainingConfig:
    """Configuration for training simulation."""
    total_steps: int = 20
    batch_size: int = 8
    model_accuracy: float = 0.7
    learning_rate: float = 3e-6
    save_every: int = 10
    log_every: int = 1
    mode: str = "normal"  # normal, fast, detailed


class MockTrainingSimulator:
    """Simulates the complete training pipeline."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.metrics: List[TrainingMetrics] = []
        self.mock_server_process: Optional[subprocess.Popen] = None
        self.current_step = 0
        
        # Simulation parameters
        self.base_loss = 2.5
        self.base_reward = 0.1
        self.base_accuracy = 0.1
        
        random.seed(42)  # For reproducible simulation
    
    def start_mock_server(self, port: int = 8888) -> bool:
        """Start the mock inference server."""
        console.print(f"🚀 Starting mock inference server on port {port}...", style="bold blue")
        
        server_script = env_dir / "mock_server.py"
        self.mock_server_process = subprocess.Popen([
            str(server_script),
            "--port", str(port),
            "--mode", "masking_aware",
            "--accuracy", str(self.config.model_accuracy),
            "--host", "127.0.0.1"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        for _ in range(20):
            try:
                import requests
                response = requests.get(f"http://localhost:{port}/health", timeout=1)
                if response.status_code == 200:
                    console.print("   ✅ Mock server started successfully")
                    return True
            except:
                time.sleep(0.5)
        
        console.print("   ❌ Failed to start mock server", style="red")
        return False
    
    def stop_mock_server(self):
        """Stop the mock inference server."""
        if self.mock_server_process:
            self.mock_server_process.terminate()
            try:
                self.mock_server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.mock_server_process.kill()
            console.print("   ✅ Mock server stopped")
    
    def load_environment(self) -> bool:
        """Load and validate the environment."""
        console.print("📚 Loading Second Occurrence Masking environment...", style="bold blue")
        
        try:
            # Add prime-rl to path
            prime_rl_src = env_dir.parent.parent / "src"
            sys.path.insert(0, str(prime_rl_src))
            
            from prime_rl.environments.registry import load_environment
            
            # Load environment with smaller dataset for simulation
            self.env = load_environment("second-occurrence-masking", {
                "num_examples": 100,  # Small for faster simulation
                "min_length": 30,
                "max_length": 150,
                "min_masks": 1,
                "max_masks": 3
            })
            
            console.print(f"   ✅ Environment loaded with {len(self.env.dataset)} examples")
            
            # Show sample
            sample = self.env.dataset[0]
            console.print(f"   📝 Sample question: {sample['question'][:80]}...")
            console.print(f"   🎯 Sample answer: {sample['answer']}")
            
            return True
            
        except Exception as e:
            console.print(f"   ❌ Failed to load environment: {e}", style="red")
            return False
    
    def simulate_training_step(self, step: int) -> TrainingMetrics:
        """Simulate a single training step."""
        
        # Simulate realistic learning curves
        progress = step / self.config.total_steps
        
        # Loss decreases with training (with noise)
        loss_trend = self.base_loss * (1 - 0.7 * progress)
        loss_noise = random.gauss(0, 0.1)
        loss = max(0.1, loss_trend + loss_noise)
        
        # Reward increases with training (with noise)
        reward_trend = self.base_reward + (self.config.model_accuracy - self.base_reward) * progress
        reward_noise = random.gauss(0, 0.05)
        reward = max(0.0, min(1.0, reward_trend + reward_noise))
        
        # Accuracy improves more slowly
        acc_trend = self.base_accuracy + (self.config.model_accuracy - self.base_accuracy) * (progress ** 1.5)
        acc_noise = random.gauss(0, 0.02)
        accuracy = max(0.0, min(1.0, acc_trend + acc_noise))
        
        # Simulate processing speed
        examples_per_sec = random.uniform(8, 15) * self.config.batch_size
        
        # Simulate memory usage
        memory_usage = random.uniform(2.1, 3.2)  # GB
        
        return TrainingMetrics(
            step=step,
            loss=loss,
            reward=reward,
            accuracy=accuracy,
            examples_per_sec=examples_per_sec,
            memory_usage=memory_usage
        )
    
    def test_inference(self) -> bool:
        """Test inference with the mock server."""
        try:
            from openai import OpenAI
            
            client = OpenAI(
                base_url="http://localhost:8888/v1",
                api_key="mock"
            )
            
            # Get a test example from environment
            sample = self.env.dataset[0]
            info = sample.get("info", {})
            masked_text = info.get("masked_text", "")
            target_words = info.get("target_words", [])
            
            if not masked_text:
                return True  # Skip if no masked text
            
            # Test inference
            response = client.chat.completions.create(
                model="mock-masking_aware",
                messages=[
                    {"role": "system", "content": "Fill in [MASK] tokens with original words."},
                    {"role": "user", "content": f"Fill in the [MASK] tokens: {masked_text}"}
                ],
                max_tokens=50,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content
            console.print(f"   🧪 Test inference: '{answer}' (expected: {target_words})")
            return True
            
        except Exception as e:
            console.print(f"   ❌ Inference test failed: {e}", style="red")
            return False
    
    def create_training_display(self) -> Layout:
        """Create rich display layout for training."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=7)
        )
        
        layout["body"].split_row(
            Layout(name="metrics", ratio=1),
            Layout(name="progress", ratio=1)
        )
        
        return layout
    
    def update_training_display(self, layout: Layout, progress_obj: Progress, task_id):
        """Update the training display with current metrics."""
        
        # Header
        layout["header"].update(Panel(
            f"🎭 Second Occurrence Masking - Mock Training Simulation\n"
            f"Steps: {self.current_step}/{self.config.total_steps} | "
            f"Batch Size: {self.config.batch_size} | "
            f"Model Accuracy: {self.config.model_accuracy:.1%}",
            style="bold blue"
        ))
        
        # Metrics table
        if self.metrics:
            table = Table(box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Current", style="yellow")
            table.add_column("Best", style="green")
            table.add_column("Trend", style="magenta")
            
            latest = self.metrics[-1]
            best_reward = max(m.reward for m in self.metrics)
            best_accuracy = max(m.accuracy for m in self.metrics)
            min_loss = min(m.loss for m in self.metrics)
            
            # Calculate trends
            if len(self.metrics) >= 5:
                recent_rewards = [m.reward for m in self.metrics[-5:]]
                reward_trend = "📈" if recent_rewards[-1] > recent_rewards[0] else "📉"
                
                recent_losses = [m.loss for m in self.metrics[-5:]]
                loss_trend = "📉" if recent_losses[-1] < recent_losses[0] else "📈"
            else:
                reward_trend = "➡️"
                loss_trend = "➡️"
            
            table.add_row("Loss", f"{latest.loss:.3f}", f"{min_loss:.3f}", loss_trend)
            table.add_row("Reward", f"{latest.reward:.3f}", f"{best_reward:.3f}", reward_trend)
            table.add_row("Accuracy", f"{latest.accuracy:.1%}", f"{best_accuracy:.1%}", reward_trend)
            table.add_row("Speed", f"{latest.examples_per_sec:.1f} ex/s", "-", "⚡")
            table.add_row("Memory", f"{latest.memory_usage:.1f} GB", "-", "💾")
            
            layout["metrics"].update(Panel(table, title="📊 Training Metrics", border_style="green"))
        
        # Progress
        layout["progress"].update(Panel(progress_obj, title="🚀 Training Progress", border_style="blue"))
        
        # Footer with recent logs
        recent_logs = []
        if len(self.metrics) >= 3:
            for m in self.metrics[-3:]:
                recent_logs.append(f"Step {m.step:2d}: loss={m.loss:.3f}, reward={m.reward:.3f}, acc={m.accuracy:.1%}")
        
        footer_text = "\n".join(recent_logs) if recent_logs else "Starting training..."
        layout["footer"].update(Panel(footer_text, title="📋 Recent Steps", border_style="yellow"))
    
    def simulate_checkpointing(self, step: int):
        """Simulate saving checkpoints."""
        if step % self.config.save_every == 0:
            # Simulate checkpoint saving time
            if self.config.mode != "fast":
                time.sleep(0.2)
            console.print(f"   💾 Checkpoint saved at step {step}")
    
    def run_training(self):
        """Run the complete training simulation."""
        
        # Initialize
        if not self.load_environment():
            return False
        
        if not self.start_mock_server():
            return False
        
        try:
            # Test inference
            console.print("🧪 Testing inference connection...", style="bold blue")
            if not self.test_inference():
                return False
            console.print("   ✅ Inference test passed")
            
            # Setup progress tracking
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
                TimeElapsedColumn(),
            ) as progress:
                
                task = progress.add_task("Training...", total=self.config.total_steps)
                layout = self.create_training_display()
                
                with Live(layout, refresh_per_second=4):
                    for step in range(1, self.config.total_steps + 1):
                        self.current_step = step
                        
                        # Simulate training step
                        metrics = self.simulate_training_step(step)
                        self.metrics.append(metrics)
                        
                        # Update display
                        progress.update(task, completed=step)
                        self.update_training_display(layout, progress, task)
                        
                        # Simulate step time
                        if self.config.mode == "fast":
                            time.sleep(0.05)
                        elif self.config.mode == "detailed":
                            time.sleep(0.5)
                        else:
                            time.sleep(0.2)
                        
                        # Simulate checkpointing
                        self.simulate_checkpointing(step)
            
            # Training completed
            self.show_final_results()
            return True
            
        finally:
            self.stop_mock_server()
    
    def show_final_results(self):
        """Show final training results."""
        console.print("\n" + "="*60)
        console.print("🎉 [bold green]Training Simulation Completed![/bold green]")
        console.print("="*60)
        
        if not self.metrics:
            console.print("❌ No metrics recorded")
            return
        
        # Final metrics
        final = self.metrics[-1]
        best_reward = max(m.reward for m in self.metrics)
        best_accuracy = max(m.accuracy for m in self.metrics)
        min_loss = min(m.loss for m in self.metrics)
        
        # Create summary table
        table = Table(title="📊 Final Training Summary", box=box.DOUBLE_EDGE)
        table.add_column("Metric", style="cyan")
        table.add_column("Final", style="yellow")
        table.add_column("Best", style="green")
        table.add_column("Improvement", style="magenta")
        
        initial = self.metrics[0]
        reward_improvement = ((final.reward - initial.reward) / initial.reward * 100) if initial.reward > 0 else 0
        loss_improvement = ((initial.loss - final.loss) / initial.loss * 100) if initial.loss > 0 else 0
        acc_improvement = ((final.accuracy - initial.accuracy) / initial.accuracy * 100) if initial.accuracy > 0 else 0
        
        table.add_row("Loss", f"{final.loss:.3f}", f"{min_loss:.3f}", f"{loss_improvement:+.1f}%")
        table.add_row("Reward", f"{final.reward:.3f}", f"{best_reward:.3f}", f"{reward_improvement:+.1f}%")
        table.add_row("Accuracy", f"{final.accuracy:.1%}", f"{best_accuracy:.1%}", f"{acc_improvement:+.1f}%")
        
        console.print(table)
        
        # Performance summary
        avg_speed = sum(m.examples_per_sec for m in self.metrics) / len(self.metrics)
        total_examples = avg_speed * len(self.metrics) * 0.2  # Approximate
        
        console.print(f"\n📈 [bold]Performance Summary:[/bold]")
        console.print(f"   • Total steps: {len(self.metrics)}")
        console.print(f"   • Average speed: {avg_speed:.1f} examples/sec")
        console.print(f"   • Estimated examples processed: {total_examples:.0f}")
        console.print(f"   • Final reward: {final.reward:.3f} ({final.reward*100:.1f}% mask accuracy)")
        
        # Success criteria
        console.print(f"\n🎯 [bold]Success Criteria:[/bold]")
        success_items = []
        
        if final.reward > 0.3:
            success_items.append("✅ Reward > 0.3 (reasonable mask filling)")
        else:
            success_items.append("❌ Reward ≤ 0.3 (poor mask filling)")
        
        if final.loss < initial.loss * 0.8:
            success_items.append("✅ Loss reduced by >20%")
        else:
            success_items.append("❌ Loss not sufficiently reduced")
        
        if final.accuracy > 0.4:
            success_items.append("✅ Accuracy > 40%")
        else:
            success_items.append("❌ Accuracy ≤ 40%")
        
        for item in success_items:
            console.print(f"   {item}")
        
        # Recommendations
        console.print(f"\n💡 [bold]Recommendations:[/bold]")
        if final.reward < 0.5:
            console.print("   • Consider increasing model size or training steps")
            console.print("   • Tune hyperparameters (learning rate, batch size)")
        
        if final.loss > 1.0:
            console.print("   • Loss still high - may need longer training")
        
        console.print("   • Test with real models using: [cyan]cp inference_config_ollama.toml inference_config.toml[/cyan]")
        console.print("   • Scale up with: [cyan]cp inference_config.toml inference_config.toml[/cyan] (cloud inference)")


def main():
    parser = argparse.ArgumentParser(description="Mock training simulator for Second Occurrence Masking")
    parser.add_argument("--steps", type=int, default=20, 
                       help="Number of training steps to simulate")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for simulation")
    parser.add_argument("--accuracy", type=float, default=0.7,
                       help="Simulated model accuracy (0.0-1.0)")
    parser.add_argument("--learning-rate", type=float, default=3e-6,
                       help="Learning rate (for display)")
    parser.add_argument("--mode", choices=["fast", "normal", "detailed"], default="normal",
                       help="Simulation speed: fast=quick test, normal=realistic, detailed=slow/verbose")
    parser.add_argument("--save-every", type=int, default=10,
                       help="Save checkpoint every N steps")
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        total_steps=args.steps,
        batch_size=args.batch_size,
        model_accuracy=args.accuracy,
        learning_rate=args.learning_rate,
        mode=args.mode,
        save_every=args.save_every
    )
    
    # Print startup info
    console.print("🎭 [bold]Second Occurrence Masking - Mock Training Simulator[/bold]")
    console.print(f"   Mode: {config.mode}")
    console.print(f"   Steps: {config.total_steps}")
    console.print(f"   Batch Size: {config.batch_size}")
    console.print(f"   Model Accuracy: {config.model_accuracy:.1%}")
    console.print()
    
    # Run simulation
    simulator = MockTrainingSimulator(config)
    
    try:
        success = simulator.run_training()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        console.print("\n⚠️  Training interrupted by user", style="yellow")
        simulator.stop_mock_server()
        sys.exit(1)


if __name__ == "__main__":
    main()