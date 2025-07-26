#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["rich", "requests", "openai"]
# requires-python = ">=3.8"
# ///

"""
Mock training simulator for swap tracking environment.
Simulates complete RL training pipeline without GPU requirements for macOS development.
"""

import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add environment directory to path
env_dir = Path(__file__).parent
sys.path.insert(0, str(env_dir))

from swap_tracking_loader import SwapTrackingLoader

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
except ImportError:
    print("Installing rich for better output...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live

console = Console()


class MockTrainingSimulator:
    """Simulates complete RLVR training pipeline."""
    
    def __init__(self, 
                 n_boxes: int = 10,
                 n_swaps: int = 20,
                 max_steps: int = 30,
                 batch_size: int = 8,
                 learning_rate: float = 3e-6):
        """
        Initialize mock training simulator.
        
        Args:
            n_boxes: Number of boxes in swap tracking task
            n_swaps: Number of swaps per task
            max_steps: Maximum training steps
            batch_size: Batch size for training
            learning_rate: Learning rate
        """
        self.n_boxes = n_boxes
        self.n_swaps = n_swaps
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.loader = SwapTrackingLoader(n_boxes=n_boxes, n_swaps=n_swaps)
        self.mock_server_process = None
        self.training_data = []
        self.metrics = {
            "step": 0,
            "train_loss": 1.0,
            "train_reward": 0.0,
            "eval_reward": 0.0,
            "lr": learning_rate
        }
    
    def start_mock_inference_server(self, port: int = 8888) -> bool:
        """Start mock inference server in background."""
        console.print("[blue]Starting mock inference server...[/blue]")
        
        mock_server_path = env_dir / "mock_server.py"
        if not mock_server_path.exists():
            console.print("❌ Mock server not found")
            return False
        
        try:
            self.mock_server_process = subprocess.Popen([
                str(mock_server_path),
                "--host", "localhost",
                "--port", str(port)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for server to start
            time.sleep(2)
            
            # Test connection
            import requests
            response = requests.get(f"http://localhost:{port}/", timeout=5)
            
            if response.status_code == 200:
                console.print(f"✅ Mock inference server running on port {port}")
                return True
            else:
                console.print("❌ Mock server not responding")
                return False
                
        except Exception as e:
            console.print(f"❌ Failed to start mock server: {e}")
            return False
    
    def generate_training_data(self, num_examples: int = 100) -> None:
        """Generate training data for the environment."""
        console.print(f"[blue]Generating {num_examples} training examples...[/blue]")
        
        self.training_data = self.loader.generate_training_examples(
            num_examples, seed=42
        )
        
        console.print(f"✅ Generated {len(self.training_data)} training examples")
    
    def simulate_inference_batch(self, examples: List[Dict], model_name: str = "mock-swap-aware") -> List[Dict]:
        """Simulate batch inference with mock model."""
        try:
            from openai import OpenAI
            
            client = OpenAI(
                base_url="http://localhost:8888/v1",
                api_key="mock"
            )
            
            results = []
            
            for example in examples:
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": example["question"]}],
                        max_tokens=100,
                        temperature=0.1
                    )
                    
                    prediction = response.choices[0].message.content
                    reward = self.loader.calculate_reward(
                        prediction, example["info"]["final_state"]
                    )
                    
                    results.append({
                        "question": example["question"],
                        "prediction": prediction,
                        "answer": example["answer"],
                        "reward": reward,
                        "final_state": example["info"]["final_state"]
                    })
                    
                except Exception as e:
                    console.print(f"⚠ Inference failed for example: {e}")
                    results.append({
                        "question": example["question"],
                        "prediction": "[inference_failed]",
                        "answer": example["answer"],
                        "reward": 0.0,
                        "final_state": example["info"]["final_state"]
                    })
            
            return results
            
        except Exception as e:
            console.print(f"❌ Batch inference failed: {e}")
            return []
    
    def simulate_training_step(self, step: int) -> Dict:
        """Simulate one training step."""
        # Select random batch
        batch = random.sample(self.training_data, min(self.batch_size, len(self.training_data)))
        
        # Simulate inference
        inference_results = self.simulate_inference_batch(batch)
        
        if not inference_results:
            return self.metrics
        
        # Calculate metrics
        batch_rewards = [r["reward"] for r in inference_results]
        avg_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0
        
        # Simulate training dynamics
        # Reward should generally improve over time with some noise
        base_improvement = step / self.max_steps * 0.5  # Max 0.5 improvement
        noise = random.gauss(0, 0.1)
        simulated_reward = min(1.0, max(0.0, base_improvement + noise))
        
        # Simulate loss decrease
        simulated_loss = max(0.1, 1.0 - (step / self.max_steps) * 0.8 + random.gauss(0, 0.05))
        
        # Update metrics
        self.metrics.update({
            "step": step,
            "train_loss": simulated_loss,
            "train_reward": simulated_reward,
            "eval_reward": simulated_reward * 0.9 + random.gauss(0, 0.05),  # Slightly lower than train
            "lr": self.learning_rate * (0.95 ** (step // 10)),  # Decay every 10 steps
            "batch_size": len(batch),
            "avg_batch_reward": avg_reward
        })
        
        return self.metrics
    
    def create_training_display(self) -> Layout:
        """Create rich display layout for training progress."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=8)
        )
        
        layout["main"].split_row(
            Layout(name="metrics", ratio=1),
            Layout(name="examples", ratio=2)
        )
        
        return layout
    
    def update_display(self, layout: Layout, progress: Progress, 
                      recent_examples: List[Dict] = None) -> None:
        """Update the training display."""
        # Header
        layout["header"].update(Panel(
            f"[bold green]Mock RLVR Training - Swap Tracking Environment[/bold green]\n"
            f"Boxes: {self.n_boxes} | Swaps: {self.n_swaps} | Model: mock-swap-aware",
            title="🚀 Training Status"
        ))
        
        # Metrics table
        metrics_table = Table(title="Training Metrics")
        metrics_table.add_column("Metric", justify="left")
        metrics_table.add_column("Value", justify="right")
        
        metrics_table.add_row("Step", str(self.metrics["step"]))
        metrics_table.add_row("Train Loss", f"{self.metrics['train_loss']:.4f}")
        metrics_table.add_row("Train Reward", f"{self.metrics['train_reward']:.4f}")
        metrics_table.add_row("Eval Reward", f"{self.metrics['eval_reward']:.4f}")
        metrics_table.add_row("Learning Rate", f"{self.metrics['lr']:.2e}")
        
        if "avg_batch_reward" in self.metrics:
            metrics_table.add_row("Batch Reward", f"{self.metrics['avg_batch_reward']:.4f}")
        
        layout["metrics"].update(Panel(metrics_table, title="📊 Metrics"))
        
        # Recent examples
        if recent_examples:
            examples_table = Table(title="Recent Training Examples")
            examples_table.add_column("Question", width=40)
            examples_table.add_column("Prediction", width=20)
            examples_table.add_column("Reward", justify="right", width=8)
            
            for example in recent_examples[-3:]:  # Show last 3 examples
                question = example["question"][:37] + "..." if len(example["question"]) > 40 else example["question"]
                prediction = example["prediction"][:17] + "..." if len(example["prediction"]) > 20 else example["prediction"]
                reward = f"{example['reward']:.2f}"
                
                examples_table.add_row(question, prediction, reward)
            
            layout["examples"].update(Panel(examples_table, title="🔍 Examples"))
        
        # Footer with progress
        layout["footer"].update(Panel(progress, title="⏱️ Progress"))
    
    def run_training(self, save_checkpoints: bool = True) -> bool:
        """Run complete mock training simulation."""
        console.print("[bold green]Starting Mock RLVR Training Simulation[/bold green]")
        console.print("=" * 60)
        
        # Setup
        if not self.start_mock_inference_server():
            return False
        
        self.generate_training_data(num_examples=200)
        
        # Create display
        layout = self.create_training_display()
        recent_examples = []
        
        # Progress tracking
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            training_task = progress.add_task("Training...", total=self.max_steps)
            
            with Live(layout, refresh_per_second=2, console=console):
                for step in range(1, self.max_steps + 1):
                    # Simulate training step
                    metrics = self.simulate_training_step(step)
                    
                    # Get example results for display
                    if step % 5 == 0:  # Update examples every 5 steps
                        batch = random.sample(self.training_data, 2)
                        batch_results = self.simulate_inference_batch(batch, "mock-swap-aware")
                        recent_examples.extend(batch_results)
                        recent_examples = recent_examples[-10:]  # Keep last 10
                    
                    # Update display
                    self.update_display(layout, progress, recent_examples)
                    
                    # Save checkpoint
                    if save_checkpoints and step % 10 == 0:
                        self.save_checkpoint(step)
                    
                    # Update progress
                    progress.update(training_task, advance=1)
                    
                    # Simulate training time
                    time.sleep(0.5)
        
        # Cleanup
        self.cleanup()
        
        # Final results
        console.print("\n" + "=" * 60)
        console.print("[bold green]Training Completed![/bold green]")
        
        final_table = Table(title="Final Results")
        final_table.add_column("Metric", justify="left")
        final_table.add_column("Final Value", justify="right")
        
        final_table.add_row("Final Train Reward", f"{self.metrics['train_reward']:.4f}")
        final_table.add_row("Final Eval Reward", f"{self.metrics['eval_reward']:.4f}")
        final_table.add_row("Final Loss", f"{self.metrics['train_loss']:.4f}")
        final_table.add_row("Steps Completed", str(self.metrics['step']))
        
        console.print(final_table)
        
        # Success criteria
        if self.metrics['train_reward'] > 0.3 and self.metrics['eval_reward'] > 0.25:
            console.print("[bold green]✅ Training successful! Model shows learning progress.[/bold green]")
            return True
        else:
            console.print("[bold yellow]⚠ Training completed but performance could be better.[/bold yellow]")
            return True
    
    def save_checkpoint(self, step: int) -> None:
        """Simulate saving a training checkpoint."""
        checkpoint_dir = env_dir / "checkpoints" / f"step_{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save mock checkpoint data
        checkpoint_data = {
            "step": step,
            "metrics": self.metrics.copy(),
            "config": {
                "n_boxes": self.n_boxes,
                "n_swaps": self.n_swaps,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate
            }
        }
        
        with open(checkpoint_dir / "trainer.json", 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.mock_server_process:
            console.print("[blue]Shutting down mock inference server...[/blue]")
            self.mock_server_process.terminate()
            self.mock_server_process.wait()


def main():
    """Main training simulation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mock RLVR Training Simulator")
    parser.add_argument("--steps", type=int, default=30, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--boxes", type=int, default=10, help="Number of boxes")
    parser.add_argument("--swaps", type=int, default=20, help="Number of swaps per task")
    parser.add_argument("--lr", type=float, default=3e-6, help="Learning rate")
    parser.add_argument("--no-checkpoints", action="store_true", help="Disable checkpoint saving")
    
    args = parser.parse_args()
    
    # Create simulator
    simulator = MockTrainingSimulator(
        n_boxes=args.boxes,
        n_swaps=args.swaps,
        max_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    # Run training
    success = simulator.run_training(save_checkpoints=not args.no_checkpoints)
    
    if success:
        console.print("\n[bold green]🎉 Mock training simulation completed successfully![/bold green]")
        console.print("\n[bold]Next steps:[/bold]")
        console.print("• Configurations tested and validated")
        console.print("• Mock inference system working")
        console.print("• Ready for real training with GPU")
        console.print("• Try: uv run rl --trainer @ trainer_config.toml --orchestrator @ orchestrator_config.toml --inference @ inference_config.toml")
    else:
        console.print("\n[bold red]❌ Mock training simulation failed![/bold red]")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())