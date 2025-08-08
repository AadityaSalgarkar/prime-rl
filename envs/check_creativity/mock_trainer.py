#!/usr/bin/env python3
"""
Mock Trainer for Creativity Environment

Provides a complete training simulation for macOS development and testing
without requiring CUDA or actual model training infrastructure.
"""

import json
import time
import random
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np

# Rich console for better output formatting
try:
    from rich.console import Console
    from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MockTrainingMetrics:
    """Mock training metrics for simulation."""
    step: int
    epoch: float
    train_loss: float
    eval_loss: Optional[float]
    creativity_score: float
    reward: float
    learning_rate: float
    grad_norm: float
    throughput: float  # samples/sec
    
    # Component creativity metrics
    entropy_score: float
    distinct_ratio: float
    uncommon_words: float
    bigram_diversity: float
    sentence_variance: float
    word_variance: float
    ending_variety: float


@dataclass
class MockTrainingConfig:
    """Mock training configuration."""
    max_steps: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-5
    eval_every: int = 10
    save_every: int = 25
    log_every: int = 5
    
    # Creativity-specific settings
    creativity_target: float = 6.0
    improvement_rate: float = 0.02
    noise_level: float = 0.1
    
    # Mock model settings
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    mock_model_size: str = "0.5B"
    
    # Simulation behavior
    simulate_cuda: bool = True
    simulate_memory_usage: bool = True
    simulate_realistic_timing: bool = True


class MockCreativityTrainer:
    """
    Mock trainer that simulates RLVR training for creativity enhancement.
    
    Provides realistic training simulation including:
    - Progressive improvement in creativity metrics
    - Realistic loss curves and learning dynamics
    - Memory and GPU simulation for macOS development
    - Comprehensive logging and checkpointing
    """
    
    def __init__(self, config: MockTrainingConfig):
        """
        Initialize mock trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.console = Console() if RICH_AVAILABLE else None
        
        # Training state
        self.current_step = 0
        self.current_epoch = 0.0
        self.best_creativity_score = 0.0
        self.training_history: List[MockTrainingMetrics] = []
        
        # Simulation parameters
        self.base_creativity = 3.5
        self.creativity_trend = 0.0
        self.loss_baseline = 2.5
        self.convergence_rate = 0.98
        
        # Mock system resources
        self.mock_gpu_memory = 8192  # MB
        self.mock_cpu_count = 8
        self.mock_memory_usage = 0.0
        
        logger.info(f"Initialized MockCreativityTrainer for {config.max_steps} steps")
    
    def _simulate_step_metrics(self) -> MockTrainingMetrics:
        """Simulate realistic training metrics for a single step."""
        progress = self.current_step / self.config.max_steps
        
        # Simulate progressive improvement in creativity
        creativity_improvement = progress * self.config.improvement_rate * self.config.max_steps
        noise = random.gauss(0, self.config.noise_level)
        creativity_score = self.base_creativity + creativity_improvement + noise
        creativity_score = max(1.0, min(10.0, creativity_score))  # Clamp to reasonable range
        
        # Simulate learning dynamics
        train_loss = self.loss_baseline * (self.convergence_rate ** self.current_step) + random.gauss(0, 0.1)
        train_loss = max(0.1, train_loss)
        
        # Simulate reward based on creativity improvement
        reward_base = 0.5 + (creativity_score - self.base_creativity) / (10.0 - self.base_creativity) * 0.4
        reward = max(0.0, min(1.0, reward_base + random.gauss(0, 0.05)))
        
        # Simulate component metrics
        entropy_base = 3.0 + progress * 2.5
        distinct_base = 0.5 + progress * 0.3
        uncommon_base = 0.1 + progress * 0.2
        bigram_base = 0.6 + progress * 0.25
        
        # Add some realistic correlations and noise
        entropy_score = max(2.0, min(7.5, entropy_base + random.gauss(0, 0.3)))
        distinct_ratio = max(0.3, min(0.9, distinct_base + random.gauss(0, 0.05)))
        uncommon_words = max(0.05, min(0.4, uncommon_base + random.gauss(0, 0.03)))
        bigram_diversity = max(0.5, min(0.95, bigram_base + random.gauss(0, 0.05)))
        
        sentence_variance = max(0.5, min(4.0, 1.5 + progress * 1.5 + random.gauss(0, 0.2)))
        word_variance = max(1.0, min(3.5, 1.8 + progress * 1.2 + random.gauss(0, 0.15)))
        ending_variety = max(0.3, min(2.5, 0.8 + progress * 1.2 + random.gauss(0, 0.1)))
        
        # Learning rate schedule
        lr_decay = 0.95 ** (self.current_step // 20)
        learning_rate = self.config.learning_rate * lr_decay
        
        # Simulate other training metrics
        grad_norm = random.uniform(0.5, 2.5) * (1.0 - progress * 0.3)  # Decreasing grad norm
        throughput = random.uniform(8.0, 12.0)  # samples per second
        
        # Calculate epoch based on step and batch size
        steps_per_epoch = 1000 // self.config.batch_size  # Assume 1000 samples per epoch
        epoch = self.current_step / steps_per_epoch
        
        # Eval loss (computed periodically)
        eval_loss = None
        if self.current_step % self.config.eval_every == 0:
            eval_loss = train_loss + random.gauss(0, 0.05)  # Slightly noisy version of train loss
        
        return MockTrainingMetrics(
            step=self.current_step,
            epoch=epoch,
            train_loss=train_loss,
            eval_loss=eval_loss,
            creativity_score=creativity_score,
            reward=reward,
            learning_rate=learning_rate,
            grad_norm=grad_norm,
            throughput=throughput,
            entropy_score=entropy_score,
            distinct_ratio=distinct_ratio,
            uncommon_words=uncommon_words,
            bigram_diversity=bigram_diversity,
            sentence_variance=sentence_variance,
            word_variance=word_variance,
            ending_variety=ending_variety
        )
    
    def _simulate_processing_time(self):
        """Simulate realistic processing time."""
        if self.config.simulate_realistic_timing:
            # Base time per step
            base_time = 0.1 + random.uniform(0.05, 0.15)
            
            # Add occasional slower steps (checkpointing, evaluation)
            if self.current_step % self.config.save_every == 0:
                base_time += random.uniform(0.5, 1.0)
            elif self.current_step % self.config.eval_every == 0:
                base_time += random.uniform(0.2, 0.4)
            
            time.sleep(base_time)
    
    def _update_mock_system_resources(self):
        """Update mock system resource usage."""
        if self.config.simulate_memory_usage:
            # Simulate memory usage that increases slightly over time
            base_memory = 2048  # MB
            step_memory = self.current_step * 0.5  # Slight memory growth
            noise_memory = random.uniform(-100, 100)
            
            self.mock_memory_usage = min(self.mock_gpu_memory * 0.8, base_memory + step_memory + noise_memory)
    
    def _log_metrics_rich(self, metrics: MockTrainingMetrics):
        """Log metrics using rich console formatting."""
        if not self.console:
            return
        
        # Create a comprehensive metrics table
        table = Table(title=f"Training Step {metrics.step}", box=box.ROUNDED)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta", justify="right")
        table.add_column("Progress", style="green")
        
        # Core metrics
        table.add_row("Epoch", f"{metrics.epoch:.2f}", "")
        table.add_row("Train Loss", f"{metrics.train_loss:.4f}", "")
        if metrics.eval_loss is not None:
            table.add_row("Eval Loss", f"{metrics.eval_loss:.4f}", "")
        
        table.add_row("", "", "")  # Separator
        
        # Creativity metrics
        creativity_progress = "🟢" if metrics.creativity_score > self.config.creativity_target else "🟡" if metrics.creativity_score > 4.0 else "🔴"
        table.add_row("Creativity Score", f"{metrics.creativity_score:.3f}", creativity_progress)
        table.add_row("Reward", f"{metrics.reward:.3f}", "")
        
        table.add_row("", "", "")  # Separator
        
        # Component metrics
        table.add_row("Entropy", f"{metrics.entropy_score:.3f}", "")
        table.add_row("Distinct Ratio", f"{metrics.distinct_ratio:.3f}", "")
        table.add_row("Uncommon Words", f"{metrics.uncommon_words:.3f}", "")
        table.add_row("Bigram Diversity", f"{metrics.bigram_diversity:.3f}", "")
        
        table.add_row("", "", "")  # Separator
        
        # Training metrics
        table.add_row("Learning Rate", f"{metrics.learning_rate:.2e}", "")
        table.add_row("Grad Norm", f"{metrics.grad_norm:.3f}", "")
        table.add_row("Throughput", f"{metrics.throughput:.1f} samples/s", "")
        
        # System resources
        if self.config.simulate_memory_usage:
            memory_pct = (self.mock_memory_usage / self.mock_gpu_memory) * 100
            table.add_row("GPU Memory", f"{self.mock_memory_usage:.0f}/{self.mock_gpu_memory:.0f} MB ({memory_pct:.1f}%)", "")
        
        self.console.print(table)
        
        # Update best score tracking
        if metrics.creativity_score > self.best_creativity_score:
            self.best_creativity_score = metrics.creativity_score
            self.console.print(f"🎉 New best creativity score: {self.best_creativity_score:.3f}", style="bold green")
    
    def _log_metrics_standard(self, metrics: MockTrainingMetrics):
        """Log metrics using standard logging."""
        log_msg = (
            f"Step {metrics.step:4d} | "
            f"Loss: {metrics.train_loss:.4f} | "
            f"Creativity: {metrics.creativity_score:.3f} | "
            f"Reward: {metrics.reward:.3f} | "
            f"LR: {metrics.learning_rate:.2e} | "
            f"Throughput: {metrics.throughput:.1f} samples/s"
        )
        
        if metrics.eval_loss is not None:
            log_msg += f" | Eval Loss: {metrics.eval_loss:.4f}"
        
        logger.info(log_msg)
        
        # Component metrics (less frequent)
        if self.current_step % (self.config.log_every * 2) == 0:
            component_msg = (
                f"Components - "
                f"Entropy: {metrics.entropy_score:.3f}, "
                f"Distinct: {metrics.distinct_ratio:.3f}, "
                f"Uncommon: {metrics.uncommon_words:.3f}, "
                f"Bigrams: {metrics.bigram_diversity:.3f}"
            )
            logger.info(component_msg)
    
    def _save_checkpoint(self, metrics: MockTrainingMetrics):
        """Save training checkpoint."""
        checkpoint_data = {
            "step": self.current_step,
            "epoch": metrics.epoch,
            "best_creativity_score": self.best_creativity_score,
            "config": asdict(self.config),
            "current_metrics": asdict(metrics),
            "training_history": [asdict(m) for m in self.training_history[-50:]],  # Last 50 steps
        }
        
        checkpoint_path = Path(f"mock_checkpoint_step_{self.current_step}.json")
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Also save best model if this is the best score
        if metrics.creativity_score == self.best_creativity_score:
            best_model_path = Path("mock_best_model.json")
            with open(best_model_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            logger.info(f"Saved best model to {best_model_path}")
    
    def _print_training_summary(self):
        """Print final training summary."""
        if not self.training_history:
            return
        
        final_metrics = self.training_history[-1]
        initial_metrics = self.training_history[0] if len(self.training_history) > 1 else final_metrics
        
        if self.console:
            # Rich summary table
            summary_table = Table(title="Training Summary", box=box.DOUBLE_EDGE)
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Initial", style="yellow", justify="right")
            summary_table.add_column("Final", style="green", justify="right")
            summary_table.add_column("Improvement", style="magenta", justify="right")
            
            creativity_improvement = final_metrics.creativity_score - initial_metrics.creativity_score
            reward_improvement = final_metrics.reward - initial_metrics.reward
            loss_improvement = initial_metrics.train_loss - final_metrics.train_loss  # Loss goes down
            
            summary_table.add_row(
                "Creativity Score",
                f"{initial_metrics.creativity_score:.3f}",
                f"{final_metrics.creativity_score:.3f}",
                f"+{creativity_improvement:.3f}"
            )
            
            summary_table.add_row(
                "Reward",
                f"{initial_metrics.reward:.3f}",
                f"{final_metrics.reward:.3f}",
                f"+{reward_improvement:.3f}"
            )
            
            summary_table.add_row(
                "Training Loss",
                f"{initial_metrics.train_loss:.4f}",
                f"{final_metrics.train_loss:.4f}",
                f"-{loss_improvement:.4f}"
            )
            
            summary_table.add_row(
                "Best Creativity",
                "-",
                f"{self.best_creativity_score:.3f}",
                "-"
            )
            
            self.console.print(summary_table)
            
            # Success message
            if final_metrics.creativity_score >= self.config.creativity_target:
                self.console.print(
                    Panel(
                        f"🎉 Training completed successfully!\nAchieved creativity target of {self.config.creativity_target:.1f}",
                        title="Success",
                        style="green"
                    )
                )
            else:
                remaining = self.config.creativity_target - final_metrics.creativity_score
                self.console.print(
                    Panel(
                        f"Training completed. {remaining:.2f} points away from target of {self.config.creativity_target:.1f}",
                        title="Training Complete",
                        style="yellow"
                    )
                )
        else:
            # Standard logging summary
            logger.info("=" * 60)
            logger.info("TRAINING SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total steps completed: {self.current_step}")
            logger.info(f"Initial creativity score: {initial_metrics.creativity_score:.3f}")
            logger.info(f"Final creativity score: {final_metrics.creativity_score:.3f}")
            logger.info(f"Best creativity score: {self.best_creativity_score:.3f}")
            logger.info(f"Improvement: +{final_metrics.creativity_score - initial_metrics.creativity_score:.3f}")
            logger.info(f"Target achieved: {final_metrics.creativity_score >= self.config.creativity_target}")
            logger.info("=" * 60)
    
    def train(self):
        """Run the mock training simulation."""
        if self.console:
            self.console.print(
                Panel(
                    f"Starting Mock Creativity Training\n"
                    f"Model: {self.config.model_name}\n"
                    f"Max Steps: {self.config.max_steps}\n"
                    f"Target Creativity: {self.config.creativity_target:.1f}\n"
                    f"Batch Size: {self.config.batch_size}",
                    title="Mock Training Configuration",
                    style="blue"
                )
            )
        else:
            logger.info("Starting Mock Creativity Training")
            logger.info(f"Configuration: {self.config}")
        
        # Training loop with progress bar
        if RICH_AVAILABLE and self.console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=self.console
            ) as progress:
                
                train_task = progress.add_task("Training...", total=self.config.max_steps)
                
                for step in range(self.config.max_steps):
                    self.current_step = step
                    
                    # Simulate processing time
                    self._simulate_processing_time()
                    
                    # Update system resources
                    self._update_mock_system_resources()
                    
                    # Generate step metrics
                    metrics = self._simulate_step_metrics()
                    self.training_history.append(metrics)
                    
                    # Log metrics
                    if step % self.config.log_every == 0:
                        self._log_metrics_rich(metrics)
                    
                    # Save checkpoint
                    if step % self.config.save_every == 0 and step > 0:
                        self._save_checkpoint(metrics)
                    
                    # Update progress
                    progress.update(train_task, advance=1)
                    
                    # Early stopping check
                    if metrics.creativity_score >= self.config.creativity_target * 1.1:  # 10% above target
                        progress.console.print("🎯 Exceeded creativity target! Early stopping.", style="bold green")
                        break
        else:
            # Standard training loop
            for step in range(self.config.max_steps):
                self.current_step = step
                
                # Simulate processing time
                self._simulate_processing_time()
                
                # Update system resources
                self._update_mock_system_resources()
                
                # Generate step metrics
                metrics = self._simulate_step_metrics()
                self.training_history.append(metrics)
                
                # Log metrics
                if step % self.config.log_every == 0:
                    self._log_metrics_standard(metrics)
                
                # Save checkpoint
                if step % self.config.save_every == 0 and step > 0:
                    self._save_checkpoint(metrics)
                
                # Progress indication
                if step % (self.config.max_steps // 10) == 0 and step > 0:
                    progress_pct = (step / self.config.max_steps) * 100
                    logger.info(f"Training progress: {progress_pct:.1f}%")
                
                # Early stopping check
                if metrics.creativity_score >= self.config.creativity_target * 1.1:
                    logger.info("Exceeded creativity target! Early stopping.")
                    break
        
        # Final checkpoint
        if self.training_history:
            self._save_checkpoint(self.training_history[-1])
        
        # Print summary
        self._print_training_summary()
        
        return self.training_history


def load_config_from_toml(config_path: Path) -> MockTrainingConfig:
    """Load training configuration from TOML file."""
    try:
        import toml
        config_data = toml.load(config_path)
        
        # Extract relevant fields
        config_params = {}
        
        if 'max_steps' in config_data:
            config_params['max_steps'] = config_data['max_steps']
        
        if 'data' in config_data:
            data_config = config_data['data']
            if 'batch_size' in data_config:
                config_params['batch_size'] = data_config['batch_size']
        
        if 'optimizer' in config_data:
            opt_config = config_data['optimizer']
            if 'learning_rate' in opt_config:
                config_params['learning_rate'] = opt_config['learning_rate']
        
        if 'training' in config_data:
            train_config = config_data['training']
            if 'eval_every' in train_config:
                config_params['eval_every'] = train_config['eval_every']
            if 'save_every' in train_config:
                config_params['save_every'] = train_config['save_every']
            if 'log_every' in train_config:
                config_params['log_every'] = train_config['log_every']
        
        if 'model' in config_data:
            model_config = config_data['model']
            if 'model_name' in model_config:
                config_params['model_name'] = model_config['model_name']
        
        return MockTrainingConfig(**config_params)
        
    except Exception as e:
        logger.warning(f"Could not load TOML config from {config_path}: {e}")
        return MockTrainingConfig()


def main():
    """Main function for running mock training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mock Creativity Training Simulator")
    parser.add_argument("--config", type=Path, help="Path to training configuration TOML file")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--creativity-target", type=float, default=6.0, help="Target creativity score")
    parser.add_argument("--no-rich", action="store_true", help="Disable rich console output")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and args.config.exists():
        config = load_config_from_toml(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        config = MockTrainingConfig()
    
    # Override with command line arguments
    if args.max_steps != 100:
        config.max_steps = args.max_steps
    if args.batch_size != 8:
        config.batch_size = args.batch_size
    if args.learning_rate != 1e-5:
        config.learning_rate = args.learning_rate
    if args.creativity_target != 6.0:
        config.creativity_target = args.creativity_target
    
    # Disable rich if requested or not available
    if args.no_rich or not RICH_AVAILABLE:
        logger.info("Using standard console output (rich not available or disabled)")
    
    # Run training simulation
    trainer = MockCreativityTrainer(config)
    
    try:
        training_history = trainer.train()
        logger.info(f"Mock training completed successfully with {len(training_history)} steps")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())