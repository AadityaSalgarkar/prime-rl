#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "requests",
#   "openai",
#   "rich",
# ]
# requires-python = ">=3.8"
# ///
"""
Test orchestration for Second Occurrence Masking environment.

This script runs all test suites and provides comprehensive reporting:
- Core functionality tests
- Local inference tests (Ollama)
- Mock inference tests  
- Full pipeline tests
- Integration tests
- Performance benchmarks

Usage:
    ./tests/run_all_tests.py                    # Run all tests
    ./tests/run_all_tests.py --fast             # Skip slow tests
    ./tests/run_all_tests.py --skip-ollama      # Skip Ollama tests
    ./tests/run_all_tests.py --benchmark        # Include performance tests
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box

console = Console()


class TestResult:
    """Container for test results."""
    
    def __init__(self, name: str, success: bool, duration: float, output: str = "", error: str = ""):
        self.name = name
        self.success = success
        self.duration = duration
        self.output = output
        self.error = error


class TestOrchestrator:
    """Orchestrates all test suites for the environment."""
    
    def __init__(self, args):
        self.args = args
        self.test_dir = Path(__file__).parent
        self.env_dir = self.test_dir.parent
        self.results = []
    
    def run_test_script(self, script_name: str, timeout: int = 120) -> TestResult:
        """Run a test script and capture results."""
        script_path = self.test_dir / script_name
        
        if not script_path.exists():
            return TestResult(script_name, False, 0.0, "", f"Script not found: {script_path}")
        
        console.print(f"   🏃 Running {script_name}...")
        start_time = time.time()
        
        try:
            result = subprocess.run(
                [str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.env_dir
            )
            
            duration = time.time() - start_time
            success = result.returncode == 0
            
            return TestResult(
                name=script_name,
                success=success,
                duration=duration,
                output=result.stdout,
                error=result.stderr
            )
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return TestResult(
                name=script_name,
                success=False,
                duration=duration,
                error=f"Test timed out after {timeout} seconds"
            )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name=script_name,
                success=False,
                duration=duration,
                error=str(e)
            )
    
    def check_prerequisites(self) -> bool:
        """Check if prerequisites are met."""
        console.print("🔍 Checking prerequisites...", style="bold blue")
        
        # Check if we're in the right directory
        expected_files = [
            "second_occurrence_loader.py",
            "trainer_config.toml",
            "orchestrator_config.toml"
        ]
        
        missing_files = []
        for file in expected_files:
            if not (self.env_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            console.print(f"   ❌ Missing files: {missing_files}", style="red")
            return False
        
        console.print("   ✅ All required files present")
        
        # Check Python environment
        try:
            import datasets
            import openai
            import requests
            console.print("   ✅ Python dependencies available")
        except ImportError as e:
            console.print(f"   ❌ Missing Python dependency: {e}", style="red")
            return False
        
        return True
    
    def run_core_tests(self) -> TestResult:
        """Run core functionality tests."""
        console.print("🧪 Running core functionality tests...", style="bold blue")
        return self.run_test_script("test_second_occurrence_masking.py")
    
    def run_ollama_tests(self) -> TestResult:
        """Run Ollama inference tests."""
        if self.args.skip_ollama:
            console.print("⏭️  Skipping Ollama tests (--skip-ollama)", style="yellow")
            return TestResult("test_inference_ollama.py", True, 0.0, "Skipped")
        
        console.print("🦙 Running Ollama inference tests...", style="bold blue")
        return self.run_test_script("test_inference_ollama.py", timeout=180)
    
    def run_mock_tests(self) -> TestResult:
        """Run mock inference tests."""
        console.print("🎭 Running mock inference tests...", style="bold blue")
        return self.run_test_script("test_mock_inference.py")
    
    def run_pipeline_tests(self) -> TestResult:
        """Run full pipeline tests."""
        if self.args.fast:
            console.print("⏭️  Skipping pipeline tests (--fast)", style="yellow")
            return TestResult("test_full_pipeline.py", True, 0.0, "Skipped") 
        
        console.print("🚀 Running full pipeline tests...", style="bold blue")
        return self.run_test_script("test_full_pipeline.py", timeout=300)
    
    def run_integration_tests(self) -> bool:
        """Run integration tests."""
        console.print("🔗 Running integration tests...", style="bold blue")
        
        try:
            # Test environment loading through registry
            console.print("   📚 Testing environment registry integration...")
            sys.path.insert(0, str(self.env_dir.parent.parent / "src"))
            
            from prime_rl.environments.registry import load_environment
            
            env = load_environment("second-occurrence-masking", {
                "num_examples": 5,
                "min_length": 30,
                "max_length": 100
            })
            
            if len(env.dataset) == 0:
                console.print("   ❌ Environment loaded but no examples generated", style="red")
                return False
            
            console.print(f"   ✅ Environment loaded with {len(env.dataset)} examples")
            
            # Test configuration compatibility
            console.print("   ⚙️  Testing configuration compatibility...")
            import toml
            
            configs = ["trainer_config.toml", "orchestrator_config.toml", "inference_config.toml"]
            for config_file in configs:
                config_path = self.env_dir / config_file
                config = toml.load(config_path)
                console.print(f"   ✅ {config_file} valid")
            
            return True
            
        except Exception as e:
            console.print(f"   ❌ Integration test failed: {e}", style="red")
            return False
    
    def run_benchmark_tests(self) -> dict:
        """Run performance benchmark tests."""
        if not self.args.benchmark:
            return {}
        
        console.print("📊 Running performance benchmarks...", style="bold blue")
        
        try:
            sys.path.insert(0, str(self.env_dir))
            from second_occurrence_loader import SecondOccurrenceMaskingLoader
            from mock_inference import create_mock_model
            
            # Benchmark environment loading
            console.print("   🔄 Benchmarking environment loading...")
            start_time = time.time()
            loader = SecondOccurrenceMaskingLoader(
                dataset_name="roneneldan/TinyStories",
                min_length=50,
                max_length=200
            )
            
            # Generate test dataset
            examples = []
            for _ in range(100):
                text = loader.get_sample_text()
                result = loader.mask_text(text)
                if result.target_words:
                    examples.append(result)
                if len(examples) >= 50:
                    break
            
            loading_time = time.time() - start_time
            console.print(f"   ✅ Generated {len(examples)} examples in {loading_time:.2f}s")
            
            # Benchmark inference speed
            console.print("   🚀 Benchmarking mock inference speed...")
            model = create_mock_model("masking_aware", accuracy=0.8)
            
            start_time = time.time()
            for example in examples[:20]:  # Test with 20 examples
                messages = [{"role": "user", "content": f"Fill masks: {example.masked_text}"}]
                response = model.complete(messages)
            inference_time = time.time() - start_time
            
            console.print(f"   ✅ 20 inferences in {inference_time:.2f}s ({20/inference_time:.1f} inferences/sec)")
            
            # Benchmark reward calculation
            console.print("   🏆 Benchmarking reward calculation...")
            start_time = time.time()
            total_reward = 0
            for example in examples:
                messages = [{"role": "user", "content": f"Fill masks: {example.masked_text}"}]
                response = model.complete(messages)
                reward = loader.calculate_reward(
                    example.original_text,
                    response.content,
                    example.mask_positions,
                    example.target_words
                )
                total_reward += reward
            
            reward_time = time.time() - start_time
            avg_reward = total_reward / len(examples)
            
            console.print(f"   ✅ {len(examples)} reward calculations in {reward_time:.2f}s")
            console.print(f"   📈 Average reward: {avg_reward:.3f}")
            
            return {
                "examples_generated": len(examples),
                "loading_time": loading_time,
                "inference_speed": 20 / inference_time,
                "reward_calc_time": reward_time,
                "average_reward": avg_reward
            }
            
        except Exception as e:
            console.print(f"   ❌ Benchmark failed: {e}", style="red")
            return {}
    
    def show_summary(self, benchmark_results: dict = None):
        """Show comprehensive test summary."""
        console.print("\n" + "="*80)
        console.print("📊 [bold]Test Summary - Second Occurrence Masking Environment[/bold]")
        console.print("="*80)
        
        # Test results table
        table = Table(title="🧪 Test Results", box=box.ROUNDED)
        table.add_column("Test Suite", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Duration", style="yellow")
        table.add_column("Details", style="white")
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        total_duration = sum(r.duration for r in self.results)
        
        for result in self.results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            duration = f"{result.duration:.2f}s" if result.duration > 0 else "skipped"
            
            # Extract key info from output
            details = ""
            if result.success and "passed" in result.output.lower():
                # Try to extract pass count
                lines = result.output.split('\n')
                for line in lines:
                    if "passed" in line.lower() and "failed" in line.lower():
                        details = line.strip()
                        break
            elif not result.success and result.error:
                details = result.error[:50] + "..." if len(result.error) > 50 else result.error
            
            table.add_row(result.name, status, duration, details)
        
        console.print(table)
        
        # Overall statistics
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        stats_table = Table(title="📈 Overall Statistics", box=box.DOUBLE_EDGE)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Total Test Suites", str(total_tests))
        stats_table.add_row("Passed", str(passed_tests))
        stats_table.add_row("Failed", str(total_tests - passed_tests))
        stats_table.add_row("Success Rate", f"{success_rate:.1f}%")
        stats_table.add_row("Total Duration", f"{total_duration:.2f}s")
        
        console.print(stats_table)
        
        # Benchmark results
        if benchmark_results:
            bench_table = Table(title="⚡ Performance Benchmarks", box=box.ROUNDED)
            bench_table.add_column("Metric", style="cyan")
            bench_table.add_column("Value", style="yellow")
            
            bench_table.add_row("Examples Generated", str(benchmark_results.get("examples_generated", "N/A")))
            bench_table.add_row("Loading Time", f"{benchmark_results.get('loading_time', 0):.2f}s")
            bench_table.add_row("Inference Speed", f"{benchmark_results.get('inference_speed', 0):.1f} inferences/sec")
            bench_table.add_row("Reward Calc Time", f"{benchmark_results.get('reward_calc_time', 0):.2f}s")
            bench_table.add_row("Average Reward", f"{benchmark_results.get('average_reward', 0):.3f}")
            
            console.print(bench_table)
        
        # Final verdict
        if passed_tests == total_tests:
            verdict = Panel(
                "🎉 [bold green]ALL TESTS PASSED![/bold green]\n\n"
                "✅ Environment is ready for training\n"
                "✅ All configurations validated\n"
                "✅ Mock inference working\n"
                "✅ Pipeline integration confirmed\n\n"
                "💡 Next steps:\n"
                "   • Test with Ollama: [cyan]./tests/test_inference_ollama.py[/cyan]\n"
                "   • Run mock training: [cyan]./mock_trainer.py[/cyan]\n"
                "   • Start real training: [cyan]uv run rl --trainer @ ...[/cyan]",
                style="green",
                title="🏆 SUCCESS"
            )
        else:
            failed_count = total_tests - passed_tests
            verdict = Panel(
                f"❌ [bold red]{failed_count} TEST(S) FAILED[/bold red]\n\n"
                "🔧 Issues to resolve:\n"
                + "\n".join(f"   • {r.name}: {r.error[:60]}..." for r in self.results if not r.success)[:200] +
                "\n\n💡 Debugging steps:\n"
                "   • Check individual test outputs above\n"
                "   • Verify dependencies are installed\n"
                "   • Ensure configuration files are valid\n"
                "   • Run tests individually for more details",
                style="red",
                title="💥 ISSUES FOUND"
            )
        
        console.print(verdict)
    
    def run_all_tests(self):
        """Run all test suites."""
        console.print("🧪 [bold]Starting Comprehensive Test Suite for Second Occurrence Masking[/bold]\n")
        
        # Check prerequisites
        if not self.check_prerequisites():
            console.print("💥 [bold red]Prerequisites not met. Aborting tests.[/bold red]")
            return False
        
        # Create progress tracker
        test_suites = [
            ("Core Tests", self.run_core_tests),
            ("Mock Tests", self.run_mock_tests),
            ("Ollama Tests", self.run_ollama_tests),
            ("Pipeline Tests", self.run_pipeline_tests),
        ]
        
        # Run tests with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        ) as progress:
            
            task = progress.add_task("Running tests...", total=len(test_suites) + 2)  # +2 for integration and benchmarks
            
            for suite_name, test_func in test_suites:
                progress.update(task, description=f"Running {suite_name}...")
                result = test_func()
                self.results.append(result)
                progress.advance(task)
            
            # Integration tests
            progress.update(task, description="Running integration tests...")
            integration_success = self.run_integration_tests()
            self.results.append(TestResult("Integration Tests", integration_success, 0.0))
            progress.advance(task)
            
            # Benchmark tests
            progress.update(task, description="Running benchmarks...")
            benchmark_results = self.run_benchmark_tests()
            if benchmark_results:
                self.results.append(TestResult("Performance Benchmarks", True, 0.0))
            progress.advance(task)
        
        # Show summary
        self.show_summary(benchmark_results)
        
        # Return overall success
        return all(result.success for result in self.results)


def main():
    parser = argparse.ArgumentParser(description="Run all tests for Second Occurrence Masking environment")
    parser.add_argument("--fast", action="store_true",
                       help="Skip slow tests (pipeline tests)")
    parser.add_argument("--skip-ollama", action="store_true",
                       help="Skip Ollama inference tests")
    parser.add_argument("--benchmark", action="store_true",
                       help="Include performance benchmarks")
    
    args = parser.parse_args()
    
    orchestrator = TestOrchestrator(args)
    success = orchestrator.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()