#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["requests", "openai", "rich"]
# requires-python = ">=3.8"
# ///

"""
Test orchestration system for swap tracking environment.
Runs complete test suite and provides comprehensive reporting.
"""

import subprocess
import sys
import time
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.columns import Columns
except ImportError:
    print("Installing rich for better output...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.columns import Columns

console = Console()


class TestOrchestrator:
    """Orchestrates and manages all test suites for swap tracking environment."""
    
    def __init__(self):
        self.env_dir = Path(__file__).parent.parent
        self.tests_dir = self.env_dir / "tests"
        self.results = {}
        
        # Define test suites
        self.test_suites = [
            {
                "name": "Core Functionality",
                "script": "test_swap_tracking.py",
                "description": "Basic environment logic and data generation",
                "required": True,
                "estimated_time": 10
            },
            {
                "name": "Ollama Integration",
                "script": "test_inference_ollama.py", 
                "description": "Local inference with Ollama models",
                "required": False,
                "estimated_time": 30
            },
            {
                "name": "Mock Inference",
                "script": "test_mock_inference.py",
                "description": "Mock inference system for training pipeline",
                "required": True,
                "estimated_time": 20
            },
            {
                "name": "Full Pipeline",
                "script": "test_full_pipeline.py",
                "description": "Complete training workflow validation",
                "required": True,
                "estimated_time": 15
            }
        ]
    
    def check_prerequisites(self) -> bool:
        """Check if all test scripts exist and are executable."""
        console.print("[bold blue]Checking Prerequisites[/bold blue]")
        
        missing_scripts = []
        
        for suite in self.test_suites:
            script_path = self.tests_dir / suite["script"]
            if not script_path.exists():
                missing_scripts.append(suite["script"])
            elif not script_path.is_file():
                missing_scripts.append(f"{suite['script']} (not a file)")
        
        if missing_scripts:
            console.print("❌ Missing test scripts:")
            for script in missing_scripts:
                console.print(f"   • {script}")
            return False
        
        console.print("✓ All test scripts found")
        
        # Check core loader
        loader_path = self.env_dir / "swap_tracking_loader.py"
        if not loader_path.exists():
            console.print("❌ Missing core loader: swap_tracking_loader.py")
            return False
        
        console.print("✓ Core loader found")
        
        # Check config files
        config_files = [
            "trainer_config.toml",
            "orchestrator_config.toml", 
            "inference_config.toml"
        ]
        
        missing_configs = []
        for config_file in config_files:
            if not (self.env_dir / config_file).exists():
                missing_configs.append(config_file)
        
        if missing_configs:
            console.print("❌ Missing config files:")
            for config in missing_configs:
                console.print(f"   • {config}")
            return False
        
        console.print("✓ All config files found")
        return True
    
    def run_test_suite(self, suite: dict) -> dict:
        """Run a single test suite."""
        script_path = self.tests_dir / suite["script"]
        
        console.print(f"[yellow]Running {suite['name']}...[/yellow]")
        
        try:
            start_time = time.time()
            
            # Run test script
            result = subprocess.run(
                [str(script_path)],
                cwd=str(self.env_dir),
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            success = result.returncode == 0
            
            return {
                "name": suite["name"],
                "success": success,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "required": suite["required"]
            }
            
        except subprocess.TimeoutExpired:
            return {
                "name": suite["name"],
                "success": False,
                "duration": 120,
                "stdout": "",
                "stderr": "Test timed out after 2 minutes",
                "returncode": -1,
                "required": suite["required"]
            }
        except Exception as e:
            return {
                "name": suite["name"],
                "success": False,
                "duration": 0,
                "stdout": "",
                "stderr": f"Failed to run test: {e}",
                "returncode": -1,
                "required": suite["required"]
            }
    
    def display_test_progress(self, current_suite: str, progress: float) -> None:
        """Display current test progress."""
        progress_bar = "█" * int(progress * 20) + "░" * (20 - int(progress * 20))
        console.print(f"[bold]Progress:[/bold] [{progress_bar}] {progress*100:.1f}%")
        console.print(f"[bold]Current:[/bold] {current_suite}")
    
    def generate_summary_report(self) -> None:
        """Generate comprehensive summary report."""
        console.print("\n" + "=" * 70)
        console.print("[bold green]SWAP TRACKING ENVIRONMENT - TEST SUMMARY[/bold green]")
        console.print("=" * 70)
        
        # Results table
        results_table = Table(title="Test Results")
        results_table.add_column("Test Suite", justify="left")
        results_table.add_column("Status", justify="center")
        results_table.add_column("Duration", justify="right")
        results_table.add_column("Required", justify="center")
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r["success"])
        failed_tests = total_tests - passed_tests
        
        required_tests = [r for r in self.results.values() if r["required"]]
        required_passed = sum(1 for r in required_tests if r["success"])
        required_failed = len(required_tests) - required_passed
        
        for result in self.results.values():
            status = "✓ PASS" if result["success"] else "❌ FAIL"
            duration = f"{result['duration']:.1f}s"
            required = "Yes" if result["required"] else "No"
            
            results_table.add_row(
                result["name"],
                status,
                duration,
                required
            )
        
        console.print(results_table)
        
        # Summary panels
        summary_panels = []
        
        # Overall results
        overall_panel = Panel(
            f"[bold]Total Tests:[/bold] {total_tests}\n"
            f"[bold green]Passed:[/bold green] {passed_tests}\n"
            f"[bold red]Failed:[/bold red] {failed_tests}\n"
            f"[bold]Success Rate:[/bold] {(passed_tests/total_tests)*100:.1f}%",
            title="📊 Overall Results"
        )
        summary_panels.append(overall_panel)
        
        # Required tests
        required_panel = Panel(
            f"[bold]Required Tests:[/bold] {len(required_tests)}\n"
            f"[bold green]Passed:[/bold green] {required_passed}\n"
            f"[bold red]Failed:[/bold red] {required_failed}\n"
            f"[bold]Success Rate:[/bold] {(required_passed/len(required_tests))*100:.1f}%",
            title="🔧 Required Tests"
        )
        summary_panels.append(required_panel)
        
        console.print(Columns(summary_panels))
        
        # Failure details
        failed_results = [r for r in self.results.values() if not r["success"]]
        if failed_results:
            console.print("\n[bold red]FAILURE DETAILS[/bold red]")
            for result in failed_results:
                console.print(f"\n[bold red]❌ {result['name']}[/bold red]")
                if result["stderr"]:
                    console.print(f"[red]Error:[/red] {result['stderr'][:200]}...")
                if result["returncode"] != 0:
                    console.print(f"[red]Exit code:[/red] {result['returncode']}")
        
        # Final assessment
        console.print("\n" + "=" * 70)
        
        if required_failed == 0:
            console.print("[bold green]🎉 ALL REQUIRED TESTS PASSED![/bold green]")
            console.print("\n[bold]Environment Status:[/bold] ✅ Ready for use")
            console.print("\n[bold]What works:[/bold]")
            console.print("• Core environment logic")
            console.print("• Mock inference system")
            console.print("• Training pipeline integration")
            console.print("• Configuration files")
            
            if failed_tests > 0:
                console.print(f"\n[bold yellow]Note:[/bold yellow] {failed_tests} optional tests failed")
                console.print("These are likely due to missing external dependencies (e.g., Ollama)")
            
            console.print("\n[bold]Next steps:[/bold]")
            console.print("1. Copy mock config: cp inference_config_mock.toml inference_config.toml")
            console.print("2. Test mock training: ./mock_trainer.py --steps 10")
            console.print("3. Run real training when ready")
            
        else:
            console.print("[bold red]❌ REQUIRED TESTS FAILED![/bold red]")
            console.print(f"\n[bold]Environment Status:[/bold] ❌ Not ready ({required_failed} critical failures)")
            console.print("\n[bold]Failed required tests:[/bold]")
            for result in failed_results:
                if result["required"]:
                    console.print(f"• {result['name']}")
            
            console.print("\n[bold]Recommended actions:[/bold]")
            console.print("1. Check error messages above")
            console.print("2. Fix failing tests")
            console.print("3. Re-run test suite")
    
    def run_all_tests(self, include_optional: bool = True) -> bool:
        """Run all test suites with progress tracking."""
        console.print("[bold green]SWAP TRACKING ENVIRONMENT - COMPREHENSIVE TEST SUITE[/bold green]")
        console.print("=" * 70)
        
        # Check prerequisites
        if not self.check_prerequisites():
            console.print("\n[bold red]❌ Prerequisites check failed![/bold red]")
            return False
        
        # Filter test suites
        suites_to_run = self.test_suites
        if not include_optional:
            suites_to_run = [s for s in self.test_suites if s["required"]]
        
        console.print(f"\n[bold]Running {len(suites_to_run)} test suites...[/bold]")
        
        # Run tests with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for i, suite in enumerate(suites_to_run):
                task = progress.add_task(f"Running {suite['name']}...", total=None)
                
                result = self.run_test_suite(suite)
                self.results[suite["name"]] = result
                
                status = "✓" if result["success"] else "❌"
                progress.update(task, description=f"{status} {suite['name']} ({result['duration']:.1f}s)")
                progress.remove_task(task)
        
        # Generate report
        self.generate_summary_report()
        
        # Return success if all required tests passed
        required_results = [r for r in self.results.values() if r["required"]]
        all_required_passed = all(r["success"] for r in required_results)
        
        return all_required_passed


def main():
    """Main test orchestration function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Swap Tracking Environment Test Suite")
    parser.add_argument("--required-only", action="store_true", 
                       help="Run only required tests (skip optional like Ollama)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show verbose output from tests")
    
    args = parser.parse_args()
    
    orchestrator = TestOrchestrator()
    
    try:
        success = orchestrator.run_all_tests(include_optional=not args.required_only)
        
        if args.verbose:
            console.print("\n[bold]VERBOSE OUTPUT:[/bold]")
            for name, result in orchestrator.results.items():
                console.print(f"\n[bold]{name}:[/bold]")
                if result["stdout"]:
                    console.print(result["stdout"])
                if result["stderr"]:
                    console.print(f"[red]{result['stderr']}[/red]")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        console.print("\n[bold yellow]⚠ Tests interrupted by user[/bold yellow]")
        return 1
    except Exception as e:
        console.print(f"\n[bold red]❌ Test orchestration failed: {e}[/bold red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())