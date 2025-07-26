#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["requests", "openai"]
# requires-python = ">=3.8"
# ///
"""
Run all tests for the thesaurus replacement environment.
Combines core functionality tests and Ollama integration tests.

Run with: ./tests/run_all_tests.py
"""

import subprocess
import sys
from pathlib import Path

def run_test_script(script_path: Path, test_name: str) -> bool:
    """Run a test script and return success status."""
    print(f"\n{'='*60}")
    print(f"🧪 Running {test_name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [str(script_path)],
            cwd=script_path.parent.parent,
            capture_output=False,
            text=True
        )
        
        success = result.returncode == 0
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"\n{status} {test_name}")
        return success
        
    except Exception as e:
        print(f"❌ FAILED {test_name}: {e}")
        return False

def main():
    """Run all test suites."""
    print("🚀 Running All Thesaurus Replacement Environment Tests")
    
    tests_dir = Path(__file__).parent
    
    test_suites = [
        (tests_dir / "test_thesaurus.py", "Core Environment Tests"),
        (tests_dir / "test_inference_ollama.py", "Ollama Integration Tests"),
        (tests_dir / "test_mock_inference.py", "Mock Inference Tests"),
    ]
    
    results = []
    for script_path, test_name in test_suites:
        if script_path.exists():
            result = run_test_script(script_path, test_name)
            results.append(result)
        else:
            print(f"❌ Test script not found: {script_path}")
            results.append(False)
    
    print(f"\n{'='*60}")
    print(f"📊 OVERALL TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Test suites passed: {sum(results)}/{len(results)}")
    
    for i, (script_path, test_name) in enumerate(test_suites):
        status = "✅" if results[i] else "❌"
        print(f"  {status} {test_name}")
    
    if all(results):
        print("\n🎉 All test suites passed!")
        print("✅ Environment is ready for production use")
        return 0
    else:
        print("\n💥 Some test suites failed")
        print("ℹ️  Check individual test outputs for details")
        return 1

if __name__ == "__main__":
    sys.exit(main())