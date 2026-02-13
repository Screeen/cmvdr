#!/usr/bin/env python3
"""
Integration test for cmvdr-plot CLI with PathSafeLoader.

This test verifies that:
1. PathSafeLoader can be imported
2. PathSafeLoader can deserialize pathlib.PosixPath from YAML
3. The CLI help message works
"""
import sys
import tempfile
from pathlib import Path

def test_path_safe_loader():
    """Test that PathSafeLoader can deserialize Path objects from YAML."""
    from cmvdr.util.config import PathSafeLoader
    import yaml

    test_yaml = """datasets_path: !!python/object/apply:pathlib.PosixPath
- /Users
- test
fs: 16000
"""

    try:
        result = yaml.load(test_yaml, Loader=PathSafeLoader)
        assert isinstance(result['datasets_path'], Path), f"Expected Path, got {type(result['datasets_path'])}"
        assert str(result['datasets_path']) == "/Users/test", f"Unexpected path: {result['datasets_path']}"
        assert result['fs'] == 16000
        print("✓ PathSafeLoader test passed")
        return True
    except Exception as e:
        print(f"✗ PathSafeLoader test failed: {e}")
        return False


def test_cli_help():
    """Test that the CLI can display help."""
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "cmvdr.cli.plot_results", "-h"],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        output = (result.stdout or "") + (result.stderr or "")
        if "Regenerate plots from saved experiment results" in output:
            print("✓ CLI help test passed")
            return True

    print(f"✗ CLI help test failed (returncode={result.returncode})")
    return False


if __name__ == "__main__":
    print("Running cmvdr-plot integration tests...\n")

    tests = [
        ("PathSafeLoader", test_path_safe_loader),
        ("CLI help", test_cli_help),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {name} test error: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)

