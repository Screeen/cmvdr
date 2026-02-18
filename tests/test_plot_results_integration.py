"""
Integration test for cmvdr-plot CLI with PathSafeLoader.

This test verifies that:
1. PathSafeLoader can be imported
2. PathSafeLoader can deserialize pathlib.PosixPath from YAML
3. The CLI help message works
"""
import subprocess
import sys
import unittest
from pathlib import Path

import yaml

from cmvdr.util.config import PathSafeLoader


class TestPlotResultsIntegration(unittest.TestCase):
    def test_path_safe_loader(self):
        """Test that PathSafeLoader can deserialize Path objects from YAML."""
        test_yaml = """datasets_path: !!python/object/apply:pathlib.PosixPath
- /Users
- test
fs: 16000
"""

        result = yaml.load(test_yaml, Loader=PathSafeLoader)
        self.assertIsInstance(result['datasets_path'], Path)
        self.assertEqual(str(result['datasets_path']), "/Users/test")
        self.assertEqual(result['fs'], 16000)

    def test_cli_help(self):
        """Test that the CLI can display help."""
        result = subprocess.run(
            [sys.executable, "-m", "cmvdr.cli.plot_results", "-h"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        output = (result.stdout or "") + (result.stderr or "")
        self.assertIn("Regenerate plots from saved experiment results", output)


if __name__ == "__main__":
    unittest.main()

