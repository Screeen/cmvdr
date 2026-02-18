import subprocess
import sys
import unittest


class TestCmvdrPlotCli(unittest.TestCase):
    def test_help_message(self):
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

