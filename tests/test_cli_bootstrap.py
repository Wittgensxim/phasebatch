import subprocess
import sys
import unittest
from pathlib import Path


class CliBootstrapTests(unittest.TestCase):
    def test_module_help_runs(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "phasebatch", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("analyze", result.stdout)
        self.assertIn("batch", result.stdout)

    def test_analyze_help_runs(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "phasebatch", "analyze", "--help"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--input", result.stdout)
        self.assertIn("--passes", result.stdout)


if __name__ == "__main__":
    unittest.main()
