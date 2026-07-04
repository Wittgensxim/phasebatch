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
        self.assertIn("explore", result.stdout)
        self.assertIn("explore-batches", result.stdout)
        self.assertIn("batchify", result.stdout)

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

    def test_explore_help_runs(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "phasebatch", "explore", "--help"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--max-depth", result.stdout)
        self.assertIn("--frontier-policy", result.stdout)

    def test_batchify_help_runs(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "phasebatch", "batchify", "--help"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--state-dir", result.stdout)
        self.assertIn("--max-component-size", result.stdout)
        self.assertIn("--max-batch-candidates", result.stdout)
        self.assertIn("--validate-batches", result.stdout)

    def test_explore_batches_help_runs(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "phasebatch", "explore-batches", "--help"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--input", result.stdout)
        self.assertIn("--max-depth", result.stdout)
        self.assertIn("--max-component-size", result.stdout)
        self.assertIn("--max-batch-candidates", result.stdout)
        self.assertIn("--validate-batches", result.stdout)
        self.assertIn("--allow-sampled-batches", result.stdout)


if __name__ == "__main__":
    unittest.main()
