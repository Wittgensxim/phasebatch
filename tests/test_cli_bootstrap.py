import subprocess
import sys
import tempfile
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

    def test_analyze_parses_required_arguments(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            passes = Path(tmp) / "passes.yaml"
            passes.write_text("passes:\n  - mem2reg\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "phasebatch",
                    "analyze",
                    "--input",
                    "x.c",
                    "--out",
                    "outputs/x",
                    "--passes",
                    str(passes),
                ],
                cwd=repo,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"command": "analyze"', result.stdout)
        self.assertIn('"input": "x.c"', result.stdout)
        self.assertIn('"pass_count": 1', result.stdout)


if __name__ == "__main__":
    unittest.main()
