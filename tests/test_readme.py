import unittest
from pathlib import Path


class ReadmeTests(unittest.TestCase):
    def test_readme_documents_batchify_command(self) -> None:
        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")

        self.assertIn("python -m phasebatch batchify", readme)
        self.assertIn("--state-dir", readme)
        self.assertIn("--max-component-size", readme)
        self.assertIn("--max-batch-candidates", readme)
        self.assertIn("does not run opt", readme)


if __name__ == "__main__":
    unittest.main()
