import unittest
from pathlib import Path


class ReadmeTests(unittest.TestCase):
    def test_readme_documents_batchify_command(self) -> None:
        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")

        self.assertIn("python -m phasebatch batchify", readme)
        self.assertIn("--state-dir", readme)
        self.assertIn("--max-component-size", readme)
        self.assertIn("--max-batch-candidates", readme)
        self.assertIn("--validate-batches", readme)
        self.assertIn("batch_validation.csv", readme)
        self.assertIn("python -m phasebatch explore-batches", readme)
        self.assertIn("batch_state_transitions.csv", readme)
        self.assertIn("batch_explore_summary.md", readme)
        self.assertIn("--allow-sampled-batches", readme)
        self.assertIn("--max-batches-per-state", readme)
        self.assertIn("--max-frontier-states", readme)
        self.assertIn("--batch-frontier-policy", readme)
        self.assertIn("skipped_batches.csv", readme)
        self.assertIn("does not run opt", readme)


if __name__ == "__main__":
    unittest.main()
