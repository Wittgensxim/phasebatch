import unittest
from pathlib import Path


class ReadmeTests(unittest.TestCase):
    def test_readme_documents_only_the_maintained_mainline(self) -> None:
        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")

        for command in [
            "python.exe -m phasebatch batchify",
            "python.exe -m phasebatch explore-batches",
            "python.exe -m phasebatch optimize-batches",
            "python.exe -m phasebatch optimize-staged",
            "python.exe -m phasebatch verify-opt-worker",
            "python.exe -m phasebatch run-advisor-report-zh",
            "python.exe -m phasebatch summarize-advisor-report-zh",
        ]:
            with self.subTest(command=command):
                self.assertIn(command, readme)

        for artifact in [
            "batch_validation.csv",
            "batch_correctness.csv",
            "coverage_report.csv",
            "footprint_overlap.csv",
            "optimized_pipeline.txt",
            "chosen_path.csv",
            "exact_status.txt",
        ]:
            with self.subTest(artifact=artifact):
                self.assertIn(artifact, readme)

        self.assertIn("--batch-construction-mode pairwise", readme)
        self.assertIn("--opt-backend worker", readme)
        self.assertIn("diagnostics only", readme)

    def test_readme_does_not_advertise_removed_commands(self) -> None:
        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")

        for removed in [
            "compare-cegar-pairwise",
            "run-mainline",
            "run-method-comparison",
            "run-budgeted-sensitivity",
            "run-core-v1-budgeted-study",
            "run-v2-extension-study",
            "run-v3-loop-smoke",
        ]:
            with self.subTest(command=removed):
                self.assertNotIn(removed, readme)


if __name__ == "__main__":
    unittest.main()
