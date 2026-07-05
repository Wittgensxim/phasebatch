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
        self.assertIn("batch_correctness.csv", readme)
        self.assertIn("coverage_report.csv", readme)
        self.assertIn("aggregate_coverage_summary.csv", readme)
        self.assertIn("footprint_overlap.csv", readme)
        self.assertIn("aggregate_overlap_summary.csv", readme)
        self.assertIn("diagnostics only", readme)
        self.assertIn("batch_validation.csv", readme)
        self.assertIn("python -m phasebatch explore-batches", readme)
        self.assertIn("python -m phasebatch run-mainline", readme)
        self.assertIn("mainline_runs.csv", readme)
        self.assertIn("mainline_aggregate_overlap.csv", readme)
        self.assertIn("mainline_summary.md", readme)
        self.assertIn("python -m phasebatch summarize-mainline", readme)
        self.assertIn("python -m phasebatch eval-batches", readme)
        self.assertIn("python -m phasebatch optimize-batches", readme)
        self.assertIn("--mode exact", readme)
        self.assertIn("--max-states", readme)
        self.assertIn("exact_status.txt", readme)
        self.assertIn("exact_incomplete", readme)
        self.assertIn("optimized_pipeline.txt", readme)
        self.assertIn("chosen_path.csv", readme)
        self.assertIn("objective is used only for path selection", readme)
        self.assertIn("--objective ir-inst-count", readme)
        self.assertIn("objective_signal.csv", readme)
        self.assertIn("objective_eval.csv", readme)
        self.assertIn("aggregate_objective_signal.csv", readme)
        self.assertIn("--eval-objective ir-inst-count", readme)
        self.assertIn("objective_summary.md", readme)
        self.assertIn("batch_state_transitions.csv", readme)
        self.assertIn("aggregate_batch_summary.csv", readme)
        self.assertIn("batch_explore_summary.md", readme)
        self.assertIn("--allow-sampled-batches", readme)
        self.assertIn("--max-batches-per-state", readme)
        self.assertIn("--max-frontier-states", readme)
        self.assertIn("--batch-frontier-policy", readme)
        self.assertIn("skipped_batches.csv", readme)
        self.assertIn("does not run opt", readme)


if __name__ == "__main__":
    unittest.main()
