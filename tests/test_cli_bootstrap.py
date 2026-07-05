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
        self.assertIn("optimize-batches", result.stdout)
        self.assertIn("run-mainline", result.stdout)
        self.assertIn("run-passset-smoke", result.stdout)
        self.assertIn("run-v3-loop-smoke", result.stdout)
        self.assertIn("summarize-passsets", result.stdout)
        self.assertIn("audit-passes", result.stdout)
        self.assertIn("batchify", result.stdout)
        self.assertIn("eval-batches", result.stdout)
        self.assertIn("compare-baselines", result.stdout)
        self.assertIn("summarize-mainline", result.stdout)
        self.assertIn("summarize-final", result.stdout)
        self.assertIn("summarize-reduction", result.stdout)
        self.assertIn("export-evidence-pack", result.stdout)
        self.assertIn("diagnose-paths", result.stdout)
        self.assertIn("run-reduction-study", result.stdout)
        self.assertIn("run-budgeted-sensitivity", result.stdout)
        self.assertIn("summarize-exact-reduction-study", result.stdout)
        self.assertIn("summarize-core-v1-case-study", result.stdout)
        self.assertIn("summarize-components", result.stdout)
        self.assertIn("run-core-v1-budgeted-study", result.stdout)
        self.assertIn("select-and-run-exact-reference", result.stdout)
        self.assertIn("run-v2-extension-study", result.stdout)
        self.assertIn("replay-final-pipeline", result.stdout)
        self.assertIn("export-case-studies", result.stdout)
        self.assertIn("visualize-dag", result.stdout)

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

    def test_visualize_dag_help_runs(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "phasebatch", "visualize-dag", "--help"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--run-dir", result.stdout)
        self.assertIn("--formats", result.stdout)
        self.assertIn("--max-full-nodes", result.stdout)

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
        self.assertIn("--allow-sampled-batches", result.stdout)

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
        self.assertIn("--max-batches-per-state", result.stdout)
        self.assertIn("--max-frontier-states", result.stdout)
        self.assertIn("--batch-frontier-policy", result.stdout)
        self.assertIn("--validate-batches", result.stdout)
        self.assertIn("--allow-sampled-batches", result.stdout)

    def test_eval_batches_help_runs(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "phasebatch", "eval-batches", "--help"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--run-dir", result.stdout)
        self.assertIn("--objective", result.stdout)
        self.assertIn("ir-inst-count", result.stdout)
        self.assertIn("--recursive", result.stdout)

    def test_compare_baselines_help_runs(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "phasebatch", "compare-baselines", "--help"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--run-dir", result.stdout)
        self.assertIn("--passes", result.stdout)
        self.assertIn("--objective", result.stdout)
        self.assertIn("--methods", result.stdout)
        self.assertIn("--include-default-pipelines", result.stdout)
        self.assertIn("--random-trials", result.stdout)
        self.assertIn("--include-llvm-defaults", result.stdout)

    def test_replay_final_pipeline_help_runs(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "phasebatch", "replay-final-pipeline", "--help"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--run-dir", result.stdout)
        self.assertIn("--timeout", result.stdout)

    def test_run_mainline_help_runs(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "phasebatch", "run-mainline", "--help"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--inputs", result.stdout)
        self.assertIn("--overwrite", result.stdout)
        self.assertIn("--continue-on-error", result.stdout)
        self.assertIn("--max-depth", result.stdout)
        self.assertIn("--max-component-size", result.stdout)
        self.assertIn("--max-batches-per-state", result.stdout)
        self.assertIn("--eval-objective", result.stdout)

    def test_run_method_comparison_help_runs(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "phasebatch", "run-method-comparison", "--help"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--inputs", result.stdout)
        self.assertIn("--optimizer-mode", result.stdout)
        self.assertIn("--baseline-max-rounds", result.stdout)
        self.assertIn("--include-default-pipelines", result.stdout)

    def test_run_passset_smoke_help_runs(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "phasebatch", "run-passset-smoke", "--help"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--inputs", result.stdout)
        self.assertIn("--passsets", result.stdout)
        self.assertIn("--optimizer-mode", result.stdout)
        self.assertIn("--validate-batches", result.stdout)
        self.assertIn("--overwrite", result.stdout)
        self.assertIn("--continue-on-error", result.stdout)

    def test_run_v3_loop_smoke_help_runs(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "phasebatch", "run-v3-loop-smoke", "--help"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--inputs", result.stdout)
        self.assertIn("--passes", result.stdout)
        self.assertIn("--optimizer-mode", result.stdout)
        self.assertIn("--beam-width", result.stdout)
        self.assertIn("--validate-batches", result.stdout)
        self.assertIn("--continue-on-error", result.stdout)

    def test_summarize_passsets_help_runs(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "phasebatch", "summarize-passsets", "--help"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--inputs", result.stdout)
        self.assertIn("--out", result.stdout)

    def test_audit_passes_help_runs(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "phasebatch", "audit-passes", "--help"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--input", result.stdout)
        self.assertIn("--passes", result.stdout)
        self.assertIn("--out", result.stdout)
        self.assertIn("--timeout", result.stdout)
        self.assertIn("--jobs", result.stdout)

    def test_summarize_mainline_help_runs(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "phasebatch", "summarize-mainline", "--help"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--run-dir", result.stdout)

    def test_summarize_final_help_runs(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "phasebatch", "summarize-final", "--help"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--run-dir", result.stdout)

    def test_summarize_reduction_help_runs(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "phasebatch", "summarize-reduction", "--help"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--run-dir", result.stdout)

    def test_export_evidence_pack_help_runs(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "phasebatch", "export-evidence-pack", "--help"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--run-dir", result.stdout)

    def test_diagnose_paths_help_runs(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "phasebatch", "diagnose-paths", "--help"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--run-dir", result.stdout)
        self.assertIn("--baseline-dir", result.stdout)
        self.assertIn("--timeout", result.stdout)

    def test_run_reduction_study_help_runs(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "phasebatch", "run-reduction-study", "--help"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--inputs", result.stdout)
        self.assertIn("--optimizer-mode", result.stdout)
        self.assertIn("--max-rounds", result.stdout)
        self.assertIn("--max-states", result.stdout)
        self.assertIn("--validate-batches", result.stdout)
        self.assertIn("--continue-on-error", result.stdout)
        self.assertIn("--summarize-components", result.stdout)

    def test_run_budgeted_sensitivity_help_runs(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "phasebatch", "run-budgeted-sensitivity", "--help"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--inputs", result.stdout)
        self.assertIn("--beam-widths", result.stdout)
        self.assertIn("--max-states-list", result.stdout)
        self.assertIn("--max-batches-per-state", result.stdout)
        self.assertIn("--batch-frontier-policy", result.stdout)
        self.assertIn("--exact-reference", result.stdout)
        self.assertIn("--summarize-components", result.stdout)

    def test_run_core_v1_budgeted_study_help_runs(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "phasebatch", "run-core-v1-budgeted-study", "--help"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--inputs", result.stdout)
        self.assertIn("--beam-width", result.stdout)
        self.assertIn("--max-states", result.stdout)
        self.assertIn("--baseline-methods", result.stdout)
        self.assertIn("--random-trials", result.stdout)

    def test_select_and_run_exact_reference_help_runs(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "phasebatch", "select-and-run-exact-reference", "--help"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--budgeted-study-dir", result.stdout)
        self.assertIn("--num-easy", result.stdout)
        self.assertIn("--num-medium", result.stdout)
        self.assertIn("--num-hard", result.stdout)

    def test_run_v2_extension_study_help_runs(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "phasebatch", "run-v2-extension-study", "--help"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--v1-passes", result.stdout)
        self.assertIn("--v2-passes", result.stdout)
        self.assertIn("--random-trials", result.stdout)
        self.assertIn("--continue-on-error", result.stdout)

    def test_summarize_exact_reduction_study_help_runs(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "phasebatch", "summarize-exact-reduction-study", "--help"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--run-dirs", result.stdout)
        self.assertIn("--root-dir", result.stdout)
        self.assertIn("--out", result.stdout)
        self.assertIn("--label", result.stdout)
        self.assertIn("--summarize-components", result.stdout)

    def test_summarize_core_v1_case_study_help_runs(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "phasebatch", "summarize-core-v1-case-study", "--help"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--exact-method-summary", result.stdout)
        self.assertIn("--exact-reduction-summary", result.stdout)
        self.assertIn("--budgeted-sensitivity-summary", result.stdout)
        self.assertIn("--nbody-round-study", result.stdout)
        self.assertIn("--puzzle-case-study", result.stdout)

    def test_summarize_components_help_runs(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "phasebatch", "summarize-components", "--help"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--run-dir", result.stdout)
        self.assertIn("--run-dirs", result.stdout)
        self.assertIn("--out", result.stdout)


if __name__ == "__main__":
    unittest.main()
