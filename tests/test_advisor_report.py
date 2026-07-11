import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.advisor_report import run_advisor_report_zh, summarize_advisor_report_zh


class AdvisorReportTests(unittest.TestCase):
    def test_summarize_does_not_run_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
            mock.patch("phasebatch.advisor_report.optimize_batches") as optimizer, \
            mock.patch("phasebatch.advisor_report.summarize_advisor_metrics", return_value={"programs": 0}), \
            mock.patch("phasebatch.advisor_report.generate_advisor_figures", return_value={"figures": 9}), \
            mock.patch("phasebatch.advisor_report.generate_advisor_dags", return_value={"programs": 0}), \
            mock.patch("phasebatch.advisor_report.generate_advisor_markdown", return_value={"advisor_report_zh": "report.md"}):
            result = summarize_advisor_report_zh(Path(tmp))

        optimizer.assert_not_called()
        self.assertEqual(result["metrics"]["programs"], 0)

    def test_summarize_recovers_rolling_scope_from_program_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp)
            run_dir = study / "programs" / "demo" / "optimize"
            run_dir.mkdir(parents=True)
            (run_dir / "metadata.json").write_text(
                '{"mode":"rolling-exact","rolling_window_depth":2,'
                '"max_rolling_windows":0,"rolling_windows_completed":3,'
                '"rolling_committed_depth":6,"rolling_closure_reason":"no_active_passes"}',
                encoding="utf-8",
            )
            with mock.patch("phasebatch.advisor_report.summarize_advisor_metrics", return_value={"programs": 1}), \
                mock.patch("phasebatch.advisor_report.generate_advisor_figures", return_value={"figures": 9}), \
                mock.patch("phasebatch.advisor_report.generate_advisor_dags", return_value={"programs": 1}), \
                mock.patch("phasebatch.advisor_report.generate_advisor_markdown", return_value={"advisor_report_zh": "report.md"}) as markdown:
                summarize_advisor_report_zh(study)

            metadata = markdown.call_args.kwargs["metadata"]

        self.assertEqual(metadata["rolling_window_depth"], 2)
        self.assertEqual(metadata["max_rolling_windows"], 0)
        self.assertEqual(metadata["rolling_windows_completed"], 3)
        self.assertEqual(metadata["rolling_committed_depth"], 6)
        self.assertEqual(metadata["rolling_closure_reason"], "no_active_passes")

    def test_run_enforces_stable_mainline_and_resume_skips_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "SingleSource" / "demo.c"
            source.parent.mkdir(parents=True)
            source.write_text("int main(void) { return 0; }\n", encoding="utf-8")
            passes = root / "passes.yaml"
            passes.write_text("passes:\n  - instcombine\n", encoding="utf-8")
            out = root / "study"
            discovery = {
                "selected": [{"name": "demo", "path": str(source)}],
                "selected_count": 1,
                "benchmark_candidates_csv": str(out / "benchmark_candidates.csv"),
                "benchmark_selection_csv": str(out / "benchmark_selection.csv"),
                "selected_benchmarks_yaml": str(out / "selected_benchmarks.yaml"),
            }

            def fake_optimize(_input, optimize_dir, _passes, **_kwargs):
                optimize_dir.mkdir(parents=True, exist_ok=True)
                (optimize_dir / "optimize_summary.md").write_text("ok\n", encoding="utf-8")
                _write_csv(
                    optimize_dir / "states.csv",
                    ["state_id", "depth"],
                    [{"state_id": "S0000", "depth": "0"}],
                )
                _write_csv(
                    optimize_dir / "states" / "S0000" / "pass_profile.csv",
                    ["pass", "success", "active"],
                    [{"pass": "instcombine", "success": "true", "active": "false"}],
                )
                return {"states": 1, "transitions": 0}

            with mock.patch("phasebatch.advisor_report.discover_advisor_benchmarks", return_value=discovery), \
                mock.patch("phasebatch.advisor_report.optimize_batches", side_effect=fake_optimize) as optimizer, \
                mock.patch("phasebatch.advisor_report.summarize_advisor_report_zh", return_value={"ok": True}), \
                mock.patch("phasebatch.advisor_report.collect_toolchain", return_value={"tools": {}}):
                first = run_advisor_report_zh(
                    test_suite_root=root,
                    out_dir=out,
                    passes_path=passes,
                    num_programs=1,
                    resume=False,
                )
                second = run_advisor_report_zh(
                    test_suite_root=root,
                    out_dir=out,
                    passes_path=passes,
                    num_programs=1,
                    resume=True,
                )

            kwargs = optimizer.call_args.kwargs
            run_rows = _read_csv(out / "study_runs.csv")

        self.assertEqual(optimizer.call_count, 1)
        self.assertEqual(kwargs["pair_testing_mode"], "full")
        self.assertEqual(kwargs["batch_construction_mode"], "pairwise")
        self.assertEqual(kwargs["batch_validation_mode"], "auto")
        self.assertTrue(kwargs["validate_batches"])
        self.assertEqual(kwargs["budgeted_validation_strategy"], "all")
        self.assertEqual(kwargs["mode"], "rolling-exact")
        self.assertEqual(kwargs["rolling_window_depth"], 2)
        self.assertEqual(kwargs["rolling_frontier_width"], 5)
        self.assertEqual(kwargs["max_rolling_windows"], 0)
        self.assertIsNone(kwargs["max_pairs"])
        self.assertFalse(kwargs["allow_sampled_batches"])
        self.assertEqual(run_rows[0]["status"], "success")
        self.assertEqual(first["successes"], 1)
        self.assertEqual(second["successes"], 1)

    def test_rejects_non_stable_pair_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "pair_testing_mode must be full"):
                run_advisor_report_zh(
                    test_suite_root=root,
                    out_dir=root / "out",
                    passes_path=root / "passes.yaml",
                    pair_testing_mode="lazy",
                )


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
