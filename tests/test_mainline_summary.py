import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from phasebatch.mainline_summary import generate_mainline_summary


class MainlineSummaryTests(unittest.TestCase):
    def test_summary_generated_with_all_aggregate_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _write_complete_run(run_dir)

            path = generate_mainline_summary(run_dir)
            text = path.read_text(encoding="utf-8")

        self.assertEqual(path.name, "mainline_summary.md")
        self.assertIn("# Mainline Summary", text)
        self.assertIn("- total programs: 2", text)
        self.assertIn("- successful programs: 1", text)
        self.assertIn("- failed programs: 1", text)
        self.assertIn("- total states: 3", text)
        self.assertIn("- total batch transitions: 2", text)
        self.assertIn("## Program Status", text)
        self.assertIn("| ok | success | 3 | 2 | 1 | 10.00 |  |", text)
        self.assertIn("| bad | failed | 0 | 0 | 0 | 5.00 | boom |", text)
        self.assertIn("## State Relation Summary", text)
        self.assertIn("| ok | 1 | 2 | 3.00 | 3.00 | 2.00 | 1.00 | 0.00 |", text)
        self.assertIn("## Batch Reduction Summary", text)
        self.assertIn("avg log10 naive orderings", text)
        self.assertNotIn("avg naive orderings", text)
        self.assertIn("| ok | 0 | 1 | 2.00 | 3.00 | 3.00 | 10.00 | 2 | 1 |", text)
        self.assertIn("## Batch Validation Summary", text)
        self.assertIn("avg candidates | total candidates", text)
        self.assertIn("| ok | 0 | 2.00 | 2 | 1 | 1 | 0 | 0 | 0 | 0 |", text)
        self.assertIn("## Coverage Summary", text)
        self.assertIn("terminal not covered due max depth", text)
        self.assertNotIn("not executed due max depth", text)
        self.assertIn("## Coarse Footprint / Overlap Diagnostics", text)
        self.assertIn("Coarse footprint labels are diagnostic only and are not used as hard independence proof.", text)
        self.assertIn("## Key Observations", text)
        self.assertIn("## Missing Outputs / Failures", text)

    def test_summary_generated_with_some_missing_aggregate_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _write_csv(
                run_dir / "mainline_runs.csv",
                ["program", "input_path", "output_dir", "status", "error_message", "total_time_ms"],
                [{"program": "ok", "input_path": "ok.c", "output_dir": "ok", "status": "success", "error_message": "", "total_time_ms": "1"}],
            )

            path = generate_mainline_summary(run_dir)
            text = path.read_text(encoding="utf-8")

        self.assertIn("## Warnings", text)
        self.assertIn("missing input CSV: mainline_aggregate_states.csv", text)
        self.assertIn("## Program Status", text)

    def test_dropped_coverage_creates_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _write_complete_run(run_dir, dropped="2")

            text = generate_mainline_summary(run_dir).read_text(encoding="utf-8")

        self.assertIn("**WARNING**", text)
        self.assertIn("| ok | 0 | 4 | 1 | 0 | 0 | 0 | 3 | 0 | **WARNING** 2 |", text)

    def test_terminal_depth_coverage_is_separate_from_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _write_complete_run(run_dir, terminal_not_executed="5")

            text = generate_mainline_summary(run_dir).read_text(encoding="utf-8")

        self.assertIn("| ok | 0 | 2.00 | 2 | 1 | 1 | 0 | 0 | 0 | 0 |", text)
        self.assertIn("| ok | 1 | 3.00 | 3 | 0 | 0 | 0 | 0 | 0 | 3 |", text)
        self.assertIn("| ok | 1 | 5 | 0 | 0 | 0 | 0 | 0 | 5 | 0 |", text)

    def test_failed_program_appears_in_program_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _write_complete_run(run_dir)

            text = generate_mainline_summary(run_dir).read_text(encoding="utf-8")

        self.assertIn("| bad | failed | 0 | 0 | 0 | 5.00 | boom |", text)

    def test_summarize_mainline_command_regenerates_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _write_complete_run(run_dir)
            summary_path = run_dir / "mainline_summary.md"
            summary_path.write_text("stale\n", encoding="utf-8")

            result = subprocess.run(
                [sys.executable, "-m", "phasebatch", "summarize-mainline", "--run-dir", str(run_dir)],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )
            text = summary_path.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("mainline_summary.md", result.stdout)
        self.assertIn("# Mainline Summary", text)
        self.assertNotEqual(text, "stale\n")


def _write_complete_run(run_dir: Path, dropped: str = "0", terminal_not_executed: str = "0") -> None:
    _write_csv(
        run_dir / "mainline_runs.csv",
        ["program", "input_path", "output_dir", "status", "error_message", "total_time_ms"],
        [
            {"program": "ok", "input_path": "ok.c", "output_dir": "ok", "status": "success", "error_message": "", "total_time_ms": "10.00"},
            {"program": "bad", "input_path": "bad.c", "output_dir": "bad", "status": "failed", "error_message": "boom", "total_time_ms": "5.00"},
        ],
    )
    _write_csv(
        run_dir / "mainline_aggregate_states.csv",
        [
            "program",
            "depth",
            "num_states",
            "avg_active_passes",
            "avg_pairs_tested",
            "avg_dynamic_commute",
            "avg_order_sensitive",
            "avg_unknown",
        ],
        [
            {"program": "ok", "depth": "0", "num_states": "1", "avg_active_passes": "4.00", "avg_pairs_tested": "6.00", "avg_dynamic_commute": "4.00", "avg_order_sensitive": "2.00", "avg_unknown": "0.00"},
            {"program": "ok", "depth": "1", "num_states": "2", "avg_active_passes": "3.00", "avg_pairs_tested": "3.00", "avg_dynamic_commute": "2.00", "avg_order_sensitive": "1.00", "avg_unknown": "0.00"},
        ],
    )
    _write_csv(
        run_dir / "mainline_aggregate_batches.csv",
        [
            "program",
            "depth",
            "states",
            "avg_candidates",
            "avg_batch_size",
            "avg_reduction",
            "executed",
            "skipped",
            "all_permutations_same",
            "sampled_same",
            "mismatch",
            "failed",
            "not_validated",
        ],
        [
            {"program": "ok", "depth": "0", "states": "1", "avg_candidates": "2.00", "avg_batch_size": "3.00", "avg_reduction": "10.00", "executed": "2", "skipped": "1", "all_permutations_same": "1", "sampled_same": "1", "mismatch": "0", "failed": "0", "not_validated": "0"},
            {"program": "ok", "depth": "1", "states": "1", "avg_candidates": "3.00", "avg_batch_size": "2.00", "avg_reduction": "2.00", "executed": "0", "skipped": "0", "all_permutations_same": "0", "sampled_same": "0", "mismatch": "0", "failed": "0", "not_validated": "0"},
        ],
    )
    _write_csv(
        run_dir / "mainline_aggregate_coverage.csv",
        [
            "program",
            "depth",
            "active_passes",
            "certified_covered",
            "heuristic_covered",
            "unresolved_conflict",
            "validation_rejected",
            "unvalidated_covered",
            "failed_or_unknown",
            "not_executed_due_to_max_depth",
            "dropped_active_passes",
        ],
        [
            {"program": "ok", "depth": "0", "active_passes": "4", "certified_covered": "1", "heuristic_covered": "0", "unresolved_conflict": "0", "validation_rejected": "0", "unvalidated_covered": "2", "failed_or_unknown": "1", "not_executed_due_to_max_depth": "0", "dropped_active_passes": dropped},
            {"program": "ok", "depth": "1", "active_passes": terminal_not_executed, "certified_covered": "0", "heuristic_covered": "0", "unresolved_conflict": "0", "validation_rejected": "0", "unvalidated_covered": "0", "failed_or_unknown": "0", "not_executed_due_to_max_depth": terminal_not_executed, "dropped_active_passes": "0"},
        ],
    )
    _write_csv(
        run_dir / "mainline_aggregate_overlap.csv",
        [
            "program",
            "depth",
            "total_pairs",
            "disjoint_write",
            "same_function_overlap",
            "same_block_overlap",
            "unknown_overlap",
            "overlap_and_commute",
            "overlap_and_order_sensitive",
        ],
        [
            {"program": "ok", "depth": "0", "total_pairs": "6", "disjoint_write": "1", "same_function_overlap": "2", "same_block_overlap": "3", "unknown_overlap": "0", "overlap_and_commute": "4", "overlap_and_order_sensitive": "1"},
        ],
    )
    _write_csv(
        run_dir / "mainline_missing_outputs.csv",
        ["program", "expected_file", "status"],
        [{"program": "ok", "expected_file": "aggregate_by_depth.csv", "status": "present"}],
    )
    _write_csv(
        run_dir / "ok" / "states.csv",
        ["state_id", "depth"],
        [
            {"state_id": "S0000", "depth": "0"},
            {"state_id": "S0001", "depth": "1"},
        ],
    )
    _write_csv(
        run_dir / "ok" / "states" / "S0000" / "batch_summary.csv",
        ["program", "state_id", "naive_orderings_estimate"],
        [{"program": "ok", "state_id": "S0000", "naive_orderings_estimate": "1000"}],
    )
    _write_csv(
        run_dir / "ok" / "states" / "S0001" / "batch_summary.csv",
        ["program", "state_id", "naive_orderings_estimate"],
        [{"program": "ok", "state_id": "S0001", "naive_orderings_estimate": "1000000"}],
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
