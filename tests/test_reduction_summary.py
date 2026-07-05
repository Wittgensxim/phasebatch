import csv
import math
import tempfile
import unittest
from pathlib import Path

from phasebatch.reduction_summary import summarize_reduction


class ReductionSummaryTests(unittest.TestCase):
    def test_reduction_report_is_generated_from_mock_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_mock_run(run_dir)

            result = summarize_reduction(run_dir)

            by_state = _read_csv(Path(result["reduction_by_state_csv"]))
            summary = _read_csv(Path(result["reduction_summary_csv"]))
            markdown = Path(result["reduction_summary_md"]).read_text(encoding="utf-8")

        rows = {row["state_id"]: row for row in by_state}
        self.assertEqual(len(by_state), 3)
        self.assertEqual(rows["S0000"]["active_passes"], "4")
        self.assertAlmostEqual(float(rows["S0000"]["naive_orderings_log10"]), math.log10(24), places=5)
        self.assertEqual(rows["S0000"]["batch_candidates"], "3")
        self.assertEqual(rows["S0000"]["certified_batches"], "2")
        self.assertEqual(rows["S0000"]["executable_batches"], "2")
        self.assertEqual(rows["S0000"]["sampled_batches"], "1")
        self.assertEqual(rows["S0000"]["skipped_batches"], "1")
        self.assertEqual(rows["S0000"]["dropped_active_passes"], "1")
        self.assertAlmostEqual(float(rows["S0000"]["local_reduction_ratio"]), 12.0, places=5)
        self.assertEqual(rows["S0000"]["selected_on_final_path"], "true")

        self.assertEqual(rows["S0001"]["executable_batches"], "0")
        self.assertEqual(rows["S0001"]["no_executable_batches"], "true")
        self.assertEqual(rows["S0001"]["local_reduction_ratio"], "6")
        self.assertNotIn("inf", rows["S0001"]["local_reduction_log10"].lower())
        self.assertEqual(rows["S0001"]["selected_on_final_path"], "true")

        self.assertEqual(rows["S0002"]["active_passes"], "1")
        self.assertEqual(rows["S0002"]["local_reduction_ratio"], "1")
        self.assertEqual(rows["S0002"]["selected_on_final_path"], "false")

        self.assertEqual(summary[0]["total_states"], "3")
        self.assertEqual(summary[0]["total_executed_transitions"], "2")
        self.assertEqual(summary[0]["selected_path_steps"], "2")
        self.assertEqual(summary[0]["selected_path_pass_invocations"], "5")
        self.assertEqual(summary[0]["final_pipeline_length"], "5")
        self.assertIn("# Reduction Evidence Summary", markdown)
        self.assertIn("Objective is not used as commutation proof.", markdown)


def _make_mock_run(run_dir: Path) -> None:
    states_dir = run_dir / "states"
    s0 = states_dir / "S0000"
    s1 = states_dir / "S0001"
    s2 = states_dir / "S0002"
    for path in [s0, s1, s2]:
        path.mkdir(parents=True, exist_ok=True)

    _write_csv(
        run_dir / "states.csv",
        ["program", "state_id", "depth", "state_hash", "state_dir"],
        [
            {"program": "mock", "state_id": "S0000", "depth": "0", "state_hash": "h0", "state_dir": str(s0)},
            {"program": "mock", "state_id": "S0001", "depth": "1", "state_hash": "h1", "state_dir": str(s1)},
            {"program": "mock", "state_id": "S0002", "depth": "1", "state_hash": "h2", "state_dir": str(s2)},
        ],
    )
    _write_csv(
        run_dir / "batch_state_transitions.csv",
        ["parent_state_id", "child_state_id"],
        [
            {"parent_state_id": "S0000", "child_state_id": "S0001"},
            {"parent_state_id": "S0000", "child_state_id": "S0002"},
        ],
    )
    _write_csv(
        run_dir / "chosen_path.csv",
        ["parent_state_id", "child_state_id", "batch_passes"],
        [
            {"parent_state_id": "S0000", "child_state_id": "S0001", "batch_passes": "a;b"},
            {"parent_state_id": "S0001", "child_state_id": "S0003", "batch_passes": "c;d;e"},
        ],
    )
    (run_dir / "optimized_pipeline.txt").write_text("a,b,c,d,e\n", encoding="utf-8")

    _write_csv(
        s0 / "per_state_summary.csv",
        [
            "program",
            "state_id",
            "depth",
            "state_hash",
            "active_passes",
            "pairs_tested",
            "dynamic_commute",
            "order_sensitive",
            "unknown",
        ],
        [
            {
                "program": "mock",
                "state_id": "S0000",
                "depth": "0",
                "state_hash": "h0",
                "active_passes": "4",
                "pairs_tested": "6",
                "dynamic_commute": "4",
                "order_sensitive": "2",
                "unknown": "0",
            }
        ],
    )
    _write_csv(
        s0 / "batch_candidates.csv",
        ["batch_id", "batch_passes"],
        [
            {"batch_id": "B0000", "batch_passes": "a;b"},
            {"batch_id": "B0001", "batch_passes": "c;d"},
            {"batch_id": "B0002", "batch_passes": "e"},
        ],
    )
    _write_csv(
        s0 / "batch_correctness.csv",
        ["batch_id", "correctness_class", "can_execute"],
        [
            {"batch_id": "B0000", "correctness_class": "certified_batch", "can_execute": "true"},
            {"batch_id": "B0001", "correctness_class": "certified_batch", "can_execute": "true"},
            {"batch_id": "B0002", "correctness_class": "sampled_batch", "can_execute": "false"},
        ],
    )
    _write_csv(
        s0 / "coverage_summary.csv",
        ["dropped_active_passes"],
        [{"dropped_active_passes": "1"}],
    )

    _write_csv(
        s1 / "pass_profile.csv",
        ["pass", "success", "active"],
        [
            {"pass": "a", "success": "true", "active": "true"},
            {"pass": "b", "success": "true", "active": "true"},
            {"pass": "c", "success": "true", "active": "true"},
            {"pass": "d", "success": "true", "active": "false"},
        ],
    )
    _write_csv(
        s1 / "pair_relation.csv",
        ["final_relation"],
        [
            {"final_relation": "final_commute"},
            {"final_relation": "final_order_sensitive"},
            {"final_relation": "final_unknown"},
        ],
    )
    _write_csv(
        s1 / "batch_candidates.csv",
        ["batch_id", "batch_passes"],
        [],
    )
    _write_csv(
        s1 / "batch_correctness.csv",
        ["batch_id", "correctness_class", "can_execute"],
        [],
    )

    _write_csv(
        s2 / "per_state_summary.csv",
        ["active_passes", "pairs_tested", "dynamic_commute", "order_sensitive", "unknown"],
        [{"active_passes": "1", "pairs_tested": "0", "dynamic_commute": "0", "order_sensitive": "0", "unknown": "0"}],
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
