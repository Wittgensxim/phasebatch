import csv
import tempfile
import unittest
from pathlib import Path

from phasebatch.pair_scheduling import write_pair_scheduling_summary


class PairSchedulingSummaryTests(unittest.TestCase):
    def test_write_pair_scheduling_summary_aggregates_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            s0 = run_dir / "states" / "S0000"
            s1 = run_dir / "states" / "S0001"
            s0.mkdir(parents=True)
            s1.mkdir(parents=True)
            _write_csv(
                s0 / "per_state_summary.csv",
                ["program", "state_id", "depth", "active_passes"],
                [{"program": "run", "state_id": "S0000", "depth": "0", "active_passes": "3"}],
            )
            _write_csv(
                s0 / "pair_relation.csv",
                [
                    "program",
                    "state_id",
                    "pass_a",
                    "pass_b",
                    "dynamic_relation",
                    "final_relation",
                    "failure_kind",
                    "pair_testing_mode",
                    "skipped_by_budget",
                    "cache_hit",
                ],
                [
                    {"program": "run", "state_id": "S0000", "pass_a": "A", "pass_b": "B", "dynamic_relation": "dynamic_commute", "final_relation": "final_commute", "failure_kind": "", "pair_testing_mode": "lazy", "skipped_by_budget": "false", "cache_hit": "true"},
                    {"program": "run", "state_id": "S0000", "pass_a": "A", "pass_b": "C", "dynamic_relation": "not_tested", "final_relation": "final_unknown", "failure_kind": "lazy_budget", "pair_testing_mode": "lazy", "skipped_by_budget": "true", "cache_hit": "false"},
                    {"program": "run", "state_id": "S0000", "pass_a": "B", "pass_b": "C", "dynamic_relation": "dynamic_order_sensitive", "final_relation": "final_order_sensitive", "failure_kind": "", "pair_testing_mode": "lazy", "skipped_by_budget": "false", "cache_hit": "false"},
                ],
            )
            _write_csv(
                s1 / "per_state_summary.csv",
                ["program", "state_id", "depth", "active_passes"],
                [{"program": "run", "state_id": "S0001", "depth": "1", "active_passes": "1"}],
            )
            _write_csv(s1 / "pair_relation.csv", ["program", "state_id"], [])

            result = write_pair_scheduling_summary(run_dir)
            rows = _read_csv(run_dir / "pair_scheduling_summary.csv")
            markdown = (run_dir / "pair_scheduling_summary.md").read_text(encoding="utf-8")

        self.assertEqual(result["pair_scheduling_summary_csv"], str(run_dir / "pair_scheduling_summary.csv"))
        self.assertEqual(rows[0]["pair_testing_mode"], "lazy")
        self.assertEqual(rows[0]["total_pairs"], "3")
        self.assertEqual(rows[0]["tested_pairs"], "2")
        self.assertEqual(rows[0]["skipped_pairs"], "1")
        self.assertEqual(rows[0]["unknown_pairs"], "1")
        self.assertEqual(rows[0]["cache_hits"], "1")
        self.assertEqual(rows[0]["cache_misses"], "1")
        self.assertIn("Lazy pair testing can reduce cost but cannot create commute evidence. Untested pairs are treated as unknown/conflict.", markdown)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
