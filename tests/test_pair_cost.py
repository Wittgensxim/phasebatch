import csv
import tempfile
import unittest
from pathlib import Path

from phasebatch.pair_cost import write_pair_cost_summary


class PairCostTests(unittest.TestCase):
    def test_write_pair_cost_summary_aggregates_state_pair_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            s0 = root / "states" / "S0000"
            s1 = root / "states" / "S0001"
            s0.mkdir(parents=True)
            s1.mkdir(parents=True)
            fields = [
                "program",
                "state_id",
                "depth",
                "cache_hit",
                "pair_test_opt_runs",
                "avoided_opt_runs",
                "reused_single_pass_outputs",
                "pair_test_time_ms",
                "pair_test_pass_invocations_baseline",
                "pair_test_pass_invocations_actual",
                "pair_test_pass_invocations_saved",
                "llvm_diff_time_ms",
                "comparator_time_ms",
            ]
            _write_csv(
                s0 / "pair_relation.csv",
                fields,
                [
                    {
                        "program": "branch",
                        "state_id": "S0000",
                        "depth": "0",
                        "cache_hit": "false",
                        "pair_test_opt_runs": "2",
                        "avoided_opt_runs": "0",
                        "reused_single_pass_outputs": "true",
                        "pair_test_time_ms": "4.5",
                        "pair_test_pass_invocations_baseline": "4",
                        "pair_test_pass_invocations_actual": "2",
                        "pair_test_pass_invocations_saved": "2",
                        "llvm_diff_time_ms": "",
                        "comparator_time_ms": "0.5",
                    }
                ],
            )
            _write_csv(
                s1 / "pair_relation.csv",
                fields,
                [
                    {
                        "program": "branch",
                        "state_id": "S0001",
                        "depth": "1",
                        "cache_hit": "true",
                        "pair_test_opt_runs": "0",
                        "avoided_opt_runs": "2",
                        "reused_single_pass_outputs": "false",
                        "pair_test_time_ms": "0.0",
                        "pair_test_pass_invocations_baseline": "4",
                        "pair_test_pass_invocations_actual": "0",
                        "pair_test_pass_invocations_saved": "4",
                        "llvm_diff_time_ms": "",
                        "comparator_time_ms": "0.0",
                    }
                ],
            )
            s2 = root / "states" / "S0002"
            s2.mkdir(parents=True)
            _write_csv(s2 / "pair_relation.csv", fields, [])

            result = write_pair_cost_summary(root)
            with (root / "pair_cost_summary.csv").open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            markdown = (root / "pair_cost_summary.md").read_text(encoding="utf-8")

        self.assertEqual(result["pair_cost_summary_csv"], str(root / "pair_cost_summary.csv"))
        self.assertEqual(row["program"], "branch")
        self.assertEqual(row["states"], "3")
        self.assertEqual(row["pair_rows"], "2")
        self.assertEqual(row["cache_hits"], "1")
        self.assertEqual(row["cache_misses"], "1")
        self.assertEqual(row["cache_hit_rate"], "0.5000")
        self.assertEqual(row["pair_test_opt_runs"], "2")
        self.assertEqual(row["avoided_opt_runs"], "2")
        self.assertEqual(row["reused_single_pass_pairs"], "1")
        self.assertEqual(row["pair_test_pass_invocations_baseline"], "8")
        self.assertEqual(row["pair_test_pass_invocations_actual"], "2")
        self.assertEqual(row["pair_test_pass_invocations_saved"], "6")
        self.assertEqual(row["pair_test_time_ms"], "4.500")
        self.assertEqual(row["llvm_diff_time_ms"], "")
        self.assertEqual(row["comparator_time_ms"], "0.500")
        self.assertIn("# Pair Detection Cost Summary", markdown)
        self.assertIn("- pass invocations saved: 6", markdown)
        self.assertIn("| 1 | 1 | 1 | 0 | 1.0000 | 0.000 |", markdown)
        self.assertIn("Pair-result memoization is keyed by canonical IR state and pass identity. It is not reused across different IR states.", markdown)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
