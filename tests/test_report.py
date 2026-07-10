import csv
import json
import tempfile
import unittest
from pathlib import Path

from phasebatch.report import write_aggregate_report, write_per_state_summary, write_summary


class ReportTests(unittest.TestCase):
    def test_write_summary_mentions_core_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "metadata.json").write_text(
                json.dumps({"input": "x.c", "tools": {"opt": {"version": "LLVM version 23"}}}),
                encoding="utf-8",
            )
            _write_csv(
                out / "pass_profile.csv",
                ["program", "pass", "active", "time_ms"],
                [
                    {"program": "x", "pass": "a", "active": "true", "time_ms": "1"},
                    {"program": "x", "pass": "b", "active": "false", "time_ms": "2"},
                ],
            )
            _write_csv(
                out / "pair_relation.csv",
                ["program", "pass_a", "pass_b", "dynamic_relation", "final_relation", "equality_tier", "can_hard_fold", "time_ms"],
                [
                    {
                        "program": "x",
                        "pass_a": "a",
                        "pass_b": "b",
                        "dynamic_relation": "dynamic_commute",
                        "final_relation": "final_commute",
                        "equality_tier": "structural_diff",
                        "can_hard_fold": "true",
                        "time_ms": "3",
                    }
                ],
            )
            _write_csv(
                out / "cluster_distribution.csv",
                ["program", "graph_type", "max_size", "median_size"],
                [{"program": "x", "graph_type": "noncommute_graph", "max_size": "0", "median_size": "0"}],
            )

            summary = write_summary(out)
            text = summary.read_text(encoding="utf-8")

        self.assertIn("# Summary", text)
        self.assertIn("active passes", text)
        self.assertIn("dynamic_commute", text)
        self.assertIn("# Equality Tier Summary", text)
        self.assertIn("| tier | count | hard_fold |", text)
        self.assertIn("| structural_diff | 1 | 1 |", text)

    def test_write_aggregate_report_includes_equality_tier_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program_dir = root / "program"
            program_dir.mkdir()
            _write_csv(
                program_dir / "pass_profile.csv",
                ["program", "pass"],
                [{"program": "x", "pass": "a"}],
            )
            _write_csv(
                program_dir / "pair_relation.csv",
                ["program", "pass_a", "pass_b", "equality_tier", "can_hard_fold"],
                [
                    {"program": "x", "pass_a": "a", "pass_b": "b", "equality_tier": "canonical_hash", "can_hard_fold": "true"},
                    {"program": "x", "pass_a": "a", "pass_b": "c", "equality_tier": "different", "can_hard_fold": "false"},
                ],
            )
            _write_csv(
                program_dir / "per_state_summary.csv",
                [
                    "program",
                    "valid_passes",
                    "active_passes",
                    "pairs_tested",
                    "dynamic_commute",
                    "order_sensitive",
                    "unknown",
                    "max_conflict_component",
                    "total_time_ms",
                ],
                [
                    {
                        "program": "x",
                        "valid_passes": "2",
                        "active_passes": "2",
                        "pairs_tested": "2",
                        "dynamic_commute": "1",
                        "order_sensitive": "1",
                        "unknown": "0",
                        "max_conflict_component": "1",
                        "total_time_ms": "4",
                    }
                ],
            )
            _write_csv(program_dir / "cluster_distribution.csv", ["program"], [])

            summary = write_aggregate_report(root, [program_dir])
            text = summary.read_text(encoding="utf-8")

        self.assertIn("# Equality Tier Summary", text)
        self.assertIn("| canonical_hash | 1 | 1 |", text)
        self.assertIn("| different | 1 | 0 |", text)

    def test_write_per_state_summary_includes_state_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            _write_csv(
                out / "pass_profile.csv",
                ["program", "state_id", "depth", "parent_state_id", "transition_pass", "pass", "success", "active"],
                [{"program": "x", "state_id": "S0001", "depth": "1", "parent_state_id": "S0000", "transition_pass": "mem2reg", "pass": "a", "success": "true", "active": "true"}],
            )
            _write_csv(
                out / "pair_relation.csv",
                ["program", "state_id", "depth", "parent_state_id", "transition_pass", "dynamic_relation", "static_relation"],
                [{"program": "x", "state_id": "S0001", "depth": "1", "parent_state_id": "S0000", "transition_pass": "mem2reg", "dynamic_relation": "dynamic_commute", "static_relation": "static_disjoint_function"}],
            )
            _write_csv(
                out / "cluster_distribution.csv",
                ["program", "graph_type", "max_size", "median_size"],
                [{"program": "x", "graph_type": "noncommute_graph", "max_size": "0", "median_size": "0"}],
            )

            summary_path = write_per_state_summary(
                out,
                "x",
                "hash",
                state_id="S0001",
                depth=1,
                parent_state_id="S0000",
                transition_pass="mem2reg",
                pass_set_size=1,
                valid_passes=1,
                invalid_passes=0,
                profile_time_ms=1.0,
                pair_time_ms=2.0,
                total_time_ms=3.0,
            )
            with summary_path.open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))

        self.assertEqual(row["state_id"], "S0001")
        self.assertEqual(row["depth"], "1")
        self.assertEqual(row["parent_state_id"], "S0000")
        self.assertEqual(row["transition_pass"], "mem2reg")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
