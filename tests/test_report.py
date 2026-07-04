import csv
import json
import tempfile
import unittest
from pathlib import Path

from phasebatch.report import write_summary


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
                ["program", "pass_a", "pass_b", "dynamic_relation", "final_relation", "time_ms"],
                [{"program": "x", "pass_a": "a", "pass_b": "b", "dynamic_relation": "dynamic_commute", "final_relation": "final_commute", "time_ms": "3"}],
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


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
