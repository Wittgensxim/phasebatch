import csv
import tempfile
import unittest
from pathlib import Path

from phasebatch.batcher import build_batch_family


class BatcherTests(unittest.TestCase):
    def test_fully_commuting_four_passes_produces_one_full_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_state(state_dir, ["A", "B", "C", "D"], _all_pairs(["A", "B", "C", "D"], "final_commute"))

            result = build_batch_family(state_dir)
            candidates = _read_csv(state_dir / "batch_candidates.csv")
            summary = _read_csv(state_dir / "batch_summary.csv")[0]

        self.assertEqual(result["batch_candidates"], 1)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["batch_passes"], "A;B;C;D")
        self.assertEqual(candidates[0]["is_exact"], "true")
        self.assertEqual(summary["active_pairs"], "6")
        self.assertEqual(summary["commute_pairs"], "6")
        self.assertEqual(summary["conflict_pairs"], "0")
        self.assertEqual(summary["naive_orderings_estimate"], "24")

    def test_fully_conflicting_three_passes_with_two_independent_passes_produces_three_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            passes = ["A", "B", "C", "D", "E"]
            relations = _all_pairs(passes, "final_commute")
            relations.update({("A", "B"): "final_order_sensitive", ("A", "C"): "final_order_sensitive", ("B", "C"): "final_order_sensitive"})
            _write_state(state_dir, passes, relations)

            result = build_batch_family(state_dir)
            batches = {row["batch_passes"] for row in _read_csv(state_dir / "batch_candidates.csv")}
            summary = _read_csv(state_dir / "batch_summary.csv")[0]

        self.assertEqual(result["batch_candidates"], 3)
        self.assertEqual(batches, {"A;D;E", "B;D;E", "C;D;E"})
        self.assertEqual(summary["conflict_pairs"], "3")
        self.assertEqual(summary["batch_reduction_estimate"], "40.00")

    def test_path_conflict_component_uses_maximal_independent_sets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            relations = {("A", "B"): "final_order_sensitive", ("A", "C"): "final_commute", ("B", "C"): "final_order_sensitive"}
            _write_state(state_dir, ["A", "B", "C"], relations)

            build_result = build_batch_family(state_dir)
            batches = {row["batch_passes"] for row in _read_csv(state_dir / "batch_candidates.csv")}

        self.assertEqual(build_result["batch_candidates"], 2)
        self.assertEqual(batches, {"A;C", "B"})

    def test_unknown_relation_is_conservative_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_state(state_dir, ["A", "B"], {("A", "B"): "final_unknown"})

            result = build_batch_family(state_dir)
            batches = {row["batch_passes"] for row in _read_csv(state_dir / "batch_candidates.csv")}
            components = _read_csv(state_dir / "batch_components.csv")
            summary_text = (state_dir / "batch_summary.md").read_text(encoding="utf-8")

        self.assertEqual(result["batch_candidates"], 2)
        self.assertEqual(batches, {"A", "B"})
        self.assertEqual(components[0]["conflict_edges"], "A--B")
        self.assertIn("Batch Summary", summary_text)

    def test_pair_relation_keys_are_unordered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_state(state_dir, ["B", "A"], {("A", "B"): "final_commute"})

            result = build_batch_family(state_dir)
            candidates = _read_csv(state_dir / "batch_candidates.csv")

        self.assertEqual(result["batch_candidates"], 1)
        self.assertEqual(candidates[0]["batch_passes"], "B;A")


def _write_state(state_dir: Path, passes: list[str], relations: dict[tuple[str, str], str]) -> None:
    _write_csv(
        state_dir / "pass_profile.csv",
        ["program", "state_id", "state_hash", "pass", "success", "active"],
        [
            {"program": "testprog", "state_id": "S0000", "state_hash": "hash0", "pass": pass_name, "success": "true", "active": "true"}
            for pass_name in passes
        ],
    )
    _write_csv(
        state_dir / "pair_relation.csv",
        ["program", "state_id", "state_hash", "pass_a", "pass_b", "final_relation"],
        [
            {
                "program": "testprog",
                "state_id": "S0000",
                "state_hash": "hash0",
                "pass_a": pass_a,
                "pass_b": pass_b,
                "final_relation": relation,
            }
            for (pass_a, pass_b), relation in relations.items()
        ],
    )


def _all_pairs(passes: list[str], relation: str) -> dict[tuple[str, str], str]:
    return {(passes[i], passes[j]): relation for i in range(len(passes)) for j in range(i + 1, len(passes))}


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
