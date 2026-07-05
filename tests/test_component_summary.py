import csv
import tempfile
import unittest
from pathlib import Path

from phasebatch.component_summary import summarize_components


class ComponentSummaryTests(unittest.TestCase):
    def test_component_reports_reconstruct_conflict_graph_and_dot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_mock_run(run_dir, "mock")

            result = summarize_components(run_dir=run_dir)

            components = _read_csv(Path(result["component_summary_csv"]))
            edges = _read_csv(Path(result["component_edges_csv"]))
            programs = _read_csv(Path(result["component_program_summary_csv"]))
            markdown = Path(result["component_summary_md"]).read_text(encoding="utf-8")
            dot_path = run_dir / "components" / "mock_S0000_interaction.dot"
            dot = dot_path.read_text(encoding="utf-8")

        by_component = {row["component_passes"]: row for row in components}
        self.assertEqual({row["program"] for row in edges}, {"mock"})
        self.assertEqual(len(edges), 6)
        self.assertEqual(len(components), 3)
        self.assertEqual(by_component["A;B;C"]["component_size"], "3")
        self.assertEqual(by_component["A;B;C"]["conflict_edges"], "2")
        self.assertEqual(by_component["A;B;C"]["commute_pairs_inside_component"], "1")
        self.assertEqual(by_component["A;B;C"]["order_sensitive_edges"], "1")
        self.assertEqual(by_component["A;B;C"]["unknown_edges"], "1")
        self.assertEqual(by_component["A;B;C"]["is_singleton"], "false")
        self.assertEqual(by_component["A;B;C"]["batch_candidates_contributed"], "2")
        self.assertEqual(by_component["D"]["is_singleton"], "true")
        self.assertEqual(programs[0]["max_component_size"], "3")
        self.assertEqual(programs[0]["singleton_components"], "2")
        self.assertIn("# Component / Interaction Graph Summary", markdown)
        self.assertIn("Conflict components are built from non-commuting or unknown pass pairs.", markdown)
        self.assertIn("This graph is state-local.", markdown)
        self.assertIn("| mock | 4 | 1 | 1 |", markdown)
        self.assertNotIn("| optimize |", markdown)
        self.assertIn("A -- B [color=red", dot)
        self.assertIn("B -- C [color=gray", dot)
        self.assertNotIn("A -- C", dot)

    def test_multi_run_aggregation_writes_all_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_a = root / "run_a"
            run_b = root / "run_b"
            out_dir = root / "summary"
            _make_mock_run(run_a, "alpha")
            _make_mock_run(run_b, "beta")

            result = summarize_components(run_dirs=[run_a, run_b], out_dir=out_dir)

            all_components = _read_csv(Path(result["component_summary_all_csv"]))
            all_edges = _read_csv(Path(result["component_edges_all_csv"]))
            programs = _read_csv(out_dir / "component_program_summary.csv")
            markdown_exists = (out_dir / "component_summary.md").exists()

        self.assertEqual({row["program"] for row in programs}, {"alpha", "beta"})
        self.assertEqual(len(all_components), 6)
        self.assertEqual(len(all_edges), 12)
        self.assertTrue(markdown_exists)


def _make_mock_run(run_dir: Path, program: str) -> None:
    states_dir = run_dir / "states"
    s0 = states_dir / "S0000"
    s1 = states_dir / "S0001"
    for path in [s0, s1]:
        path.mkdir(parents=True, exist_ok=True)

    _write_csv(
        run_dir / "states.csv",
        ["program", "state_id", "depth", "state_hash", "state_dir"],
        [
            {"program": program, "state_id": "S0000", "depth": "0", "state_hash": "h0", "state_dir": str(s0)},
            {"program": program, "state_id": "S0001", "depth": "1", "state_hash": "h1", "state_dir": str(s1)},
        ],
    )
    _write_csv(
        run_dir / "chosen_path.csv",
        ["parent_state_id", "child_state_id"],
        [{"parent_state_id": "S0000", "child_state_id": "S0001"}],
    )

    _write_csv(
        s0 / "pass_profile.csv",
        ["program", "state_id", "depth", "state_hash", "pass", "success", "active"],
        [
            {"program": program, "state_id": "S0000", "depth": "0", "state_hash": "h0", "pass": "A", "success": "true", "active": "true"},
            {"program": program, "state_id": "S0000", "depth": "0", "state_hash": "h0", "pass": "B", "success": "true", "active": "true"},
            {"program": program, "state_id": "S0000", "depth": "0", "state_hash": "h0", "pass": "C", "success": "true", "active": "true"},
            {"program": program, "state_id": "S0000", "depth": "0", "state_hash": "h0", "pass": "D", "success": "true", "active": "true"},
        ],
    )
    _write_csv(
        s0 / "pair_relation.csv",
        ["program", "state_id", "depth", "pass_a", "pass_b", "final_relation", "same_hash", "validation_status"],
        [
            {"program": "optimize", "state_id": "S0000", "depth": "0", "pass_a": "A", "pass_b": "B", "final_relation": "final_order_sensitive", "same_hash": "false", "validation_status": ""},
            {"program": "optimize", "state_id": "S0000", "depth": "0", "pass_a": "B", "pass_b": "C", "final_relation": "final_unknown", "same_hash": "", "validation_status": ""},
            {"program": "optimize", "state_id": "S0000", "depth": "0", "pass_a": "A", "pass_b": "C", "final_relation": "final_commute", "same_hash": "true", "validation_status": ""},
            {"program": "optimize", "state_id": "S0000", "depth": "0", "pass_a": "A", "pass_b": "D", "final_relation": "final_commute", "same_hash": "true", "validation_status": ""},
            {"program": "optimize", "state_id": "S0000", "depth": "0", "pass_a": "B", "pass_b": "D", "final_relation": "final_commute", "same_hash": "true", "validation_status": ""},
            {"program": "optimize", "state_id": "S0000", "depth": "0", "pass_a": "C", "pass_b": "D", "final_relation": "final_commute", "same_hash": "true", "validation_status": ""},
        ],
    )
    _write_csv(
        s0 / "batch_candidates.csv",
        ["batch_id", "batch_passes", "component_choices"],
        [
            {"batch_id": "B0000", "batch_passes": "A;C;D", "component_choices": "C0000:A;C|C0001:D"},
            {"batch_id": "B0001", "batch_passes": "B;D", "component_choices": "C0000:B|C0001:D"},
        ],
    )
    _write_csv(
        s0 / "batch_correctness.csv",
        ["batch_id", "validation_status"],
        [
            {"batch_id": "B0000", "validation_status": "all_permutations_same"},
            {"batch_id": "B0001", "validation_status": "all_permutations_same"},
        ],
    )

    _write_csv(
        s1 / "pass_profile.csv",
        ["program", "state_id", "depth", "state_hash", "pass", "success", "active"],
        [{"program": program, "state_id": "S0001", "depth": "1", "state_hash": "h1", "pass": "E", "success": "true", "active": "true"}],
    )
    _write_csv(s1 / "pair_relation.csv", ["pass_a", "pass_b", "final_relation"], [])


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
