import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.dag_visualizer import _run_dot, visualize_dag


class DagVisualizerTests(unittest.TestCase):
    def test_dot_generation_marks_duplicate_and_selected_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            out_dir = root / "viz"
            _make_mock_run(run_dir)

            result = visualize_dag(
                run_dir,
                out_dir,
                view="all",
                formats=["dot"],
                max_full_nodes=200,
                include_selected_path=True,
                include_depth_overview=True,
            )

            full_dot = (out_dir / "state_dag_full.dot").read_text(encoding="utf-8")
            selected_dot = (out_dir / "state_dag_selected.dot").read_text(encoding="utf-8")
            paths = _read_csv(out_dir / "dag_paths.csv")

        self.assertEqual(result["unique_states"], 4)
        for state_id in ["S0000", "S0001", "S0002", "S0003"]:
            self.assertIn(state_id, full_dot)
        self.assertIn('S0002 -> S0003', full_dot)
        self.assertIn('style="dashed"', full_dot)
        self.assertIn('duplicate -> S0003', full_dot)
        self.assertIn('color="green"', full_dot)
        self.assertIn('penwidth=3', full_dot)
        self.assertIn("S0000", selected_dot)
        self.assertIn("S0001", selected_dot)
        self.assertIn("S0003", selected_dot)
        self.assertEqual([row["batch_id"] for row in paths], ["B0000", "B0002"])

    def test_depth_rank_grouping_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            out_dir = Path(tmp) / "viz"
            _make_mock_run(run_dir)

            visualize_dag(run_dir, out_dir, view="all", formats=["dot"], max_full_nodes=200)

            dot = (out_dir / "state_dag_full.dot").read_text(encoding="utf-8")

        self.assertIn("{ rank=same; S0000; }", dot)
        self.assertIn("{ rank=same; S0001; S0002; }", dot)
        self.assertIn("{ rank=same; S0003; }", dot)

    def test_selected_only_view_limits_graph_to_selected_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            out_dir = Path(tmp) / "viz"
            _make_mock_run(run_dir)

            visualize_dag(run_dir, out_dir, view="selected-only", formats=["dot"], max_full_nodes=200)

            dot = (out_dir / "state_dag_selected.dot").read_text(encoding="utf-8")

        self.assertIn("S0000", dot)
        self.assertIn("S0001", dot)
        self.assertIn("S0003", dot)
        self.assertNotIn("S0002", dot)
        self.assertNotIn("B0001", dot)

    def test_depth_overview_and_metrics_are_generated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            out_dir = Path(tmp) / "viz"
            _make_mock_run(run_dir)

            visualize_dag(run_dir, out_dir, view="depth-overview", formats=["dot"], max_full_nodes=200)

            overview = (out_dir / "depth_overview.dot").read_text(encoding="utf-8")
            depth_rows = _read_csv(out_dir / "dag_depth_metrics.csv")

        self.assertIn("D0 -> D1", overview)
        self.assertIn("D1 -> D2", overview)
        by_depth = {row["depth"]: row for row in depth_rows}
        self.assertEqual(by_depth["1"]["states"], "2")
        self.assertEqual(by_depth["1"]["outgoing_transitions"], "2")
        self.assertEqual(by_depth["2"]["incoming_transitions"], "2")

    def test_graphviz_missing_writes_dot_and_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            out_dir = Path(tmp) / "viz"
            _make_mock_run(run_dir)

            with mock.patch("phasebatch.dag_visualizer.shutil.which", return_value=None):
                visualize_dag(run_dir, out_dir, view="all", formats=["dot", "svg"], max_full_nodes=200)

            summary = (out_dir / "dag_summary.md").read_text(encoding="utf-8")
            full_dot_exists = (out_dir / "state_dag_full.dot").exists()
            full_svg_exists = (out_dir / "state_dag_full.svg").exists()

        self.assertTrue(full_dot_exists)
        self.assertFalse(full_svg_exists)
        self.assertIn("graphviz unavailable", summary)

    def test_large_graph_skips_full_render_but_keeps_selected_and_depth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            out_dir = Path(tmp) / "viz"
            _make_mock_run(run_dir)

            with mock.patch("phasebatch.dag_visualizer.shutil.which", return_value="dot"), mock.patch("phasebatch.dag_visualizer._run_dot") as fake_dot:
                visualize_dag(
                    run_dir,
                    out_dir,
                    view="all",
                    formats=["dot", "svg"],
                    max_full_nodes=2,
                    include_selected_path=True,
                    include_depth_overview=True,
                )

            rendered_inputs = [call.args[0].name for call in fake_dot.call_args_list]
            summary = (out_dir / "dag_summary.md").read_text(encoding="utf-8")
            full_dot_exists = (out_dir / "state_dag_full.dot").exists()

        self.assertTrue(full_dot_exists)
        self.assertNotIn("state_dag_full.dot", rendered_inputs)
        self.assertIn("state_dag_selected.dot", rendered_inputs)
        self.assertIn("depth_overview.dot", rendered_inputs)
        self.assertIn("graph too large for full render", summary)

    def test_metrics_csv_has_run_level_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            out_dir = Path(tmp) / "viz"
            _make_mock_run(run_dir)

            visualize_dag(run_dir, out_dir, view="all", formats=["dot"], max_full_nodes=200)

            metrics = _read_csv(out_dir / "dag_metrics.csv")[0]

        self.assertEqual(metrics["unique_states"], "4")
        self.assertEqual(metrics["transitions"], "4")
        self.assertEqual(metrics["duplicate_transitions"], "1")
        self.assertEqual(metrics["merge_rate"], "0.25")
        self.assertEqual(metrics["max_depth"], "2")

    def test_run_dot_uses_resolved_batch_entrypoint_on_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_dot = root / "dot.cmd"
            fake_dot.write_text("@echo off\r\ncopy /Y %2 %4 >NUL\r\n", encoding="utf-8")
            dot_file = root / "input.dot"
            output_file = root / "output.svg"
            dot_file.write_text("digraph G { A -> B }\n", encoding="utf-8")

            with mock.patch("phasebatch.dag_visualizer.shutil.which", return_value=str(fake_dot)):
                warning = _run_dot(dot_file, "svg", output_file)
            output_exists = output_file.exists()

        self.assertEqual(warning, "")
        self.assertTrue(output_exists)

    def test_run_dot_skips_windows_batch_entrypoint_on_non_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_dot = root / "dot.cmd"
            fake_dot.write_text("@echo off\r\n", encoding="utf-8")
            dot_file = root / "input.dot"
            output_file = root / "output.svg"
            dot_file.write_text("digraph G { A -> B }\n", encoding="utf-8")

            with mock.patch("os.name", "posix"), \
                mock.patch("phasebatch.dag_visualizer.shutil.which", return_value=str(fake_dot)), \
                mock.patch("phasebatch.dag_visualizer.subprocess.run") as fake_run:
                warning = _run_dot(dot_file, "svg", output_file)

        self.assertIn("graphviz render skipped", warning)
        self.assertIn(".cmd/.bat dot entrypoint cannot run on non-Windows", warning)
        fake_run.assert_not_called()


def _make_mock_run(run_dir: Path) -> None:
    for state_id in ["S0000", "S0001", "S0002", "S0003"]:
        (run_dir / "states" / state_id).mkdir(parents=True, exist_ok=True)

    _write_csv(
        run_dir / "states.csv",
        ["program", "state_id", "state_hash", "depth", "state_dir", "active_passes", "pairs_tested", "dynamic_commute", "order_sensitive", "unknown"],
        [
            {"program": "mock", "state_id": "S0000", "state_hash": "h0", "depth": "0", "state_dir": str(run_dir / "states" / "S0000"), "active_passes": "3", "pairs_tested": "3", "dynamic_commute": "2", "order_sensitive": "1", "unknown": "0"},
            {"program": "mock", "state_id": "S0001", "state_hash": "h1", "depth": "1", "state_dir": str(run_dir / "states" / "S0001"), "active_passes": "2", "pairs_tested": "1", "dynamic_commute": "1", "order_sensitive": "0", "unknown": "0"},
            {"program": "mock", "state_id": "S0002", "state_hash": "h2", "depth": "1", "state_dir": str(run_dir / "states" / "S0002"), "active_passes": "2", "pairs_tested": "1", "dynamic_commute": "0", "order_sensitive": "1", "unknown": "0"},
            {"program": "mock", "state_id": "S0003", "state_hash": "h3", "depth": "2", "state_dir": str(run_dir / "states" / "S0003"), "active_passes": "0", "pairs_tested": "0", "dynamic_commute": "0", "order_sensitive": "0", "unknown": "0"},
        ],
    )
    _write_csv(
        run_dir / "state_dag.csv",
        [
            "program",
            "source_state_id",
            "target_state_id",
            "source_hash",
            "target_hash",
            "transition_kind",
            "batch_id",
            "batch_passes",
            "canonical_order",
            "validation_status",
            "correctness_class",
            "is_duplicate",
            "duplicate_of",
        ],
        [
            {"program": "mock", "source_state_id": "S0000", "target_state_id": "S0001", "source_hash": "h0", "target_hash": "h1", "transition_kind": "batch", "batch_id": "B0000", "batch_passes": "mem2reg;sroa", "canonical_order": "mem2reg;sroa", "validation_status": "all_permutations_same", "correctness_class": "certified_batch", "is_duplicate": "false", "duplicate_of": ""},
            {"program": "mock", "source_state_id": "S0000", "target_state_id": "S0002", "source_hash": "h0", "target_hash": "h2", "transition_kind": "batch", "batch_id": "B0001", "batch_passes": "gvn", "canonical_order": "gvn", "validation_status": "sampled_same", "correctness_class": "sampled_batch", "is_duplicate": "false", "duplicate_of": ""},
            {"program": "mock", "source_state_id": "S0001", "target_state_id": "S0003", "source_hash": "h1", "target_hash": "h3", "transition_kind": "batch", "batch_id": "B0002", "batch_passes": "dce", "canonical_order": "dce", "validation_status": "all_permutations_same", "correctness_class": "certified_batch", "is_duplicate": "false", "duplicate_of": ""},
            {"program": "mock", "source_state_id": "S0002", "target_state_id": "S0003", "source_hash": "h2", "target_hash": "h3", "transition_kind": "batch", "batch_id": "B0003", "batch_passes": "adce", "canonical_order": "adce", "validation_status": "all_permutations_same", "correctness_class": "certified_batch", "is_duplicate": "true", "duplicate_of": "S0003"},
        ],
    )
    _write_csv(
        run_dir / "chosen_path.csv",
        ["step", "parent_state_id", "batch_id", "child_state_id", "batch_passes", "canonical_order", "validation_status", "correctness_class", "ir_inst_before", "ir_inst_after", "ir_inst_delta"],
        [
            {"step": "0", "parent_state_id": "S0000", "batch_id": "B0000", "child_state_id": "S0001", "batch_passes": "mem2reg;sroa", "canonical_order": "mem2reg;sroa", "validation_status": "all_permutations_same", "correctness_class": "certified_batch", "ir_inst_before": "10", "ir_inst_after": "7", "ir_inst_delta": "-3"},
            {"step": "1", "parent_state_id": "S0001", "batch_id": "B0002", "child_state_id": "S0003", "batch_passes": "dce", "canonical_order": "dce", "validation_status": "all_permutations_same", "correctness_class": "certified_batch", "ir_inst_before": "7", "ir_inst_after": "5", "ir_inst_delta": "-2"},
        ],
    )
    _write_csv(
        run_dir / "leaf_states.csv",
        ["state_id", "is_leaf", "selected_as_final", "leaf_reason", "objective_value"],
        [
            {"state_id": "S0003", "is_leaf": "true", "selected_as_final": "true", "leaf_reason": "no_active_passes", "objective_value": "5"},
        ],
    )
    _write_csv(
        run_dir / "reduction_by_state.csv",
        ["state_id", "local_reduction_log10"],
        [
            {"state_id": "S0000", "local_reduction_log10": "0.78"},
            {"state_id": "S0001", "local_reduction_log10": "0.30"},
            {"state_id": "S0002", "local_reduction_log10": "0.30"},
            {"state_id": "S0003", "local_reduction_log10": "0.00"},
        ],
    )
    _write_csv(
        run_dir / "chosen_path_summary.csv",
        ["program", "selected_final_state", "root_ir_inst_count", "final_ir_inst_count", "path_steps", "total_pass_invocations"],
        [{"program": "mock", "selected_final_state": "S0003", "root_ir_inst_count": "10", "final_ir_inst_count": "5", "path_steps": "2", "total_pass_invocations": "3"}],
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
