import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from phasebatch.final_summary import generate_final_summary


BOUNDARY_TEXT = (
    "The batch reduction layer only treats a batch as hard-foldable when its correctness evidence supports it, "
    "such as all_permutations_same validation. Objective values are used only for search ranking, path selection, "
    "and evaluation; they are not independence or commutation proof."
)


class FinalSummaryTests(unittest.TestCase):
    def test_final_summary_generated_with_all_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _write_complete_run(run_dir)

            path = generate_final_summary(run_dir)
            text = path.read_text(encoding="utf-8")

        self.assertEqual(path.name, "final_summary.md")
        for heading in [
            "# Final Optimization Summary",
            "## 1. Run Configuration",
            "## 2. Final Result",
            "## Final Pipeline Replay Verification",
            "## 3. Chosen Batch Path",
            "## 4. Why Each Batch Was Executable",
            "## 5. State Changes Along the Path",
            "## 6. Objective Signal Along the Path",
            "## 7. Baseline Comparison",
            "## 8. Reproducibility Artifacts",
            "## 9. Correctness Boundary",
        ]:
            self.assertIn(heading, text)
        self.assertIn("- selected mode: exact", text)
        self.assertIn("- exact_status: exact_complete", text)
        self.assertIn("- final IR instruction count: 3", text)

    def test_missing_baseline_results_warns_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _write_complete_run(run_dir, include_baselines=False)

            text = generate_final_summary(run_dir).read_text(encoding="utf-8")

        self.assertIn("## Warnings", text)
        self.assertIn("Baseline results are missing. Run compare-baselines or use --run-baselines.", text)
        self.assertIn("## 7. Baseline Comparison", text)

    def test_baseline_comparison_identifies_best_and_comparisons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _write_complete_run(run_dir)

            text = generate_final_summary(run_dir).read_text(encoding="utf-8")
            index = _read_csv(run_dir / "final_summary_index.csv")[0]

        self.assertIn("best method by final IR inst count: optimized_pipeline (3)", text)
        self.assertIn("optimized_pipeline beats greedy: true", text)
        self.assertIn("optimized_pipeline beats random best: true", text)
        self.assertIn("optimized_pipeline beats config_order_once: true", text)
        self.assertEqual(index["best_baseline_method"], "greedy_single_pass")
        self.assertEqual(index["best_baseline_inst_count"], "5")
        self.assertEqual(index["optimized_beats_greedy"], "true")
        self.assertEqual(index["optimized_beats_random"], "true")
        self.assertEqual(index["optimized_beats_config_order"], "true")

    def test_correctness_boundary_text_exists_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _write_complete_run(run_dir)

            text = generate_final_summary(run_dir).read_text(encoding="utf-8")

        self.assertIn(BOUNDARY_TEXT, text)

    def test_chosen_path_table_includes_objective_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _write_complete_run(run_dir)

            text = generate_final_summary(run_dir).read_text(encoding="utf-8")

        self.assertIn("| step | parent state | batch id | batch passes | validation | correctness | child state | active passes before -> after | IR inst before -> after | delta |", text)
        self.assertIn("| 0 | S0000 | B0000 | mem2reg;sroa | all_permutations_same | certified_batch | S0001 | 4 -> 2 | 10 -> 3 | -7 |", text)
        self.assertIn("Objective signals are used only for path selection and evaluation. They are not used as commutation proof.", text)
        self.assertIn("- correctness_class: certified_batch", text)
        self.assertIn("- reason: all tested permutations produced identical canonical IR", text)

    def test_final_summary_index_csv_generated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _write_complete_run(run_dir)

            generate_final_summary(run_dir)
            rows = _read_csv(run_dir / "final_summary_index.csv")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["program"], "run")
        self.assertEqual(rows[0]["final_state"], "S0001")
        self.assertEqual(rows[0]["root_ir_inst_count"], "10")
        self.assertEqual(rows[0]["final_ir_inst_count"], "3")
        self.assertEqual(rows[0]["reduction_pct"], "70.00")
        self.assertEqual(rows[0]["path_steps"], "1")
        self.assertEqual(rows[0]["pass_invocations"], "2")
        self.assertEqual(rows[0]["optimized_pipeline"], "mem2reg,sroa")
        self.assertEqual(rows[0]["replay_status"], "success")
        self.assertEqual(rows[0]["replay_hashes_match"], "true")

    def test_final_summary_includes_pipeline_replay_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _write_complete_run(run_dir)

            text = generate_final_summary(run_dir).read_text(encoding="utf-8")

        self.assertIn("## Final Pipeline Replay Verification", text)
        self.assertIn("- replay status: success", text)
        self.assertIn("- hashes match: true", text)
        self.assertIn("- replayed_final.ll path:", text)

    def test_final_summary_warns_on_pipeline_replay_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _write_complete_run(run_dir, replay_hashes_match="false", replay_status="mismatch")

            text = generate_final_summary(run_dir).read_text(encoding="utf-8")

        self.assertIn("**WARNING** final pipeline replay did not reproduce final.ll.", text)

    def test_summarize_final_command_regenerates_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _write_complete_run(run_dir)
            summary_path = run_dir / "final_summary.md"
            summary_path.write_text("stale\n", encoding="utf-8")

            result = subprocess.run(
                [sys.executable, "-m", "phasebatch", "summarize-final", "--run-dir", str(run_dir)],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )
            text = summary_path.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("final_summary.md", result.stdout)
        self.assertIn("# Final Optimization Summary", text)
        self.assertNotEqual(text, "stale\n")


def _write_complete_run(
    run_dir: Path,
    *,
    include_baselines: bool = True,
    replay_status: str = "success",
    replay_hashes_match: str = "true",
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        run_dir / "chosen_path.csv",
        [
            "step",
            "round",
            "parent_state_id",
            "parent_depth",
            "parent_state_hash",
            "batch_id",
            "batch_passes",
            "batch_size",
            "canonical_order",
            "validation_status",
            "correctness_class",
            "can_hard_fold",
            "can_execute",
            "child_state_id",
            "child_depth",
            "child_state_hash",
            "is_duplicate_transition",
            "duplicate_of",
            "parent_ir_path",
            "child_ir_path",
            "parent_active_passes",
            "child_active_passes",
            "parent_tested_pairs",
            "child_tested_pairs",
            "parent_commute_pairs",
            "child_commute_pairs",
            "parent_order_sensitive_pairs",
            "child_order_sensitive_pairs",
            "parent_unknown_pairs",
            "child_unknown_pairs",
            "ir_inst_before",
            "ir_inst_after",
            "ir_inst_delta",
            "ir_inst_reduction_pct",
            "selection_reason",
        ],
        [
            {
                "step": "0",
                "round": "0",
                "parent_state_id": "S0000",
                "parent_depth": "0",
                "parent_state_hash": "root-hash",
                "batch_id": "B0000",
                "batch_passes": "mem2reg;sroa",
                "batch_size": "2",
                "canonical_order": "mem2reg;sroa",
                "validation_status": "all_permutations_same",
                "correctness_class": "certified_batch",
                "can_hard_fold": "true",
                "can_execute": "true",
                "child_state_id": "S0001",
                "child_depth": "1",
                "child_state_hash": "child-hash",
                "is_duplicate_transition": "false",
                "duplicate_of": "",
                "parent_ir_path": str(run_dir / "states" / "S0000" / "input.ll"),
                "child_ir_path": str(run_dir / "states" / "S0001" / "input.ll"),
                "parent_active_passes": "4",
                "child_active_passes": "2",
                "parent_tested_pairs": "6",
                "child_tested_pairs": "1",
                "parent_commute_pairs": "5",
                "child_commute_pairs": "1",
                "parent_order_sensitive_pairs": "1",
                "child_order_sensitive_pairs": "0",
                "parent_unknown_pairs": "0",
                "child_unknown_pairs": "0",
                "ir_inst_before": "10",
                "ir_inst_after": "3",
                "ir_inst_delta": "-7",
                "ir_inst_reduction_pct": "70.00",
                "selection_reason": "selected_final_path",
            }
        ],
    )
    _write_csv(
        run_dir / "chosen_path_summary.csv",
        [
            "program",
            "selected_final_state",
            "path_steps",
            "total_pass_invocations",
            "unique_pass_types",
            "root_ir_inst_count",
            "final_ir_inst_count",
            "total_ir_inst_delta",
            "total_ir_inst_reduction_pct",
            "all_batches_certified",
            "any_sampled_batch",
            "any_rejected_batch",
            "any_unvalidated_batch",
            "replay_verified",
        ],
        [
            {
                "program": "run",
                "selected_final_state": "S0001",
                "path_steps": "1",
                "total_pass_invocations": "2",
                "unique_pass_types": "2",
                "root_ir_inst_count": "10",
                "final_ir_inst_count": "3",
                "total_ir_inst_delta": "-7",
                "total_ir_inst_reduction_pct": "70.00",
                "all_batches_certified": "true",
                "any_sampled_batch": "false",
                "any_rejected_batch": "false",
                "any_unvalidated_batch": "false",
                "replay_verified": "false",
            }
        ],
    )
    _write_csv(
        run_dir / "states.csv",
        ["program", "state_id", "depth", "state_hash", "ir_path", "state_dir", "active_passes", "pairs_tested", "dynamic_commute", "order_sensitive", "unknown"],
        [
            {"program": "run", "state_id": "S0000", "depth": "0", "state_hash": "root-hash", "ir_path": "states/S0000/input.ll", "state_dir": "states/S0000", "active_passes": "4", "pairs_tested": "6", "dynamic_commute": "5", "order_sensitive": "1", "unknown": "0"},
            {"program": "run", "state_id": "S0001", "depth": "1", "state_hash": "child-hash", "ir_path": "states/S0001/input.ll", "state_dir": "states/S0001", "active_passes": "2", "pairs_tested": "1", "dynamic_commute": "1", "order_sensitive": "0", "unknown": "0"},
        ],
    )
    _write_csv(
        run_dir / "enable_suppress.csv",
        ["program", "parent_state_id", "child_state_id", "transition_pass", "affected_pass", "relation"],
        [
            {"program": "run", "parent_state_id": "S0000", "child_state_id": "S0001", "transition_pass": "mem2reg;sroa", "affected_pass": "instcombine", "relation": "enable"},
            {"program": "run", "parent_state_id": "S0000", "child_state_id": "S0001", "transition_pass": "mem2reg;sroa", "affected_pass": "gvn", "relation": "effect_changed"},
        ],
    )
    _write_csv(
        run_dir / "relation_flip.csv",
        ["program", "parent_state_id", "child_state_id", "transition_pass", "pass_a", "pass_b", "parent_relation", "child_relation", "flip_kind"],
        [
            {"program": "run", "parent_state_id": "S0000", "child_state_id": "S0001", "transition_pass": "mem2reg;sroa", "pass_a": "a", "pass_b": "b", "parent_relation": "final_commute", "child_relation": "final_order_sensitive", "flip_kind": "commute_to_sensitive"},
            {"program": "run", "parent_state_id": "S0000", "child_state_id": "S0001", "transition_pass": "mem2reg;sroa", "pass_a": "c", "pass_b": "d", "parent_relation": "final_commute", "child_relation": "", "flip_kind": "active_pair_to_missing"},
        ],
    )
    if include_baselines:
        _write_csv(
            run_dir / "baseline_results.csv",
            [
                "method",
                "status",
                "final_ir_path",
                "final_ir_hash",
                "final_ir_inst_count",
                "root_ir_inst_count",
                "ir_inst_delta",
                "ir_inst_reduction_pct",
                "pass_sequence",
                "pass_invocations",
                "states_analyzed",
                "opt_runs",
                "time_ms",
                "error_message",
            ],
            [
                {"method": "root", "status": "success", "final_ir_path": "root.ll", "final_ir_hash": "h0", "final_ir_inst_count": "10", "root_ir_inst_count": "10", "ir_inst_delta": "0", "ir_inst_reduction_pct": "0.00", "pass_sequence": "", "pass_invocations": "0", "states_analyzed": "0", "opt_runs": "0", "time_ms": "0.00", "error_message": ""},
                {"method": "optimized_pipeline", "status": "success", "final_ir_path": "opt.ll", "final_ir_hash": "h1", "final_ir_inst_count": "3", "root_ir_inst_count": "10", "ir_inst_delta": "-7", "ir_inst_reduction_pct": "70.00", "pass_sequence": "mem2reg;sroa", "pass_invocations": "2", "states_analyzed": "0", "opt_runs": "1", "time_ms": "1.00", "error_message": ""},
                {"method": "config_order_once", "status": "success", "final_ir_path": "config.ll", "final_ir_hash": "h2", "final_ir_inst_count": "6", "root_ir_inst_count": "10", "ir_inst_delta": "-4", "ir_inst_reduction_pct": "40.00", "pass_sequence": "mem2reg;sroa;gvn", "pass_invocations": "3", "states_analyzed": "0", "opt_runs": "1", "time_ms": "2.00", "error_message": ""},
                {"method": "greedy_single_pass", "status": "success", "final_ir_path": "greedy.ll", "final_ir_hash": "h3", "final_ir_inst_count": "5", "root_ir_inst_count": "10", "ir_inst_delta": "-5", "ir_inst_reduction_pct": "50.00", "pass_sequence": "sroa;gvn", "pass_invocations": "2", "states_analyzed": "2", "opt_runs": "6", "time_ms": "3.00", "error_message": ""},
                {"method": "random_single_pass_best", "status": "success", "final_ir_path": "random.ll", "final_ir_hash": "h4", "final_ir_inst_count": "7", "root_ir_inst_count": "10", "ir_inst_delta": "-3", "ir_inst_reduction_pct": "30.00", "pass_sequence": "gvn", "pass_invocations": "1", "states_analyzed": "5", "opt_runs": "10", "time_ms": "4.00", "error_message": ""},
            ],
        )
    _write_csv(
        run_dir / "pipeline_replay.csv",
        [
            "program",
            "root_ir_path",
            "optimized_pipeline",
            "replay_output_path",
            "final_ir_path",
            "replay_hash",
            "final_hash",
            "hashes_match",
            "replay_status",
            "error_message",
            "time_ms",
        ],
        [
            {
                "program": "run",
                "root_ir_path": str(run_dir / "states" / "S0000" / "input.ll"),
                "optimized_pipeline": "mem2reg,sroa",
                "replay_output_path": str(run_dir / "replayed_final.ll"),
                "final_ir_path": str(run_dir / "final.ll"),
                "replay_hash": "hash-a",
                "final_hash": "hash-a" if replay_hashes_match == "true" else "hash-b",
                "hashes_match": replay_hashes_match,
                "replay_status": replay_status,
                "error_message": "",
                "time_ms": "1.0",
            }
        ],
    )
    (run_dir / "optimized_pipeline.txt").write_text("mem2reg,sroa\n", encoding="utf-8")
    (run_dir / "exact_status.txt").write_text("exact_complete\n", encoding="utf-8")
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "input": "benchmarks/tiny/branch.c",
                "pass_config": "configs/core_passes.yaml",
                "mode": "exact",
                "objective": "ir-inst-count",
                "max_rounds": 2,
                "beam_width": 8,
                "max_batches_per_state": 20,
                "batch_selection_policy": "score",
                "frontier_selection_policy": "score",
                "tools": {
                    "clang": {"path": "clang", "version": "clang version test"},
                    "opt": {"path": "opt", "version": "LLVM opt test"},
                },
            }
        ),
        encoding="utf-8",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
