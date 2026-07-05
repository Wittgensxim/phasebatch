import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from phasebatch.case_studies import export_case_studies, select_representative_state


class CaseStudyExporterTests(unittest.TestCase):
    def test_representative_state_selection_uses_reduction_active_and_relation_flips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            program_dir = _write_program_output(run_dir, "branch")

            selected = select_representative_state(program_dir)

        self.assertEqual(selected, "S0002")

    def test_case_study_generated_from_mock_program_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            program_dir = _write_program_output(run_dir, "branch")

            result = export_case_studies(run_dir, max_pairs=20, max_batches=10)
            case_path = program_dir / "case_study_branch.md"
            index_path = run_dir / "case_studies_index.md"
            case_exists = case_path.exists()
            index_exists = index_path.exists()
            text = case_path.read_text(encoding="utf-8")
            index = index_path.read_text(encoding="utf-8")

        self.assertEqual(result["case_studies"], 1)
        self.assertTrue(case_exists)
        self.assertTrue(index_exists)
        self.assertIn("# Case Study: branch", text)
        self.assertIn("## Selected State", text)
        self.assertIn("- state_id: S0002", text)
        self.assertIn("- parent_state_id: S0000", text)
        self.assertIn("## Active Passes", text)
        self.assertIn("| instcombine | -2 | 1 | 2 | f | f:bb1;f:bb2 |", text)
        self.assertIn("## Pair Relations", text)
        self.assertIn("| instcombine | gvn | final_order_sensitive | order_sensitive | false | same_block |", text)
        self.assertIn("## Conflict / Component Structure", text)
        self.assertIn("## Batch Candidates", text)
        self.assertIn("| B0000 | 2 | instcombine;gvn | certified_batch | true | true |", text)
        self.assertIn("## Batch Validation", text)
        self.assertIn("| B0000 | all_permutations_same | 2 | 2 | 0 |", text)
        self.assertIn("## Coverage", text)
        self.assertIn("- certified_covered: 2", text)
        self.assertIn("## Reduction Estimate", text)
        self.assertIn("- batch_reduction_estimate: 20.00", text)
        self.assertIn("## State Transition Evidence", text)
        self.assertIn("| B0000 | S0003 | 2 | all_permutations_same | false |", text)
        self.assertIn("parent transition: S0000 -> S0002", text)
        self.assertIn("## Interpretation", text)
        self.assertIn("| branch | branch/case_study_branch.md | S0002 | 4 | 2 | 20.00 | 0 |", index)

    def test_missing_files_do_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            program_dir = run_dir / "tiny"
            program_dir.mkdir()
            _write_csv(
                run_dir / "mainline_runs.csv",
                ["program", "input_path", "output_dir", "status", "error_message", "total_time_ms"],
                [{"program": "tiny", "input_path": "tiny.c", "output_dir": str(program_dir), "status": "success", "error_message": "", "total_time_ms": "1"}],
            )
            _write_csv(
                program_dir / "states.csv",
                ["program", "state_id", "state_hash", "depth", "parent_state_id", "transition_pass", "state_dir", "active_passes"],
                [{"program": "tiny", "state_id": "S0000", "state_hash": "h0", "depth": "0", "parent_state_id": "", "transition_pass": "", "state_dir": str(program_dir / "states" / "S0000"), "active_passes": "0"}],
            )

            result = export_case_studies(run_dir)
            text = (program_dir / "case_study_tiny.md").read_text(encoding="utf-8")

        self.assertEqual(result["case_studies"], 1)
        self.assertIn("Missing: pass_profile.csv", text)
        self.assertIn("Missing: pair_relation.csv", text)
        self.assertIn("Missing: batch_summary.csv", text)

    def test_pair_and_batch_limits_truncate_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            program_dir = _write_program_output(run_dir, "branch")

            export_case_studies(run_dir, max_pairs=2, max_batches=1)
            text = (program_dir / "case_study_branch.md").read_text(encoding="utf-8")

        self.assertIn("Showing first 2 of 3 pair relations.", text)
        self.assertIn("| instcombine | gvn | final_order_sensitive | order_sensitive | false | same_block |", text)
        self.assertIn("| simplifycfg | instcombine | final_commute | dynamic_commute | true | disjoint_function |", text)
        self.assertNotIn("| mem2reg | reassociate | final_unknown | unknown |  | unknown |", text)
        self.assertIn("Showing first 1 of 2 batch candidates.", text)
        self.assertIn("| B0000 | 2 | instcombine;gvn | certified_batch | true | true |", text)
        self.assertNotIn("| B0001 | 1 | simplifycfg | rejected_batch | false | false |", text)

    def test_export_case_studies_help_runs(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "phasebatch", "export-case-studies", "--help"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--run-dir", result.stdout)
        self.assertIn("--max-pairs", result.stdout)
        self.assertIn("--max-batches", result.stdout)


def _write_program_output(run_dir: Path, program: str) -> Path:
    program_dir = run_dir / program
    program_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        run_dir / "mainline_runs.csv",
        ["program", "input_path", "output_dir", "status", "error_message", "total_time_ms"],
        [
            {"program": program, "input_path": f"{program}.c", "output_dir": str(program_dir), "status": "success", "error_message": "", "total_time_ms": "10"},
            {"program": "failed", "input_path": "failed.c", "output_dir": str(run_dir / "failed"), "status": "failed", "error_message": "boom", "total_time_ms": "1"},
        ],
    )
    state_rows = [
        _state_row(program_dir, program, "S0000", "h0", "0", "", "", "2"),
        _state_row(program_dir, program, "S0001", "h1", "1", "S0000", "instcombine", "4"),
        _state_row(program_dir, program, "S0002", "h2", "1", "S0000", "instcombine;gvn", "4"),
    ]
    _write_csv(
        program_dir / "states.csv",
        [
            "program",
            "state_id",
            "state_hash",
            "depth",
            "parent_state_id",
            "transition_pass",
            "state_dir",
            "active_passes",
            "pairs_tested",
            "dynamic_commute",
            "order_sensitive",
            "unknown",
            "max_conflict_component",
        ],
        state_rows,
    )
    _write_csv(
        program_dir / "relation_flip.csv",
        ["program", "parent_state_id", "child_state_id", "transition_pass", "pass_a", "pass_b", "parent_relation", "child_relation", "flip_kind"],
        [
            {"program": program, "parent_state_id": "S0000", "child_state_id": "S0001", "transition_pass": "instcombine", "pass_a": "a", "pass_b": "b", "parent_relation": "final_commute", "child_relation": "final_order_sensitive", "flip_kind": "commute_to_sensitive"},
            {"program": program, "parent_state_id": "S0000", "child_state_id": "S0002", "transition_pass": "instcombine;gvn", "pass_a": "a", "pass_b": "b", "parent_relation": "final_commute", "child_relation": "final_order_sensitive", "flip_kind": "commute_to_sensitive"},
            {"program": program, "parent_state_id": "S0000", "child_state_id": "S0002", "transition_pass": "instcombine;gvn", "pass_a": "c", "pass_b": "d", "parent_relation": "final_order_sensitive", "child_relation": "final_commute", "flip_kind": "sensitive_to_commute"},
        ],
    )
    _write_csv(
        program_dir / "batch_state_transitions.csv",
        ["program", "parent_state_id", "child_state_id", "batch_id", "batch_passes", "batch_size", "parent_hash", "child_hash", "is_duplicate", "duplicate_of", "validation_status"],
        [
            {"program": program, "parent_state_id": "S0000", "child_state_id": "S0002", "batch_id": "B9999", "batch_passes": "instcombine;gvn", "batch_size": "2", "parent_hash": "h0", "child_hash": "h2", "is_duplicate": "false", "duplicate_of": "", "validation_status": "all_permutations_same"},
            {"program": program, "parent_state_id": "S0002", "child_state_id": "S0003", "batch_id": "B0000", "batch_passes": "instcombine;gvn", "batch_size": "2", "parent_hash": "h2", "child_hash": "h3", "is_duplicate": "false", "duplicate_of": "", "validation_status": "all_permutations_same"},
        ],
    )
    _write_state_dir(program_dir, program, "S0000", "1.00", "2")
    _write_state_dir(program_dir, program, "S0001", "20.00", "4")
    _write_state_dir(program_dir, program, "S0002", "20.00", "4", full=True)
    return program_dir


def _state_row(program_dir: Path, program: str, state_id: str, state_hash: str, depth: str, parent: str, transition: str, active: str) -> dict[str, str]:
    return {
        "program": program,
        "state_id": state_id,
        "state_hash": state_hash,
        "depth": depth,
        "parent_state_id": parent,
        "transition_pass": transition,
        "state_dir": str(program_dir / "states" / state_id),
        "active_passes": active,
        "pairs_tested": "3",
        "dynamic_commute": "1",
        "order_sensitive": "1",
        "unknown": "1",
        "max_conflict_component": "2",
    }


def _write_state_dir(program_dir: Path, program: str, state_id: str, reduction: str, active_passes: str, full: bool = False) -> None:
    state_dir = program_dir / "states" / state_id
    state_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        state_dir / "per_state_summary.csv",
        ["program", "state_id", "depth", "parent_state_id", "transition_pass", "state_hash", "active_passes", "pairs_tested", "dynamic_commute", "order_sensitive", "unknown", "max_conflict_component"],
        [{"program": program, "state_id": state_id, "depth": "1" if state_id != "S0000" else "0", "parent_state_id": "S0000" if state_id != "S0000" else "", "transition_pass": "instcombine;gvn" if state_id == "S0002" else "", "state_hash": f"h{state_id[-1]}", "active_passes": active_passes, "pairs_tested": "3", "dynamic_commute": "1", "order_sensitive": "1", "unknown": "1", "max_conflict_component": "2"}],
    )
    _write_csv(
        state_dir / "batch_summary.csv",
        ["program", "state_id", "state_hash", "active_passes", "active_pairs", "commute_pairs", "conflict_pairs", "conflict_components", "max_component_size", "batch_candidates", "exact_components", "unresolved_components", "naive_orderings_estimate", "batch_reduction_estimate"],
        [{"program": program, "state_id": state_id, "state_hash": f"h{state_id[-1]}", "active_passes": active_passes, "active_pairs": "3", "commute_pairs": "1", "conflict_pairs": "2", "conflict_components": "1", "max_component_size": "2", "batch_candidates": "2", "exact_components": "1", "unresolved_components": "0", "naive_orderings_estimate": "40", "batch_reduction_estimate": reduction}],
    )
    if not full:
        return
    _write_csv(
        state_dir / "pass_profile.csv",
        ["program", "state_id", "state_hash", "pass", "success", "active", "inst_delta", "funcs_changed", "blocks_changed", "changed_functions", "changed_blocks"],
        [
            {"program": program, "state_id": state_id, "state_hash": "h2", "pass": "instcombine", "success": "true", "active": "true", "inst_delta": "-2", "funcs_changed": "1", "blocks_changed": "2", "changed_functions": "f", "changed_blocks": "f:bb1;f:bb2"},
            {"program": program, "state_id": state_id, "state_hash": "h2", "pass": "gvn", "success": "true", "active": "true", "inst_delta": "-1", "funcs_changed": "1", "blocks_changed": "1", "changed_functions": "g", "changed_blocks": "g:bb1"},
        ],
    )
    _write_csv(
        state_dir / "pair_relation.csv",
        ["program", "state_id", "pass_a", "pass_b", "final_relation", "dynamic_relation", "same_hash", "static_relation"],
        [
            {"program": program, "state_id": state_id, "pass_a": "simplifycfg", "pass_b": "instcombine", "final_relation": "final_commute", "dynamic_relation": "dynamic_commute", "same_hash": "true", "static_relation": "disjoint_function"},
            {"program": program, "state_id": state_id, "pass_a": "mem2reg", "pass_b": "reassociate", "final_relation": "final_unknown", "dynamic_relation": "unknown", "same_hash": "", "static_relation": "unknown"},
            {"program": program, "state_id": state_id, "pass_a": "instcombine", "pass_b": "gvn", "final_relation": "final_order_sensitive", "dynamic_relation": "order_sensitive", "same_hash": "false", "static_relation": "same_block"},
        ],
    )
    _write_csv(
        state_dir / "batch_components.csv",
        ["program", "state_id", "state_hash", "component_id", "component_size", "component_passes", "is_exact", "num_local_alternatives", "unresolved_reason"],
        [{"program": program, "state_id": state_id, "state_hash": "h2", "component_id": "C0000", "component_size": "2", "component_passes": "instcombine;gvn", "is_exact": "true", "num_local_alternatives": "1", "unresolved_reason": ""}],
    )
    _write_csv(
        state_dir / "batch_candidates.csv",
        ["program", "state_id", "state_hash", "batch_id", "batch_passes", "batch_size"],
        [
            {"program": program, "state_id": state_id, "state_hash": "h2", "batch_id": "B0000", "batch_passes": "instcombine;gvn", "batch_size": "2"},
            {"program": program, "state_id": state_id, "state_hash": "h2", "batch_id": "B0001", "batch_passes": "simplifycfg", "batch_size": "1"},
        ],
    )
    _write_csv(
        state_dir / "batch_correctness.csv",
        ["program", "state_id", "state_hash", "batch_id", "batch_passes", "batch_size", "validation_status", "correctness_class", "can_hard_fold", "can_execute", "reason"],
        [
            {"program": program, "state_id": state_id, "state_hash": "h2", "batch_id": "B0000", "batch_passes": "instcombine;gvn", "batch_size": "2", "validation_status": "all_permutations_same", "correctness_class": "certified_batch", "can_hard_fold": "true", "can_execute": "true", "reason": ""},
            {"program": program, "state_id": state_id, "state_hash": "h2", "batch_id": "B0001", "batch_passes": "simplifycfg", "batch_size": "1", "validation_status": "mismatch", "correctness_class": "rejected_batch", "can_hard_fold": "false", "can_execute": "false", "reason": "validation_mismatch"},
        ],
    )
    _write_csv(
        state_dir / "batch_validation.csv",
        ["program", "state_id", "state_hash", "batch_id", "batch_size", "canonical_order", "tested_orders", "same_hash_count", "different_hash_count", "validation_status"],
        [{"program": program, "state_id": state_id, "state_hash": "h2", "batch_id": "B0000", "batch_size": "2", "canonical_order": "instcombine;gvn", "tested_orders": "2", "same_hash_count": "2", "different_hash_count": "0", "validation_status": "all_permutations_same"}],
    )
    _write_csv(
        state_dir / "coverage_report.csv",
        ["program", "state_id", "state_hash", "active_pass", "coverage_status", "covered_by_batch_ids", "component_ids", "correctness_classes", "reason"],
        [
            {"program": program, "state_id": state_id, "state_hash": "h2", "active_pass": "instcombine", "coverage_status": "certified_covered", "covered_by_batch_ids": "B0000", "component_ids": "C0000", "correctness_classes": "certified_batch", "reason": ""},
            {"program": program, "state_id": state_id, "state_hash": "h2", "active_pass": "gvn", "coverage_status": "certified_covered", "covered_by_batch_ids": "B0000", "component_ids": "C0000", "correctness_classes": "certified_batch", "reason": ""},
        ],
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
