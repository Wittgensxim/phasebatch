import csv
import math
import tempfile
import unittest
from pathlib import Path

from phasebatch.exact_reduction_study import summarize_exact_reduction_study


class ExactReductionStudyTests(unittest.TestCase):
    def test_study_aggregates_reduction_evidence_from_existing_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_a = root / "alpha" / "optimize"
            run_b = root / "beta" / "optimize"
            _make_run(run_a, program="alpha", dropped="2")
            _make_run(run_b, program="beta", missing_correctness=True)
            out_dir = root / "study"

            result = summarize_exact_reduction_study([run_a, run_b], out_dir, label="smoke")

            runs = _read_csv(out_dir / "exact_reduction_runs.csv")
            by_state = _read_csv(out_dir / "reduction_by_state_all.csv")
            by_program = _read_csv(out_dir / "reduction_by_program.csv")
            evidence = _read_csv(out_dir / "evidence_by_batch_all.csv")
            selected = _read_csv(out_dir / "selected_path_evidence.csv")
            coverage = _read_csv(out_dir / "coverage_by_program.csv")
            markdown = (out_dir / "exact_reduction_summary.md").read_text(encoding="utf-8")

        self.assertEqual(result["programs"], 2)
        self.assertEqual(result["successes"], 2)
        self.assertEqual(len(runs), 2)
        self.assertEqual({row["program"] for row in by_program}, {"alpha", "beta"})

        alpha_s0 = next(row for row in by_state if row["program"] == "alpha" and row["state_id"] == "S0000")
        self.assertAlmostEqual(float(alpha_s0["naive_orderings_log10"]), math.log10(24), places=5)
        self.assertEqual(alpha_s0["executable_batches"], "2")
        self.assertEqual(alpha_s0["local_reduction_ratio_capped"], "12")
        self.assertEqual(alpha_s0["selected_on_final_path"], "true")

        beta_s0 = next(row for row in by_state if row["program"] == "beta" and row["state_id"] == "S0000")
        self.assertEqual(beta_s0["executable_batches"], "0")
        self.assertEqual(beta_s0["no_executable_batches"], "true")
        self.assertNotIn("inf", beta_s0["local_reduction_log10"].lower())
        self.assertGreater(float(beta_s0["local_reduction_log10"]), 0.0)

        strengths = {(row["program"], row["batch_id"]): row["evidence_strength"] for row in evidence}
        self.assertEqual(strengths[("alpha", "B0000")], "strong")
        self.assertEqual(strengths[("alpha", "B0001")], "weak")
        self.assertEqual(strengths[("alpha", "B0002")], "rejected")
        self.assertEqual(strengths[("beta", "B0000")], "unknown")

        self.assertEqual(len([row for row in selected if row["program"] == "alpha"]), 1)
        self.assertEqual(selected[0]["ir_inst_delta"], "-3")
        alpha_coverage = next(row for row in coverage if row["program"] == "alpha")
        self.assertEqual(alpha_coverage["dropped_active_passes"], "2")
        self.assertIn("**WARNING**", markdown)
        self.assertIn("Search-space reduction is state-local.", markdown)
        self.assertIn("## Selected Path Evidence", markdown)

    def test_root_dir_discovery_finds_optimize_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_run(root / "alpha" / "optimize", program="alpha")
            _make_run(root / "beta" / "optimize", program="beta")
            out_dir = root / "out"

            summarize_exact_reduction_study([], out_dir, label="discovered", root_dir=root)

            runs = _read_csv(out_dir / "exact_reduction_runs.csv")

        self.assertEqual({row["program"] for row in runs}, {"alpha", "beta"})


def _make_run(run_dir: Path, *, program: str, dropped: str = "0", missing_correctness: bool = False) -> None:
    states_dir = run_dir / "states"
    s0 = states_dir / "S0000"
    s1 = states_dir / "S0001"
    s0.mkdir(parents=True, exist_ok=True)
    s1.mkdir(parents=True, exist_ok=True)
    (run_dir / "optimized_pipeline.txt").write_text("a,b\n", encoding="utf-8")
    (run_dir / "exact_status.txt").write_text("exact_complete\n", encoding="utf-8")
    (run_dir / "final.ll").write_text("define i32 @f(){ ret i32 0 }\n", encoding="utf-8")
    _write_csv(run_dir / "pipeline_replay.csv", ["replay_status", "hashes_match"], [{"replay_status": "success", "hashes_match": "true"}])
    _write_csv(
        run_dir / "states.csv",
        ["program", "state_id", "depth", "state_hash", "state_dir"],
        [
            {"program": program, "state_id": "S0000", "depth": "0", "state_hash": "h0", "state_dir": str(s0)},
            {"program": program, "state_id": "S0001", "depth": "1", "state_hash": "h1", "state_dir": str(s1)},
        ],
    )
    _write_csv(
        run_dir / "state_dag.csv",
        [
            "program",
            "source_state_id",
            "target_state_id",
            "batch_id",
            "batch_passes",
            "canonical_order",
            "validation_status",
            "correctness_class",
            "is_duplicate",
            "duplicate_of",
        ],
        [
            {"program": program, "source_state_id": "S0000", "target_state_id": "S0001", "batch_id": "B0000", "batch_passes": "a;b", "canonical_order": "a;b", "validation_status": "all_permutations_same", "correctness_class": "certified_batch", "is_duplicate": "false", "duplicate_of": ""},
            {"program": program, "source_state_id": "S0000", "target_state_id": "S0002", "batch_id": "B0001", "batch_passes": "c;d", "canonical_order": "c;d", "validation_status": "sampled_same", "correctness_class": "sampled_batch", "is_duplicate": "true", "duplicate_of": "S0001"},
            {"program": program, "source_state_id": "S0000", "target_state_id": "S0003", "batch_id": "B0002", "batch_passes": "e;f", "canonical_order": "e;f", "validation_status": "mismatch", "correctness_class": "rejected_batch", "is_duplicate": "false", "duplicate_of": ""},
        ],
    )
    _write_csv(
        run_dir / "batch_state_transitions.csv",
        ["program", "parent_state_id", "child_state_id", "batch_id", "validation_status"],
        [{"program": program, "parent_state_id": "S0000", "child_state_id": "S0001", "batch_id": "B0000", "validation_status": "all_permutations_same"}],
    )
    _write_csv(
        run_dir / "chosen_path.csv",
        ["step", "parent_state_id", "child_state_id", "batch_id", "batch_passes", "canonical_order", "ir_inst_before", "ir_inst_after", "ir_inst_delta"],
        [{"step": "0", "parent_state_id": "S0000", "child_state_id": "S0001", "batch_id": "B0000", "batch_passes": "a;b", "canonical_order": "a;b", "ir_inst_before": "10", "ir_inst_after": "7", "ir_inst_delta": "-3"}],
    )
    _write_csv(
        run_dir / "chosen_path_summary.csv",
        ["program", "selected_final_state", "path_steps", "total_pass_invocations", "final_ir_inst_count"],
        [{"program": program, "selected_final_state": "S0001", "path_steps": "1", "total_pass_invocations": "2", "final_ir_inst_count": "7"}],
    )
    _write_csv(
        s0 / "per_state_summary.csv",
        ["program", "state_id", "depth", "state_hash", "active_passes", "pairs_tested", "dynamic_commute", "order_sensitive", "unknown"],
        [{"program": program, "state_id": "S0000", "depth": "0", "state_hash": "h0", "active_passes": "4", "pairs_tested": "6", "dynamic_commute": "3", "order_sensitive": "2", "unknown": "1"}],
    )
    _write_csv(
        s0 / "batch_candidates.csv",
        ["batch_id", "batch_passes", "canonical_order"],
        [
            {"batch_id": "B0000", "batch_passes": "a;b", "canonical_order": "a;b"},
            {"batch_id": "B0001", "batch_passes": "c;d", "canonical_order": "c;d"},
            {"batch_id": "B0002", "batch_passes": "e;f", "canonical_order": "e;f"},
        ],
    )
    _write_csv(
        s0 / "batch_validation.csv",
        ["batch_id", "canonical_order", "tested_orders", "same_hash_count", "different_hash_count", "validation_status", "canonical_hash", "first_mismatch_order", "first_mismatch_hash"],
        [
            {"batch_id": "B0000", "canonical_order": "a;b", "tested_orders": "2", "same_hash_count": "2", "different_hash_count": "0", "validation_status": "all_permutations_same", "canonical_hash": "hash0", "first_mismatch_order": "", "first_mismatch_hash": ""},
            {"batch_id": "B0001", "canonical_order": "c;d", "tested_orders": "20", "same_hash_count": "20", "different_hash_count": "0", "validation_status": "sampled_same", "canonical_hash": "hash1", "first_mismatch_order": "", "first_mismatch_hash": ""},
            {"batch_id": "B0002", "canonical_order": "e;f", "tested_orders": "2", "same_hash_count": "1", "different_hash_count": "1", "validation_status": "mismatch", "canonical_hash": "hash2", "first_mismatch_order": "f;e", "first_mismatch_hash": "bad"},
        ],
    )
    if not missing_correctness:
        _write_csv(
            s0 / "batch_correctness.csv",
            ["batch_id", "batch_passes", "validation_status", "correctness_class", "can_hard_fold", "can_execute"],
            [
                {"batch_id": "B0000", "batch_passes": "a;b", "validation_status": "all_permutations_same", "correctness_class": "certified_batch", "can_hard_fold": "true", "can_execute": "true"},
                {"batch_id": "B0001", "batch_passes": "c;d", "validation_status": "sampled_same", "correctness_class": "sampled_batch", "can_hard_fold": "false", "can_execute": "true"},
                {"batch_id": "B0002", "batch_passes": "e;f", "validation_status": "mismatch", "correctness_class": "rejected_batch", "can_hard_fold": "false", "can_execute": "false"},
            ],
        )
    _write_csv(
        s0 / "coverage_summary.csv",
        ["program", "state_id", "active_passes", "certified_covered", "heuristic_covered", "unresolved_conflict", "validation_rejected", "unvalidated_covered", "failed_or_unknown", "not_executed_due_to_max_depth", "dropped_active_passes"],
        [{"program": program, "state_id": "S0000", "active_passes": "4", "certified_covered": "2", "heuristic_covered": "1", "unresolved_conflict": "0", "validation_rejected": "1", "unvalidated_covered": "0", "failed_or_unknown": "0", "not_executed_due_to_max_depth": "0", "dropped_active_passes": dropped}],
    )
    _write_csv(
        s1 / "per_state_summary.csv",
        ["program", "state_id", "depth", "state_hash", "active_passes", "pairs_tested", "dynamic_commute", "order_sensitive", "unknown"],
        [{"program": program, "state_id": "S0001", "depth": "1", "state_hash": "h1", "active_passes": "0", "pairs_tested": "0", "dynamic_commute": "0", "order_sensitive": "0", "unknown": "0"}],
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
