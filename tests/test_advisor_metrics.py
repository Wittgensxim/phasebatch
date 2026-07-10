import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from phasebatch.advisor_metrics import (
    component_statistics,
    connected_components,
    local_reduction_metrics,
    percentile_nearest_rank,
    small_cluster_abba_summary,
    summarize_advisor_metrics,
    summarize_pair_relations,
)


class AdvisorMetricTests(unittest.TestCase):
    def test_pair_component_abba_and_reduction_helpers(self) -> None:
        pair_rows = [
            {"pass_a": "a", "pass_b": "b", "final_relation": "final_commute", "equality_tier": "canonical_hash", "time_ms": "2"},
            {"pass_a": "a", "pass_b": "c", "final_relation": "final_unknown", "equality_tier": "failed", "failure_kind": "comparator_failed", "ab_success": "true", "ba_success": "true", "time_ms": "3"},
            {"pass_a": "b", "pass_b": "c", "final_relation": "final_order_sensitive", "equality_tier": "different", "equality_reason": "llvm_diff_difference", "time_ms": "5"},
        ]

        relation = summarize_pair_relations(pair_rows)
        components = connected_components(["a", "b", "c"], [("a", "b")])
        stats = component_statistics(components)
        abba = small_cluster_abba_summary(
            "overlap",
            [{"component_passes": "a;b;c", "component_size": "3"}],
            {("p", "S0000"): pair_rows},
            component_state_keys=[("p", "S0000")],
        )[0]
        reduction = local_reduction_metrics(3, 0)

        self.assertEqual(relation["commute_pairs"], 1)
        self.assertEqual(relation["comparator_failed"], 1)
        self.assertAlmostEqual(relation["commute_ratio"], 1 / 3)
        self.assertEqual(components, [["a", "b"], ["c"]])
        self.assertEqual(stats["mean_component_size"], 1.5)
        self.assertEqual(stats["median_component_size"], 1.5)
        self.assertEqual(stats["p90_component_size"], 2)
        self.assertEqual(percentile_nearest_rank([1, 2, 3, 10], 0.9), 10)
        self.assertEqual(abba["tested_pairs"], "2")
        self.assertEqual(abba["unknown_pairs"], "1")
        self.assertEqual(abba["ab_ba_equal_ratio"], "0.5")
        self.assertAlmostEqual(float(reduction["naive_orderings_log10"]), math.log10(6), places=6)
        self.assertEqual(reduction["local_reduction_log10"], reduction["naive_orderings_log10"])

    def test_study_summary_writes_required_tables_and_preserves_missing_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp)
            run = _build_run(study)
            result = summarize_advisor_metrics(study)

            required = [
                "program_summary.csv",
                "pair_relation_summary.csv",
                "pair_relation_by_state.csv",
                "equality_tier_summary_all.csv",
                "overlap_component_program_summary.csv",
                "conflict_component_program_summary.csv",
                "small_overlap_cluster_abba.csv",
                "unknown_failure_summary.csv",
                "coverage_summary_all.csv",
                "batch_reduction_program_summary.csv",
                "cost_breakdown_by_program.csv",
                "top_conflict_passes.csv",
                "state_aware_by_depth.csv",
                "relation_flip_examples.csv",
                "missing_outputs.csv",
            ]
            coverage = _read_csv(study / "coverage_summary_all.csv")[0]
            effects = _read_csv(study / "state_transition_effects.csv")
            flips = _read_csv(study / "relation_flip_examples.csv")
            overlap = _read_csv(study / "overlap_components_by_state.csv")
            top_conflict = _read_csv(study / "top_conflict_passes.csv")
            missing_names = {Path(row["output"]).name for row in _read_csv(study / "missing_outputs.csv")}
            all_required_exist = all((study / name).exists() for name in required)

        self.assertEqual(result["programs"], 1)
        self.assertTrue(all_required_exist)
        self.assertEqual(coverage["dropped_active_passes"], "")
        self.assertIn("coverage_summary.csv", missing_names)
        self.assertNotIn("batch_candidates.csv", missing_names)
        self.assertIn("enable", {row["effect_kind"] for row in effects})
        self.assertIn("suppress", {row["effect_kind"] for row in effects})
        self.assertIn("effect_changed", {row["effect_kind"] for row in effects})
        self.assertIn("sensitive_to_commute", {row["flip_kind"] for row in flips})
        self.assertIn("pair_availability_change", {row["flip_kind"] for row in flips})
        self.assertTrue(any(row["component_size"] == "1" for row in overlap))
        conflict_scores = [float(row["weighted_conflict_score"]) for row in top_conflict]
        self.assertEqual(conflict_scores, sorted(conflict_scores, reverse=True))
        self.assertEqual(run.name, "optimize")


def _build_run(study: Path) -> Path:
    run = study / "programs" / "demo" / "optimize"
    s0 = run / "states" / "S0000"
    s1 = run / "states" / "S0001"
    s0.mkdir(parents=True)
    s1.mkdir(parents=True)
    (run / "metadata.json").write_text(
        json.dumps({"input": "demo.c", "pair_testing_mode": "full", "batch_construction_mode": "pairwise"}),
        encoding="utf-8",
    )
    (run / "optimized_pipeline.txt").write_text("a,b\n", encoding="utf-8")
    _write_csv(
        run / "states.csv",
        ["program", "state_id", "depth", "parent_state_id", "transition_pass", "state_dir", "is_duplicate", "ir_instructions", "active_passes"],
        [
            {"program": "demo", "state_id": "S0000", "depth": "0", "parent_state_id": "", "transition_pass": "", "state_dir": str(s0), "is_duplicate": "false", "ir_instructions": "20", "active_passes": "3"},
            {"program": "demo", "state_id": "S0001", "depth": "1", "parent_state_id": "S0000", "transition_pass": "a;b", "state_dir": str(s1), "is_duplicate": "false", "ir_instructions": "10", "active_passes": "3"},
        ],
    )
    _write_csv(
        run / "state_dag.csv",
        ["source_state_id", "target_state_id", "batch_id", "batch_passes", "correctness_class"],
        [{"source_state_id": "S0000", "target_state_id": "S0001", "batch_id": "B0", "batch_passes": "a;b", "correctness_class": "certified_batch"}],
    )
    _write_csv(
        run / "chosen_path.csv",
        ["parent_state_id", "child_state_id"],
        [{"parent_state_id": "S0000", "child_state_id": "S0001"}],
    )
    _write_csv(run / "valid_passes.csv", ["pass"], [{"pass": name} for name in ["a", "b", "c", "d"]])
    _write_csv(run / "invalid_passes.csv", ["pass"], [])
    _write_csv(
        run / "optimizer_timing.csv",
        ["optimizer_total_time_ms", "profiling_time_ms", "pair_testing_time_ms", "batch_validation_time_ms", "batch_apply_time_ms", "total_opt_invocations"],
        [{"optimizer_total_time_ms": "100", "profiling_time_ms": "10", "pair_testing_time_ms": "20", "batch_validation_time_ms": "40", "batch_apply_time_ms": "5", "total_opt_invocations": "12"}],
    )
    _write_csv(
        run / "pair_cost_summary.csv",
        ["pair_test_pass_invocations_baseline", "pair_test_pass_invocations_actual", "pair_test_pass_invocations_saved", "cache_hits", "cache_misses", "comparator_time_ms"],
        [{"pair_test_pass_invocations_baseline": "12", "pair_test_pass_invocations_actual": "6", "pair_test_pass_invocations_saved": "6", "cache_hits": "2", "cache_misses": "4", "comparator_time_ms": "7"}],
    )

    profile_fields = ["pass", "success", "active", "inst_delta", "changed_functions", "changed_blocks", "time_ms"]
    _write_csv(
        s0 / "pass_profile.csv",
        profile_fields,
        [
            {"pass": "a", "success": "true", "active": "true", "inst_delta": "-1", "changed_functions": "f", "changed_blocks": "f::b"},
            {"pass": "b", "success": "true", "active": "true", "inst_delta": "-2", "changed_functions": "f", "changed_blocks": "f::b"},
            {"pass": "c", "success": "true", "active": "true", "inst_delta": "-3", "changed_functions": "g", "changed_blocks": "g::b"},
        ],
    )
    _write_csv(
        s1 / "pass_profile.csv",
        profile_fields,
        [
            {"pass": "a", "success": "true", "active": "false", "inst_delta": "0", "changed_functions": "[]", "changed_blocks": "[]"},
            {"pass": "b", "success": "true", "active": "true", "inst_delta": "-4", "changed_functions": "h", "changed_blocks": "h::b"},
            {"pass": "c", "success": "true", "active": "true", "inst_delta": "-3", "changed_functions": "g", "changed_blocks": "g::b"},
            {"pass": "d", "success": "true", "active": "true", "inst_delta": "-1", "changed_functions": "h", "changed_blocks": "h::c"},
        ],
    )
    pair_fields = ["pass_a", "pass_b", "final_relation", "equality_tier", "failure_kind", "time_ms", "skipped_by_budget"]
    _write_csv(
        s0 / "pair_relation.csv",
        pair_fields,
        [
            {"pass_a": "a", "pass_b": "b", "final_relation": "final_commute", "equality_tier": "canonical_hash", "time_ms": "2"},
            {"pass_a": "a", "pass_b": "c", "final_relation": "final_unknown", "equality_tier": "failed", "failure_kind": "timeout", "time_ms": "3"},
            {"pass_a": "b", "pass_b": "c", "final_relation": "final_order_sensitive", "equality_tier": "different", "time_ms": "5"},
        ],
    )
    _write_csv(
        s1 / "pair_relation.csv",
        pair_fields,
        [
            {"pass_a": "b", "pass_b": "c", "final_relation": "final_commute", "equality_tier": "structural_diff", "time_ms": "4"},
            {"pass_a": "b", "pass_b": "d", "final_relation": "final_unknown", "equality_tier": "failed", "failure_kind": "comparator_failed", "time_ms": "6"},
            {"pass_a": "c", "pass_b": "d", "final_relation": "final_commute", "equality_tier": "canonical_hash", "time_ms": "2"},
        ],
    )
    overlap_fields = ["pass_a", "pass_b", "overlap_kind"]
    _write_csv(
        s0 / "footprint_overlap.csv",
        overlap_fields,
        [
            {"pass_a": "a", "pass_b": "b", "overlap_kind": "same_block_overlap"},
            {"pass_a": "a", "pass_b": "c", "overlap_kind": "unknown_overlap"},
            {"pass_a": "b", "pass_b": "c", "overlap_kind": "disjoint_write"},
        ],
    )
    _write_csv(
        s1 / "footprint_overlap.csv",
        overlap_fields,
        [
            {"pass_a": "b", "pass_b": "c", "overlap_kind": "same_function_overlap"},
            {"pass_a": "b", "pass_b": "d", "overlap_kind": "unknown_overlap"},
            {"pass_a": "c", "pass_b": "d", "overlap_kind": "disjoint_write"},
        ],
    )
    _write_csv(s0 / "batch_candidates.csv", ["batch_id"], [{"batch_id": "B0"}, {"batch_id": "B1"}])
    _write_csv(s1 / "batch_candidates.csv", ["batch_id"], [])
    _write_csv(
        s0 / "batch_validation_ladder_summary.csv",
        ["hard_certified_batches", "executable_batches", "sampled_batches", "bounded_batches", "rejected_batches", "failed_batches", "unvalidated_batches", "validation_time_ms", "validation_pass_invocations_baseline", "validation_pass_invocations_actual", "validation_pass_invocations_saved", "validation_transition_cache_hits", "validation_equivalence_cache_hits"],
        [{"hard_certified_batches": "1", "executable_batches": "1", "sampled_batches": "0", "bounded_batches": "0", "rejected_batches": "1", "failed_batches": "0", "unvalidated_batches": "0", "validation_time_ms": "30", "validation_pass_invocations_baseline": "8", "validation_pass_invocations_actual": "4", "validation_pass_invocations_saved": "4", "validation_transition_cache_hits": "2", "validation_equivalence_cache_hits": "1"}],
    )
    _write_csv(
        s1 / "batch_validation_ladder_summary.csv",
        ["hard_certified_batches", "executable_batches", "sampled_batches", "bounded_batches", "rejected_batches", "failed_batches", "unvalidated_batches", "validation_time_ms"],
        [{"hard_certified_batches": "1", "executable_batches": "1", "sampled_batches": "0", "bounded_batches": "0", "rejected_batches": "0", "failed_batches": "0", "unvalidated_batches": "0", "validation_time_ms": "10"}],
    )
    _write_csv(
        s0 / "coverage_summary.csv",
        ["active_passes", "certified_covered", "heuristic_covered", "unresolved_conflict", "validation_rejected", "unvalidated_covered", "failed_or_unknown", "not_executed_due_to_max_depth", "dropped_active_passes"],
        [{"active_passes": "3", "certified_covered": "3", "heuristic_covered": "0", "unresolved_conflict": "0", "validation_rejected": "0", "unvalidated_covered": "0", "failed_or_unknown": "0", "not_executed_due_to_max_depth": "0", "dropped_active_passes": "0"}],
    )
    _write_csv(s0 / "per_state_summary.csv", ["profile_time_ms", "pair_time_ms", "total_time_ms"], [{"profile_time_ms": "5", "pair_time_ms": "10", "total_time_ms": "20"}])
    _write_csv(s1 / "per_state_summary.csv", ["profile_time_ms", "pair_time_ms", "total_time_ms"], [{"profile_time_ms": "5", "pair_time_ms": "10", "total_time_ms": "20"}])
    return run


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
