import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.batch_explorer import _select_candidate_batches, _select_next_frontier, explore_batches
from phasebatch.schema import BATCH_VALIDATION_FIELDS, RunResult


class BatchExplorerTests(unittest.TestCase):
    def test_depth_one_batch_explore_caches_duplicate_batch_successors_with_certified_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, fake_analyze = _run_fake_batch_explore(
                Path(tmp),
                validate_batches=True,
                validation_statuses={"B0000": "all_permutations_same", "B0001": "all_permutations_same"},
            )

            states = _read_csv(out_dir / "states.csv")
            transitions = _read_csv(out_dir / "batch_state_transitions.csv")
            skipped = _read_csv(out_dir / "skipped_batches.csv")
            correctness = _read_csv(out_dir / "states" / "S0000" / "batch_correctness.csv")
            enable_suppress = _read_csv(out_dir / "enable_suppress.csv")
            relation_flips = _read_csv(out_dir / "relation_flip.csv")
            aggregate = _read_csv(out_dir / "aggregate_by_depth.csv")
            aggregate_coverage = _read_csv(out_dir / "aggregate_coverage_summary.csv")
            aggregate_overlap = _read_csv(out_dir / "aggregate_overlap_summary.csv")
            multistate_summary = (out_dir / "multistate_summary.md").read_text(encoding="utf-8")
            summary = (out_dir / "batch_explore_summary.md").read_text(encoding="utf-8")
            root_coverage_exists = (out_dir / "states" / "S0000" / "coverage_report.csv").exists()
            root_footprint = _read_csv(out_dir / "states" / "S0000" / "footprint_overlap.csv")
            child_footprint_exists = (out_dir / "states" / "S0001" / "footprint_overlap.csv").exists()

        self.assertEqual(result["states"], 3)
        self.assertEqual(result["batch_transitions"], 2)
        self.assertEqual(fake_analyze.call_count, 2)
        self.assertEqual([row["state_id"] for row in states], ["S0000", "S0001", "S0002"])
        self.assertEqual(states[2]["is_duplicate"], "true")
        self.assertEqual(states[2]["duplicate_of"], "S0001")
        self.assertEqual([row["batch_id"] for row in transitions], ["B0000", "B0001"])
        self.assertEqual(transitions[1]["is_duplicate"], "true")
        self.assertEqual(transitions[1]["duplicate_of"], "S0001")
        self.assertEqual(transitions[0]["validation_status"], "all_permutations_same")
        self.assertEqual({row["correctness_class"] for row in correctness}, {"certified_batch"})
        self.assertEqual(skipped, [])
        self.assertEqual(len(enable_suppress), 4)
        self.assertIn("suppress", {row["relation"] for row in enable_suppress})
        self.assertIn("effect_changed", {row["relation"] for row in enable_suppress})
        self.assertEqual(len(relation_flips), 2)
        self.assertEqual({row["flip_kind"] for row in relation_flips}, {"sensitive_to_commute"})
        self.assertEqual(aggregate[1]["state_cache_hits"], "1")
        self.assertEqual(aggregate[1]["suppress_count"], "2")
        self.assertEqual(aggregate[1]["effect_changed_count"], "2")
        self.assertEqual(aggregate[1]["relation_flip_count"], "2")
        self.assertEqual(aggregate[1]["true_relation_flip_count"], "2")
        self.assertTrue(root_coverage_exists)
        self.assertEqual(len(root_footprint), 1)
        self.assertTrue(child_footprint_exists)
        self.assertEqual([row["depth"] for row in aggregate_coverage], ["0", "1"])
        self.assertEqual(aggregate_coverage[0]["dropped_active_passes"], "0")
        self.assertEqual(aggregate_coverage[1]["not_executed_due_to_max_depth"], "1")
        self.assertEqual(aggregate_coverage[1]["unvalidated_covered"], "0")
        self.assertEqual([row["depth"] for row in aggregate_overlap], ["0", "1"])
        self.assertEqual(aggregate_overlap[0]["total_pairs"], "1")
        self.assertEqual(aggregate_overlap[0]["unknown_overlap"], "1")
        self.assertIn("Enable/suppress counts", multistate_summary)
        self.assertIn("True relation flips among pairs active in both states", multistate_summary)
        self.assertIn("Batch Explore Summary", summary)
        self.assertIn("Coverage Invariant", summary)
        self.assertIn("Coarse Footprint / Overlap Diagnostics", summary)
        self.assertIn("These coarse footprint labels are diagnostics only. They are not used as hard independence proof in this MVP.", summary)
        self.assertIn("total batch candidates: 2", summary)
        self.assertIn("executed batches: 2", summary)
        self.assertIn("skipped batches: 0", summary)
        self.assertEqual(result["enable_suppress_csv"], str(out_dir / "enable_suppress.csv"))
        self.assertEqual(result["relation_flip_csv"], str(out_dir / "relation_flip.csv"))
        self.assertEqual(result["batch_state_transitions_csv"], str(out_dir / "batch_state_transitions.csv"))
        self.assertEqual(result["skipped_batches_csv"], str(out_dir / "skipped_batches.csv"))
        self.assertEqual(result["aggregate_batch_summary_csv"], str(out_dir / "aggregate_batch_summary.csv"))
        self.assertEqual(result["aggregate_coverage_summary_csv"], str(out_dir / "aggregate_coverage_summary.csv"))
        self.assertEqual(result["aggregate_overlap_summary_csv"], str(out_dir / "aggregate_overlap_summary.csv"))
        self.assertEqual(result["batch_explore_summary"], str(out_dir / "batch_explore_summary.md"))

    def test_validation_gate_skips_mismatch_and_failed_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, fake_analyze = _run_fake_batch_explore(
                Path(tmp),
                validate_batches=True,
                validation_statuses={"B0000": "all_permutations_same", "B0001": "mismatch"},
            )

            states = _read_csv(out_dir / "states.csv")
            transitions = _read_csv(out_dir / "batch_state_transitions.csv")
            skipped = _read_csv(out_dir / "skipped_batches.csv")
            summary = (out_dir / "batch_explore_summary.md").read_text(encoding="utf-8")

        self.assertEqual(result["states"], 2)
        self.assertEqual(result["batch_transitions"], 1)
        self.assertEqual(fake_analyze.call_count, 2)
        self.assertEqual([row["state_id"] for row in states], ["S0000", "S0001"])
        self.assertEqual([row["batch_id"] for row in transitions], ["B0000"])
        self.assertEqual(transitions[0]["validation_status"], "all_permutations_same")
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["batch_id"], "B0001")
        self.assertEqual(skipped[0]["validation_status"], "mismatch")
        self.assertEqual(skipped[0]["correctness_class"], "rejected_batch")
        self.assertEqual(skipped[0]["skip_reason"], "validation_mismatch")
        self.assertIn("total batch candidates: 2", summary)
        self.assertIn("executed batches: 1", summary)
        self.assertIn("skipped batches: 1", summary)
        self.assertIn("| mismatch | 1 |", summary)

    def test_validation_gate_skips_sampled_batches_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, _fake_analyze = _run_fake_batch_explore(
                Path(tmp),
                validate_batches=True,
                validation_statuses={"B0000": "all_permutations_same", "B0001": "sampled_same"},
            )

            transitions = _read_csv(out_dir / "batch_state_transitions.csv")
            skipped = _read_csv(out_dir / "skipped_batches.csv")

        self.assertEqual(result["batch_transitions"], 1)
        self.assertEqual([row["batch_id"] for row in transitions], ["B0000"])
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["batch_id"], "B0001")
        self.assertEqual(skipped[0]["validation_status"], "sampled_same")
        self.assertEqual(skipped[0]["correctness_class"], "sampled_batch")
        self.assertEqual(skipped[0]["skip_reason"], "sampled_not_allowed")

    def test_correctness_gate_skips_failed_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, _fake_analyze = _run_fake_batch_explore(
                Path(tmp),
                validate_batches=True,
                validation_statuses={"B0000": "all_permutations_same", "B0001": "failed"},
            )

            transitions = _read_csv(out_dir / "batch_state_transitions.csv")
            skipped = _read_csv(out_dir / "skipped_batches.csv")

        self.assertEqual(result["batch_transitions"], 1)
        self.assertEqual([row["batch_id"] for row in transitions], ["B0000"])
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["batch_id"], "B0001")
        self.assertEqual(skipped[0]["validation_status"], "failed")
        self.assertEqual(skipped[0]["correctness_class"], "failed_batch")
        self.assertEqual(skipped[0]["skip_reason"], "validation_failed")

    def test_allow_sampled_batches_executes_sampled_same_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, _fake_analyze = _run_fake_batch_explore(
                Path(tmp),
                validate_batches=True,
                allow_sampled_batches=True,
                validation_statuses={"B0000": "all_permutations_same", "B0001": "sampled_same"},
            )

            transitions = _read_csv(out_dir / "batch_state_transitions.csv")
            skipped = _read_csv(out_dir / "skipped_batches.csv")
            summary = (out_dir / "batch_explore_summary.md").read_text(encoding="utf-8")

        self.assertEqual(result["batch_transitions"], 2)
        self.assertEqual([row["validation_status"] for row in transitions], ["all_permutations_same", "sampled_same"])
        self.assertEqual(skipped, [])
        self.assertIn("executed batches: 2", summary)
        self.assertIn("skipped batches: 0", summary)

    def test_depth_two_batch_explore_expands_non_duplicate_frontier_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, fake_analyze = _run_fake_batch_explore(
                Path(tmp),
                validate_batches=True,
                validation_statuses={"B0000": "all_permutations_same", "B0001": "all_permutations_same"},
                max_depth=2,
                unique_outputs=True,
            )

            states = _read_csv(out_dir / "states.csv")
            transitions = _read_csv(out_dir / "batch_state_transitions.csv")
            aggregate = _read_csv(out_dir / "aggregate_by_depth.csv")
            summary = (out_dir / "batch_explore_summary.md").read_text(encoding="utf-8")
            child_input_exists = (out_dir / "states" / "S0001" / "input.ll").exists()
            depth_two_batches_exist = (out_dir / "states" / "S0003" / "batch_candidates.csv").exists()

        self.assertEqual(result["states"], 5)
        self.assertEqual(result["batch_transitions"], 4)
        self.assertEqual(fake_analyze.call_count, 5)
        self.assertEqual([row["depth"] for row in states], ["0", "1", "1", "2", "2"])
        self.assertEqual([row["parent_state_id"] for row in transitions], ["S0000", "S0000", "S0001", "S0002"])
        self.assertEqual({row["is_duplicate"] for row in states}, {"false"})
        self.assertTrue(child_input_exists)
        self.assertTrue(depth_two_batches_exist)
        self.assertEqual(aggregate[2]["depth"], "2")
        self.assertEqual(aggregate[2]["num_states"], "2")
        self.assertIn("states explored: 5", summary)
        self.assertIn("batch transitions: 4", summary)
        self.assertIn("total batch candidates: 4", summary)

    def test_aggregate_batch_summary_groups_batch_metrics_by_depth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, _fake_analyze = _run_fake_batch_explore(
                Path(tmp),
                validate_batches=True,
                allow_sampled_batches=True,
                validation_statuses={"B0000": "all_permutations_same", "B0001": "sampled_same"},
                max_depth=2,
                unique_outputs=True,
            )

            aggregate = _read_csv(out_dir / "aggregate_batch_summary.csv")
            summary = (out_dir / "batch_explore_summary.md").read_text(encoding="utf-8")

        self.assertEqual(result["aggregate_batch_summary_csv"], str(out_dir / "aggregate_batch_summary.csv"))
        self.assertEqual([row["depth"] for row in aggregate], ["0", "1", "2"])
        self.assertEqual(aggregate[0]["states"], "1")
        self.assertEqual(aggregate[0]["avg_candidates"], "2.00")
        self.assertEqual(aggregate[0]["avg_batch_size"], "1.00")
        self.assertEqual(aggregate[0]["avg_reduction"], "1.00")
        self.assertEqual(aggregate[0]["executed"], "2")
        self.assertEqual(aggregate[0]["skipped"], "0")
        self.assertEqual(aggregate[0]["all_permutations_same"], "1")
        self.assertEqual(aggregate[0]["sampled_same"], "1")
        self.assertEqual(aggregate[0]["validation_counts"], "all_permutations_same=1; sampled_same=1")
        self.assertEqual(aggregate[1]["states"], "2")
        self.assertEqual(aggregate[1]["avg_candidates"], "1.00")
        self.assertEqual(aggregate[1]["executed"], "2")
        self.assertEqual(aggregate[1]["all_permutations_same"], "2")
        self.assertEqual(aggregate[2]["states"], "2")
        self.assertEqual(aggregate[2]["executed"], "0")
        self.assertEqual(aggregate[2]["validation_counts"], "")
        self.assertIn("## By-depth Batch Summary", summary)
        self.assertIn("| depth | states | avg candidates | avg batch size | avg reduction | executed | skipped | validation counts |", summary)
        self.assertIn("| 0 | 1 | 2.00 | 1.00 | 1.00 | 2 | 0 | all_permutations_same=1; sampled_same=1 |", summary)

    def test_max_batches_per_state_caps_applied_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, _fake_analyze = _run_fake_batch_explore(
                Path(tmp),
                validate_batches=True,
                validation_statuses={"B0000": "all_permutations_same", "B0001": "all_permutations_same"},
                unique_outputs=True,
                max_batches_per_state=1,
            )

            transitions = _read_csv(out_dir / "batch_state_transitions.csv")
            summary = (out_dir / "batch_explore_summary.md").read_text(encoding="utf-8")

        self.assertEqual(result["states"], 2)
        self.assertEqual(result["batch_transitions"], 1)
        self.assertEqual([row["batch_id"] for row in transitions], ["B0000"])
        self.assertIn("total batch candidates: 2", summary)
        self.assertIn("selected batch candidates: 1", summary)

    def test_certified_first_policy_sorts_before_batch_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, _fake_analyze = _run_fake_batch_explore(
                Path(tmp),
                validate_batches=True,
                validation_statuses={"B0000": "not_validated", "B0001": "all_permutations_same"},
                unique_outputs=True,
                max_batches_per_state=1,
                batch_frontier_policy="certified-first",
            )

            transitions = _read_csv(out_dir / "batch_state_transitions.csv")
            skipped = _read_csv(out_dir / "skipped_batches.csv")

        self.assertEqual(result["batch_transitions"], 1)
        self.assertEqual(skipped, [])
        self.assertEqual(transitions[0]["batch_id"], "B0001")
        self.assertEqual(transitions[0]["validation_status"], "all_permutations_same")

    def test_max_frontier_states_caps_next_depth_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, _fake_analyze = _run_fake_batch_explore(
                Path(tmp),
                validate_batches=True,
                validation_statuses={"B0000": "all_permutations_same", "B0001": "all_permutations_same"},
                max_depth=2,
                unique_outputs=True,
                max_frontier_states=1,
            )

            states = _read_csv(out_dir / "states.csv")
            transitions = _read_csv(out_dir / "batch_state_transitions.csv")

        self.assertEqual(result["states"], 4)
        self.assertEqual(result["batch_transitions"], 3)
        self.assertEqual([row["depth"] for row in states], ["0", "1", "1", "2"])
        self.assertEqual([row["parent_state_id"] for row in transitions], ["S0000", "S0000", "S0001"])

    def test_largest_batch_policy_prefers_larger_candidates(self) -> None:
        rows = [
            {"batch_id": "small", "batch_size": "1"},
            {"batch_id": "large", "batch_size": "4"},
            {"batch_id": "medium", "batch_size": "2"},
        ]

        selected = _select_candidate_batches(
            rows,
            validation_map={},
            policy="largest-batch",
            max_batches_per_state=2,
        )

        self.assertEqual([row["batch_id"] for row in selected], ["large", "medium"])

    def test_diverse_hash_policy_keeps_unique_hashes_first(self) -> None:
        rows = [
            {"state_id": "S0001", "state_hash": "h1"},
            {"state_id": "S0002", "state_hash": "h1"},
            {"state_id": "S0003", "state_hash": "h2"},
        ]

        selected = _select_next_frontier(rows, max_frontier_states=2, batch_frontier_policy="diverse-hash")

        self.assertEqual([row["state_id"] for row in selected], ["S0001", "S0003"])


def _run_fake_batch_explore(
    root: Path,
    *,
    validate_batches: bool,
    allow_sampled_batches: bool = False,
    validation_statuses: dict[str, str] | None = None,
    max_depth: int = 1,
    unique_outputs: bool = False,
    max_batches_per_state: int = 20,
    max_frontier_states: int = 20,
    batch_frontier_policy: str = "all",
):
    input_path = root / "input.ll"
    input_path.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
    passes_path = root / "passes.yaml"
    passes_path.write_text("passes:\n  - pass-a\n  - pass-b\n", encoding="utf-8")
    out_dir = root / "batch_explore"
    prepared_ir = out_dir / "input.ll"

    def fake_prepare(src, out, tools, timeout):
        out.mkdir(parents=True, exist_ok=True)
        prepared_ir.write_text(input_path.read_text(encoding="utf-8"), encoding="utf-8")
        return prepared_ir

    def fake_analyze_state(
        input_ll,
        state_dir,
        tools,
        *,
        valid_passes,
        invalid_rows,
        configured_pass_count,
        jobs,
        timeout,
        max_pairs,
        program,
        state_id,
        depth,
        parent_state_id,
        transition_pass,
    ):
        state_dir = Path(state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        if state_id == "S0000":
            _write_csv(
                state_dir / "pass_profile.csv",
                [
                    "program",
                    "state_id",
                    "depth",
                    "parent_state_id",
                    "transition_pass",
                    "state_hash",
                    "pass",
                    "success",
                    "active",
                    "output_hash",
                    "output_path",
                    "inst_delta",
                    "blocks_changed",
                    "changed_functions",
                ],
                [
                    {
                        "program": "batch_explore",
                        "state_id": "S0000",
                        "depth": "0",
                        "parent_state_id": "",
                        "transition_pass": "",
                        "state_hash": "root-hash",
                        "pass": "pass-a",
                        "success": "true",
                        "active": "true",
                        "output_hash": "a-hash",
                        "output_path": str(state_dir / "a.ll"),
                        "inst_delta": "-1",
                        "blocks_changed": "1",
                        "changed_functions": "f",
                    },
                    {
                        "program": "batch_explore",
                        "state_id": "S0000",
                        "depth": "0",
                        "parent_state_id": "",
                        "transition_pass": "",
                        "state_hash": "root-hash",
                        "pass": "pass-b",
                        "success": "true",
                        "active": "true",
                        "output_hash": "b-hash",
                        "output_path": str(state_dir / "b.ll"),
                        "inst_delta": "-1",
                        "blocks_changed": "1",
                        "changed_functions": "f",
                    },
                ],
            )
            _write_csv(
                state_dir / "pair_relation.csv",
                ["program", "state_id", "pass_a", "pass_b", "final_relation"],
                [
                    {
                        "program": "batch_explore",
                        "state_id": "S0000",
                        "pass_a": "pass-a",
                        "pass_b": "pass-b",
                        "final_relation": "final_order_sensitive",
                    }
                ],
            )
            _write_summary(state_dir, "S0000", "root-hash", "2", "0", "1", "1", "11")
            return {"program": "batch_explore", "state_id": "S0000", "summary_path": str(state_dir / "summary.md")}

        _write_csv(
            state_dir / "pass_profile.csv",
            [
                "program",
                "state_id",
                "depth",
                "parent_state_id",
                "transition_pass",
                "state_hash",
                "pass",
                "success",
                "active",
                "output_hash",
                "output_path",
                "inst_delta",
                "blocks_changed",
                "changed_functions",
            ],
            [
                {
                    "program": "batch_explore",
                    "state_id": state_id,
                    "depth": str(depth),
                    "parent_state_id": parent_state_id,
                    "transition_pass": transition_pass,
                    "state_hash": "child-hash",
                    "pass": "pass-a",
                    "success": "true",
                    "active": "false",
                    "output_hash": "",
                    "output_path": "",
                    "inst_delta": "0",
                    "blocks_changed": "0",
                    "changed_functions": "",
                },
                {
                    "program": "batch_explore",
                    "state_id": state_id,
                    "depth": str(depth),
                    "parent_state_id": parent_state_id,
                    "transition_pass": transition_pass,
                    "state_hash": "child-hash",
                    "pass": "pass-b",
                    "success": "true",
                    "active": "true",
                    "output_hash": "next-hash",
                    "output_path": str(state_dir / "next.ll"),
                    "inst_delta": "-3",
                    "blocks_changed": "2",
                    "changed_functions": "f,g",
                },
            ],
        )
        _write_csv(
            state_dir / "pair_relation.csv",
            ["program", "state_id", "pass_a", "pass_b", "final_relation"],
            [
                {
                    "program": "batch_explore",
                    "state_id": state_id,
                    "pass_a": "pass-a",
                    "pass_b": "pass-b",
                    "final_relation": "final_commute",
                }
            ],
        )
        _write_summary(state_dir, state_id, "child-hash", "1", "1", "1", "1", "7")
        return {"program": "batch_explore", "state_id": state_id, "summary_path": str(state_dir / "summary.md")}

    output_counter = {"value": 0}

    def fake_run_opt(opt, src, passes, out, timeout):
        if unique_outputs:
            output_counter["value"] += 1
            text = f"define i32 @f() {{\n  ret i32 {output_counter['value']}\n}}\n"
        else:
            text = "define i32 @f() {\n  ret i32 1\n}\n"
        out.write_text(text, encoding="utf-8")
        return RunResult([opt], 0, "", "", 1.0)

    def fake_validate_batch_candidates(state_dir, tools, timeout, jobs):
        rows = []
        status_by_batch = validation_statuses or {}
        for candidate in _read_csv(Path(state_dir) / "batch_candidates.csv"):
            rows.append(
                {
                    "program": candidate.get("program", ""),
                    "state_id": candidate.get("state_id", ""),
                    "state_hash": candidate.get("state_hash", ""),
                    "batch_id": candidate.get("batch_id", ""),
                    "batch_size": candidate.get("batch_size", ""),
                    "canonical_order": candidate.get("canonical_order", ""),
                    "tested_orders": "1",
                    "same_hash_count": "1",
                    "different_hash_count": "0",
                    "validation_status": status_by_batch.get(candidate.get("batch_id", ""), "not_validated"),
                    "canonical_hash": "hash",
                    "first_mismatch_order": "",
                    "first_mismatch_hash": "",
                    "time_ms": "1",
                }
            )
        _write_csv(Path(state_dir) / "batch_validation.csv", BATCH_VALIDATION_FIELDS, rows)
        return {"validated_batches": len(rows), "batch_validation_csv": str(Path(state_dir) / "batch_validation.csv")}

    with mock.patch(
        "phasebatch.batch_explorer.collect_toolchain",
        return_value={"tools": {"opt": {"path": "opt", "version": "LLVM"}}},
    ), mock.patch(
        "phasebatch.batch_explorer.prepare_input_ir",
        side_effect=fake_prepare,
    ), mock.patch(
        "phasebatch.batch_explorer.validate_passes",
        return_value=(["pass-a", "pass-b"], []),
    ), mock.patch(
        "phasebatch.batch_explorer.analyze_state",
        side_effect=fake_analyze_state,
    ) as fake_analyze, mock.patch(
        "phasebatch.batch_explorer.run_opt",
        side_effect=fake_run_opt,
    ), mock.patch(
        "phasebatch.batch_explorer.validate_batch_candidates",
        side_effect=fake_validate_batch_candidates,
    ):
        result = explore_batches(
            input_path,
            out_dir,
            passes_path,
            jobs=1,
            timeout=1,
            max_pairs=5,
            max_depth=max_depth,
            max_component_size=10,
            max_batch_candidates=50,
            validate_batches=validate_batches,
            allow_sampled_batches=allow_sampled_batches,
            max_batches_per_state=max_batches_per_state,
            max_frontier_states=max_frontier_states,
            batch_frontier_policy=batch_frontier_policy,
        )

    return result, out_dir, fake_analyze


def _write_summary(
    state_dir: Path,
    state_id: str,
    state_hash: str,
    active_passes: str,
    dormant_passes: str,
    pairs_tested: str,
    max_conflict_component: str,
    total_time_ms: str,
) -> None:
    _write_csv(
        state_dir / "per_state_summary.csv",
        [
            "program",
            "state_id",
            "state_hash",
            "active_passes",
            "dormant_passes",
            "pairs_tested",
            "dynamic_commute",
            "order_sensitive",
            "unknown",
            "max_conflict_component",
            "total_time_ms",
        ],
        [
            {
                "program": "batch_explore",
                "state_id": state_id,
                "state_hash": state_hash,
                "active_passes": active_passes,
                "dormant_passes": dormant_passes,
                "pairs_tested": pairs_tested,
                "dynamic_commute": "0",
                "order_sensitive": pairs_tested,
                "unknown": "0",
                "max_conflict_component": max_conflict_component,
                "total_time_ms": total_time_ms,
            }
        ],
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
