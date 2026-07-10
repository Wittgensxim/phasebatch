import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.optimizer import (
    _flatten_pipeline,
    _flatten_pipeline_names,
    _run_timed_batch_validation,
    _select_diversity_preserving_beam,
    _state_pair_matrix_complete,
    _write_optimizer_timing,
    optimize_batches,
    score_frontier_state,
)
from phasebatch.pass_config import PassRegistry, PassSpec
from phasebatch.schema import BATCH_VALIDATION_FIELDS, RunResult


class OptimizerTests(unittest.TestCase):
    def test_lazy_mode_never_reports_a_complete_pair_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_csv(
                state_dir / "pass_profile.csv",
                ["pass", "success", "active"],
                [
                    {"pass": "a", "success": "true", "active": "true"},
                    {"pass": "b", "success": "true", "active": "true"},
                ],
            )
            _write_csv(
                state_dir / "pair_relation.csv",
                ["pass_a", "pass_b", "dynamic_relation", "failure_kind", "skipped_by_budget"],
                [
                    {
                        "pass_a": "a",
                        "pass_b": "b",
                        "dynamic_relation": "dynamic_commute",
                        "failure_kind": "",
                        "skipped_by_budget": "false",
                    }
                ],
            )

            self.assertTrue(_state_pair_matrix_complete(state_dir, pair_testing_mode="full"))
            self.assertFalse(_state_pair_matrix_complete(state_dir, pair_testing_mode="lazy"))

    def test_score_beam_preserves_distinct_static_feature_buckets(self) -> None:
        def row(state_id, *, objective, calls, memory, branches, novelty=0.0, score=0.5):
            return {
                "state_id": state_id,
                "objective_value": str(objective),
                "direct_calls": str(calls),
                "memory_ops": str(memory),
                "branches": str(branches),
                "novelty_score": str(novelty),
                "final_state_score": str(score),
                "pareto_kept": "true",
            }

        rows = [
            row("Sscore", objective=50, calls=5, memory=5, branches=5, score=1.0),
            row("Sobjective", objective=1, calls=5, memory=5, branches=5),
            row("Scall", objective=50, calls=0, memory=5, branches=5),
            row("Smemory", objective=50, calls=5, memory=0, branches=5),
            row("Sbranch", objective=50, calls=5, memory=5, branches=0),
            row("Snovel", objective=50, calls=5, memory=5, branches=5, novelty=1.0),
        ]

        selected = _select_diversity_preserving_beam(rows, beam_width=6)

        self.assertEqual(
            {bucket for _state_id, bucket in selected},
            {
                "objective_bucket",
                "direct_call_bucket",
                "memory_bucket",
                "branch_bucket",
                "novelty_bucket",
                "score_fill",
            },
        )

    def test_timed_validation_call_accumulates_non_overlapping_wall_time(self) -> None:
        context = {"timing": {"batch_validation_wall_time_ms": 25.0}}

        with mock.patch("phasebatch.optimizer.time.perf_counter", side_effect=[10.0, 10.125]):
            result = _run_timed_batch_validation(context, lambda: "done")

        self.assertEqual(result, "done")
        self.assertEqual(context["timing"]["batch_validation_wall_time_ms"], 150.0)

    def test_optimizer_timing_uses_actual_validation_opt_invocations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            state_dir = out_dir / "states" / "S0000"
            state_dir.mkdir(parents=True)
            _write_csv(
                out_dir / "states.csv",
                ["state_id", "state_dir", "is_duplicate"],
                [{"state_id": "S0000", "state_dir": str(state_dir), "is_duplicate": "false"}],
            )
            _write_csv(
                state_dir / "batch_validation.csv",
                ["tested_orders", "validation_opt_invocations", "time_ms"],
                [{"tested_orders": "10", "validation_opt_invocations": "6", "time_ms": "12.5"}],
            )

            path = _write_optimizer_timing(
                out_dir,
                {
                    "program": "testprog",
                    "timing": {"batch_validation_wall_time_ms": 7.25},
                },
                elapsed_ms=20.0,
            )
            row = _read_csv(path)[0]

        self.assertEqual(row["total_opt_invocations"], "6")
        self.assertEqual(row["batch_validation_time_ms"], "7.250")

    def test_optimize_batches_help_exists(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "phasebatch", "optimize-batches", "--help"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--input", result.stdout)
        self.assertIn("--mode", result.stdout)
        self.assertIn("--max-rounds", result.stdout)
        self.assertIn("--max-states", result.stdout)
        self.assertIn("--max-component-size", result.stdout)
        self.assertIn("--max-batch-candidates", result.stdout)
        self.assertIn("--beam-width", result.stdout)
        self.assertIn("--batch-frontier-policy", result.stdout)
        self.assertIn("--batch-selection-policy", result.stdout)
        self.assertIn("--frontier-selection-policy", result.stdout)
        self.assertIn("--selection-seed", result.stdout)
        self.assertIn("--allow-sampled-batches", result.stdout)
        self.assertIn("--allow-bounded-validation", result.stdout)
        self.assertIn("--batch-validation-mode", result.stdout)
        self.assertIn("--max-permutation-factorial", result.stdout)
        self.assertIn("--max-validation-sequences", result.stdout)
        self.assertIn("--pair-testing-mode", result.stdout)
        self.assertIn("--pair-test-budget-per-state", result.stdout)
        self.assertIn("--pair-priority-policy", result.stdout)
        self.assertIn("--batch-construction-mode", result.stdout)
        self.assertIn("--exact-fail-on-incomplete", result.stdout)
        self.assertIn("--batchify-terminal-states", result.stdout)
        self.assertIn("--run-baselines", result.stdout)
        self.assertIn("--verify-final-pipeline", result.stdout)
        self.assertIn("--llvm-diff", result.stdout)
        self.assertIn("--keep-ir-artifacts", result.stdout)

    def test_exact_mode_rejects_on_demand_validation(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Exact mode requires budgeted_validation_strategy=all",
        ):
            optimize_batches(
                Path("input.ll"),
                Path("out"),
                Path("passes.yaml"),
                mode="exact",
                objective="ir-inst-count",
                max_rounds=1,
                max_batches_per_state=2,
                budgeted_validation_strategy="on-demand",
                validate_batches=True,
                allow_sampled_batches=False,
                jobs=1,
                timeout=1,
                max_pairs=None,
            )

    def test_optimizer_rejects_non_pairwise_batch_construction(self) -> None:
        with self.assertRaisesRegex(ValueError, "only supports pairwise"):
            optimize_batches(
                Path("input.ll"),
                Path("out"),
                Path("passes.yaml"),
                mode="budgeted",
                objective="ir-inst-count",
                max_rounds=1,
                max_batches_per_state=2,
                validate_batches=True,
                allow_sampled_batches=False,
                batch_construction_mode="experimental",
                jobs=1,
                timeout=1,
                max_pairs=None,
            )

    def test_budgeted_on_demand_stops_after_enough_executable_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, _fake_analyze, attempts = _run_fake_optimizer(
                Path(tmp),
                validation_statuses={
                    "B0000": "mismatch",
                    "B0001": "all_permutations_same",
                    "B0002": "all_permutations_same",
                    "B0003": "all_permutations_same",
                },
                candidate_ids=["B0000", "B0001", "B0002", "B0003"],
                batch_orders={
                    "B0000": "pass-a;pass-b;pass-c",
                    "B0001": "pass-a;pass-b",
                    "B0002": "pass-a",
                    "B0003": "pass-b",
                },
                child_instruction_counts={"B0001": 2, "B0002": 1},
                max_rounds=1,
                max_batches_per_state=2,
                budgeted_validation_strategy="on-demand",
                return_validation_attempts=True,
            )
            validation_rows = _read_csv(out_dir / "states" / "S0000" / "batch_validation.csv")
            transitions = _read_csv(out_dir / "batch_state_transitions.csv")
            optimize_summary = (out_dir / "optimize_summary.md").read_text(encoding="utf-8")
            final_summary = (out_dir / "final_summary.md").read_text(encoding="utf-8")

        self.assertEqual(attempts, ["B0000", "B0001", "B0002"])
        self.assertEqual(
            [row["validation_status"] for row in validation_rows],
            ["mismatch", "all_permutations_same", "all_permutations_same", "not_validated"],
        )
        self.assertEqual(
            validation_rows[3]["validation_incomplete_reason"],
            "budgeted_on_demand_not_selected",
        )
        self.assertEqual([row["batch_id"] for row in transitions], ["B0001", "B0002"])
        self.assertEqual(result["batch_transitions"], 2)
        self.assertEqual(result["budgeted_validation_strategy"], "on-demand")
        self.assertIn("- budgeted_validation_strategy: on-demand", optimize_summary)
        self.assertIn("- budgeted_validation_strategy: on-demand", final_summary)
        self.assertIn("## Validation Cost", final_summary)

    def test_budgeted_validate_all_strategy_keeps_full_validation_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _result, out_dir, _fake_analyze, attempts = _run_fake_optimizer(
                Path(tmp),
                validation_statuses={
                    "B0000": "mismatch",
                    "B0001": "all_permutations_same",
                    "B0002": "all_permutations_same",
                    "B0003": "all_permutations_same",
                },
                candidate_ids=["B0000", "B0001", "B0002", "B0003"],
                max_rounds=1,
                max_batches_per_state=2,
                budgeted_validation_strategy="all",
                return_validation_attempts=True,
            )
            rows = _read_csv(out_dir / "states" / "S0000" / "batch_validation.csv")

        self.assertEqual(attempts, ["B0000", "B0001", "B0002", "B0003"])
        self.assertNotIn("not_validated", [row["validation_status"] for row in rows])

    def test_flattened_pipeline_uses_pipeline_text_and_names_are_separate(self) -> None:
        registry = PassRegistry.from_specs(
            [
                PassSpec("mem2reg", "mem2reg", ["mem2reg"], "scalar", "v1", True),
                PassSpec("licm", "function(loop(licm))", ["function(loop(licm))"], "loop", "v3", True),
                PassSpec("dce", "dce", ["dce"], "cleanup", "v1", True),
            ]
        )
        rows = [{"canonical_order": "mem2reg;licm"}, {"canonical_order": "dce"}]

        self.assertEqual(_flatten_pipeline_names(rows), "mem2reg,licm,dce")
        self.assertEqual(_flatten_pipeline(rows, registry), "mem2reg,function(loop(licm)),dce")

    def test_max_rounds_one_writes_optimizer_outputs_and_selects_best_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, _fake_analyze = _run_fake_optimizer(
                Path(tmp),
                validation_statuses={"B0000": "all_permutations_same", "B0001": "all_permutations_same"},
                child_instruction_counts={"B0000": 2, "B0001": 1},
            )

            states = _read_csv(out_dir / "states.csv")
            dag = _read_csv(out_dir / "state_dag.csv")
            leaves = _read_csv(out_dir / "leaf_states.csv")
            chosen = _read_csv(out_dir / "chosen_path.csv")
            pipeline = (out_dir / "optimized_pipeline.txt").read_text(encoding="utf-8").strip()
            summary = (out_dir / "optimize_summary.md").read_text(encoding="utf-8")
            output_exists = {
                name: (out_dir / name).exists()
                for name in [
            "states.csv",
            "state_dag.csv",
            "leaf_states.csv",
            "chosen_path.csv",
            "optimized_batches.txt",
            "optimized_pipeline.txt",
            "final.ll",
            "optimize_summary.md",
            "pair_cost_summary.csv",
            "pair_cost_summary.md",
                ]
            }

        self.assertEqual(result["selected_final_state"], "S0002")
        self.assertTrue(result["pair_cost_summary_csv"].endswith("pair_cost_summary.csv"))
        for name, exists in output_exists.items():
            self.assertTrue(exists, name)
        self.assertEqual([row["state_id"] for row in states], ["S0000", "S0001", "S0002"])
        self.assertEqual([row["batch_id"] for row in dag], ["B0000", "B0001"])
        self.assertEqual([row["selected_as_final"] for row in leaves], ["false", "false", "true"])
        self.assertEqual(chosen[0]["child_state_id"], "S0002")
        self.assertEqual(chosen[0]["ir_inst_delta"], "-2")
        self.assertEqual(pipeline, "pass-c")
        self.assertIn("Objective is used only for path selection, not as commutation proof.", summary)

    def test_optimize_batches_passes_batcher_limits_and_records_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _result, out_dir, _fake_analyze, fake_build = _run_fake_optimizer(
                Path(tmp),
                validation_statuses={"B0000": "all_permutations_same"},
                candidate_ids=["B0000"],
                child_instruction_counts={"B0000": 1},
                max_component_size=17,
                max_batch_candidates=23,
                return_build_mock=True,
            )

            summary = (out_dir / "optimize_summary.md").read_text(encoding="utf-8")

        build_kwargs = fake_build.call_args.kwargs
        self.assertEqual(build_kwargs["max_component_size"], 17)
        self.assertEqual(build_kwargs["max_batch_candidates"], 23)
        self.assertIn("- max_component_size: 17", summary)
        self.assertIn("- max_batch_candidates: 23", summary)

    def test_rejected_and_failed_batches_are_not_executed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, fake_analyze = _run_fake_optimizer(
                Path(tmp),
                validation_statuses={"B0000": "mismatch", "B0001": "failed"},
            )

            transitions = _read_csv(out_dir / "batch_state_transitions.csv")
            dag = _read_csv(out_dir / "state_dag.csv")

        self.assertEqual(result["batch_transitions"], 0)
        self.assertEqual(transitions, [])
        self.assertEqual(dag, [])
        self.assertEqual(fake_analyze.call_count, 1)

    def test_sampled_batches_require_allow_sampled_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, _fake_analyze = _run_fake_optimizer(
                Path(tmp) / "default",
                validation_statuses={"B0000": "sampled_same"},
                candidate_ids=["B0000"],
                child_instruction_counts={"B0000": 1},
                allow_sampled_batches=False,
            )
            default_transitions = _read_csv(out_dir / "batch_state_transitions.csv")

        self.assertEqual(result["batch_transitions"], 0)
        self.assertEqual(default_transitions, [])

        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, _fake_analyze = _run_fake_optimizer(
                Path(tmp) / "allowed",
                validation_statuses={"B0000": "sampled_same"},
                candidate_ids=["B0000"],
                child_instruction_counts={"B0000": 1},
                allow_sampled_batches=True,
            )
            allowed_transitions = _read_csv(out_dir / "batch_state_transitions.csv")

        self.assertEqual(result["batch_transitions"], 1)
        self.assertEqual(allowed_transitions[0]["correctness_class"], "sampled_batch")

    def test_bounded_batches_require_allow_bounded_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, _fake_analyze = _run_fake_optimizer(
                Path(tmp) / "default",
                validation_statuses={"B0000": "bounded_same"},
                candidate_ids=["B0000"],
                child_instruction_counts={"B0000": 1},
                allow_bounded_validation=False,
            )
            default_transitions = _read_csv(out_dir / "batch_state_transitions.csv")

        self.assertEqual(result["batch_transitions"], 0)
        self.assertEqual(default_transitions, [])

        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, _fake_analyze = _run_fake_optimizer(
                Path(tmp) / "allowed",
                validation_statuses={"B0000": "bounded_same"},
                candidate_ids=["B0000"],
                child_instruction_counts={"B0000": 1},
                allow_bounded_validation=True,
            )
            allowed_transitions = _read_csv(out_dir / "batch_state_transitions.csv")
            correctness = _read_csv(out_dir / "states" / "S0000" / "batch_correctness.csv")

        self.assertEqual(result["batch_transitions"], 1)
        self.assertEqual(allowed_transitions[0]["correctness_class"], "bounded_batch")
        self.assertEqual(correctness[0]["can_hard_fold"], "false")

    def test_duplicate_child_states_are_merged_by_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, fake_analyze = _run_fake_optimizer(
                Path(tmp),
                validation_statuses={"B0000": "all_permutations_same", "B0001": "all_permutations_same"},
                child_instruction_counts={"B0000": 1, "B0001": 1},
            )

            states = _read_csv(out_dir / "states.csv")
            dag = _read_csv(out_dir / "state_dag.csv")
            chosen = _read_csv(out_dir / "chosen_path.csv")

        self.assertEqual(result["duplicate_states"], 1)
        self.assertEqual(fake_analyze.call_count, 2)
        self.assertEqual(states[2]["is_duplicate"], "true")
        self.assertEqual(states[2]["duplicate_of"], "S0001")
        self.assertEqual(dag[1]["is_duplicate"], "true")
        self.assertEqual(dag[1]["duplicate_of"], "S0001")
        self.assertEqual(chosen[0]["child_state_id"], "S0002")
        self.assertEqual(chosen[0]["is_duplicate_transition"], "true")
        self.assertEqual(chosen[0]["duplicate_of"], "S0001")
        self.assertEqual(Path(chosen[0]["child_ir_path"]).name, "input.ll")
        self.assertIn(str(Path("states") / "S0001" / "input.ll"), chosen[0]["child_ir_path"])

    def test_structurally_equal_hash_different_states_are_not_merged(self) -> None:
        left = "define i32 @f(i32 %x) {\n  %a = add i32 %x, 0\n  ret i32 %a\n}\n"
        right = "define i32 @f(i32 %x) {\n  %b = add i32 %x, 0\n  ret i32 %b\n}\n"
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, fake_analyze = _run_fake_optimizer(
                Path(tmp),
                validation_statuses={"B0000": "all_permutations_same", "B0001": "all_permutations_same"},
                candidate_ids=["B0000", "B0001"],
                child_ir_texts={"B0000": left, "B0001": right},
            )

            states = _read_csv(out_dir / "states.csv")
            dag = _read_csv(out_dir / "state_dag.csv")

        self.assertEqual(result["duplicate_states"], 0)
        self.assertEqual(fake_analyze.call_count, 3)
        self.assertEqual([row["is_duplicate"] for row in states], ["false", "false", "false"])
        self.assertNotEqual(states[1]["state_hash"], states[2]["state_hash"])
        self.assertEqual([row["is_duplicate"] for row in dag], ["false", "false"])

    def test_budgeted_max_rounds_two_creates_depth_two_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, _fake_analyze = _run_fake_optimizer(
                Path(tmp),
                validation_statuses={("S0000", "B0000"): "all_permutations_same", ("S0001", "B0000"): "all_permutations_same"},
                candidate_ids_by_state={"S0000": ["B0000"], "S0001": ["B0000"], "S0002": []},
                child_instruction_counts={("S0000", "B0000"): 2, ("S0001", "B0000"): 1},
                max_rounds=2,
            )

            states = _read_csv(out_dir / "states.csv")
            chosen = _read_csv(out_dir / "chosen_path.csv")

        self.assertEqual(result["selected_final_state"], "S0002")
        self.assertIn("2", {row["depth"] for row in states})
        self.assertEqual([row["child_state_id"] for row in chosen], ["S0001", "S0002"])

    def test_budgeted_beam_width_limits_next_frontier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _result, out_dir, _fake_analyze = _run_fake_optimizer(
                Path(tmp),
                validation_statuses={
                    ("S0000", "B0000"): "all_permutations_same",
                    ("S0000", "B0001"): "all_permutations_same",
                    ("S0001", "B0000"): "all_permutations_same",
                    ("S0002", "B0000"): "all_permutations_same",
                },
                candidate_ids_by_state={"S0000": ["B0000", "B0001"], "S0001": ["B0000"], "S0002": ["B0000"]},
                child_instruction_counts={("S0000", "B0000"): 2, ("S0000", "B0001"): 4, ("S0001", "B0000"): 1, ("S0002", "B0000"): 1},
                max_rounds=2,
                beam_width=1,
            )

            transitions = _read_csv(out_dir / "batch_state_transitions.csv")
            parents = [row["parent_state_id"] for row in transitions]

        self.assertEqual(parents.count("S0000"), 2)
        self.assertEqual(parents.count("S0001"), 1)
        self.assertNotIn("S0002", parents[2:])

    def test_budgeted_max_batches_per_state_limits_executed_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, _fake_analyze = _run_fake_optimizer(
                Path(tmp),
                validation_statuses={"B0000": "all_permutations_same", "B0001": "all_permutations_same", "B0002": "all_permutations_same"},
                candidate_ids=["B0000", "B0001", "B0002"],
                child_instruction_counts={"B0000": 1, "B0001": 2, "B0002": 4},
                max_batches_per_state=2,
            )

            transitions = _read_csv(out_dir / "batch_state_transitions.csv")

        self.assertEqual(result["batch_transitions"], 2)
        self.assertEqual([row["batch_id"] for row in transitions], ["B0000", "B0001"])

    def test_budgeted_max_states_marks_budget_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, _fake_analyze = _run_fake_optimizer(
                Path(tmp),
                validation_statuses={"B0000": "all_permutations_same", "B0001": "all_permutations_same"},
                child_instruction_counts={"B0000": 2, "B0001": 1},
                max_states=2,
            )

            states = _read_csv(out_dir / "states.csv")
            summary = (out_dir / "optimize_summary.md").read_text(encoding="utf-8")
            events = _read_csv(out_dir / "optimizer_events.csv")

        self.assertEqual(len([row for row in states if row["is_duplicate"] != "true"]), 2)
        self.assertTrue(result["budget_exhausted"])
        self.assertIn("budget_exhausted: true", summary)
        self.assertIn("stop_reason: max_states_reached", summary)
        self.assertIn("budget_exhausted", {row["event_type"] for row in events})

    def test_budgeted_incumbent_can_be_from_earlier_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, _fake_analyze = _run_fake_optimizer(
                Path(tmp),
                validation_statuses={("S0000", "B0000"): "all_permutations_same", ("S0001", "B0000"): "all_permutations_same"},
                candidate_ids_by_state={"S0000": ["B0000"], "S0001": ["B0000"]},
                child_instruction_counts={("S0000", "B0000"): 1, ("S0001", "B0000"): 2},
                max_rounds=2,
            )

            chosen = _read_csv(out_dir / "chosen_path.csv")

        self.assertEqual(result["selected_final_state"], "S0001")
        self.assertEqual([row["child_state_id"] for row in chosen], ["S0001"])

    def test_budgeted_diverse_policy_keeps_different_batch_signatures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _result, out_dir, _fake_analyze = _run_fake_optimizer(
                Path(tmp),
                validation_statuses={"B0000": "all_permutations_same", "B0001": "all_permutations_same", "B0002": "all_permutations_same"},
                candidate_ids=["B0000", "B0001", "B0002"],
                batch_orders={("S0000", "B0000"): "pass-a", ("S0000", "B0001"): "pass-a", ("S0000", "B0002"): "pass-b"},
                child_instruction_counts={"B0000": 1, "B0001": 2, "B0002": 4},
                max_rounds=1,
                beam_width=2,
                batch_frontier_policy="diverse",
            )

            scores = _read_csv(out_dir / "frontier_scores.csv")
            selected = [row["last_batch_id"] for row in scores if row["round"] == "0" and row["selected_for_frontier"] == "true"]

        self.assertEqual(selected, ["B0000", "B0002"])

    def test_budgeted_defaults_to_score_policies_and_records_frontier_buckets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _result, out_dir, _fake_analyze = _run_fake_optimizer(
                Path(tmp),
                validation_statuses={"B0000": "all_permutations_same"},
                candidate_ids=["B0000"],
                child_instruction_counts={"B0000": 1},
                beam_width=1,
            )

            scores = _read_csv(out_dir / "frontier_scores.csv")
            events = _read_csv(out_dir / "optimizer_events.csv")
            summary = (out_dir / "optimize_summary.md").read_text(encoding="utf-8")

        self.assertTrue(scores)
        self.assertEqual(scores[0]["policy"], "score")
        self.assertEqual(scores[0]["selection_bucket"], "objective_bucket")
        self.assertIn("batch_selection_policy: score", summary)
        self.assertIn("frontier_selection_policy: score", summary)
        self.assertIn("apply_batch", {row["event_type"] for row in events})
        self.assertIn("select_frontier", {row["event_type"] for row in events})

    def test_budgeted_chosen_path_reconstructs_path_to_incumbent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, _fake_analyze = _run_fake_optimizer(
                Path(tmp),
                validation_statuses={("S0000", "B0000"): "all_permutations_same", ("S0001", "B0000"): "all_permutations_same"},
                candidate_ids_by_state={"S0000": ["B0000"], "S0001": ["B0000"]},
                batch_orders={("S0000", "B0000"): "pass-a", ("S0001", "B0000"): "pass-b"},
                child_instruction_counts={("S0000", "B0000"): 2, ("S0001", "B0000"): 1},
                max_rounds=2,
            )

            chosen = _read_csv(out_dir / "chosen_path.csv")
            pipeline = (out_dir / "optimized_pipeline.txt").read_text(encoding="utf-8").strip()

        self.assertEqual(result["selected_final_state"], "S0002")
        self.assertEqual([(row["parent_state_id"], row["child_state_id"]) for row in chosen], [("S0000", "S0001"), ("S0001", "S0002")])
        self.assertEqual(pipeline, "pass-a,pass-b")

    def test_chosen_path_artifacts_are_enriched_for_budgeted_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, _fake_analyze = _run_fake_optimizer(
                Path(tmp),
                validation_statuses={"B0000": "all_permutations_same", "B0001": "all_permutations_same"},
                child_instruction_counts={"B0000": 2, "B0001": 1},
            )

            chosen = _read_csv(out_dir / "chosen_path.csv")
            summary = _read_csv(out_dir / "chosen_path_summary.csv")
            optimized_batches = (out_dir / "optimized_batches.txt").read_text(encoding="utf-8")
            readable = (out_dir / "optimized_pipeline_readable.txt").read_text(encoding="utf-8")
            final_state = (out_dir / "final_state.txt").read_text(encoding="utf-8")
            path_artifacts = (out_dir / "path_artifacts.md").read_text(encoding="utf-8")

        required = {
            "step",
            "round",
            "parent_depth",
            "parent_state_hash",
            "batch_size",
            "can_hard_fold",
            "can_execute",
            "child_depth",
            "child_state_hash",
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
            "ir_inst_reduction_pct",
            "selection_reason",
        }
        self.assertEqual(result["selected_final_state"], "S0002")
        self.assertTrue(required.issubset(chosen[0].keys()))
        self.assertEqual(chosen[0]["round"], "0")
        self.assertEqual(chosen[0]["parent_depth"], "0")
        self.assertEqual(chosen[0]["child_depth"], "1")
        self.assertEqual(chosen[0]["batch_size"], "1")
        self.assertEqual(chosen[0]["can_hard_fold"], "true")
        self.assertEqual(chosen[0]["can_execute"], "true")
        self.assertEqual(chosen[0]["parent_active_passes"], "2")
        self.assertEqual(chosen[0]["child_active_passes"], "2")
        self.assertEqual(chosen[0]["parent_tested_pairs"], "1")
        self.assertEqual(chosen[0]["parent_commute_pairs"], "1")
        self.assertEqual(chosen[0]["ir_inst_reduction_pct"], "66.67")
        self.assertEqual(summary[0]["path_steps"], "1")
        self.assertEqual(summary[0]["total_pass_invocations"], "1")
        self.assertEqual(summary[0]["unique_pass_types"], "1")
        self.assertEqual(summary[0]["root_ir_inst_count"], "3")
        self.assertEqual(summary[0]["final_ir_inst_count"], "1")
        self.assertEqual(summary[0]["all_batches_certified"], "true")
        self.assertEqual(summary[0]["replay_verified"], "not_run")
        self.assertIn("can_hard_fold: true", optimized_batches)
        self.assertIn("objective_before: 3", optimized_batches)
        self.assertIn("# Step 0: B0001 from S0000 to S0002", readable)
        self.assertIn("pass-c", readable)
        self.assertIn("selected_final_state=S0002", final_state)
        self.assertIn("final_objective=1", final_state)
        self.assertIn("# Chosen Path", path_artifacts)
        self.assertIn(
            "Every hard-folded batch in the chosen path must be supported by batch correctness evidence. Objective values are reported only for path selection and evaluation; they are not used as commutation proof.",
            path_artifacts,
        )

    def test_optimized_pipeline_keeps_repeated_passes_from_chosen_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _result, out_dir, _fake_analyze = _run_fake_optimizer(
                Path(tmp),
                validation_statuses={("S0000", "B0000"): "all_permutations_same", ("S0001", "B0000"): "all_permutations_same"},
                candidate_ids_by_state={"S0000": ["B0000"], "S0001": ["B0000"]},
                batch_orders={("S0000", "B0000"): "pass-a;pass-a", ("S0001", "B0000"): "pass-a"},
                child_instruction_counts={("S0000", "B0000"): 2, ("S0001", "B0000"): 1},
                max_rounds=2,
            )

            chosen = _read_csv(out_dir / "chosen_path.csv")
            pipeline = (out_dir / "optimized_pipeline.txt").read_text(encoding="utf-8").strip()
            summary = _read_csv(out_dir / "chosen_path_summary.csv")

        self.assertEqual([row["canonical_order"] for row in chosen], ["pass-a;pass-a", "pass-a"])
        self.assertEqual(pipeline, "pass-a,pass-a,pass-a")
        self.assertEqual(summary[0]["total_pass_invocations"], "3")
        self.assertEqual(summary[0]["unique_pass_types"], "1")

    def test_batch_candidate_scores_generated_with_evidence_and_risk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _result, out_dir, _fake_analyze = _run_fake_optimizer(
                Path(tmp),
                validation_statuses={"B0000": "all_permutations_same", "B0001": "sampled_same"},
                candidate_ids=["B0000", "B0001"],
                child_instruction_counts={"B0000": 1, "B0001": 2},
                allow_sampled_batches=True,
                batch_frontier_policy="score",
            )

            rows = _read_csv(out_dir / "states" / "S0000" / "batch_candidate_scores.csv")
            by_id = {row["batch_id"]: row for row in rows}

        self.assertTrue(rows)
        self.assertEqual(by_id["B0000"]["evidence_score"], "1.0000")
        self.assertEqual(by_id["B0001"]["evidence_score"], "0.5000")
        self.assertGreater(float(by_id["B0001"]["risk_penalty"]), 0.0)
        self.assertIn("selected_for_execution", rows[0])

    def test_frontier_scores_include_detailed_scoring_columns_and_pareto(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _result, out_dir, _fake_analyze = _run_fake_optimizer(
                Path(tmp),
                validation_statuses={"B0000": "all_permutations_same", "B0001": "all_permutations_same"},
                batch_orders={("S0000", "B0000"): "pass-a", ("S0000", "B0001"): "pass-a"},
                child_instruction_counts={"B0000": 1, "B0001": 2},
                beam_width=1,
                batch_frontier_policy="score",
            )

            scores = _read_csv(out_dir / "frontier_scores.csv")

        required = {
            "root_inst_count",
            "parent_inst_count",
            "child_inst_count",
            "enable_count_from_parent",
            "effect_changed_count_from_parent",
            "parent_gain",
            "objective_score",
            "direct_call_score",
            "memory_score",
            "branch_score",
            "direct_calls",
            "memory_ops",
            "branches",
            "future_potential_score",
            "evidence_quality_score",
            "novelty_score",
            "cost_score",
            "risk_penalty",
            "final_state_score",
            "pareto_kept",
            "selection_bucket",
        }
        self.assertTrue(required.issubset(scores[0].keys()))
        self.assertIn("false", {row["pareto_kept"] for row in scores})

    def test_score_policy_preserves_objective_best_and_novelty_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _result, out_dir, _fake_analyze = _run_fake_optimizer(
                Path(tmp),
                validation_statuses={"B0000": "all_permutations_same", "B0001": "all_permutations_same", "B0002": "all_permutations_same"},
                candidate_ids=["B0000", "B0001", "B0002"],
                batch_orders={("S0000", "B0000"): "pass-a", ("S0000", "B0001"): "pass-a", ("S0000", "B0002"): "pass-b"},
                child_instruction_counts={"B0000": 1, "B0001": 2, "B0002": 4},
                beam_width=2,
                batch_frontier_policy="score",
            )

            selected = [
                row["last_batch_id"]
                for row in _read_csv(out_dir / "frontier_scores.csv")
                if row["selected_for_frontier"] == "true"
            ]

        self.assertIn("B0000", selected)
        self.assertIn("B0002", selected)

    def test_frontier_future_potential_uses_enable_and_effect_changed_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child_dir = root / "child"
            child_dir.mkdir()
            _write_csv(
                child_dir / "batch_summary.csv",
                [
                    "program",
                    "state_id",
                    "state_hash",
                    "active_passes",
                    "batch_candidates",
                    "conflict_components",
                    "unresolved_components",
                ],
                [
                    {
                        "program": "p",
                        "state_id": "S0001",
                        "state_hash": "h",
                        "active_passes": "2",
                        "batch_candidates": "1",
                        "conflict_components": "1",
                        "unresolved_components": "0",
                    }
                ],
            )
            _write_csv(
                child_dir / "batch_correctness.csv",
                ["batch_id", "correctness_class", "can_execute"],
                [{"batch_id": "B0000", "correctness_class": "certified_batch", "can_execute": "true"}],
            )
            _write_csv(
                child_dir / "coverage_summary.csv",
                [
                    "active_passes",
                    "certified_covered",
                    "heuristic_covered",
                    "unvalidated_covered",
                    "failed_or_unknown",
                    "dropped_active_passes",
                ],
                [
                    {
                        "active_passes": "2",
                        "certified_covered": "2",
                        "heuristic_covered": "0",
                        "unvalidated_covered": "0",
                        "failed_or_unknown": "0",
                        "dropped_active_passes": "0",
                    }
                ],
            )
            child_state = {"state_id": "S0001", "state_dir": str(child_dir), "active_passes": "2", "total_time_ms": "0"}
            parent_state = {"state_id": "S0000"}
            context = {"objective_by_state": {"S0000": 10, "S0001": 8}, "root_inst_count": 10, "configured_pass_count": 10}

            low = score_frontier_state(child_state, parent_state, {"parent_state_id": "S0000"}, context)
            high = score_frontier_state(
                child_state,
                parent_state,
                {"parent_state_id": "S0000", "enable_count_from_parent": "3", "effect_changed_count_from_parent": "2"},
                context,
            )

        self.assertGreater(float(high["future_potential_score"]), float(low["future_potential_score"]))

    def test_auto_mode_selects_budgeted_when_root_is_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, _fake_analyze = _run_fake_optimizer(
                Path(tmp),
                mode="auto",
                validation_statuses={"B0000": "all_permutations_same"},
                candidate_ids=["B0000"],
                unresolved_states={"S0000"},
            )

            summary = (out_dir / "optimize_summary.md").read_text(encoding="utf-8")

        self.assertEqual(result["selected_mode"], "budgeted")
        self.assertIn("selected_mode: budgeted", summary)
        self.assertIn("auto_reason:", summary)

    def test_auto_mode_selects_exact_for_small_clean_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, _fake_analyze = _run_fake_optimizer(
                Path(tmp),
                mode="auto",
                validation_statuses={"B0000": "all_permutations_same"},
                candidate_ids=["B0000"],
                child_instruction_counts={"B0000": 1},
            )

            summary = (out_dir / "optimize_summary.md").read_text(encoding="utf-8")

        self.assertEqual(result["selected_mode"], "exact")
        self.assertEqual(result["exact_status"], "exact_complete")
        self.assertIn("selected_mode: exact", summary)
        self.assertIn("auto_reason:", summary)

    def test_optimize_summary_contains_research_facing_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _result, out_dir, _fake_analyze = _run_fake_optimizer(
                Path(tmp),
                validation_statuses={"B0000": "all_permutations_same"},
                candidate_ids=["B0000"],
                child_instruction_counts={"B0000": 1},
                batch_frontier_policy="score",
            )

            summary = (out_dir / "optimize_summary.md").read_text(encoding="utf-8")

        self.assertIn("## Frontier Selection Policy", summary)
        self.assertIn("## Equality Tier Summary", summary)
        self.assertIn("| tier | count | hard_fold |", summary)
        self.assertIn("| structural_diff | 2 | 2 |", summary)
        self.assertIn("## Correctness Boundary", summary)
        self.assertIn(
            "Batch correctness is based on certified canonical-IR equality or explicit validation status. Objective scores are used only for search ranking and final path selection; they are not used as commutation or independence proof.",
            summary,
        )

    def test_exact_mode_expands_multiple_rounds_and_flattens_chosen_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, _fake_analyze = _run_fake_exact_optimizer(
                Path(tmp),
                candidate_ids_by_state={"S0000": ["B0000"], "S0001": ["B0000"], "S0002": []},
                validation_statuses={("S0000", "B0000"): "all_permutations_same", ("S0001", "B0000"): "all_permutations_same"},
                child_instruction_counts={("S0000", "B0000"): 2, ("S0001", "B0000"): 1},
                max_rounds=2,
            )

            states = _read_csv(out_dir / "states.csv")
            leaves = _read_csv(out_dir / "leaf_states.csv")
            chosen = _read_csv(out_dir / "chosen_path.csv")
            pipeline = (out_dir / "optimized_pipeline.txt").read_text(encoding="utf-8").strip()
            path_summary = _read_csv(out_dir / "chosen_path_summary.csv")
            path_artifacts = (out_dir / "path_artifacts.md").read_text(encoding="utf-8")
            exact_status = (out_dir / "exact_status.txt").read_text(encoding="utf-8")
            summary = (out_dir / "optimize_summary.md").read_text(encoding="utf-8")

        self.assertEqual(result["selected_final_state"], "S0002")
        self.assertEqual(result["exact_status"], "exact_complete")
        self.assertEqual([row["state_id"] for row in states], ["S0000", "S0001", "S0002"])
        self.assertEqual([row["child_state_id"] for row in chosen], ["S0001", "S0002"])
        self.assertEqual([row["step"] for row in chosen], ["0", "1"])
        self.assertEqual(pipeline, "root-a,child-a")
        self.assertEqual(chosen[0]["round"], "0")
        self.assertEqual(chosen[1]["round"], "1")
        self.assertEqual(path_summary[0]["path_steps"], "2")
        self.assertEqual(path_summary[0]["total_pass_invocations"], "2")
        self.assertIn("## Path Table", path_artifacts)
        self.assertIn("exact_complete", exact_status)
        self.assertIn("selected_mode: exact", summary)
        self.assertEqual({row["state_id"]: row["leaf_reason"] for row in leaves}["S0002"], "no_active_passes")

    def test_exact_mode_does_not_execute_sampled_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, fake_analyze = _run_fake_exact_optimizer(
                Path(tmp),
                candidate_ids_by_state={"S0000": ["B0000"]},
                validation_statuses={("S0000", "B0000"): "sampled_same"},
                child_instruction_counts={("S0000", "B0000"): 1},
                max_rounds=1,
            )

            transitions = _read_csv(out_dir / "batch_state_transitions.csv")
            leaves = _read_csv(out_dir / "leaf_states.csv")

        self.assertEqual(result["batch_transitions"], 0)
        self.assertEqual(transitions, [])
        self.assertEqual(fake_analyze.call_count, 1)
        self.assertEqual(leaves[0]["leaf_reason"], "exact_incomplete")
        self.assertEqual(result["exact_status"], "exact_incomplete")
        self.assertIn("non_certified_batch_validation:S0000", result["exact_incomplete_reasons"])

    def test_exact_mode_does_not_execute_bounded_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, fake_analyze = _run_fake_exact_optimizer(
                Path(tmp),
                candidate_ids_by_state={"S0000": ["B0000"]},
                validation_statuses={("S0000", "B0000"): "bounded_same"},
                child_instruction_counts={("S0000", "B0000"): 1},
                max_rounds=1,
            )

            transitions = _read_csv(out_dir / "batch_state_transitions.csv")
            leaves = _read_csv(out_dir / "leaf_states.csv")

        self.assertEqual(result["batch_transitions"], 0)
        self.assertEqual(transitions, [])
        self.assertEqual(fake_analyze.call_count, 1)
        self.assertEqual(leaves[0]["leaf_reason"], "exact_incomplete")
        self.assertEqual(result["exact_status"], "exact_incomplete")
        self.assertIn("non_certified_batch_validation:S0000", result["exact_incomplete_reasons"])

    def test_exact_mode_rejects_allow_sampled_batches_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Exact mode does not allow sampled batches"):
                _run_fake_exact_optimizer(
                    Path(tmp),
                    candidate_ids_by_state={"S0000": ["B0000"]},
                    validation_statuses={("S0000", "B0000"): "sampled_same"},
                    allow_sampled_batches=True,
                )

    def test_exact_mode_marks_incomplete_when_batch_candidates_are_truncated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, _fake_analyze = _run_fake_exact_optimizer(
                Path(tmp),
                candidate_ids_by_state={"S0000": ["B0000"]},
                validation_statuses={("S0000", "B0000"): "all_permutations_same"},
                truncated_states={"S0000"},
                max_rounds=1,
            )

            transitions = _read_csv(out_dir / "batch_state_transitions.csv")
            status_text = (out_dir / "exact_status.txt").read_text(encoding="utf-8")

        self.assertEqual(result["exact_status"], "exact_incomplete")
        self.assertIn("truncated_batch_candidates:S0000", result["exact_incomplete_reasons"])
        self.assertIn("truncated_batch_candidates:S0000", status_text)
        self.assertEqual(transitions, [])

    def test_exact_mode_marks_incomplete_when_unresolved_component_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, _fake_analyze = _run_fake_exact_optimizer(
                Path(tmp),
                candidate_ids_by_state={"S0000": ["B0000"]},
                validation_statuses={("S0000", "B0000"): "all_permutations_same"},
                unresolved_states={"S0000"},
                max_rounds=1,
            )

            status_text = (out_dir / "exact_status.txt").read_text(encoding="utf-8")

        self.assertEqual(result["exact_status"], "exact_incomplete")
        self.assertIn("unresolved_components:S0000", status_text)

    def test_exact_mode_marks_incomplete_when_pair_testing_hits_max_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, _fake_analyze = _run_fake_exact_optimizer(
                Path(tmp),
                candidate_ids_by_state={"S0000": ["B0000"]},
                validation_statuses={("S0000", "B0000"): "all_permutations_same"},
                child_instruction_counts={("S0000", "B0000"): 1},
                max_pairs_truncated_states={"S0000"},
                max_rounds=1,
            )

            status_text = (out_dir / "exact_status.txt").read_text(encoding="utf-8")
            summary = (out_dir / "optimize_summary.md").read_text(encoding="utf-8")

        self.assertEqual(result["exact_status"], "exact_incomplete")
        self.assertIn("pair_relation_incomplete_max_pairs:S0000", result["exact_incomplete_reasons"])
        self.assertIn("pair_relation_incomplete_max_pairs:S0000", status_text)
        self.assertIn("pair_relation_incomplete_max_pairs:S0000", summary)

    def test_exact_mode_marks_incomplete_when_lazy_pair_budget_skips_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, _fake_analyze = _run_fake_exact_optimizer(
                Path(tmp),
                candidate_ids_by_state={"S0000": ["B0000"]},
                validation_statuses={("S0000", "B0000"): "all_permutations_same"},
                child_instruction_counts={("S0000", "B0000"): 1},
                lazy_budget_states={"S0000"},
                pair_testing_mode="lazy",
                pair_test_budget_per_state=1,
                max_rounds=1,
            )

            status_text = (out_dir / "exact_status.txt").read_text(encoding="utf-8")
            transitions = _read_csv(out_dir / "batch_state_transitions.csv")

        self.assertEqual(result["exact_status"], "exact_incomplete")
        self.assertIn("pair_relation_incomplete_lazy_budget:S0000", result["exact_incomplete_reasons"])
        self.assertIn("pair_relation_incomplete_lazy_budget:S0000", status_text)
        self.assertEqual(transitions, [])

    def test_exact_mode_batchifies_terminal_frontier_without_applying_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, _fake_analyze = _run_fake_exact_optimizer(
                Path(tmp),
                candidate_ids_by_state={"S0000": ["B0000"], "S0001": ["B0000"]},
                validation_statuses={("S0000", "B0000"): "all_permutations_same", ("S0001", "B0000"): "all_permutations_same"},
                child_instruction_counts={("S0000", "B0000"): 1, ("S0001", "B0000"): 0},
                max_rounds=1,
            )

            terminal_dir = out_dir / "states" / "S0001"
            states = _read_csv(out_dir / "states.csv")
            transitions = _read_csv(out_dir / "batch_state_transitions.csv")
            terminal_summary = _read_csv(terminal_dir / "batch_summary.csv")
            terminal_correctness = _read_csv(terminal_dir / "batch_correctness.csv")
            terminal_coverage_exists = (terminal_dir / "coverage_summary.csv").exists()
            summary = (out_dir / "optimize_summary.md").read_text(encoding="utf-8")

        self.assertEqual(result["selected_final_state"], "S0001")
        self.assertEqual([row["state_id"] for row in states], ["S0000", "S0001"])
        self.assertEqual(len(transitions), 1)
        self.assertTrue(terminal_coverage_exists)
        self.assertEqual(terminal_summary[0]["batch_candidates"], "1")
        self.assertEqual(terminal_correctness[0]["can_execute"], "true")
        self.assertEqual(result["selected_final_state_truncated"], "true")
        self.assertEqual(result["remaining_active_passes"], "root-a;child-a")
        self.assertEqual(result["remaining_executable_batches"], "B0000")
        self.assertIn("- batchify_terminal_states: true", summary)
        self.assertIn("- selected_final_state_truncated: true", summary)
        self.assertIn("- remaining_active_passes: root-a;child-a", summary)
        self.assertIn("- remaining_executable_batches: B0000", summary)

    def test_exact_mode_marks_incomplete_when_state_cap_exceeded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, _fake_analyze = _run_fake_exact_optimizer(
                Path(tmp),
                candidate_ids_by_state={"S0000": ["B0000", "B0001"]},
                validation_statuses={("S0000", "B0000"): "all_permutations_same", ("S0000", "B0001"): "all_permutations_same"},
                child_instruction_counts={("S0000", "B0000"): 2, ("S0000", "B0001"): 1},
                max_rounds=1,
                max_states=2,
            )

            states = _read_csv(out_dir / "states.csv")
            status_text = (out_dir / "exact_status.txt").read_text(encoding="utf-8")

        self.assertEqual(result["exact_status"], "exact_incomplete")
        self.assertEqual([row["state_id"] for row in states], ["S0000", "S0001"])
        self.assertIn("state_cap_exceeded", status_text)

    def test_exact_mode_merges_duplicate_states_and_preserves_duplicate_incoming_edge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, fake_analyze = _run_fake_exact_optimizer(
                Path(tmp),
                candidate_ids_by_state={"S0000": ["B0000", "B0001"]},
                validation_statuses={("S0000", "B0000"): "all_permutations_same", ("S0000", "B0001"): "all_permutations_same"},
                child_instruction_counts={("S0000", "B0000"): 1, ("S0000", "B0001"): 1},
                max_rounds=1,
            )

            states = _read_csv(out_dir / "states.csv")
            dag = _read_csv(out_dir / "state_dag.csv")

        self.assertEqual(result["duplicate_states"], 0)
        self.assertEqual(result["duplicate_transitions"], 1)
        self.assertEqual(fake_analyze.call_count, 2)
        self.assertEqual([row["state_id"] for row in states], ["S0000", "S0001"])
        self.assertEqual(len(dag), 2)
        self.assertEqual(dag[1]["target_state_id"], "S0001")
        self.assertEqual(dag[1]["is_duplicate"], "true")
        self.assertEqual(dag[1]["duplicate_of"], "S0001")

    def test_optimize_batches_generates_final_summary_after_optional_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            optimized_dir = root / "optimized"
            with mock.patch(
                "phasebatch.baselines.compare_baselines",
                return_value={
                    "baseline_results_csv": str(optimized_dir / "baseline_results.csv"),
                    "random_baseline_trials_csv": str(optimized_dir / "random_baseline_trials.csv"),
                    "baselines_dir": str(optimized_dir / "baselines"),
                },
            ) as fake_baselines, mock.patch(
                "phasebatch.final_summary.generate_final_summary",
                return_value=optimized_dir / "final_summary.md",
            ) as fake_summary:
                result, out_dir, _fake_analyze = _run_fake_optimizer(
                    root,
                    validation_statuses={"B0000": "all_permutations_same"},
                    candidate_ids=["B0000"],
                    child_instruction_counts={"B0000": 1},
                    run_baselines=True,
                )

        fake_baselines.assert_called_once()
        fake_summary.assert_called_once_with(out_dir)
        self.assertEqual(result["final_summary"], str(out_dir / "final_summary.md"))

    def test_verify_final_pipeline_updates_chosen_path_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            optimized_dir = root / "optimized"
            with mock.patch(
                "phasebatch.pipeline_replay.replay_optimized_pipeline",
                return_value={
                    "replay_status": "success",
                    "hashes_match": "true",
                    "pipeline_replay_csv": str(optimized_dir / "pipeline_replay.csv"),
                },
            ) as fake_replay:
                _result, out_dir, _fake_analyze = _run_fake_optimizer(
                    root,
                    validation_statuses={"B0000": "all_permutations_same"},
                    candidate_ids=["B0000"],
                    child_instruction_counts={"B0000": 1},
                    verify_final_pipeline=True,
                )

            summary = _read_csv(out_dir / "chosen_path_summary.csv")
            optimize_summary = (out_dir / "optimize_summary.md").read_text(encoding="utf-8")

        fake_replay.assert_called_once_with(out_dir, timeout=1)
        self.assertEqual(summary[0]["replay_verified"], "true")
        self.assertIn("## Final Pipeline Replay Verification", optimize_summary)
        self.assertIn("replay_status: success", optimize_summary)

    def test_optimize_batches_writes_equality_tier_summary_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, _fake_analyze = _run_fake_optimizer(
                Path(tmp),
                validation_statuses={"B0000": "all_permutations_same"},
                candidate_ids=["B0000"],
                child_instruction_counts={"B0000": 1},
                verify_final_pipeline=False,
            )

            csv_path = out_dir / "equality_tier_summary.csv"
            md_path = out_dir / "equality_tier_summary.md"
            rows = _read_csv(csv_path)
            markdown = md_path.read_text(encoding="utf-8")

        self.assertEqual(result["equality_tier_summary_csv"], str(csv_path))
        self.assertEqual(result["equality_tier_summary_md"], str(md_path))
        self.assertTrue(rows)
        self.assertIn("# Equality Tier Summary", markdown)

    def test_budgeted_lazy_pair_testing_records_pair_scheduling_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, _fake_analyze = _run_fake_optimizer(
                Path(tmp),
                validation_statuses={"B0000": "all_permutations_same"},
                candidate_ids=["B0000"],
                child_instruction_counts={"B0000": 1},
                pair_testing_mode="lazy",
                pair_test_budget_per_state=1,
            )

            rows = _read_csv(out_dir / "pair_scheduling_summary.csv")
            markdown = (out_dir / "pair_scheduling_summary.md").read_text(encoding="utf-8")

        self.assertEqual(result["pair_scheduling_summary_csv"], str(out_dir / "pair_scheduling_summary.csv"))
        self.assertTrue(rows)
        self.assertTrue(any(row["pair_testing_mode"] == "lazy" for row in rows))
        self.assertIn("Lazy pair testing can reduce cost", markdown)

    def test_optimize_batches_cleans_ir_artifacts_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, _fake_analyze = _run_fake_optimizer(
                Path(tmp),
                validation_statuses={"B0000": "all_permutations_same"},
                candidate_ids=["B0000"],
                child_instruction_counts={"B0000": 1},
                keep_ir_artifacts=None,
            )

            remaining_ll = list(out_dir.rglob("*.ll"))

        self.assertEqual(result["ir_artifacts_cleaned"], "true")
        self.assertGreater(int(result["deleted_ir_artifacts"]), 0)
        self.assertEqual(remaining_ll, [])

    def test_optimize_batches_can_keep_ir_artifacts_for_postprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, out_dir, fake_analyze = _run_fake_optimizer(
                Path(tmp),
                validation_statuses={"B0000": "all_permutations_same"},
                candidate_ids=["B0000"],
                child_instruction_counts={"B0000": 1},
                keep_ir_artifacts=True,
            )

            remaining_ll = list(out_dir.rglob("*.ll"))

        self.assertEqual(result["ir_artifacts_cleaned"], "false")
        self.assertTrue(remaining_ll)
        self.assertTrue(fake_analyze.call_args_list[0].args[2]["_keep_ir_artifacts"])


def _run_fake_optimizer(
    root: Path,
    *,
    validation_statuses: dict,
    mode: str = "budgeted",
    child_instruction_counts: dict | None = None,
    child_ir_texts: dict | None = None,
    candidate_ids: list[str] | None = None,
    candidate_ids_by_state: dict[str, list[str]] | None = None,
    batch_orders: dict | None = None,
    unresolved_states: set[str] | None = None,
    truncated_states: set[str] | None = None,
    allow_sampled_batches: bool = False,
    allow_bounded_validation: bool = False,
    max_rounds: int = 1,
    beam_width: int = 8,
    max_states: int = 5000,
    max_batches_per_state: int = 20,
    budgeted_validation_strategy: str = "all",
    max_component_size: int = 10,
    max_batch_candidates: int = 200,
    batch_frontier_policy: str | None = None,
    batch_selection_policy: str | None = None,
    frontier_selection_policy: str | None = None,
    selection_seed: int = 0,
    run_baselines: bool = False,
    verify_final_pipeline: bool = False,
    keep_ir_artifacts: bool | None = True,
    pair_testing_mode: str = "full",
    pair_test_budget_per_state: int = 0,
    pair_priority_policy: str = "mixed",
    return_build_mock: bool = False,
    return_validation_attempts: bool = False,
):
    root.mkdir(parents=True, exist_ok=True)
    input_path = root / "input.ll"
    input_path.write_text(_ir_with_instruction_count(3), encoding="utf-8")
    passes_path = root / "passes.yaml"
    passes_path.write_text("passes:\n  - pass-a\n  - pass-b\n  - pass-c\n", encoding="utf-8")
    out_dir = root / "optimized"
    prepared_ir = out_dir / "input.ll"
    candidate_ids = candidate_ids or ["B0000", "B0001"]
    candidate_ids_by_state = candidate_ids_by_state or {"S0000": candidate_ids}
    child_instruction_counts = child_instruction_counts or {}
    child_ir_texts = child_ir_texts or {}
    batch_orders = batch_orders or {}
    unresolved_states = unresolved_states or set()
    truncated_states = truncated_states or set()
    validation_attempts: list[str] = []

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
        **_pair_scheduling_kwargs,
    ):
        state_dir = Path(state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
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
                "inst_delta",
                "blocks_changed",
                "changed_functions",
            ],
            [
                {
                    "program": program,
                    "state_id": state_id,
                    "depth": str(depth),
                    "parent_state_id": parent_state_id,
                    "transition_pass": transition_pass,
                    "state_hash": f"{state_id}-hash",
                    "pass": "pass-a",
                    "success": "true",
                    "active": "true",
                    "inst_delta": "-1",
                    "blocks_changed": "1",
                    "changed_functions": "f",
                },
                {
                    "program": program,
                    "state_id": state_id,
                    "depth": str(depth),
                    "parent_state_id": parent_state_id,
                    "transition_pass": transition_pass,
                    "state_hash": f"{state_id}-hash",
                    "pass": "pass-b",
                    "success": "true",
                    "active": "true",
                    "inst_delta": "-1",
                    "blocks_changed": "1",
                    "changed_functions": "f",
                },
            ],
        )
        _write_csv(
            state_dir / "pair_relation.csv",
            [
                "program",
                "state_id",
                "pass_a",
                "pass_b",
                "dynamic_relation",
                "final_relation",
                "failure_kind",
                "equality_tier",
                "can_hard_fold",
                "pair_testing_mode",
                "skipped_by_budget",
                "cache_hit",
            ],
            [
                {
                    "program": program,
                    "state_id": state_id,
                    "pass_a": "pass-a",
                    "pass_b": "pass-b",
                    "dynamic_relation": "not_tested" if pair_testing_mode == "lazy" else "dynamic_commute",
                    "final_relation": "final_unknown" if pair_testing_mode == "lazy" else "final_commute",
                    "failure_kind": "lazy_budget" if pair_testing_mode == "lazy" else "",
                    "equality_tier": "failed" if pair_testing_mode == "lazy" else "structural_diff",
                    "can_hard_fold": "false" if pair_testing_mode == "lazy" else "true",
                    "pair_testing_mode": pair_testing_mode,
                    "skipped_by_budget": "true" if pair_testing_mode == "lazy" else "false",
                    "cache_hit": "false",
                }
            ],
        )
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
                    "program": program,
                    "state_id": state_id,
                    "state_hash": f"{state_id}-hash",
                    "active_passes": "2",
                    "dormant_passes": "0",
                    "pairs_tested": "1",
                    "dynamic_commute": "1",
                    "order_sensitive": "0",
                    "unknown": "0",
                    "max_conflict_component": "1",
                    "total_time_ms": "1",
                }
            ],
        )
        return {"program": program, "state_id": state_id, "summary_path": str(state_dir / "summary.md")}

    def fake_build_batch_family(state_dir, max_component_size=10, max_batch_candidates=200):
        state_dir = Path(state_dir)
        state_id = state_dir.name
        state_candidate_ids = candidate_ids_by_state.get(state_id, [])
        rows = []
        for batch_id in state_candidate_ids:
            order = batch_orders.get((state_id, batch_id), batch_orders.get(batch_id))
            if order:
                pass
            elif batch_id == "B0000":
                order = "pass-a;pass-b"
            elif batch_id == "B0001":
                order = "pass-c"
            else:
                order = "pass-a"
            rows.append(
                {
                    "program": out_dir.name,
                    "state_id": state_id,
                    "state_hash": f"{state_id}-hash",
                    "batch_id": batch_id,
                    "batch_passes": order,
                    "batch_size": str(len(order.split(";"))),
                    "component_choices": "",
                    "is_exact": "false" if state_id in unresolved_states else "true",
                    "num_conflict_components": "1",
                    "unresolved_components": "1" if state_id in unresolved_states else "0",
                    "canonical_order": order,
                }
            )
        _write_csv(
            state_dir / "batch_candidates.csv",
            [
                "program",
                "state_id",
                "state_hash",
                "batch_id",
                "batch_passes",
                "batch_size",
                "component_choices",
                "is_exact",
                "num_conflict_components",
                "unresolved_components",
                "canonical_order",
            ],
            rows,
        )
        _write_csv(
            state_dir / "batch_summary.csv",
            [
                "program",
                "state_id",
                "state_hash",
                "active_passes",
                "active_pairs",
                "commute_pairs",
                "conflict_pairs",
                "conflict_components",
                "max_component_size",
                "batch_candidates",
                "exact_components",
                "unresolved_components",
                "naive_orderings_estimate",
                "batch_reduction_estimate",
                "truncated",
                "max_batch_candidates",
            ],
            [
                {
                    "program": out_dir.name,
                    "state_id": state_id,
                    "state_hash": f"{state_id}-hash",
                    "active_passes": "2",
                    "active_pairs": "1",
                    "commute_pairs": "1",
                    "conflict_pairs": "0",
                    "conflict_components": "1",
                    "max_component_size": "10",
                    "batch_candidates": str(len(rows)),
                    "exact_components": "0" if state_id in unresolved_states else "1",
                    "unresolved_components": "1" if state_id in unresolved_states else "0",
                    "naive_orderings_estimate": "2",
                    "batch_reduction_estimate": "1.00",
                    "truncated": "true" if state_id in truncated_states else "false",
                    "max_batch_candidates": str(max_batch_candidates),
                }
            ],
        )
        return {"batch_candidates": len(rows), "truncated": state_id in truncated_states}

    def fake_validate_batch_candidates(state_dir, tools, timeout, jobs, candidate_ids=None, runtime=None):
        del runtime
        rows = []
        state_dir = Path(state_dir)
        selected_ids = None if candidate_ids is None else set(candidate_ids)
        existing_by_id = {
            row.get("batch_id", ""): row
            for row in _read_csv(state_dir / "batch_validation.csv")
            if row.get("batch_id")
        } if (state_dir / "batch_validation.csv").exists() else {}
        for candidate in _read_csv(state_dir / "batch_candidates.csv"):
            key = (candidate.get("state_id", ""), candidate.get("batch_id", ""))
            batch_id = candidate.get("batch_id", "")
            existing = existing_by_id.get(batch_id)
            if existing and not (
                existing.get("validation_status") == "not_validated"
                and existing.get("validation_incomplete_reason") == "budgeted_on_demand_not_selected"
            ):
                rows.append(existing)
                continue
            if selected_ids is not None and batch_id not in selected_ids:
                status = "not_validated"
                incomplete_reason = "budgeted_on_demand_not_selected"
            else:
                validation_attempts.append(batch_id)
                status = validation_statuses.get(key, validation_statuses.get(batch_id, "not_validated"))
                incomplete_reason = ""
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
                    "validation_status": status,
                    "validation_tier": "exhaustive_all_permutations" if status == "all_permutations_same" else "unvalidated",
                    "validation_complete": "true" if status == "all_permutations_same" else "false",
                    "validation_hard_certificate": "true" if status == "all_permutations_same" else "false",
                    "validation_incomplete_reason": incomplete_reason,
                    "canonical_hash": "hash",
                    "first_mismatch_order": "",
                    "first_mismatch_hash": "",
                    "time_ms": "1",
                }
            )
        _write_csv(state_dir / "batch_validation.csv", BATCH_VALIDATION_FIELDS, rows)
        return {"validated_batches": len(rows)}

    def fake_run_opt(opt, src, passes, out, timeout):
        out = Path(out)
        parent_state = out.parents[2].name
        batch_id = out.stem.split("_")[-1]
        text = child_ir_texts.get((parent_state, batch_id), child_ir_texts.get(batch_id))
        if text is None:
            count = child_instruction_counts.get((parent_state, batch_id), child_instruction_counts.get(batch_id, 2))
            text = _ir_with_instruction_count(count)
        out.write_text(text, encoding="utf-8")
        return RunResult([opt], 0, "", "", 1.0)

    with mock.patch("phasebatch.optimizer.collect_toolchain", return_value={"tools": {"clang": {"path": "clang"}, "opt": {"path": "opt"}}}), \
        mock.patch("phasebatch.optimizer.prepare_input_ir", side_effect=fake_prepare), \
        mock.patch("phasebatch.optimizer.validate_passes", return_value=(["pass-a", "pass-b", "pass-c"], [])), \
        mock.patch("phasebatch.optimizer.analyze_state", side_effect=fake_analyze_state) as fake_analyze, \
        mock.patch("phasebatch.optimizer.build_batch_family", side_effect=fake_build_batch_family) as fake_build, \
        mock.patch("phasebatch.optimizer.validate_batch_candidates", side_effect=fake_validate_batch_candidates), \
        mock.patch("phasebatch.optimizer.run_opt", side_effect=fake_run_opt):
        optimize_kwargs = {
            "mode": mode,
            "objective": "ir-inst-count",
            "max_rounds": max_rounds,
            "max_batches_per_state": max_batches_per_state,
            "budgeted_validation_strategy": budgeted_validation_strategy,
            "max_component_size": max_component_size,
            "max_batch_candidates": max_batch_candidates,
            "beam_width": beam_width,
            "max_states": max_states,
            "batch_frontier_policy": batch_frontier_policy,
            "batch_selection_policy": batch_selection_policy,
            "frontier_selection_policy": frontier_selection_policy,
            "selection_seed": selection_seed,
            "validate_batches": True,
            "allow_sampled_batches": allow_sampled_batches,
            "allow_bounded_validation": allow_bounded_validation,
            "jobs": 1,
            "timeout": 1,
            "max_pairs": 10,
            "run_baselines": run_baselines,
            "verify_final_pipeline": verify_final_pipeline,
            "pair_testing_mode": pair_testing_mode,
            "pair_test_budget_per_state": pair_test_budget_per_state,
            "pair_priority_policy": pair_priority_policy,
        }
        if keep_ir_artifacts is not None:
            optimize_kwargs["keep_ir_artifacts"] = keep_ir_artifacts
        result = optimize_batches(input_path, out_dir, passes_path, **optimize_kwargs)

    if return_build_mock:
        return result, out_dir, fake_analyze, fake_build
    if return_validation_attempts:
        return result, out_dir, fake_analyze, validation_attempts
    return result, out_dir, fake_analyze


def _run_fake_exact_optimizer(
    root: Path,
    *,
    candidate_ids_by_state: dict[str, list[str]],
    validation_statuses: dict[tuple[str, str], str],
    child_instruction_counts: dict[tuple[str, str], int] | None = None,
    truncated_states: set[str] | None = None,
    unresolved_states: set[str] | None = None,
    max_pairs_truncated_states: set[str] | None = None,
    lazy_budget_states: set[str] | None = None,
    allow_sampled_batches: bool = False,
    allow_bounded_validation: bool = False,
    pair_testing_mode: str = "full",
    pair_test_budget_per_state: int = 0,
    pair_priority_policy: str = "mixed",
    batchify_terminal_states: bool = True,
    max_rounds: int = 1,
    max_states: int = 5000,
):
    root.mkdir(parents=True, exist_ok=True)
    input_path = root / "input.ll"
    input_path.write_text(_ir_with_instruction_count(3), encoding="utf-8")
    passes_path = root / "passes.yaml"
    passes_path.write_text("passes:\n  - root-a\n  - child-a\n  - alt-a\n", encoding="utf-8")
    out_dir = root / "optimized"
    prepared_ir = out_dir / "input.ll"
    child_instruction_counts = child_instruction_counts or {}
    truncated_states = truncated_states or set()
    unresolved_states = unresolved_states or set()
    max_pairs_truncated_states = max_pairs_truncated_states or set()
    lazy_budget_states = lazy_budget_states or set()

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
        **_pair_scheduling_kwargs,
    ):
        state_dir = Path(state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        active = "0" if candidate_ids_by_state.get(state_id) == [] else "2"
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
                "inst_delta",
                "blocks_changed",
                "changed_functions",
            ],
            [
                {
                    "program": program,
                    "state_id": state_id,
                    "depth": str(depth),
                    "parent_state_id": parent_state_id,
                    "transition_pass": transition_pass,
                    "state_hash": f"{state_id}-hash",
                    "pass": "root-a",
                    "success": "true",
                    "active": "true" if active != "0" else "false",
                    "inst_delta": "-1",
                    "blocks_changed": "1",
                    "changed_functions": "f",
                },
                {
                    "program": program,
                    "state_id": state_id,
                    "depth": str(depth),
                    "parent_state_id": parent_state_id,
                    "transition_pass": transition_pass,
                    "state_hash": f"{state_id}-hash",
                    "pass": "child-a",
                    "success": "true",
                    "active": "true" if active != "0" else "false",
                    "inst_delta": "-1",
                    "blocks_changed": "1",
                    "changed_functions": "f",
                },
            ],
        )
        lazy_budget = state_id in lazy_budget_states
        max_pairs_truncated = state_id in max_pairs_truncated_states
        _write_csv(
            state_dir / "pair_relation.csv",
            [
                "program",
                "state_id",
                "state_hash",
                "pass_a",
                "pass_b",
                "dynamic_relation",
                "failure_kind",
                "final_relation",
                "equality_tier",
                "can_hard_fold",
                "pair_testing_mode",
                "skipped_by_budget",
            ],
            [
                {
                    "program": program,
                    "state_id": state_id,
                    "state_hash": f"{state_id}-hash",
                    "pass_a": "root-a",
                    "pass_b": "child-a",
                    "dynamic_relation": "not_tested" if (max_pairs_truncated or lazy_budget) else "dynamic_commute",
                    "failure_kind": "lazy_budget" if lazy_budget else ("max_pairs" if max_pairs_truncated else ""),
                    "final_relation": "final_unknown" if (max_pairs_truncated or lazy_budget) else "final_commute",
                    "equality_tier": "failed" if (max_pairs_truncated or lazy_budget) else "canonical_hash",
                    "can_hard_fold": "false" if (max_pairs_truncated or lazy_budget) else "true",
                    "pair_testing_mode": "lazy" if lazy_budget else pair_testing_mode,
                    "skipped_by_budget": "true" if lazy_budget else "false",
                }
            ],
        )
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
                    "program": program,
                    "state_id": state_id,
                    "state_hash": f"{state_id}-hash",
                    "active_passes": active,
                    "dormant_passes": "0",
                    "pairs_tested": "1" if active != "0" else "0",
                    "dynamic_commute": "1" if active != "0" else "0",
                    "order_sensitive": "0",
                    "unknown": "0",
                    "max_conflict_component": "1" if active != "0" else "0",
                    "total_time_ms": "1",
                }
            ],
        )
        return {"program": program, "state_id": state_id, "summary_path": str(state_dir / "summary.md")}

    def fake_build_batch_family(state_dir, max_component_size=10, max_batch_candidates=200):
        state_dir = Path(state_dir)
        state_id = state_dir.name
        candidate_ids = candidate_ids_by_state.get(state_id, [])
        rows = []
        for batch_id in candidate_ids:
            order = {
                ("S0000", "B0000"): "root-a",
                ("S0000", "B0001"): "alt-a",
                ("S0001", "B0000"): "child-a",
            }.get((state_id, batch_id), "root-a")
            rows.append(
                {
                    "program": out_dir.name,
                    "state_id": state_id,
                    "state_hash": f"{state_id}-hash",
                    "batch_id": batch_id,
                    "batch_passes": order,
                    "batch_size": str(len(order.split(";"))),
                    "component_choices": "",
                    "is_exact": "false" if state_id in unresolved_states else "true",
                    "num_conflict_components": "1" if candidate_ids else "0",
                    "unresolved_components": "1" if state_id in unresolved_states else "0",
                    "canonical_order": order,
                }
            )
        _write_csv(
            state_dir / "batch_candidates.csv",
            [
                "program",
                "state_id",
                "state_hash",
                "batch_id",
                "batch_passes",
                "batch_size",
                "component_choices",
                "is_exact",
                "num_conflict_components",
                "unresolved_components",
                "canonical_order",
            ],
            rows,
        )
        _write_csv(
            state_dir / "batch_components.csv",
            [
                "program",
                "state_id",
                "state_hash",
                "component_id",
                "component_size",
                "component_passes",
                "conflict_edges",
                "commute_edges",
                "is_exact",
                "num_local_alternatives",
                "unresolved_reason",
            ],
            [
                {
                    "program": out_dir.name,
                    "state_id": state_id,
                    "state_hash": f"{state_id}-hash",
                    "component_id": "C0000",
                    "component_size": "2",
                    "component_passes": "root-a;child-a",
                    "conflict_edges": "",
                    "commute_edges": "root-a--child-a",
                    "is_exact": "false" if state_id in unresolved_states else "true",
                    "num_local_alternatives": "1",
                    "unresolved_reason": "too_large" if state_id in unresolved_states else "",
                }
            ] if candidate_ids else [],
        )
        _write_csv(
            state_dir / "batch_summary.csv",
            [
                "program",
                "state_id",
                "state_hash",
                "active_passes",
                "active_pairs",
                "commute_pairs",
                "conflict_pairs",
                "conflict_components",
                "max_component_size",
                "batch_candidates",
                "exact_components",
                "unresolved_components",
                "naive_orderings_estimate",
                "batch_reduction_estimate",
                "truncated",
                "max_batch_candidates",
            ],
            [
                {
                    "program": out_dir.name,
                    "state_id": state_id,
                    "state_hash": f"{state_id}-hash",
                    "active_passes": "0" if not candidate_ids else "2",
                    "active_pairs": "0" if not candidate_ids else "1",
                    "commute_pairs": "0" if not candidate_ids else "1",
                    "conflict_pairs": "0",
                    "conflict_components": "0" if not candidate_ids else "1",
                    "max_component_size": "0" if not candidate_ids else "2",
                    "batch_candidates": str(len(rows)),
                    "exact_components": "0" if state_id in unresolved_states else ("1" if candidate_ids else "0"),
                    "unresolved_components": "1" if state_id in unresolved_states else "0",
                    "naive_orderings_estimate": "2",
                    "batch_reduction_estimate": "1.00",
                    "truncated": "true" if state_id in truncated_states else "false",
                    "max_batch_candidates": str(max_batch_candidates),
                }
            ],
        )
        (state_dir / "batch_summary.md").write_text("# Batch Summary\n", encoding="utf-8")
        return {"batch_candidates": len(rows), "truncated": state_id in truncated_states}

    def fake_validate_batch_candidates(state_dir, tools, timeout, jobs):
        state_dir = Path(state_dir)
        rows = []
        for candidate in _read_csv(state_dir / "batch_candidates.csv"):
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
                    "validation_status": validation_statuses.get((candidate.get("state_id", ""), candidate.get("batch_id", "")), "not_validated"),
                    "canonical_hash": "hash",
                    "first_mismatch_order": "",
                    "first_mismatch_hash": "",
                    "time_ms": "1",
                }
            )
        _write_csv(state_dir / "batch_validation.csv", BATCH_VALIDATION_FIELDS, rows)
        return {"validated_batches": len(rows)}

    def fake_run_opt(opt, src, passes, out, timeout):
        out = Path(out)
        parent_state = out.parents[2].name
        batch_id = out.stem.split("_")[-1]
        count = child_instruction_counts.get((parent_state, batch_id), 2)
        out.write_text(_ir_with_instruction_count(count), encoding="utf-8")
        return RunResult([opt], 0, "", "", 1.0)

    with mock.patch("phasebatch.optimizer.collect_toolchain", return_value={"tools": {"clang": {"path": "clang"}, "opt": {"path": "opt"}}}), \
        mock.patch("phasebatch.optimizer.prepare_input_ir", side_effect=fake_prepare), \
        mock.patch("phasebatch.optimizer.validate_passes", return_value=(["root-a", "child-a", "alt-a"], [])), \
        mock.patch("phasebatch.optimizer.analyze_state", side_effect=fake_analyze_state) as fake_analyze, \
        mock.patch("phasebatch.optimizer.build_batch_family", side_effect=fake_build_batch_family), \
        mock.patch("phasebatch.optimizer.validate_batch_candidates", side_effect=fake_validate_batch_candidates), \
        mock.patch("phasebatch.optimizer.run_opt", side_effect=fake_run_opt):
        result = optimize_batches(
            input_path,
            out_dir,
            passes_path,
            mode="exact",
            objective="ir-inst-count",
            max_rounds=max_rounds,
            max_batches_per_state=20,
            validate_batches=True,
            allow_sampled_batches=allow_sampled_batches,
            allow_bounded_validation=allow_bounded_validation,
            pair_testing_mode=pair_testing_mode,
            pair_test_budget_per_state=pair_test_budget_per_state,
            pair_priority_policy=pair_priority_policy,
            jobs=1,
            timeout=1,
            max_pairs=10,
            max_states=max_states,
            batchify_terminal_states=batchify_terminal_states,
            verify_final_pipeline=False,
        )

    return result, out_dir, fake_analyze


def _ir_with_instruction_count(count: int) -> str:
    lines = ["define i32 @f(i32 %x) {", "entry:"]
    for index in range(count - 1):
        source = "%x" if index == 0 else f"%v{index - 1}"
        lines.append(f"  %v{index} = add i32 {source}, 1")
    if count > 0:
        value = "%x" if count == 1 else f"%v{count - 2}"
        lines.append(f"  ret i32 {value}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
