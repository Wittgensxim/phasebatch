import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.optimizer import _flatten_pipeline, _flatten_pipeline_names, optimize_batches, score_frontier_state
from phasebatch.pass_config import PassRegistry, PassSpec
from phasebatch.schema import BATCH_VALIDATION_FIELDS, RunResult


class OptimizerTests(unittest.TestCase):
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
        self.assertIn("--beam-width", result.stdout)
        self.assertIn("--batch-frontier-policy", result.stdout)
        self.assertIn("--batch-selection-policy", result.stdout)
        self.assertIn("--frontier-selection-policy", result.stdout)
        self.assertIn("--selection-seed", result.stdout)
        self.assertIn("--allow-sampled-batches", result.stdout)
        self.assertIn("--exact-fail-on-incomplete", result.stdout)
        self.assertIn("--run-baselines", result.stdout)
        self.assertIn("--verify-final-pipeline", result.stdout)

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
                ]
            }

        self.assertEqual(result["selected_final_state"], "S0002")
        for name, exists in output_exists.items():
            self.assertTrue(exists, name)
        self.assertEqual([row["state_id"] for row in states], ["S0000", "S0001", "S0002"])
        self.assertEqual([row["batch_id"] for row in dag], ["B0000", "B0001"])
        self.assertEqual([row["selected_as_final"] for row in leaves], ["false", "false", "true"])
        self.assertEqual(chosen[0]["child_state_id"], "S0002")
        self.assertEqual(chosen[0]["ir_inst_delta"], "-2")
        self.assertEqual(pipeline, "pass-c")
        self.assertIn("Objective is used only for path selection, not as commutation proof.", summary)

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
        self.assertEqual(leaves[0]["leaf_reason"], "no_executable_batches")

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


def _run_fake_optimizer(
    root: Path,
    *,
    validation_statuses: dict,
    mode: str = "budgeted",
    child_instruction_counts: dict | None = None,
    candidate_ids: list[str] | None = None,
    candidate_ids_by_state: dict[str, list[str]] | None = None,
    batch_orders: dict | None = None,
    unresolved_states: set[str] | None = None,
    truncated_states: set[str] | None = None,
    allow_sampled_batches: bool = False,
    max_rounds: int = 1,
    beam_width: int = 8,
    max_states: int = 5000,
    max_batches_per_state: int = 20,
    batch_frontier_policy: str | None = None,
    batch_selection_policy: str | None = None,
    frontier_selection_policy: str | None = None,
    selection_seed: int = 0,
    run_baselines: bool = False,
    verify_final_pipeline: bool = False,
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
    batch_orders = batch_orders or {}
    unresolved_states = unresolved_states or set()
    truncated_states = truncated_states or set()

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
            ["program", "state_id", "pass_a", "pass_b", "final_relation"],
            [{"program": program, "state_id": state_id, "pass_a": "pass-a", "pass_b": "pass-b", "final_relation": "final_commute"}],
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

    def fake_validate_batch_candidates(state_dir, tools, timeout, jobs):
        rows = []
        state_dir = Path(state_dir)
        for candidate in _read_csv(state_dir / "batch_candidates.csv"):
            key = (candidate.get("state_id", ""), candidate.get("batch_id", ""))
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
                    "validation_status": validation_statuses.get(key, validation_statuses.get(candidate.get("batch_id", ""), "not_validated")),
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
        count = child_instruction_counts.get((parent_state, batch_id), child_instruction_counts.get(batch_id, 2))
        out.write_text(_ir_with_instruction_count(count), encoding="utf-8")
        return RunResult([opt], 0, "", "", 1.0)

    with mock.patch("phasebatch.optimizer.collect_toolchain", return_value={"tools": {"clang": {"path": "clang"}, "opt": {"path": "opt"}}}), \
        mock.patch("phasebatch.optimizer.prepare_input_ir", side_effect=fake_prepare), \
        mock.patch("phasebatch.optimizer.validate_passes", return_value=(["pass-a", "pass-b", "pass-c"], [])), \
        mock.patch("phasebatch.optimizer.analyze_state", side_effect=fake_analyze_state) as fake_analyze, \
        mock.patch("phasebatch.optimizer.build_batch_family", side_effect=fake_build_batch_family), \
        mock.patch("phasebatch.optimizer.validate_batch_candidates", side_effect=fake_validate_batch_candidates), \
        mock.patch("phasebatch.optimizer.run_opt", side_effect=fake_run_opt):
        result = optimize_batches(
            input_path,
            out_dir,
            passes_path,
            mode=mode,
            objective="ir-inst-count",
            max_rounds=max_rounds,
            max_batches_per_state=max_batches_per_state,
            beam_width=beam_width,
            max_states=max_states,
            batch_frontier_policy=batch_frontier_policy,
            batch_selection_policy=batch_selection_policy,
            frontier_selection_policy=frontier_selection_policy,
            selection_seed=selection_seed,
            validate_batches=True,
            allow_sampled_batches=allow_sampled_batches,
            jobs=1,
            timeout=1,
            max_pairs=10,
            run_baselines=run_baselines,
            verify_final_pipeline=verify_final_pipeline,
        )

    return result, out_dir, fake_analyze


def _run_fake_exact_optimizer(
    root: Path,
    *,
    candidate_ids_by_state: dict[str, list[str]],
    validation_statuses: dict[tuple[str, str], str],
    child_instruction_counts: dict[tuple[str, str], int] | None = None,
    truncated_states: set[str] | None = None,
    unresolved_states: set[str] | None = None,
    allow_sampled_batches: bool = False,
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
        _write_csv(
            state_dir / "pair_relation.csv",
            ["program", "state_id", "state_hash", "pass_a", "pass_b", "final_relation"],
            [
                {
                    "program": program,
                    "state_id": state_id,
                    "state_hash": f"{state_id}-hash",
                    "pass_a": "root-a",
                    "pass_b": "child-a",
                    "final_relation": "final_commute",
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
            jobs=1,
            timeout=1,
            max_pairs=10,
            max_states=max_states,
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
