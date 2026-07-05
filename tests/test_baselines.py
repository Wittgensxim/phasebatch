import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.baselines import compare_baselines, run_greedy_single_pass_baseline, run_random_single_pass_baseline
from phasebatch.schema import RunResult


class BaselineComparisonTests(unittest.TestCase):
    def test_root_baseline_equals_root_ir_instruction_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, passes_path = _make_run(Path(tmp), root_count=4)

            with _fake_no_active_profiles():
                compare_baselines(run_dir, passes_path, methods=["root"], max_rounds=1, random_trials=1, seed=0, timeout=1, jobs=1)

            rows = _rows_by_method(run_dir / "baseline_results.csv")

        self.assertEqual(rows["root"]["status"], "success")
        self.assertEqual(rows["root"]["final_ir_inst_count"], "4")
        self.assertEqual(rows["root"]["root_ir_inst_count"], "4")
        self.assertEqual(rows["root"]["ir_inst_delta"], "0")
        self.assertEqual(rows["root"]["final_sequence_length"], "0")

    def test_default_o0_row_equals_root_ir_instruction_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, passes_path = _make_run(Path(tmp), root_count=4)

            compare_baselines(run_dir, passes_path, methods=["default"], max_rounds=1, random_trials=1, seed=0, timeout=1, jobs=1)

            rows = _rows_by_method(run_dir / "baseline_results.csv")
            output_exists = (run_dir / "baselines" / "default_O0" / "default_O0.ll").exists()

        self.assertTrue(output_exists)
        self.assertEqual(rows["default_O0"]["status"], "success")
        self.assertEqual(rows["default_O0"]["final_ir_inst_count"], "4")
        self.assertEqual(rows["default_O0"]["root_ir_inst_count"], "4")
        self.assertEqual(rows["default_O0"]["states_evaluated"], "1")
        self.assertEqual(rows["default_O0"]["opt_runs"], "0")
        self.assertEqual(rows["default_O0"]["final_sequence_length"], "0")

    def test_config_order_once_applies_valid_passes_in_config_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, passes_path = _make_run(
                Path(tmp),
                config_passes=["pass-b", "invalid-pass", "pass-a"],
                valid_passes=["pass-a", "pass-b"],
            )
            calls = []

            def fake_run_opt(opt, src, passes, out, timeout):
                calls.append(list(passes))
                Path(out).write_text(_ir_with_instruction_count(2), encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            with mock.patch("phasebatch.baselines.run_opt", side_effect=fake_run_opt), _fake_no_active_profiles():
                compare_baselines(run_dir, passes_path, max_rounds=1, random_trials=1, seed=0, timeout=1, jobs=1)

            rows = _rows_by_method(run_dir / "baseline_results.csv")

        self.assertIn(["pass-b", "pass-a"], calls)
        self.assertEqual(rows["config_order_once"]["pass_sequence"], "pass-b;pass-a")
        self.assertIn("skipped invalid passes: invalid-pass", rows["config_order_once"]["error_message"])

    def test_greedy_single_pass_stops_when_no_active_pass_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, passes_path = _make_run(Path(tmp), root_count=3, valid_passes=["pass-a", "pass-b"])

            with _fake_no_active_profiles():
                compare_baselines(run_dir, passes_path, max_rounds=3, random_trials=1, seed=0, timeout=1, jobs=1)

            rows = _rows_by_method(run_dir / "baseline_results.csv")

        self.assertEqual(rows["greedy_single_pass"]["status"], "success")
        self.assertEqual(rows["greedy_single_pass"]["pass_sequence"], "")
        self.assertEqual(rows["greedy_single_pass"]["states_evaluated"], "0")
        self.assertEqual(rows["greedy_single_pass"]["final_ir_inst_count"], "3")
        self.assertEqual(rows["greedy_single_pass"]["stop_reason"], "no_active_passes")

    def test_greedy_single_pass_selects_lower_instruction_count_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, passes_path = _make_run(Path(tmp), root_count=5, valid_passes=["pass-a", "pass-b"])

            def fake_run_opt(opt, src, passes, out, timeout):
                count = {"pass-a": 4, "pass-b": 2}[passes[0]]
                Path(out).write_text(_ir_with_instruction_count(count), encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            with mock.patch("phasebatch.baselines.run_opt", side_effect=fake_run_opt), _fake_no_active_profiles():
                compare_baselines(run_dir, passes_path, max_rounds=1, random_trials=1, seed=0, timeout=1, jobs=1)

            rows = _rows_by_method(run_dir / "baseline_results.csv")
            path_rows = _read_csv(run_dir / "baselines" / "greedy_single_pass" / "greedy_path.csv")

        self.assertEqual(rows["greedy_single_pass"]["pass_sequence"], "pass-b")
        self.assertEqual(rows["greedy_single_pass"]["final_ir_inst_count"], "2")
        self.assertEqual(path_rows[0]["selected_pass"], "pass-b")

    def test_run_greedy_single_pass_baseline_writes_path_summary_and_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _passes_path = _make_run(Path(tmp), root_count=5, valid_passes=["pass-a", "pass-b"])
            out_dir = run_dir / "baselines" / "greedy_single_pass"

            def fake_run_opt(opt, src, passes, out, timeout):
                count = {"pass-a": 3, "pass-b": 4}[passes[0]]
                Path(out).write_text(_ir_with_instruction_count(count), encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            with mock.patch("phasebatch.baselines.run_opt", side_effect=fake_run_opt):
                result = run_greedy_single_pass_baseline(
                    run_dir / "states" / "S0000" / "input.ll",
                    ["pass-a", "pass-b"],
                    {"opt": "opt"},
                    out_dir,
                    max_rounds=1,
                    timeout=1,
                )

            rows = _rows_by_method(run_dir / "baseline_results.csv")
            path_exists = (out_dir / "greedy_path.csv").exists()
            summary_exists = (out_dir / "greedy_summary.md").exists()
            final_exists = (out_dir / "greedy_final.ll").exists()

        self.assertEqual(result["pass_sequence"], "pass-a")
        self.assertTrue(path_exists)
        self.assertTrue(summary_exists)
        self.assertTrue(final_exists)
        self.assertEqual(rows["greedy_single_pass"]["method"], "greedy_single_pass")
        self.assertEqual(rows["greedy_single_pass"]["final_sequence_length"], "1")
        self.assertEqual(rows["greedy_single_pass"]["states_evaluated"], "1")
        self.assertEqual(rows["greedy_single_pass"]["opt_runs"], "2")

    def test_greedy_single_pass_stops_nonimproving_when_disallowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _passes_path = _make_run(Path(tmp), root_count=3, valid_passes=["pass-a"])
            out_dir = run_dir / "baselines" / "greedy_single_pass"

            def fake_run_opt(opt, src, passes, out, timeout):
                Path(out).write_text(_ir_with_instruction_count(3, opcode="mul"), encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            with mock.patch("phasebatch.baselines.run_opt", side_effect=fake_run_opt):
                result = run_greedy_single_pass_baseline(
                    run_dir / "states" / "S0000" / "input.ll",
                    ["pass-a"],
                    {"opt": "opt"},
                    out_dir,
                    max_rounds=2,
                    timeout=1,
                    allow_nonimproving=False,
                )

            path_rows = _read_csv(out_dir / "greedy_path.csv")

        self.assertEqual(result["pass_sequence"], "")
        self.assertEqual(result["stop_reason"], "no_improving_pass")
        self.assertEqual(result["states_evaluated"], "0")
        self.assertEqual(path_rows[0]["selected_pass"], "pass-a")
        self.assertEqual(path_rows[0]["stop_reason"], "no_improving_pass")

    def test_greedy_single_pass_can_continue_nonimproving_when_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _passes_path = _make_run(Path(tmp), root_count=3, valid_passes=["pass-a"])
            out_dir = run_dir / "baselines" / "greedy_single_pass"

            def fake_run_opt(opt, src, passes, out, timeout):
                Path(out).write_text(_ir_with_instruction_count(3, opcode="mul"), encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            with mock.patch("phasebatch.baselines.run_opt", side_effect=fake_run_opt):
                result = run_greedy_single_pass_baseline(
                    run_dir / "states" / "S0000" / "input.ll",
                    ["pass-a"],
                    {"opt": "opt"},
                    out_dir,
                    max_rounds=1,
                    timeout=1,
                    allow_nonimproving=True,
                )

        self.assertEqual(result["pass_sequence"], "pass-a")
        self.assertEqual(result["states_evaluated"], "1")
        self.assertEqual(result["stop_reason"], "max_rounds_reached")

    def test_random_single_pass_generates_requested_trials_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _passes_path = _make_run(Path(tmp), root_count=5, valid_passes=["pass-a", "pass-b"])
            out_dir = run_dir / "baselines" / "random_single_pass"

            def fake_run_opt(opt, src, passes, out, timeout):
                count = {"pass-a": 4, "pass-b": 2}[passes[0]]
                Path(out).write_text(_ir_with_instruction_count(count), encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            with mock.patch("phasebatch.baselines.run_opt", side_effect=fake_run_opt):
                result = run_random_single_pass_baseline(
                    run_dir / "states" / "S0000" / "input.ll",
                    ["pass-a", "pass-b"],
                    {"opt": "opt"},
                    out_dir,
                    max_rounds=1,
                    random_trials=4,
                    seed=0,
                    timeout=1,
                )

            trial_rows = _read_csv(out_dir / "random_trials.csv")
            baseline_rows = _rows_by_method(run_dir / "baseline_results.csv")
            best_path = _read_csv(out_dir / "random_best_path.csv")
            summary_exists = (out_dir / "random_summary.md").exists()
            final_exists = (out_dir / "random_best_final.ll").exists()

        self.assertEqual(len(trial_rows), 4)
        self.assertTrue(summary_exists)
        self.assertTrue(final_exists)
        self.assertEqual(result["method"], "random_single_pass_best")
        self.assertEqual(baseline_rows["random_single_pass_best"]["method"], "random_single_pass_best")
        self.assertEqual(result["states_evaluated"], "2")
        self.assertEqual(result["opt_runs"], "2")
        self.assertEqual(best_path[0]["trial"], result["best_trial"])

    def test_random_single_pass_same_seed_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, passes_path = _make_run(root / "first", valid_passes=["pass-a", "pass-b", "pass-c"])
            second, _ = _make_run(root / "second", valid_passes=["pass-a", "pass-b", "pass-c"])

            def fake_run_opt(opt, src, passes, out, timeout):
                count = {"pass-a": 7, "pass-b": 5, "pass-c": 3}[passes[0]]
                Path(out).write_text(_ir_with_instruction_count(count), encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            with mock.patch("phasebatch.baselines.run_opt", side_effect=fake_run_opt):
                compare_baselines(first, passes_path, max_rounds=2, random_trials=5, seed=7, timeout=1, jobs=1)
                compare_baselines(second, passes_path, max_rounds=2, random_trials=5, seed=7, timeout=1, jobs=1)

            first_trials = _read_csv(first / "baselines" / "random_single_pass" / "random_trials.csv")
            second_trials = _read_csv(second / "baselines" / "random_single_pass" / "random_trials.csv")
            first_best = _rows_by_method(first / "baseline_results.csv")["random_single_pass_best"]["pass_sequence"]
            second_best = _rows_by_method(second / "baseline_results.csv")["random_single_pass_best"]["pass_sequence"]

        self.assertEqual(
            [row["pass_sequence"] for row in first_trials],
            [row["pass_sequence"] for row in second_trials],
        )
        self.assertEqual(first_best, second_best)

    def test_random_single_pass_different_seed_can_change_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, passes_path = _make_run(root / "first", valid_passes=["pass-a", "pass-b", "pass-c"])
            second, _ = _make_run(root / "second", valid_passes=["pass-a", "pass-b", "pass-c"])

            def fake_run_opt(opt, src, passes, out, timeout):
                count = {"pass-a": 7, "pass-b": 5, "pass-c": 3}[passes[0]]
                Path(out).write_text(_ir_with_instruction_count(count), encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            with mock.patch("phasebatch.baselines.run_opt", side_effect=fake_run_opt):
                compare_baselines(first, passes_path, max_rounds=1, random_trials=5, seed=0, timeout=1, jobs=1)
                compare_baselines(second, passes_path, max_rounds=1, random_trials=5, seed=1, timeout=1, jobs=1)

            first_trials = _read_csv(first / "baselines" / "random_single_pass" / "random_trials.csv")
            second_trials = _read_csv(second / "baselines" / "random_single_pass" / "random_trials.csv")

        self.assertNotEqual(
            [row["pass_sequence"] for row in first_trials],
            [row["pass_sequence"] for row in second_trials],
        )

    def test_random_single_pass_best_trial_uses_instruction_count_then_length_then_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _passes_path = _make_run(Path(tmp), root_count=5, valid_passes=["pass-a", "pass-b"])
            out_dir = run_dir / "baselines" / "random_single_pass"

            def fake_run_opt(opt, src, passes, out, timeout):
                count = {"pass-a": 4, "pass-b": 2}[passes[0]]
                Path(out).write_text(_ir_with_instruction_count(count), encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            with mock.patch("phasebatch.baselines.run_opt", side_effect=fake_run_opt):
                result = run_random_single_pass_baseline(
                    run_dir / "states" / "S0000" / "input.ll",
                    ["pass-a", "pass-b"],
                    {"opt": "opt"},
                    out_dir,
                    max_rounds=1,
                    random_trials=3,
                    seed=0,
                    timeout=1,
                )

            best_path = _read_csv(out_dir / "random_best_path.csv")

        self.assertEqual(result["best_trial"], "0")
        self.assertEqual(result["pass_sequence"], "pass-b")
        self.assertEqual(result["final_ir_inst_count"], "2")
        self.assertEqual(best_path[0]["selected_pass"], "pass-b")

    def test_default_o2_oz_unsupported_is_recorded_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, passes_path = _make_run(Path(tmp))

            def fake_raw(opt, src, pipeline, out, timeout):
                return RunResult([opt], 1, "", f"unknown pipeline {pipeline}", 1.0)

            with mock.patch("phasebatch.baselines.run_opt_raw_pipeline", side_effect=fake_raw), _fake_no_active_profiles():
                compare_baselines(
                    run_dir,
                    passes_path,
                    methods=["default"],
                    max_rounds=1,
                    random_trials=1,
                    seed=0,
                    timeout=1,
                    jobs=1,
                    include_default_pipelines=True,
                )

            rows = _rows_by_method(run_dir / "baseline_results.csv")

        self.assertEqual(rows["default_O2"]["status"], "unsupported")
        self.assertEqual(rows["default_O2"]["final_ir_inst_count"], "")
        self.assertIn("unknown pipeline default<O2>", rows["default_O2"]["error_message"])
        self.assertEqual(rows["default_Oz"]["status"], "unsupported")

    def test_batch_optimizer_row_reads_final_pipeline_and_state_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, passes_path = _make_run(Path(tmp), root_count=6, optimized_pipeline="pass-a,pass-b,pass-c")
            (run_dir / "final.ll").write_text(_ir_with_instruction_count(2), encoding="utf-8")
            _write_csv(
                run_dir / "states.csv",
                ["state_id", "is_duplicate"],
                [
                    {"state_id": "S0000", "is_duplicate": "false"},
                    {"state_id": "S0001", "is_duplicate": "false"},
                    {"state_id": "S0002", "is_duplicate": "true"},
                ],
            )
            _write_csv(
                run_dir / "optimizer_events.csv",
                ["event_id", "round", "state_id", "event_type", "message"],
                [
                    {"event_id": "0", "round": "0", "state_id": "S0000", "event_type": "apply_batch", "message": ""},
                    {"event_id": "1", "round": "0", "state_id": "S0001", "event_type": "analyze_state", "message": ""},
                    {"event_id": "2", "round": "0", "state_id": "S0001", "event_type": "update_incumbent", "message": ""},
                ],
            )
            _write_csv(
                run_dir / "states" / "S0000" / "per_state_summary.csv",
                ["profile_time_ms", "pair_time_ms", "total_time_ms"],
                [{"profile_time_ms": "10.5", "pair_time_ms": "20.25", "total_time_ms": "35.0"}],
            )
            _write_csv(
                run_dir / "states" / "S0000" / "pass_profile.csv",
                ["pass"],
                [{"pass": "pass-a"}],
            )
            _write_csv(
                run_dir / "states" / "S0001" / "per_state_summary.csv",
                ["profile_time_ms", "pair_time_ms", "total_time_ms"],
                [{"profile_time_ms": "3.5", "pair_time_ms": "4.25", "total_time_ms": "9.0"}],
            )
            _write_csv(
                run_dir / "states" / "S0001" / "pass_profile.csv",
                ["pass"],
                [{"pass": "pass-b"}],
            )
            _write_csv(
                run_dir / "states" / "S0000" / "batch_validation.csv",
                ["tested_orders", "time_ms"],
                [{"tested_orders": "6", "time_ms": "7.5"}],
            )
            _write_csv(
                run_dir / "states" / "S0001" / "batch_validation.csv",
                ["tested_orders", "time_ms"],
                [{"tested_orders": "2", "time_ms": "1.5"}],
            )
            _write_csv(
                run_dir / "batch_state_transitions.csv",
                ["parent_state_id", "child_state_id", "batch_id"],
                [{"parent_state_id": "S0000", "child_state_id": "S0001", "batch_id": "B0000"}],
            )

            compare_baselines(run_dir, passes_path, methods=["batch"], max_rounds=1, random_trials=1, seed=0, timeout=1, jobs=1)

            rows = _rows_by_method(run_dir / "baseline_results.csv")

        self.assertEqual(rows["batch_optimizer"]["status"], "success")
        self.assertEqual(rows["batch_optimizer"]["final_ir_inst_count"], "2")
        self.assertEqual(rows["batch_optimizer"]["root_ir_inst_count"], "6")
        self.assertEqual(rows["batch_optimizer"]["pass_sequence"], "pass-a,pass-b,pass-c")
        self.assertEqual(rows["batch_optimizer"]["final_sequence_length"], "3")
        self.assertEqual(rows["batch_optimizer"]["states_evaluated"], "2")
        self.assertEqual(rows["batch_optimizer"]["opt_runs"], "2")
        self.assertEqual(rows["batch_optimizer"]["analysis_time_ms"], "44.000")
        self.assertEqual(rows["batch_optimizer"]["profiling_time_ms"], "14.000")
        self.assertEqual(rows["batch_optimizer"]["pair_testing_time_ms"], "24.500")
        self.assertEqual(rows["batch_optimizer"]["batch_validation_time_ms"], "9.000")
        self.assertEqual(rows["batch_optimizer"]["batch_apply_time_ms"], "")
        self.assertEqual(rows["batch_optimizer"]["optimizer_total_time_ms"], "53.000")
        self.assertEqual(rows["batch_optimizer"]["time_ms"], "53.000")
        self.assertEqual(rows["batch_optimizer"]["total_opt_invocations"], "13")

    def test_method_comparison_is_generated_with_objective_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, passes_path = _make_run(Path(tmp), root_count=4, optimized_pipeline="pass-a")
            (run_dir / "final.ll").write_text(_ir_with_instruction_count(2), encoding="utf-8")

            compare_baselines(run_dir, passes_path, methods=["batch"], max_rounds=1, random_trials=1, seed=0, timeout=1, jobs=1)

            summary = (run_dir / "method_comparison.md").read_text(encoding="utf-8")

        self.assertIn("# Method Comparison", summary)
        self.assertIn("under the IR instruction count objective in this run", summary)
        self.assertIn("objective is not proof", summary)

    def test_optimized_pipeline_replays_pipeline_and_records_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, passes_path = _make_run(Path(tmp), optimized_pipeline="pass-b,pass-a")
            calls = []

            def fake_run_opt(opt, src, passes, out, timeout):
                calls.append(list(passes))
                Path(out).write_text(_ir_with_instruction_count(1), encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            with mock.patch("phasebatch.baselines.run_opt", side_effect=fake_run_opt), _fake_no_active_profiles():
                compare_baselines(run_dir, passes_path, max_rounds=1, random_trials=1, seed=0, timeout=1, jobs=1)

            rows = _rows_by_method(run_dir / "baseline_results.csv")

        self.assertIn(["pass-b", "pass-a"], calls)
        self.assertEqual(rows["optimized_pipeline"]["status"], "success")
        self.assertEqual(rows["optimized_pipeline"]["pass_sequence"], "pass-b;pass-a")
        self.assertEqual(rows["optimized_pipeline"]["final_ir_inst_count"], "1")

    def test_baseline_results_contains_required_methods(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, passes_path = _make_run(Path(tmp))

            with _fake_successful_run_opt(), _fake_no_active_profiles():
                result = compare_baselines(run_dir, passes_path, max_rounds=1, random_trials=2, seed=0, timeout=1, jobs=1)

            rows = _rows_by_method(run_dir / "baseline_results.csv")
            baselines_dir_exists = (run_dir / "baselines").is_dir()
            trials_csv_exists = (run_dir / "random_baseline_trials.csv").exists()

        self.assertEqual(result["baseline_results_csv"], str(run_dir / "baseline_results.csv"))
        self.assertTrue(baselines_dir_exists)
        self.assertTrue(trials_csv_exists)
        self.assertTrue(
            {"default_O0", "batch_optimizer", "greedy_single_pass", "random_single_pass_best"}.issubset(rows)
        )

    def test_methods_all_runs_default_greedy_random_and_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, passes_path = _make_run(Path(tmp), optimized_pipeline="pass-a,pass-b")
            (run_dir / "final.ll").write_text(_ir_with_instruction_count(1), encoding="utf-8")

            with _fake_successful_run_opt(), _fake_no_active_profiles():
                compare_baselines(run_dir, passes_path, methods=["all"], max_rounds=1, random_trials=2, seed=0, timeout=1, jobs=1)

            rows = _rows_by_method(run_dir / "baseline_results.csv")

        self.assertTrue(
            {"default_O0", "batch_optimizer", "greedy_single_pass", "random_single_pass_best"}.issubset(rows)
        )


def _make_run(
    root: Path,
    *,
    root_count: int = 3,
    config_passes: list[str] | None = None,
    valid_passes: list[str] | None = None,
    optimized_pipeline: str = "",
) -> tuple[Path, Path]:
    run_dir = root / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    state_dir = run_dir / "states" / "S0000"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "input.ll").write_text(_ir_with_instruction_count(root_count), encoding="utf-8")
    (run_dir / "optimized_pipeline.txt").write_text(optimized_pipeline + ("\n" if optimized_pipeline else ""), encoding="utf-8")
    (run_dir / "metadata.json").write_text(
        json.dumps({"tools": {"opt": {"path": "opt"}}, "input": "input.c"}),
        encoding="utf-8",
    )
    config_passes = config_passes or ["pass-a", "pass-b"]
    passes_path = root / "passes.yaml"
    passes_path.parent.mkdir(parents=True, exist_ok=True)
    passes_path.write_text("passes:\n" + "".join(f"  - {pass_name}\n" for pass_name in config_passes), encoding="utf-8")
    valid_passes = valid_passes if valid_passes is not None else config_passes
    _write_csv(
        run_dir / "valid_passes.csv",
        ["pass", "valid", "reason", "test_time_ms"],
        [{"pass": pass_name, "valid": "true", "reason": "ok", "test_time_ms": "1"} for pass_name in valid_passes],
    )
    return run_dir, passes_path


def _fake_no_active_profiles():
    return mock.patch(
        "phasebatch.baselines.profile_passes",
        return_value=[{"pass": "pass-a", "success": "true", "active": "false", "output_path": ""}],
    )


def _fake_successful_run_opt():
    def fake_run_opt(opt, src, passes, out, timeout):
        Path(out).write_text(_ir_with_instruction_count(2), encoding="utf-8")
        return RunResult([opt], 0, "", "", 1.0)

    return mock.patch("phasebatch.baselines.run_opt", side_effect=fake_run_opt)


def _ir_with_instruction_count(count: int, opcode: str = "add") -> str:
    lines = ["define i32 @f(i32 %x) {", "entry:"]
    for index in range(count - 1):
        source = "%x" if index == 0 else f"%v{index - 1}"
        lines.append(f"  %v{index} = {opcode} i32 {source}, 1")
    if count > 0:
        value = "%x" if count == 1 else f"%v{count - 2}"
        lines.append(f"  ret i32 {value}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _rows_by_method(path: Path) -> dict[str, dict[str, str]]:
    return {row["method"]: row for row in _read_csv(path)}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
