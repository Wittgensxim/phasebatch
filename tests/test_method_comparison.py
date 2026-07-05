import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.method_comparison import run_method_comparison


class MethodComparisonRunnerTests(unittest.TestCase):
    def test_expands_glob_inputs_and_writes_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "a.c"
            second = root / "b.ll"
            passes = root / "passes.yaml"
            first.write_text("int a(void){return 0;}\n", encoding="utf-8")
            second.write_text("define i32 @b() { ret i32 0 }\n", encoding="utf-8")
            passes.write_text("passes:\n  - instcombine\n", encoding="utf-8")

            with mock.patch("phasebatch.method_comparison.run_optimizer", side_effect=_fake_optimize), \
                mock.patch("phasebatch.method_comparison.run_baseline_comparison", side_effect=_fake_compare):
                result = run_method_comparison(
                    [str(root / "*.c"), str(second)],
                    root / "out",
                    passes,
                    optimizer_mode="budgeted",
                    objective="ir-inst-count",
                    max_rounds=3,
                    beam_width=4,
                    max_states=500,
                    max_batches_per_state=10,
                    batch_frontier_policy="score",
                    validate_batches=True,
                    baseline_max_rounds=5,
                    random_trials=20,
                    seed=0,
                    include_default_pipelines=True,
                    jobs=1,
                    timeout=1,
                    max_pairs=20,
                )

            runs = _read_csv(root / "out" / "method_comparison_runs.csv")
            rows = _read_csv(root / "out" / "method_comparison_results.csv")

        self.assertEqual(result["programs"], 2)
        self.assertEqual([row["program"] for row in runs], ["a", "b"])
        self.assertIn("default_O0", {row["method"] for row in rows})
        self.assertIn("batch_optimizer", {row["method"] for row in rows})

    def test_one_mock_program_copies_per_program_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "one.c"
            passes = root / "passes.yaml"
            input_path.write_text("int one(void){return 0;}\n", encoding="utf-8")
            passes.write_text("passes:\n  - instcombine\n", encoding="utf-8")

            with mock.patch("phasebatch.method_comparison.run_optimizer", side_effect=_fake_optimize), \
                mock.patch("phasebatch.method_comparison.run_baseline_comparison", side_effect=_fake_compare):
                run_method_comparison(
                    [str(input_path)],
                    root / "out",
                    passes,
                    optimizer_mode="exact",
                    objective="ir-inst-count",
                    max_rounds=2,
                    beam_width=4,
                    max_states=500,
                    max_batches_per_state=10,
                    batch_frontier_policy="score",
                    validate_batches=True,
                    baseline_max_rounds=5,
                    random_trials=3,
                    seed=7,
                    include_default_pipelines=False,
                    jobs=1,
                    timeout=1,
                    max_pairs=None,
                )

            baseline_exists = (root / "out" / "one" / "baseline_results.csv").exists()
            summary_exists = (root / "out" / "one" / "method_comparison.md").exists()
            baselines_exists = (root / "out" / "one" / "baselines").is_dir()

        self.assertTrue(baseline_exists)
        self.assertTrue(summary_exists)
        self.assertTrue(baselines_exists)

    def test_continue_on_error_records_failed_program(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "good.c"
            bad = root / "bad.c"
            passes = root / "passes.yaml"
            good.write_text("int good(void){return 0;}\n", encoding="utf-8")
            bad.write_text("int bad(void){return 0;}\n", encoding="utf-8")
            passes.write_text("passes:\n  - instcombine\n", encoding="utf-8")

            def fake_optimizer(input_path, out_dir, passes_path, **kwargs):
                if Path(input_path).name == "bad.c":
                    raise RuntimeError("optimizer boom")
                return _fake_optimize(input_path, out_dir, passes_path, **kwargs)

            with mock.patch("phasebatch.method_comparison.run_optimizer", side_effect=fake_optimizer), \
                mock.patch("phasebatch.method_comparison.run_baseline_comparison", side_effect=_fake_compare):
                result = run_method_comparison(
                    [str(good), str(bad)],
                    root / "out",
                    passes,
                    optimizer_mode="budgeted",
                    objective="ir-inst-count",
                    max_rounds=1,
                    beam_width=4,
                    max_states=500,
                    max_batches_per_state=10,
                    batch_frontier_policy="score",
                    validate_batches=False,
                    baseline_max_rounds=1,
                    random_trials=1,
                    seed=0,
                    include_default_pipelines=False,
                    jobs=1,
                    timeout=1,
                    max_pairs=None,
                    continue_on_error=True,
                )

            runs = _read_csv(root / "out" / "method_comparison_runs.csv")
            failures = _read_csv(root / "out" / "method_comparison_failures.csv")

        self.assertEqual(result["successes"], 1)
        self.assertEqual(result["failures"], 1)
        self.assertEqual([row["status"] for row in runs], ["success", "failed"])
        self.assertIn("optimizer boom", failures[0]["error_message"])

    def test_unsupported_default_pipeline_is_recorded_not_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "one.c"
            passes = root / "passes.yaml"
            input_path.write_text("int one(void){return 0;}\n", encoding="utf-8")
            passes.write_text("passes:\n  - instcombine\n", encoding="utf-8")

            with mock.patch("phasebatch.method_comparison.run_optimizer", side_effect=_fake_optimize), \
                mock.patch("phasebatch.method_comparison.run_baseline_comparison", side_effect=_fake_compare_with_unsupported_default):
                result = run_method_comparison(
                    [str(input_path)],
                    root / "out",
                    passes,
                    optimizer_mode="budgeted",
                    objective="ir-inst-count",
                    max_rounds=1,
                    beam_width=4,
                    max_states=500,
                    max_batches_per_state=10,
                    batch_frontier_policy="score",
                    validate_batches=False,
                    baseline_max_rounds=1,
                    random_trials=1,
                    seed=0,
                    include_default_pipelines=True,
                    jobs=1,
                    timeout=1,
                    max_pairs=None,
                )

            rows = _read_csv(root / "out" / "method_comparison_results.csv")
            failures = _read_csv(root / "out" / "method_comparison_failures.csv")

        self.assertEqual(result["failures"], 0)
        self.assertIn("unsupported", {row["status"] for row in rows})
        self.assertIn("default_Oz", {row["method"] for row in failures})

    def test_summary_computes_batch_wins_and_best_excludes_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "one.c"
            passes = root / "passes.yaml"
            input_path.write_text("int one(void){return 0;}\n", encoding="utf-8")
            passes.write_text("passes:\n  - instcombine\n", encoding="utf-8")

            with mock.patch("phasebatch.method_comparison.run_optimizer", side_effect=_fake_optimize), \
                mock.patch("phasebatch.method_comparison.run_baseline_comparison", side_effect=_fake_compare_with_unsupported_default):
                run_method_comparison(
                    [str(input_path)],
                    root / "out",
                    passes,
                    optimizer_mode="budgeted",
                    objective="ir-inst-count",
                    max_rounds=1,
                    beam_width=4,
                    max_states=500,
                    max_batches_per_state=10,
                    batch_frontier_policy="score",
                    validate_batches=False,
                    baseline_max_rounds=1,
                    random_trials=1,
                    seed=0,
                    include_default_pipelines=True,
                    jobs=1,
                    timeout=1,
                    max_pairs=None,
                )

            summary = (root / "out" / "method_comparison_summary.md").read_text(encoding="utf-8")

        self.assertIn("batch beats default_O0: 1", summary)
        self.assertIn("| one | batch_optimizer | 8 |", summary)
        self.assertIn("IR instruction count is an evaluation objective. It is not used as commutation or independence proof.", summary)

    def test_result_table_has_required_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "one.c"
            passes = root / "passes.yaml"
            input_path.write_text("int one(void){return 0;}\n", encoding="utf-8")
            passes.write_text("passes:\n  - instcombine\n", encoding="utf-8")

            with mock.patch("phasebatch.method_comparison.run_optimizer", side_effect=_fake_optimize), \
                mock.patch("phasebatch.method_comparison.run_baseline_comparison", side_effect=_fake_compare):
                run_method_comparison(
                    [str(input_path)],
                    root / "out",
                    passes,
                    optimizer_mode="budgeted",
                    objective="ir-inst-count",
                    max_rounds=1,
                    beam_width=4,
                    max_states=500,
                    max_batches_per_state=10,
                    batch_frontier_policy="score",
                    validate_batches=False,
                    baseline_max_rounds=1,
                    random_trials=1,
                    seed=0,
                    include_default_pipelines=False,
                    jobs=1,
                    timeout=1,
                    max_pairs=None,
                )

            rows = _read_csv(root / "out" / "method_comparison_results.csv")

        for column in ["method", "final_ir_inst_count", "states_evaluated", "opt_runs", "final_sequence_length"]:
            self.assertIn(column, rows[0])


def _fake_optimize(input_path: Path, out_dir: Path, passes_path: Path, **kwargs) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "final.ll").write_text("define i32 @f() {\nentry:\n  ret i32 0\n}\n", encoding="utf-8")
    (out_dir / "optimized_pipeline.txt").write_text("instcombine\n", encoding="utf-8")
    _write_csv(out_dir / "states.csv", ["state_id", "is_duplicate"], [{"state_id": "S0000", "is_duplicate": "false"}])
    _write_csv(out_dir / "state_dag.csv", ["source_state_id", "target_state_id"], [])
    return {"states": 1, "selected_final_state": "S0000"}


def _fake_compare(run_dir: Path, passes_path: Path, **kwargs) -> dict:
    run_dir = Path(run_dir)
    rows = [
        _result_row("default_O0", "success", "12", "0", "0", "0"),
        _result_row("greedy_single_pass", "success", "9", "2", "4", "1"),
        _result_row("random_single_pass_best", "success", "10", "2", "4", "1"),
        _result_row("batch_optimizer", "success", "8", "3", "6", "2"),
    ]
    _write_csv(run_dir / "baseline_results.csv", list(rows[0]), rows)
    (run_dir / "method_comparison.md").write_text("# Method Comparison\n", encoding="utf-8")
    (run_dir / "baselines").mkdir(exist_ok=True)
    return {"rows": len(rows)}


def _fake_compare_with_unsupported_default(run_dir: Path, passes_path: Path, **kwargs) -> dict:
    run_dir = Path(run_dir)
    rows = [
        _result_row("default_O0", "success", "12", "0", "0", "0"),
        _result_row("default_O2", "unsupported", "", "1", "1", "1"),
        _result_row("default_Oz", "unsupported", "", "1", "1", "1"),
        _result_row("greedy_single_pass", "success", "9", "2", "4", "1"),
        _result_row("random_single_pass_best", "success", "10", "2", "4", "1"),
        _result_row("batch_optimizer", "success", "8", "3", "6", "2"),
    ]
    _write_csv(run_dir / "baseline_results.csv", list(rows[0]), rows)
    (run_dir / "method_comparison.md").write_text("# Method Comparison\n", encoding="utf-8")
    (run_dir / "baselines").mkdir(exist_ok=True)
    return {"rows": len(rows)}


def _result_row(method: str, status: str, final_inst: str, states: str, opt_runs: str, sequence_length: str) -> dict[str, str]:
    root = "12"
    delta = "" if not final_inst else str(int(final_inst) - int(root))
    reduction = "" if not final_inst else f"{((int(root) - int(final_inst)) / int(root)) * 100:.2f}"
    return {
        "program": "",
        "method": method,
        "status": status,
        "final_ir_path": f"{method}.ll" if final_inst else "",
        "final_ir_hash": "hash" if final_inst else "",
        "final_ir_inst_count": final_inst,
        "root_ir_inst_count": root,
        "ir_inst_delta": delta,
        "ir_inst_reduction_pct": reduction,
        "pass_sequence": method,
        "final_sequence_length": sequence_length,
        "states_evaluated": states,
        "opt_runs": opt_runs,
        "time_ms": "1.0",
        "stop_reason": "",
        "error_message": "unsupported" if status == "unsupported" else "",
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
