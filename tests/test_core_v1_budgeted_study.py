import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.core_v1_budgeted_study import run_core_v1_budgeted_study


class CoreV1BudgetedStudyTests(unittest.TestCase):
    def test_expands_glob_inputs_and_aggregates_methods_reduction_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = _make_inputs(root, ["a.c", "b.ll"])
            passes = root / "passes.yaml"
            passes.write_text("passes:\n  - instcombine\n", encoding="utf-8")
            out_dir = root / "out"

            with _patched_successful_study():
                result = run_core_v1_budgeted_study(
                    [str(root / "inputs" / "*.c"), str(inputs[1])],
                    out_dir,
                    passes,
                    objective="ir-inst-count",
                    max_rounds=4,
                    beam_width=4,
                    max_states=500,
                    max_batches_per_state=20,
                    batch_frontier_policy="score",
                    validate_batches=True,
                    jobs=1,
                    timeout=1,
                    max_pairs=20,
                    baseline_methods=["default", "greedy", "random", "batch"],
                    random_trials=3,
                    seed=0,
                )

            runs = _read_csv(out_dir / "budgeted_study_runs.csv")
            methods = _read_csv(out_dir / "budgeted_study_methods.csv")
            reduction = _read_csv(out_dir / "budgeted_study_reduction.csv")
            evidence = _read_csv(out_dir / "budgeted_study_evidence.csv")
            summary = (out_dir / "budgeted_study_summary.md").read_text(encoding="utf-8")

        self.assertEqual(result["programs"], 2)
        self.assertEqual(result["successes"], 2)
        self.assertEqual([row["program"] for row in runs], ["a", "b"])
        self.assertEqual({row["program"] for row in methods}, {"a", "b"})
        self.assertIn("batch_optimizer", {row["method"] for row in methods})
        self.assertIn("config_order_once", {row["method"] for row in methods})
        self.assertEqual(reduction[0]["total_certified_batches"], "3")
        self.assertEqual(evidence[0]["selected_strong_certificates"], "2")
        self.assertIn("# Core-v1 Budgeted Study Summary", summary)
        self.assertIn("Budgeted mode changes search coverage, not batch correctness.", summary)

    def test_continue_on_error_records_failed_program_and_keeps_going(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = _make_inputs(root, ["bad.c", "good.c"])
            passes = root / "passes.yaml"
            passes.write_text("passes:\n  - instcombine\n", encoding="utf-8")
            out_dir = root / "out"

            def fake_optimizer(input_path, out_dir, passes_path, **kwargs):
                if Path(input_path).stem == "bad":
                    raise RuntimeError("optimizer boom")
                return _fake_optimizer(input_path, out_dir, passes_path, **kwargs)

            with _patched_successful_study(optimizer=fake_optimizer):
                result = run_core_v1_budgeted_study(
                    [str(path) for path in inputs],
                    out_dir,
                    passes,
                    objective="ir-inst-count",
                    max_rounds=4,
                    beam_width=4,
                    max_states=500,
                    max_batches_per_state=20,
                    batch_frontier_policy="score",
                    validate_batches=True,
                    jobs=1,
                    timeout=1,
                    max_pairs=20,
                    baseline_methods=["all"],
                    random_trials=3,
                    seed=0,
                    continue_on_error=True,
                )

            runs = _read_csv(out_dir / "budgeted_study_runs.csv")
            failures = _read_csv(out_dir / "failures.csv")

        self.assertEqual(result["successes"], 1)
        self.assertEqual(result["failures"], 1)
        self.assertEqual([row["status"] for row in runs], ["failed", "success"])
        self.assertEqual(failures[0]["stage"], "optimize")
        self.assertIn("optimizer boom", failures[0]["error_message"])

    def test_missing_input_is_recorded_when_continue_on_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = _make_inputs(root, ["good.c"])[0]
            missing = root / "inputs" / "missing.c"
            passes = root / "passes.yaml"
            passes.write_text("passes:\n  - instcombine\n", encoding="utf-8")
            out_dir = root / "out"

            with _patched_successful_study():
                result = run_core_v1_budgeted_study(
                    [str(missing), str(good)],
                    out_dir,
                    passes,
                    objective="ir-inst-count",
                    max_rounds=4,
                    beam_width=4,
                    max_states=500,
                    max_batches_per_state=20,
                    batch_frontier_policy="score",
                    validate_batches=True,
                    jobs=1,
                    timeout=1,
                    max_pairs=20,
                    baseline_methods=["all"],
                    random_trials=3,
                    seed=0,
                    continue_on_error=True,
                )

            failures = _read_csv(out_dir / "failures.csv")

        self.assertEqual(result["programs"], 2)
        self.assertEqual(result["failures"], 1)
        self.assertEqual(failures[0]["stage"], "input")
        self.assertIn("missing input", failures[0]["error_message"])

    def test_unmatched_glob_warns_but_does_not_fail_when_other_inputs_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = _make_inputs(root, ["good.c"])[0]
            passes = root / "passes.yaml"
            passes.write_text("passes:\n  - instcombine\n", encoding="utf-8")
            out_dir = root / "out"
            warnings: list[str] = []

            with _patched_successful_study():
                result = run_core_v1_budgeted_study(
                    [str(root / "inputs" / "none*.c"), str(good)],
                    out_dir,
                    passes,
                    objective="ir-inst-count",
                    max_rounds=4,
                    beam_width=4,
                    max_states=500,
                    max_batches_per_state=20,
                    batch_frontier_policy="score",
                    validate_batches=True,
                    jobs=1,
                    timeout=1,
                    max_pairs=20,
                    baseline_methods=["all"],
                    random_trials=3,
                    seed=0,
                    warn=warnings.append,
                )

            runs = _read_csv(out_dir / "budgeted_study_runs.csv")
            failures = _read_csv(out_dir / "failures.csv")

        self.assertEqual(result["successes"], 1)
        self.assertEqual([row["program"] for row in runs], ["good"])
        self.assertEqual(failures[0]["stage"], "input_glob")
        self.assertIn("matched no files", warnings[0])

    def test_missing_reduction_and_evidence_do_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = _make_inputs(root, ["one.c"])[0]
            passes = root / "passes.yaml"
            passes.write_text("passes:\n  - instcombine\n", encoding="utf-8")
            out_dir = root / "out"

            with mock.patch("phasebatch.core_v1_budgeted_study.run_optimizer", side_effect=_fake_optimizer), \
                mock.patch("phasebatch.core_v1_budgeted_study.run_baseline_comparison", side_effect=_fake_baselines), \
                mock.patch("phasebatch.core_v1_budgeted_study.run_reduction_summary", side_effect=RuntimeError("missing reduction")), \
                mock.patch("phasebatch.core_v1_budgeted_study.run_evidence_pack", side_effect=RuntimeError("missing evidence")):
                result = run_core_v1_budgeted_study(
                    [str(input_path)],
                    out_dir,
                    passes,
                    objective="ir-inst-count",
                    max_rounds=4,
                    beam_width=4,
                    max_states=500,
                    max_batches_per_state=20,
                    batch_frontier_policy="score",
                    validate_batches=True,
                    jobs=1,
                    timeout=1,
                    max_pairs=20,
                    baseline_methods=["all"],
                    random_trials=3,
                    seed=0,
                    continue_on_error=True,
                )

            runs = _read_csv(out_dir / "budgeted_study_runs.csv")
            failures = _read_csv(out_dir / "failures.csv")

        self.assertEqual(result["successes"], 1)
        self.assertEqual(runs[0]["reduction_status"], "failed")
        self.assertEqual(runs[0]["evidence_status"], "failed")
        self.assertEqual({row["stage"] for row in failures}, {"reduction", "evidence"})

    def test_summary_reports_win_tie_loss_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = _make_inputs(root, ["win.c", "tie.c", "loss.c"])
            passes = root / "passes.yaml"
            passes.write_text("passes:\n  - instcombine\n", encoding="utf-8")
            out_dir = root / "out"

            with _patched_successful_study(baselines=_fake_varied_baselines):
                run_core_v1_budgeted_study(
                    [str(path) for path in inputs],
                    out_dir,
                    passes,
                    objective="ir-inst-count",
                    max_rounds=4,
                    beam_width=4,
                    max_states=500,
                    max_batches_per_state=20,
                    batch_frontier_policy="score",
                    validate_batches=True,
                    jobs=1,
                    timeout=1,
                    max_pairs=20,
                    baseline_methods=["all"],
                    random_trials=3,
                    seed=0,
                )

            summary = (out_dir / "budgeted_study_summary.md").read_text(encoding="utf-8")

        self.assertIn("- batch vs greedy_single_pass: wins=1 ties=1 losses=1", summary)


def _patched_successful_study(optimizer=None, baselines=None):
    optimizer = optimizer or _fake_optimizer
    baselines = baselines or _fake_baselines
    return mock.patch.multiple(
        "phasebatch.core_v1_budgeted_study",
        run_optimizer=mock.Mock(side_effect=optimizer),
        run_baseline_comparison=mock.Mock(side_effect=baselines),
        run_reduction_summary=mock.Mock(side_effect=_fake_reduction),
        run_evidence_pack=mock.Mock(side_effect=_fake_evidence),
    )


def _make_inputs(root: Path, names: list[str]) -> list[Path]:
    inputs = root / "inputs"
    inputs.mkdir(exist_ok=True)
    paths = []
    for name in names:
        path = inputs / name
        path.write_text("int f(void){return 0;}\n", encoding="utf-8")
        paths.append(path)
    return paths


def _fake_optimizer(input_path: Path, out_dir: Path, passes_path: Path, **kwargs) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    program = Path(input_path).stem
    _write_csv(
        out_dir / "states.csv",
        ["program", "state_id", "depth", "is_duplicate"],
        [{"program": program, "state_id": "S0000", "depth": "0", "is_duplicate": "false"}],
    )
    _write_csv(
        out_dir / "batch_state_transitions.csv",
        ["program", "parent_state_id", "child_state_id"],
        [{"program": program, "parent_state_id": "S0000", "child_state_id": "S0001"}],
    )
    _write_csv(
        out_dir / "chosen_path_summary.csv",
        [
            "program",
            "selected_final_state",
            "final_ir_inst_count",
            "root_ir_inst_count",
            "ir_inst_delta",
            "ir_inst_reduction_pct",
            "path_steps",
        ],
        [
            {
                "program": program,
                "selected_final_state": "S0001",
                "final_ir_inst_count": "8",
                "root_ir_inst_count": "12",
                "ir_inst_delta": "-4",
                "ir_inst_reduction_pct": "33.33",
                "path_steps": "1",
            }
        ],
    )
    _write_csv(
        out_dir / "optimizer_timing.csv",
        ["optimizer_total_time_ms"],
        [{"optimizer_total_time_ms": "123.000"}],
    )
    (out_dir / "optimized_pipeline.txt").write_text("instcombine,simplifycfg\n", encoding="utf-8")
    _write_csv(out_dir / "leaf_states.csv", ["state_id", "selected_as_final", "leaf_reason"], [{"state_id": "S0001", "selected_as_final": "true", "leaf_reason": "max_rounds_reached"}])
    return {"states": 1, "batch_transitions": 1}


def _fake_baselines(run_dir: Path, passes_path: Path, **kwargs) -> dict:
    run_dir = Path(run_dir)
    rows = [
        _method_row("default_O0", "success", "12"),
        _method_row("batch_optimizer", "success", "8"),
        _method_row("config_order_once", "success", "9"),
        _method_row("greedy_single_pass", "success", "7"),
        _method_row("random_single_pass_best", "success", "10"),
        _method_row("default_O2", "unsupported", ""),
        _method_row("default_Oz", "unsupported", ""),
    ]
    _write_csv(run_dir / "baseline_results.csv", list(rows[0]), rows)
    (run_dir / "baselines").mkdir(exist_ok=True)
    return {"rows": len(rows)}


def _fake_varied_baselines(run_dir: Path, passes_path: Path, **kwargs) -> dict:
    program = Path(run_dir).parent.name
    greedy = {"win": "9", "tie": "8", "loss": "7"}.get(program, "9")
    rows = [
        _method_row("default_O0", "success", "12"),
        _method_row("batch_optimizer", "success", "8"),
        _method_row("config_order_once", "success", "9"),
        _method_row("greedy_single_pass", "success", greedy),
        _method_row("random_single_pass_best", "success", "10"),
    ]
    _write_csv(run_dir / "baseline_results.csv", list(rows[0]), rows)
    (run_dir / "baselines").mkdir(exist_ok=True)
    return {"rows": len(rows)}


def _fake_reduction(run_dir: Path) -> dict:
    program = Path(run_dir).parent.name
    _write_csv(
        run_dir / "reduction_summary.csv",
        [
            "program",
            "total_states",
            "max_depth",
            "total_active_passes",
            "total_tested_pairs",
            "total_commute_pairs",
            "total_order_sensitive_pairs",
            "total_unknown_pairs",
            "total_batch_candidates",
            "total_certified_batches",
            "total_executable_batches",
            "total_executed_transitions",
            "total_skipped_batches",
            "total_dropped_active_passes",
            "avg_local_reduction_log10",
            "max_local_reduction_log10",
        ],
        [
            {
                "program": program,
                "total_states": "2",
                "max_depth": "1",
                "total_active_passes": "4",
                "total_tested_pairs": "6",
                "total_commute_pairs": "4",
                "total_order_sensitive_pairs": "2",
                "total_unknown_pairs": "0",
                "total_batch_candidates": "4",
                "total_certified_batches": "3",
                "total_executable_batches": "3",
                "total_executed_transitions": "1",
                "total_skipped_batches": "1",
                "total_dropped_active_passes": "0",
                "avg_local_reduction_log10": "0.9",
                "max_local_reduction_log10": "1.2",
            }
        ],
    )
    (run_dir / "reduction_summary.md").write_text("# Reduction\n", encoding="utf-8")
    return {"states": 2}


def _fake_evidence(run_dir: Path) -> dict:
    program = Path(run_dir).parent.name
    _write_csv(
        run_dir / "evidence_pack.csv",
        [
            "program",
            "selected_path_batches",
            "selected_strong_certificates",
            "selected_weak_certificates",
            "executed_batches",
            "executed_strong_certificates",
            "executed_weak_certificates",
            "executed_rejected",
            "dropped_active_passes",
            "replay_status",
            "replay_hashes_match",
        ],
        [
            {
                "program": program,
                "selected_path_batches": "2",
                "selected_strong_certificates": "2",
                "selected_weak_certificates": "0",
                "executed_batches": "3",
                "executed_strong_certificates": "3",
                "executed_weak_certificates": "0",
                "executed_rejected": "0",
                "dropped_active_passes": "0",
                "replay_status": "success",
                "replay_hashes_match": "true",
            }
        ],
    )
    (run_dir / "evidence_pack.md").write_text("# Evidence\n", encoding="utf-8")
    return {"selected_batches": 2}


def _method_row(method: str, status: str, final_inst: str) -> dict[str, str]:
    root = "12"
    delta = "" if not final_inst else str(int(final_inst) - int(root))
    reduction = "" if not final_inst else f"{((int(root) - int(final_inst)) / int(root)) * 100:.2f}"
    return {
        "program": "",
        "method": method,
        "status": status,
        "final_ir_path": "",
        "final_ir_hash": "",
        "final_ir_inst_count": final_inst,
        "root_ir_inst_count": root,
        "ir_inst_delta": delta,
        "ir_inst_reduction_pct": reduction,
        "pass_sequence": method,
        "final_sequence_length": "1" if final_inst else "",
        "states_evaluated": "1" if final_inst else "",
        "opt_runs": "1" if final_inst else "",
        "time_ms": "1.0",
        "error_message": "unsupported" if status == "unsupported" else "",
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
