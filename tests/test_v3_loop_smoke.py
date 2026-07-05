import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.v3_loop_smoke import run_v3_loop_smoke


class V3LoopSmokeTests(unittest.TestCase):
    def test_missing_inputs_are_skipped_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing = root / "loop.c"
            missing = root / "missing.c"
            passes = root / "middleend_passes_v3.yaml"
            existing.write_text("int f(void){for(int i=0;i<3;i++){} return 0;}\n", encoding="utf-8")
            passes.write_text("passes:\n  - name: licm\n    pipeline: licm\n    category: loop\n", encoding="utf-8")
            warnings: list[str] = []

            with mock.patch("phasebatch.v3_loop_smoke.run_pass_audit", side_effect=_fake_audit), \
                mock.patch("phasebatch.v3_loop_smoke.run_optimizer", side_effect=_fake_optimize), \
                mock.patch("phasebatch.v3_loop_smoke.run_baseline_comparison", side_effect=_fake_compare):
                result = run_v3_loop_smoke(
                    [str(missing), str(existing)],
                    root / "out",
                    passes,
                    optimizer_mode="budgeted",
                    objective="ir-inst-count",
                    max_rounds=3,
                    beam_width=4,
                    max_states=800,
                    max_batches_per_state=12,
                    batch_frontier_policy="score",
                    validate_batches=True,
                    jobs=1,
                    timeout=1,
                    max_pairs=1000,
                    warn=warnings.append,
                )

            runs = _read_csv(root / "out" / "v3_loop_runs.csv")

        self.assertEqual(result["programs_attempted"], 1)
        self.assertEqual(runs[0]["program"], "loop")
        self.assertIn("skipping missing input", warnings[0])

    def test_audit_resolved_passes_are_used_for_optimize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "loop.c"
            passes = root / "middleend_passes_v3.yaml"
            input_path.write_text("int f(void){return 0;}\n", encoding="utf-8")
            passes.write_text("passes:\n  - name: licm\n    pipeline: licm\n    category: loop\n", encoding="utf-8")

            with mock.patch("phasebatch.v3_loop_smoke.run_pass_audit", side_effect=_fake_audit), \
                mock.patch("phasebatch.v3_loop_smoke.run_optimizer", side_effect=_fake_optimize) as fake_optimizer, \
                mock.patch("phasebatch.v3_loop_smoke.run_baseline_comparison", side_effect=_fake_compare):
                run_v3_loop_smoke(
                    [str(input_path)],
                    root / "out",
                    passes,
                    optimizer_mode="budgeted",
                    objective="ir-inst-count",
                    max_rounds=3,
                    beam_width=4,
                    max_states=800,
                    max_batches_per_state=12,
                    batch_frontier_policy="score",
                    validate_batches=True,
                    jobs=1,
                    timeout=1,
                    max_pairs=1000,
                )

            optimize_passes = fake_optimizer.call_args.args[2]

        self.assertEqual(Path(optimize_passes).name, "resolved_passes.yaml")
        self.assertIn("audit", str(optimize_passes))

    def test_resolved_loop_pipelines_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "loop.c"
            passes = root / "middleend_passes_v3.yaml"
            input_path.write_text("int f(void){return 0;}\n", encoding="utf-8")
            passes.write_text("passes:\n  - name: licm\n    pipeline_candidates:\n      - licm\n      - function(loop(licm))\n    category: loop\n", encoding="utf-8")

            with mock.patch("phasebatch.v3_loop_smoke.run_pass_audit", side_effect=_fake_audit), \
                mock.patch("phasebatch.v3_loop_smoke.run_optimizer", side_effect=_fake_optimize), \
                mock.patch("phasebatch.v3_loop_smoke.run_baseline_comparison", side_effect=_fake_compare):
                run_v3_loop_smoke(
                    [str(input_path)],
                    root / "out",
                    passes,
                    optimizer_mode="budgeted",
                    objective="ir-inst-count",
                    max_rounds=3,
                    beam_width=4,
                    max_states=800,
                    max_batches_per_state=12,
                    batch_frontier_policy="score",
                    validate_batches=True,
                    jobs=1,
                    timeout=1,
                    max_pairs=1000,
                )

            summary = (root / "out" / "v3_loop_summary.md").read_text(encoding="utf-8")
            runs = _read_csv(root / "out" / "v3_loop_runs.csv")

        self.assertIn("licm=function(loop(licm))", summary)
        self.assertEqual(runs[0]["valid_loop_passes"], "2")
        self.assertEqual(runs[0]["invalid_loop_passes"], "1")

    def test_summary_csv_and_markdown_are_generated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "loop.c"
            passes = root / "middleend_passes_v3.yaml"
            input_path.write_text("int f(void){return 0;}\n", encoding="utf-8")
            passes.write_text("passes:\n  - name: licm\n    pipeline: licm\n    category: loop\n", encoding="utf-8")

            with mock.patch("phasebatch.v3_loop_smoke.run_pass_audit", side_effect=_fake_audit), \
                mock.patch("phasebatch.v3_loop_smoke.run_optimizer", side_effect=_fake_optimize), \
                mock.patch("phasebatch.v3_loop_smoke.run_baseline_comparison", side_effect=_fake_compare):
                result = run_v3_loop_smoke(
                    [str(input_path)],
                    root / "out",
                    passes,
                    optimizer_mode="budgeted",
                    objective="ir-inst-count",
                    max_rounds=3,
                    beam_width=4,
                    max_states=800,
                    max_batches_per_state=12,
                    batch_frontier_policy="score",
                    validate_batches=True,
                    jobs=1,
                    timeout=1,
                    max_pairs=1000,
                )

            summary_rows = _read_csv(root / "out" / "v3_loop_summary.csv")
            summary_text = (root / "out" / "v3_loop_summary.md").read_text(encoding="utf-8")

        self.assertEqual(result["successes"], 1)
        self.assertEqual(summary_rows[0]["active_loop_passes_depth0"], "1")
        self.assertIn("# V3 Middle-End / Loop Pass Smoke Summary", summary_text)
        self.assertIn("Loop passes may require nested New Pass Manager pipeline syntax.", summary_text)

    def test_invalid_loop_passes_do_not_crash_whole_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "loop.c"
            passes = root / "middleend_passes_v3.yaml"
            input_path.write_text("int f(void){return 0;}\n", encoding="utf-8")
            passes.write_text("passes:\n  - name: bad-loop\n    pipeline: bad-loop\n    category: loop\n", encoding="utf-8")

            with mock.patch("phasebatch.v3_loop_smoke.run_pass_audit", side_effect=_fake_audit), \
                mock.patch("phasebatch.v3_loop_smoke.run_optimizer", side_effect=_fake_optimize), \
                mock.patch("phasebatch.v3_loop_smoke.run_baseline_comparison", side_effect=_fake_compare):
                result = run_v3_loop_smoke(
                    [str(input_path)],
                    root / "out",
                    passes,
                    optimizer_mode="budgeted",
                    objective="ir-inst-count",
                    max_rounds=3,
                    beam_width=4,
                    max_states=800,
                    max_batches_per_state=12,
                    batch_frontier_policy="score",
                    validate_batches=True,
                    jobs=1,
                    timeout=1,
                    max_pairs=1000,
                    continue_on_error=True,
                )

            runs = _read_csv(root / "out" / "v3_loop_runs.csv")

        self.assertEqual(result["failures"], 0)
        self.assertEqual(runs[0]["status"], "success")
        self.assertEqual(runs[0]["invalid_loop_passes"], "1")


def _fake_audit(input_path: Path, passes_path: Path, out_dir: Path, **kwargs) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _audit_row("instcombine", "scalar", "instcombine", "true"),
        _audit_row("licm", "loop", "function(loop(licm))", "true"),
        _audit_row("loop-rotate", "loop", "function(loop(loop-rotate))", "true"),
        _audit_row("bad-loop", "loop", "", "false"),
    ]
    _write_csv(out_dir / "pass_audit.csv", list(rows[0]), rows)
    _write_csv(
        out_dir / "invalid_passes.csv",
        ["pass", "category", "stage", "attempted_candidates", "failure_kind", "stderr_summary"],
        [{"pass": "bad-loop", "category": "loop", "stage": "v3", "attempted_candidates": "bad-loop", "failure_kind": "failed", "stderr_summary": "bad"}],
    )
    (out_dir / "resolved_passes.yaml").write_text(
        "passes:\n"
        "  - name: instcombine\n    pipeline: instcombine\n    category: scalar\n"
        "  - name: licm\n    pipeline: function(loop(licm))\n    category: loop\n",
        encoding="utf-8",
    )
    return {"valid_passes": 3, "invalid_passes": 1, "resolved_passes_yaml": str(out_dir / "resolved_passes.yaml")}


def _audit_row(name: str, category: str, pipeline: str, valid: str) -> dict[str, str]:
    return {
        "pass": name,
        "category": category,
        "stage": "v3" if category == "loop" else "v2",
        "enabled": "true",
        "candidate_index": "1" if pipeline.startswith("function(") else "0",
        "candidate_pipeline": pipeline,
        "resolved_pipeline": pipeline,
        "recognized_by_opt": valid,
        "valid_on_input": valid,
        "active_on_input": "true" if valid == "true" else "false",
        "input_hash": "h0",
        "output_hash": "h1" if valid == "true" else "",
        "ir_inst_before": "10",
        "ir_inst_after": "9" if valid == "true" else "",
        "inst_delta": "-1" if valid == "true" else "",
        "time_ms": "1",
        "failure_kind": "" if valid == "true" else "failed",
        "stderr_summary": "",
        "recommended_action": "keep" if valid == "true" else "drop_invalid",
    }


def _fake_optimize(input_path: Path, out_dir: Path, passes_path: Path, **kwargs) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_dir = out_dir / "states" / "S0000"
    state_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "states.csv", ["state_id", "is_duplicate"], [{"state_id": "S0000", "is_duplicate": "false"}, {"state_id": "S0001", "is_duplicate": "false"}])
    _write_csv(out_dir / "batch_state_transitions.csv", ["parent_state_id", "child_state_id", "correctness_class", "validation_status"], [{"parent_state_id": "S0000", "child_state_id": "S0001", "correctness_class": "certified_batch", "validation_status": "all_permutations_same"}])
    _write_csv(state_dir / "per_state_summary.csv", ["active_passes", "pairs_tested", "dynamic_commute", "order_sensitive", "unknown"], [{"active_passes": "5", "pairs_tested": "10", "dynamic_commute": "7", "order_sensitive": "2", "unknown": "1"}])
    _write_csv(state_dir / "pass_profile.csv", ["pass", "success", "active"], [{"pass": "instcombine", "success": "true", "active": "true"}, {"pass": "licm", "success": "true", "active": "true"}, {"pass": "loop-rotate", "success": "true", "active": "false"}])
    _write_csv(state_dir / "batch_summary.csv", ["batch_candidates", "max_component_size"], [{"batch_candidates": "4", "max_component_size": "3"}])
    _write_csv(
        state_dir / "batch_correctness.csv",
        ["batch_id", "correctness_class", "validation_status", "can_execute"],
        [
            {"batch_id": "B0000", "correctness_class": "certified_batch", "validation_status": "all_permutations_same", "can_execute": "true"},
            {"batch_id": "B0001", "correctness_class": "sampled_batch", "validation_status": "sampled_same", "can_execute": "false"},
        ],
    )
    _write_csv(state_dir / "coverage_summary.csv", ["dropped_active_passes"], [{"dropped_active_passes": "0"}])
    _write_csv(out_dir / "chosen_path_summary.csv", ["selected_final_state", "final_ir_inst_count"], [{"selected_final_state": "S0001", "final_ir_inst_count": "8"}])
    (out_dir / "optimized_pipeline_names.txt").write_text("instcombine,licm\n", encoding="utf-8")
    return {"states": 2, "selected_final_state": "S0001"}


def _fake_compare(run_dir: Path, passes_path: Path, **kwargs) -> dict:
    run_dir = Path(run_dir)
    (run_dir / "baselines").mkdir(parents=True, exist_ok=True)
    _write_csv(run_dir / "baseline_results.csv", ["method", "status"], [{"method": "batch_optimizer", "status": "success"}])
    return {"rows": 1}


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
