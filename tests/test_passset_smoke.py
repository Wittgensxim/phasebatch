import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.passset_smoke import run_passset_smoke


class PasssetSmokeTests(unittest.TestCase):
    def test_runs_multiple_passsets_and_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "branch.c"
            v1 = root / "core_passes_v1.yaml"
            v2 = root / "scalar_passes_v2.yaml"
            input_path.write_text("int f(void){return 0;}\n", encoding="utf-8")
            v1.write_text("passes:\n  - instcombine\n", encoding="utf-8")
            v2.write_text("passes:\n  - instcombine\n  - simplifycfg\n", encoding="utf-8")

            with mock.patch("phasebatch.passset_smoke.run_pass_audit", side_effect=_fake_audit), \
                mock.patch("phasebatch.passset_smoke.run_optimizer", side_effect=_fake_optimize), \
                mock.patch("phasebatch.passset_smoke.run_baseline_comparison", side_effect=_fake_compare):
                result = run_passset_smoke(
                    [str(input_path)],
                    [v1, v2],
                    root / "out",
                    optimizer_mode="exact",
                    objective="ir-inst-count",
                    max_rounds=2,
                    beam_width=8,
                    max_states=5000,
                    max_batches_per_state=20,
                    batch_frontier_policy="score",
                    validate_batches=True,
                    jobs=1,
                    timeout=1,
                    max_pairs=600,
                )

            runs = _read_csv(root / "out" / "passset_smoke_runs.csv")
            v1_audit_exists = (root / "out" / "branch" / "core_passes_v1" / "audit" / "resolved_passes.yaml").exists()
            v2_optimize_exists = (root / "out" / "branch" / "scalar_passes_v2" / "optimize" / "chosen_path_summary.csv").exists()

        self.assertEqual(result["runs"], 2)
        self.assertEqual({row["passset"] for row in runs}, {"core_passes_v1", "scalar_passes_v2"})
        self.assertTrue(v1_audit_exists)
        self.assertTrue(v2_optimize_exists)

    def test_continue_on_error_records_failed_run_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "branch.c"
            v1 = root / "core_passes_v1.yaml"
            v2 = root / "scalar_passes_v2.yaml"
            input_path.write_text("int f(void){return 0;}\n", encoding="utf-8")
            v1.write_text("passes:\n  - instcombine\n", encoding="utf-8")
            v2.write_text("passes:\n  - simplifycfg\n", encoding="utf-8")

            def fake_optimizer(input_path, out_dir, passes_path, **kwargs):
                if "scalar_passes_v2" in str(out_dir):
                    raise RuntimeError("optimizer boom")
                return _fake_optimize(input_path, out_dir, passes_path, **kwargs)

            with mock.patch("phasebatch.passset_smoke.run_pass_audit", side_effect=_fake_audit), \
                mock.patch("phasebatch.passset_smoke.run_optimizer", side_effect=fake_optimizer), \
                mock.patch("phasebatch.passset_smoke.run_baseline_comparison", side_effect=_fake_compare):
                result = run_passset_smoke(
                    [str(input_path)],
                    [v1, v2],
                    root / "out",
                    optimizer_mode="exact",
                    objective="ir-inst-count",
                    max_rounds=2,
                    beam_width=8,
                    max_states=5000,
                    max_batches_per_state=20,
                    batch_frontier_policy="score",
                    validate_batches=True,
                    jobs=1,
                    timeout=1,
                    max_pairs=600,
                    continue_on_error=True,
                )

            runs = _read_csv(root / "out" / "passset_smoke_runs.csv")

        self.assertEqual(result["successes"], 1)
        self.assertEqual(result["failures"], 1)
        self.assertEqual([row["optimize_status"] for row in runs], ["success", "failed"])
        self.assertIn("optimizer boom", runs[1]["error_message"])

    def test_audit_failure_marks_optimize_not_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "branch.c"
            v1 = root / "core_passes_v1.yaml"
            input_path.write_text("int f(void){return 0;}\n", encoding="utf-8")
            v1.write_text("passes:\n  - instcombine\n", encoding="utf-8")

            with mock.patch("phasebatch.passset_smoke.run_pass_audit", side_effect=RuntimeError("audit boom")), \
                mock.patch("phasebatch.passset_smoke.run_optimizer") as fake_optimizer:
                run_passset_smoke(
                    [str(input_path)],
                    [v1],
                    root / "out",
                    optimizer_mode="exact",
                    objective="ir-inst-count",
                    max_rounds=2,
                    beam_width=8,
                    max_states=5000,
                    max_batches_per_state=20,
                    batch_frontier_policy="score",
                    validate_batches=True,
                    jobs=1,
                    timeout=1,
                    max_pairs=600,
                    continue_on_error=True,
                )

            runs = _read_csv(root / "out" / "passset_smoke_runs.csv")

        fake_optimizer.assert_not_called()
        self.assertEqual(runs[0]["audit_status"], "failed")
        self.assertEqual(runs[0]["optimize_status"], "not_run")
        self.assertIn("audit boom", runs[0]["error_message"])

    def test_comparison_contains_v1_v2_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "branch.c"
            v1 = root / "core_passes_v1.yaml"
            v2 = root / "scalar_passes_v2.yaml"
            input_path.write_text("int f(void){return 0;}\n", encoding="utf-8")
            v1.write_text("passes:\n  - instcombine\n", encoding="utf-8")
            v2.write_text("passes:\n  - instcombine\n  - simplifycfg\n", encoding="utf-8")

            with mock.patch("phasebatch.passset_smoke.run_pass_audit", side_effect=_fake_audit), \
                mock.patch("phasebatch.passset_smoke.run_optimizer", side_effect=_fake_optimize), \
                mock.patch("phasebatch.passset_smoke.run_baseline_comparison", side_effect=_fake_compare):
                run_passset_smoke(
                    [str(input_path)],
                    [v1, v2],
                    root / "out",
                    optimizer_mode="exact",
                    objective="ir-inst-count",
                    max_rounds=2,
                    beam_width=8,
                    max_states=5000,
                    max_batches_per_state=20,
                    batch_frontier_policy="score",
                    validate_batches=True,
                    jobs=1,
                    timeout=1,
                    max_pairs=600,
                )

            comparison = _read_csv(root / "out" / "passset_comparison.csv")
            metrics = {row["metric"]: row for row in comparison}

        self.assertIn("valid_passes", metrics)
        self.assertEqual(metrics["valid_passes"]["v1_value"], "3")
        self.assertEqual(metrics["valid_passes"]["v2_value"], "5")
        self.assertEqual(metrics["active_passes_depth0"]["delta"], "2")

    def test_summary_is_generated_with_required_sections_and_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "branch.c"
            v1 = root / "core_passes_v1.yaml"
            v2 = root / "scalar_passes_v2.yaml"
            input_path.write_text("int f(void){return 0;}\n", encoding="utf-8")
            v1.write_text("passes:\n  - instcombine\n", encoding="utf-8")
            v2.write_text("passes:\n  - simplifycfg\n", encoding="utf-8")

            with mock.patch("phasebatch.passset_smoke.run_pass_audit", side_effect=_fake_audit), \
                mock.patch("phasebatch.passset_smoke.run_optimizer", side_effect=_fake_optimize), \
                mock.patch("phasebatch.passset_smoke.run_baseline_comparison", side_effect=_fake_compare):
                run_passset_smoke(
                    [str(input_path)],
                    [v1, v2],
                    root / "out",
                    optimizer_mode="exact",
                    objective="ir-inst-count",
                    max_rounds=2,
                    beam_width=8,
                    max_states=5000,
                    max_batches_per_state=20,
                    batch_frontier_policy="score",
                    validate_batches=True,
                    jobs=1,
                    timeout=1,
                    max_pairs=600,
                )

            summary = (root / "out" / "passset_smoke_summary.md").read_text(encoding="utf-8")

        self.assertIn("# Pass Set Smoke Summary", summary)
        self.assertIn("## Depth-0 Relation Changes", summary)
        self.assertIn("## Batch Changes", summary)
        self.assertIn("Adding passes may increase active pairs and validation cost. Objective values are evaluation signals, not commutation proof.", summary)

    def test_dropped_active_passes_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "branch.c"
            v1 = root / "core_passes_v1.yaml"
            v2 = root / "scalar_passes_v2.yaml"
            input_path.write_text("int f(void){return 0;}\n", encoding="utf-8")
            v1.write_text("passes:\n  - instcombine\n", encoding="utf-8")
            v2.write_text("passes:\n  - simplifycfg\n", encoding="utf-8")

            with mock.patch("phasebatch.passset_smoke.run_pass_audit", side_effect=_fake_audit), \
                mock.patch("phasebatch.passset_smoke.run_optimizer", side_effect=_fake_optimize), \
                mock.patch("phasebatch.passset_smoke.run_baseline_comparison", side_effect=_fake_compare):
                run_passset_smoke(
                    [str(input_path)],
                    [v1, v2],
                    root / "out",
                    optimizer_mode="exact",
                    objective="ir-inst-count",
                    max_rounds=2,
                    beam_width=8,
                    max_states=5000,
                    max_batches_per_state=20,
                    batch_frontier_policy="score",
                    validate_batches=True,
                    jobs=1,
                    timeout=1,
                    max_pairs=600,
                )

            comparison = _read_csv(root / "out" / "passset_comparison.csv")
            dropped = next(row for row in comparison if row["metric"] == "dropped_active_passes")
            runs = _read_csv(root / "out" / "passset_smoke_runs.csv")

        self.assertEqual(dropped["v1_value"], "0")
        self.assertEqual(dropped["v2_value"], "1")
        self.assertIn("active_passes_depth0", runs[0])


def _fake_audit(input_path: Path, passes_path: Path, out_dir: Path, **kwargs) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    valid = 5 if "v2" in Path(passes_path).stem else 3
    invalid = 1 if "v2" in Path(passes_path).stem else 0
    rows = [
        {
            "pass": f"p{index}",
            "category": "scalar",
            "stage": "middleend",
            "enabled": "true",
            "candidate_index": "0",
            "candidate_pipeline": f"p{index}",
            "resolved_pipeline": f"p{index}",
            "recognized_by_opt": "true",
            "valid_on_input": "true",
            "active_on_input": "true",
            "input_hash": "h0",
            "output_hash": f"h{index}",
            "ir_inst_before": "10",
            "ir_inst_after": "9",
            "inst_delta": "-1",
            "time_ms": "1",
            "failure_kind": "",
            "stderr_summary": "",
            "recommended_action": "keep",
        }
        for index in range(valid)
    ]
    _write_csv(out_dir / "pass_audit.csv", list(rows[0]), rows)
    _write_csv(out_dir / "invalid_passes.csv", ["pass", "category", "stage", "attempted_candidates", "failure_kind", "stderr_summary"], [{"pass": "bad", "category": "", "stage": "", "attempted_candidates": "bad", "failure_kind": "failed", "stderr_summary": "bad"}] * invalid)
    (out_dir / "resolved_passes.yaml").write_text("passes:\n  - name: instcombine\n    pipeline: instcombine\n", encoding="utf-8")
    return {"valid_passes": valid, "invalid_passes": invalid, "resolved_passes_yaml": str(out_dir / "resolved_passes.yaml")}


def _fake_optimize(input_path: Path, out_dir: Path, passes_path: Path, **kwargs) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    is_v2 = "scalar_passes_v2" in str(out_dir)
    active = "6" if is_v2 else "4"
    pairs = "15" if is_v2 else "6"
    commute = "9" if is_v2 else "4"
    sensitive = "5" if is_v2 else "2"
    candidates = "7" if is_v2 else "3"
    final_inst = "8" if is_v2 else "10"
    dropped = "1" if is_v2 else "0"
    state_dir = out_dir / "states" / "S0000"
    state_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "states.csv", ["state_id", "is_duplicate"], [{"state_id": "S0000", "is_duplicate": "false"}, {"state_id": "S0001", "is_duplicate": "false"}])
    _write_csv(out_dir / "batch_state_transitions.csv", ["parent_state_id", "child_state_id"], [{"parent_state_id": "S0000", "child_state_id": "S0001"}])
    _write_csv(state_dir / "per_state_summary.csv", ["active_passes", "pairs_tested", "dynamic_commute", "order_sensitive", "unknown"], [{"active_passes": active, "pairs_tested": pairs, "dynamic_commute": commute, "order_sensitive": sensitive, "unknown": "0"}])
    _write_csv(state_dir / "batch_summary.csv", ["batch_candidates", "batch_reduction_estimate"], [{"batch_candidates": candidates, "batch_reduction_estimate": "12"}])
    _write_csv(
        state_dir / "batch_correctness.csv",
        ["batch_id", "correctness_class", "validation_status", "can_execute"],
        [
            {"batch_id": "B0000", "correctness_class": "certified_batch", "validation_status": "all_permutations_same", "can_execute": "true"},
            {"batch_id": "B0001", "correctness_class": "sampled_batch", "validation_status": "sampled_same", "can_execute": "false"},
        ],
    )
    _write_csv(state_dir / "coverage_summary.csv", ["dropped_active_passes"], [{"dropped_active_passes": dropped}])
    _write_csv(
        out_dir / "chosen_path_summary.csv",
        ["selected_final_state", "root_ir_inst_count", "final_ir_inst_count", "total_ir_inst_delta"],
        [{"selected_final_state": "S0001", "root_ir_inst_count": "12", "final_ir_inst_count": final_inst, "total_ir_inst_delta": str(int(final_inst) - 12)}],
    )
    (out_dir / "optimized_pipeline_names.txt").write_text("instcombine,simplifycfg\n" if is_v2 else "instcombine\n", encoding="utf-8")
    return {"states": 2, "selected_final_state": "S0001"}


def _fake_compare(run_dir: Path, passes_path: Path, **kwargs) -> dict:
    run_dir = Path(run_dir)
    rows = [
        {
            "program": "",
            "method": "batch_optimizer",
            "status": "success",
            "final_ir_inst_count": "8",
            "root_ir_inst_count": "12",
            "ir_inst_delta": "-4",
            "ir_inst_reduction_pct": "33.33",
            "states_evaluated": "2",
            "opt_runs": "3",
            "final_sequence_length": "2",
            "pass_sequence": "instcombine",
            "time_ms": "1",
            "error_message": "",
        }
    ]
    _write_csv(run_dir / "baseline_results.csv", list(rows[0]), rows)
    (run_dir / "baselines").mkdir(exist_ok=True)
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
