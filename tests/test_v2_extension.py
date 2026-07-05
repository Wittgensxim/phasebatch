import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.v2_extension import run_v2_extension_study


class V2ExtensionStudyTests(unittest.TestCase):
    def test_creates_scalar_passes_v2_config_if_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = _make_input(root, "case.c")
            v1 = root / "core_passes.yaml"
            v2 = root / "scalar_passes_v2.yaml"
            v1.write_text("passes:\n  - instcombine\n  - simplifycfg\n", encoding="utf-8")

            with _patched_successful_study():
                run_v2_extension_study(
                    [str(input_path)],
                    root / "out",
                    v1,
                    v2,
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
                    random_trials=3,
                    seed=0,
                )

            text = v2.read_text(encoding="utf-8")

        self.assertIn("name: sccp", text)
        self.assertIn("name: dse", text)
        self.assertIn("name: memcpyopt", text)
        self.assertIn("name: sink", text)
        self.assertIn("name: tailcallelim", text)

    def test_v2_comparison_works_on_mock_v1_v2_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = _make_input(root, "case.c")
            v1, v2 = _make_pass_configs(root)

            with _patched_successful_study():
                result = run_v2_extension_study(
                    [str(input_path)],
                    root / "out",
                    v1,
                    v2,
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
                    random_trials=3,
                    seed=0,
                )

            comparison = _read_csv(root / "out" / "v2_extension_comparison.csv")
            runs = _read_csv(root / "out" / "v2_extension_runs.csv")
            row = comparison[0]

        self.assertEqual(result["programs"], 1)
        self.assertEqual(row["v1_valid_passes"], "14")
        self.assertEqual(row["v2_valid_passes"], "19")
        self.assertEqual(row["v1_active_depth0"], "4")
        self.assertEqual(row["v2_active_depth0"], "7")
        self.assertEqual(row["v2_minus_v1_final_inst"], "-2")
        self.assertEqual({r["passset"] for r in runs}, {"v1", "v2"})

    def test_missing_audit_falls_back_to_v2_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = _make_input(root, "case.c")
            v1, v2 = _make_pass_configs(root)

            with mock.patch("phasebatch.v2_extension.run_optimizer", side_effect=_fake_optimizer), \
                mock.patch("phasebatch.v2_extension.run_pass_audit", side_effect=NotImplementedError("no audit")):
                result = run_v2_extension_study(
                    [str(input_path)],
                    root / "out",
                    v1,
                    v2,
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
                    random_trials=3,
                    seed=0,
                    continue_on_error=True,
                )

            failures = _read_csv(root / "out" / "failures.csv")
            runs = _read_csv(root / "out" / "v2_extension_runs.csv")

        self.assertEqual(result["successes"], 1)
        self.assertEqual(failures[0]["stage"], "audit")
        self.assertIn("no audit", failures[0]["error_message"])
        self.assertEqual([row for row in runs if row["passset"] == "v2" and row["stage"] == "optimize"][0]["status"], "success")

    def test_dropped_active_pass_warning_appears_in_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = _make_input(root, "case.c")
            v1, v2 = _make_pass_configs(root)

            with _patched_successful_study(dropped_v2="2"):
                run_v2_extension_study(
                    [str(input_path)],
                    root / "out",
                    v1,
                    v2,
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
                    random_trials=3,
                    seed=0,
                )

            comparison = _read_csv(root / "out" / "v2_extension_comparison.csv")
            summary = (root / "out" / "v2_extension_summary.md").read_text(encoding="utf-8")

        self.assertEqual(comparison[0]["v2_dropped_active_passes"], "2")
        self.assertIn("investigate v2 failures", summary)
        self.assertIn("dropped active passes", summary)

    def test_summary_contains_correctness_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = _make_input(root, "case.c")
            v1, v2 = _make_pass_configs(root)

            with _patched_successful_study():
                run_v2_extension_study(
                    [str(input_path)],
                    root / "out",
                    v1,
                    v2,
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
                    random_trials=3,
                    seed=0,
                )

            summary = (root / "out" / "v2_extension_summary.md").read_text(encoding="utf-8")

        self.assertIn("# V2 Scalar Pass Set Extension Summary", summary)
        self.assertIn("V2 is a scalability extension.", summary)
        self.assertIn("Adding passes changes the explored search space. It does not change the rule that only certified/executable batches may be hard-folded.", summary)

    def test_invalid_pass_program_count_is_not_double_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = _make_input(root, "case.c")
            v1, v2 = _make_pass_configs(root)

            with mock.patch("phasebatch.v2_extension.run_optimizer", side_effect=_fake_optimizer), \
                mock.patch("phasebatch.v2_extension.run_pass_audit", side_effect=_fake_audit_with_invalid):
                run_v2_extension_study(
                    [str(input_path)],
                    root / "out",
                    v1,
                    v2,
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
                    random_trials=3,
                    seed=0,
                )

            summary = (root / "out" / "v2_extension_summary.md").read_text(encoding="utf-8")

        self.assertIn("invalid-pass programs=1", summary)


def _patched_successful_study(dropped_v2: str = "0"):
    return mock.patch.multiple(
        "phasebatch.v2_extension",
        run_optimizer=mock.Mock(side_effect=lambda input_path, out_dir, passes_path, **kwargs: _fake_optimizer(input_path, out_dir, passes_path, dropped_v2=dropped_v2, **kwargs)),
        run_pass_audit=mock.Mock(side_effect=_fake_audit),
    )


def _make_input(root: Path, name: str) -> Path:
    path = root / name
    path.write_text("int f(void){return 0;}\n", encoding="utf-8")
    return path


def _make_pass_configs(root: Path) -> tuple[Path, Path]:
    v1 = root / "core_passes.yaml"
    v2 = root / "scalar_passes_v2.yaml"
    v1.write_text("passes:\n  - instcombine\n", encoding="utf-8")
    v2.write_text("passes:\n  - instcombine\n  - sccp\n  - dse\n", encoding="utf-8")
    return v1, v2


def _fake_audit(input_path: Path, passes_path: Path, out_dir: Path, **kwargs) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "pass_audit.csv", ["pass", "valid_on_input"], [{"pass": "p", "valid_on_input": "true"}])
    _write_csv(out_dir / "invalid_passes.csv", ["pass"], [])
    resolved = out_dir / "resolved_passes.yaml"
    resolved.write_text("passes:\n  - instcombine\n", encoding="utf-8")
    return {"valid_passes": 19, "invalid_passes": 0, "resolved_passes_yaml": str(resolved)}


def _fake_audit_with_invalid(input_path: Path, passes_path: Path, out_dir: Path, **kwargs) -> dict:
    result = _fake_audit(input_path, passes_path, out_dir, **kwargs)
    result["invalid_passes"] = 1
    _write_csv(out_dir / "invalid_passes.csv", ["pass"], [{"pass": "bad"}])
    return result


def _fake_optimizer(input_path: Path, out_dir: Path, passes_path: Path, dropped_v2: str = "0", **kwargs) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    is_v2 = "v2" in str(out_dir).replace("\\", "/")
    state_dir = out_dir / "states" / "S0000"
    state_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "valid": "19" if is_v2 else "14",
        "invalid": "0",
        "active": "7" if is_v2 else "4",
        "pairs": "21" if is_v2 else "6",
        "commute": "14" if is_v2 else "4",
        "sensitive": "7" if is_v2 else "2",
        "candidates": "8" if is_v2 else "3",
        "final": "8" if is_v2 else "10",
        "states": "5" if is_v2 else "3",
        "time": "200" if is_v2 else "100",
        "dropped": dropped_v2 if is_v2 else "0",
    }
    _write_csv(out_dir / "valid_passes.csv", ["pass"], [{"pass": f"p{i}"} for i in range(int(metrics["valid"]))])
    _write_csv(out_dir / "invalid_passes.csv", ["pass"], [{"pass": f"bad{i}"} for i in range(int(metrics["invalid"]))])
    _write_csv(
        out_dir / "states.csv",
        ["state_id"],
        [{"state_id": f"S{i:04d}"} for i in range(int(metrics["states"]))],
    )
    _write_csv(out_dir / "batch_state_transitions.csv", ["parent_state_id", "child_state_id"], [{"parent_state_id": "S0000", "child_state_id": "S0001"}])
    _write_csv(
        state_dir / "per_state_summary.csv",
        ["active_passes", "pairs_tested", "dynamic_commute", "order_sensitive"],
        [{"active_passes": metrics["active"], "pairs_tested": metrics["pairs"], "dynamic_commute": metrics["commute"], "order_sensitive": metrics["sensitive"]}],
    )
    _write_csv(state_dir / "batch_summary.csv", ["batch_candidates"], [{"batch_candidates": metrics["candidates"]}])
    _write_csv(
        state_dir / "batch_correctness.csv",
        ["correctness_class", "can_execute"],
        [
            {"correctness_class": "certified_batch", "can_execute": "true"},
            {"correctness_class": "sampled_batch", "can_execute": "false"},
        ],
    )
    _write_csv(state_dir / "coverage_summary.csv", ["dropped_active_passes"], [{"dropped_active_passes": metrics["dropped"]}])
    _write_csv(
        out_dir / "chosen_path_summary.csv",
        ["selected_final_state", "final_ir_inst_count"],
        [{"selected_final_state": "S0001", "final_ir_inst_count": metrics["final"]}],
    )
    _write_csv(out_dir / "optimizer_timing.csv", ["optimizer_total_time_ms"], [{"optimizer_total_time_ms": metrics["time"]}])
    (out_dir / "optimized_pipeline.txt").write_text("instcombine\n", encoding="utf-8")
    return {"states": int(metrics["states"])}


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
