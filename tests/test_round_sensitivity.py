import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.round_sensitivity import generate_round_sensitivity, run_round_sensitivity


class RoundSensitivityTests(unittest.TestCase):
    def test_generate_round_sensitivity_reads_existing_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            r2 = _make_opt_run(root / "round_2", 2, states=22, transitions=22, final_inst=223, pipeline="a,b,c,d,e,f", final_state="S0014", exact_status="exact_complete", leaf_reason="max_rounds_reached")
            r3 = _make_opt_run(root / "round_3", 3, states=44, transitions=53, final_inst=214, pipeline="a,b,c,d,e,f,g,h,i", final_state="S0030", exact_status="exact_complete", leaf_reason="max_rounds_reached")
            r4 = _make_opt_run(root / "round_4", 4, states=61, transitions=76, final_inst=211, pipeline="a,b,c,d,e,f,g,h,i", final_state="S0049", exact_status="exact_complete", leaf_reason="no_active_passes")

            result = generate_round_sensitivity([r2, r3, r4], root / "report", input_label="n-body.c", passes_label="core_passes.yaml")

            rows = _read_csv(Path(result["round_sensitivity_csv"]))
            summary = Path(result["round_sensitivity_md"]).read_text(encoding="utf-8")

        self.assertEqual([row["max_rounds"] for row in rows], ["2", "3", "4"])
        self.assertEqual(rows[1]["states_reached"], "44")
        self.assertEqual(rows[1]["transitions"], "53")
        self.assertEqual(rows[1]["final_inst"], "214")
        self.assertEqual(rows[1]["pipeline_len"], "9")
        self.assertEqual(rows[1]["exact_status"], "exact_complete")
        self.assertEqual(rows[1]["selected_final_state"], "S0030")
        self.assertEqual(rows[1]["truncated"], "true")
        self.assertEqual(rows[2]["truncated"], "false")
        self.assertIn("# Round Sensitivity Report", summary)
        self.assertIn("batch-state exact search convergence curve", summary)

    def test_selected_max_round_state_reports_remaining_active_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            r3 = _make_opt_run(
                root / "round_3",
                3,
                states=44,
                transitions=53,
                final_inst=214,
                pipeline="a,b,c,d,e,f,g,h,i",
                final_state="S0030",
                exact_status="exact_complete",
                leaf_reason="max_rounds_reached",
                remaining_active_passes=["early-cse", "gvn"],
            )

            result = generate_round_sensitivity([r3], root / "report")

            rows = _read_csv(Path(result["round_sensitivity_csv"]))
            summary = Path(result["round_sensitivity_md"]).read_text(encoding="utf-8")

        self.assertEqual(rows[0]["selected_final_state_is_terminal"], "false")
        self.assertEqual(rows[0]["selected_final_state_stop_reason"], "max_rounds_reached")
        self.assertEqual(rows[0]["selected_final_state_truncated"], "true")
        self.assertEqual(rows[0]["remaining_active_pass_count"], "2")
        self.assertEqual(rows[0]["remaining_active_passes"], "early-cse;gvn")
        self.assertEqual(rows[0]["remaining_executable_batches"], "not_evaluated_at_terminal_depth")
        self.assertIn("remaining active passes", summary)
        self.assertIn("early-cse;gvn", summary)

    def test_run_round_sensitivity_runs_optimizer_for_each_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            input_path = root / "input.c"
            passes_path = root / "passes.yaml"
            input_path.write_text("int main(void){return 0;}\n", encoding="utf-8")
            passes_path.write_text("passes:\n  - mem2reg\n", encoding="utf-8")

            def fake_optimize(input_arg, out_arg, passes_arg, **kwargs):
                max_rounds = kwargs["max_rounds"]
                _make_opt_run(Path(out_arg), max_rounds, states=max_rounds + 1, transitions=max_rounds * 2, final_inst=10 - max_rounds, pipeline="mem2reg", final_state=f"S{max_rounds:04d}", exact_status="exact_complete", leaf_reason="no_active_passes")
                return {"out_dir": str(out_arg), "states": max_rounds + 1}

            with mock.patch("phasebatch.round_sensitivity._optimize_batches", side_effect=fake_optimize) as fake:
                result = run_round_sensitivity(
                    input_path,
                    out_dir,
                    passes_path,
                    rounds=[1, 2],
                    optimizer_mode="exact",
                    objective="ir-inst-count",
                    beam_width=8,
                    max_states=100,
                    max_batches_per_state=20,
                    batch_frontier_policy=None,
                    validate_batches=True,
                    jobs=4,
                    timeout=5,
                    max_pairs=50,
                )

            rows = _read_csv(Path(result["round_sensitivity_csv"]))
            summary_exists = Path(result["round_sensitivity_md"]).exists()

        self.assertEqual(fake.call_count, 2)
        self.assertEqual(fake.call_args_list[0].kwargs["max_rounds"], 1)
        self.assertEqual(fake.call_args_list[0].kwargs["validate_batches"], True)
        self.assertEqual(rows[0]["max_rounds"], "1")
        self.assertEqual(rows[1]["max_rounds"], "2")
        self.assertTrue(summary_exists)


def _make_opt_run(
    run_dir: Path,
    max_rounds: int,
    *,
    states: int,
    transitions: int,
    final_inst: int,
    pipeline: str,
    final_state: str,
    exact_status: str,
    leaf_reason: str,
    remaining_active_passes: list[str] | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        run_dir / "states.csv",
        ["state_id"],
        [{"state_id": f"S{i:04d}"} for i in range(states)],
    )
    _write_csv(
        run_dir / "batch_state_transitions.csv",
        ["parent_state_id", "child_state_id"],
        [{"parent_state_id": "S0000", "child_state_id": f"S{i + 1:04d}"} for i in range(transitions)],
    )
    _write_csv(
        run_dir / "chosen_path_summary.csv",
        [
            "selected_final_state",
            "final_ir_inst_count",
            "total_pass_invocations",
        ],
        [
            {
                "selected_final_state": final_state,
                "final_ir_inst_count": str(final_inst),
                "total_pass_invocations": str(len([part for part in pipeline.split(",") if part])),
            }
        ],
    )
    _write_csv(
        run_dir / "leaf_states.csv",
        ["state_id", "selected_as_final", "leaf_reason"],
        [{"state_id": final_state, "selected_as_final": "true", "leaf_reason": leaf_reason}],
    )
    if remaining_active_passes is not None:
        state_dir = run_dir / "states" / final_state
        rows = [
            {"pass": pass_name, "success": "true", "active": "true"}
            for pass_name in remaining_active_passes
        ]
        rows.append({"pass": "mem2reg", "success": "true", "active": "false"})
        _write_csv(state_dir / "pass_profile.csv", ["pass", "success", "active"], rows)
    _write_csv(
        run_dir / "optimizer_timing.csv",
        ["optimizer_total_time_ms"],
        [{"optimizer_total_time_ms": str(max_rounds * 100)}],
    )
    (run_dir / "optimized_pipeline.txt").write_text(pipeline + "\n", encoding="utf-8")
    (run_dir / "exact_status.txt").write_text(exact_status + "\n", encoding="utf-8")
    (run_dir / "final_state.txt").write_text(f"selected_final_state={final_state}\nfinal_objective={final_inst}\n", encoding="utf-8")
    return run_dir


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
