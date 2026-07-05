import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.exact_reference import select_and_run_exact_reference


class ExactReferenceTests(unittest.TestCase):
    def test_selects_easy_medium_and_hard_and_runs_exact_only_for_selected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            budgeted = root / "budgeted"
            _write_budgeted_study(
                budgeted,
                [
                    _program("easy", states=2, time_ms=100, reduction=1.0, sensitive=1, tested=10, batch_inst=10, greedy_inst=10),
                    _program("medium", states=20, time_ms=1000, reduction=3.0, sensitive=10, tested=40, batch_inst=20, greedy_inst=20),
                    _program("hard", states=80, time_ms=9000, reduction=8.0, sensitive=70, tested=100, batch_inst=50, greedy_inst=40, random_inst=39),
                    _program("spare", states=30, time_ms=2000, reduction=4.0, sensitive=20, tested=60, batch_inst=30, greedy_inst=30),
                ],
            )
            out_dir = root / "exact_ref"
            passes = root / "passes.yaml"
            passes.write_text("passes:\n  - instcombine\n", encoding="utf-8")
            calls: list[Path] = []

            def fake_optimizer(input_path, out_dir, passes_path, **kwargs):
                calls.append(Path(input_path))
                final = {"easy": "9", "medium": "19", "hard": "38"}[Path(input_path).stem]
                return _fake_exact_run(Path(input_path).stem, Path(out_dir), final_inst=final, states=5)

            with mock.patch("phasebatch.exact_reference.run_optimizer", side_effect=fake_optimizer):
                result = select_and_run_exact_reference(
                    budgeted,
                    out_dir,
                    passes,
                    objective="ir-inst-count",
                    max_rounds=4,
                    max_states=5000,
                    validate_batches=True,
                    jobs=1,
                    timeout=1,
                    max_pairs=20,
                    num_easy=1,
                    num_medium=1,
                    num_hard=1,
                )

            selection = _read_csv(out_dir / "exact_reference_selection.csv")
            runs = _read_csv(out_dir / "exact_reference_runs.csv")

        self.assertEqual(result["selected_programs"], 3)
        self.assertEqual([(row["program"], row["category"]) for row in selection], [("easy", "easy"), ("medium", "medium"), ("hard", "hard")])
        self.assertEqual([path.stem for path in calls], ["easy", "medium", "hard"])
        self.assertEqual({row["status"] for row in runs}, {"success"})

    def test_results_join_budgeted_exact_and_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            budgeted = root / "budgeted"
            _write_budgeted_study(
                budgeted,
                [_program("case", states=5, time_ms=500, reduction=2.0, sensitive=2, tested=10, batch_inst=10, greedy_inst=9, random_inst=12, config_inst=11)],
            )
            out_dir = root / "exact_ref"
            passes = root / "passes.yaml"
            passes.write_text("passes:\n  - instcombine\n", encoding="utf-8")

            with mock.patch(
                "phasebatch.exact_reference.run_optimizer",
                side_effect=lambda input_path, out_dir, passes_path, **kwargs: _fake_exact_run("case", Path(out_dir), final_inst="8", states=20),
            ):
                select_and_run_exact_reference(
                    budgeted,
                    out_dir,
                    passes,
                    objective="ir-inst-count",
                    max_rounds=4,
                    max_states=5000,
                    validate_batches=True,
                    jobs=1,
                    timeout=1,
                    max_pairs=20,
                    num_easy=1,
                    num_medium=0,
                    num_hard=0,
                )

            result_row = _read_csv(out_dir / "exact_reference_results.csv")[0]

        self.assertEqual(result_row["budgeted_inst"], "10")
        self.assertEqual(result_row["exact_inst"], "8")
        self.assertEqual(result_row["gap_budgeted_to_exact"], "2")
        self.assertEqual(result_row["greedy_inst"], "9")
        self.assertEqual(result_row["random_best_inst"], "12")
        self.assertEqual(result_row["config_order_inst"], "11")
        self.assertEqual(result_row["exact_vs_greedy"], "win")
        self.assertEqual(result_row["exact_vs_random"], "win")
        self.assertEqual(result_row["budgeted_matched_exact"], "false")

    def test_insufficient_candidates_records_warning_and_summary_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            budgeted = root / "budgeted"
            _write_budgeted_study(
                budgeted,
                [_program("only", states=3, time_ms=200, reduction=1.0, sensitive=1, tested=10, batch_inst=7, greedy_inst=7)],
            )
            out_dir = root / "exact_ref"
            passes = root / "passes.yaml"
            passes.write_text("passes:\n  - instcombine\n", encoding="utf-8")
            warnings: list[str] = []

            with mock.patch(
                "phasebatch.exact_reference.run_optimizer",
                side_effect=lambda input_path, out_dir, passes_path, **kwargs: _fake_exact_run("only", Path(out_dir), final_inst="7", states=4),
            ):
                select_and_run_exact_reference(
                    budgeted,
                    out_dir,
                    passes,
                    objective="ir-inst-count",
                    max_rounds=4,
                    max_states=5000,
                    validate_batches=True,
                    jobs=1,
                    timeout=1,
                    max_pairs=20,
                    num_easy=2,
                    num_medium=2,
                    num_hard=2,
                    warn=warnings.append,
                )

            selection = _read_csv(out_dir / "exact_reference_selection.csv")
            failures = _read_csv(out_dir / "failures.csv")
            summary = (out_dir / "exact_reference_summary.md").read_text(encoding="utf-8")

        self.assertEqual(len(selection), 1)
        self.assertTrue(warnings)
        self.assertEqual(failures[0]["stage"], "selection")
        self.assertIn("not enough candidates", failures[0]["error_message"])
        self.assertIn("Exact reference is complete only within the current certified batch-state graph", summary)

    def test_continue_on_error_records_exact_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            budgeted = root / "budgeted"
            _write_budgeted_study(
                budgeted,
                [_program("bad", states=2, time_ms=100, reduction=1.0, sensitive=1, tested=10, batch_inst=10, greedy_inst=10)],
            )
            out_dir = root / "exact_ref"
            passes = root / "passes.yaml"
            passes.write_text("passes:\n  - instcombine\n", encoding="utf-8")

            with mock.patch("phasebatch.exact_reference.run_optimizer", side_effect=RuntimeError("exact boom")):
                result = select_and_run_exact_reference(
                    budgeted,
                    out_dir,
                    passes,
                    objective="ir-inst-count",
                    max_rounds=4,
                    max_states=5000,
                    validate_batches=True,
                    jobs=1,
                    timeout=1,
                    max_pairs=20,
                    num_easy=1,
                    num_medium=0,
                    num_hard=0,
                    continue_on_error=True,
                )

            runs = _read_csv(out_dir / "exact_reference_runs.csv")
            failures = _read_csv(out_dir / "failures.csv")

        self.assertEqual(result["failures"], 1)
        self.assertEqual(runs[0]["status"], "failed")
        self.assertEqual(failures[0]["stage"], "exact")
        self.assertIn("exact boom", failures[0]["error_message"])


def _program(
    name: str,
    *,
    states: int,
    time_ms: int,
    reduction: float,
    sensitive: int,
    tested: int,
    batch_inst: int,
    greedy_inst: int,
    random_inst: int | None = None,
    config_inst: int | None = None,
) -> dict:
    return {
        "program": name,
        "input_path": f"{name}.c",
        "states": str(states),
        "time_ms": str(time_ms),
        "max_local_reduction_log10": str(reduction),
        "total_order_sensitive_pairs": str(sensitive),
        "total_tested_pairs": str(tested),
        "batch_inst": str(batch_inst),
        "greedy_inst": str(greedy_inst),
        "random_inst": str(random_inst if random_inst is not None else greedy_inst),
        "config_inst": str(config_inst if config_inst is not None else batch_inst),
    }


def _write_budgeted_study(root: Path, programs: list[dict]) -> None:
    root.mkdir(parents=True)
    input_dir = root / "inputs"
    input_dir.mkdir()
    runs = []
    methods = []
    reductions = []
    evidence = []
    for program in programs:
        input_path = input_dir / program["input_path"]
        input_path.write_text("int f(void){return 0;}\n", encoding="utf-8")
        runs.append(
            {
                "program": program["program"],
                "input_path": str(input_path),
                "status": "success",
                "final_ir_inst_count": program["batch_inst"],
                "states_reached": program["states"],
                "transitions": str(max(0, int(program["states"]) - 1)),
                "pipeline_length": "2",
                "time_ms": program["time_ms"],
            }
        )
        for method, inst in [
            ("batch_optimizer", program["batch_inst"]),
            ("greedy_single_pass", program["greedy_inst"]),
            ("random_single_pass_best", program["random_inst"]),
            ("config_order_once", program["config_inst"]),
        ]:
            methods.append(
                {
                    "program": program["program"],
                    "method": method,
                    "status": "success",
                    "final_ir_inst_count": inst,
                }
            )
        reductions.append(
            {
                "program": program["program"],
                "states": program["states"],
                "total_tested_pairs": program["total_tested_pairs"],
                "total_order_sensitive_pairs": program["total_order_sensitive_pairs"],
                "max_local_reduction_log10": program["max_local_reduction_log10"],
                "total_dropped_active_passes": "0",
            }
        )
        evidence.append(
            {
                "program": program["program"],
                "selected_path_batches": "2",
                "selected_strong_certificates": "2",
                "selected_weak_certificates": "0",
                "dropped_active_passes": "0",
            }
        )
    _write_csv(root / "budgeted_study_runs.csv", list(runs[0]), runs)
    _write_csv(root / "budgeted_study_methods.csv", list(methods[0]), methods)
    _write_csv(root / "budgeted_study_reduction.csv", list(reductions[0]), reductions)
    _write_csv(root / "budgeted_study_evidence.csv", list(evidence[0]), evidence)


def _fake_exact_run(program: str, out_dir: Path, *, final_inst: str, states: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        out_dir / "states.csv",
        ["program", "state_id"],
        [{"program": program, "state_id": f"S{i:04d}"} for i in range(states)],
    )
    _write_csv(
        out_dir / "batch_state_transitions.csv",
        ["program", "parent_state_id", "child_state_id"],
        [{"program": program, "parent_state_id": "S0000", "child_state_id": "S0001"}],
    )
    _write_csv(
        out_dir / "chosen_path_summary.csv",
        ["program", "final_ir_inst_count"],
        [{"program": program, "final_ir_inst_count": final_inst}],
    )
    _write_csv(out_dir / "optimizer_timing.csv", ["optimizer_total_time_ms"], [{"optimizer_total_time_ms": "1000"}])
    (out_dir / "exact_status.txt").write_text("exact_complete\n", encoding="utf-8")
    return {"exact_status": "exact_complete", "states": states, "batch_transitions": 1, "final_objective_value": final_inst}


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
