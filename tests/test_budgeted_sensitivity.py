import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.budgeted_sensitivity import run_budgeted_sensitivity


class BudgetedSensitivityTests(unittest.TestCase):
    def test_budgeted_sensitivity_runs_matrix_joins_reference_and_selects_best(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = _make_inputs(root, ["alpha.c", "beta.c"])
            passes = root / "passes.yaml"
            passes.write_text("passes:\n  - instcombine\n", encoding="utf-8")
            reference = root / "reference.csv"
            _write_csv(
                reference,
                ["program", "batch", "greedy", "random best", "config once", "batch states", "batch time ms"],
                [
                    {"program": "alpha", "batch": "88", "greedy": "91", "random best": "92", "config once": "93", "batch states": "200", "batch time ms": "1000"},
                    {"program": "beta", "batch": "79", "greedy": "82", "random best": "83", "config once": "84", "batch states": "300", "batch time ms": "1200"},
                ],
            )
            out_dir = root / "out"

            with mock.patch("phasebatch.budgeted_sensitivity.run_optimizer", side_effect=_fake_optimizer) as fake_run:
                result = run_budgeted_sensitivity(
                    [str(path) for path in inputs],
                    out_dir,
                    passes,
                    objective="ir-inst-count",
                    max_rounds=4,
                    beam_widths=[4, 8],
                    max_states_list=[100, 200],
                    max_batches_per_state=20,
                    batch_frontier_policy="score",
                    validate_batches=True,
                    jobs=1,
                    timeout=1,
                    max_pairs=300,
                    exact_reference=reference,
                    overwrite=False,
                    continue_on_error=False,
                )

            runs = _read_csv(out_dir / "budgeted_sensitivity_runs.csv")
            results = _read_csv(out_dir / "budgeted_sensitivity_results.csv")
            best = _read_csv(out_dir / "budgeted_sensitivity_best.csv")
            summary = (out_dir / "budgeted_sensitivity_summary.md").read_text(encoding="utf-8")

        self.assertEqual(result["attempted_runs"], 8)
        self.assertEqual(fake_run.call_count, 8)
        self.assertEqual(len(runs), 8)
        self.assertEqual(len(results), 8)
        self.assertEqual({row["exact_r4_inst"] for row in results if row["program"] == "alpha"}, {"88"})
        alpha_best = next(row for row in best if row["program"] == "alpha")
        self.assertEqual(alpha_best["best_beam_width"], "4")
        self.assertEqual(alpha_best["best_max_states"], "100")
        self.assertEqual(alpha_best["best_final_ir_inst_count"], "90")
        self.assertEqual(alpha_best["gap_to_exact"], "2")
        self.assertEqual(alpha_best["beat_greedy"], "true")
        self.assertIn("## Budgeted vs Exact", summary)
        self.assertIn("Budgeted mode changes search coverage, not batch correctness.", summary)

    def test_failures_and_missing_reference_do_not_crash_when_continuing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = _make_inputs(root, ["good.c"])[0]
            missing = root / "missing.c"
            passes = root / "passes.yaml"
            passes.write_text("passes:\n  - instcombine\n", encoding="utf-8")
            out_dir = root / "out"

            def fake_optimizer(input_path, out_dir, passes_path, **kwargs):
                if kwargs["beam_width"] == 8:
                    raise RuntimeError("beam failed")
                return _fake_optimizer(input_path, out_dir, passes_path, **kwargs)

            with mock.patch("phasebatch.budgeted_sensitivity.run_optimizer", side_effect=fake_optimizer):
                result = run_budgeted_sensitivity(
                    [str(good), str(missing)],
                    out_dir,
                    passes,
                    objective="ir-inst-count",
                    max_rounds=3,
                    beam_widths=[4, 8],
                    max_states_list=[50],
                    max_batches_per_state=10,
                    batch_frontier_policy="score",
                    validate_batches=True,
                    jobs=1,
                    timeout=1,
                    max_pairs=300,
                    exact_reference=root / "does_not_exist.csv",
                    overwrite=False,
                    continue_on_error=True,
                )

            runs = _read_csv(out_dir / "budgeted_sensitivity_runs.csv")
            failures = _read_csv(out_dir / "failures.csv")
            summary = (out_dir / "budgeted_sensitivity_summary.md").read_text(encoding="utf-8")

        self.assertEqual(result["failures"], 2)
        self.assertEqual(sum(1 for row in runs if row["status"] == "failed"), 2)
        self.assertEqual({row["stage"] for row in failures}, {"input", "optimize"})
        self.assertIn("Exact reference was not available.", summary)


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
    program = Path(input_path).stem
    beam = int(kwargs["beam_width"])
    max_states = int(kwargs["max_states"])
    out_dir.mkdir(parents=True, exist_ok=True)
    final_count = _final_count(program, beam, max_states)
    states = 50 if (beam, max_states) == (4, 100) else beam * 10 + max_states // 10
    transitions = states + 1
    time_ms = 10 if (beam, max_states) == (4, 100) else beam * 2 + max_states / 10
    _write_csv(
        out_dir / "chosen_path_summary.csv",
        [
            "selected_final_state",
            "root_ir_inst_count",
            "final_ir_inst_count",
            "total_ir_inst_delta",
            "ir_inst_reduction_pct",
        ],
        [
            {
                "selected_final_state": "S0001",
                "root_ir_inst_count": "100",
                "final_ir_inst_count": str(final_count),
                "total_ir_inst_delta": str(final_count - 100),
                "ir_inst_reduction_pct": "10.00",
            }
        ],
    )
    _write_csv(
        out_dir / "states.csv",
        ["program", "state_id", "is_duplicate"],
        [{"program": program, "state_id": f"S{i:04d}", "is_duplicate": "true" if i == 1 else "false"} for i in range(3)],
    )
    _write_csv(
        out_dir / "batch_state_transitions.csv",
        ["program", "parent_state_id", "child_state_id"],
        [{"program": program, "parent_state_id": "S0000", "child_state_id": "S0001"} for _ in range(2)],
    )
    _write_csv(
        out_dir / "optimizer_timing.csv",
        ["optimizer_total_time_ms"],
        [{"optimizer_total_time_ms": f"{time_ms:.3f}"}],
    )
    (out_dir / "optimized_pipeline.txt").write_text("mem2reg,gvn\n", encoding="utf-8")
    (out_dir / "exact_status.txt").write_text("budgeted\n", encoding="utf-8")
    _write_csv(
        out_dir / "leaf_states.csv",
        ["state_id", "leaf_reason"],
        [{"state_id": "S0001", "leaf_reason": "max_rounds_reached"}],
    )
    return {"states": states, "batch_transitions": transitions}


def _final_count(program: str, beam: int, max_states: int) -> int:
    if program == "alpha":
        if (beam, max_states) in {(4, 100), (8, 200)}:
            return 90
        return 94
    if beam == 8 and max_states == 200:
        return 80
    return 85


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
