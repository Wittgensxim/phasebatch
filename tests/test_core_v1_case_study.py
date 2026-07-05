import csv
import tempfile
import unittest
from pathlib import Path

from phasebatch.core_v1_case_study import summarize_core_v1_case_study


class CoreV1CaseStudyTests(unittest.TestCase):
    def test_core_v1_case_study_is_generated_from_mock_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            method = root / "method" / "five_program_method_summary.csv"
            reduction = root / "reduction" / "exact_reduction_summary.md"
            budgeted = root / "budgeted" / "budgeted_sensitivity_summary.md"
            _write_method_summary(method)
            _write_reduction_inputs(reduction)
            _write_budgeted_inputs(budgeted)
            out_dir = root / "out"

            result = summarize_core_v1_case_study(
                method,
                reduction,
                budgeted,
                out_dir,
                label="core_v1_exact_r4",
                nbody_round_study=root / "missing_nbody.md",
                puzzle_case_study=root / "missing_puzzle.md",
            )

            summary = (out_dir / "core_v1_case_study_summary.md").read_text(encoding="utf-8")
            numbers = _read_csv(out_dir / "core_v1_case_study_numbers.csv")
            claims = _read_csv(out_dir / "core_v1_key_claims.csv")
            figures = _read_csv(out_dir / "core_v1_figures_data.csv")
            missing = _read_csv(out_dir / "missing_inputs.csv")

        self.assertEqual(result["programs"], 2)
        self.assertIn("# Core-v1 Case Study Summary", summary)
        self.assertIn("Core-v1 is used as the controlled setting", summary)
        self.assertIn("## 3. Exact r4 Method Comparison", summary)
        self.assertIn("## 4. Exact r4 Reduction Evidence", summary)
        self.assertIn("## 6. Budgeted Sensitivity", summary)
        self.assertIn("All reduction claims are state-local.", summary)
        self.assertIn("WARNING", summary)
        self.assertEqual({row["program"] for row in numbers}, {"alpha", "beta"})
        alpha = next(row for row in numbers if row["program"] == "alpha")
        self.assertEqual(alpha["root_inst"], "100")
        self.assertEqual(alpha["exact_batch_inst"], "50")
        self.assertEqual(alpha["best_budgeted_inst"], "50")
        self.assertEqual(alpha["strong_certs"], "9")
        self.assertEqual(alpha["dropped_active_passes"], "0")
        self.assertGreaterEqual(len(claims), 5)
        self.assertTrue(any(row["claim_id"] == "C4" for row in claims))
        self.assertTrue(any(row["figure"] == "exact_vs_budgeted_final_inst" for row in figures))
        self.assertEqual({row["input_name"] for row in missing}, {"nbody_round_study", "puzzle_case_study"})

    def test_optional_notes_are_included_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            method = root / "method.csv"
            reduction = root / "reduction" / "exact_reduction_summary.md"
            budgeted = root / "budgeted" / "budgeted_sensitivity_summary.md"
            nbody = root / "nbody.md"
            puzzle = root / "puzzle.md"
            _write_method_summary(method)
            _write_reduction_inputs(reduction)
            _write_budgeted_inputs(budgeted)
            nbody.write_text("r2=223\nr3=214\nr4=211\nremaining active passes: early-cse;gvn\n", encoding="utf-8")
            puzzle.write_text("puzzle requires beam 16 in this run.\n", encoding="utf-8")

            summarize_core_v1_case_study(
                method,
                reduction,
                budgeted,
                root / "out",
                label="core",
                nbody_round_study=nbody,
                puzzle_case_study=puzzle,
            )

            summary = (root / "out" / "core_v1_case_study_summary.md").read_text(encoding="utf-8")
            missing = _read_csv(root / "out" / "missing_inputs.csv")

        self.assertIn("remaining active passes: early-cse;gvn", summary)
        self.assertIn("puzzle requires beam 16", summary)
        self.assertEqual(missing, [])


def _write_method_summary(path: Path) -> None:
    _write_csv(
        path,
        [
            "program",
            "root",
            "batch",
            "greedy",
            "random_best",
            "config_once",
            "batch_vs_greedy",
            "batch_vs_random",
            "batch_vs_config",
            "batch_states",
            "batch_transitions",
            "batch_time_ms",
        ],
        [
            {"program": "alpha", "root": "100", "batch": "50", "greedy": "51", "random_best": "52", "config_once": "53", "batch_vs_greedy": "1", "batch_vs_random": "2", "batch_vs_config": "3", "batch_states": "10", "batch_transitions": "9", "batch_time_ms": "1000"},
            {"program": "beta", "root": "80", "batch": "40", "greedy": "40", "random_best": "39", "config_once": "41", "batch_vs_greedy": "0", "batch_vs_random": "-1", "batch_vs_config": "1", "batch_states": "8", "batch_transitions": "7", "batch_time_ms": "900"},
        ],
    )


def _write_reduction_inputs(summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("# Exact r4 Reduction Evidence Summary\n", encoding="utf-8")
    _write_csv(
        summary_path.parent / "exact_reduction_runs.csv",
        ["program", "states_reached", "transitions"],
        [
            {"program": "alpha", "states_reached": "10", "transitions": "9"},
            {"program": "beta", "states_reached": "8", "transitions": "7"},
        ],
    )
    _write_csv(
        summary_path.parent / "reduction_by_program.csv",
        [
            "program",
            "states",
            "total_batch_candidates",
            "total_executable_batches",
            "avg_active_passes",
            "avg_local_reduction_log10",
            "max_local_reduction_log10",
            "total_tested_pairs",
            "total_commute_pairs",
            "total_order_sensitive_pairs",
            "total_unknown_pairs",
        ],
        [
            {"program": "alpha", "states": "10", "total_batch_candidates": "11", "total_executable_batches": "9", "avg_active_passes": "3.5", "avg_local_reduction_log10": "2.0", "max_local_reduction_log10": "4.0", "total_tested_pairs": "20", "total_commute_pairs": "12", "total_order_sensitive_pairs": "8", "total_unknown_pairs": "0"},
            {"program": "beta", "states": "8", "total_batch_candidates": "7", "total_executable_batches": "7", "avg_active_passes": "2.0", "avg_local_reduction_log10": "1.0", "max_local_reduction_log10": "2.0", "total_tested_pairs": "10", "total_commute_pairs": "7", "total_order_sensitive_pairs": "3", "total_unknown_pairs": "0"},
        ],
    )
    _write_csv(
        summary_path.parent / "coverage_by_program.csv",
        ["program", "dropped_active_passes"],
        [
            {"program": "alpha", "dropped_active_passes": "0"},
            {"program": "beta", "dropped_active_passes": "0"},
        ],
    )
    _write_csv(
        summary_path.parent / "evidence_by_batch_all.csv",
        ["program", "evidence_strength"],
        [{"program": "alpha", "evidence_strength": "strong"} for _ in range(9)]
        + [{"program": "beta", "evidence_strength": "strong"} for _ in range(7)],
    )
    _write_csv(
        summary_path.parent / "selected_path_evidence.csv",
        ["program", "evidence_strength"],
        [
            {"program": "alpha", "evidence_strength": "strong"},
            {"program": "beta", "evidence_strength": "strong"},
        ],
    )


def _write_budgeted_inputs(summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        "# Budgeted Sensitivity Summary\n\n"
        "## Budgeted vs Exact\n\n"
        "- programs matching exact: 2\n"
        "- average gap to exact: 0\n"
        "- average state reduction relative to exact: 50.00%\n"
        "- average time reduction relative to exact: 25.00%\n\n"
        "## Budgeted vs Baselines\n\n"
        "- budgeted vs greedy: wins=1 ties=1 losses=0\n"
        "- budgeted vs random best: wins=1 ties=0 losses=1\n"
        "- budgeted vs config order once: wins=2 ties=0 losses=0\n",
        encoding="utf-8",
    )
    _write_csv(
        summary_path.parent / "budgeted_sensitivity_best.csv",
        [
            "program",
            "best_beam_width",
            "best_max_states",
            "best_final_ir_inst_count",
            "exact_r4_inst",
            "gap_to_exact",
            "states_reached",
            "time_ms",
        ],
        [
            {"program": "alpha", "best_beam_width": "4", "best_max_states": "100", "best_final_ir_inst_count": "50", "exact_r4_inst": "50", "gap_to_exact": "0", "states_reached": "5", "time_ms": "500"},
            {"program": "beta", "best_beam_width": "8", "best_max_states": "200", "best_final_ir_inst_count": "40", "exact_r4_inst": "40", "gap_to_exact": "0", "states_reached": "6", "time_ms": "600"},
        ],
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
