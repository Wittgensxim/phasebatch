import csv
import tempfile
import unittest
from pathlib import Path

from phasebatch.passset_summary import summarize_passsets


class PasssetSummaryTests(unittest.TestCase):
    def test_reads_v1_v2_smoke_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            smoke = root / "passset_smoke"
            _write_passset_smoke(smoke)

            result = summarize_passsets([smoke], root / "report")
            matrix = _read_csv(root / "report" / "passset_comparison_matrix.csv")

        self.assertEqual(result["matrix_rows"], 2)
        self.assertEqual({row["passset"] for row in matrix}, {"v1", "v2"})
        self.assertEqual(next(row for row in matrix if row["passset"] == "v2")["active_passes_depth0"], "9")

    def test_reads_v3_loop_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v3 = root / "v3_loop"
            _write_v3_loop(v3)

            summarize_passsets([v3], root / "report")
            matrix = _read_csv(root / "report" / "passset_comparison_matrix.csv")
            report = (root / "report" / "passset_comparison_report.md").read_text(encoding="utf-8")

        self.assertEqual(matrix[0]["passset"], "v3")
        self.assertEqual(matrix[0]["valid_passes"], "25")
        self.assertIn("## V3 vs V2", report)
        self.assertIn("loop passes resolved successfully", report)

    def test_missing_input_directory_warns_not_crashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            smoke = root / "passset_smoke"
            missing = root / "missing"
            _write_passset_smoke(smoke)
            warnings: list[str] = []

            result = summarize_passsets([smoke, missing], root / "report", warn=warnings.append)
            failures = _read_csv(root / "report" / "passset_failures.csv")

        self.assertEqual(result["failures"], 1)
        self.assertIn("missing input directory", warnings[0])
        self.assertEqual(failures[0]["status"], "missing")

    def test_matrix_and_report_are_generated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            smoke = root / "passset_smoke"
            v3 = root / "v3_loop"
            _write_passset_smoke(smoke)
            _write_v3_loop(v3)

            summarize_passsets([smoke, v3], root / "report")
            matrix_exists = (root / "report" / "passset_comparison_matrix.csv").exists()
            report_exists = (root / "report" / "passset_comparison_report.md").exists()
            failures_exists = (root / "report" / "passset_failures.csv").exists()

        self.assertTrue(matrix_exists)
        self.assertTrue(report_exists)
        self.assertTrue(failures_exists)

    def test_recommendation_section_is_generated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            smoke = root / "passset_smoke"
            v3 = root / "v3_loop"
            _write_passset_smoke(smoke)
            _write_v3_loop(v3)

            summarize_passsets([smoke, v3], root / "report")
            report = (root / "report" / "passset_comparison_report.md").read_text(encoding="utf-8")

        self.assertIn("## Recommendation", report)
        self.assertIn("use v3 only for loop-heavy case studies", report)


def _write_passset_smoke(root: Path) -> None:
    _write_csv(
        root / "passset_smoke_runs.csv",
        [
            "program",
            "passset",
            "audit_status",
            "optimize_status",
            "valid_passes",
            "invalid_passes",
            "active_passes_depth0",
            "states_reached",
            "transitions",
            "selected_final_state",
            "root_ir_inst_count",
            "final_ir_inst_count",
            "ir_inst_delta",
            "time_ms",
            "error_message",
        ],
        [
            _smoke_run("branch", "core_passes_v1", "14", "0", "8", "4", "3", "12", "2", "-10", "100.0"),
            _smoke_run("branch", "scalar_passes_v2", "19", "0", "9", "5", "4", "12", "2", "-10", "140.0"),
        ],
    )
    for passset, pass_label, active, pairs, commute, sensitive, candidates, certified in [
        ("core_passes_v1", "v1", "8", "28", "17", "11", "4", "4"),
        ("scalar_passes_v2", "v2", "9", "36", "22", "14", "4", "4"),
    ]:
        base = root / "branch" / passset
        _write_audit(base / "audit", valid="14" if pass_label == "v1" else "19", invalid="0", loop=False)
        _write_optimize(
            base / "optimize",
            active=active,
            pairs=pairs,
            commute=commute,
            sensitive=sensitive,
            unknown="0",
            candidates=candidates,
            certified=certified,
            sampled="0",
            skipped="0",
            dropped="0",
            states="3" if pass_label == "v1" else "4",
            transitions="2" if pass_label == "v1" else "3",
            final_inst="2",
            root_inst="12",
            pipeline="instcombine,simplifycfg",
        )


def _write_v3_loop(root: Path) -> None:
    _write_csv(
        root / "v3_loop_runs.csv",
        [
            "program",
            "input_path",
            "status",
            "valid_passes",
            "invalid_passes",
            "valid_loop_passes",
            "invalid_loop_passes",
            "active_loop_passes_depth0",
            "total_active_passes_depth0",
            "states_reached",
            "transitions",
            "exact_or_budgeted",
            "final_ir_inst_count",
            "optimized_pipeline_length",
            "time_ms",
            "error_message",
        ],
        [
            {
                "program": "n-body",
                "input_path": "n-body.c",
                "status": "success",
                "valid_passes": "25",
                "invalid_passes": "0",
                "valid_loop_passes": "6",
                "invalid_loop_passes": "0",
                "active_loop_passes_depth0": "2",
                "total_active_passes_depth0": "14",
                "states_reached": "94",
                "transitions": "93",
                "exact_or_budgeted": "budgeted",
                "final_ir_inst_count": "214",
                "optimized_pipeline_length": "6",
                "time_ms": "500.0",
                "error_message": "",
            }
        ],
    )
    _write_csv(
        root / "v3_loop_summary.csv",
        [
            "program",
            "valid_passes",
            "valid_loop_passes",
            "active_passes_depth0",
            "active_loop_passes_depth0",
            "tested_pairs_depth0",
            "commute_pairs_depth0",
            "sensitive_pairs_depth0",
            "batch_candidates_depth0",
            "certified_batches_depth0",
            "sampled_batches_depth0",
            "skipped_batches_depth0",
            "max_component_size_depth0",
            "states_reached",
            "transitions",
            "final_ir_inst_count",
            "dropped_active_passes",
        ],
        [
            {
                "program": "n-body",
                "valid_passes": "25",
                "valid_loop_passes": "6",
                "active_passes_depth0": "14",
                "active_loop_passes_depth0": "2",
                "tested_pairs_depth0": "91",
                "commute_pairs_depth0": "47",
                "sensitive_pairs_depth0": "19",
                "batch_candidates_depth0": "14",
                "certified_batches_depth0": "14",
                "sampled_batches_depth0": "0",
                "skipped_batches_depth0": "0",
                "max_component_size_depth0": "14",
                "states_reached": "94",
                "transitions": "93",
                "final_ir_inst_count": "214",
                "dropped_active_passes": "0",
            }
        ],
    )
    _write_audit(root / "n-body" / "audit", valid="25", invalid="0", loop=True)
    _write_optimize(
        root / "n-body" / "optimize",
        active="14",
        pairs="91",
        commute="47",
        sensitive="19",
        unknown="0",
        candidates="14",
        certified="14",
        sampled="0",
        skipped="0",
        dropped="0",
        states="94",
        transitions="93",
        final_inst="214",
        root_inst="300",
        pipeline="instcombine,licm,loop-rotate",
    )


def _smoke_run(program: str, passset: str, valid: str, invalid: str, active: str, states: str, transitions: str, root: str, final: str, delta: str, time_ms: str) -> dict[str, str]:
    return {
        "program": program,
        "passset": passset,
        "audit_status": "success",
        "optimize_status": "success",
        "valid_passes": valid,
        "invalid_passes": invalid,
        "active_passes_depth0": active,
        "states_reached": states,
        "transitions": transitions,
        "selected_final_state": "S0001",
        "root_ir_inst_count": root,
        "final_ir_inst_count": final,
        "ir_inst_delta": delta,
        "time_ms": time_ms,
        "error_message": "",
    }


def _write_audit(path: Path, valid: str, invalid: str, loop: bool) -> None:
    rows = [
        {
            "pass": "instcombine",
            "category": "scalar",
            "stage": "v1",
            "valid_on_input": "true",
            "resolved_pipeline": "instcombine",
        }
    ]
    if loop:
        rows.extend(
            [
                {"pass": "licm", "category": "loop", "stage": "v3", "valid_on_input": "true", "resolved_pipeline": "licm"},
                {"pass": "loop-rotate", "category": "loop", "stage": "v3", "valid_on_input": "true", "resolved_pipeline": "loop-rotate"},
            ]
        )
    _write_csv(path / "pass_audit.csv", ["pass", "category", "stage", "valid_on_input", "resolved_pipeline"], rows)
    invalid_rows = [{"pass": "bad", "category": "loop", "stage": "v3", "failure_kind": "failed", "stderr_summary": "bad"}] * int(invalid)
    _write_csv(path / "invalid_passes.csv", ["pass", "category", "stage", "failure_kind", "stderr_summary"], invalid_rows)


def _write_optimize(
    path: Path,
    *,
    active: str,
    pairs: str,
    commute: str,
    sensitive: str,
    unknown: str,
    candidates: str,
    certified: str,
    sampled: str,
    skipped: str,
    dropped: str,
    states: str,
    transitions: str,
    final_inst: str,
    root_inst: str,
    pipeline: str,
) -> None:
    state_dir = path / "states" / "S0000"
    _write_csv(state_dir / "per_state_summary.csv", ["active_passes", "pairs_tested", "dynamic_commute", "order_sensitive", "unknown"], [{"active_passes": active, "pairs_tested": pairs, "dynamic_commute": commute, "order_sensitive": sensitive, "unknown": unknown}])
    _write_csv(state_dir / "batch_summary.csv", ["batch_candidates"], [{"batch_candidates": candidates}])
    correctness_rows = [{"batch_id": f"B{i:04d}", "correctness_class": "certified_batch", "can_execute": "true"} for i in range(int(certified))]
    correctness_rows.extend({"batch_id": f"S{i:04d}", "correctness_class": "sampled_batch", "can_execute": "false"} for i in range(int(sampled)))
    _write_csv(state_dir / "batch_correctness.csv", ["batch_id", "correctness_class", "can_execute"], correctness_rows)
    _write_csv(state_dir / "coverage_summary.csv", ["dropped_active_passes"], [{"dropped_active_passes": dropped}])
    _write_csv(path / "states.csv", ["state_id"], [{"state_id": f"S{i:04d}"} for i in range(int(states))])
    _write_csv(path / "batch_state_transitions.csv", ["parent_state_id", "child_state_id"], [{"parent_state_id": "S0000", "child_state_id": f"S{i + 1:04d}"} for i in range(int(transitions))])
    _write_csv(path / "chosen_path_summary.csv", ["root_ir_inst_count", "final_ir_inst_count"], [{"root_ir_inst_count": root_inst, "final_ir_inst_count": final_inst}])
    (path / "optimized_pipeline_names.txt").parent.mkdir(parents=True, exist_ok=True)
    (path / "optimized_pipeline_names.txt").write_text(pipeline, encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
