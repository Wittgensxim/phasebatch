import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.path_diagnostic import diagnose_paths
from phasebatch.schema import RunResult


class PathDiagnosticTests(unittest.TestCase):
    def test_diagnose_paths_generates_comparison_prefixes_and_missed_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_mock_run(run_dir, include_greedy=True)

            def fake_run_opt(opt, input_ll, passes, output_ll, timeout):
                count = _count_for_passes(passes)
                output_ll.write_text(f"; inst_count={count}\n", encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            def fake_count(path):
                text = Path(path).read_text(encoding="utf-8", errors="replace")
                marker = "inst_count="
                return int(text.split(marker, 1)[1].splitlines()[0]) if marker in text else 12

            with mock.patch("phasebatch.path_diagnostic.run_opt", side_effect=fake_run_opt), mock.patch(
                "phasebatch.path_diagnostic.count_ir_instructions", side_effect=fake_count
            ):
                result = diagnose_paths(run_dir, timeout=1)

            comparison = _read_csv(Path(result["pipeline_comparison_csv"]))
            batch_prefix = _read_csv(Path(result["prefix_eval_batch_csv"]))
            greedy_prefix = _read_csv(Path(result["prefix_eval_greedy_csv"]))
            missed = _read_csv(Path(result["missed_pass_diagnostic_csv"]))
            markdown = Path(result["path_diagnostic_md"]).read_text(encoding="utf-8")

        self.assertEqual({row["method"] for row in comparison}, {"batch_optimizer", "greedy_single_pass", "random_single_pass_best", "config_order_once", "default_O0"})
        self.assertEqual(batch_prefix[0]["prefix_len"], "0")
        self.assertEqual(batch_prefix[-1]["ir_inst_count"], "9")
        self.assertEqual(greedy_prefix[2]["ir_inst_count"], "7")
        self.assertIn("first prefix where greedy is lower", markdown)
        self.assertIn("This diagnostic identifies likely reasons; it does not prove global optimality.", markdown)

        by_pass = {row["greedy_pass"]: row for row in missed}
        self.assertEqual(by_pass["pass-a"]["appears_in_batch_pipeline"], "true")
        self.assertEqual(by_pass["pass-a"]["diagnostic_reason"], "pass included in batch pipeline")
        self.assertEqual(by_pass["pass-b"]["appears_in_any_sampled_root_batch"], "true")
        self.assertEqual(by_pass["pass-b"]["appears_in_any_order_sensitive_pair"], "true")
        self.assertEqual(by_pass["pass-b"]["diagnostic_reason"], "pass only appears in sampled batch")

    def test_missing_greedy_path_does_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_mock_run(run_dir, include_greedy=False)

            with mock.patch("phasebatch.path_diagnostic.run_opt") as fake_run_opt:
                result = diagnose_paths(run_dir, timeout=1)

            greedy_prefix = _read_csv(Path(result["prefix_eval_greedy_csv"]))
            missed = _read_csv(Path(result["missed_pass_diagnostic_csv"]))

        self.assertGreaterEqual(fake_run_opt.call_count, 1)
        self.assertEqual(greedy_prefix, [])
        self.assertEqual(missed, [])


def _make_mock_run(run_dir: Path, *, include_greedy: bool) -> None:
    root_state = run_dir / "states" / "S0000"
    root_state.mkdir(parents=True, exist_ok=True)
    (root_state / "input.ll").write_text("; inst_count=12\n", encoding="utf-8")
    (run_dir / "metadata.json").write_text(json.dumps({"tools": {"opt": {"path": "opt"}}}), encoding="utf-8")
    (run_dir / "optimized_pipeline.txt").write_text("pass-a,pass-c\n", encoding="utf-8")

    baseline_rows = [
        {"method": "default_O0", "status": "success", "final_ir_inst_count": "12", "ir_inst_delta": "0", "ir_inst_reduction_pct": "0", "pass_sequence": "", "final_sequence_length": "0", "states_evaluated": "1", "opt_runs": "0"},
        {"method": "batch_optimizer", "status": "success", "final_ir_inst_count": "9", "ir_inst_delta": "-3", "ir_inst_reduction_pct": "25", "pass_sequence": "pass-a;pass-c", "final_sequence_length": "2", "states_evaluated": "2", "opt_runs": "2"},
        {"method": "random_single_pass_best", "status": "success", "final_ir_inst_count": "11", "ir_inst_delta": "-1", "ir_inst_reduction_pct": "8.33", "pass_sequence": "pass-d", "final_sequence_length": "1", "states_evaluated": "1", "opt_runs": "1"},
        {"method": "config_order_once", "status": "success", "final_ir_inst_count": "10", "ir_inst_delta": "-2", "ir_inst_reduction_pct": "16.67", "pass_sequence": "pass-a;pass-b;pass-c", "final_sequence_length": "3", "states_evaluated": "1", "opt_runs": "1"},
    ]
    if include_greedy:
        baseline_rows.append({"method": "greedy_single_pass", "status": "success", "final_ir_inst_count": "7", "ir_inst_delta": "-5", "ir_inst_reduction_pct": "41.67", "pass_sequence": "pass-a;pass-b", "final_sequence_length": "2", "states_evaluated": "2", "opt_runs": "2"})
    _write_csv(run_dir / "baseline_results.csv", baseline_rows[0].keys(), baseline_rows)

    if include_greedy:
        _write_csv(
            run_dir / "baselines" / "greedy_single_pass" / "greedy_path.csv",
            ["round", "selected_pass"],
            [{"round": "0", "selected_pass": "pass-a"}, {"round": "1", "selected_pass": "pass-b"}],
        )
    _write_csv(
        run_dir / "baselines" / "random_single_pass" / "random_best_path.csv",
        ["round", "selected_pass"],
        [{"round": "0", "selected_pass": "pass-d"}],
    )
    _write_csv(
        root_state / "batch_candidates.csv",
        ["batch_id", "batch_passes"],
        [{"batch_id": "B0000", "batch_passes": "pass-a;pass-c"}, {"batch_id": "B0001", "batch_passes": "pass-b"}],
    )
    _write_csv(
        root_state / "batch_correctness.csv",
        ["batch_id", "batch_passes", "correctness_class"],
        [{"batch_id": "B0000", "batch_passes": "pass-a;pass-c", "correctness_class": "certified_batch"}, {"batch_id": "B0001", "batch_passes": "pass-b", "correctness_class": "sampled_batch"}],
    )
    _write_csv(
        root_state / "pair_relation.csv",
        ["pass_a", "pass_b", "final_relation"],
        [{"pass_a": "pass-b", "pass_b": "pass-c", "final_relation": "final_order_sensitive"}],
    )


def _count_for_passes(passes: list[str]) -> int:
    key = tuple(passes)
    return {
        ("pass-a",): 10,
        ("pass-a", "pass-c"): 9,
        ("pass-a", "pass-b"): 7,
        ("pass-d",): 11,
    }.get(key, 8)


def _write_csv(path: Path, fieldnames, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
