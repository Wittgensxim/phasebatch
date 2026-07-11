import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.ir_equivalence import EqualityResult
from phasebatch.pipeline_replay import replay_optimized_pipeline
from phasebatch.schema import RunResult


class PipelineReplayTests(unittest.TestCase):
    def test_replay_succeeds_when_pipeline_reproduces_final_ir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _make_run(Path(tmp), pipeline="pass-a,pass-b", final_count=1)

            def fake_run_opt(opt, input_ll, passes, output_ll, timeout):
                self.assertEqual(passes, ["pass-a", "pass-b"])
                Path(output_ll).write_text(_ir(1), encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0, output_path=Path(output_ll))

            with mock.patch("phasebatch.pipeline_replay.run_opt", side_effect=fake_run_opt):
                result = replay_optimized_pipeline(run_dir, timeout=5)

            rows = _read_csv(run_dir / "pipeline_replay.csv")
            replay_output_exists = (run_dir / "replayed_final.ll").exists()

        self.assertEqual(result["replay_status"], "success")
        self.assertEqual(result["hashes_match"], "true")
        self.assertTrue(replay_output_exists)
        self.assertEqual(rows[0]["optimized_pipeline"], "pass-a,pass-b")
        self.assertEqual(rows[0]["hashes_match"], "true")

    def test_replay_preserves_chosen_batch_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _make_run(Path(tmp), pipeline="pass-a,pass-b,pass-c", final_count=1)
            _write_csv(
                run_dir / "chosen_path.csv",
                ["step", "canonical_order"],
                [
                    {"step": "0", "canonical_order": "pass-a;pass-b"},
                    {"step": "1", "canonical_order": "pass-c"},
                ],
            )
            calls: list[tuple[Path, list[str], Path]] = []

            def fake_run_opt(opt, input_ll, passes, output_ll, timeout):
                del opt, timeout
                calls.append((Path(input_ll), list(passes), Path(output_ll)))
                Path(output_ll).write_text(_ir(2 if len(calls) == 1 else 1), encoding="utf-8")
                return RunResult(["opt"], 0, "", "", 1.0, output_path=Path(output_ll))

            with mock.patch("phasebatch.pipeline_replay.run_opt", side_effect=fake_run_opt):
                result = replay_optimized_pipeline(run_dir, timeout=5)

        self.assertEqual(result["replay_status"], "success")
        self.assertEqual([passes for _input, passes, _output in calls], [["pass-a", "pass-b"], ["pass-c"]])
        self.assertEqual(calls[1][0], calls[0][2])

    def test_replay_detects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _make_run(Path(tmp), pipeline="pass-a", final_count=1)

            def fake_run_opt(opt, input_ll, passes, output_ll, timeout):
                Path(output_ll).write_text(_ir(2), encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0, output_path=Path(output_ll))

            different = EqualityResult(
                equal=False,
                tier="different",
                can_hard_fold=False,
                reason="llvm_diff_difference",
                text_hash_equal=False,
                llvm_diff_equal=False,
                left_hash="replay-hash",
                right_hash="final-hash",
            )

            with mock.patch("phasebatch.pipeline_replay.run_opt", side_effect=fake_run_opt), \
                mock.patch("phasebatch.pipeline_replay.compare_ir_equivalence", return_value=different):
                result = replay_optimized_pipeline(run_dir)

            rows = _read_csv(run_dir / "pipeline_replay.csv")

        self.assertEqual(result["replay_status"], "mismatch")
        self.assertEqual(result["hashes_match"], "false")
        self.assertEqual(rows[0]["replay_status"], "mismatch")

    def test_replay_uses_structural_equality_when_hashes_differ(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _make_run(Path(tmp), pipeline="pass-a", final_count=1)

            def fake_run_opt(opt, input_ll, passes, output_ll, timeout):
                Path(output_ll).write_text("define i32 @f(i32 %x) {\n  %a = add i32 %x, 0\n  ret i32 %a\n}\n", encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0, output_path=Path(output_ll))

            structural = EqualityResult(
                equal=True,
                tier="structural_diff",
                can_hard_fold=True,
                reason="llvm_diff_equal_and_module_fingerprint_equal",
                text_hash_equal=False,
                llvm_diff_equal=True,
                module_fingerprint_equal=True,
                left_hash="replay-hash",
                right_hash="final-hash",
            )

            with mock.patch("phasebatch.pipeline_replay.run_opt", side_effect=fake_run_opt), \
                mock.patch("phasebatch.pipeline_replay.compare_ir_equivalence", return_value=structural):
                result = replay_optimized_pipeline(run_dir, timeout=5)

            rows = _read_csv(run_dir / "pipeline_replay.csv")

        self.assertEqual(result["replay_status"], "success")
        self.assertEqual(result["hashes_match"], "true")
        self.assertEqual(rows[0]["replay_hash"], "replay-hash")
        self.assertEqual(rows[0]["final_hash"], "final-hash")
        self.assertEqual(rows[0]["text_hash_equal"], "false")
        self.assertEqual(rows[0]["llvm_diff_equal"], "true")
        self.assertEqual(rows[0]["module_fingerprint_equal"], "true")
        self.assertEqual(rows[0]["equality_tier"], "structural_diff")
        self.assertEqual(rows[0]["equality_reason"], "llvm_diff_equal_and_module_fingerprint_equal")
        self.assertEqual(rows[0]["can_hard_fold"], "true")

    def test_replay_mismatch_records_equality_difference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _make_run(Path(tmp), pipeline="pass-a", final_count=1)

            def fake_run_opt(opt, input_ll, passes, output_ll, timeout):
                Path(output_ll).write_text(_ir(2), encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0, output_path=Path(output_ll))

            different = EqualityResult(
                equal=False,
                tier="different",
                can_hard_fold=False,
                reason="module_fingerprint_difference",
                text_hash_equal=False,
                llvm_diff_equal=True,
                module_fingerprint_equal=False,
                left_hash="replay-hash",
                right_hash="final-hash",
            )

            with mock.patch("phasebatch.pipeline_replay.run_opt", side_effect=fake_run_opt), \
                mock.patch("phasebatch.pipeline_replay.compare_ir_equivalence", return_value=different):
                result = replay_optimized_pipeline(run_dir)

            rows = _read_csv(run_dir / "pipeline_replay.csv")

        self.assertEqual(result["replay_status"], "mismatch")
        self.assertEqual(result["hashes_match"], "false")
        self.assertEqual(rows[0]["equality_tier"], "different")
        self.assertEqual(rows[0]["equality_reason"], "module_fingerprint_difference")
        self.assertEqual(rows[0]["can_hard_fold"], "false")

    def test_replay_comparator_failure_records_failed_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _make_run(Path(tmp), pipeline="pass-a", final_count=1)

            def fake_run_opt(opt, input_ll, passes, output_ll, timeout):
                Path(output_ll).write_text(_ir(1), encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0, output_path=Path(output_ll))

            failed = EqualityResult(
                equal=False,
                tier="failed",
                can_hard_fold=False,
                reason="tool_failed",
                text_hash_equal=False,
                left_hash="replay-hash",
                right_hash="final-hash",
                error_message="llvm-diff not found",
            )

            with mock.patch("phasebatch.pipeline_replay.run_opt", side_effect=fake_run_opt), \
                mock.patch("phasebatch.pipeline_replay.compare_ir_equivalence", return_value=failed):
                result = replay_optimized_pipeline(run_dir)

            rows = _read_csv(run_dir / "pipeline_replay.csv")

        self.assertEqual(result["replay_status"], "failed")
        self.assertEqual(result["hashes_match"], "false")
        self.assertEqual(rows[0]["equality_tier"], "failed")
        self.assertEqual(rows[0]["equality_reason"], "tool_failed")
        self.assertEqual(rows[0]["can_hard_fold"], "false")
        self.assertIn("llvm-diff not found", rows[0]["error_message"])

    def test_empty_pipeline_compares_root_and_final(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _make_run(Path(tmp), pipeline="", root_count=1, final_count=1)

            result = replay_optimized_pipeline(run_dir)
            rows = _read_csv(run_dir / "pipeline_replay.csv")
            replay_output_exists = (run_dir / "replayed_final.ll").exists()

        self.assertEqual(result["replay_status"], "success")
        self.assertEqual(result["hashes_match"], "true")
        self.assertEqual(rows[0]["optimized_pipeline"], "")
        self.assertTrue(replay_output_exists)

    def test_pipeline_replay_csv_generated_on_failed_opt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _make_run(Path(tmp), pipeline="bad-pass", final_count=1)

            def fake_run_opt(opt, input_ll, passes, output_ll, timeout):
                return RunResult([opt], 1, "", "bad pass", 1.0, output_path=Path(output_ll))

            with mock.patch("phasebatch.pipeline_replay.run_opt", side_effect=fake_run_opt):
                result = replay_optimized_pipeline(run_dir)

            rows = _read_csv(run_dir / "pipeline_replay.csv")

        self.assertEqual(result["replay_status"], "failed")
        self.assertEqual(rows[0]["error_message"], "bad pass")


def _make_run(root: Path, *, pipeline: str, root_count: int = 2, final_count: int = 1) -> Path:
    run_dir = root / "run"
    (run_dir / "states" / "S0000").mkdir(parents=True, exist_ok=True)
    (run_dir / "states" / "S0000" / "input.ll").write_text(_ir(root_count), encoding="utf-8")
    (run_dir / "final.ll").write_text(_ir(final_count), encoding="utf-8")
    (run_dir / "optimized_pipeline.txt").write_text(pipeline + ("\n" if pipeline else ""), encoding="utf-8")
    (run_dir / "metadata.json").write_text(json.dumps({"tools": {"opt": {"path": "opt"}}}), encoding="utf-8")
    return run_dir


def _ir(count: int) -> str:
    lines = ["define i32 @f(i32 %x) {", "entry:"]
    for index in range(max(0, count - 1)):
        source = "%x" if index == 0 else f"%v{index - 1}"
        lines.append(f"  %v{index} = add i32 {source}, 1")
    value = "%x" if count <= 1 else f"%v{count - 2}"
    lines.append(f"  ret i32 {value}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
