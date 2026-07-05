import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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

    def test_replay_detects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _make_run(Path(tmp), pipeline="pass-a", final_count=1)

            def fake_run_opt(opt, input_ll, passes, output_ll, timeout):
                Path(output_ll).write_text(_ir(2), encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0, output_path=Path(output_ll))

            with mock.patch("phasebatch.pipeline_replay.run_opt", side_effect=fake_run_opt):
                result = replay_optimized_pipeline(run_dir)

            rows = _read_csv(run_dir / "pipeline_replay.csv")

        self.assertEqual(result["replay_status"], "mismatch")
        self.assertEqual(result["hashes_match"], "false")
        self.assertEqual(rows[0]["replay_status"], "mismatch")

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
