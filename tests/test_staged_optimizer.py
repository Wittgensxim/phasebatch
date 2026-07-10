import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.schema import RunResult
from phasebatch.runtime_rerank import RuntimeCandidate, RuntimeRerankResult
from phasebatch.staged_optimizer import optimize_staged


class StagedOptimizerTests(unittest.TestCase):
    def test_hands_off_stage_ir_replays_in_order_and_cleans_ir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("root\n", encoding="utf-8")
            passes = root / "passes.yaml"
            passes.write_text("passes:\n  - instcombine\n", encoding="utf-8")
            manifest = root / "staged.yaml"
            manifest.write_text(
                "root_ir_mode: inlinable-unoptimized\n"
                "stages:\n"
                "  - id: first\n"
                "    passes: passes.yaml\n"
                "    mode: exact\n"
                "    max_rounds: 1\n"
                "  - id: second\n"
                "    passes: passes.yaml\n"
                "    mode: budgeted\n"
                "    max_rounds: 2\n",
                encoding="utf-8",
            )
            out_dir = root / "out"
            calls = []

            def fake_stage_runner(input_path, stage_dir, passes_path, **kwargs):
                calls.append((Path(input_path), Path(stage_dir), Path(passes_path), kwargs))
                stage_dir = Path(stage_dir)
                (stage_dir / "states" / "S0000").mkdir(parents=True)
                source_text = Path(input_path).read_text(encoding="utf-8")
                (stage_dir / "states" / "S0000" / "input.ll").write_text(source_text, encoding="utf-8")
                stage_number = len(calls)
                (stage_dir / "final.ll").write_text(f"stage-{stage_number}\n", encoding="utf-8")
                (stage_dir / "optimized_pipeline.txt").write_text(f"pass-{stage_number}\n", encoding="utf-8")
                (stage_dir / "metadata.json").write_text(
                    json.dumps(
                        {
                            "exact_status": "exact_complete" if kwargs["mode"] == "exact" else "not_applicable",
                            "pair_matrix_complete": True,
                        }
                    ),
                    encoding="utf-8",
                )
                return {
                    "states": stage_number + 1,
                    "batch_transitions": stage_number,
                    "selected_final_state": f"S{stage_number:04d}",
                }

            replay_index = 0

            def fake_run_opt(_opt, _input, _passes, output, _timeout):
                nonlocal replay_index
                replay_index += 1
                Path(output).parent.mkdir(parents=True, exist_ok=True)
                Path(output).write_text(f"stage-{replay_index}\n", encoding="utf-8")
                return RunResult(
                    command=["opt"],
                    returncode=0,
                    stdout="",
                    stderr="",
                    time_ms=1.0,
                    output_path=Path(output),
                )

            with mock.patch("phasebatch.staged_optimizer.collect_toolchain", return_value={"tools": {"opt": {"path": "opt", "version": None}}}), \
                mock.patch("phasebatch.staged_optimizer.run_opt", side_effect=fake_run_opt):
                result = optimize_staged(
                    input_ll,
                    out_dir,
                    manifest,
                    jobs=2,
                    timeout=5,
                    keep_ir_artifacts=False,
                    stage_runner=fake_stage_runner,
                )

            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][3]["root_ir_mode"], "inlinable-unoptimized")
            self.assertTrue(calls[0][3]["keep_ir_artifacts"])
            self.assertEqual(calls[1][0].read_text(encoding="utf-8") if calls[1][0].exists() else "stage-1\n", "stage-1\n")
            self.assertEqual(replay_index, 2)
            self.assertTrue(result["replay_verified"])
            self.assertTrue((out_dir / "staged_summary.csv").exists())
            self.assertTrue((out_dir / "staged_pipeline.csv").exists())
            self.assertEqual(list(out_dir.rglob("*.ll")), [])

    def test_runtime_winner_replaces_static_stage_handoff_and_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("root\n", encoding="utf-8")
            passes = root / "passes.yaml"
            passes.write_text("passes:\n  - pass-a\n  - pass-b\n", encoding="utf-8")
            manifest = root / "staged.yaml"
            manifest.write_text(
                "root_ir_mode: inlinable-unoptimized\n"
                "runtime:\n"
                "  enabled: true\n"
                "  top_k: 2\n"
                "  warmups: 1\n"
                "  trials: 2\n"
                "stages:\n"
                "  - id: final\n"
                "    passes: passes.yaml\n"
                "    mode: budgeted\n"
                "    runtime_rerank: true\n",
                encoding="utf-8",
            )
            out_dir = root / "out"
            winner_ir = out_dir / "winner.ll"

            def fake_stage_runner(input_path, stage_dir, passes_path, **kwargs):
                del input_path, passes_path, kwargs
                stage_dir = Path(stage_dir)
                (stage_dir / "states" / "S0000").mkdir(parents=True)
                (stage_dir / "states" / "S0000" / "input.ll").write_text("root\n", encoding="utf-8")
                (stage_dir / "final.ll").write_text("static\n", encoding="utf-8")
                (stage_dir / "optimized_pipeline.txt").write_text("pass-a\n", encoding="utf-8")
                (stage_dir / "metadata.json").write_text(
                    json.dumps({"exact_status": "not_applicable", "pair_matrix_complete": True}),
                    encoding="utf-8",
                )
                winner_ir.parent.mkdir(parents=True, exist_ok=True)
                winner_ir.write_text("winner\n", encoding="utf-8")
                return {"states": 3, "batch_transitions": 2, "selected_final_state": "S0001"}

            candidate = RuntimeCandidate(
                state_id="S0002",
                state_hash="winner-hash",
                ir_path=winner_ir,
                objective_value=20.0,
                pipeline=("pass-b",),
                selected_as_final=False,
                direct_calls=0,
                memory_ops=0,
                branches=0,
            )
            runtime_result = RuntimeRerankResult(
                candidates=(candidate,),
                winner=candidate,
                winner_summary={"state_id": "S0002", "median_ms": "1.250", "eligible": "true"},
                reason="runtime_median",
                candidates_csv=out_dir / "runtime_candidates.csv",
                trials_csv=out_dir / "runtime_trials.csv",
                summary_csv=out_dir / "runtime_summary.csv",
                selection_md=out_dir / "runtime_selection.md",
            )

            def fake_run_opt(_opt, _input, passes_arg, output, _timeout):
                self.assertEqual(passes_arg, ["pass-b"])
                Path(output).parent.mkdir(parents=True, exist_ok=True)
                Path(output).write_text("winner\n", encoding="utf-8")
                return RunResult(
                    command=["opt"],
                    returncode=0,
                    stdout="",
                    stderr="",
                    time_ms=1.0,
                    output_path=Path(output),
                )

            with mock.patch("phasebatch.staged_optimizer.collect_toolchain", return_value={"tools": {"opt": {"path": "opt", "version": None}}}), \
                mock.patch("phasebatch.staged_optimizer.rerank_terminal_states", return_value=runtime_result) as fake_rerank, \
                mock.patch("phasebatch.staged_optimizer.run_opt", side_effect=fake_run_opt):
                result = optimize_staged(
                    input_ll,
                    out_dir,
                    manifest,
                    jobs=2,
                    timeout=5,
                    keep_ir_artifacts=True,
                    stage_runner=fake_stage_runner,
                )

            pipeline_rows = _read_csv(out_dir / "staged_pipeline.csv")
            summary_rows = _read_csv(out_dir / "staged_summary.csv")

        fake_rerank.assert_called_once()
        self.assertTrue(result["replay_verified"])
        self.assertEqual(pipeline_rows[0]["pipeline"], "pass-b")
        self.assertEqual(pipeline_rows[0]["selection_source"], "runtime_median")
        self.assertEqual(summary_rows[0]["selected_final_state"], "S0002")
        self.assertEqual(summary_rows[0]["runtime_median_ms"], "1.250")

    def test_required_transition_replaces_identity_stage_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("root\n", encoding="utf-8")
            passes = root / "passes.yaml"
            passes.write_text("passes:\n  - inline\n", encoding="utf-8")
            manifest = root / "staged.yaml"
            manifest.write_text(
                "stages:\n"
                "  - id: ipo\n"
                "    passes: passes.yaml\n"
                "    mode: exact\n"
                "    require_transition: true\n",
                encoding="utf-8",
            )
            out_dir = root / "out"

            def fake_stage_runner(input_path, stage_dir, passes_path, **kwargs):
                del input_path, passes_path, kwargs
                stage_dir = Path(stage_dir)
                root_ir = stage_dir / "states" / "S0000" / "input.ll"
                child_ir = stage_dir / "states" / "S0001" / "input.ll"
                root_ir.parent.mkdir(parents=True)
                child_ir.parent.mkdir(parents=True)
                root_ir.write_text("root\n", encoding="utf-8")
                child_ir.write_text("inlined\n", encoding="utf-8")
                (stage_dir / "final.ll").write_text("root\n", encoding="utf-8")
                (stage_dir / "optimized_pipeline.txt").write_text("", encoding="utf-8")
                (stage_dir / "metadata.json").write_text(
                    json.dumps({"exact_status": "exact_complete", "pair_matrix_complete": True}),
                    encoding="utf-8",
                )
                _write_csv(
                    stage_dir / "states.csv",
                    ["state_id", "state_hash", "parent_state_id", "transition_pass", "ir_path"],
                    [
                        {"state_id": "S0000", "state_hash": "h0", "parent_state_id": "", "transition_pass": "", "ir_path": str(root_ir)},
                        {"state_id": "S0001", "state_hash": "h1", "parent_state_id": "S0000", "transition_pass": "inline", "ir_path": str(child_ir)},
                    ],
                )
                _write_csv(
                    stage_dir / "leaf_states.csv",
                    ["state_id", "objective_value", "is_leaf", "selected_as_final"],
                    [
                        {"state_id": "S0000", "objective_value": "10", "is_leaf": "false", "selected_as_final": "true"},
                        {"state_id": "S0001", "objective_value": "20", "is_leaf": "true", "selected_as_final": "false"},
                    ],
                )
                return {"states": 2, "batch_transitions": 1, "selected_final_state": "S0000"}

            def fake_run_opt(_opt, _input, passes_arg, output, _timeout):
                self.assertEqual(passes_arg, ["inline"])
                Path(output).parent.mkdir(parents=True, exist_ok=True)
                Path(output).write_text("inlined\n", encoding="utf-8")
                return RunResult(
                    command=["opt"],
                    returncode=0,
                    stdout="",
                    stderr="",
                    time_ms=1.0,
                    output_path=Path(output),
                )

            with mock.patch("phasebatch.staged_optimizer.collect_toolchain", return_value={"tools": {"opt": {"path": "opt", "version": None}}}), \
                mock.patch("phasebatch.staged_optimizer.run_opt", side_effect=fake_run_opt):
                result = optimize_staged(
                    input_ll,
                    out_dir,
                    manifest,
                    jobs=1,
                    timeout=5,
                    keep_ir_artifacts=True,
                    stage_runner=fake_stage_runner,
                )

            pipeline_rows = _read_csv(out_dir / "staged_pipeline.csv")
            summary_rows = _read_csv(out_dir / "staged_summary.csv")

        self.assertTrue(result["replay_verified"])
        self.assertEqual(pipeline_rows[0]["pipeline"], "inline")
        self.assertEqual(pipeline_rows[0]["selection_source"], "required_transition")
        self.assertEqual(summary_rows[0]["selected_final_state"], "S0001")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
