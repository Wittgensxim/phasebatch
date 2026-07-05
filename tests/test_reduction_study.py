import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.reduction_study import run_reduction_study


class ReductionStudyTests(unittest.TestCase):
    def test_run_reduction_study_handles_multiple_inputs_and_writes_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = _make_inputs(root, ["a.c", "b.c"])
            passes = root / "passes.yaml"
            passes.write_text("passes:\n  - instcombine\n", encoding="utf-8")
            out_dir = root / "out"

            with mock.patch("phasebatch.reduction_study.run_optimizer", side_effect=_fake_optimizer), \
                mock.patch("phasebatch.reduction_study.run_reduction_summary", side_effect=_fake_reduction), \
                mock.patch("phasebatch.reduction_study.run_evidence_pack", side_effect=_fake_evidence), \
                mock.patch("phasebatch.reduction_study.run_replay", side_effect=_fake_replay):
                result = run_reduction_study(
                    [str(path) for path in inputs],
                    out_dir,
                    passes,
                    optimizer_mode="exact",
                    objective="ir-inst-count",
                    max_rounds=2,
                    max_states=5000,
                    validate_batches=True,
                    jobs=1,
                    timeout=1,
                    max_pairs=10,
                    overwrite=False,
                    continue_on_error=False,
                )

            runs = _read_csv(out_dir / "reduction_study_runs.csv")
            summary = _read_csv(out_dir / "reduction_study_summary.csv")
            evidence = _read_csv(out_dir / "evidence_quality_summary.csv")
            markdown = (out_dir / "reduction_study_summary.md").read_text(encoding="utf-8")
            reduction_by_state_exists = (out_dir / "a" / "reduction_by_state.csv").exists()
            evidence_pack_exists = (out_dir / "a" / "evidence_pack.md").exists()

        self.assertEqual(result["programs"], 2)
        self.assertEqual(result["successes"], 2)
        self.assertEqual([row["status"] for row in runs], ["success", "success"])
        self.assertEqual({row["program"] for row in summary}, {"a", "b"})
        self.assertEqual(summary[0]["total_certified_batches"], "3")
        self.assertEqual(evidence[0]["selected_strong_certificates"], "2")
        self.assertIn("Reduction claims are state-local.", markdown)
        self.assertTrue(reduction_by_state_exists)
        self.assertTrue(evidence_pack_exists)

    def test_run_reduction_study_records_failure_when_continue_on_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = _make_inputs(root, ["bad.c", "good.c"])
            passes = root / "passes.yaml"
            passes.write_text("passes:\n  - instcombine\n", encoding="utf-8")
            out_dir = root / "out"

            def fake_optimizer(input_path, out_dir, passes_path, **kwargs):
                if Path(input_path).stem == "bad":
                    raise RuntimeError("boom")
                return _fake_optimizer(input_path, out_dir, passes_path, **kwargs)

            with mock.patch("phasebatch.reduction_study.run_optimizer", side_effect=fake_optimizer), \
                mock.patch("phasebatch.reduction_study.run_reduction_summary", side_effect=_fake_reduction), \
                mock.patch("phasebatch.reduction_study.run_evidence_pack", side_effect=_fake_evidence), \
                mock.patch("phasebatch.reduction_study.run_replay", side_effect=_fake_replay):
                result = run_reduction_study(
                    [str(path) for path in inputs],
                    out_dir,
                    passes,
                    optimizer_mode="exact",
                    objective="ir-inst-count",
                    max_rounds=2,
                    max_states=5000,
                    validate_batches=True,
                    jobs=1,
                    timeout=1,
                    max_pairs=10,
                    overwrite=False,
                    continue_on_error=True,
                )

            runs = _read_csv(out_dir / "reduction_study_runs.csv")
            summary = _read_csv(out_dir / "reduction_study_summary.csv")

        self.assertEqual(result["successes"], 1)
        self.assertEqual(result["failures"], 1)
        self.assertEqual(runs[0]["status"], "failed")
        self.assertEqual(runs[0]["optimize_status"], "failed")
        self.assertIn("boom", runs[0]["error_message"])
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["program"], "good")


def _make_inputs(root: Path, names: list[str]) -> list[Path]:
    inputs = root / "inputs"
    inputs.mkdir()
    paths = []
    for name in names:
        path = inputs / name
        path.write_text("int f(void){return 0;}\n", encoding="utf-8")
        paths.append(path)
    return paths


def _fake_optimizer(input_path: Path, out_dir: Path, passes_path: Path, **kwargs) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        out_dir / "states.csv",
        ["program", "state_id", "depth"],
        [{"program": Path(input_path).stem, "state_id": "S0000", "depth": "0"}],
    )
    _write_csv(
        out_dir / "batch_state_transitions.csv",
        ["program", "parent_state_id", "child_state_id"],
        [{"program": Path(input_path).stem, "parent_state_id": "S0000", "child_state_id": "S0001"}],
    )
    return {"states": 1, "batch_transitions": 1}


def _fake_reduction(run_dir: Path) -> dict:
    program = Path(run_dir).parent.name
    _write_csv(
        run_dir / "reduction_by_state.csv",
        ["program", "state_id", "dropped_active_passes"],
        [{"program": program, "state_id": "S0000", "dropped_active_passes": "0"}],
    )
    _write_csv(
        run_dir / "reduction_summary.csv",
        [
            "program",
            "total_states",
            "max_depth",
            "total_active_passes",
            "total_tested_pairs",
            "total_commute_pairs",
            "total_order_sensitive_pairs",
            "unknown_pairs",
            "total_batch_candidates",
            "total_certified_batches",
            "total_executable_batches",
            "total_executed_transitions",
            "total_skipped_batches",
            "total_dropped_active_passes",
            "avg_local_reduction_log10",
            "max_local_reduction_log10",
            "selected_path_steps",
            "final_pipeline_length",
        ],
        [
            {
                "program": program,
                "total_states": "1",
                "max_depth": "0",
                "total_active_passes": "4",
                "total_tested_pairs": "6",
                "total_commute_pairs": "4",
                "total_order_sensitive_pairs": "2",
                "unknown_pairs": "0",
                "total_batch_candidates": "4",
                "total_certified_batches": "3",
                "total_executable_batches": "3",
                "total_executed_transitions": "1",
                "total_skipped_batches": "1",
                "total_dropped_active_passes": "0",
                "avg_local_reduction_log10": "0.9",
                "max_local_reduction_log10": "1.2",
                "selected_path_steps": "1",
                "final_pipeline_length": "2",
            }
        ],
    )
    (run_dir / "reduction_summary.md").write_text("# Reduction Evidence Summary\n", encoding="utf-8")
    return {"states": 1}


def _fake_evidence(run_dir: Path) -> dict:
    program = Path(run_dir).parent.name
    _write_csv(
        run_dir / "evidence_pack.csv",
        [
            "program",
            "selected_path_batches",
            "selected_strong_certificates",
            "selected_weak_certificates",
            "selected_rejected",
            "executed_batches",
            "executed_strong_certificates",
            "executed_weak_certificates",
            "executed_rejected",
            "dropped_active_passes",
            "replay_status",
            "replay_hashes_match",
        ],
        [
            {
                "program": program,
                "selected_path_batches": "2",
                "selected_strong_certificates": "2",
                "selected_weak_certificates": "0",
                "selected_rejected": "0",
                "executed_batches": "3",
                "executed_strong_certificates": "3",
                "executed_weak_certificates": "0",
                "executed_rejected": "0",
                "dropped_active_passes": "0",
                "replay_status": "success",
                "replay_hashes_match": "true",
            }
        ],
    )
    (run_dir / "evidence_pack.md").write_text("# Evidence Pack\n", encoding="utf-8")
    return {"selected_batches": 2, "executed_batches": 3}


def _fake_replay(run_dir: Path, timeout: int = 10) -> dict:
    _write_csv(
        run_dir / "pipeline_replay.csv",
        ["replay_status", "hashes_match"],
        [{"replay_status": "success", "hashes_match": "true"}],
    )
    return {"replay_status": "success", "hashes_match": "true"}


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
