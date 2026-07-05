import csv
import tempfile
import unittest
from pathlib import Path

from phasebatch.evidence_pack import export_evidence_pack


class EvidencePackTests(unittest.TestCase):
    def test_evidence_pack_classifies_selected_and_executed_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_mock_run(run_dir)

            result = export_evidence_pack(run_dir)

            selected = _read_csv(Path(result["selected_batch_certificates_csv"]))
            executed = _read_csv(Path(result["executed_batch_certificates_csv"]))
            summary = _read_csv(Path(result["evidence_pack_csv"]))
            markdown = Path(result["evidence_pack_md"]).read_text(encoding="utf-8")

        strengths = {row["batch_id"]: row["evidence_strength"] for row in selected}
        self.assertEqual(strengths["B0000"], "strong")
        self.assertEqual(strengths["B0001"], "weak")
        self.assertEqual(strengths["B0002"], "rejected")
        self.assertEqual(strengths["B0003"], "unknown")
        self.assertEqual(selected[0]["tested_orders"], "2")
        self.assertEqual(selected[0]["same_hash_count"], "2")
        self.assertEqual(selected[2]["different_hash_count"], "1")
        self.assertEqual(selected[3]["evidence_note"], "missing validation evidence")

        duplicate = next(row for row in executed if row["batch_id"] == "B0001")
        self.assertEqual(duplicate["is_duplicate_transition"], "true")
        self.assertEqual(duplicate["duplicate_of"], "S0001")
        self.assertEqual(duplicate["evidence_strength"], "weak")

        self.assertEqual(summary[0]["selected_path_batches"], "4")
        self.assertEqual(summary[0]["selected_strong_certificates"], "1")
        self.assertEqual(summary[0]["selected_weak_certificates"], "1")
        self.assertEqual(summary[0]["selected_rejected"], "1")
        self.assertEqual(summary[0]["executed_batches"], "3")
        self.assertEqual(summary[0]["replay_status"], "success")
        self.assertEqual(summary[0]["replay_hashes_match"], "true")
        self.assertEqual(summary[0]["dropped_active_passes"], "2")
        self.assertIn("# Evidence Pack", markdown)
        self.assertIn("Only strong certificates may be used for hard folding. Weak or objective-only evidence is not used as commutation proof.", markdown)
        self.assertIn("success", markdown)


def _make_mock_run(run_dir: Path) -> None:
    states_dir = run_dir / "states"
    for state_id in ["S0000", "S0001", "S0002", "S0003"]:
        (states_dir / state_id).mkdir(parents=True, exist_ok=True)

    _write_csv(
        run_dir / "states.csv",
        ["program", "state_id", "state_dir"],
        [
            {"program": "mock", "state_id": state_id, "state_dir": str(states_dir / state_id)}
            for state_id in ["S0000", "S0001", "S0002", "S0003"]
        ],
    )
    _write_csv(
        run_dir / "chosen_path.csv",
        ["step", "parent_state_id", "child_state_id", "batch_id", "batch_passes", "canonical_order"],
        [
            {"step": "0", "parent_state_id": "S0000", "child_state_id": "S0001", "batch_id": "B0000", "batch_passes": "a;b", "canonical_order": "a;b"},
            {"step": "1", "parent_state_id": "S0001", "child_state_id": "S0002", "batch_id": "B0001", "batch_passes": "c;d", "canonical_order": "c;d"},
            {"step": "2", "parent_state_id": "S0002", "child_state_id": "S0003", "batch_id": "B0002", "batch_passes": "e;f", "canonical_order": "e;f"},
            {"step": "3", "parent_state_id": "S0003", "child_state_id": "S0004", "batch_id": "B0003", "batch_passes": "g;h", "canonical_order": "g;h"},
        ],
    )
    _write_csv(
        run_dir / "state_dag.csv",
        [
            "program",
            "source_state_id",
            "target_state_id",
            "batch_id",
            "batch_passes",
            "validation_status",
            "correctness_class",
            "is_duplicate",
            "duplicate_of",
        ],
        [
            {"program": "mock", "source_state_id": "S0000", "target_state_id": "S0001", "batch_id": "B0000", "batch_passes": "a;b", "validation_status": "all_permutations_same", "correctness_class": "certified_batch", "is_duplicate": "false", "duplicate_of": ""},
            {"program": "mock", "source_state_id": "S0001", "target_state_id": "S0002", "batch_id": "B0001", "batch_passes": "c;d", "validation_status": "sampled_same", "correctness_class": "sampled_batch", "is_duplicate": "true", "duplicate_of": "S0001"},
            {"program": "mock", "source_state_id": "S0002", "target_state_id": "S0003", "batch_id": "B0002", "batch_passes": "e;f", "validation_status": "mismatch", "correctness_class": "rejected_batch", "is_duplicate": "false", "duplicate_of": ""},
        ],
    )
    _write_csv(
        run_dir / "pipeline_replay.csv",
        ["replay_status", "hashes_match"],
        [{"replay_status": "success", "hashes_match": "true"}],
    )

    _state_evidence(states_dir / "S0000", "B0000", "all_permutations_same", "certified_batch", "true", "true", tested="2", same="2", different="0")
    _state_evidence(states_dir / "S0001", "B0001", "sampled_same", "sampled_batch", "false", "false", tested="20", same="20", different="0")
    _state_evidence(states_dir / "S0002", "B0002", "mismatch", "rejected_batch", "false", "false", tested="2", same="1", different="1", mismatch_order="f;e", mismatch_hash="bad")
    _write_csv(
        states_dir / "S0003" / "batch_candidates.csv",
        ["batch_id", "batch_passes", "canonical_order"],
        [{"batch_id": "B0003", "batch_passes": "g;h", "canonical_order": "g;h"}],
    )
    _write_csv(
        states_dir / "S0000" / "coverage_report.csv",
        ["active_pass", "coverage_status"],
        [{"active_pass": "a", "coverage_status": "dropped"}],
    )
    _write_csv(
        states_dir / "S0001" / "coverage_report.csv",
        ["active_pass", "coverage_status"],
        [{"active_pass": "c", "coverage_status": "certified_covered"}],
    )
    _write_csv(
        states_dir / "S0002" / "coverage_report.csv",
        ["active_pass", "coverage_status"],
        [{"active_pass": "e", "coverage_status": "dropped"}],
    )


def _state_evidence(
    state_dir: Path,
    batch_id: str,
    validation_status: str,
    correctness_class: str,
    can_hard_fold: str,
    can_execute: str,
    *,
    tested: str,
    same: str,
    different: str,
    mismatch_order: str = "",
    mismatch_hash: str = "",
) -> None:
    _write_csv(
        state_dir / "batch_validation.csv",
        [
            "batch_id",
            "canonical_order",
            "tested_orders",
            "same_hash_count",
            "different_hash_count",
            "validation_status",
            "canonical_hash",
            "first_mismatch_order",
            "first_mismatch_hash",
        ],
        [
            {
                "batch_id": batch_id,
                "canonical_order": "order",
                "tested_orders": tested,
                "same_hash_count": same,
                "different_hash_count": different,
                "validation_status": validation_status,
                "canonical_hash": "hash",
                "first_mismatch_order": mismatch_order,
                "first_mismatch_hash": mismatch_hash,
            }
        ],
    )
    _write_csv(
        state_dir / "batch_correctness.csv",
        ["batch_id", "batch_passes", "validation_status", "correctness_class", "can_hard_fold", "can_execute"],
        [
            {
                "batch_id": batch_id,
                "batch_passes": "passes",
                "validation_status": validation_status,
                "correctness_class": correctness_class,
                "can_hard_fold": can_hard_fold,
                "can_execute": can_execute,
            }
        ],
    )
    _write_csv(
        state_dir / "batch_candidates.csv",
        ["batch_id", "batch_passes", "canonical_order"],
        [{"batch_id": batch_id, "batch_passes": "passes", "canonical_order": "order"}],
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
