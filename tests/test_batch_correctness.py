import csv
import tempfile
import unittest
from pathlib import Path

from phasebatch.batch_correctness import classify_batch_correctness


class BatchCorrectnessTests(unittest.TestCase):
    def test_all_permutations_same_is_certified_and_executable(self) -> None:
        rows = _classify_one("all_permutations_same")

        self.assertEqual(rows[0]["correctness_class"], "certified_batch")
        self.assertEqual(rows[0]["can_hard_fold"], "true")
        self.assertEqual(rows[0]["can_execute"], "true")

    def test_sampled_same_is_not_executable_by_default(self) -> None:
        rows = _classify_one("sampled_same")

        self.assertEqual(rows[0]["correctness_class"], "sampled_batch")
        self.assertEqual(rows[0]["can_hard_fold"], "false")
        self.assertEqual(rows[0]["can_execute"], "false")

    def test_sampled_same_can_execute_when_explicitly_allowed(self) -> None:
        rows = _classify_one("sampled_same", allow_sampled_batches=True)

        self.assertEqual(rows[0]["correctness_class"], "sampled_batch")
        self.assertEqual(rows[0]["can_hard_fold"], "false")
        self.assertEqual(rows[0]["can_execute"], "true")

    def test_mismatch_is_rejected(self) -> None:
        rows = _classify_one("mismatch")

        self.assertEqual(rows[0]["correctness_class"], "rejected_batch")
        self.assertEqual(rows[0]["can_execute"], "false")

    def test_missing_validation_file_marks_all_batches_unvalidated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_candidates(state_dir)

            rows = classify_batch_correctness(state_dir)
            written = _read_csv(state_dir / "batch_correctness.csv")

        self.assertEqual(rows[0]["validation_status"], "not_validated")
        self.assertEqual(rows[0]["correctness_class"], "unvalidated_batch")
        self.assertEqual(rows[0]["can_execute"], "false")
        self.assertEqual(written, rows)


def _classify_one(status: str, allow_sampled_batches: bool = False) -> list[dict[str, str]]:
    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp)
        _write_candidates(state_dir)
        _write_validation(state_dir, status)
        return classify_batch_correctness(state_dir, allow_sampled_batches=allow_sampled_batches)


def _write_candidates(state_dir: Path) -> None:
    _write_csv(
        state_dir / "batch_candidates.csv",
        [
            "program",
            "state_id",
            "state_hash",
            "batch_id",
            "batch_passes",
            "batch_size",
            "component_choices",
            "is_exact",
            "num_conflict_components",
            "unresolved_components",
            "canonical_order",
        ],
        [
            {
                "program": "testprog",
                "state_id": "S0000",
                "state_hash": "hash0",
                "batch_id": "B0000",
                "batch_passes": "A;B",
                "batch_size": "2",
                "component_choices": "",
                "is_exact": "true",
                "num_conflict_components": "1",
                "unresolved_components": "0",
                "canonical_order": "A;B",
            }
        ],
    )


def _write_validation(state_dir: Path, status: str) -> None:
    _write_csv(
        state_dir / "batch_validation.csv",
        [
            "program",
            "state_id",
            "state_hash",
            "batch_id",
            "batch_size",
            "canonical_order",
            "tested_orders",
            "same_hash_count",
            "different_hash_count",
            "validation_status",
            "canonical_hash",
            "first_mismatch_order",
            "first_mismatch_hash",
            "time_ms",
        ],
        [
            {
                "program": "testprog",
                "state_id": "S0000",
                "state_hash": "hash0",
                "batch_id": "B0000",
                "batch_size": "2",
                "canonical_order": "A;B",
                "tested_orders": "2",
                "same_hash_count": "2",
                "different_hash_count": "0",
                "validation_status": status,
                "canonical_hash": "hash",
                "first_mismatch_order": "",
                "first_mismatch_hash": "",
                "time_ms": "1",
            }
        ],
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
