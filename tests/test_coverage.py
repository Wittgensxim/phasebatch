import csv
import tempfile
import unittest
from pathlib import Path

from phasebatch.coverage import build_coverage_report


class CoverageReportTests(unittest.TestCase):
    def test_all_active_passes_in_certified_batch_are_certified_covered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_profiles(state_dir, ["A", "B"], inactive=["C"])
            _write_candidates(state_dir, [{"batch_id": "B0000", "batch_passes": "A;B"}])
            _write_correctness(state_dir, [{"batch_id": "B0000", "batch_passes": "A;B", "correctness_class": "certified_batch"}])
            _write_components(state_dir, [{"component_id": "C0000", "component_passes": "A;B", "is_exact": "true"}])

            rows = build_coverage_report(state_dir)
            summary = _read_csv(state_dir / "coverage_summary.csv")[0]

        self.assertEqual({row["active_pass"] for row in rows}, {"A", "B"})
        self.assertEqual({row["coverage_status"] for row in rows}, {"certified_covered"})
        self.assertEqual(summary["active_passes"], "2")
        self.assertEqual(summary["certified_covered"], "2")
        self.assertEqual(summary["dropped_active_passes"], "0")

    def test_active_pass_only_in_sampled_batch_is_heuristic_covered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_profiles(state_dir, ["A"])
            _write_candidates(state_dir, [{"batch_id": "B0000", "batch_passes": "A"}])
            _write_correctness(state_dir, [{"batch_id": "B0000", "batch_passes": "A", "correctness_class": "sampled_batch"}])

            rows = build_coverage_report(state_dir)

        self.assertEqual(rows[0]["coverage_status"], "heuristic_covered")
        self.assertEqual(rows[0]["correctness_classes"], "sampled_batch")

    def test_active_pass_only_in_rejected_batch_is_validation_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_profiles(state_dir, ["A"])
            _write_candidates(state_dir, [{"batch_id": "B0000", "batch_passes": "A"}])
            _write_correctness(state_dir, [{"batch_id": "B0000", "batch_passes": "A", "correctness_class": "rejected_batch"}])

            rows = build_coverage_report(state_dir)

        self.assertEqual(rows[0]["coverage_status"], "validation_rejected")

    def test_active_pass_in_unresolved_component_is_unresolved_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_profiles(state_dir, ["A"])
            _write_candidates(state_dir, [])
            _write_components(
                state_dir,
                [{"component_id": "C0000", "component_passes": "A", "is_exact": "false", "unresolved_reason": "component_size>10"}],
            )

            rows = build_coverage_report(state_dir)

        self.assertEqual(rows[0]["coverage_status"], "unresolved_conflict")
        self.assertEqual(rows[0]["component_ids"], "C0000")

    def test_active_pass_appears_nowhere_is_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_profiles(state_dir, ["A"])
            _write_candidates(state_dir, [])
            _write_components(state_dir, [])

            rows = build_coverage_report(state_dir)
            summary = _read_csv(state_dir / "coverage_summary.csv")[0]

        self.assertEqual(rows[0]["coverage_status"], "dropped")
        self.assertEqual(summary["dropped_active_passes"], "1")

    def test_terminal_state_active_pass_is_not_executed_due_to_max_depth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_profiles(state_dir, ["A"])
            _write_candidates(state_dir, [{"batch_id": "B0000", "batch_passes": "A"}])
            _write_correctness(state_dir, [{"batch_id": "B0000", "batch_passes": "A", "correctness_class": "unvalidated_batch"}])
            _write_components(state_dir, [{"component_id": "C0000", "component_passes": "A", "is_exact": "true"}])

            rows = build_coverage_report(state_dir, terminal_not_validated=True)
            summary = _read_csv(state_dir / "coverage_summary.csv")[0]

        self.assertEqual(rows[0]["coverage_status"], "not_executed_due_to_max_depth")
        self.assertEqual(rows[0]["reason"], "state reached max depth; candidate batches were built but not validated or executed")
        self.assertEqual(summary["not_executed_due_to_max_depth"], "1")
        self.assertEqual(summary["unvalidated_covered"], "0")


def _write_profiles(state_dir: Path, active: list[str], inactive: list[str] | None = None) -> None:
    rows = [
        {
            "program": "testprog",
            "state_id": "S0000",
            "state_hash": "hash0",
            "pass": pass_name,
            "success": "True",
            "active": "yes",
        }
        for pass_name in active
    ]
    rows.extend(
        {
            "program": "testprog",
            "state_id": "S0000",
            "state_hash": "hash0",
            "pass": pass_name,
            "success": "true",
            "active": "false",
        }
        for pass_name in (inactive or [])
    )
    _write_csv(state_dir / "pass_profile.csv", ["program", "state_id", "state_hash", "pass", "success", "active"], rows)


def _write_candidates(state_dir: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["program", "state_id", "state_hash", "batch_id", "batch_passes", "batch_size"]
    for row in rows:
        row.setdefault("program", "testprog")
        row.setdefault("state_id", "S0000")
        row.setdefault("state_hash", "hash0")
        row.setdefault("batch_size", str(len([part for part in row.get("batch_passes", "").split(";") if part])))
    _write_csv(state_dir / "batch_candidates.csv", fieldnames, rows)


def _write_correctness(state_dir: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "program",
        "state_id",
        "state_hash",
        "batch_id",
        "batch_passes",
        "batch_size",
        "validation_status",
        "correctness_class",
        "can_hard_fold",
        "can_execute",
        "reason",
    ]
    for row in rows:
        row.setdefault("program", "testprog")
        row.setdefault("state_id", "S0000")
        row.setdefault("state_hash", "hash0")
        row.setdefault("batch_size", str(len([part for part in row.get("batch_passes", "").split(";") if part])))
        row.setdefault("validation_status", "")
        row.setdefault("can_hard_fold", "false")
        row.setdefault("can_execute", "false")
        row.setdefault("reason", "")
    _write_csv(state_dir / "batch_correctness.csv", fieldnames, rows)


def _write_components(state_dir: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "program",
        "state_id",
        "state_hash",
        "component_id",
        "component_size",
        "component_passes",
        "conflict_edges",
        "commute_edges",
        "is_exact",
        "num_local_alternatives",
        "unresolved_reason",
    ]
    for row in rows:
        row.setdefault("program", "testprog")
        row.setdefault("state_id", "S0000")
        row.setdefault("state_hash", "hash0")
        row.setdefault("component_size", "1")
        row.setdefault("conflict_edges", "")
        row.setdefault("commute_edges", "")
        row.setdefault("is_exact", "true")
        row.setdefault("num_local_alternatives", "1")
        row.setdefault("unresolved_reason", "")
    _write_csv(state_dir / "batch_components.csv", fieldnames, rows)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
