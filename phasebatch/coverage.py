from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from .schema import COVERAGE_REPORT_FIELDS, COVERAGE_SUMMARY_FIELDS


COVERAGE_STATUSES = [
    "certified_covered",
    "heuristic_covered",
    "unresolved_conflict",
    "validation_rejected",
    "unvalidated_covered",
    "failed_or_unknown",
    "not_executed_due_to_max_depth",
    "dropped",
]


def build_coverage_report(state_dir: Path, terminal_not_validated: bool = False) -> list[dict]:
    state_dir = Path(state_dir)
    active_profiles = [
        row
        for row in _read_csv(state_dir / "pass_profile.csv")
        if row.get("pass") and _is_true(row.get("success")) and _is_true(row.get("active"))
    ]
    candidates = _read_csv(state_dir / "batch_candidates.csv")
    components = _read_csv(state_dir / "batch_components.csv")
    correctness_by_batch = {
        row.get("batch_id", ""): row
        for row in _read_csv(state_dir / "batch_correctness.csv")
        if row.get("batch_id")
    }
    program, state_id, state_hash = _state_identity(active_profiles, candidates, components)

    candidate_passes = [(row, _split_passes(row.get("batch_passes", ""))) for row in candidates]
    component_passes = [(row, _split_passes(row.get("component_passes", ""))) for row in components]

    rows = []
    for profile in active_profiles:
        active_pass = profile["pass"]
        covering_candidates = [row for row, passes in candidate_passes if active_pass in passes]
        containing_components = [row for row, passes in component_passes if active_pass in passes]
        unresolved_components = [row for row in containing_components if _component_unresolved(row)]
        correctness_classes = [
            _correctness_class(correctness_by_batch.get(row.get("batch_id", "")))
            for row in covering_candidates
        ]
        coverage_status, reason = _coverage_status(
            covering_candidates=covering_candidates,
            containing_components=containing_components,
            unresolved_components=unresolved_components,
            correctness_classes=correctness_classes,
            terminal_not_validated=terminal_not_validated,
        )
        rows.append(
            {
                "program": profile.get("program", program),
                "state_id": profile.get("state_id", state_id),
                "state_hash": profile.get("state_hash", state_hash),
                "active_pass": active_pass,
                "coverage_status": coverage_status,
                "covered_by_batch_ids": _join_unique(row.get("batch_id", "") for row in covering_candidates),
                "component_ids": _join_unique(row.get("component_id", "") for row in containing_components),
                "correctness_classes": _join_unique(correctness_classes),
                "reason": reason,
            }
        )

    _write_csv(state_dir / "coverage_report.csv", COVERAGE_REPORT_FIELDS, rows)
    summary_rows = [_coverage_summary(program, state_id, state_hash, rows)]
    _write_csv(state_dir / "coverage_summary.csv", COVERAGE_SUMMARY_FIELDS, summary_rows)
    _append_coverage_summary(state_dir / "batch_summary.md", summary_rows[0])
    return rows


def _coverage_status(
    *,
    covering_candidates: list[dict],
    containing_components: list[dict],
    unresolved_components: list[dict],
    correctness_classes: list[str],
    terminal_not_validated: bool,
) -> tuple[str, str]:
    class_set = set(correctness_classes)
    if "certified_batch" in class_set:
        return "certified_covered", "active pass appears in at least one certified batch"
    if "sampled_batch" in class_set:
        return "heuristic_covered", "active pass appears in sampled batch only; not a hard certificate"
    if terminal_not_validated and (covering_candidates or containing_components):
        return (
            "not_executed_due_to_max_depth",
            "state reached max depth; candidate batches were built but not validated or executed",
        )
    if unresolved_components:
        return "unresolved_conflict", "active pass belongs to an unresolved conflict component"
    if covering_candidates and class_set and class_set <= {"rejected_batch", "failed_batch"}:
        return "validation_rejected", "active pass appears only in rejected or failed batches"
    if covering_candidates and (not class_set or class_set <= {"unvalidated_batch", "unknown_batch"}):
        return "unvalidated_covered", "active pass appears only in unvalidated or unknown batches"
    if covering_candidates:
        return "failed_or_unknown", "batch data is incomplete or has mixed non-executable correctness classes"
    if containing_components:
        return "failed_or_unknown", "active pass appears in a component but no batch candidate covers it"
    return "dropped", "active pass appears in no batch candidate or conflict component"


def _coverage_summary(program: str, state_id: str, state_hash: str, rows: list[dict]) -> dict:
    counts = Counter(row.get("coverage_status", "") for row in rows)
    return {
        "program": program,
        "state_id": state_id,
        "state_hash": state_hash,
        "active_passes": str(len(rows)),
        "certified_covered": str(counts.get("certified_covered", 0)),
        "heuristic_covered": str(counts.get("heuristic_covered", 0)),
        "unresolved_conflict": str(counts.get("unresolved_conflict", 0)),
        "validation_rejected": str(counts.get("validation_rejected", 0)),
        "unvalidated_covered": str(counts.get("unvalidated_covered", 0)),
        "failed_or_unknown": str(counts.get("failed_or_unknown", 0)),
        "not_executed_due_to_max_depth": str(counts.get("not_executed_due_to_max_depth", 0)),
        "dropped_active_passes": str(counts.get("dropped", 0)),
    }


def _append_coverage_summary(path: Path, summary: dict) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Batch Summary\n"
    marker = "\n## Coverage Invariant\n"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n"
    lines = [
        existing.rstrip(),
        "",
        "## Coverage Invariant",
        "",
        f"- total active passes: {summary['active_passes']}",
        f"- certified covered: {summary['certified_covered']}",
        f"- heuristic covered: {summary['heuristic_covered']}",
        f"- unresolved conflict: {summary['unresolved_conflict']}",
        f"- rejected: {summary['validation_rejected']}",
        f"- unknown/unvalidated: {int(summary['unvalidated_covered']) + int(summary['failed_or_unknown'])}",
        f"- not executed due max depth: {summary['not_executed_due_to_max_depth']}",
        f"- dropped: {summary['dropped_active_passes']}",
    ]
    if int(summary["dropped_active_passes"]) > 0:
        lines.append("- WARNING: at least one active pass was dropped from batch coverage.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _state_identity(*row_groups: list[dict]) -> tuple[str, str, str]:
    for rows in row_groups:
        if rows:
            row = rows[0]
            return row.get("program", ""), row.get("state_id", ""), row.get("state_hash", "")
    return "", "", ""


def _component_unresolved(row: dict) -> bool:
    return not _is_true(row.get("is_exact")) or bool(row.get("unresolved_reason"))


def _correctness_class(row: dict | None) -> str:
    if not row:
        return "unknown_batch"
    return row.get("correctness_class") or "unknown_batch"


def _split_passes(value: str) -> set[str]:
    return {part for part in str(value or "").split(";") if part}


def _join_unique(values) -> str:
    seen = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return ";".join(seen)


def _is_true(value: object) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
