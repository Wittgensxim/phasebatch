from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from .schema import BATCH_CORRECTNESS_FIELDS


def classify_batch_correctness(state_dir: Path, allow_sampled_batches: bool = False) -> list[dict]:
    state_dir = Path(state_dir)
    candidates = _read_csv(state_dir / "batch_candidates.csv")
    validation_by_id = {
        row.get("batch_id", ""): row
        for row in _read_csv(state_dir / "batch_validation.csv")
        if row.get("batch_id")
    }
    rows = []
    for candidate in candidates:
        validation = validation_by_id.get(candidate.get("batch_id", ""), {})
        status = validation.get("validation_status") or "not_validated"
        classification = classify_validation_status(status, allow_sampled_batches=allow_sampled_batches)
        rows.append(
            {
                "program": candidate.get("program", validation.get("program", "")),
                "state_id": candidate.get("state_id", validation.get("state_id", "")),
                "state_hash": candidate.get("state_hash", validation.get("state_hash", "")),
                "batch_id": candidate.get("batch_id", ""),
                "batch_passes": candidate.get("batch_passes", ""),
                "batch_size": candidate.get("batch_size", validation.get("batch_size", "")),
                "validation_status": status,
                **classification,
            }
        )

    _write_csv(state_dir / "batch_correctness.csv", BATCH_CORRECTNESS_FIELDS, rows)
    _append_correctness_summary(state_dir / "batch_summary.md", rows)
    return rows


def classify_validation_status(status: str, *, allow_sampled_batches: bool = False) -> dict:
    normalized = status or "not_validated"
    if normalized == "all_permutations_same":
        return {
            "correctness_class": "certified_batch",
            "can_hard_fold": "true",
            "can_execute": "true",
            "reason": "all tested permutations produced identical canonical IR",
        }
    if normalized == "sampled_same":
        return {
            "correctness_class": "sampled_batch",
            "can_hard_fold": "false",
            "can_execute": _bool(allow_sampled_batches),
            "reason": "only sampled permutations matched; not a hard certificate",
        }
    if normalized == "mismatch":
        return {
            "correctness_class": "rejected_batch",
            "can_hard_fold": "false",
            "can_execute": "false",
            "reason": "at least one tested ordering produced a different canonical IR",
        }
    if normalized == "failed":
        return {
            "correctness_class": "failed_batch",
            "can_hard_fold": "false",
            "can_execute": "false",
            "reason": "validation failed, crashed, timed out, or produced invalid IR",
        }
    if normalized == "not_validated":
        return {
            "correctness_class": "unvalidated_batch",
            "can_hard_fold": "false",
            "can_execute": "false",
            "reason": "batch was not validated",
        }
    return {
        "correctness_class": "unknown_batch",
        "can_hard_fold": "false",
        "can_execute": "false",
        "reason": "unknown validation status",
    }


def skip_reason_for_correctness(row: dict) -> str:
    if row.get("can_execute") == "true":
        return ""
    correctness_class = row.get("correctness_class", "")
    if correctness_class == "sampled_batch":
        return "sampled_not_allowed"
    if correctness_class == "rejected_batch":
        return "validation_mismatch"
    if correctness_class == "failed_batch":
        return "validation_failed"
    if correctness_class == "unvalidated_batch":
        return "not_validated"
    return "unknown_correctness_class"


def _append_correctness_summary(path: Path, rows: list[dict]) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Batch Summary\n"
    marker = "\n## Correctness\n"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n"
    class_counts = Counter(row.get("correctness_class", "") for row in rows)
    executable_count = sum(1 for row in rows if row.get("can_execute") == "true")
    lines = [
        existing.rstrip(),
        "",
        "## Correctness",
        "",
        f"- total batch candidates: {len(rows)}",
        f"- certified_batch count: {class_counts.get('certified_batch', 0)}",
        f"- sampled_batch count: {class_counts.get('sampled_batch', 0)}",
        f"- rejected_batch count: {class_counts.get('rejected_batch', 0)}",
        f"- failed_batch count: {class_counts.get('failed_batch', 0)}",
        f"- unvalidated_batch count: {class_counts.get('unvalidated_batch', 0)}",
        f"- executable batch count: {executable_count}",
        f"- skipped batch count: {len(rows) - executable_count}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _bool(value: bool) -> str:
    return "true" if value else "false"
