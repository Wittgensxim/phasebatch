from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from .schema import (
    EVIDENCE_PACK_FIELDS,
    EXECUTED_BATCH_CERTIFICATE_FIELDS,
    SELECTED_BATCH_CERTIFICATE_FIELDS,
)


def export_evidence_pack(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    state_dirs = _state_dirs(run_dir)
    selected_rows = [_selected_certificate(row, state_dirs) for row in _read_csv(run_dir / "chosen_path.csv")]
    executed_rows = [_executed_certificate(row, state_dirs) for row in _executed_transition_rows(run_dir)]
    replay = _first_row(run_dir / "pipeline_replay.csv")
    dropped = _dropped_active_passes(state_dirs.values())
    summary_rows = [_summary_row(run_dir, selected_rows, executed_rows, replay, dropped)]

    selected_path = run_dir / "selected_batch_certificates.csv"
    executed_path = run_dir / "executed_batch_certificates.csv"
    summary_path = run_dir / "evidence_pack.csv"
    md_path = run_dir / "evidence_pack.md"
    _write_csv(selected_path, SELECTED_BATCH_CERTIFICATE_FIELDS, selected_rows)
    _write_csv(executed_path, EXECUTED_BATCH_CERTIFICATE_FIELDS, executed_rows)
    _write_csv(summary_path, EVIDENCE_PACK_FIELDS, summary_rows)
    _write_markdown(md_path, selected_rows, executed_rows, summary_rows[0], replay)
    return {
        "evidence_pack_csv": str(summary_path),
        "evidence_pack_md": str(md_path),
        "selected_batch_certificates_csv": str(selected_path),
        "executed_batch_certificates_csv": str(executed_path),
        "selected_batches": len(selected_rows),
        "executed_batches": len(executed_rows),
    }


def _selected_certificate(row: dict, state_dirs: dict[str, Path]) -> dict:
    parent_id = row.get("parent_state_id", "")
    batch_id = row.get("batch_id", "")
    evidence = _batch_evidence(state_dirs.get(parent_id), batch_id)
    validation_status = evidence["validation"].get("validation_status") or row.get("validation_status", "")
    correctness_class = evidence["correctness"].get("correctness_class") or row.get("correctness_class", "")
    strength, note = _evidence_strength(validation_status, correctness_class)
    return {
        "step": row.get("step", ""),
        "parent_state_id": parent_id,
        "batch_id": batch_id,
        "batch_passes": row.get("batch_passes") or evidence["candidate"].get("batch_passes", ""),
        "canonical_order": row.get("canonical_order") or evidence["validation"].get("canonical_order") or evidence["candidate"].get("canonical_order", ""),
        "validation_status": validation_status,
        "correctness_class": correctness_class,
        "can_hard_fold": evidence["correctness"].get("can_hard_fold", ""),
        "can_execute": evidence["correctness"].get("can_execute", ""),
        "tested_orders": evidence["validation"].get("tested_orders", ""),
        "same_hash_count": evidence["validation"].get("same_hash_count", ""),
        "different_hash_count": evidence["validation"].get("different_hash_count", ""),
        "canonical_hash": evidence["validation"].get("canonical_hash", ""),
        "first_mismatch_order": evidence["validation"].get("first_mismatch_order", ""),
        "first_mismatch_hash": evidence["validation"].get("first_mismatch_hash", ""),
        "evidence_strength": strength,
        "evidence_note": note,
    }


def _executed_certificate(row: dict, state_dirs: dict[str, Path]) -> dict:
    parent_id = row.get("source_state_id") or row.get("parent_state_id", "")
    child_id = row.get("target_state_id") or row.get("child_state_id", "")
    batch_id = row.get("batch_id", "")
    evidence = _batch_evidence(state_dirs.get(parent_id), batch_id)
    validation_status = evidence["validation"].get("validation_status") or row.get("validation_status", "")
    correctness_class = evidence["correctness"].get("correctness_class") or row.get("correctness_class", "")
    strength, note = _evidence_strength(validation_status, correctness_class)
    return {
        "parent_state_id": parent_id,
        "child_state_id": child_id,
        "batch_id": batch_id,
        "batch_passes": row.get("batch_passes") or evidence["candidate"].get("batch_passes", ""),
        "validation_status": validation_status,
        "correctness_class": correctness_class,
        "can_hard_fold": evidence["correctness"].get("can_hard_fold", ""),
        "can_execute": evidence["correctness"].get("can_execute", ""),
        "is_duplicate_transition": row.get("is_duplicate_transition") or row.get("is_duplicate", ""),
        "duplicate_of": row.get("duplicate_of", ""),
        "evidence_strength": strength,
        "evidence_note": note,
    }


def _evidence_strength(validation_status: str, correctness_class: str) -> tuple[str, str]:
    if validation_status == "all_permutations_same" and correctness_class == "certified_batch":
        return "strong", "all tested permutations produced identical canonical IR"
    if validation_status == "sampled_same":
        return "weak", "sampled permutations matched; not a hard certificate"
    if validation_status == "mismatch":
        return "rejected", "at least one tested order produced different canonical IR"
    if validation_status == "failed":
        return "failed", "validation failed"
    return "unknown", "missing validation evidence"


def _summary_row(
    run_dir: Path,
    selected_rows: list[dict],
    executed_rows: list[dict],
    replay: dict,
    dropped_active_passes: int,
) -> dict:
    selected = Counter(row.get("evidence_strength", "") for row in selected_rows)
    executed = Counter(row.get("evidence_strength", "") for row in executed_rows)
    return {
        "program": _program(run_dir),
        "selected_path_batches": str(len(selected_rows)),
        "selected_strong_certificates": str(selected.get("strong", 0)),
        "selected_weak_certificates": str(selected.get("weak", 0)),
        "selected_rejected": str(selected.get("rejected", 0)),
        "executed_batches": str(len(executed_rows)),
        "executed_strong_certificates": str(executed.get("strong", 0)),
        "executed_weak_certificates": str(executed.get("weak", 0)),
        "executed_rejected": str(executed.get("rejected", 0)),
        "replay_status": replay.get("replay_status", ""),
        "replay_hashes_match": replay.get("hashes_match", ""),
        "dropped_active_passes": str(dropped_active_passes),
    }


def _write_markdown(path: Path, selected_rows: list[dict], executed_rows: list[dict], summary: dict, replay: dict) -> None:
    lines = [
        "# Evidence Pack",
        "",
        "## Selected Path Certificates",
        "",
        *_markdown_table(
            ["step", "parent", "batch", "validation", "correctness", "evidence strength", "tested orders", "different hashes"],
            [
                [
                    row.get("step", ""),
                    row.get("parent_state_id", ""),
                    row.get("batch_id", ""),
                    row.get("validation_status", ""),
                    row.get("correctness_class", ""),
                    row.get("evidence_strength", ""),
                    row.get("tested_orders", ""),
                    row.get("different_hash_count", ""),
                ]
                for row in selected_rows
            ],
        ),
        "",
        "## All Executed Batch Certificates",
        "",
        *_markdown_table(
            ["parent", "child", "batch", "validation", "correctness", "evidence strength", "duplicate"],
            [
                [
                    row.get("parent_state_id", ""),
                    row.get("child_state_id", ""),
                    row.get("batch_id", ""),
                    row.get("validation_status", ""),
                    row.get("correctness_class", ""),
                    row.get("evidence_strength", ""),
                    row.get("is_duplicate_transition", ""),
                ]
                for row in executed_rows
            ],
        ),
        "",
        "## Coverage",
        "",
        f"- dropped active passes: {summary.get('dropped_active_passes', '0')}",
        "",
        "## Final Pipeline Replay",
        "",
    ]
    if replay:
        lines.extend(
            [
                f"- replay status: {replay.get('replay_status', '')}",
                f"- hashes match: {replay.get('hashes_match', '')}",
            ]
        )
    else:
        lines.append('- WARNING: "Pipeline replay was not run."')
    lines.extend(
        [
            "",
            "## Correctness Boundary",
            "",
            "Only strong certificates may be used for hard folding. Weak or objective-only evidence is not used as commutation proof.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _batch_evidence(state_dir: Path | None, batch_id: str) -> dict[str, dict]:
    if state_dir is None:
        return {"validation": {}, "correctness": {}, "candidate": {}}
    return {
        "validation": _row_by_batch(state_dir / "batch_validation.csv", batch_id),
        "correctness": _row_by_batch(state_dir / "batch_correctness.csv", batch_id),
        "candidate": _row_by_batch(state_dir / "batch_candidates.csv", batch_id),
    }


def _row_by_batch(path: Path, batch_id: str) -> dict:
    for row in _read_csv(path):
        if row.get("batch_id", "") == batch_id:
            return row
    return {}


def _executed_transition_rows(run_dir: Path) -> list[dict]:
    rows = _read_csv(run_dir / "state_dag.csv")
    if rows:
        return rows
    return _read_csv(run_dir / "batch_state_transitions.csv")


def _state_dirs(run_dir: Path) -> dict[str, Path]:
    mapping = {
        row.get("state_id", ""): Path(row.get("state_dir") or run_dir / "states" / row.get("state_id", ""))
        for row in _read_csv(run_dir / "states.csv")
        if row.get("state_id")
    }
    states_root = run_dir / "states"
    if states_root.exists():
        for path in states_root.iterdir():
            if path.is_dir():
                mapping.setdefault(path.name, path)
    return mapping


def _dropped_active_passes(state_dirs) -> int:
    total = 0
    for state_dir in state_dirs:
        summary = _first_row(state_dir / "coverage_summary.csv")
        if summary:
            total += _int(summary.get("dropped_active_passes"))
        else:
            total += sum(1 for row in _read_csv(state_dir / "coverage_report.csv") if row.get("coverage_status") == "dropped")
    return total


def _program(run_dir: Path) -> str:
    for path in [run_dir / "states.csv", run_dir / "state_dag.csv", run_dir / "batch_state_transitions.csv"]:
        row = _first_row(path)
        if row.get("program"):
            return row["program"]
    return run_dir.name


def _first_row(path: Path) -> dict:
    rows = _read_csv(path)
    return rows[0] if rows else {}


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


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_escape_cell(value) for value in row) + " |")
    return lines


def _escape_cell(value: object) -> str:
    return " ".join(str(value).splitlines()).replace("|", "\\|")


def _int(value: object) -> int:
    try:
        return int(float(str(value or "0")))
    except ValueError:
        return 0
