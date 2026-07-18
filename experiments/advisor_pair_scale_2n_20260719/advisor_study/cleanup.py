"""Recoverable, space-bounded cleanup for isolated dynamic IR artifacts.

The study never discards an observation.  It first publishes a *planned*
ledger, then this module moves only hash-bound non-witness materializations
into an experiment-local quarantine before deleting them.  A failed move or
delete therefore leaves an exact, hash-bound path in the ledger rather than a
misleading all-or-nothing deletion claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import time
from typing import Any
from uuid import uuid4


CLEANUP_LEDGER_SCHEMA_VERSION = "advisor-pair-scale-cleanup-v2"
_SHA256_CHARS = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class CleanupResult:
    """Updated evidence rows plus a deterministic cleanup ledger payload."""

    pair_rows: tuple[dict[str, object], ...]
    directional_rows_by_group: Mapping[str, tuple[dict[str, object], ...]]
    ledger: dict[str, object]


def plan_intermediate_artifact_cleanup(
    *,
    isolation_root: Path,
    study_manifest_id: str,
    pair_rows: Sequence[Mapping[str, object]],
    directional_rows_by_group: Mapping[str, Sequence[Mapping[str, object]]],
    false_authorizations: Sequence[Mapping[str, object]],
) -> CleanupResult:
    """Create a no-delete hand-off which records exactly what may be reclaimed."""

    return _compact(
        isolation_root=isolation_root,
        study_manifest_id=study_manifest_id,
        pair_rows=pair_rows,
        directional_rows_by_group=directional_rows_by_group,
        false_authorizations=false_authorizations,
        mode="plan",
    )


def compact_intermediate_artifacts(
    *,
    isolation_root: Path,
    study_manifest_id: str,
    pair_rows: Sequence[Mapping[str, object]],
    directional_rows_by_group: Mapping[str, Sequence[Mapping[str, object]]],
    false_authorizations: Sequence[Mapping[str, object]],
    protected_pair_ids: Sequence[str] = (),
    protected_directionals: Sequence[tuple[str, str, str]] = (),
    planned_ledger: Mapping[str, object] | None = None,
) -> CleanupResult:
    """Reclaim only pre-published, hash-bound, non-witness materializations.

    Pair AB/BA outputs and 2N ``second_round.ll`` are the only reclaimable
    kinds.  Roots, profiles, merged inputs, terminal failures and all false
    authorization witnesses are retained.  Each reclaim attempts a quarantine
    move first; group moves are intentionally fail-closed rather than best
    effort.
    """

    return _compact(
        isolation_root=isolation_root,
        study_manifest_id=study_manifest_id,
        pair_rows=pair_rows,
        directional_rows_by_group=directional_rows_by_group,
        false_authorizations=false_authorizations,
        mode="execute",
        protected_pair_ids=set(str(value) for value in protected_pair_ids),
        protected_directionals={tuple(str(part) for part in value) for value in protected_directionals},
        planned_ledger=planned_ledger,
    )


def cleanup_journals_resolved(*, isolation_root: Path, ledger: Mapping[str, object]) -> bool:
    """Return whether every materialized cleanup journal is terminal.

    The CLI may advance the active hand-off from ``planned`` to ``complete``
    only after this check.  A process crash leaves a ``*_prepared`` state and
    therefore retains the already-published planned pointer for recovery.
    """

    entries = ledger.get("entries")
    if not isinstance(entries, list):
        return False
    journal_root = Path(isolation_root).resolve(strict=False) / "raw" / "cleanup-journal"
    terminal = {"deleted", "retained", "original"}
    for entry in entries:
        if not isinstance(entry, Mapping):
            return False
        cleanup_id = entry.get("cleanup_id")
        if not isinstance(cleanup_id, str) or len(cleanup_id) != 64:
            return False
        path = journal_root / f"{cleanup_id}.json"
        if not path.is_file():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            artifacts = raw.get("artifacts") if isinstance(raw, Mapping) else None
            if raw.get("cleanup_id") != cleanup_id or not isinstance(artifacts, Mapping):
                return False
            if any(not isinstance(value, Mapping) or str(value.get("state", "")) not in terminal for value in artifacts.values()):
                return False
        except (OSError, ValueError, json.JSONDecodeError):
            return False
    return True


def _compact(
    *,
    isolation_root: Path,
    study_manifest_id: str,
    pair_rows: Sequence[Mapping[str, object]],
    directional_rows_by_group: Mapping[str, Sequence[Mapping[str, object]]],
    false_authorizations: Sequence[Mapping[str, object]],
    mode: str,
    protected_pair_ids: set[str] | None = None,
    protected_directionals: set[tuple[str, str, str]] | None = None,
    planned_ledger: Mapping[str, object] | None = None,
) -> CleanupResult:
    if mode not in {"plan", "execute"}:
        raise ValueError("cleanup mode must be plan or execute")
    root = Path(isolation_root).resolve(strict=False)
    manifest = _nonempty(study_manifest_id, "study_manifest_id")
    observed_pair_ids, observed_directionals = _false_authorization_protection(false_authorizations)
    protected_pair_ids = observed_pair_ids | (protected_pair_ids or set())
    protected_directionals = observed_directionals | (protected_directionals or set())
    planned_entries = _planned_entries(planned_ledger, manifest)
    entries: list[dict[str, object]] = []
    compact_pairs: list[dict[str, object]] = []
    pair_path_references: dict[Path, int] = {}
    for raw in pair_rows:
        for _name, path, _digest in _pair_files(raw):
            if str(path):
                resolved = path.resolve(strict=False)
                pair_path_references[resolved] = pair_path_references.get(resolved, 0) + 1

    for raw in pair_rows:
        row = dict(raw)
        files = _pair_files(row)
        identity = _entry_identity(manifest, "pair_ab_ba", row)
        planned = planned_entries.get(("pair_ab_ba", str(row.get("row_id", ""))))
        reason = _pair_retention_reason(
            row, files, root, manifest, protected_pair_ids,
            path_references=pair_path_references,
            require_files=planned is None,
        )
        if reason:
            _mark_retained(row, "pair", reason)
            entries.append(_ledger_entry(identity, row, files, "retained", reason))
        elif mode == "plan":
            row["cleanup_status"] = "planned_nonwitness"
            entries.append(_ledger_entry(
                identity, row, files, "planned", "",
                artifacts=_planned_artifact_records(files, root, str(identity["cleanup_id"])),
            ))
        else:
            outcome = _quarantine_then_delete(
                files, root, str(identity["cleanup_id"]),
                planned_artifacts=_entry_artifacts(planned),
            )
            _apply_pair_outcome(row, outcome)
            entries.append(_ledger_entry(
                identity, row, files, str(outcome["cleanup_status"]),
                str(outcome["retention_reason"]), artifacts=outcome["artifacts"],
            ))
        compact_pairs.append(row)

    compact_directionals: dict[str, tuple[dict[str, object], ...]] = {}
    for group_id in sorted(directional_rows_by_group):
        compact: list[dict[str, object]] = []
        for raw in directional_rows_by_group[group_id]:
            row = dict(raw)
            files = _second_file(row)
            key = (str(row.get("group_id", "")), str(row.get("program_id", "")), str(row.get("action_id", "")))
            identity = _entry_identity(manifest, "two_n_second_round", row)
            planned = planned_entries.get(("two_n_second_round", str(row.get("row_id", ""))))
            reason = _second_retention_reason(
                row, files, root, manifest, protected_directionals, key,
                require_files=planned is None,
            )
            if reason:
                _mark_retained(row, "directional", reason)
                entries.append(_ledger_entry(identity, row, files, "retained", reason))
            elif mode == "plan":
                row["cleanup_status"] = "planned_nonwitness"
                row.setdefault("second_output_materialized", "true")
                entries.append(_ledger_entry(
                    identity, row, files, "planned", "",
                    artifacts=_planned_artifact_records(files, root, str(identity["cleanup_id"])),
                ))
            else:
                outcome = _quarantine_then_delete(
                    files, root, str(identity["cleanup_id"]),
                    planned_artifacts=_entry_artifacts(planned),
                )
                _apply_directional_outcome(row, outcome)
                entries.append(_ledger_entry(
                    identity, row, files, str(outcome["cleanup_status"]),
                    str(outcome["retention_reason"]), artifacts=outcome["artifacts"],
                ))
            compact.append(row)
        compact_directionals[group_id] = tuple(compact)

    ordered = sorted(entries, key=lambda item: str(item["cleanup_id"]))
    return CleanupResult(
        pair_rows=tuple(compact_pairs),
        directional_rows_by_group=compact_directionals,
        ledger={
            "schema_version": CLEANUP_LEDGER_SCHEMA_VERSION,
            "cleanup_state": "planned" if mode == "plan" else "complete",
            "study_manifest_id": manifest,
            "authority_granted": False,
            "proved_commute": False,
            "protected_pair_row_ids": sorted(protected_pair_ids),
            "protected_directionals": [list(value) for value in sorted(protected_directionals)],
            "entries": ordered,
            "summary": _summary(ordered),
        },
    )


def _pair_files(row: Mapping[str, object]) -> tuple[tuple[str, Path, str], ...]:
    return (
        ("AB", _path(row.get("ab_output_path")), str(row.get("ab_output_sha256", ""))),
        ("BA", _path(row.get("ba_output_path")), str(row.get("ba_output_sha256", ""))),
    )


def _second_file(row: Mapping[str, object]) -> tuple[tuple[str, Path, str], ...]:
    return (("second_round", _path(row.get("second_output_path")), str(row.get("second_output_sha256", ""))),)


def _pair_retention_reason(
    row: Mapping[str, object], files: Sequence[tuple[str, Path, str]], root: Path,
    manifest: str, protected_pair_ids: set[str], *,
    path_references: Mapping[Path, int] | None = None,
    require_files: bool = True,
) -> str:
    if str(row.get("study_manifest_id", "")) != manifest:
        return "retained_manifest_mismatch"
    if str(row.get("row_id", "")) in protected_pair_ids:
        return "retained_false_authorization"
    if any(str(row.get(field, "")) != "success" for field in ("ab_status", "ba_status", "ab_verifier_status", "ba_verifier_status")):
        return "retained_terminal_failure_witness"
    if str(row.get("dynamic_result", "")) != "commute":
        return "retained_noncommuting_or_unresolved"
    if str(row.get("artifact_available", "")).lower() != "true" or str(row.get("artifact_materialized", "")).lower() != "true":
        return "retained_unmaterialized"
    if path_references is not None and any(
        path_references.get(path.resolve(strict=False), 0) != 1
        for _name, path, _digest in files
    ):
        return "retained_referenced"
    if require_files and not _files_bound(files, root):
        return "retained_hash_or_provenance_unpersisted"
    return ""


def _second_retention_reason(
    row: Mapping[str, object], files: Sequence[tuple[str, Path, str]], root: Path,
    manifest: str, protected_directionals: set[tuple[str, str, str]], key: tuple[str, str, str], *,
    require_files: bool = True,
) -> str:
    if str(row.get("study_manifest_id", "")) != manifest:
        return "retained_manifest_mismatch"
    if key in protected_directionals:
        return "retained_false_authorization"
    if str(row.get("second_round_status", "")) != "success" or str(row.get("verifier_status", "")) != "success":
        return "retained_terminal_failure_witness"
    merged = _path(row.get("merged_input_path"))
    if str(row.get("merged_input_status", "")) != "complete" or not _file_bound(merged, str(row.get("merged_input_sha256", "")), root):
        return "retained_hash_or_provenance_unpersisted"
    if require_files and not _files_bound(files, root):
        return "retained_hash_or_provenance_unpersisted"
    return ""


def _quarantine_then_delete(
    files: Sequence[tuple[str, Path, str]], root: Path, cleanup_id: str, *,
    planned_artifacts: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Journal every move/delete so a planned crash can be reconciled exactly."""

    records = _copy_records(planned_artifacts) if planned_artifacts is not None else _planned_artifact_records(files, root, cleanup_id)
    _validate_planned_artifact_records(files, root, cleanup_id, records)
    journal_path = root / "raw" / "cleanup-journal" / f"{cleanup_id}.json"
    try:
        journal = _load_or_create_journal(journal_path, cleanup_id, records)
    except (OSError, ValueError):
        return _outcome("retained", "retained_cleanup_journal_invalid", records)

    # Reconcile an interrupted move/delete from its pre-operation tombstone.
    for name, record in records.items():
        state = _journal_state(journal, name)
        original = Path(str(record["original_path"]))
        quarantine = Path(str(record["quarantine_path"]))
        digest = str(record["sha256"])
        if not _cleanup_path_safe(original, root) or not _cleanup_path_safe(quarantine, root):
            return _outcome("retained", "retained_cleanup_path_became_unsafe", records)
        if quarantine.is_file():
            if not _file_bound(quarantine, digest, root):
                return _outcome("retained", "retained_quarantine_hash_drift", records)
            record["actual_path"] = str(quarantine)
            _journal_transition(journal_path, journal, name, "quarantined", str(quarantine))
        elif original.is_file():
            if not _file_bound(original, digest, root):
                return _outcome("retained", "retained_cleanup_race", records)
            record["actual_path"] = str(original)
        elif state in {"delete_prepared", "deleted"}:
            # The durable pre-delete tombstone plus absence from both locations
            # is the recovery proof of deletion; it is never inferred without
            # that journal record.
            record["actual_path"] = ""
            record["reclaimed"] = True
            _journal_transition(journal_path, journal, name, "deleted", "")
        else:
            return _outcome("retained", "retained_recovery_location_unknown", records)

    # Complete moves only after their durable move-prepared tombstone exists.
    for name, record in records.items():
        if bool(record["reclaimed"]) or str(record["actual_path"]) == str(record["quarantine_path"]):
            continue
        original = Path(str(record["original_path"]))
        quarantine = Path(str(record["quarantine_path"]))
        digest = str(record["sha256"])
        if (
            not _file_bound(original, digest, root)
            or not _cleanup_path_safe(quarantine, root)
        ):
            return _outcome("retained", "retained_cleanup_race", records)
        try:
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            if (
                not _file_bound(original, digest, root)
                or not _cleanup_path_safe(quarantine, root)
            ):
                return _outcome("retained", "retained_cleanup_path_became_unsafe", records)
            _journal_transition(journal_path, journal, name, "move_prepared", str(original))
            if (
                not _file_bound(original, digest, root)
                or not _cleanup_path_safe(quarantine, root)
            ):
                return _outcome("retained", "retained_cleanup_path_became_unsafe", records)
            os.replace(original, quarantine)
        except OSError:
            # The operation did not complete as an exception path.  Mark the
            # journal terminal only when the original evidence is still bound;
            # otherwise leave the prepared tombstone for crash recovery.
            if _file_bound(original, digest, root):
                _journal_transition(journal_path, journal, name, "retained", str(original))
            return _outcome("retained", "retained_quarantine_move_failed", records)
        record["actual_path"] = str(quarantine)
        _journal_transition(journal_path, journal, name, "quarantined", str(quarantine))

    # Delete from quarantine only after a durable delete-prepared tombstone.
    for name, record in records.items():
        if bool(record["reclaimed"]):
            continue
        target = Path(str(record["quarantine_path"]))
        digest = str(record["sha256"])
        if not _file_bound(target, digest, root) or not _cleanup_path_safe(target, root):
            return _outcome("retained", "retained_quarantine_hash_drift", records)
        try:
            _journal_transition(journal_path, journal, name, "delete_prepared", str(target))
            if not _file_bound(target, digest, root) or not _cleanup_path_safe(target, root):
                return _outcome("retained", "retained_cleanup_path_became_unsafe", records)
            target.unlink()
        except OSError as error:
            if any(bool(item["reclaimed"]) for item in records.values()):
                raise RuntimeError(
                    "partial pair cleanup remains recoverable in the planned checkpoint"
                ) from error
            if _file_bound(target, digest, root):
                _journal_transition(journal_path, journal, name, "retained", str(target))
            return _outcome("retained", "retained_quarantine_delete_failed", records)
        record["actual_path"] = ""
        record["reclaimed"] = True
        _journal_transition(journal_path, journal, name, "deleted", "")
    return _outcome("reclaimed", "", records)


def _outcome(status: str, reason: str, artifacts: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    return {"cleanup_status": status, "retention_reason": reason, "artifacts": artifacts}


def _apply_pair_outcome(row: dict[str, object], outcome: Mapping[str, object]) -> None:
    artifacts = outcome["artifacts"]
    assert isinstance(artifacts, Mapping)
    actual_paths: list[str] = []
    for name, field in (("AB", "ab_output_path"), ("BA", "ba_output_path")):
        artifact = artifacts[name]
        assert isinstance(artifact, Mapping)
        actual = str(artifact["actual_path"])
        actual_paths.append(actual)
        if actual:
            row[field] = actual
    if str(outcome["cleanup_status"]) == "reclaimed":
        row["artifact_materialized"] = "false"
    elif all(actual_paths):
        # A failed quarantine move can leave one artifact at its original path
        # and the other in quarantine.  Both paths still bind real evidence.
        row["artifact_materialized"] = "true"
    row["cleanup_status"] = (
        "reclaimed_nonwitness" if str(outcome["cleanup_status"]) == "reclaimed"
        else str(outcome["retention_reason"])
    )


def _apply_directional_outcome(row: dict[str, object], outcome: Mapping[str, object]) -> None:
    artifacts = outcome["artifacts"]
    assert isinstance(artifacts, Mapping)
    artifact = artifacts["second_round"]
    assert isinstance(artifact, Mapping)
    actual = str(artifact["actual_path"])
    if actual:
        row["second_output_path"] = actual
    if str(outcome["cleanup_status"]) == "reclaimed":
        row["second_output_materialized"] = "false"
    else:
        row["second_output_materialized"] = "true"
    row["cleanup_status"] = (
        "reclaimed_nonwitness" if str(outcome["cleanup_status"]) == "reclaimed"
        else str(outcome["retention_reason"])
    )


def _mark_retained(row: dict[str, object], kind: str, reason: str) -> None:
    row["cleanup_status"] = reason
    if kind == "directional":
        row.setdefault("second_output_materialized", "true")


def _entry_identity(manifest: str, artifact_kind: str, row: Mapping[str, object]) -> dict[str, object]:
    identity = {
        "study_manifest_id": manifest,
        "artifact_kind": artifact_kind,
        "source_row_id": str(row.get("row_id", "")),
        "group_id": str(row.get("group_id", "")),
        "program_id": str(row.get("program_id", "")),
        "action_id": str(row.get("action_id", "")),
    }
    identity["cleanup_id"] = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
    return identity


def _ledger_entry(
    identity: Mapping[str, object], row: Mapping[str, object], files: Sequence[tuple[str, Path, str]],
    cleanup_status: str, retention_reason: str, *, artifacts: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    records = artifacts or _artifact_records(files)
    ordered = [dict(records[name]) for name, _path_value, _digest in files]
    reclaimed = [item for item in ordered if item["reclaimed"]]
    retained = [item for item in ordered if not item["reclaimed"]]
    planned = ordered if cleanup_status == "planned" else []
    materialized_field = "artifact_materialized" if identity["artifact_kind"] == "pair_ab_ba" else "second_output_materialized"
    return {
        **identity,
        "cleanup_status": cleanup_status,
        "row_cleanup_status": str(row.get("cleanup_status", "")),
        "retention_reason": retention_reason,
        "artifact_materialized": str(row.get(materialized_field, "")),
        "file_count": len(ordered),
        "size_bytes": sum(_integer(item["size_bytes"]) for item in ordered),
        "reclaimed_file_count": len(reclaimed),
        "reclaimed_bytes": sum(_integer(item["size_bytes"]) for item in reclaimed),
        "retained_file_count": len(retained),
        "retained_bytes": sum(_integer(item["size_bytes"]) for item in retained),
        "planned_file_count": len(planned),
        "planned_bytes": sum(_integer(item["size_bytes"]) for item in planned),
        "artifacts": ordered,
    }


def _artifact_records(files: Sequence[tuple[str, Path, str]]) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for name, path, digest in files:
        # ``Path(\"\")`` renders as the process directory, whereas the raw
        # row deliberately records an absent artifact as an empty path.
        serialized = "" if path == Path() else str(path)
        records[name] = {
            "name": name,
            "original_path": serialized,
            "actual_path": serialized,
            "quarantine_path": "",
            "sha256": digest,
            "size_bytes": path.stat().st_size if serialized and path.is_file() else 0,
            "reclaimed": False,
        }
    return records


def _planned_artifact_records(
    files: Sequence[tuple[str, Path, str]], root: Path, cleanup_id: str,
) -> dict[str, dict[str, object]]:
    records = _artifact_records(files)
    quarantine_root = root / "raw" / "cleanup-quarantine" / cleanup_id
    for name, path, _digest in files:
        records[name]["quarantine_path"] = str(quarantine_root / path.name)
    return records


def _validate_planned_artifact_records(
    files: Sequence[tuple[str, Path, str]],
    root: Path,
    cleanup_id: str,
    records: Mapping[str, Mapping[str, object]],
) -> None:
    """Bind every destructive target to its row and deterministic quarantine."""

    expected_names = {name for name, _path, _digest in files}
    expected_fields = {
        "name",
        "original_path",
        "actual_path",
        "quarantine_path",
        "sha256",
        "size_bytes",
        "reclaimed",
    }
    if set(records) != expected_names or not _sha256(cleanup_id):
        raise ValueError("planned cleanup artifact binding is invalid")
    seen_quarantine: set[Path] = set()
    for name, path, digest in files:
        record = records.get(name)
        if not isinstance(record, Mapping):
            raise ValueError("planned cleanup artifact binding is invalid")
        original = path.resolve(strict=False)
        quarantine = (
            root / "raw" / "cleanup-quarantine" / cleanup_id / path.name
        )
        supplied_original = Path(str(record.get("original_path", "")))
        supplied_quarantine = Path(str(record.get("quarantine_path", "")))
        size_bytes = record.get("size_bytes")
        if (
            set(record) != expected_fields
            or str(record.get("name", "")) != name
            or str(record.get("original_path", "")) != str(original)
            or str(record.get("actual_path", "")) != str(original)
            or str(record.get("quarantine_path", "")) != str(quarantine)
            or str(record.get("sha256", "")) != digest
            or not _sha256(digest)
            or record.get("reclaimed") is not False
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
            or supplied_quarantine in seen_quarantine
            or not _cleanup_path_safe(supplied_original, root)
            or not _cleanup_path_safe(supplied_quarantine, root)
        ):
            raise ValueError("planned cleanup artifact binding is invalid")
        seen_quarantine.add(supplied_quarantine)
        physical = (
            supplied_original
            if supplied_original.is_file()
            else supplied_quarantine
            if supplied_quarantine.is_file()
            else None
        )
        if physical is not None:
            try:
                if physical.stat().st_size != size_bytes:
                    raise ValueError("planned cleanup artifact binding is invalid")
            except OSError as error:
                raise ValueError("planned cleanup artifact binding is invalid") from error


def _copy_records(source: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, object]]:
    return {str(name): dict(value) for name, value in source.items()}


def _planned_entries(
    ledger: Mapping[str, object] | None, manifest: str,
) -> dict[tuple[str, str], Mapping[str, object]]:
    if ledger is None:
        return {}
    if ledger.get("cleanup_state") != "planned" or ledger.get("study_manifest_id") != manifest:
        raise ValueError("planned cleanup ledger is not bound to this study")
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        raise ValueError("planned cleanup ledger entries are malformed")
    indexed: dict[tuple[str, str], Mapping[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("planned cleanup ledger entry is malformed")
        key = (str(entry.get("artifact_kind", "")), str(entry.get("source_row_id", "")))
        if key in indexed or not all(key):
            raise ValueError("planned cleanup ledger entry binding is ambiguous")
        indexed[key] = entry
    return indexed


def _entry_artifacts(entry: Mapping[str, object] | None) -> dict[str, dict[str, object]] | None:
    if entry is None:
        return None
    artifacts = entry.get("artifacts")
    if not isinstance(artifacts, list) or not all(isinstance(item, Mapping) for item in artifacts):
        raise ValueError("planned cleanup ledger artifacts are malformed")
    copied = {str(item.get("name", "")): dict(item) for item in artifacts}
    if not copied or "" in copied:
        raise ValueError("planned cleanup ledger artifact names are malformed")
    return copied


def _load_or_create_journal(
    path: Path, cleanup_id: str, records: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("cleanup_id") != cleanup_id or not isinstance(raw.get("artifacts"), dict):
            raise ValueError("cleanup journal is malformed")
        journal = raw
    else:
        journal = {
            "schema_version": "advisor-pair-scale-cleanup-journal-v1",
            "cleanup_id": cleanup_id,
            "artifacts": {
                name: {
                    "original_path": str(record["original_path"]),
                    "quarantine_path": str(record["quarantine_path"]),
                    "sha256": str(record["sha256"]),
                    "state": "original",
                    "actual_path": str(record["original_path"]),
                }
                for name, record in sorted(records.items())
            },
        }
        _write_journal(path, journal)
    artifacts = journal["artifacts"]
    assert isinstance(artifacts, dict)
    if set(artifacts) != set(records):
        raise ValueError("cleanup journal artifacts mismatch")
    for name, record in records.items():
        item = artifacts[name]
        if not isinstance(item, Mapping) or any(
            str(item.get(field, "")) != str(record[field])
            for field in ("original_path", "quarantine_path", "sha256")
        ):
            raise ValueError("cleanup journal binding mismatch")
    return journal


def _journal_state(journal: Mapping[str, object], name: str) -> str:
    artifacts = journal.get("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(artifacts.get(name), Mapping):
        raise ValueError("cleanup journal artifact state is missing")
    state = str(artifacts[name].get("state", ""))
    if state not in {"original", "move_prepared", "quarantined", "delete_prepared", "deleted", "retained"}:
        raise ValueError("cleanup journal state is invalid")
    return state


def _journal_transition(
    path: Path, journal: dict[str, object], name: str, state: str, actual_path: str,
) -> None:
    artifacts = journal["artifacts"]
    assert isinstance(artifacts, dict) and isinstance(artifacts[name], dict)
    artifacts[name]["state"] = state
    artifacts[name]["actual_path"] = actual_path
    _write_journal(path, journal)


def _write_journal(path: Path, journal: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(journal, sort_keys=True, separators=(",", ":")) + "\n"
    last_error: PermissionError | None = None
    # Defender/indexing can transiently hold a just-written journal on Windows.
    # A unique staging name avoids clobbering a crash tombstone; bounded retries
    # retain the old journal if the target remains unavailable.
    for attempt in range(6):
        # Keep the transient basename short enough for ordinary Windows path
        # limits even though the durable journal name contains a full SHA-256.
        staging = path.with_name(f".{uuid4().hex}.journal-tmp")
        staging.write_text(content, encoding="utf-8", newline="\n")
        try:
            os.replace(staging, path)
            return
        except PermissionError as error:
            last_error = error
            try:
                staging.unlink()
            except OSError:
                pass
            time.sleep(0.05 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _summary(entries: Sequence[Mapping[str, object]]) -> dict[str, int]:
    return {
        field: sum(_integer(item.get(field)) for item in entries)
        for field in (
            "reclaimed_file_count", "reclaimed_bytes", "retained_file_count",
            "retained_bytes", "planned_file_count", "planned_bytes",
        )
    }


def _false_authorization_protection(rows: Sequence[Mapping[str, object]]) -> tuple[set[str], set[tuple[str, str, str]]]:
    pair_ids: set[str] = set()
    directionals: set[tuple[str, str, str]] = set()
    for witness in rows:
        case = witness.get("case", witness)
        if not isinstance(case, Mapping):
            continue
        observed = case.get("pair_observation", case.get("pair_observation_row", {}))
        if isinstance(observed, Mapping) and str(observed.get("row_id", "")):
            pair_ids.add(str(observed["row_id"]))
        advisor = case.get("advisor_pair_row", witness.get("advisor_pair_row", {}))
        if isinstance(advisor, Mapping):
            group, program = str(advisor.get("group_id", "")), str(advisor.get("program_id", ""))
            for field in ("action_a_id", "action_b_id"):
                action = str(advisor.get(field, ""))
                if group and program and action:
                    directionals.add((group, program, action))
    return pair_ids, directionals


def _files_bound(files: Sequence[tuple[str, Path, str]], root: Path) -> bool:
    return bool(files) and all(_file_bound(path, digest, root) for _name, path, digest in files)


def _file_bound(path: Path, digest: str, root: Path) -> bool:
    if (
        not _sha256(digest)
        or not _cleanup_path_safe(path, root)
        or not path.is_file()
    ):
        return False
    try:
        return _sha256_file(path) == digest
    except OSError:
        return False


def _path(value: object) -> Path:
    return Path(str(value)).resolve(strict=False) if str(value) else Path()


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _cleanup_path_safe(path: Path, root: Path) -> bool:
    """Require an in-root lexical path with no symlink or Windows reparse hop."""

    root = root.resolve(strict=False)
    supplied = Path(path)
    try:
        relative = supplied.relative_to(root)
        supplied.resolve(strict=False).relative_to(root)
    except ValueError:
        return False
    current = root
    for part in relative.parts:
        current /= part
        if not os.path.lexists(current):
            continue
        try:
            metadata = current.lstat()
        except OSError:
            return False
        attributes = int(getattr(metadata, "st_file_attributes", 0))
        reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
        if current.is_symlink() or bool(attributes & reparse_flag):
            return False
    return True


def _identity(path: Path) -> tuple[int, int, int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: str) -> bool:
    return len(value) == 64 and all(character in _SHA256_CHARS for character in value.lower())


def _nonempty(value: object, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _integer(value: object) -> int:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
