from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

import advisor_study.cleanup as cleanup_module
from advisor_study.cleanup import (
    cleanup_journals_resolved,
    compact_intermediate_artifacts,
    plan_intermediate_artifact_cleanup,
)
from advisor_study.cli import _project_row


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, content: str) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    return _sha256(path), path.stat().st_size


def _successful_pair(root: Path, row_id: str = "pair-1") -> dict[str, object]:
    ab = root / "pairs" / row_id / "AB.ll"
    ba = root / "pairs" / row_id / "BA.ll"
    ab_sha, _ = _write(ab, "AB\n")
    ba_sha, _ = _write(ba, "BA\n")
    return {
        "row_id": row_id,
        "study_manifest_id": "manifest",
        "program_id": "p1",
        "action_a_id": "A",
        "action_b_id": "B",
        "ab_status": "success",
        "ba_status": "success",
        "ab_verifier_status": "success",
        "ba_verifier_status": "success",
        "dynamic_result": "commute",
        "ab_output_path": str(ab),
        "ba_output_path": str(ba),
        "ab_output_sha256": ab_sha,
        "ba_output_sha256": ba_sha,
        "artifact_available": "true",
        "artifact_materialized": "true",
    }


@pytest.mark.parametrize("dynamic_result", ("order_sensitive", "failed", "timeout", "unknown", ""))
def test_cleanup_retains_every_pair_not_proven_commuting(
    tmp_path: Path, dynamic_result: str,
) -> None:
    pair = _successful_pair(tmp_path)
    pair["dynamic_result"] = dynamic_result
    ab = Path(str(pair["ab_output_path"]))
    ba = Path(str(pair["ba_output_path"]))

    result = compact_intermediate_artifacts(
        isolation_root=tmp_path,
        study_manifest_id="manifest",
        pair_rows=[pair],
        directional_rows_by_group={},
        false_authorizations=(),
    )

    assert ab.is_file() and ba.is_file()
    assert result.ledger["entries"][0]["retention_reason"] == (
        "retained_noncommuting_or_unresolved"
    )


def test_cleanup_retains_pair_outputs_referenced_by_more_than_one_row(
    tmp_path: Path,
) -> None:
    first = _successful_pair(tmp_path, "shared")
    second = {**first, "row_id": "also-shared", "action_a_id": "C"}

    result = compact_intermediate_artifacts(
        isolation_root=tmp_path,
        study_manifest_id="manifest",
        pair_rows=[first, second],
        directional_rows_by_group={},
        false_authorizations=(),
    )

    assert Path(str(first["ab_output_path"])).is_file()
    assert Path(str(first["ba_output_path"])).is_file()
    assert {entry["retention_reason"] for entry in result.ledger["entries"]} == {
        "retained_referenced"
    }


def _successful_directional(root: Path, row_id: str = "directional-1") -> dict[str, object]:
    merged = root / "two_n" / "A" / "merged_input.ll"
    second = root / "two_n" / "A" / "second_round.ll"
    merged_sha, _ = _write(merged, "merged\n")
    second_sha, _ = _write(second, "second\n")
    return {
        "row_id": row_id,
        "study_manifest_id": "manifest",
        "program_id": "p1",
        "group_id": "U14",
        "action_id": "A",
        "merged_input_status": "complete",
        "merged_input_path": str(merged),
        "merged_input_sha256": merged_sha,
        "second_round_status": "success",
        "verifier_status": "success",
        "second_output_path": str(second),
        "second_output_sha256": second_sha,
    }


def test_cleanup_reclaims_only_hash_bound_nonwitness_pair_and_second_round_outputs(
    tmp_path: Path,
) -> None:
    pair = _successful_pair(tmp_path)
    directional = _successful_directional(tmp_path)
    ab, ba = Path(str(pair["ab_output_path"])), Path(str(pair["ba_output_path"]))
    merged, second = Path(str(directional["merged_input_path"])), Path(str(directional["second_output_path"]))
    expected_bytes = ab.stat().st_size + ba.stat().st_size + second.stat().st_size

    result = compact_intermediate_artifacts(
        isolation_root=tmp_path,
        study_manifest_id="manifest",
        pair_rows=[pair],
        directional_rows_by_group={"U14": [directional]},
        false_authorizations=(),
    )

    assert not ab.exists() and not ba.exists() and not second.exists()
    assert merged.is_file()
    assert result.pair_rows[0]["artifact_materialized"] == "false"
    assert result.pair_rows[0]["ab_output_path"] == str(ab)
    assert result.pair_rows[0]["ab_output_sha256"] == pair["ab_output_sha256"]
    assert result.directional_rows_by_group["U14"][0]["second_output_materialized"] == "false"
    assert result.directional_rows_by_group["U14"][0]["merged_input_path"] == str(merged)
    assert result.ledger["summary"] == {
        "reclaimed_file_count": 3,
        "reclaimed_bytes": expected_bytes,
        "retained_file_count": 0,
        "retained_bytes": 0,
        "planned_file_count": 0,
        "planned_bytes": 0,
    }
    assert {row["artifact_kind"] for row in result.ledger["entries"]} == {
        "pair_ab_ba",
        "two_n_second_round",
    }


def test_cleanup_plan_is_a_no_delete_recovery_point_with_exact_candidate_bytes(
    tmp_path: Path,
) -> None:
    pair = _successful_pair(tmp_path)
    directional = _successful_directional(tmp_path)
    paths = [
        Path(str(pair["ab_output_path"])), Path(str(pair["ba_output_path"])),
        Path(str(directional["second_output_path"])),
    ]
    expected_bytes = sum(path.stat().st_size for path in paths)

    planned = plan_intermediate_artifact_cleanup(
        isolation_root=tmp_path,
        study_manifest_id="manifest",
        pair_rows=[pair],
        directional_rows_by_group={"U14": [directional]},
        false_authorizations=(),
    )

    assert all(path.is_file() for path in paths)
    assert planned.ledger["cleanup_state"] == "planned"
    assert planned.ledger["summary"] == {
        "reclaimed_file_count": 0,
        "reclaimed_bytes": 0,
        "retained_file_count": 3,
        "retained_bytes": expected_bytes,
        "planned_file_count": 3,
        "planned_bytes": expected_bytes,
    }
    assert {entry["cleanup_status"] for entry in planned.ledger["entries"]} == {"planned"}


def test_cleanup_never_deletes_false_authorization_or_terminal_failure_witnesses(
    tmp_path: Path,
) -> None:
    protected_pair = _successful_pair(tmp_path, "pair-protected")
    protected_directional = _successful_directional(tmp_path, "directional-protected")
    terminal_pair = _successful_pair(tmp_path, "pair-terminal")
    terminal_pair["ab_status"] = "failed"
    terminal_pair["ab_verifier_status"] = "not_run"

    result = compact_intermediate_artifacts(
        isolation_root=tmp_path,
        study_manifest_id="manifest",
        pair_rows=[protected_pair, terminal_pair],
        directional_rows_by_group={"U14": [protected_directional]},
        false_authorizations=[
            {
                "case": {
                    "pair_observation": {"row_id": "pair-protected"},
                    "advisor_pair_row": {
                        "group_id": "U14",
                        "program_id": "p1",
                        "action_a_id": "A",
                        "action_b_id": "B",
                    },
                }
            }
        ],
    )

    assert Path(str(protected_pair["ab_output_path"])).is_file()
    assert Path(str(protected_pair["ba_output_path"])).is_file()
    assert Path(str(protected_directional["second_output_path"])).is_file()
    assert Path(str(terminal_pair["ab_output_path"])).is_file()
    assert {entry["retention_reason"] for entry in result.ledger["entries"]} == {
        "retained_false_authorization",
        "retained_terminal_failure_witness",
    }


def test_cleanup_rejects_unpersisted_or_hash_drifting_artifacts_before_deletion(
    tmp_path: Path,
) -> None:
    pair = _successful_pair(tmp_path)
    Path(str(pair["ab_output_path"])).write_text("drift\n", encoding="utf-8")

    result = compact_intermediate_artifacts(
        isolation_root=tmp_path,
        study_manifest_id="manifest",
        pair_rows=[pair],
        directional_rows_by_group={},
        false_authorizations=(),
    )

    assert Path(str(pair["ab_output_path"])).is_file()
    assert Path(str(pair["ba_output_path"])).is_file()
    assert result.pair_rows[0]["artifact_materialized"] == "true"
    assert result.ledger["entries"][0]["retention_reason"] == "retained_hash_or_provenance_unpersisted"


def test_cleanup_quarantines_pair_before_delete_and_records_exact_partial_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = _successful_pair(tmp_path)
    ab, ba = Path(str(pair["ab_output_path"])), Path(str(pair["ba_output_path"]))
    original_replace = cleanup_module.os.replace

    def fail_second_move(source: object, target: object) -> None:
        if Path(source).name == "BA.ll":
            raise OSError("injected second quarantine move failure")
        original_replace(source, target)

    monkeypatch.setattr(cleanup_module.os, "replace", fail_second_move)
    result = compact_intermediate_artifacts(
        isolation_root=tmp_path,
        study_manifest_id="manifest",
        pair_rows=[pair],
        directional_rows_by_group={},
        false_authorizations=(),
    )

    entry = result.ledger["entries"][0]
    artifacts = {item["name"]: item for item in entry["artifacts"]}
    assert entry["cleanup_status"] == "retained"
    assert entry["retention_reason"] == "retained_quarantine_move_failed"
    assert not ab.exists() and ba.is_file()
    assert Path(str(artifacts["AB"]["actual_path"])).is_file()
    assert artifacts["BA"]["actual_path"] == str(ba)
    assert result.pair_rows[0]["ab_output_path"] == artifacts["AB"]["actual_path"]
    assert result.pair_rows[0]["ba_output_path"] == str(ba)
    # A partial cleanup retains both artifacts (one in quarantine and one at
    # its original path), so the row must remain truthfully materialized.
    assert result.pair_rows[0]["artifact_materialized"] == "true"


@pytest.mark.parametrize("crash_point", ("move", "delete"))
def test_planned_cleanup_recovers_exactly_after_move_or_delete_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crash_point: str,
) -> None:
    """A durable pre-operation tombstone makes an interrupted plan resumable."""

    pair = _successful_pair(tmp_path, f"crash-{crash_point}")
    planned = plan_intermediate_artifact_cleanup(
        isolation_root=tmp_path,
        study_manifest_id="manifest",
        pair_rows=[pair],
        directional_rows_by_group={},
        false_authorizations=(),
    )
    ab, ba = Path(str(pair["ab_output_path"])), Path(str(pair["ba_output_path"]))
    if crash_point == "move":
        original_replace = cleanup_module.os.replace

        def crash_after_ab_move(source: object, target: object) -> None:
            original_replace(source, target)
            if Path(source).name == "AB.ll":
                raise KeyboardInterrupt("injected crash after AB move")

        monkeypatch.setattr(cleanup_module.os, "replace", crash_after_ab_move)
    else:
        original_unlink = Path.unlink

        def crash_after_ab_delete(path: Path, *args: object, **kwargs: object) -> None:
            original_unlink(path, *args, **kwargs)
            if path.name == "AB.ll":
                raise KeyboardInterrupt("injected crash after AB delete")

        monkeypatch.setattr(Path, "unlink", crash_after_ab_delete)
    with pytest.raises(KeyboardInterrupt, match="injected crash"):
        compact_intermediate_artifacts(
            isolation_root=tmp_path,
            study_manifest_id="manifest",
            pair_rows=planned.pair_rows,
            directional_rows_by_group={},
            false_authorizations=(),
            planned_ledger=planned.ledger,
        )
    planned_entry = planned.ledger["entries"][0]
    journal_path = tmp_path / "raw" / "cleanup-journal" / f"{planned_entry['cleanup_id']}.json"
    interrupted = __import__("json").loads(journal_path.read_text(encoding="utf-8"))
    ab_journal = interrupted["artifacts"]["AB"]
    if crash_point == "move":
        assert Path(ab_journal["quarantine_path"]).is_file()
        assert ab_journal["state"] == "move_prepared"
    else:
        assert not ab.exists()
        assert ab_journal["state"] == "delete_prepared"
    assert cleanup_journals_resolved(isolation_root=tmp_path, ledger=planned.ledger) is False
    monkeypatch.undo()

    resumed = compact_intermediate_artifacts(
        isolation_root=tmp_path,
        study_manifest_id="manifest",
        pair_rows=planned.pair_rows,
        directional_rows_by_group={},
        false_authorizations=(),
        planned_ledger=planned.ledger,
    )

    assert not ab.exists() and not ba.exists()
    assert resumed.pair_rows[0]["artifact_materialized"] == "false"
    entry = resumed.ledger["entries"][0]
    assert entry["cleanup_status"] == "reclaimed"
    assert cleanup_journals_resolved(isolation_root=tmp_path, ledger=resumed.ledger) is True
    journal = tmp_path / "raw" / "cleanup-journal" / f"{entry['cleanup_id']}.json"
    persisted = __import__("json").loads(journal.read_text(encoding="utf-8"))
    assert {item["state"] for item in persisted["artifacts"].values()} == {"deleted"}


def test_projected_successful_pair_persists_verifier_predicate_and_resumes_cleanup(
    tmp_path: Path,
) -> None:
    """The raw projection retains the predicate required by later recovery."""

    pair = _successful_pair(tmp_path, "projected")
    pair.update({"group_id": "Uall", "program_family": "fixture"})
    projected = _project_row("pair_observations.csv", pair)
    assert projected["ab_verifier_status"] == "success"
    assert projected["ba_verifier_status"] == "success"
    planned = plan_intermediate_artifact_cleanup(
        isolation_root=tmp_path,
        study_manifest_id="manifest",
        pair_rows=[projected],
        directional_rows_by_group={},
        false_authorizations=(),
    )
    resumed = compact_intermediate_artifacts(
        isolation_root=tmp_path,
        study_manifest_id="manifest",
        pair_rows=planned.pair_rows,
        directional_rows_by_group={},
        false_authorizations=(),
        planned_ledger=planned.ledger,
    )
    assert resumed.ledger["entries"][0]["cleanup_status"] == "reclaimed"


def test_journal_publication_retries_transient_windows_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = _successful_pair(tmp_path, "journal-retry")
    planned = plan_intermediate_artifact_cleanup(
        isolation_root=tmp_path,
        study_manifest_id="manifest",
        pair_rows=[pair],
        directional_rows_by_group={},
        false_authorizations=(),
    )
    original_replace = cleanup_module.os.replace
    failures = 0

    def transient_journal_lock(source: object, target: object) -> None:
        nonlocal failures
        target_path = Path(target)
        if target_path.parent.name == "cleanup-journal" and target_path.suffix == ".json" and failures < 2:
            failures += 1
            raise PermissionError("injected transient Windows journal lock")
        original_replace(source, target)

    monkeypatch.setattr(cleanup_module.os, "replace", transient_journal_lock)
    result = compact_intermediate_artifacts(
        isolation_root=tmp_path,
        study_manifest_id="manifest",
        pair_rows=planned.pair_rows,
        directional_rows_by_group={},
        false_authorizations=(),
        planned_ledger=planned.ledger,
    )

    assert failures == 2
    assert result.ledger["entries"][0]["cleanup_status"] == "reclaimed"
    assert cleanup_journals_resolved(isolation_root=tmp_path, ledger=result.ledger)


def test_planned_cleanup_rejects_out_of_root_quarantine_before_artifact_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = _successful_pair(tmp_path, "forged-quarantine")
    planned = plan_intermediate_artifact_cleanup(
        isolation_root=tmp_path,
        study_manifest_id="manifest",
        pair_rows=[pair],
        directional_rows_by_group={},
        false_authorizations=(),
    )
    forged = copy.deepcopy(planned.ledger)
    entry = forged["entries"][0]
    artifacts = {item["name"]: item for item in entry["artifacts"]}
    outside = tmp_path.parent / f"{tmp_path.name}-outside" / "AB.ll"
    artifacts["AB"]["quarantine_path"] = str(outside)
    entry["artifacts"] = [artifacts["AB"], artifacts["BA"]]

    original_replace = cleanup_module.os.replace
    original_unlink = Path.unlink
    artifact_moves: list[tuple[Path, Path]] = []
    artifact_deletes: list[Path] = []

    def observe_replace(source: object, target: object) -> None:
        source_path, target_path = Path(source), Path(target)
        if source_path.suffix == ".ll":
            artifact_moves.append((source_path, target_path))
        original_replace(source, target)

    def observe_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path.suffix == ".ll":
            artifact_deletes.append(path)
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(cleanup_module.os, "replace", observe_replace)
    monkeypatch.setattr(Path, "unlink", observe_unlink)

    with pytest.raises(ValueError, match="planned cleanup artifact binding"):
        compact_intermediate_artifacts(
            isolation_root=tmp_path,
            study_manifest_id="manifest",
            pair_rows=planned.pair_rows,
            directional_rows_by_group={},
            false_authorizations=(),
            planned_ledger=forged,
        )

    assert artifact_moves == []
    assert artifact_deletes == []
    assert Path(str(pair["ab_output_path"])).read_text(encoding="utf-8") == "AB\n"
    assert Path(str(pair["ba_output_path"])).read_text(encoding="utf-8") == "BA\n"
    assert not outside.exists()


def test_planned_cleanup_rejects_symlinked_artifact_before_move_or_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = _successful_pair(tmp_path, "symlinked-original")
    planned = plan_intermediate_artifact_cleanup(
        isolation_root=tmp_path,
        study_manifest_id="manifest",
        pair_rows=[pair],
        directional_rows_by_group={},
        false_authorizations=(),
    )
    ab = Path(str(pair["ab_output_path"]))
    target = tmp_path / "same-bytes-target.ll"
    target.write_text("AB\n", encoding="utf-8", newline="\n")
    ab.unlink()
    try:
        ab.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    original_replace = cleanup_module.os.replace
    original_unlink = Path.unlink
    artifact_moves: list[tuple[Path, Path]] = []
    artifact_deletes: list[Path] = []

    def observe_replace(source: object, destination: object) -> None:
        source_path, destination_path = Path(source), Path(destination)
        if source_path.suffix == ".ll":
            artifact_moves.append((source_path, destination_path))
        original_replace(source, destination)

    def observe_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path.suffix == ".ll":
            artifact_deletes.append(path)
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(cleanup_module.os, "replace", observe_replace)
    monkeypatch.setattr(Path, "unlink", observe_unlink)

    with pytest.raises(ValueError, match="planned cleanup artifact binding"):
        compact_intermediate_artifacts(
            isolation_root=tmp_path,
            study_manifest_id="manifest",
            pair_rows=planned.pair_rows,
            directional_rows_by_group={},
            false_authorizations=(),
            planned_ledger=planned.ledger,
        )

    assert artifact_moves == []
    assert artifact_deletes == []
    assert ab.is_symlink()
    assert target.read_text(encoding="utf-8") == "AB\n"


def test_planned_cleanup_rejects_windows_reparse_quarantine_ancestor_before_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = _successful_pair(tmp_path, "reparse-quarantine")
    planned = plan_intermediate_artifact_cleanup(
        isolation_root=tmp_path,
        study_manifest_id="manifest",
        pair_rows=[pair],
        directional_rows_by_group={},
        false_authorizations=(),
    )
    quarantine_root = tmp_path / "raw" / "cleanup-quarantine"
    quarantine_root.mkdir(parents=True)
    original_lstat = Path.lstat
    quarantine_metadata = original_lstat(quarantine_root)
    original_replace = cleanup_module.os.replace
    original_unlink = Path.unlink
    artifact_moves: list[tuple[Path, Path]] = []
    artifact_deletes: list[Path] = []

    class _ReparseMetadata:
        st_file_attributes = cleanup_module.stat.FILE_ATTRIBUTE_REPARSE_POINT
        st_mode = quarantine_metadata.st_mode

    def mark_quarantine_as_reparse(path: Path) -> object:
        if path == quarantine_root:
            return _ReparseMetadata()
        return original_lstat(path)

    def reject_artifact_move(source: object, destination: object) -> None:
        source_path, destination_path = Path(source), Path(destination)
        if source_path.suffix == ".ll":
            artifact_moves.append((source_path, destination_path))
        original_replace(source, destination)

    def reject_artifact_delete(path: Path, *args: object, **kwargs: object) -> None:
        if path.suffix == ".ll":
            artifact_deletes.append(path)
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", mark_quarantine_as_reparse)
    monkeypatch.setattr(cleanup_module.os, "replace", reject_artifact_move)
    monkeypatch.setattr(Path, "unlink", reject_artifact_delete)

    with pytest.raises(ValueError, match="planned cleanup artifact binding"):
        compact_intermediate_artifacts(
            isolation_root=tmp_path,
            study_manifest_id="manifest",
            pair_rows=planned.pair_rows,
            directional_rows_by_group={},
            false_authorizations=(),
            planned_ledger=planned.ledger,
        )

    assert artifact_moves == []
    assert artifact_deletes == []
    assert Path(str(pair["ab_output_path"])).is_file()
    assert Path(str(pair["ba_output_path"])).is_file()
