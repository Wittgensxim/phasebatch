"""Content-addressed execution for the isolated pair/advisor-2N study.

This module is deliberately an adapter over injected experiment runners.  It
does not import Phasebatch authority, create a certificate, or select a batch.
Its only responsibility is preserving exact experiment evidence and sharing
the one Uall profile/pair execution without changing either oracle semantics or
the three independent group-level 2N checks.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Iterator
from uuid import uuid4

from .schema import canonical_row_id


_GROUP_IDS = ("U14", "U30", "Uall")
_REPLAY_ARTIFACTS = (
    "S",
    "A",
    "B",
    "AB",
    "BA",
    "merged_input",
    "second_round_output",
)
_REPLAY_STAGE_NAMES = ("A", "B", "AB", "BA")
RAW_EXECUTION_SEMANTICS_REVISION = (
    "advisor-pair-scale-raw-v6-rebased-command-binding"
)
STAGE_COMPLETION_SCHEMA_VERSION = "advisor-pair-scale-stage/v2"


@dataclass(frozen=True)
class OrchestrationDependencies:
    """Experiment-only side effects, supplied by the standalone CLI/tests."""

    profile_uall: Callable[[Path, tuple[object, ...], Path], Sequence[Mapping[str, object]]]
    run_uall_pairs: Callable[
        [Path, list[dict[str, object]], dict[str, object], Path],
        Sequence[Mapping[str, object]],
    ]
    run_group_two_n: Callable[
        [Path, str, dict[str, object], list[dict[str, object]], Path, list[dict[str, object]]],
        object,
    ]
    replay_worker: Callable[[dict[str, object], int, Path], Mapping[str, object]]
    replay_external_opt: Callable[[dict[str, object], int, Path], Mapping[str, object]]
    replay_two_n: Callable[[dict[str, object], int, Path], Mapping[str, object]]


@dataclass(frozen=True)
class OrchestrationResult:
    out_dir: Path
    study_manifest_id: str
    profile_rows: Mapping[str, tuple[dict[str, object], ...]]
    pair_views: Mapping[str, tuple[dict[str, object], ...]]
    two_n_results: Mapping[str, Mapping[str, object]]
    false_authorizations: tuple[dict[str, object], ...]
    stage_paths: Mapping[str, str]


def run_study_orchestration(
    *,
    out_dir: Path,
    isolation_root: Path,
    study_manifest_id: str,
    programs: Mapping[str, Path],
    groups: Mapping[str, Sequence[object]],
    dependencies: OrchestrationDependencies,
) -> OrchestrationResult:
    """Serialize one complete writer over the isolated raw evidence root."""

    root = _require_isolated_output(Path(out_dir), Path(isolation_root))
    with _exclusive_output_lock(root):
        return _run_study_orchestration_locked(
            out_dir=root,
            isolation_root=isolation_root,
            study_manifest_id=study_manifest_id,
            programs=programs,
            groups=groups,
            dependencies=dependencies,
        )


def _run_study_orchestration_locked(
    *,
    out_dir: Path,
    isolation_root: Path,
    study_manifest_id: str,
    programs: Mapping[str, Path],
    groups: Mapping[str, Sequence[object]],
    dependencies: OrchestrationDependencies,
) -> OrchestrationResult:
    """Run/resume raw evidence with Uall sharing and mandatory witness replay.

    Every callback writes only below its supplied directory.  A callback result
    is retained only behind a completion marker whose listed file hashes still
    match; any incomplete or mismatched stage is recreated in place.  Reusing a
    valid marker never rewrites evidence or invokes the associated runner.
    """

    root = _require_isolated_output(Path(out_dir), Path(isolation_root))
    manifest = _safe_digest(study_manifest_id, "study_manifest_id")
    _validate_groups(groups)
    if not programs:
        raise ValueError("orchestration requires at least one program")
    normalized_programs: list[tuple[str, Path]] = []
    seen_program_ids: set[str] = set()
    for raw_program_id, raw_root_ir in programs.items():
        program_id = _safe_name(raw_program_id, "program_id")
        if program_id in seen_program_ids:
            raise ValueError(f"duplicate normalized program_id: {program_id}")
        seen_program_ids.add(program_id)
        normalized_programs.append((program_id, Path(raw_root_ir)))
    # Windows still commonly imposes a 260-character path limit.  The full
    # digest remains in records/completion binding; the path component is its
    # deterministic 64-bit prefix to leave room for materialized IR names.
    raw_root = root / "raw" / _path_component(manifest) / "p"
    profiles_by_program: dict[str, tuple[dict[str, object], ...]] = {}
    views_by_group: dict[str, list[dict[str, object]]] = {group: [] for group in _GROUP_IDS}
    two_n_by_group: dict[str, dict[str, object]] = {
        group: {"group_rows": [], "directional_rows": [], "pair_rows": []}
        for group in _GROUP_IDS
    }
    all_pairs_by_program: dict[str, list[dict[str, object]]] = {}
    stage_paths: dict[str, str] = {}

    uall_actions = _action_map(groups["Uall"])
    for program_id, root_ir_raw in sorted(normalized_programs, key=lambda item: item[0]):
        root_ir = Path(root_ir_raw).resolve(strict=False)
        if not root_ir.is_file():
            raise ValueError(f"root IR is missing: {root_ir}")
        program_digest = _digest_text(f"{program_id}\0{_sha256_file(root_ir)}")
        program_dir = raw_root / _path_component(program_digest)
        profile_dir = program_dir / "profiles"

        profile_input = _execution_input_digest({
            "study_manifest_id": study_manifest_id,
            "program_id": program_id,
            "root_sha256": _sha256_file(root_ir),
            "uall_action_ids": sorted(uall_actions),
        })

        def make_profiles(directory: Path) -> object:
            rows = _canonical_rows(dependencies.profile_uall(root_ir, tuple(uall_actions.values()), directory))
            _require_exact_action_rows(rows, uall_actions, "Uall profile")
            _validate_first_round_artifacts(rows, directory)
            for row in rows:
                action_digest = _path_component(_digest_text(str(row["action_id"])))
                _write_json(
                    directory / action_digest / "profile.json",
                    _stage_record(
                        stage="profile",
                        study_manifest_id=study_manifest_id,
                        program_id=program_id,
                        group_id="Uall",
                        action_id=str(row["action_id"]),
                        payload=row,
                    ),
                )
            return {"rows": rows}

        profile_payload, profile_reused = _run_or_reuse_stage(
            profile_dir, make_profiles, expected_input_sha256=profile_input, isolation_root=root
        )
        profile_rows = _canonical_rows(_require_mapping_list(profile_payload, "rows"))
        profiles_by_program[program_id] = tuple(profile_rows)
        stage_paths[f"{program_id}:profiles"] = _relative(root, profile_dir)

        pair_group_digest = _digest_text("Uall\0" + "\0".join(sorted(uall_actions)))
        pair_dir = program_dir / "q" / _path_component(pair_group_digest)

        pair_input = _execution_input_digest({
            "study_manifest_id": study_manifest_id,
            "program_id": program_id,
            "uall_action_ids": sorted(uall_actions),
            "profiles": profile_rows,
            "cleanup_provenance_version": 1,
        })

        def make_pairs(directory: Path) -> object:
            rows = _canonical_rows(
                dependencies.run_uall_pairs(root_ir, profile_rows, uall_actions, directory)
            )
            _require_pair_rows(rows, uall_actions)
            for row in rows:
                pair_digest = _path_component(_ordered_pair_digest(
                    str(row["action_a_id"]), str(row["action_b_id"])
                ))
                _write_json(
                    directory / pair_digest / "pair.json",
                    _stage_record(
                        stage="pair_oracle",
                        study_manifest_id=study_manifest_id,
                        program_id=program_id,
                        group_id="Uall",
                        pair_id=str(row.get("row_id", pair_digest)),
                        payload=row,
                    ),
                )
            return {"rows": rows}

        pair_payload, _ = _run_or_reuse_stage(
            pair_dir, make_pairs, expected_input_sha256=pair_input, isolation_root=root
        )
        uall_pair_rows = _canonical_rows(_require_mapping_list(pair_payload, "rows"))
        all_pairs_by_program[program_id] = uall_pair_rows
        stage_paths[f"{program_id}:pairs:Uall"] = _relative(root, pair_dir)

        for group_id in _GROUP_IDS:
            action_map = _action_map(groups[group_id])
            group_digest = _digest_text(group_id + "\0" + "\0".join(sorted(action_map)))
            view_dir = program_dir / "v" / _path_component(group_digest)

            view_input = _execution_input_digest({
                "study_manifest_id": study_manifest_id,
                "program_id": program_id,
                "group_id": group_id,
                "action_ids": sorted(action_map),
                "uall_pair_rows": uall_pair_rows,
                "profile_input_sha256": profile_input,
                "profile_rows": profile_rows,
            })

            def make_view(directory: Path, *, action_ids: frozenset[str] = frozenset(action_map), name: str = group_id) -> object:
                rows = [
                    {**row, "group_id": name, "reused_uall_pair_artifact": "true"}
                    for row in uall_pair_rows
                    if str(row["action_a_id"]) in action_ids and str(row["action_b_id"]) in action_ids
                ]
                payload = {"rows": _canonical_rows(rows)}
                _write_json(
                    directory / "view_evidence.json",
                    _stage_record(
                        stage="pair_view",
                        study_manifest_id=study_manifest_id,
                        program_id=program_id,
                        group_id=name,
                        payload={
                            "execution_status": "complete",
                            "artifact_available": "true",
                            "artifact_materialized": "true",
                            "first_round_artifact_sha256": _digest_text(_canonical_json(profile_rows)),
                            "reused_uall_pair_artifacts": "true",
                        },
                    ),
                )
                return payload

            view_payload, _ = _run_or_reuse_stage(
                view_dir, make_view, expected_input_sha256=view_input, isolation_root=root
            )
            view_rows = _canonical_rows(_require_mapping_list(view_payload, "rows"))
            views_by_group[group_id].extend(view_rows)
            stage_paths[f"{program_id}:view:{group_id}"] = _relative(root, view_dir)

            two_n_dir = program_dir / "n" / _path_component(group_digest)

            two_n_input = _execution_input_digest({
                "study_manifest_id": study_manifest_id,
                "program_id": program_id,
                "group_id": group_id,
                "action_ids": sorted(action_map),
                "profiles": profile_rows,
                "pair_view": view_rows,
                "cleanup_provenance_version": 1,
            })

            def make_two_n(directory: Path, *, name: str = group_id, actions: dict[str, object] = action_map, view: list[dict[str, object]] = view_rows) -> object:
                value = dependencies.run_group_two_n(root_ir, name, actions, profile_rows, directory, view)
                normalized = _normalize_two_n_result(value)
                _validate_two_n_result(
                    normalized,
                    study_manifest_id=study_manifest_id,
                    program_id=program_id,
                    group_id=name,
                    actions=actions,
                    pair_view=view,
                )
                _validate_two_n_materialization(normalized["directional_rows"], directory)
                for row in normalized["directional_rows"]:
                    action_id = str(row.get("action_id", ""))
                    if action_id:
                        _write_json(
                            directory / _path_component(_digest_text(action_id)) / "directional.json",
                            _stage_record(
                                stage="advisor_2n",
                                study_manifest_id=study_manifest_id,
                                program_id=program_id,
                                group_id=name,
                                action_id=action_id,
                                payload=row,
                            ),
                        )
                merged_inputs = _materialized_merged_inputs(directory)
                _write_json(
                    directory / "two_n_evidence.json",
                    _stage_record(
                        stage="advisor_2n",
                        study_manifest_id=study_manifest_id,
                        program_id=program_id,
                        group_id=name,
                        payload={
                            "execution_status": "complete",
                            "artifact_available": "true" if merged_inputs or not normalized["directional_rows"] else "false",
                            "artifact_materialized": "true" if merged_inputs or not normalized["directional_rows"] else "false",
                            "first_round_artifact_sha256": _digest_text(_canonical_json(profile_rows)),
                            "merged_inputs": merged_inputs,
                        },
                    ),
                )
                return normalized

            two_n_payload, _ = _run_or_reuse_stage(
                two_n_dir, make_two_n, expected_input_sha256=two_n_input, isolation_root=root
            )
            normalized_two_n = _normalize_two_n_result(two_n_payload)
            for key in ("group_rows", "directional_rows", "pair_rows"):
                two_n_by_group[group_id][key].extend(normalized_two_n[key])
            stage_paths[f"{program_id}:two_n:{group_id}"] = _relative(root, two_n_dir)

    false_authorizations: list[dict[str, object]] = []
    for group_id in _GROUP_IDS:
        for row in two_n_by_group[group_id]["pair_rows"]:
            if str(row.get("false_authorization", "")).lower() != "true":
                continue
            program_id = _safe_name(str(row.get("program_id", "")) or _only_program_id(normalized_programs), "program_id")
            original = _find_original_pair(
                all_pairs_by_program[program_id], str(row.get("pair_observation_row_id", ""))
            )
            expected_pair_stages = _validate_original_pair(
                original,
                study_manifest_id=study_manifest_id,
                program_id=program_id,
                action_a_id=str(row.get("action_a_id", "")),
                action_b_id=str(row.get("action_b_id", "")),
                profiles=profiles_by_program[program_id],
            )
            directional_by_action = {
                str(directional.get("action_id", "")): directional
                for directional in two_n_by_group[group_id]["directional_rows"]
                if str(directional.get("program_id", "")) == program_id
            }
            row_witnesses: list[dict[str, object]] = []
            for authorized_action_id in _authorized_direction_ids(row):
                directional = directional_by_action.get(authorized_action_id)
                if directional is None:
                    raise ValueError(
                        "false authorization has no exact authorized directional row"
                    )
                case = {
                    "study_manifest_id": study_manifest_id,
                    "program_id": program_id,
                    "group_id": group_id,
                    "authorized_action_id": authorized_action_id,
                    "advisor_pair_row": dict(row),
                    "pair_observation": original,
                    "profile_input_sha256": _digest_text(
                        _canonical_json(profiles_by_program[program_id])
                    ),
                    "expected_pair_stages": expected_pair_stages,
                    "expected_hard_state_hashes": {
                        name: str(value["hard_state_id"])
                        for name, value in expected_pair_stages.items()
                        if value["execution_status"] == "success"
                    },
                    "expected_two_n": _expected_two_n_values(row, directional),
                }
                case_id = canonical_row_id(
                    "false-authorization",
                    study_manifest_id,
                    program_id,
                    group_id,
                    row.get("action_a_id", ""),
                    row.get("action_b_id", ""),
                    authorized_action_id,
                    row.get("pair_observation_row_id", ""),
                )
                replay_dir = (
                    root
                    / "replay"
                    / "false_authorizations"
                    / _path_component(_digest_text(case_id))
                )

                replay_input = _execution_input_digest(case)

                def make_replay(
                    directory: Path,
                    *,
                    replay_case: dict[str, object] = case,
                    replay_case_id: str = case_id,
                ) -> object:
                    started_ns = time.perf_counter_ns()
                    worker = _run_replay_family(
                        "worker", dependencies.replay_worker, replay_case, directory
                    )
                    external = _run_replay_family(
                        "external_opt",
                        dependencies.replay_external_opt,
                        replay_case,
                        directory,
                    )
                    two_n = _run_replay_family(
                        "two_n", dependencies.replay_two_n, replay_case, directory
                    )
                    stable, replay_status, family_statuses = _stable_replay(
                        worker, external, two_n, replay_case
                    )
                    return {
                        "case_id": replay_case_id,
                        "case": replay_case,
                        "worker": worker,
                        "external_opt": external,
                        "two_n": two_n,
                        "stable_false_authorization": "true" if stable else "false",
                        "replay_status": replay_status,
                        "family_statuses": family_statuses,
                        "replay_time_ms": max(
                            0, (time.perf_counter_ns() - started_ns) // 1_000_000
                        ),
                    }

                replay_payload, _ = _run_or_reuse_stage(
                    replay_dir,
                    make_replay,
                    expected_input_sha256=replay_input,
                    isolation_root=root,
                )
                witness = dict(replay_payload)
                witness["replay_relative_path"] = _relative(root, replay_dir)
                false_authorizations.append(witness)
                row_witnesses.append(witness)
                stage_paths[f"replay:{case_id}"] = _relative(root, replay_dir)
            _writeback_replay_results(row, row_witnesses)

    return OrchestrationResult(
        out_dir=root,
        study_manifest_id=study_manifest_id,
        profile_rows=profiles_by_program,
        pair_views={group: tuple(_canonical_rows(rows)) for group, rows in views_by_group.items()},
        two_n_results={group: _freeze_two_n(value) for group, value in two_n_by_group.items()},
        false_authorizations=tuple(sorted(false_authorizations, key=lambda row: str(row["case_id"]))),
        stage_paths=dict(sorted(stage_paths.items())),
    )


def _run_or_reuse_stage(
    directory: Path,
    produce: Callable[[Path], object],
    *,
    expected_input_sha256: str,
    isolation_root: Path,
) -> tuple[dict[str, object], bool]:
    _require_stage_path(directory, isolation_root)
    directory.parent.mkdir(parents=True, exist_ok=True)
    _recover_stage_publication(directory, isolation_root)
    cached = _load_complete_stage(directory, expected_input_sha256)
    if cached is not None:
        return cached, True
    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}.stage-", dir=directory.parent))
    _require_stage_path(staging, isolation_root)
    backup = _stage_backup_path(directory)
    moved_old = False
    try:
        prefixes = _rebase_path_prefixes(staging, directory)
        payload = _rebase_staging_paths(
            _json_value(produce(staging)), staging, directory, prefixes=prefixes
        )
        if not isinstance(payload, dict):
            raise ValueError("stage producer must return a mapping")
        _rebase_json_evidence_paths(staging, directory, prefixes=prefixes)
        _write_json(staging / "result.json", payload)
        _write_complete(staging, expected_input_sha256)
        if _load_complete_stage(staging, expected_input_sha256) is None:
            raise ValueError("staged completion marker failed self-validation")
        if directory.exists():
            if backup.exists():
                raise RuntimeError("stage publication backup was not recovered")
            os.replace(directory, backup)
            moved_old = True
        os.replace(staging, directory)
        published = _load_complete_stage(directory, expected_input_sha256)
        if published is None:
            raise ValueError("published active stage failed self-validation")
        if backup.exists():
            _safe_rmtree(backup, isolation_root)
    except Exception:
        if staging.exists():
            _safe_rmtree(staging, isolation_root)
        if moved_old and backup.exists() and not directory.exists():
            try:
                os.replace(backup, directory)
            except OSError as restore_error:
                raise RuntimeError(
                    "stage publication failed and old active could not be restored"
                ) from restore_error
        raise
    return published, False


def _stage_backup_path(directory: Path) -> Path:
    return directory.parent / f".{directory.name}.publication-backup"


def _load_any_complete_stage(directory: Path) -> dict[str, object] | None:
    """Validate a current-schema stage against the input digest it records."""

    completion = directory / "complete.json"
    if not completion.is_file():
        return None
    try:
        marker = json.loads(completion.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    input_digest = marker.get("input_sha256")
    if not isinstance(input_digest, str) or len(input_digest) != 64 or any(
        character not in "0123456789abcdef" for character in input_digest.lower()
    ):
        return None
    return _load_complete_stage(directory, input_digest)


def _recover_stage_publication(directory: Path, isolation_root: Path) -> None:
    """Resolve the two crash states of the active/backup publish protocol."""

    backup = _stage_backup_path(directory)
    _require_stage_path(backup, isolation_root)
    if not backup.exists():
        return
    backup_valid = _load_any_complete_stage(backup)
    if not directory.exists():
        if backup_valid is None:
            raise RuntimeError("stage publication has only an invalid backup")
        os.replace(backup, directory)
        return

    active_valid = _load_any_complete_stage(directory)
    if active_valid is not None:
        # The new active is self-hashed under the current schema/revision.  It
        # is now the durable copy, so the specific old backup may be removed.
        _safe_rmtree(backup, isolation_root)
        return
    if backup_valid is None:
        raise RuntimeError("stage publication active and backup are both invalid")

    # Preserve the invalid active for diagnosis, then restore the last valid
    # stage.  Never recursively delete an ambiguous publication state.
    orphan = directory.parent / f".{directory.name}.invalid-active-{uuid4().hex}"
    _require_stage_path(orphan, isolation_root)
    os.replace(directory, orphan)
    try:
        os.replace(backup, directory)
    except OSError:
        if not directory.exists() and orphan.exists():
            os.replace(orphan, directory)
        raise


def _load_complete_stage(directory: Path, expected_input_sha256: str) -> dict[str, object] | None:
    completion = directory / "complete.json"
    result = directory / "result.json"
    if not completion.is_file() or not result.is_file():
        return None
    try:
        marker = json.loads(completion.read_text(encoding="utf-8"))
        files = marker["files"]
        if (
            marker.get("schema_version") != STAGE_COMPLETION_SCHEMA_VERSION
            or marker.get("raw_execution_semantics_revision")
            != RAW_EXECUTION_SEMANTICS_REVISION
            or marker.get("input_sha256") != expected_input_sha256
            or marker.get("full_digest_identity") != expected_input_sha256
            or not isinstance(files, dict)
            or not files
        ):
            return None
        actual = _stage_file_hashes(directory)
        if files != actual:
            return None
        payload = json.loads(result.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_complete(directory: Path, input_sha256: str) -> None:
    files = _stage_file_hashes(directory)
    if "result.json" not in files:
        raise ValueError("content-addressed stage must write result.json")
    _write_json(
        directory / "complete.json",
        {
            "schema_version": STAGE_COMPLETION_SCHEMA_VERSION,
            "raw_execution_semantics_revision": RAW_EXECUTION_SEMANTICS_REVISION,
            "input_sha256": input_sha256,
            "full_digest_identity": input_sha256,
            "files": files,
        },
    )


def _stage_file_hashes(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): _sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "complete.json"
    }


def _rebase_json_evidence_paths(
    staging: Path,
    final_directory: Path,
    *,
    prefixes: tuple[str, str] | None = None,
) -> None:
    canonical_prefixes = prefixes or _rebase_path_prefixes(staging, final_directory)
    for path in sorted(staging.rglob("*.json")):
        if path.name == "complete.json":
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"stage JSON evidence is malformed: {path}") from error
        _write_json(
            path,
            _rebase_staging_paths(value, staging, final_directory, prefixes=canonical_prefixes),
        )


def _rebase_path_prefixes(staging: Path, final_directory: Path) -> tuple[str, str]:
    """Resolve the two stage roots exactly once before recursive path rebasing."""

    old = str(staging.resolve(strict=False))
    new = str(final_directory.resolve(strict=False))
    return old, new


def _rebase_staging_paths(
    value: object,
    staging: Path,
    final_directory: Path,
    *,
    prefixes: tuple[str, str] | None = None,
) -> object:
    old, new = prefixes or _rebase_path_prefixes(staging, final_directory)
    return _rebase_staging_paths_with_prefixes(value, old, new)


def _rebase_staging_paths_with_prefixes(value: object, old: str, new: str) -> object:
    if isinstance(value, str):
        if value == old:
            return new
        # A sibling such as ``<staging>-old`` is not an artifact inside the
        # staged directory.  Rebase only a complete, canonical path segment.
        return new + value[len(old) :] if value.startswith(old + os.sep) else value
    if isinstance(value, Mapping):
        rebased = {
            str(key): _rebase_staging_paths_with_prefixes(item, old, new)
            for key, item in value.items()
        }
        _rebind_rebased_command_hash(value, rebased)
        return rebased
    if isinstance(value, list):
        return [_rebase_staging_paths_with_prefixes(item, old, new) for item in value]
    if isinstance(value, tuple):
        return [_rebase_staging_paths_with_prefixes(item, old, new) for item in value]
    return value


def _rebind_rebased_command_hash(
    staged: Mapping[object, object], published: dict[str, object]
) -> None:
    """Preserve an exact command/hash binding across atomic path publication.

    Commands are evidence, but materialization arguments name files below the
    private staging directory.  Publication rebases those arguments to the
    durable stage path.  Validate the command identity that was actually
    supplied by the producer before replacing its hash with the identity of
    the published command.  An absent or forged source hash must never be
    repaired into apparently valid evidence.
    """

    if "command" not in staged or staged.get("command") == published.get("command"):
        return
    staged_command = staged.get("command")
    published_command = published.get("command")
    if not isinstance(staged_command, (list, tuple)) or not isinstance(
        published_command, list
    ):
        raise ValueError("staged command is malformed during path rebase")
    supplied = str(staged.get("command_sha256", "")).lower()
    actual = _digest_text("\0".join(str(part) for part in staged_command))
    if not _sha256_shaped(supplied) or supplied != actual:
        raise ValueError("staged command hash mismatch during path rebase")
    published["command_sha256"] = _digest_text(
        "\0".join(str(part) for part in published_command)
    )


def _run_replay_family(
    name: str,
    runner: Callable[[dict[str, object], int, Path], Mapping[str, object]],
    case: dict[str, object],
    directory: Path,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for repetition in (1, 2):
        repeat_dir = directory / name / f"repeat-{repetition}"
        repeat_dir.mkdir(parents=True, exist_ok=True)
        raw = dict(runner(case, repetition, repeat_dir))
        record = _validate_replay_record(raw, repeat_dir)
        _write_json(repeat_dir / "record.json", record)
        (repeat_dir / "stderr.txt").write_text(str(record["stderr"]), encoding="utf-8", newline="\n")
        (repeat_dir / "command.txt").write_text("\0".join(record["command"]), encoding="utf-8", newline="\n")
        records.append(record)
    return records


def _validate_replay_record(raw: Mapping[str, object], directory: Path) -> dict[str, object]:
    status = str(raw.get("status", "")).strip()
    hard_state_hashes = raw.get("hard_state_hashes")
    artifact_sha256 = raw.get("artifact_sha256")
    artifacts = raw.get("artifacts")
    command = raw.get("command")
    if (
        not status
        or not isinstance(hard_state_hashes, Mapping)
        or not isinstance(artifact_sha256, Mapping)
        or not isinstance(artifacts, Mapping)
    ):
        raise ValueError(
            "replay record requires status, hard_state_hashes, artifact_sha256, and artifacts"
        )
    if not isinstance(command, (list, tuple)) or not all(str(part) for part in command):
        raise ValueError("replay record requires non-empty command identity")
    normalized_state_hashes: dict[str, str] = {}
    normalized_artifact_sha256: dict[str, str] = {}
    normalized_artifacts: dict[str, str] = {}
    artifact_names = {str(name) for name in artifacts}
    if "S" not in artifact_names or not artifact_names.issubset(_REPLAY_ARTIFACTS):
        raise ValueError("replay record artifacts are missing S or contain unknown names")
    if set(artifact_sha256) != artifact_names or not set(hard_state_hashes).issubset(
        artifact_names
    ):
        raise ValueError("replay record artifact/hash names disagree")
    for name in sorted(artifact_names):
        state_digest = str(hard_state_hashes.get(name, ""))
        artifact_digest = str(artifact_sha256.get(name, ""))
        artifact = Path(str(artifacts.get(name, ""))).resolve(strict=False)
        if state_digest and not _sha256_shaped(state_digest):
            raise ValueError(f"replay record has invalid {name} hard-state hash")
        if not _sha256_shaped(artifact_digest):
            raise ValueError(f"replay record has invalid {name} artifact sha256")
        try:
            artifact.relative_to(directory.resolve())
        except ValueError as error:
            raise ValueError(f"replay artifact escapes its isolated directory: {artifact}") from error
        if not artifact.is_file() or _sha256_file(artifact) != artifact_digest.lower():
            raise ValueError(f"replay artifact is missing: {artifact}")
        if state_digest:
            normalized_state_hashes[name] = state_digest.lower()
        normalized_artifact_sha256[name] = artifact_digest.lower()
        normalized_artifacts[name] = str(artifact)
    stderr = str(raw.get("stderr", ""))
    two_n_result = raw.get("two_n_result", {})
    if two_n_result and not isinstance(two_n_result, Mapping):
        raise ValueError("replay two_n_result must be a mapping when present")
    stage_results = _validate_replay_stage_results(raw.get("stage_results"))
    normalized_two_n_result = {
        str(key): str(value) for key, value in dict(two_n_result).items()
    }
    merge_status = str(raw.get("merge_status", "complete"))
    merge_error_fingerprint = str(raw.get("merge_error_fingerprint", ""))
    if merge_status not in {"complete", "error", "not_applicable"}:
        raise ValueError(
            "replay merge_status must be complete, error, or not_applicable"
        )
    if merge_status == "error" and not _sha256_shaped(merge_error_fingerprint):
        raise ValueError("failed replay merge requires an error fingerprint")
    if merge_status == "not_applicable" and (
        merge_error_fingerprint or "merged_input" in normalized_artifact_sha256
    ):
        raise ValueError(
            "pair-only replay cannot carry merge failure or merged-input evidence"
        )
    _validate_replay_evidence_bindings(
        stage_results=stage_results,
        artifact_sha256=normalized_artifact_sha256,
        hard_state_hashes=normalized_state_hashes,
        two_n_result=normalized_two_n_result,
        merge_status=merge_status,
    )
    return {
        "status": status,
        "hard_state_hashes": normalized_state_hashes,
        "artifact_sha256": normalized_artifact_sha256,
        "artifacts": normalized_artifacts,
        "stderr": stderr,
        "stderr_sha256": _digest_text(stderr),
        "command": [str(part) for part in command],
        "command_sha256": _digest_text("\0".join(str(part) for part in command)),
        "two_n_result": normalized_two_n_result,
        "stage_results": stage_results,
        "merge_status": merge_status,
        "merge_error_fingerprint": merge_error_fingerprint,
    }


def _validate_replay_stage_results(value: object) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping) or set(value) != set(_REPLAY_STAGE_NAMES):
        raise ValueError("replay record requires exact A/B/AB/BA stage_results")
    normalized: dict[str, dict[str, str]] = {}
    for name in _REPLAY_STAGE_NAMES:
        raw = value[name]
        if not isinstance(raw, Mapping):
            raise ValueError(f"replay {name} stage result must be a mapping")
        stage = {
            field: str(raw.get(field, ""))
            for field in (
                "execution_status",
                "verifier_status",
                "hard_state_id",
                "output_sha256",
                "command_sha256",
                "stderr_sha256",
                "error_fingerprint",
            )
        }
        if not stage["execution_status"] or not stage["verifier_status"]:
            raise ValueError(f"replay {name} stage lacks execution/verifier status")
        for field in ("hard_state_id", "output_sha256"):
            if stage[field] and not _sha256_shaped(stage[field]):
                raise ValueError(f"replay {name} stage has invalid {field}")
        for field in ("command_sha256", "stderr_sha256", "error_fingerprint"):
            if not _sha256_shaped(stage[field]):
                raise ValueError(f"replay {name} stage has invalid {field}")
        if stage["execution_status"] == "success" and (
            stage["verifier_status"] != "success"
            or not stage["hard_state_id"]
            or not stage["output_sha256"]
        ):
            raise ValueError(f"replay {name} successful stage lacks verified output identity")
        if stage["execution_status"] != "success" and stage[
            "error_fingerprint"
        ] != _terminal_error_fingerprint(
            stage["execution_status"],
            stage["verifier_status"],
            stage["stderr_sha256"],
        ):
            raise ValueError(
                f"replay {name} terminal fingerprint is not bound to stage evidence"
            )
        normalized[name] = stage
    return normalized


def _validate_replay_evidence_bindings(
    *,
    stage_results: Mapping[str, Mapping[str, str]],
    artifact_sha256: Mapping[str, str],
    hard_state_hashes: Mapping[str, str],
    two_n_result: Mapping[str, str],
    merge_status: str,
) -> None:
    """Bind every replay claim to the exact bytes retained in its record."""

    if "S" not in hard_state_hashes:
        raise ValueError("replay root artifact S lacks a hard-state identity")
    for name in _REPLAY_STAGE_NAMES:
        stage = stage_results[name]
        status = stage["execution_status"]
        artifact_present = name in artifact_sha256
        if status == "success":
            if not artifact_present:
                raise ValueError(f"replay {name} successful stage lacks its artifact")
            if stage["output_sha256"] != artifact_sha256[name]:
                raise ValueError(f"replay {name} output claim does not bind its artifact")
            if stage["hard_state_id"] != hard_state_hashes.get(name, ""):
                raise ValueError(f"replay {name} hard-state claim does not bind its artifact")
            continue
        if stage["hard_state_id"]:
            raise ValueError(f"replay {name} terminal stage cannot claim a hard state")
        if artifact_present:
            if stage["output_sha256"] != artifact_sha256[name]:
                raise ValueError(f"replay {name} terminal output does not bind its artifact")
        elif stage["output_sha256"]:
            raise ValueError(f"replay {name} terminal output claim lacks its artifact")
        if status == "not_run" and artifact_present:
            raise ValueError(f"replay {name} not-run stage cannot retain a fabricated artifact")

    if merge_status == "complete" and (
        "merged_input" not in artifact_sha256
        or "merged_input" not in hard_state_hashes
    ):
        raise ValueError("completed replay merge lacks bound merged_input evidence")

    if not two_n_result:
        return
    if "second_round_output" not in artifact_sha256:
        raise ValueError("2N replay lacks a bound second_round_output artifact")
    if "second_round_output" not in hard_state_hashes:
        raise ValueError("2N replay second output lacks a hard-state identity")
    if "merged_input" not in artifact_sha256 or "merged_input" not in hard_state_hashes:
        raise ValueError("2N replay lacks a bound full-group merged_input artifact")
    for result_field, artifact_name, evidence in (
        ("merged_input_sha256", "merged_input", artifact_sha256),
        ("merged_input_hard_state_id", "merged_input", hard_state_hashes),
        ("second_output_sha256", "second_round_output", artifact_sha256),
        ("second_output_hard_state_id", "second_round_output", hard_state_hashes),
    ):
        claimed = two_n_result.get(result_field, "")
        if not _sha256_shaped(claimed) or claimed != evidence[artifact_name]:
            raise ValueError(
                f"2N replay {result_field} does not bind {artifact_name} artifact"
            )


def _validate_replay_record_evidence_claims(record: Mapping[str, object]) -> None:
    """Revalidate persisted replay semantics without trusting stored labels."""

    artifacts = record.get("artifacts")
    artifact_sha256 = record.get("artifact_sha256")
    hard_state_hashes = record.get("hard_state_hashes")
    two_n_result = record.get("two_n_result")
    if (
        not isinstance(artifacts, Mapping)
        or not isinstance(artifact_sha256, Mapping)
        or not isinstance(hard_state_hashes, Mapping)
        or not isinstance(two_n_result, Mapping)
    ):
        raise ValueError("replay evidence maps are malformed")
    artifact_names = {str(name) for name in artifacts}
    if (
        "S" not in artifact_names
        or not artifact_names.issubset(_REPLAY_ARTIFACTS)
        or {str(name) for name in artifact_sha256} != artifact_names
        or not {str(name) for name in hard_state_hashes}.issubset(artifact_names)
    ):
        raise ValueError("replay artifact evidence names disagree")
    normalized_artifact_sha256 = {
        str(name): str(value) for name, value in artifact_sha256.items()
    }
    normalized_hard_states = {
        str(name): str(value) for name, value in hard_state_hashes.items()
    }
    normalized_two_n = {str(name): str(value) for name, value in two_n_result.items()}
    stages = _validate_replay_stage_results(record.get("stage_results"))
    merge_status = str(record.get("merge_status", ""))
    merge_error_fingerprint = str(record.get("merge_error_fingerprint", ""))
    if merge_status not in {"complete", "error", "not_applicable"}:
        raise ValueError("replay merge status is invalid")
    if merge_status == "error" and not _sha256_shaped(merge_error_fingerprint):
        raise ValueError("failed replay merge requires an error fingerprint")
    if merge_status == "not_applicable" and (
        merge_error_fingerprint or "merged_input" in artifact_names
    ):
        raise ValueError("pair-only replay carries merge evidence")
    _validate_replay_evidence_bindings(
        stage_results=stages,
        artifact_sha256=normalized_artifact_sha256,
        hard_state_hashes=normalized_hard_states,
        two_n_result=normalized_two_n,
        merge_status=merge_status,
    )


def _stable_replay(
    worker: list[dict[str, object]],
    external: list[dict[str, object]],
    two_n: list[dict[str, object]],
    case: Mapping[str, object],
) -> tuple[bool, str, dict[str, str]]:
    family_records = {
        "worker": worker,
        "external_opt": external,
        "two_n": two_n,
    }
    statuses = {
        name: _replay_family_status(
            records,
            case,
            require_two_n=(name == "two_n"),
            family=name,
        )
        for name, records in family_records.items()
    }
    stable = all(value == "stable" for value in statuses.values())
    return stable, ("stable" if stable else _aggregate_replay_status(statuses.values())), statuses


def _replay_family_status(
    records: Sequence[Mapping[str, object]],
    case: Mapping[str, object],
    *,
    require_two_n: bool,
    family: str = "worker",
) -> str:
    if len(records) != 2:
        return "unavailable"
    transport = [str(record.get("status", "")) for record in records]
    if any(status != "success" for status in transport):
        return "timeout" if "timeout" in transport else "failed"
    if _replay_signature(records[0]) != _replay_signature(records[1]):
        return "nondeterministic"
    expected_stages = case.get("expected_pair_stages")
    actual_stages = records[0].get("stage_results")
    if not isinstance(expected_stages, Mapping) or not isinstance(actual_stages, Mapping):
        return "mismatch"
    for name, expected in expected_stages.items():
        actual = actual_stages.get(name)
        if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
            return "mismatch"
        if (
            str(actual.get("execution_status", ""))
            != str(expected.get("execution_status", ""))
            or str(actual.get("verifier_status", ""))
            != str(expected.get("verifier_status", ""))
        ):
            return "mismatch"
        if str(expected.get("execution_status", "")) == "success" and str(
            actual.get("hard_state_id", "")
        ) != str(expected.get("hard_state_id", "")):
            return "mismatch"
    if require_two_n:
        expected_two_n = case.get("expected_two_n")
        actual_two_n = records[0].get("two_n_result")
        if not isinstance(expected_two_n, Mapping) or not isinstance(actual_two_n, Mapping):
            return "mismatch"
        expected_semantics = _semantic_two_n_result(expected_two_n)
        actual_semantics = _semantic_two_n_result(actual_two_n)
        if any(
            actual_semantics.get(name, "") != value
            for name, value in expected_semantics.items()
        ) or not _sha256_shaped(actual_two_n.get("second_output_hard_state_id", "")):
            return "mismatch"
    return "stable"


def _aggregate_replay_status(statuses: Sequence[str] | Any) -> str:
    values = {str(value) for value in statuses}
    for status in (
        "nondeterministic",
        "mismatch",
        "timeout",
        "failed",
        "unavailable",
        "unknown",
    ):
        if status in values:
            return status
    return "unknown"


def _semantic_replay_stages(value: object) -> dict[str, dict[str, str]]:
    """Project complete replay evidence onto repeat-stability semantics."""

    semantic_stages: dict[str, dict[str, str]] = {}
    if not isinstance(value, Mapping):
        return semantic_stages
    for name in _REPLAY_STAGE_NAMES:
        stage = value.get(name)
        if not isinstance(stage, Mapping):
            continue
        execution_status = str(stage.get("execution_status", ""))
        semantics = {
            "execution_status": execution_status,
            "verifier_status": str(stage.get("verifier_status", "")),
        }
        if execution_status == "success":
            semantics["hard_state_id"] = str(stage.get("hard_state_id", ""))
        semantic_stages[name] = semantics
    return semantic_stages


def _semantic_two_n_result(value: object) -> dict[str, str]:
    """Keep 2N effects/statuses/hard states, never artifact byte identity."""

    if not isinstance(value, Mapping):
        return {}
    semantic_fields = (
        "two_n_pair_status",
        "action_a_directional_status",
        "action_b_directional_status",
        "directional_status",
        "first_round_effect_sha256",
        "second_round_effect_sha256",
        "merged_input_hard_state_id",
        "second_output_hard_state_id",
    )
    return {
        field: str(value[field])
        for field in semantic_fields
        if field in value
    }


def _replay_signature(record: Mapping[str, object]) -> str:
    stages = record.get("stage_results")
    return _canonical_json(
        {
            "status": record["status"],
            "hard_state_hashes": record["hard_state_hashes"],
            "two_n_result": _semantic_two_n_result(
                record.get("two_n_result", {})
            ),
            "stage_results": _semantic_replay_stages(stages),
            "merge_status": record.get("merge_status", ""),
        }
    )


def _normalize_two_n_result(value: object) -> dict[str, list[dict[str, object]]]:
    if is_dataclass(value):
        # ``Advisor2NGroupResult`` deliberately exposes read-only
        # MappingProxyType rows.  dataclasses.asdict deep-copies those rows
        # and fails before evidence can be persisted, so read each field
        # explicitly and normalize mappings/sequences without copying opaque
        # implementation objects.
        value = {field.name: getattr(value, field.name) for field in fields(value)}
    value = _json_value(value)
    if not isinstance(value, Mapping):
        raise ValueError("group 2N runner must return a mapping or dataclass")
    if "group_row" in value:
        group_rows = [value["group_row"]]
    else:
        group_rows = value.get("group_rows", [])
    return {
        "group_rows": _canonical_rows(_as_mapping_sequence(group_rows, "group_rows")),
        "directional_rows": _canonical_rows(_as_mapping_sequence(value.get("directional_rows", []), "directional_rows")),
        "pair_rows": _canonical_rows(_as_mapping_sequence(value.get("pair_rows", []), "pair_rows")),
    }


def _validate_two_n_result(
    value: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    study_manifest_id: str,
    program_id: str,
    group_id: str,
    actions: Mapping[str, object],
    pair_view: Sequence[Mapping[str, object]],
) -> None:
    group_rows = value["group_rows"]
    if len(group_rows) != 1:
        raise ValueError("each group 2N run requires exactly one group result")
    for row in group_rows:
        _require_two_n_binding(row, study_manifest_id, program_id, group_id)
    directional_ids: list[str] = []
    for row in value["directional_rows"]:
        _require_two_n_binding(row, study_manifest_id, program_id, group_id)
        action_id = str(row.get("action_id", ""))
        if action_id not in actions:
            raise ValueError("2N directional row action is outside its exact group")
        directional_ids.append(action_id)
    if len(directional_ids) != len(set(directional_ids)) or set(directional_ids) != set(actions):
        raise ValueError("2N directional rows must cover each exact group action once")
    expected_pairs = {str(row.get("row_id", "")): row for row in pair_view}
    observed_pair_ids: list[str] = []
    for row in value["pair_rows"]:
        _require_two_n_binding(row, study_manifest_id, program_id, group_id)
        left, right = str(row.get("action_a_id", "")), str(row.get("action_b_id", ""))
        if left not in actions or right not in actions or left >= right:
            raise ValueError("2N pair row endpoints are outside its exact group")
        observation_id = str(row.get("pair_observation_row_id", ""))
        observed = expected_pairs.get(observation_id)
        if observed is None:
            raise ValueError("2N pair row does not join an exact group pair observation")
        if (str(observed.get("action_a_id")), str(observed.get("action_b_id"))) != (left, right):
            raise ValueError("2N pair row endpoints disagree with its pair observation")
        if "dynamic_result" in row and str(row["dynamic_result"]) != str(observed.get("dynamic_result", "")):
            raise ValueError("2N pair row dynamic truth disagrees with pair observation")
        observed_pair_ids.append(observation_id)
    if len(observed_pair_ids) != len(set(observed_pair_ids)) or set(observed_pair_ids) != set(expected_pairs):
        raise ValueError("2N pair rows must cover each exact group pair observation once")


def _require_two_n_binding(
    row: Mapping[str, object], study_manifest_id: str, program_id: str, group_id: str
) -> None:
    for field, value in (
        ("study_manifest_id", study_manifest_id),
        ("program_id", program_id),
        ("group_id", group_id),
    ):
        if str(row.get(field, "")) != value:
            raise ValueError(f"2N row does not exactly bind {field}")


def _freeze_two_n(value: Mapping[str, list[dict[str, object]]]) -> Mapping[str, object]:
    return {name: tuple(_canonical_rows(rows)) for name, rows in value.items()}


def _validate_groups(groups: Mapping[str, Sequence[object]]) -> None:
    if set(groups) != set(_GROUP_IDS):
        raise ValueError("groups must be exactly U14, U30, and Uall")
    uall = _action_map(groups["Uall"])
    if len(uall) < 2:
        raise ValueError("Uall requires at least two actions")
    for group in ("U14", "U30"):
        action_ids = set(_action_map(groups[group]))
        if not action_ids.issubset(uall):
            raise ValueError(f"{group} actions must reuse exact Uall action IDs")
    u14 = set(_action_map(groups["U14"]))
    u30 = set(_action_map(groups["U30"]))
    if not u14.issubset(u30):
        raise ValueError("U14 actions must be an exact subset of U30 action IDs")


def _action_map(actions: Sequence[object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for action in actions:
        action_id = str(action.get("action_id", "") if isinstance(action, Mapping) else getattr(action, "action_id", ""))
        if not action_id or action_id in result:
            raise ValueError("configured actions require unique non-empty action_id values")
        result[action_id] = action
    return dict(sorted(result.items()))


def _require_exact_action_rows(rows: Sequence[Mapping[str, object]], actions: Mapping[str, object], label: str) -> None:
    found = [str(row.get("action_id", "")) for row in rows]
    if len(found) != len(set(found)) or set(found) != set(actions):
        raise ValueError(f"{label} rows must bind exactly the configured Uall action IDs")


def _require_pair_rows(rows: Sequence[Mapping[str, object]], actions: Mapping[str, object]) -> None:
    seen: set[tuple[str, str]] = set()
    for row in rows:
        left, right = str(row.get("action_a_id", "")), str(row.get("action_b_id", ""))
        if left not in actions or right not in actions or left >= right or (left, right) in seen:
            raise ValueError("Uall pair rows must be unique canonical pairs of Uall action IDs")
        seen.add((left, right))
    expected = len(actions) * (len(actions) - 1) // 2
    if len(seen) != expected:
        raise ValueError("Uall pair runner did not return the complete unordered pair matrix")


def _find_original_pair(rows: Sequence[Mapping[str, object]], row_id: str) -> dict[str, object]:
    for row in rows:
        if str(row.get("row_id", "")) == row_id:
            return dict(row)
    raise ValueError(f"false authorization has no exact AB/BA pair observation: {row_id}")


def _validate_original_pair(
    row: Mapping[str, object],
    *,
    study_manifest_id: str,
    program_id: str,
    action_a_id: str,
    action_b_id: str,
    profiles: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, str]]:
    expected = {
        "study_manifest_id": study_manifest_id,
        "program_id": program_id,
        "action_a_id": action_a_id,
        "action_b_id": action_b_id,
    }
    for field, value in expected.items():
        if str(row.get(field, "")) != value:
            raise ValueError(f"AB/BA pair observation does not exactly bind {field}")
    dynamic = str(row.get("dynamic_result", ""))
    stages: dict[str, dict[str, str]] = {}
    profile_by_action: dict[str, Mapping[str, object]] = {}
    for profile in profiles:
        action_id = str(profile.get("action_id", ""))
        if not action_id or action_id in profile_by_action:
            raise ValueError("first-round profile identity is empty or ambiguous")
        profile_by_action[action_id] = profile
    for name, prefix, action_id in (
        ("A", "a", action_a_id),
        ("B", "b", action_b_id),
    ):
        profile = profile_by_action.get(action_id)
        if profile is None:
            raise ValueError(
                f"pair first-round {name} has no exact profile identity"
            )
        status = str(profile.get("execution_status", ""))
        verifier = str(profile.get("verifier_status", ""))
        hard_state_id = str(profile.get("output_hard_state_id", ""))
        output_sha256 = str(profile.get("output_sha256", ""))
        if (
            status != "success"
            or verifier != "success"
            or not _sha256_shaped(hard_state_id)
            or not _sha256_shaped(output_sha256)
            or str(row.get(f"{prefix}_status", "")) != status
            or str(row.get(f"{prefix}_verifier_status", "")) != verifier
            or str(row.get(f"{prefix}_hard_state_id", "")) != hard_state_id
            or str(row.get(f"{prefix}_output_sha256", "")) != output_sha256
        ):
            raise ValueError(
                f"pair first-round {name} evidence is not cross-bound to its profile"
            )
        stages[name] = {
            "execution_status": status,
            "verifier_status": verifier,
            "hard_state_id": hard_state_id,
            "output_sha256": output_sha256,
            "source_action_id": action_id,
            "source_stderr_sha256": "",
            "error_fingerprint": "",
        }
    statuses: list[str] = []
    for direction in ("ab", "ba"):
        name = direction.upper()
        status = str(row.get(f"{direction}_status", ""))
        verifier = str(row.get(f"{direction}_verifier_status", ""))
        digest = str(row.get(f"{direction}_hard_state_id", ""))
        output_sha256 = str(row.get(f"{direction}_output_sha256", ""))
        if status == "success":
            if (
                verifier != "success"
                or not _sha256_shaped(digest)
                or not _sha256_shaped(output_sha256)
            ):
                raise ValueError(
                    f"AB/BA pair observation {name} has invalid successful evidence"
                )
        elif status not in {"error", "invalid"}:
            raise ValueError(
                f"AB/BA pair observation {name} is not replayable terminal evidence"
            )
        elif verifier not in {"invalid", "not_run"}:
            raise ValueError(
                f"AB/BA pair observation {name} lacks exact terminal verifier evidence"
            )
        source_stderr_sha256 = str(
            row.get(f"{direction}_stderr_sha256", "")
        )
        if status != "success" and not _sha256_shaped(source_stderr_sha256):
            raise ValueError(
                f"AB/BA pair observation {name} lacks stage-local stderr evidence"
            )
        stages[name] = {
            "execution_status": status,
            "verifier_status": verifier,
            "hard_state_id": digest if status == "success" else "",
            "output_sha256": output_sha256 if status == "success" else "",
            "source_pair_row_id": str(row.get("row_id", "")),
            "source_stderr_sha256": source_stderr_sha256 if status != "success" else "",
            "error_fingerprint": (
                _terminal_error_fingerprint(status, verifier, source_stderr_sha256)
                if status != "success"
                else ""
            ),
        }
        statuses.append(status)
    if dynamic == "order_sensitive" and statuses != ["success", "success"]:
        raise ValueError("order-sensitive AB/BA evidence requires two successful orders")
    if dynamic == "failed" and not (
        statuses.count("success")
        + sum(status in {"error", "invalid"} for status in statuses)
        == 2
        and sum(status in {"error", "invalid"} for status in statuses) >= 1
        and _sha256_shaped(row.get("command_sha256", ""))
        and _sha256_shaped(row.get("stderr_sha256", ""))
    ):
        raise ValueError(
            "failed AB/BA evidence requires one or two replayable terminal orders"
        )
    if dynamic not in {"order_sensitive", "failed"}:
        raise ValueError("false authorization requires order-sensitive or failed AB/BA truth")
    return stages


def _expected_two_n_values(
    row: Mapping[str, object], directional: Mapping[str, object]
) -> dict[str, str]:
    fields = ("two_n_pair_status", "action_a_directional_status", "action_b_directional_status")
    values = {field: str(row.get(field, "")) for field in fields}
    if any(not value for value in values.values()):
        raise ValueError("false authorization requires complete 2N pair evidence")
    directional_values = {
        field: str(directional.get(field, ""))
        for field in (
            "directional_status",
            "first_round_effect_sha256",
            "second_round_effect_sha256",
            "merged_input_sha256",
            "merged_input_hard_state_id",
            "second_output_sha256",
        )
    }
    if directional_values["directional_status"] != "authorized_all_others" or any(
        not _sha256_shaped(directional_values[field])
        for field in (
            "first_round_effect_sha256",
            "second_round_effect_sha256",
            "merged_input_sha256",
            "merged_input_hard_state_id",
            "second_output_sha256",
        )
    ):
        raise ValueError("false authorization requires complete authorized 2N effect evidence")
    values.update(directional_values)
    return values


def _writeback_replay_results(
    row: dict[str, object], witnesses: Sequence[Mapping[str, object]]
) -> None:
    expected_directions = set(_authorized_direction_ids(row))
    by_direction: dict[str, Mapping[str, object]] = {}
    for witness in witnesses:
        case = witness.get("case")
        action_id = str(case.get("authorized_action_id", "")) if isinstance(case, Mapping) else ""
        if action_id not in expected_directions or action_id in by_direction:
            raise ValueError("false-authorization replay cases do not exactly cover directions")
        by_direction[action_id] = witness
    if set(by_direction) != expected_directions:
        raise ValueError("false-authorization replay cases do not exactly cover directions")
    family_fields = {
        "worker": "worker_replay_status",
        "external_opt": "external_opt_replay_status",
        "two_n": "two_n_replay_status",
    }
    for family, field in family_fields.items():
        statuses = []
        for witness in by_direction.values():
            family_statuses = witness.get("family_statuses")
            statuses.append(
                str(family_statuses.get(family, "unknown"))
                if isinstance(family_statuses, Mapping)
                else "unknown"
            )
        row[field] = (
            "stable"
            if statuses and all(status == "stable" for status in statuses)
            else _aggregate_replay_status(statuses)
        )
    stable = all(
        str(row[field]) == "stable" for field in family_fields.values()
    ) and all(
        str(witness.get("stable_false_authorization", "")).lower() == "true"
        for witness in by_direction.values()
    )
    row["stable_false_authorization"] = "true" if stable else "false"
    case_ids = sorted(str(witness.get("case_id", "")) for witness in by_direction.values())
    if any(not case_id for case_id in case_ids):
        raise ValueError("false-authorization replay case_id is empty")
    row["replay_artifact_id"] = canonical_row_id(
        "false-authorization-replay-set", str(row.get("row_id", "")), *case_ids
    )
    row["replay_time_ms"] = sum(
        max(0, int(witness.get("replay_time_ms", 0)))
        for witness in by_direction.values()
    )
    try:
        source_ids = json.loads(str(row.get("source_row_ids", "[]")))
    except json.JSONDecodeError:
        source_ids = []
    if not isinstance(source_ids, list):
        source_ids = []
    row["source_row_ids"] = _canonical_json(
        sorted({str(value) for value in (*source_ids, *case_ids) if str(value)})
    )
    row["fail_closed_reason"] = (
        ""
        if stable
        else "false_authorization_replay_not_stable:"
        + _aggregate_replay_status(
            str(row[field]) for field in family_fields.values()
        )
    )


def _authorized_direction_ids(row: Mapping[str, object]) -> tuple[str, ...]:
    """Return one replay case per Pi direction actually authorized by 2N."""

    result: list[str] = []
    for endpoint, status_field in (
        ("action_a_id", "action_a_directional_status"),
        ("action_b_id", "action_b_directional_status"),
    ):
        if str(row.get(status_field, "")) == "authorized_all_others":
            action_id = str(row.get(endpoint, ""))
            if not action_id:
                raise ValueError("false authorization has an empty authorized endpoint")
            result.append(action_id)
    if not result:
        raise ValueError("false authorization has no authorized Pi direction")
    return tuple(result)


def _validate_first_round_artifacts(rows: Sequence[Mapping[str, object]], directory: Path) -> None:
    root = directory.resolve(strict=False)
    for row in rows:
        if str(row.get("execution_status", "")) != "success":
            continue
        artifact = Path(str(row.get("output_path", ""))).resolve(strict=False)
        try:
            artifact.relative_to(root)
        except ValueError as error:
            raise ValueError(f"first-round artifact escapes isolated profile stage: {artifact}") from error
        expected = str(row.get("output_sha256", ""))
        if not artifact.is_file() or len(expected) != 64 or _sha256_file(artifact) != expected:
            raise ValueError("first-round artifact is missing or sha256-mismatched")


def _materialized_merged_inputs(directory: Path) -> dict[str, str]:
    root = directory.resolve(strict=False)
    inputs: dict[str, str] = {}
    for path in sorted(root.rglob("merged_input.ll")):
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as error:  # pragma: no cover - rglob is rooted, retained fail-closed
            raise ValueError(f"merged input escapes 2N stage: {path}") from error
        inputs[relative] = _sha256_file(path)
    return inputs


def _validate_two_n_materialization(rows: Sequence[Mapping[str, object]], directory: Path) -> None:
    root = directory.resolve(strict=False)
    for row in rows:
        action_id = str(row.get("action_id", ""))
        if str(row.get("merged_input_status", "")) != "complete":
            continue
        merged = root / action_id / "merged_input.ll"
        expected = str(row.get("merged_input_sha256", ""))
        if not merged.is_file() or len(expected) != 64 or _sha256_file(merged) != expected:
            raise ValueError("2N merged input is missing or sha256-mismatched")
        if str(row.get("second_round_status", "")) == "success":
            second = root / action_id / "second_round.ll"
            if not second.is_file():
                raise ValueError("2N second-round output is not materialized")


def _only_program_id(programs: Sequence[tuple[str, Path]]) -> str:
    if len(programs) != 1:
        raise ValueError("false authorization row requires program_id for multi-program study")
    return programs[0][0]


def _stage_record(
    *,
    stage: str,
    study_manifest_id: str,
    program_id: str,
    group_id: str,
    payload: Mapping[str, object],
    action_id: str = "",
    pair_id: str = "",
) -> dict[str, object]:
    command = payload.get("command", [])
    command_values = [str(part) for part in command] if isinstance(command, (list, tuple)) else []
    stderr = str(payload.get("stderr", ""))
    artifact_path = Path(str(payload.get("output_path", ""))).resolve(strict=False)
    artifact_sha256 = _sha256_file(artifact_path) if artifact_path.is_file() else ""
    return {
        "stage": stage,
        "study_manifest_id": study_manifest_id,
        "program_id": program_id,
        "group_id": group_id,
        "action_id": action_id,
        "pair_id": pair_id,
        "status": str(payload.get("execution_status", payload.get("dynamic_result", "complete"))),
        "command": command_values,
        "command_sha256": _digest_text("\0".join(command_values)),
        "stderr": stderr,
        "stderr_sha256": _digest_text(stderr),
        "artifact_available": str(payload.get("artifact_available", "unknown")),
        "artifact_materialized": str(payload.get("artifact_materialized", "unknown")),
        "artifact_path": str(artifact_path) if artifact_path.is_file() else "",
        "artifact_sha256": artifact_sha256,
        "payload": dict(payload),
    }


def _canonical_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [dict(row) for row in sorted(rows, key=lambda row: _canonical_json(dict(row)))]


def _as_mapping_sequence(value: object, name: str) -> list[Mapping[str, object]]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(row, Mapping) for row in value):
        raise ValueError(f"{name} must be a sequence of mappings")
    return list(value)


def _require_mapping_list(value: Mapping[str, object], name: str) -> list[Mapping[str, object]]:
    return _as_mapping_sequence(value.get(name), name)


def _require_isolated_output(out_dir: Path, isolation_root: Path) -> Path:
    root = isolation_root.resolve(strict=False)
    target = out_dir.resolve(strict=False)
    allowed = {root / "output" / "smoke", root / "output" / "formal"}
    if target not in allowed:
        raise ValueError("orchestration output must be exactly isolated output/smoke or output/formal")
    target.mkdir(parents=True, exist_ok=True)
    return target


@contextmanager
def _exclusive_output_lock(out_dir: Path) -> Iterator[None]:
    """Hold a crash-released, cross-process writer lock for one output root.

    The lock file itself is intentionally durable.  Windows/POSIX kernel byte
    locks belong to the open handle, so an exception, process termination, or
    crash releases ownership without a PID-age heuristic or stale-file delete.
    """

    root = Path(out_dir).resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".orchestration-writer.lock"
    handle = lock_path.open("a+b")
    acquired = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - production study runs on Windows.
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, IOError) as error:
            raise RuntimeError(
                f"orchestration output already has an active writer: {root}"
            ) from error
        acquired = True
        yield
    finally:
        if acquired:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:  # pragma: no cover - production study runs on Windows.
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except (OSError, IOError):
                # Closing the handle below is the final kernel-backed release
                # even if an explicit unlock reports an abnormal condition.
                pass
        handle.close()


def _require_stage_path(path: Path, isolation_root: Path) -> Path:
    root = isolation_root.resolve(strict=False)
    target = path.resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"stage path escapes isolated experiment root: {target}") from error
    return target


def _safe_rmtree(path: Path, isolation_root: Path) -> None:
    target = _require_stage_path(path, isolation_root)
    if target == isolation_root.resolve(strict=False) or target.name in {"output", "raw", "replay"}:
        raise ValueError(f"refusing broad stage deletion: {target}")
    shutil.rmtree(target)


def _safe_name(value: object, label: str) -> str:
    text = str(value).strip()
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"unsafe {label}: {value!r}")
    return text


def _safe_digest(value: object, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} must be non-empty")
    return _digest_text(text)


def _sha256_shaped(value: object) -> bool:
    digest = str(value).lower()
    return len(digest) == 64 and all(
        character in "0123456789abcdef" for character in digest
    )


def _ordered_pair_digest(left: str, right: str) -> str:
    return _digest_text(left + "\0" + right)


def _path_component(digest: str) -> str:
    """A deterministic digest component short enough for Windows materialization."""

    return digest[:16]


def _relative(root: Path, path: Path) -> str:
    return path.resolve(strict=False).relative_to(root).as_posix()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _terminal_error_fingerprint(
    execution_status: str, verifier_status: str, stderr_sha256: str
) -> str:
    """Canonical identity for one typed terminal pass outcome."""

    return _digest_text(
        _canonical_json(
            {
                "execution_status": execution_status,
                "verifier_status": verifier_status,
                "stderr_sha256": stderr_sha256,
            }
        )
    )


def _execution_input_digest(value: Mapping[str, object]) -> str:
    """Bind every raw stage/checkpoint input to executable semantics."""

    payload = dict(value)
    payload["raw_execution_semantics_revision"] = RAW_EXECUTION_SEMANTICS_REVISION
    return _digest_text(_canonical_json(payload))


def _canonical_json(value: object) -> str:
    return json.dumps(_json_value(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _json_value(value: object) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_value({field.name: getattr(value, field.name) for field in fields(value)})
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"stage result is not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(value) + "\n", encoding="utf-8", newline="\n")
