"""Per-program limitation evidence, checkpoints, and recoverable compaction.

This module is intentionally independent of the CLI adapter.  It gives the
formal runner one narrow integration point after each one-program orchestration
result: publish a hash-valid planned checkpoint, execute the existing
hash-bound cleanup policy, then atomically select a complete checkpoint before
the next program starts.  A complete checkpoint is authoritative only for
experiment resume; it never grants optimizer or Phasebatch authority.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence
import uuid

from .cleanup import (
    cleanup_journals_resolved,
    compact_intermediate_artifacts,
    plan_intermediate_artifact_cleanup,
)
from .direct_merge import derive_two_n_pair_fields
from .orchestration import (
    OrchestrationDependencies,
    OrchestrationResult,
    _load_any_complete_stage,
    _replay_signature,
    _stable_replay,
    _validate_replay_record_evidence_claims,
    _writeback_replay_results,
    run_study_orchestration,
)
from .schema import canonical_row_id


PROGRAM_CHECKPOINT_SCHEMA_VERSION = "advisor-pair-scale-2n/program-checkpoint-v1"
PROGRAM_CHECKPOINT_POINTER_SCHEMA_VERSION = (
    "advisor-pair-scale-2n/program-checkpoint-pointer-v1"
)
UNPUBLISHED_STAGING_CLEANUP_SCHEMA_VERSION = (
    "advisor-pair-scale-2n/unpublished-staging-cleanup-v1"
)
_GROUP_IDS = ("U14", "U30", "Uall")
_RAW_TABLES = (
    "single_pass_observations.csv",
    "pair_observations.csv",
    "advisor_2n_group_results.csv",
    "advisor_2n_directional_results.csv",
    "advisor_2n_pair_validation.csv",
)
_SHA256_CHARS = frozenset("0123456789abcdef")
PROGRAM_RUNTIME_IMPLEMENTATION_VERSION = 3
_PAIR_CLEANUP_SYNC_FIELDS = (
    "ab_output_path",
    "ab_output_sha256",
    "ba_output_path",
    "ba_output_sha256",
    "artifact_available",
    "artifact_materialized",
    "cleanup_status",
    "artifact_id",
)
_PAIR_STAGE_CLEANUP_PROJECTION_FIELDS = frozenset(
    {
        "ab_output_path",
        "ba_output_path",
        "artifact_available",
        "artifact_materialized",
        "cleanup_status",
        "reused_uall_pair_artifact",
    }
)
_DIRECTIONAL_CLEANUP_PROJECTION_FIELDS = frozenset(
    {"second_output_path", "second_output_materialized", "cleanup_status"}
)
_TWO_N_STAGE_BINDING_SCHEMA_VERSION = (
    "advisor-pair-scale-2n/two-n-stage-binding-v1"
)
_PAIR_STAGE_BINDING_SCHEMA_VERSION = (
    "advisor-pair-scale-2n/pair-stage-binding-v1"
)


def _json_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"program checkpoint value is not JSON serializable: {type(value).__name__}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        _json_value(value), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _valid_sha256(value: object) -> bool:
    text = str(value).lower()
    return len(text) == 64 and all(character in _SHA256_CHARS for character in text)


def _require_inside(path: Path, isolation_root: Path, *, label: str) -> Path:
    target = Path(path).resolve(strict=False)
    root = Path(isolation_root).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the isolated experiment output: {target}") from error
    if target == root:
        raise ValueError(f"{label} cannot be the isolated experiment output root")
    return target


def _phasebatch_hard_state_id(path: Path) -> str:
    from phasebatch.ir_equivalence import DEFAULT_HARD_STATE_POLICY, hard_state_hash

    digest = hard_state_hash(Path(path), DEFAULT_HARD_STATE_POLICY)
    if not _valid_sha256(digest):
        raise ValueError("materialized artifact hard-state hash is malformed")
    return str(digest).lower()


def _text_sha256(value: object) -> str:
    return _sha256_bytes(str(value).encode("utf-8"))


def _action_map(actions: Sequence[object]) -> dict[str, object]:
    output: dict[str, object] = {}
    for action in actions:
        action_id = str(
            action.get("action_id", "")
            if isinstance(action, Mapping)
            else getattr(action, "action_id", "")
        )
        if not action_id or action_id in output:
            raise ValueError("runtime limitation groups require unique non-empty action IDs")
        output[action_id] = action
    return dict(sorted(output.items()))


def _pairs(action_ids: Sequence[str]) -> list[tuple[str, str]]:
    ordered = sorted(str(value) for value in action_ids)
    return [
        (ordered[left], ordered[right])
        for left in range(len(ordered))
        for right in range(left + 1, len(ordered))
    ]


def program_checkpoint_input_sha256(
    *,
    study_manifest_id: str,
    program_id: str,
    root_ir_sha256: str,
    group_action_ids: Mapping[str, Sequence[str]],
    runner_semantics_id: str,
) -> str:
    """Bind resume to both frozen inputs and the implemented oracle semantics.

    ``runner_semantics_id`` is intentionally required.  A correction to Worker
    canonical hashing, AB/BA comparison, or another result-affecting adapter
    must change it so an older per-program checkpoint cannot mask the fix.
    """

    manifest = str(study_manifest_id).strip()
    program = str(program_id).strip()
    semantics = str(runner_semantics_id).strip()
    if not manifest or not program or not semantics:
        raise ValueError("checkpoint identity requires manifest, program, and runner semantics")
    if not _valid_sha256(root_ir_sha256):
        raise ValueError("checkpoint root_ir_sha256 is invalid")
    if set(group_action_ids) != set(_GROUP_IDS):
        raise ValueError("checkpoint groups must be exactly U14, U30, and Uall")
    normalized: dict[str, list[str]] = {}
    for group in _GROUP_IDS:
        values = [str(value) for value in group_action_ids[group]]
        if any(not value for value in values) or len(values) != len(set(values)):
            raise ValueError("checkpoint group action IDs must be unique and non-empty")
        normalized[group] = sorted(values)
    if not set(normalized["U14"]).issubset(normalized["U30"]) or not set(
        normalized["U30"]
    ).issubset(normalized["Uall"]):
        raise ValueError("checkpoint groups require U14 subset U30 subset Uall")
    return _sha256_bytes(
        _canonical_json(
            {
                "program_runtime_implementation_version": PROGRAM_RUNTIME_IMPLEMENTATION_VERSION,
                "study_manifest_id": manifest,
                "program_id": program,
                "root_ir_sha256": str(root_ir_sha256).lower(),
                "group_action_ids": normalized,
                "runner_semantics_id": semantics,
            }
        ).encode("utf-8")
    )


def program_checkpoint_path(out_dir: Path, input_sha256: str) -> Path:
    if not _valid_sha256(input_sha256):
        raise ValueError("program checkpoint input identity must be sha256-shaped")
    return (
        Path(out_dir).resolve(strict=False)
        / "raw"
        / "program-checkpoints"
        / str(input_sha256).lower()[:16]
    )


def _limitation_reason(provenance: Mapping[str, object]) -> str:
    required = (
        "control_id",
        "control_file_sha256",
        "program_wall_time_budget_s",
        "observed_wall_time_s",
        "limitation_kind",
        "reason",
    )
    if any(str(provenance.get(field, "")).strip() == "" for field in required):
        raise ValueError("runtime limitation provenance is incomplete")
    if provenance.get("limitation_kind") != "runtime_budget_exceeded":
        raise ValueError("runtime limitation kind must be runtime_budget_exceeded")
    if not _valid_sha256(provenance["control_id"]) or not _valid_sha256(
        provenance["control_file_sha256"]
    ):
        raise ValueError("runtime limitation control hashes are invalid")
    return (
        "runtime_budget_exceeded"
        f";budget_s={provenance['program_wall_time_budget_s']}"
        f";observed_s={provenance['observed_wall_time_s']}"
        f";control_id={provenance['control_id']}"
        f";control_file_sha256={provenance['control_file_sha256']}"
        f";reason={provenance['reason']}"
    )


def runtime_budget_limited_result(
    *,
    out_dir: Path,
    study_manifest_id: str,
    program_id: str,
    program_family: str,
    groups: Mapping[str, Sequence[object]],
    provenance: Mapping[str, object],
) -> OrchestrationResult:
    """Expand one over-budget program into full, fail-closed evidence rows."""

    if set(groups) != set(_GROUP_IDS):
        raise ValueError("runtime limitation groups must be exactly U14, U30, and Uall")
    action_maps = {group: _action_map(groups[group]) for group in _GROUP_IDS}
    if not set(action_maps["U14"]).issubset(action_maps["U30"]) or not set(
        action_maps["U30"]
    ).issubset(action_maps["Uall"]):
        raise ValueError("runtime limitation groups require U14 subset U30 subset Uall")
    manifest = str(study_manifest_id).strip()
    program = str(program_id).strip()
    if not manifest or not program:
        raise ValueError("runtime limitation requires manifest and program IDs")
    reason = _limitation_reason(provenance)
    common = {
        "study_manifest_id": manifest,
        "program_id": program,
        "authority_granted": "false",
        "proved_commute": "false",
        "fail_closed_reason": reason,
    }

    profiles: list[dict[str, object]] = []
    for action_id in action_maps["Uall"]:
        profiles.append(
            {
                **common,
                "row_id": canonical_row_id(
                    "single-pass-runtime-budget", manifest, program, action_id
                ),
                "group_id": "Uall",
                "action_id": action_id,
                "execution_status": "timeout",
                "activity_status": "unknown",
                "changed_functions_json": "[]",
                "changed_blocks_json": "[]",
                "changed_module_regions_json": "[]",
                "verifier_status": "not_run",
                "logical_pass_applications": 0,
                "physical_pass_invocations": 0,
                "cache_reused": "false",
                "artifact_available": "false",
                "artifact_materialized": "false",
                "wall_time_ms": 0,
            }
        )

    pair_views: dict[str, tuple[dict[str, object], ...]] = {}
    for group_id in _GROUP_IDS:
        rows: list[dict[str, object]] = []
        for left, right in _pairs(tuple(action_maps[group_id])):
            rows.append(
                {
                    **common,
                    "row_id": canonical_row_id(
                        "pair-runtime-budget", manifest, program, left, right
                    ),
                    "group_id": group_id,
                    "action_a_id": left,
                    "action_b_id": right,
                    "root_activity_class": "unknown",
                    "observed_relation": "observed_unknown",
                    "program_family": str(program_family),
                    "a_status": "timeout",
                    "b_status": "timeout",
                    "ab_status": "not_run",
                    "ab_output_path": "",
                    "ab_output_sha256": "",
                    "ab_verifier_status": "not_run",
                    "ba_status": "not_run",
                    "ba_output_path": "",
                    "ba_output_sha256": "",
                    "ba_verifier_status": "not_run",
                    "dynamic_result": "timeout",
                    "second_stage_logical_pass_applications": 0,
                    "second_stage_physical_pass_invocations": 0,
                    "total_logical_pass_applications": 0,
                    "total_physical_pass_invocations": 0,
                    "cache_reused": "false",
                    "artifact_available": "false",
                    "artifact_materialized": "false",
                    "cleanup_status": "retained_runtime_budget_exceeded",
                    "artifact_id": "",
                    "wall_time_ms": 0,
                }
            )
        pair_views[group_id] = tuple(rows)

    two_n_results: dict[str, Mapping[str, object]] = {}
    for group_id in _GROUP_IDS:
        actions = action_maps[group_id]
        group_row = {
            **common,
            "row_id": canonical_row_id(
                "advisor-2n-group-runtime-budget", manifest, program, group_id
            ),
            "group_id": group_id,
            "configured_n": len(actions),
            "successful_n": 0,
            "active_n": 0,
            "no_op_n": 0,
            "failed_n": 0,
            "timeout_n": len(actions),
            "round1_status": "timeout",
            "first_round_disjoint_status": "unknown",
            "all_n_merge_status": "timeout",
            "all_n_second_round_status": "timeout",
            "group_authorization_status": "group_precondition_unavailable",
            "directional_authorized_count": 0,
            "directional_unavailable_count": len(actions),
            "logical_pass_applications": 0,
            "physical_pass_invocations": 0,
            "merge_helper_calls": 0,
            "wall_time_ms": 0,
        }
        directional_rows = [
            {
                **common,
                "row_id": canonical_row_id(
                    "advisor-2n-directional-runtime-budget",
                    manifest,
                    program,
                    group_id,
                    action_id,
                ),
                "group_id": group_id,
                "action_id": action_id,
                "directional_status": "timeout",
                "first_round_status": "timeout",
                "merged_input_status": "timeout",
                "merged_input_path": "",
                "merged_input_sha256": "",
                "second_round_status": "timeout",
                "second_output_path": "",
                "second_output_sha256": "",
                "second_output_materialized": "false",
                "other_contributions_preserved": "false",
                "verifier_status": "not_run",
                "logical_pass_applications": 0,
                "physical_pass_invocations": 0,
                "merge_helper_calls": 0,
                "cleanup_status": "retained_runtime_budget_exceeded",
                "wall_time_ms": 0,
            }
            for action_id in actions
        ]
        pair_rows: list[dict[str, object]] = []
        view_by_endpoints = {
            (str(row["action_a_id"]), str(row["action_b_id"])): row
            for row in pair_views[group_id]
        }
        for left, right in _pairs(tuple(actions)):
            observed = view_by_endpoints[(left, right)]
            pair_rows.append(
                {
                    **common,
                    "row_id": canonical_row_id(
                        "advisor-2n-pair-runtime-budget",
                        manifest,
                        program,
                        group_id,
                        left,
                        right,
                    ),
                    "group_id": group_id,
                    "action_a_id": left,
                    "action_b_id": right,
                    "action_a_directional_status": "timeout",
                    "action_b_directional_status": "timeout",
                    "two_n_pair_status": "group_precondition_unavailable",
                    "pair_observation_row_id": observed["row_id"],
                    "dynamic_result": "timeout",
                    "validation_status": "ground_truth_timeout",
                    "false_authorization": "false",
                    "stable_false_authorization": "false",
                    "worker_replay_status": "not_required",
                    "external_opt_replay_status": "not_required",
                    "two_n_replay_status": "not_required",
                    "replay_time_ms": 0,
                    "source_row_ids": observed["row_id"],
                }
            )
        two_n_results[group_id] = {
            "group_rows": (group_row,),
            "directional_rows": tuple(directional_rows),
            "pair_rows": tuple(pair_rows),
        }

    return OrchestrationResult(
        out_dir=Path(out_dir).resolve(strict=False),
        study_manifest_id=manifest,
        profile_rows={program: tuple(profiles)},
        pair_views=pair_views,
        two_n_results=two_n_results,
        false_authorizations=(),
        stage_paths={},
    )


def recover_runtime_budget_limited_program(
    *,
    out_dir: Path,
    isolation_root: Path,
    study_manifest_id: str,
    program_id: str,
    root_ir: Path,
    program_family: str,
    groups: Mapping[str, Sequence[object]],
    provenance: Mapping[str, object],
    replay_worker: object,
    replay_external_opt: object,
    replay_two_n: object,
) -> OrchestrationResult:
    """Reuse every complete stage and type only the missing suffix as timeout.

    A supervisor timeout can occur between atomic stage publications.  Running
    the ordinary content-addressed orchestration with fail-closed producers
    preserves valid profile/pair/view/2N evidence byte-for-byte and publishes
    denominator-preserving rows only for stages that never completed.
    """

    if not callable(replay_worker) or not callable(replay_external_opt) or not callable(
        replay_two_n
    ):
        raise ValueError("runtime recovery requires all three replay callbacks")
    limited = runtime_budget_limited_result(
        out_dir=out_dir,
        study_manifest_id=study_manifest_id,
        program_id=program_id,
        program_family=program_family,
        groups=groups,
        provenance=provenance,
    )
    profile_template = {
        str(row["action_id"]): dict(row)
        for row in limited.profile_rows[str(program_id)]
    }
    pair_template = {
        (str(row["action_a_id"]), str(row["action_b_id"])): dict(row)
        for row in limited.pair_views["Uall"]
    }
    two_n_template = {
        group: {
            key: [dict(row) for row in limited.two_n_results[group][key]]
            for key in ("group_rows", "directional_rows", "pair_rows")
        }
        for group in _GROUP_IDS
    }

    def missing_profiles(
        _root: Path, actions: tuple[object, ...], _directory: Path
    ) -> list[dict[str, object]]:
        return [
            dict(profile_template[str(getattr(action, "action_id", ""))])
            for action in actions
        ]

    def missing_pairs(
        _root: Path,
        profile_rows: list[dict[str, object]],
        actions: dict[str, object],
        _directory: Path,
    ) -> list[dict[str, object]]:
        statuses = {
            str(row.get("action_id", "")): str(row.get("execution_status", ""))
            for row in profile_rows
        }
        rows: list[dict[str, object]] = []
        for left, right in _pairs(tuple(actions)):
            row = dict(pair_template[(left, right)])
            row["a_status"] = statuses[left]
            row["b_status"] = statuses[right]
            if statuses[left] == statuses[right] == "success":
                row["ab_status"] = "timeout"
                row["ba_status"] = "timeout"
            else:
                row["ab_status"] = "not_run"
                row["ba_status"] = "not_run"
            if statuses[left] == statuses[right] == "success" or "timeout" in {
                statuses[left],
                statuses[right],
            }:
                row["dynamic_result"] = "timeout"
            else:
                # A known first-round endpoint failure makes the AB/BA
                # precondition fail before the missing pair stage.  Preserve
                # that empirical failure while retaining the runtime-budget
                # provenance in fail_closed_reason.
                row["dynamic_result"] = "failed"
                row["fail_closed_reason"] = (
                    f"{row['fail_closed_reason']}"
                    f";pair_precondition_failed:a_status={statuses[left]}"
                    f",b_status={statuses[right]}"
                )
            rows.append(row)
        return rows

    def missing_two_n(
        _root: Path,
        group_id: str,
        actions: dict[str, object],
        profile_rows: list[dict[str, object]],
        _directory: Path,
        pair_view: list[dict[str, object]],
    ) -> dict[str, list[dict[str, object]]]:
        profile_by_action = {
            str(row.get("action_id", "")): row for row in profile_rows
        }
        group_rows = [dict(row) for row in two_n_template[group_id]["group_rows"]]
        directional_rows = [
            dict(row) for row in two_n_template[group_id]["directional_rows"]
        ]
        pair_rows = [dict(row) for row in two_n_template[group_id]["pair_rows"]]
        statuses = {
            action_id: str(profile_by_action[action_id].get("execution_status", ""))
            for action_id in actions
        }
        activities = {
            action_id: str(profile_by_action[action_id].get("activity_status", ""))
            for action_id in actions
        }
        group_row = group_rows[0]
        successful = sum(status == "success" for status in statuses.values())
        timeouts = sum(status == "timeout" for status in statuses.values())
        failed = len(actions) - successful - timeouts
        if successful == len(actions):
            round1_gate = "complete"
            unavailable_direction = "timeout"
            merge_gate = "timeout"
            second_gate = "timeout"
        elif timeouts:
            round1_gate = "timeout"
            unavailable_direction = "timeout"
            merge_gate = "unknown"
            second_gate = "unknown"
        else:
            round1_gate = "round1_precondition_failed"
            unavailable_direction = "round1_precondition_failed"
            merge_gate = "unknown"
            second_gate = "unknown"
        group_row.update(
            {
                "successful_n": successful,
                "active_n": sum(
                    statuses[action] == "success" and activities[action] == "active"
                    for action in actions
                ),
                "no_op_n": sum(
                    statuses[action] == "success" and activities[action] == "no_op"
                    for action in actions
                ),
                "failed_n": failed,
                "timeout_n": timeouts,
                "round1_status": round1_gate,
                "all_n_merge_status": merge_gate,
                "all_n_second_round_status": second_gate,
            }
        )
        for row in directional_rows:
            action_id = str(row["action_id"])
            row["first_round_status"] = statuses[action_id]
            row["directional_status"] = unavailable_direction
            row["merged_input_status"] = (
                "timeout" if round1_gate == "complete" else "unknown"
            )
            row["second_round_status"] = (
                "timeout" if round1_gate == "complete" else "not_run"
            )
        view_by_endpoints = {
            (str(row["action_a_id"]), str(row["action_b_id"])): row
            for row in pair_view
        }
        for row in pair_rows:
            endpoint = (str(row["action_a_id"]), str(row["action_b_id"]))
            observed = view_by_endpoints[endpoint]
            dynamic = str(observed.get("dynamic_result", ""))
            row["pair_observation_row_id"] = str(observed.get("row_id", ""))
            row["dynamic_result"] = dynamic
            row["action_a_directional_status"] = unavailable_direction
            row["action_b_directional_status"] = unavailable_direction
            row["two_n_pair_status"] = "group_precondition_unavailable"
            row["validation_status"] = (
                "ground_truth_timeout"
                if dynamic == "timeout"
                else "ground_truth_failed"
                if dynamic == "failed"
                else "unavailable"
            )
            row["source_row_ids"] = str(observed.get("row_id", ""))
        return {
            "group_rows": group_rows,
            "directional_rows": directional_rows,
            "pair_rows": pair_rows,
        }

    return run_study_orchestration(
        out_dir=out_dir,
        isolation_root=isolation_root,
        study_manifest_id=study_manifest_id,
        programs={str(program_id): Path(root_ir)},
        groups=groups,
        dependencies=OrchestrationDependencies(
            profile_uall=missing_profiles,
            run_uall_pairs=missing_pairs,
            run_group_two_n=missing_two_n,
            replay_worker=replay_worker,  # type: ignore[arg-type]
            replay_external_opt=replay_external_opt,  # type: ignore[arg-type]
            replay_two_n=replay_two_n,  # type: ignore[arg-type]
        ),
    )


def _unpublished_cleanup_sha256(payload: Mapping[str, object]) -> str:
    body = {
        str(key): value
        for key, value in payload.items()
        if key != "cleanup_sha256"
    }
    return _sha256_bytes(_canonical_json(body).encode("utf-8"))


def _empty_unpublished_staging_cleanup() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": UNPUBLISHED_STAGING_CLEANUP_SCHEMA_VERSION,
        "cleanup_state": "planned",
        "entries": [],
        "summary": {"entry_count": 0, "file_count": 0, "size_bytes": 0},
        "authority_granted": False,
        "proved_commute": False,
    }
    payload["cleanup_sha256"] = _unpublished_cleanup_sha256(payload)
    return payload


def _relative_to_root(path: Path, root: Path, *, label: str) -> str:
    target = _require_inside(path, root, label=label)
    return target.relative_to(root).as_posix()


def plan_unpublished_staging_cleanup(
    *,
    isolation_root: Path,
    stage_paths: Mapping[str, object],
) -> dict[str, object]:
    """Inventory only staging siblings of exact, known orchestration stages."""

    root = Path(isolation_root).resolve(strict=False)
    if not root.is_dir():
        root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    seen: set[Path] = set()
    for stage_key, raw_value in sorted(stage_paths.items()):
        supplied = Path(str(raw_value))
        final_stage = supplied if supplied.is_absolute() else root / supplied
        final_stage = _require_inside(
            final_stage, root, label=f"{stage_key} final stage path"
        )
        parent = final_stage.parent
        if not parent.is_dir():
            continue
        prefix = f".{final_stage.name}.stage-"
        for candidate in sorted(parent.iterdir(), key=lambda path: path.name):
            if not candidate.name.startswith(prefix):
                continue
            staging = _require_inside(
                candidate, root, label=f"{stage_key} unpublished staging path"
            )
            if staging in seen:
                continue
            seen.add(staging)
            if staging.is_symlink() or not staging.is_dir():
                raise ValueError(
                    f"unpublished staging candidate is not a plain directory: {staging}"
                )
            directories: list[str] = []
            files: list[dict[str, object]] = []
            for child in sorted(staging.rglob("*"), key=lambda path: path.as_posix()):
                if child.is_symlink():
                    raise ValueError(f"unpublished staging contains a symlink: {child}")
                if child.is_dir():
                    directories.append(
                        _relative_to_root(
                            child, root, label="unpublished staging directory"
                        )
                    )
                elif child.is_file():
                    data = child.read_bytes()
                    files.append(
                        {
                            "relative_path": _relative_to_root(
                                child, root, label="unpublished staging file"
                            ),
                            "sha256": _sha256_bytes(data),
                            "size_bytes": len(data),
                        }
                    )
                else:
                    raise ValueError(
                        f"unpublished staging contains a non-file entry: {child}"
                    )
            relative = _relative_to_root(
                staging, root, label="unpublished staging directory"
            )
            reason = "interrupted_unpublished_stage"
            identity = {
                "stage_key": str(stage_key),
                "final_stage_relative_path": _relative_to_root(
                    final_stage, root, label="final orchestration stage"
                ),
                "relative_path": relative,
                "reason": reason,
                "directories": directories,
                "files": files,
            }
            entries.append(
                {
                    **identity,
                    "cleanup_id": _sha256_bytes(
                        _canonical_json(identity).encode("utf-8")
                    ),
                    "status": "planned",
                    "file_count": len(files),
                    "size_bytes": sum(int(item["size_bytes"]) for item in files),
                }
            )
    entries.sort(key=lambda entry: str(entry["relative_path"]))
    payload: dict[str, object] = {
        "schema_version": UNPUBLISHED_STAGING_CLEANUP_SCHEMA_VERSION,
        "cleanup_state": "planned",
        "entries": entries,
        "summary": {
            "entry_count": len(entries),
            "file_count": sum(int(entry["file_count"]) for entry in entries),
            "size_bytes": sum(int(entry["size_bytes"]) for entry in entries),
        },
        "authority_granted": False,
        "proved_commute": False,
    }
    payload["cleanup_sha256"] = _unpublished_cleanup_sha256(payload)
    return payload


def _validate_unpublished_staging_cleanup(
    value: object,
    *,
    required_state: str | None,
    stage_paths: object,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise _checkpoint_error("unpublished staging cleanup ledger is malformed")
    ledger = json.loads(_canonical_json(value))
    if (
        ledger.get("schema_version") != UNPUBLISHED_STAGING_CLEANUP_SCHEMA_VERSION
        or ledger.get("authority_granted") is not False
        or ledger.get("proved_commute") is not False
        or ledger.get("cleanup_state") not in {"planned", "complete"}
        or ledger.get("cleanup_sha256") != _unpublished_cleanup_sha256(ledger)
    ):
        raise _checkpoint_error("unpublished staging cleanup ledger hash/state mismatch")
    if required_state is not None and ledger.get("cleanup_state") != required_state:
        raise _checkpoint_error(
            f"{required_state} checkpoint has inconsistent unpublished staging cleanup"
        )
    entries = ledger.get("entries")
    summary = ledger.get("summary")
    if (
        not isinstance(entries, list)
        or any(not isinstance(entry, Mapping) for entry in entries)
        or not isinstance(summary, Mapping)
    ):
        raise _checkpoint_error("unpublished staging cleanup entries are malformed")
    if not isinstance(stage_paths, Mapping):
        raise _checkpoint_error("unpublished staging cleanup stage paths are malformed")
    canonical_stage_paths: dict[str, str] = {}
    for raw_key, raw_value in stage_paths.items():
        stage_key = str(raw_key)
        relative = str(raw_value)
        path = Path(relative)
        if (
            not stage_key
            or not relative
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != relative
            or stage_key in canonical_stage_paths
        ):
            raise _checkpoint_error(
                "unpublished staging cleanup stage path is not canonical"
            )
        canonical_stage_paths[stage_key] = relative
    seen_paths: set[str] = set()
    seen_ids: set[str] = set()
    for entry in entries:
        relative = str(entry.get("relative_path", ""))
        files = entry.get("files")
        directories = entry.get("directories")
        if (
            not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in seen_paths
            or not isinstance(files, list)
            or any(not isinstance(item, Mapping) for item in files)
            or not isinstance(directories, list)
            or any(not isinstance(item, str) or not item for item in directories)
            or str(entry.get("reason", "")) != "interrupted_unpublished_stage"
            or entry.get("status")
            != ("planned" if ledger["cleanup_state"] == "planned" else "reclaimed")
        ):
            raise _checkpoint_error("unpublished staging cleanup entry is invalid")
        seen_paths.add(relative)
        identity = {
            "stage_key": str(entry.get("stage_key", "")),
            "final_stage_relative_path": str(
                entry.get("final_stage_relative_path", "")
            ),
            "relative_path": relative,
            "reason": str(entry.get("reason", "")),
            "directories": list(directories),
            "files": list(files),
        }
        cleanup_id = str(entry.get("cleanup_id", ""))
        stage_key = str(identity["stage_key"])
        final_relative = str(identity["final_stage_relative_path"])
        final_path = Path(final_relative)
        staging_path = Path(relative)
        if (
            not stage_key
            or not final_relative
            or canonical_stage_paths.get(stage_key) != final_relative
            or final_path.is_absolute()
            or ".." in final_path.parts
            or final_path.as_posix() != final_relative
            or staging_path.parent != final_path.parent
            or not staging_path.name.startswith(f".{final_path.name}.stage-")
            or cleanup_id
            != _sha256_bytes(_canonical_json(identity).encode("utf-8"))
            or cleanup_id in seen_ids
        ):
            raise _checkpoint_error(
                "unpublished staging cleanup stage binding identity mismatch"
            )
        seen_ids.add(cleanup_id)
        file_paths: set[str] = set()
        total = 0
        for item in files:
            path = str(item.get("relative_path", ""))
            size = item.get("size_bytes")
            if (
                not path
                or Path(path).is_absolute()
                or ".." in Path(path).parts
                or path in file_paths
                or not _valid_sha256(item.get("sha256", ""))
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
            ):
                raise _checkpoint_error("unpublished staging file binding is invalid")
            try:
                Path(path).relative_to(staging_path)
            except ValueError as error:
                raise _checkpoint_error(
                    "unpublished staging file escapes its bound stage directory"
                ) from error
            file_paths.add(path)
            total += size
        for directory in directories:
            try:
                Path(str(directory)).relative_to(staging_path)
            except ValueError as error:
                raise _checkpoint_error(
                    "unpublished staging directory escapes its bound stage directory"
                ) from error
        if entry.get("file_count") != len(files) or entry.get("size_bytes") != total:
            raise _checkpoint_error("unpublished staging cleanup counters disagree")
    if (
        summary.get("entry_count") != len(entries)
        or summary.get("file_count")
        != sum(int(entry.get("file_count", 0)) for entry in entries)
        or summary.get("size_bytes")
        != sum(int(entry.get("size_bytes", 0)) for entry in entries)
    ):
        raise _checkpoint_error("unpublished staging cleanup summary disagrees")
    return ledger


def _complete_unpublished_staging_cleanup(
    payload: Mapping[str, object], *, isolation_root: Path
) -> dict[str, object]:
    root = Path(isolation_root).resolve(strict=False)
    ledger = _validate_unpublished_staging_cleanup(
        payload.get("unpublished_staging_cleanup"),
        required_state="planned",
        stage_paths=payload.get("stage_paths"),
    )
    entries = ledger["entries"]
    assert isinstance(entries, list)
    for raw_entry in entries:
        assert isinstance(raw_entry, dict)
        stage_paths = payload.get("stage_paths")
        assert isinstance(stage_paths, Mapping)
        final_stage = _require_inside(
            root / str(stage_paths[str(raw_entry["stage_key"])]),
            root,
            label="planned final orchestration stage",
        )
        staging = _require_inside(
            root / str(raw_entry["relative_path"]),
            root,
            label="planned unpublished staging directory",
        )
        if (
            staging.parent != final_stage.parent
            or not staging.name.startswith(f".{final_stage.name}.stage-")
        ):
            raise ValueError(
                "unpublished staging cleanup stage binding changed before deletion"
            )
        expected_files = {
            str(item["relative_path"]): item
            for item in raw_entry["files"]
            if isinstance(item, Mapping)
        }
        if staging.exists():
            if staging.is_symlink() or not staging.is_dir():
                raise ValueError("unpublished staging cleanup drift: target changed type")
            observed_files: set[str] = set()
            observed_dirs: set[str] = set()
            for child in sorted(staging.rglob("*"), key=lambda path: path.as_posix()):
                if child.is_symlink():
                    raise ValueError("unpublished staging cleanup drift: symlink appeared")
                relative = _relative_to_root(
                    child, root, label="unpublished staging cleanup entry"
                )
                if child.is_dir():
                    observed_dirs.add(relative)
                elif child.is_file():
                    expected = expected_files.get(relative)
                    if expected is None:
                        raise ValueError(
                            "unpublished staging cleanup drift: unplanned file appeared"
                        )
                    data = child.read_bytes()
                    if (
                        len(data) != expected["size_bytes"]
                        or _sha256_bytes(data) != expected["sha256"]
                    ):
                        raise ValueError(
                            "unpublished staging cleanup drift: planned file changed"
                        )
                    observed_files.add(relative)
                else:
                    raise ValueError(
                        "unpublished staging cleanup drift: non-file entry appeared"
                    )
            expected_dirs = set(str(value) for value in raw_entry["directories"])
            if not observed_dirs.issubset(expected_dirs):
                raise ValueError(
                    "unpublished staging cleanup drift: unplanned directory appeared"
                )
            for relative in sorted(observed_files):
                (root / relative).unlink()
            for relative in sorted(
                observed_dirs, key=lambda value: len(Path(value).parts), reverse=True
            ):
                (root / relative).rmdir()
            staging.rmdir()
        raw_entry["status"] = "reclaimed"
    ledger["cleanup_state"] = "complete"
    ledger["cleanup_sha256"] = _unpublished_cleanup_sha256(ledger)
    output = json.loads(_canonical_json(payload))
    output["unpublished_staging_cleanup"] = ledger
    return output


def _tables_from_result(result: OrchestrationResult) -> dict[str, list[dict[str, object]]]:
    return {
        "single_pass_observations.csv": [
            dict(row) for rows in result.profile_rows.values() for row in rows
        ],
        "pair_observations.csv": [dict(row) for row in result.pair_views["Uall"]],
        "advisor_2n_group_results.csv": [
            dict(row)
            for group in _GROUP_IDS
            for row in result.two_n_results[group]["group_rows"]
        ],
        "advisor_2n_directional_results.csv": [
            dict(row)
            for group in _GROUP_IDS
            for row in result.two_n_results[group]["directional_rows"]
        ],
        "advisor_2n_pair_validation.csv": [
            dict(row)
            for group in _GROUP_IDS
            for row in result.two_n_results[group]["pair_rows"]
        ],
    }


def _summary(
    tables: Mapping[str, Sequence[Mapping[str, object]]], *, program_status: str
) -> dict[str, object]:
    return {
        "program_denominator": 1,
        "completed_program_count": 1 if program_status == "complete" else 0,
        "coverage_limitation_program_count": (
            1 if program_status == "coverage_limitation" else 0
        ),
        "row_counts": {name: len(tables[name]) for name in _RAW_TABLES},
        "table_sha256": {
            name: _sha256_bytes(_canonical_json(list(tables[name])).encode("utf-8"))
            for name in _RAW_TABLES
        },
    }


def _directional_stage_semantics(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            str(key): value
            for key, value in row.items()
            if str(key) not in _DIRECTIONAL_CLEANUP_PROJECTION_FIELDS
        }
        for row in rows
    ]


def _pair_stage_semantics(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            str(key): value
            for key, value in row.items()
            if str(key) not in _PAIR_STAGE_CLEANUP_PROJECTION_FIELDS
        }
        for row in rows
    ]


def _stage_rows(value: object, *, label: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise ValueError(f"stage result binding has malformed {label}")
    return list(value)


def _validate_two_n_pair_replay_writeback(
    staged_rows: Sequence[Mapping[str, object]],
    current_rows: Sequence[Mapping[str, object]],
    witnesses: Sequence[Mapping[str, object]],
    *,
    group: str,
) -> None:
    """Recompute every post-stage replay field from the retained witnesses."""

    current_by_id = {
        str(row.get("row_id", "")): row for row in current_rows
    }
    if (
        any(not row_id for row_id in current_by_id)
        or len(current_by_id) != len(current_rows)
    ):
        raise ValueError(f"{group} 2N replay writeback row identity is ambiguous")
    witnesses_by_row: dict[str, list[Mapping[str, object]]] = {}
    for witness in witnesses:
        case = witness.get("case")
        advisor = case.get("advisor_pair_row") if isinstance(case, Mapping) else None
        if not isinstance(advisor, Mapping):
            raise ValueError(f"{group} 2N replay writeback witness is malformed")
        if str(case.get("group_id", "")) != group:
            continue
        row_id = str(advisor.get("row_id", ""))
        witnesses_by_row.setdefault(row_id, []).append(witness)
    observed_stage_ids: set[str] = set()
    for staged in staged_rows:
        row_id = str(staged.get("row_id", ""))
        if not row_id or row_id in observed_stage_ids:
            raise ValueError(f"{group} 2N replay writeback stage identity is ambiguous")
        observed_stage_ids.add(row_id)
        current = current_by_id.get(row_id)
        if current is None:
            raise ValueError(f"{group} 2N replay writeback coverage mismatch")
        expected = dict(staged)
        row_witnesses = witnesses_by_row.pop(row_id, [])
        if str(staged.get("false_authorization", "")).lower() == "true":
            try:
                _writeback_replay_results(expected, row_witnesses)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{group} 2N replay writeback evidence is invalid"
                ) from error
        elif row_witnesses:
            raise ValueError(
                f"{group} 2N replay writeback exists for a non-counterexample row"
            )
        if _canonical_json(expected) != _canonical_json(current):
            raise ValueError(
                f"{group} 2N replay writeback disagrees with bound witnesses"
            )
    if set(current_by_id) != observed_stage_ids or witnesses_by_row:
        raise ValueError(f"{group} 2N replay writeback coverage mismatch")


def _build_pair_stage_result_binding(
    result: OrchestrationResult,
    *,
    program_id: str,
    program_status: str,
) -> dict[str, object]:
    """Bind the canonical Uall pair semantics to its immutable source stage."""

    root = Path(result.out_dir).resolve(strict=False)
    stage_key = f"{program_id}:pairs:Uall"
    relative = result.stage_paths.get(stage_key)
    if relative is None:
        if program_status == "complete":
            raise ValueError("pair stage result binding is required for a complete program")
        return {}
    stage_dir = _require_inside(
        root / str(relative), root, label="Uall pair stage directory"
    )
    stage = _load_any_complete_stage(stage_dir)
    if stage is None:
        raise ValueError("pair stage result binding is invalid for Uall")
    staged_rows = _stage_rows(stage.get("rows"), label="pair rows")
    current_rows = list(result.pair_views["Uall"])
    staged_json = _canonical_json(_pair_stage_semantics(staged_rows))
    if staged_json != _canonical_json(_pair_stage_semantics(current_rows)):
        raise ValueError(
            "pair stage result binding disagrees with checkpoint semantics"
        )
    result_path = stage_dir / "result.json"
    complete_path = stage_dir / "complete.json"
    return {
        "schema_version": _PAIR_STAGE_BINDING_SCHEMA_VERSION,
        "group_id": "Uall",
        "stage_key": stage_key,
        "stage_path": str(relative),
        "result_sha256": _sha256_bytes(result_path.read_bytes()),
        "completion_sha256": _sha256_bytes(complete_path.read_bytes()),
        "pair_semantics_sha256": _sha256_bytes(staged_json.encode("utf-8")),
        "authority_granted": False,
        "proved_commute": False,
    }


def _build_two_n_stage_result_bindings(
    result: OrchestrationResult,
    *,
    program_id: str,
    program_status: str,
) -> dict[str, dict[str, object]]:
    """Bind checkpoint semantics to the immutable pre-cleanup 2N stages."""

    root = Path(result.out_dir).resolve(strict=False)
    bindings: dict[str, dict[str, object]] = {}
    for group in _GROUP_IDS:
        stage_key = f"{program_id}:two_n:{group}"
        relative = result.stage_paths.get(stage_key)
        if relative is None:
            continue
        stage_dir = _require_inside(
            root / str(relative), root, label=f"{group} 2N stage directory"
        )
        stage = _load_any_complete_stage(stage_dir)
        if stage is None:
            raise ValueError(f"2N stage result binding is invalid for {group}")
        staged_groups = _stage_rows(stage.get("group_rows"), label="group rows")
        staged_directionals = _stage_rows(
            stage.get("directional_rows"), label="directional rows"
        )
        staged_pairs = _stage_rows(stage.get("pair_rows"), label="pair rows")
        current = result.two_n_results[group]
        current_groups = list(current["group_rows"])
        current_directionals = list(current["directional_rows"])
        current_pairs = list(current["pair_rows"])
        _validate_two_n_pair_replay_writeback(
            staged_pairs,
            current_pairs,
            result.false_authorizations,
            group=group,
        )
        staged_group_json = _canonical_json(staged_groups)
        staged_directional_json = _canonical_json(
            _directional_stage_semantics(staged_directionals)
        )
        staged_pair_json = _canonical_json(staged_pairs)
        current_pair_json = _canonical_json(current_pairs)
        if (
            staged_group_json != _canonical_json(current_groups)
            or staged_directional_json
            != _canonical_json(_directional_stage_semantics(current_directionals))
        ):
            raise ValueError(
                f"2N stage result binding disagrees with checkpoint semantics for {group}"
            )
        result_path = stage_dir / "result.json"
        complete_path = stage_dir / "complete.json"
        bindings[group] = {
            "schema_version": _TWO_N_STAGE_BINDING_SCHEMA_VERSION,
            "group_id": group,
            "stage_key": stage_key,
            "stage_path": str(relative),
            "result_sha256": _sha256_bytes(result_path.read_bytes()),
            "completion_sha256": _sha256_bytes(complete_path.read_bytes()),
            "group_rows_sha256": _sha256_bytes(staged_group_json.encode("utf-8")),
            "directional_semantics_sha256": _sha256_bytes(
                staged_directional_json.encode("utf-8")
            ),
            "pair_stage_rows_sha256": _sha256_bytes(
                staged_pair_json.encode("utf-8")
            ),
            "pair_result_rows_sha256": _sha256_bytes(
                current_pair_json.encode("utf-8")
            ),
            "authority_granted": False,
            "proved_commute": False,
        }
    stage_backing_required = program_status == "complete" or any(
        str(row.get("directional_status", ""))
        in {"authorized_all_others", "rejected_effect_changed"}
        for group in _GROUP_IDS
        for row in result.two_n_results[group]["directional_rows"]
    )
    if stage_backing_required and set(bindings) != set(_GROUP_IDS):
        raise ValueError(
            "2N stage result binding must cover U14, U30, and Uall"
        )
    return bindings


def _validate_pair_stage_result_binding(
    payload: Mapping[str, object],
    *,
    program: str,
    program_status: str,
    uall_pair_rows: Sequence[Mapping[str, object]],
) -> None:
    raw_binding = payload.get("pair_stage_result_binding")
    stage_paths = payload.get("stage_paths")
    if not isinstance(raw_binding, Mapping) or not isinstance(stage_paths, Mapping):
        raise _checkpoint_error("pair stage result binding is malformed")
    stage_key = f"{program}:pairs:Uall"
    relative = stage_paths.get(stage_key)
    if relative is None:
        if raw_binding:
            raise _checkpoint_error("pair stage result binding coverage mismatch")
        if program_status == "complete":
            raise _checkpoint_error(
                "pair stage result binding is required for a complete program"
            )
        return
    if not raw_binding:
        raise _checkpoint_error("pair stage result binding coverage mismatch")
    _require_report_only(raw_binding, label="pair stage result binding")
    expected_semantics_sha = _sha256_bytes(
        _canonical_json(_pair_stage_semantics(uall_pair_rows)).encode("utf-8")
    )
    if (
        raw_binding.get("schema_version") != _PAIR_STAGE_BINDING_SCHEMA_VERSION
        or str(raw_binding.get("group_id", "")) != "Uall"
        or str(raw_binding.get("stage_key", "")) != stage_key
        or str(raw_binding.get("stage_path", "")) != str(relative)
        or not _valid_sha256(raw_binding.get("result_sha256", ""))
        or not _valid_sha256(raw_binding.get("completion_sha256", ""))
        or str(raw_binding.get("pair_semantics_sha256", ""))
        != expected_semantics_sha
    ):
        raise _checkpoint_error(
            "pair stage result binding disagrees with checkpoint semantics"
        )


def _validate_pair_stage_result_source(
    payload: Mapping[str, object], *, isolation_root: Path
) -> None:
    """Revalidate retained Uall pair result/marker bytes after cleanup."""

    binding = payload.get("pair_stage_result_binding")
    pair_views = payload.get("pair_views")
    if not isinstance(binding, Mapping) or not isinstance(pair_views, Mapping):
        raise _checkpoint_error("pair stage source binding is malformed")
    if not binding:
        return
    current_rows = pair_views.get("Uall")
    if not isinstance(current_rows, list):
        raise _checkpoint_error("pair stage source result is malformed")
    root = Path(isolation_root).resolve(strict=False)
    stage_dir = _require_inside(
        root / str(binding.get("stage_path", "")),
        root,
        label="Uall pair stage source",
    )
    result_path = stage_dir / "result.json"
    complete_path = stage_dir / "complete.json"
    try:
        result_data = result_path.read_bytes()
        completion_data = complete_path.read_bytes()
    except OSError as error:
        raise _checkpoint_error("Uall pair stage source is missing") from error
    if (
        _sha256_bytes(result_data) != str(binding.get("result_sha256", ""))
        or _sha256_bytes(completion_data)
        != str(binding.get("completion_sha256", ""))
    ):
        raise _checkpoint_error("Uall pair stage source byte hash mismatch")
    try:
        stage = json.loads(result_data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _checkpoint_error("Uall pair stage source is malformed") from error
    if not isinstance(stage, Mapping):
        raise _checkpoint_error("Uall pair stage source result is malformed")
    staged_rows = _stage_rows(stage.get("rows"), label="pair rows")
    if _canonical_json(_pair_stage_semantics(staged_rows)) != _canonical_json(
        _pair_stage_semantics(current_rows)
    ):
        raise _checkpoint_error(
            "Uall pair stage source semantics disagree with checkpoint"
        )


def _validate_two_n_stage_result_bindings(
    payload: Mapping[str, object],
    *,
    program: str,
    program_status: str,
    two_n_groups: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
) -> None:
    raw_bindings = payload.get("two_n_stage_result_bindings")
    stage_paths = payload.get("stage_paths")
    if not isinstance(raw_bindings, Mapping) or not isinstance(stage_paths, Mapping):
        raise _checkpoint_error("2N stage result bindings are malformed")
    expected_groups = {
        group
        for group in _GROUP_IDS
        if f"{program}:two_n:{group}" in stage_paths
    }
    required = program_status == "complete" or any(
        str(row.get("directional_status", ""))
        in {"authorized_all_others", "rejected_effect_changed"}
        for group in _GROUP_IDS
        for row in two_n_groups[group]["directional_rows"]
    )
    if required:
        expected_groups = set(_GROUP_IDS)
    if set(str(group) for group in raw_bindings) != expected_groups:
        raise _checkpoint_error("2N stage result binding coverage mismatch")
    for group in expected_groups:
        binding = raw_bindings.get(group)
        if not isinstance(binding, Mapping):
            raise _checkpoint_error(f"{group} 2N stage result binding is malformed")
        _require_report_only(binding, label=f"{group} 2N stage result binding")
        stage_key = f"{program}:two_n:{group}"
        group_rows = list(two_n_groups[group]["group_rows"])
        directionals = list(two_n_groups[group]["directional_rows"])
        pair_rows = list(two_n_groups[group]["pair_rows"])
        expected_group_sha = _sha256_bytes(
            _canonical_json(group_rows).encode("utf-8")
        )
        expected_directional_sha = _sha256_bytes(
            _canonical_json(_directional_stage_semantics(directionals)).encode(
                "utf-8"
            )
        )
        expected_pair_result_sha = _sha256_bytes(
            _canonical_json(pair_rows).encode("utf-8")
        )
        if (
            binding.get("schema_version") != _TWO_N_STAGE_BINDING_SCHEMA_VERSION
            or str(binding.get("group_id", "")) != group
            or str(binding.get("stage_key", "")) != stage_key
            or str(binding.get("stage_path", ""))
            != str(stage_paths.get(stage_key, ""))
            or not _valid_sha256(binding.get("result_sha256", ""))
            or not _valid_sha256(binding.get("completion_sha256", ""))
            or str(binding.get("group_rows_sha256", "")) != expected_group_sha
            or str(binding.get("directional_semantics_sha256", ""))
            != expected_directional_sha
            or not _valid_sha256(binding.get("pair_stage_rows_sha256", ""))
            or str(binding.get("pair_result_rows_sha256", ""))
            != expected_pair_result_sha
        ):
            raise _checkpoint_error(
                f"{group} 2N stage result binding disagrees with checkpoint semantics"
            )


def _validate_two_n_stage_result_sources(
    payload: Mapping[str, object], *, isolation_root: Path
) -> None:
    """Revalidate bound result/marker bytes retained after artifact cleanup."""

    bindings = payload.get("two_n_stage_result_bindings")
    two_n = payload.get("two_n_results")
    witnesses = payload.get("false_authorizations")
    if (
        not isinstance(bindings, Mapping)
        or not isinstance(two_n, Mapping)
        or not isinstance(witnesses, list)
        or any(not isinstance(row, Mapping) for row in witnesses)
    ):
        raise _checkpoint_error("2N stage source bindings are malformed")
    root = Path(isolation_root).resolve(strict=False)
    for group, raw_binding in bindings.items():
        if not isinstance(raw_binding, Mapping):
            raise _checkpoint_error("2N stage source binding is malformed")
        stage_dir = _require_inside(
            root / str(raw_binding.get("stage_path", "")),
            root,
            label=f"{group} 2N stage source",
        )
        result_path = stage_dir / "result.json"
        complete_path = stage_dir / "complete.json"
        try:
            result_data = result_path.read_bytes()
            completion_data = complete_path.read_bytes()
        except OSError as error:
            raise _checkpoint_error(f"{group} 2N stage source is missing") from error
        if (
            _sha256_bytes(result_data)
            != str(raw_binding.get("result_sha256", ""))
            or _sha256_bytes(completion_data)
            != str(raw_binding.get("completion_sha256", ""))
        ):
            raise _checkpoint_error(f"{group} 2N stage source byte hash mismatch")
        try:
            stage = json.loads(result_data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise _checkpoint_error(f"{group} 2N stage source is malformed") from error
        group_result = two_n.get(group)
        if not isinstance(stage, Mapping) or not isinstance(group_result, Mapping):
            raise _checkpoint_error(f"{group} 2N stage source result is malformed")
        staged_groups = _stage_rows(stage.get("group_rows"), label="group rows")
        staged_directionals = _stage_rows(
            stage.get("directional_rows"), label="directional rows"
        )
        staged_pairs = _stage_rows(stage.get("pair_rows"), label="pair rows")
        current_groups = group_result.get("group_rows")
        current_directionals = group_result.get("directional_rows")
        current_pairs = group_result.get("pair_rows")
        if (
            not isinstance(current_groups, list)
            or not isinstance(current_directionals, list)
            or not isinstance(current_pairs, list)
            or _canonical_json(staged_groups) != _canonical_json(current_groups)
            or _canonical_json(_directional_stage_semantics(staged_directionals))
            != _canonical_json(_directional_stage_semantics(current_directionals))
            or _sha256_bytes(_canonical_json(staged_pairs).encode("utf-8"))
            != str(raw_binding.get("pair_stage_rows_sha256", ""))
        ):
            raise _checkpoint_error(
                f"{group} 2N stage source semantics disagree with checkpoint"
            )
        try:
            _validate_two_n_pair_replay_writeback(
                staged_pairs,
                current_pairs,
                witnesses,
                group=str(group),
            )
        except ValueError as error:
            raise _checkpoint_error(str(error)) from error


def _validate_runtime_control_provenance(
    provenance: object, *, manifest: str, program: str
) -> None:
    if not isinstance(provenance, Mapping):
        raise _checkpoint_error("canonical runtime fallback provenance is malformed")
    try:
        _limitation_reason(provenance)
    except ValueError as error:
        raise _checkpoint_error(
            "canonical runtime fallback provenance is invalid"
        ) from error
    control = provenance.get("control_payload")
    if not isinstance(control, Mapping):
        raise _checkpoint_error(
            "canonical runtime fallback requires its self-hashed control payload"
        )
    from .run_control import (
        ENFORCEMENT_MODE,
        RUNTIME_LIMITATION_KIND,
        _validate_run_control_payload,
    )

    frozen_programs = control.get("frozen_program_ids")
    control_data = (_canonical_json(control) + "\n").encode("utf-8")
    try:
        budget, skips, expected_control_id, expected_file_sha = (
            _validate_run_control_payload(
                dict(control),
                control_data,
                study_manifest_id=manifest,
                program_ids=(
                    tuple(str(value) for value in frozen_programs)
                    if isinstance(frozen_programs, list)
                    else ()
                ),
            )
        )
    except ValueError as error:
        raise _checkpoint_error(
            f"canonical runtime fallback run control is invalid: {error}"
        ) from error
    skip = skips.get(program)
    observed = provenance.get("observed_wall_time_s")
    if (
        str(provenance.get("control_id", "")) != expected_control_id
        or str(provenance.get("control_file_sha256", "")) != expected_file_sha
        or str(provenance.get("program_id", "")) != program
        or provenance.get("decision") != "skip"
        or provenance.get("enforcement_mode") != ENFORCEMENT_MODE
        or provenance.get("limitation_kind") != RUNTIME_LIMITATION_KIND
        or provenance.get("program_wall_time_budget_s") != budget
        or not isinstance(skip, Mapping)
        or isinstance(observed, bool)
        or not isinstance(observed, (int, float))
    ):
        raise _checkpoint_error(
            "canonical runtime fallback provenance is not control-bound"
        )
    assert isinstance(skip, Mapping)
    if (
        skip.get("limitation_kind") != RUNTIME_LIMITATION_KIND
        or str(skip.get("reason", "")) != str(provenance.get("reason", ""))
        or float(skip.get("observed_wall_time_s", -1)) != float(observed)
    ):
        raise _checkpoint_error(
            "canonical runtime fallback disagrees with its control skip decision"
        )


def _validate_unbound_runtime_fallback(
    payload: Mapping[str, object],
    *,
    manifest: str,
    program: str,
    program_status: str,
    limitation_kind: str,
    profile_rows: Sequence[Mapping[str, object]],
    pair_views: Mapping[str, Sequence[Mapping[str, object]]],
    two_n_groups: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
) -> None:
    """Permit no source stages only for the exact typed skip row family."""

    pair_binding = payload.get("pair_stage_result_binding")
    two_n_bindings = payload.get("two_n_stage_result_bindings")
    if not isinstance(pair_binding, Mapping) or not isinstance(two_n_bindings, Mapping):
        raise _checkpoint_error("stage result bindings are malformed")
    if pair_binding and set(str(value) for value in two_n_bindings) == set(_GROUP_IDS):
        return
    if pair_binding or two_n_bindings:
        raise _checkpoint_error(
            "stage result binding coverage is incomplete for executed evidence"
        )
    if (
        program_status != "coverage_limitation"
        or limitation_kind != "runtime_budget_exceeded"
    ):
        raise _checkpoint_error(
            "stage result bindings are required outside canonical runtime fallback"
        )
    uall_pairs = list(pair_views["Uall"])
    program_families = {
        str(row.get("program_family", "")) for row in uall_pairs
    }
    if len(program_families) != 1 or "" in program_families:
        raise _checkpoint_error(
            "canonical runtime fallback program family is ambiguous"
        )
    provenance = payload.get("run_control_provenance")
    if not isinstance(provenance, Mapping):
        raise _checkpoint_error("canonical runtime fallback provenance is malformed")
    groups = {
        group: tuple({"action_id": action_id} for action_id in sorted(
            str(row.get("action_id", ""))
            for row in two_n_groups[group]["directional_rows"]
        ))
        for group in _GROUP_IDS
    }
    try:
        expected = runtime_budget_limited_result(
            out_dir=Path("."),
            study_manifest_id=manifest,
            program_id=program,
            program_family=next(iter(program_families)),
            groups=groups,
            provenance=provenance,
        )
    except ValueError as error:
        raise _checkpoint_error("canonical runtime fallback is invalid") from error
    if _canonical_json(profile_rows) != _canonical_json(
        expected.profile_rows[program]
    ):
        raise _checkpoint_error(
            "canonical runtime fallback profile rows were rewritten"
        )
    for group in _GROUP_IDS:
        actual_pairs = list(pair_views[group])
        expected_pairs = list(expected.pair_views[group])
        pair_cleanup_matches = len(actual_pairs) == len(expected_pairs) and all(
            all(
                str(actual.get(field, "")) == str(reference.get(field, ""))
                for field in _PAIR_CLEANUP_SYNC_FIELDS
            )
            for actual, reference in zip(actual_pairs, expected_pairs)
        )
        actual_directionals = list(two_n_groups[group]["directional_rows"])
        expected_directionals = list(
            expected.two_n_results[group]["directional_rows"]
        )
        directional_cleanup_matches = (
            len(actual_directionals) == len(expected_directionals)
            and all(
                all(
                    str(actual.get(field, "")) == str(reference.get(field, ""))
                    for field in _DIRECTIONAL_CLEANUP_PROJECTION_FIELDS
                )
                for actual, reference in zip(
                    actual_directionals, expected_directionals
                )
            )
        )
        if (
            _canonical_json(_pair_stage_semantics(actual_pairs))
            != _canonical_json(_pair_stage_semantics(expected_pairs))
            or not pair_cleanup_matches
            or _canonical_json(two_n_groups[group]["group_rows"])
            != _canonical_json(expected.two_n_results[group]["group_rows"])
            or _canonical_json(
                _directional_stage_semantics(actual_directionals)
            )
            != _canonical_json(
                _directional_stage_semantics(expected_directionals)
            )
            or not directional_cleanup_matches
            or _canonical_json(two_n_groups[group]["pair_rows"])
            != _canonical_json(expected.two_n_results[group]["pair_rows"])
        ):
            raise _checkpoint_error(
                "canonical runtime fallback pair or 2N rows were rewritten"
            )


def _checkpoint_error(message: str) -> ValueError:
    return ValueError(f"program checkpoint {message}")


def _is_false(value: object) -> bool:
    return value is False or str(value).strip().lower() == "false"


def _require_report_only(row: Mapping[str, object], *, label: str) -> None:
    if not _is_false(row.get("authority_granted")) or not _is_false(
        row.get("proved_commute")
    ):
        raise _checkpoint_error(f"{label} violates report-only semantics")


def _require_rows(value: object, *, label: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise _checkpoint_error(f"{label} rows are malformed")
    return list(value)


def _pair_identity(row: Mapping[str, object], *, label: str) -> tuple[str, str, str]:
    program = str(row.get("program_id", ""))
    left = str(row.get("action_a_id", ""))
    right = str(row.get("action_b_id", ""))
    if not program or not left or not right or left >= right:
        raise _checkpoint_error(f"{label} has a noncanonical pair identity")
    return program, left, right


def _authorized_endpoint_ids(row: Mapping[str, object]) -> tuple[str, ...]:
    endpoints: list[str] = []
    for id_field, status_field in (
        ("action_a_id", "action_a_directional_status"),
        ("action_b_id", "action_b_directional_status"),
    ):
        if str(row.get(status_field, "")) == "authorized_all_others":
            action_id = str(row.get(id_field, ""))
            if not action_id:
                raise _checkpoint_error("false authorization has an empty authorized endpoint")
            endpoints.append(action_id)
    return tuple(endpoints)


def _checkpoint_replay_signature(record: Mapping[str, object]) -> str:
    # Checkpoint acceptance must recompute the exact same semantic projection
    # as live replay classification.  Evidence byte/hash/path validation stays
    # in ``_validate_checkpoint_replay_family`` below.
    return _replay_signature(record)


def _validate_checkpoint_replay_family(
    witness: Mapping[str, object], *, family: str
) -> None:
    records = witness.get(family)
    if not isinstance(records, list) or len(records) != 2 or any(
        not isinstance(record, Mapping) for record in records
    ):
        raise _checkpoint_error(
            f"false authorization replay family {family} must have exactly two repetitions"
        )
    artifact_paths_by_repetition: list[dict[str, str]] = []
    for record in records:
        assert isinstance(record, Mapping)
        status = str(record.get("status", ""))
        command = record.get("command")
        stderr = str(record.get("stderr", ""))
        artifacts = record.get("artifacts")
        artifact_sha = record.get("artifact_sha256")
        hard_states = record.get("hard_state_hashes")
        stages = record.get("stage_results")
        if (
            not status
            or not isinstance(command, list)
            or not command
            or any(not str(part) for part in command)
            or not isinstance(artifacts, Mapping)
            or not isinstance(artifact_sha, Mapping)
            or not isinstance(hard_states, Mapping)
            or not isinstance(stages, Mapping)
            or set(stages) != {"A", "B", "AB", "BA"}
        ):
            raise _checkpoint_error(
                f"false authorization replay family {family} evidence is incomplete"
            )
        if (
            str(record.get("stderr_sha256", "")) != _text_sha256(stderr)
            or str(record.get("command_sha256", ""))
            != _sha256_bytes(
                "\0".join(str(part) for part in command).encode("utf-8")
            )
        ):
            raise _checkpoint_error(
                f"false authorization replay family {family} transport hash mismatch"
            )
        artifact_names = {str(name) for name in artifacts}
        if (
            "S" not in artifact_names
            or set(str(name) for name in artifact_sha) != artifact_names
            or not set(str(name) for name in hard_states).issubset(artifact_names)
        ):
            raise _checkpoint_error(
                f"false authorization replay family {family} artifact evidence is incomplete"
            )
        for name in artifact_names:
            if not str(artifacts.get(name, "")) or not _valid_sha256(
                artifact_sha.get(name, "")
            ):
                raise _checkpoint_error(
                    f"false authorization replay family {family} artifact identity is invalid"
                )
            hard = str(hard_states.get(name, ""))
            if hard and not _valid_sha256(hard):
                raise _checkpoint_error(
                    f"false authorization replay family {family} hard-state identity is invalid"
                )
        for name in ("A", "B", "AB", "BA"):
            stage = stages[name]
            if not isinstance(stage, Mapping):
                raise _checkpoint_error(
                    f"false authorization replay family {family} stage evidence is malformed"
                )
            execution = str(stage.get("execution_status", ""))
            verifier = str(stage.get("verifier_status", ""))
            if not execution or not verifier:
                raise _checkpoint_error(
                    f"false authorization replay family {family} stage status is incomplete"
                )
            for field in ("command_sha256", "stderr_sha256", "error_fingerprint"):
                if not _valid_sha256(stage.get(field, "")):
                    raise _checkpoint_error(
                        f"false authorization replay family {family} stage signature is incomplete"
                    )
            if execution == "success":
                if (
                    verifier != "success"
                    or name not in artifact_names
                    or not _valid_sha256(stage.get("hard_state_id", ""))
                    or not _valid_sha256(stage.get("output_sha256", ""))
                ):
                    raise _checkpoint_error(
                        f"false authorization replay family {family} successful stage lacks evidence"
                    )
        merge_status = str(record.get("merge_status", ""))
        if not merge_status:
            raise _checkpoint_error(
                f"false authorization replay family {family} merge status is incomplete"
            )
        if merge_status == "complete" and "merged_input" not in artifact_names:
            raise _checkpoint_error(
                f"false authorization replay family {family} merge artifact is incomplete"
            )
        two_n_result = record.get("two_n_result")
        if not isinstance(two_n_result, Mapping):
            raise _checkpoint_error(
                f"false authorization replay family {family} 2N evidence is malformed"
            )
        if family == "two_n" and status == "success":
            if (
                "second_round_output" not in artifact_names
                or not _valid_sha256(
                    two_n_result.get("second_output_hard_state_id", "")
                )
            ):
                raise _checkpoint_error(
                    "false authorization replay family two_n lacks complete second-round evidence"
                )
        try:
            _validate_replay_record_evidence_claims(record)
        except ValueError as error:
            raise _checkpoint_error(
                f"false authorization replay family {family} evidence binding failed: {error}"
            ) from error
        artifact_paths_by_repetition.append(
            {str(name): str(path) for name, path in artifacts.items()}
        )
    for name in set(artifact_paths_by_repetition[0]).intersection(
        artifact_paths_by_repetition[1]
    ):
        if artifact_paths_by_repetition[0][name] == artifact_paths_by_repetition[1][name]:
            raise _checkpoint_error(
                f"false authorization replay family {family} repetitions share an artifact path"
            )
    family_statuses = witness.get("family_statuses")
    family_status = (
        str(family_statuses.get(family, ""))
        if isinstance(family_statuses, Mapping)
        else ""
    )
    if family_status == "stable" and (
        any(str(record.get("status", "")) != "success" for record in records)
        or _checkpoint_replay_signature(records[0])
        != _checkpoint_replay_signature(records[1])
    ):
        raise _checkpoint_error(
            f"false authorization replay family {family} stable signature mismatch"
        )


def _validate_checkpoint_replay_family_independence(
    witness: Mapping[str, object],
) -> None:
    paths_by_family: dict[str, set[str]] = {}
    commands_by_family: dict[str, set[str]] = {}
    for family in ("worker", "external_opt", "two_n"):
        records = witness.get(family)
        assert isinstance(records, list)
        family_paths: set[str] = set()
        family_commands: set[str] = set()
        for record in records:
            assert isinstance(record, Mapping)
            artifacts = record.get("artifacts")
            command = record.get("command")
            assert isinstance(artifacts, Mapping) and isinstance(command, list)
            family_paths.update(str(path) for path in artifacts.values())
            family_commands.add(_canonical_json([str(part) for part in command]))
        paths_by_family[family] = family_paths
        commands_by_family[family] = family_commands
    families = ("worker", "external_opt", "two_n")
    for index, left in enumerate(families):
        for right in families[index + 1 :]:
            if paths_by_family[left].intersection(paths_by_family[right]) or (
                commands_by_family[left].intersection(commands_by_family[right])
            ):
                raise _checkpoint_error(
                    "false authorization replay families require independent "
                    "artifact paths and commands"
                )


def _checkpoint_terminal_error_fingerprint(
    execution_status: str, verifier_status: str, stderr_sha256: str
) -> str:
    return _sha256_bytes(
        _canonical_json(
            {
                "execution_status": execution_status,
                "verifier_status": verifier_status,
                "stderr_sha256": stderr_sha256,
            }
        ).encode("utf-8")
    )


def _validate_checkpoint_false_auth_pair_binding(
    case: Mapping[str, object],
    *,
    manifest: str,
    program: str,
    left: str,
    right: str,
    observation_id: str,
    profiles_by_action: Mapping[str, Mapping[str, object]],
    source_pair: Mapping[str, object],
) -> None:
    observation = case.get("pair_observation")
    stages = case.get("expected_pair_stages")
    expected_hard_states = case.get("expected_hard_state_hashes")
    if (
        not isinstance(observation, Mapping)
        or not isinstance(stages, Mapping)
        or not isinstance(expected_hard_states, Mapping)
        or set(stages) != {"A", "B", "AB", "BA"}
        or str(observation.get("row_id", "")) != observation_id
        or str(observation.get("study_manifest_id", "")) != manifest
        or str(observation.get("program_id", "")) != program
        or str(observation.get("action_a_id", "")) != left
        or str(observation.get("action_b_id", "")) != right
        or str(source_pair.get("row_id", "")) != observation_id
    ):
        raise _checkpoint_error(
            "false authorization replay lacks exact A/B/AB/BA pair binding"
        )
    semantic_fields = (
        "a_status",
        "a_hard_state_id",
        "a_output_sha256",
        "a_verifier_status",
        "b_status",
        "b_hard_state_id",
        "b_output_sha256",
        "b_verifier_status",
        "ab_status",
        "ab_hard_state_id",
        "ab_output_sha256",
        "ab_verifier_status",
        "ab_stderr_sha256",
        "ba_status",
        "ba_hard_state_id",
        "ba_output_sha256",
        "ba_verifier_status",
        "ba_stderr_sha256",
        "dynamic_result",
    )
    if any(
        str(observation.get(field, "")) != str(source_pair.get(field, ""))
        for field in semantic_fields
    ):
        raise _checkpoint_error(
            "false authorization pair observation first-round/profile cross-binding drift"
        )
    for name, prefix, action_id in (
        ("A", "a", left),
        ("B", "b", right),
    ):
        profile = profiles_by_action.get(action_id)
        stage = stages.get(name)
        if profile is None or not isinstance(stage, Mapping):
            raise _checkpoint_error(
                "false authorization first-round stage has no exact profile"
            )
        expected_values = {
            "execution_status": str(profile.get("execution_status", "")),
            "verifier_status": str(profile.get("verifier_status", "")),
            "hard_state_id": str(profile.get("output_hard_state_id", "")),
            "output_sha256": str(profile.get("output_sha256", "")),
            "source_action_id": action_id,
            "source_stderr_sha256": "",
            "error_fingerprint": "",
        }
        if (
            expected_values["execution_status"] != "success"
            or expected_values["verifier_status"] != "success"
            or not _valid_sha256(expected_values["hard_state_id"])
            or not _valid_sha256(expected_values["output_sha256"])
            or str(observation.get(f"{prefix}_status", ""))
            != expected_values["execution_status"]
            or str(observation.get(f"{prefix}_verifier_status", ""))
            != expected_values["verifier_status"]
            or str(observation.get(f"{prefix}_hard_state_id", ""))
            != expected_values["hard_state_id"]
            or str(observation.get(f"{prefix}_output_sha256", ""))
            != expected_values["output_sha256"]
            or any(
                str(stage.get(field, "")) != value
                for field, value in expected_values.items()
            )
        ):
            raise _checkpoint_error(
                "false authorization pair first-round evidence is not bound to profile"
            )
    for name, prefix in (("AB", "ab"), ("BA", "ba")):
        stage = stages.get(name)
        if not isinstance(stage, Mapping):
            raise _checkpoint_error("false authorization pair stage is malformed")
        execution = str(observation.get(f"{prefix}_status", ""))
        verifier = str(observation.get(f"{prefix}_verifier_status", ""))
        hard_state = str(observation.get(f"{prefix}_hard_state_id", ""))
        output_sha = str(observation.get(f"{prefix}_output_sha256", ""))
        stderr_sha = str(observation.get(f"{prefix}_stderr_sha256", ""))
        expected_values = {
            "execution_status": execution,
            "verifier_status": verifier,
            "hard_state_id": hard_state if execution == "success" else "",
            "output_sha256": output_sha if execution == "success" else "",
            "source_pair_row_id": observation_id,
            "source_stderr_sha256": stderr_sha if execution != "success" else "",
            "error_fingerprint": (
                _checkpoint_terminal_error_fingerprint(
                    execution, verifier, stderr_sha
                )
                if execution != "success"
                else ""
            ),
        }
        if any(
            str(stage.get(field, "")) != value
            for field, value in expected_values.items()
        ):
            raise _checkpoint_error(
                "false authorization expected terminal stage is not pair-bound"
            )
    canonical_hard_states = {
        name: str(stage.get("hard_state_id", ""))
        for name, stage in stages.items()
        if isinstance(stage, Mapping)
        and str(stage.get("execution_status", "")) == "success"
    }
    if {
        str(name): str(value) for name, value in expected_hard_states.items()
    } != canonical_hard_states:
        raise _checkpoint_error(
            "false authorization expected hard-state map is not stage-bound"
        )


def _validate_program_checkpoint_payload(
    payload: Mapping[str, object], *, required_cleanup_state: str | None
) -> None:
    """Validate exact one-program evidence before trusting a resume marker."""

    if payload.get("schema_version") != PROGRAM_CHECKPOINT_SCHEMA_VERSION:
        raise _checkpoint_error("schema_version mismatch")
    _require_report_only(payload, label="payload")
    manifest = str(payload.get("study_manifest_id", "")).strip()
    program = str(payload.get("program_id", "")).strip()
    status = str(payload.get("program_status", ""))
    limitation = str(payload.get("limitation_kind", ""))
    if not manifest or not program:
        raise _checkpoint_error("requires non-empty manifest and program IDs")
    if status not in {"complete", "coverage_limitation"}:
        raise _checkpoint_error("has an invalid program_status")
    if status == "coverage_limitation" and not limitation:
        raise _checkpoint_error("coverage limitation requires a typed limitation_kind")
    if status == "complete" and limitation:
        raise _checkpoint_error("completed program cannot carry a limitation_kind")

    profile_rows = _require_rows(payload.get("profile_rows"), label="profile")
    if not profile_rows:
        raise _checkpoint_error("complete coverage cannot have an empty profile family")
    profile_actions: list[str] = []
    profiles_by_action: dict[str, Mapping[str, object]] = {}
    for row in profile_rows:
        _require_report_only(row, label="profile row")
        if (
            str(row.get("study_manifest_id", "")) != manifest
            or str(row.get("program_id", "")) != program
            or str(row.get("group_id", "")) != "Uall"
        ):
            raise _checkpoint_error("profile row binding mismatch")
        action_id = str(row.get("action_id", ""))
        if not action_id:
            raise _checkpoint_error("profile row action_id is empty")
        if (
            str(row.get("execution_status", "")) == "success"
            and str(row.get("artifact_materialized", "")).lower() != "true"
        ):
            raise _checkpoint_error(
                "materialized artifact flag is false for a successful profile"
            )
        profile_actions.append(action_id)
        profiles_by_action[action_id] = row
    if len(profile_actions) != len(set(profile_actions)):
        raise _checkpoint_error("profile action coverage is not unique")

    pair_views_raw = payload.get("pair_views")
    two_n_raw = payload.get("two_n_results")
    if not isinstance(pair_views_raw, Mapping) or set(pair_views_raw) != set(_GROUP_IDS):
        raise _checkpoint_error("pair views must contain exactly U14, U30, and Uall")
    if not isinstance(two_n_raw, Mapping) or set(two_n_raw) != set(_GROUP_IDS):
        raise _checkpoint_error("2N results must contain exactly U14, U30, and Uall")

    group_actions: dict[str, tuple[str, ...]] = {}
    pair_views: dict[str, list[Mapping[str, object]]] = {}
    two_n_groups: dict[str, dict[str, list[Mapping[str, object]]]] = {}
    for group in _GROUP_IDS:
        group_value = two_n_raw[group]
        if not isinstance(group_value, Mapping) or set(group_value) != {
            "group_rows",
            "directional_rows",
            "pair_rows",
        }:
            raise _checkpoint_error(f"{group} 2N result shape mismatch")
        group_rows = _require_rows(group_value["group_rows"], label=f"{group} group")
        directional = _require_rows(
            group_value["directional_rows"], label=f"{group} directional"
        )
        validation = _require_rows(group_value["pair_rows"], label=f"{group} pair")
        if len(group_rows) != 1 or not directional:
            raise _checkpoint_error(f"{group} must retain one group row and all directionals")
        action_ids: list[str] = []
        for row in (*group_rows, *directional, *validation):
            _require_report_only(row, label=f"{group} 2N row")
            if (
                str(row.get("study_manifest_id", "")) != manifest
                or str(row.get("program_id", "")) != program
                or str(row.get("group_id", "")) != group
            ):
                raise _checkpoint_error(f"{group} 2N row binding mismatch")
        for row in directional:
            action_id = str(row.get("action_id", ""))
            if not action_id:
                raise _checkpoint_error(f"{group} directional action_id is empty")
            action_ids.append(action_id)
        if len(action_ids) != len(set(action_ids)):
            raise _checkpoint_error(f"{group} directional action coverage is not unique")
        actions = tuple(sorted(action_ids))
        directional_by_action = {
            str(row.get("action_id", "")): row for row in directional
        }
        group_actions[group] = actions
        try:
            configured_n = int(group_rows[0].get("configured_n", -1))
        except (TypeError, ValueError) as error:
            raise _checkpoint_error(f"{group} configured_n is invalid") from error
        if configured_n != len(actions):
            raise _checkpoint_error(f"{group} configured_n disagrees with directionals")
        group_row = group_rows[0]
        group_profiles = [profiles_by_action[action] for action in actions]
        execution_statuses = [
            str(row.get("execution_status", "unknown")) for row in group_profiles
        ]
        directional_statuses = [
            str(row.get("directional_status", "")) for row in directional
        ]
        expected_group_fields: dict[str, object] = {
            "configured_n": len(actions),
            "successful_n": sum(value == "success" for value in execution_statuses),
            "active_n": sum(
                status_value == "success"
                and str(profile.get("activity_status", "")) == "active"
                for profile, status_value in zip(group_profiles, execution_statuses)
            ),
            "no_op_n": sum(
                status_value == "success"
                and str(profile.get("activity_status", "")) == "no_op"
                for profile, status_value in zip(group_profiles, execution_statuses)
            ),
            "failed_n": sum(
                value in {"invalid", "error", "unknown", "not_run"}
                for value in execution_statuses
            ),
            "timeout_n": sum(value == "timeout" for value in execution_statuses),
            "directional_authorized_count": sum(
                value == "authorized_all_others" for value in directional_statuses
            ),
            "directional_unavailable_count": sum(
                value != "authorized_all_others" for value in directional_statuses
            ),
        }
        merge_gate = str(group_row.get("all_n_merge_status", ""))
        second_gate = str(group_row.get("all_n_second_round_status", ""))
        status_set = set(directional_statuses)
        if status_set == {"authorized_all_others"}:
            expected_group_authorization = "authorized"
        elif merge_gate != "complete" or second_gate != "complete":
            expected_group_authorization = "group_precondition_unavailable"
        elif status_set <= {
            "authorized_all_others",
            "rejected_effect_changed",
        }:
            expected_group_authorization = "rejected"
        else:
            expected_group_authorization = "unknown"
        expected_group_fields["group_authorization_status"] = (
            expected_group_authorization
        )
        if any(
            group_row.get(field) != value
            for field, value in expected_group_fields.items()
        ):
            raise _checkpoint_error(
                f"{group} 2N group row derived fields disagree with profile and directional evidence"
            )
        source_row_ids = str(group_row.get("source_row_ids", ""))
        if source_row_ids:
            ordered_directionals = sorted(
                directional, key=lambda row: str(row.get("action_id", ""))
            )
            expected_source_ids = _canonical_json(
                [str(row.get("row_id", "")) for row in ordered_directionals]
            )
            if source_row_ids != expected_source_ids:
                raise _checkpoint_error(
                    f"{group} 2N group source rows disagree with directionals"
                )

        views = _require_rows(pair_views_raw[group], label=f"{group} pair view")
        expected_pairs = set(_pairs(actions))
        observed_pairs: dict[tuple[str, str], Mapping[str, object]] = {}
        for row in views:
            _require_report_only(row, label=f"{group} pair view row")
            if (
                str(row.get("study_manifest_id", "")) != manifest
                or str(row.get("program_id", "")) != program
                or str(row.get("group_id", "")) != group
            ):
                raise _checkpoint_error(f"{group} pair view binding mismatch")
            _program, left, right = _pair_identity(row, label=f"{group} pair view")
            if (left, right) in observed_pairs:
                raise _checkpoint_error(f"{group} pair view contains a duplicate pair")
            observed_pairs[(left, right)] = row
        if set(observed_pairs) != expected_pairs:
            raise _checkpoint_error(f"{group} pair view does not have exact pair coverage")

        observed_validation: set[tuple[str, str]] = set()
        for row in validation:
            _program, left, right = _pair_identity(row, label=f"{group} 2N pair")
            if (left, right) in observed_validation:
                raise _checkpoint_error(f"{group} 2N pair coverage is not unique")
            observed_validation.add((left, right))
            source = observed_pairs.get((left, right))
            if source is None or str(row.get("pair_observation_row_id", "")) != str(
                source.get("row_id", "")
            ):
                raise _checkpoint_error(f"{group} 2N pair observation binding mismatch")
            derived = derive_two_n_pair_fields(
                directional_by_action[left].get("directional_status", ""),
                directional_by_action[right].get("directional_status", ""),
                source.get("dynamic_result", "unknown"),
                observation_available=True,
            )
            core_fields = {
                field: value
                for field, value in derived.items()
                if field != "validation_status"
            }
            if any(
                str(row.get(field, "")) != value
                for field, value in core_fields.items()
            ):
                raise _checkpoint_error(
                    f"{group} 2N pair derived fields disagree with directional and pair evidence"
                )
            actual_validation = str(row.get("validation_status", ""))
            expected_validation = derived["validation_status"]
            limitation_validation = (
                "ground_truth_timeout"
                if derived["dynamic_result"] == "timeout"
                else "ground_truth_failed"
                if derived["dynamic_result"] == "failed"
                else ""
            )
            runtime_fallback_validation = (
                status == "coverage_limitation"
                and limitation == "runtime_budget_exceeded"
                and derived["false_authorization"] == "false"
                and not _authorized_endpoint_ids(row)
                and {
                    derived["action_a_directional_status"],
                    derived["action_b_directional_status"],
                }.issubset({"timeout", "round1_precondition_failed"})
                and bool(limitation_validation)
                and actual_validation == limitation_validation
            )
            if (
                actual_validation != expected_validation
                and not runtime_fallback_validation
            ):
                raise _checkpoint_error(
                    f"{group} 2N pair derived fields disagree with directional and pair evidence"
                )
        if observed_validation != expected_pairs:
            raise _checkpoint_error(f"{group} 2N pair rows do not have exact coverage")
        pair_views[group] = views
        two_n_groups[group] = {
            "group_rows": group_rows,
            "directional_rows": directional,
            "pair_rows": validation,
        }

    if set(group_actions["Uall"]) != set(profile_actions):
        raise _checkpoint_error("Uall profile and directional action coverage disagree")
    if not set(group_actions["U14"]).issubset(group_actions["U30"]) or not set(
        group_actions["U30"]
    ).issubset(group_actions["Uall"]):
        raise _checkpoint_error("groups require U14 subset U30 subset Uall")
    uall_cleanup = {
        _pair_identity(row, label="Uall cleanup view"): row
        for row in pair_views["Uall"]
    }
    uall_pairs_by_row_id = {
        str(row.get("row_id", "")): row for row in pair_views["Uall"]
    }
    if len(uall_pairs_by_row_id) != len(pair_views["Uall"]):
        raise _checkpoint_error("Uall pair row_id coverage is ambiguous")
    for group in _GROUP_IDS:
        for row in pair_views[group]:
            source = uall_cleanup.get(
                _pair_identity(row, label=f"{group} cleanup view")
            )
            filtered = {str(key): value for key, value in row.items() if key != "group_id"}
            canonical = (
                {
                    str(key): value
                    for key, value in source.items()
                    if key != "group_id"
                }
                if source is not None
                else None
            )
            if source is None or _canonical_json(filtered) != _canonical_json(canonical):
                raise _checkpoint_error(
                    f"{group} filtered pair view disagrees with canonical Uall evidence"
                )

    _validate_pair_stage_result_binding(
        payload,
        program=program,
        program_status=status,
        uall_pair_rows=pair_views["Uall"],
    )
    _validate_two_n_stage_result_bindings(
        payload,
        program=program,
        program_status=status,
        two_n_groups=two_n_groups,
    )
    _validate_unbound_runtime_fallback(
        payload,
        manifest=manifest,
        program=program,
        program_status=status,
        limitation_kind=limitation,
        profile_rows=profile_rows,
        pair_views=pair_views,
        two_n_groups=two_n_groups,
    )
    if status == "coverage_limitation" and limitation == "runtime_budget_exceeded":
        _validate_runtime_control_provenance(
            payload.get("run_control_provenance"),
            manifest=manifest,
            program=program,
        )

    tables_raw = payload.get("tables")
    if not isinstance(tables_raw, Mapping) or set(tables_raw) != set(_RAW_TABLES):
        raise _checkpoint_error("must contain exactly the five raw evidence tables")
    tables = {
        name: _require_rows(tables_raw[name], label=name) for name in _RAW_TABLES
    }
    expected_tables: dict[str, list[Mapping[str, object]]] = {
        "single_pass_observations.csv": profile_rows,
        "pair_observations.csv": pair_views["Uall"],
        "advisor_2n_group_results.csv": [
            row for group in _GROUP_IDS for row in two_n_groups[group]["group_rows"]
        ],
        "advisor_2n_directional_results.csv": [
            row
            for group in _GROUP_IDS
            for row in two_n_groups[group]["directional_rows"]
        ],
        "advisor_2n_pair_validation.csv": [
            row for group in _GROUP_IDS for row in two_n_groups[group]["pair_rows"]
        ],
    }
    for name in _RAW_TABLES:
        if _canonical_json(tables[name]) != _canonical_json(expected_tables[name]):
            raise _checkpoint_error(f"{name} disagrees with exact program evidence")
        row_ids = [str(row.get("row_id", "")) for row in tables[name]]
        if any(not row_id for row_id in row_ids) or len(row_ids) != len(set(row_ids)):
            raise _checkpoint_error(f"{name} row_id coverage is empty or ambiguous")

    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        raise _checkpoint_error("summary is malformed")
    expected_completed = 1 if status == "complete" else 0
    expected_limited = 1 if status == "coverage_limitation" else 0
    if (
        summary.get("program_denominator") != 1
        or summary.get("completed_program_count") != expected_completed
        or summary.get("coverage_limitation_program_count") != expected_limited
    ):
        raise _checkpoint_error("summary has an invalid explicit program denominator")
    expected_counts = {name: len(tables[name]) for name in _RAW_TABLES}
    expected_hashes = {
        name: _sha256_bytes(_canonical_json(tables[name]).encode("utf-8"))
        for name in _RAW_TABLES
    }
    if summary.get("row_counts") != expected_counts or summary.get(
        "table_sha256"
    ) != expected_hashes:
        raise _checkpoint_error("summary row_counts/table hashes disagree with evidence")

    witnesses = payload.get("false_authorizations")
    if not isinstance(witnesses, list) or any(not isinstance(row, Mapping) for row in witnesses):
        raise _checkpoint_error("false-authorization witnesses are malformed")
    expected_cases: dict[tuple[str, str], Mapping[str, object]] = {}
    for group in _GROUP_IDS:
        for row in two_n_groups[group]["pair_rows"]:
            if str(row.get("false_authorization", "")).lower() != "true":
                continue
            endpoints = _authorized_endpoint_ids(row)
            if not endpoints:
                raise _checkpoint_error(
                    "false authorization has no authorized endpoint"
                )
            for action_id in endpoints:
                key = (str(row.get("row_id", "")), action_id)
                if key in expected_cases:
                    raise _checkpoint_error(
                        "false authorization authorized endpoint is ambiguous"
                    )
                expected_cases[key] = row
    witnessed_cases: set[tuple[str, str]] = set()
    for witness in witnesses:
        case = witness.get("case")
        advisor = case.get("advisor_pair_row") if isinstance(case, Mapping) else None
        if not isinstance(case, Mapping) or not isinstance(advisor, Mapping):
            raise _checkpoint_error("false-authorization witness lacks its bound advisor row")
        if (
            str(case.get("study_manifest_id", "")) != manifest
            or str(case.get("program_id", "")) != program
        ):
            raise _checkpoint_error("false-authorization witness binding mismatch")
        _require_report_only(advisor, label="false-authorization advisor witness")
        row_id = str(advisor.get("row_id", ""))
        authorized_action = str(case.get("authorized_action_id", ""))
        key = (row_id, authorized_action)
        source = expected_cases.get(key)
        if source is None or key in witnessed_cases:
            raise _checkpoint_error(
                "false authorization authorized endpoint replay case is missing or duplicated"
            )
        witnessed_cases.add(key)
        group_id = str(source.get("group_id", ""))
        left = str(source.get("action_a_id", ""))
        right = str(source.get("action_b_id", ""))
        observation_id = str(source.get("pair_observation_row_id", ""))
        expected_case_id = canonical_row_id(
            "false-authorization",
            manifest,
            program,
            group_id,
            left,
            right,
            authorized_action,
            observation_id,
        )
        if (
            str(witness.get("case_id", "")) != expected_case_id
            or str(case.get("group_id", "")) != group_id
            or str(advisor.get("action_a_id", "")) != left
            or str(advisor.get("action_b_id", "")) != right
            or str(advisor.get("pair_observation_row_id", "")) != observation_id
            or str(advisor.get("false_authorization", "")).lower() != "true"
            or authorized_action not in _authorized_endpoint_ids(advisor)
            or any(
                str(advisor.get(field, "")) != str(source.get(field, ""))
                for field in (
                    "two_n_pair_status",
                    "action_a_directional_status",
                    "action_b_directional_status",
                    "dynamic_result",
                    "validation_status",
                )
            )
        ):
            raise _checkpoint_error(
                "false authorization authorized endpoint replay case identity mismatch"
            )
        expected_pair_stages = case.get("expected_pair_stages")
        expected_two_n = case.get("expected_two_n")
        source_pair = uall_pairs_by_row_id.get(observation_id)
        if (
            source_pair is None
            or not isinstance(expected_pair_stages, Mapping)
            or not isinstance(expected_two_n, Mapping)
            or not expected_two_n
            or str(case.get("profile_input_sha256", ""))
            != _sha256_bytes(_canonical_json(profile_rows).encode("utf-8"))
        ):
            raise _checkpoint_error(
                "false authorization replay family lacks its exact pair observation"
            )
        _validate_checkpoint_false_auth_pair_binding(
            case,
            manifest=manifest,
            program=program,
            left=left,
            right=right,
            observation_id=observation_id,
            profiles_by_action=profiles_by_action,
            source_pair=source_pair,
        )
        family_statuses = witness.get("family_statuses")
        if (
            not isinstance(family_statuses, Mapping)
            or set(family_statuses) != {"worker", "external_opt", "two_n"}
            or not str(witness.get("replay_status", ""))
            or str(witness.get("stable_false_authorization", ""))
            not in {"true", "false"}
        ):
            raise _checkpoint_error(
                "false authorization replay family status evidence is incomplete"
            )
        for family in ("worker", "external_opt", "two_n"):
            _validate_checkpoint_replay_family(witness, family=family)
        _validate_checkpoint_replay_family_independence(witness)
        stable, replay_status, recomputed_family_statuses = _stable_replay(
            [dict(record) for record in witness["worker"]],
            [dict(record) for record in witness["external_opt"]],
            [dict(record) for record in witness["two_n"]],
            case,
        )
        stored_family_statuses = {
            family: str(family_statuses[family])
            for family in ("worker", "external_opt", "two_n")
        }
        if (
            stored_family_statuses != recomputed_family_statuses
            or str(witness.get("replay_status", "")) != replay_status
            or str(witness.get("stable_false_authorization", ""))
            != ("true" if stable else "false")
        ):
            raise _checkpoint_error(
                "false authorization replay family status disagrees with recomputed evidence"
            )
    if witnessed_cases != set(expected_cases):
        raise _checkpoint_error(
            "false authorization is missing an exact authorized endpoint replay witness"
        )

    _validate_unpublished_staging_cleanup(
        payload.get("unpublished_staging_cleanup"),
        required_state=required_cleanup_state,
        stage_paths=payload.get("stage_paths"),
    )
    bindings = payload.get("materialized_artifact_bindings")
    if not isinstance(bindings, list) or any(
        not isinstance(binding, Mapping) for binding in bindings
    ):
        raise _checkpoint_error("materialized artifact bindings are malformed")
    if required_cleanup_state is None:
        return
    if required_cleanup_state == "planned" and bindings:
        raise _checkpoint_error("planned checkpoint cannot predeclare artifact bindings")
    ledger = payload.get("cleanup_ledger")
    if not isinstance(ledger, Mapping):
        raise _checkpoint_error("cleanup ledger is malformed")
    _require_report_only(ledger, label="cleanup ledger")
    if (
        ledger.get("cleanup_state") != required_cleanup_state
        or str(ledger.get("study_manifest_id", "")) != manifest
    ):
        raise _checkpoint_error(
            f"{required_cleanup_state} checkpoint has an inconsistent cleanup_state"
        )
    entries = ledger.get("entries")
    ledger_summary = ledger.get("summary")
    if not isinstance(entries, list) or any(not isinstance(entry, Mapping) for entry in entries):
        raise _checkpoint_error("cleanup ledger entries are malformed")
    if not isinstance(ledger_summary, Mapping):
        raise _checkpoint_error("cleanup ledger summary is malformed")
    affected_rows: dict[tuple[str, str], Mapping[str, object]] = {}
    for kind, rows in (
        ("pair_ab_ba", pair_views["Uall"]),
        (
            "two_n_second_round",
            [
                row
                for group in _GROUP_IDS
                for row in two_n_groups[group]["directional_rows"]
            ],
        ),
    ):
        for row in rows:
            key = (kind, str(row.get("row_id", "")))
            if not key[1] or key in affected_rows:
                raise _checkpoint_error("cleanup affected-row identity is ambiguous")
            affected_rows[key] = row
            if str(row.get("fail_closed_reason", "")).startswith(
                "runtime_budget_exceeded"
            ) and (
                str(row.get("cleanup_status", ""))
                != "retained_runtime_budget_exceeded"
            ):
                raise _checkpoint_error(
                    "runtime_budget_exceeded row lost its typed cleanup status"
                )
    entry_keys = [
        (str(entry.get("artifact_kind", "")), str(entry.get("source_row_id", "")))
        for entry in entries
    ]
    if len(entry_keys) != len(set(entry_keys)) or set(entry_keys) != set(affected_rows):
        raise _checkpoint_error("cleanup ledger does not exactly cover affected rows")
    seen_cleanup_ids: set[str] = set()
    for entry in entries:
        kind = str(entry.get("artifact_kind", ""))
        source_row_id = str(entry.get("source_row_id", ""))
        source = affected_rows[(kind, source_row_id)]
        identity = {
            "study_manifest_id": manifest,
            "artifact_kind": kind,
            "source_row_id": source_row_id,
            "group_id": str(source.get("group_id", "")),
            "program_id": str(source.get("program_id", "")),
            "action_id": str(source.get("action_id", "")),
        }
        expected_cleanup_id = _sha256_bytes(
            _canonical_json(identity).encode("utf-8")
        )
        cleanup_id = str(entry.get("cleanup_id", ""))
        if (
            cleanup_id != expected_cleanup_id
            or cleanup_id in seen_cleanup_ids
            or any(str(entry.get(field, "")) != value for field, value in identity.items())
        ):
            raise _checkpoint_error("cleanup ledger identity/hash binding mismatch")
        seen_cleanup_ids.add(cleanup_id)
        status_value = str(entry.get("cleanup_status", ""))
        materialized_field = (
            "artifact_materialized"
            if kind == "pair_ab_ba"
            else "second_output_materialized"
        )
        if (
            status_value not in {"planned", "reclaimed", "retained"}
            or str(entry.get("row_cleanup_status", ""))
            != str(source.get("cleanup_status", ""))
            or str(entry.get("artifact_materialized", ""))
            != str(source.get(materialized_field, ""))
        ):
            raise _checkpoint_error("cleanup ledger row status disagrees with evidence")
        path_fields = (
            (
                ("AB", "ab_output_path", "ab_output_sha256"),
                ("BA", "ba_output_path", "ba_output_sha256"),
            )
            if kind == "pair_ab_ba"
            else (("second_round", "second_output_path", "second_output_sha256"),)
        )
        artifacts = entry.get("artifacts")
        if not isinstance(artifacts, list) or any(
            not isinstance(item, Mapping) for item in artifacts
        ):
            raise _checkpoint_error("cleanup ledger artifacts are malformed")
        by_name = {str(item.get("name", "")): item for item in artifacts}
        if set(by_name) != {name for name, _path, _sha in path_fields}:
            raise _checkpoint_error("cleanup ledger artifact names are incomplete")
        reclaimed_count = retained_count = reclaimed_bytes = retained_bytes = 0
        for name, path_field, sha_field in path_fields:
            item = by_name[name]
            required = {
                "original_path",
                "actual_path",
                "quarantine_path",
                "sha256",
                "size_bytes",
                "reclaimed",
            }
            size = item.get("size_bytes")
            reclaimed = item.get("reclaimed")
            if (
                not required.issubset(item)
                or not isinstance(size, int)
                or isinstance(size, bool)
                or size < 0
                or not isinstance(reclaimed, bool)
                or str(item.get("sha256", "")) != str(source.get(sha_field, ""))
            ):
                raise _checkpoint_error("cleanup ledger artifact binding is invalid")
            actual = str(item.get("actual_path", ""))
            if reclaimed:
                if actual:
                    raise _checkpoint_error("reclaimed cleanup artifact still has a path")
                reclaimed_count += 1
                reclaimed_bytes += size
            else:
                if actual != str(source.get(path_field, "")):
                    raise _checkpoint_error(
                        "retained cleanup artifact path disagrees with evidence"
                    )
                retained_count += 1
                retained_bytes += size
        file_count = len(path_fields)
        if (
            entry.get("file_count") != file_count
            or entry.get("size_bytes") != reclaimed_bytes + retained_bytes
            or entry.get("reclaimed_file_count") != reclaimed_count
            or entry.get("reclaimed_bytes") != reclaimed_bytes
            or entry.get("retained_file_count") != retained_count
            or entry.get("retained_bytes") != retained_bytes
            or entry.get("planned_file_count")
            != (file_count if status_value == "planned" else 0)
            or entry.get("planned_bytes")
            != (
                reclaimed_bytes + retained_bytes if status_value == "planned" else 0
            )
        ):
            raise _checkpoint_error("cleanup ledger entry counters are inconsistent")
        if status_value == "planned" and (
            required_cleanup_state != "planned"
            or str(source.get(materialized_field, "")) != "true"
            or reclaimed_count
        ):
            raise _checkpoint_error("planned cleanup entry is not recoverable")
        if status_value == "reclaimed" and (
            required_cleanup_state != "complete"
            or str(source.get(materialized_field, "")) != "false"
            or reclaimed_count != file_count
        ):
            raise _checkpoint_error("reclaimed cleanup entry is inconsistent")
        if status_value == "retained" and not str(
            source.get("cleanup_status", "")
        ).startswith("retained_"):
            raise _checkpoint_error("retained cleanup entry lost its typed reason")
    counter_fields = (
        "reclaimed_file_count",
        "reclaimed_bytes",
        "retained_file_count",
        "retained_bytes",
        "planned_file_count",
        "planned_bytes",
    )
    for field in counter_fields:
        if ledger_summary.get(field) != sum(int(entry.get(field, 0)) for entry in entries):
            raise _checkpoint_error("cleanup ledger summary counters are inconsistent")


def _stage_boundaries(
    payload: Mapping[str, object], root: Path
) -> dict[str, Path]:
    stage_paths = payload.get("stage_paths")
    if not isinstance(stage_paths, Mapping):
        raise _checkpoint_error("stage paths are malformed")
    boundaries: dict[str, Path] = {}
    for raw_key, raw_path in stage_paths.items():
        key = str(raw_key)
        supplied = Path(str(raw_path))
        target = supplied if supplied.is_absolute() else root / supplied
        if target.is_symlink():
            raise ValueError(f"materialized artifact stage boundary is a symlink: {target}")
        boundaries[key] = _require_inside(
            target, root, label=f"materialized artifact stage {key}"
        )
    return boundaries


def _verify_row_transport_hashes(row: Mapping[str, object], *, label: str) -> None:
    if "stderr" in row:
        supplied = str(row.get("stderr_sha256", ""))
        if not _valid_sha256(supplied) or supplied != _text_sha256(row.get("stderr", "")):
            raise ValueError(f"materialized artifact {label} stderr hash mismatch")
    if "command" in row:
        command = row.get("command")
        if not isinstance(command, (list, tuple)):
            raise ValueError(f"materialized artifact {label} command is malformed")
        supplied = str(row.get("command_sha256", ""))
        actual = _sha256_bytes("\0".join(str(part) for part in command).encode("utf-8"))
        if not _valid_sha256(supplied) or supplied != actual:
            raise ValueError(f"materialized artifact {label} command hash mismatch")


def _materialized_artifact_binding(
    *,
    root: Path,
    boundary: Path,
    stage_key: str,
    source_kind: str,
    source_row_id: str,
    artifact_name: str,
    path_value: object,
    expected_sha256: object,
    expected_hard_state_id: object = "",
) -> dict[str, object]:
    supplied = Path(str(path_value))
    raw_path = supplied if supplied.is_absolute() else root / supplied
    if raw_path.is_symlink():
        raise ValueError(f"materialized artifact path is a symlink: {raw_path}")
    path = raw_path
    path = _require_inside(path, root, label="materialized artifact path")
    try:
        path.relative_to(boundary)
    except ValueError as error:
        raise ValueError(
            f"materialized artifact escapes its exact stage boundary: {path}"
        ) from error
    digest = str(expected_sha256).lower()
    if (
        path.is_symlink()
        or not path.is_file()
        or not _valid_sha256(digest)
        or _sha256_bytes(path.read_bytes()) != digest
    ):
        raise ValueError(f"materialized artifact is missing or sha256-mismatched: {path}")
    hard_state = _phasebatch_hard_state_id(path)
    expected_hard = str(expected_hard_state_id).lower()
    if expected_hard and (
        not _valid_sha256(expected_hard) or hard_state != expected_hard
    ):
        raise ValueError(f"materialized artifact hard-state mismatch: {path}")
    return {
        "source_kind": source_kind,
        "source_row_id": source_row_id,
        "artifact_name": artifact_name,
        "stage_key": stage_key,
        "stage_relative_path": boundary.relative_to(root).as_posix(),
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": digest,
        "hard_state_id": hard_state,
    }


def _build_materialized_artifact_bindings(
    payload: Mapping[str, object], *, isolation_root: Path
) -> list[dict[str, object]]:
    root = Path(isolation_root).resolve(strict=False)
    program = str(payload.get("program_id", ""))
    boundaries = _stage_boundaries(payload, root)
    bindings: list[dict[str, object]] = []

    def add(
        row: Mapping[str, object],
        *,
        source_kind: str,
        artifact_name: str,
        stage_key: str,
        path_field: str,
        sha_field: str,
        hard_field: str = "",
        materialized_field: str = "",
    ) -> None:
        path_value = str(row.get(path_field, ""))
        if (
            materialized_field
            and str(row.get(materialized_field, "")).lower() != "true"
        ):
            if path_value:
                supplied = Path(path_value)
                raw_path = supplied if supplied.is_absolute() else root / supplied
                path = _require_inside(
                    raw_path, root, label=f"unmaterialized {source_kind} artifact path"
                )
                if path.exists():
                    raise ValueError(
                        "materialized artifact flag is false while its path still exists: "
                        f"{source_kind}:{artifact_name}"
                    )
            return
        if not path_value:
            if materialized_field:
                raise ValueError(
                    f"materialized artifact is claimed without a path: {source_kind}"
                )
            return
        boundary = boundaries.get(stage_key)
        if boundary is None:
            raise ValueError(
                f"materialized artifact has no exact stage boundary: {stage_key}"
            )
        bindings.append(
            _materialized_artifact_binding(
                root=root,
                boundary=boundary,
                stage_key=stage_key,
                source_kind=source_kind,
                source_row_id=str(row.get("row_id", "")),
                artifact_name=artifact_name,
                path_value=path_value,
                expected_sha256=row.get(sha_field, ""),
                expected_hard_state_id=row.get(hard_field, "") if hard_field else "",
            )
        )

    profiles = payload.get("profile_rows")
    if not isinstance(profiles, list):
        raise _checkpoint_error("profile artifact rows are malformed")
    for row in profiles:
        if not isinstance(row, Mapping):
            raise _checkpoint_error("profile artifact row is malformed")
        _verify_row_transport_hashes(row, label="profile")
        add(
            row,
            source_kind="profile",
            artifact_name="output",
            stage_key=f"{program}:profiles",
            path_field="output_path",
            sha_field="output_sha256",
            hard_field="output_hard_state_id",
            materialized_field="artifact_materialized",
        )

    pair_views = payload.get("pair_views")
    if not isinstance(pair_views, Mapping) or not isinstance(pair_views.get("Uall"), list):
        raise _checkpoint_error("pair artifact rows are malformed")
    for row in pair_views["Uall"]:
        if not isinstance(row, Mapping):
            raise _checkpoint_error("pair artifact row is malformed")
        _verify_row_transport_hashes(row, label="pair")
        for direction in ("ab", "ba"):
            add(
                row,
                source_kind="pair",
                artifact_name=direction.upper(),
                stage_key=f"{program}:pairs:Uall",
                path_field=f"{direction}_output_path",
                sha_field=f"{direction}_output_sha256",
                hard_field=f"{direction}_hard_state_id",
                materialized_field="artifact_materialized",
            )

    two_n = payload.get("two_n_results")
    if not isinstance(two_n, Mapping):
        raise _checkpoint_error("2N artifact rows are malformed")
    for group in _GROUP_IDS:
        value = two_n.get(group)
        rows = value.get("directional_rows") if isinstance(value, Mapping) else None
        if not isinstance(rows, list):
            raise _checkpoint_error("2N directional artifact rows are malformed")
        for row in rows:
            if not isinstance(row, Mapping):
                raise _checkpoint_error("2N directional artifact row is malformed")
            _verify_row_transport_hashes(row, label="2N directional")
            add(
                row,
                source_kind="two_n_directional",
                artifact_name="merged_input",
                stage_key=f"{program}:two_n:{group}",
                path_field="merged_input_path",
                sha_field="merged_input_sha256",
                hard_field="merged_input_hard_state_id",
            )
            add(
                row,
                source_kind="two_n_directional",
                artifact_name="second_round_output",
                stage_key=f"{program}:two_n:{group}",
                path_field="second_output_path",
                sha_field="second_output_sha256",
                materialized_field="second_output_materialized",
            )

    witnesses = payload.get("false_authorizations")
    if not isinstance(witnesses, list):
        raise _checkpoint_error("replay artifact witnesses are malformed")
    for witness in witnesses:
        if not isinstance(witness, Mapping):
            raise _checkpoint_error("replay artifact witness is malformed")
        case_id = str(witness.get("case_id", ""))
        stage_key = f"replay:{case_id}"
        boundary = boundaries.get(stage_key)
        if boundary is None:
            raise ValueError(
                f"materialized replay artifact has no exact stage boundary: {case_id}"
            )
        for family in ("worker", "external_opt", "two_n"):
            records = witness.get(family)
            if not isinstance(records, list):
                raise _checkpoint_error("replay artifact family is malformed")
            for repetition, record in enumerate(records, start=1):
                if not isinstance(record, Mapping):
                    raise _checkpoint_error("replay artifact record is malformed")
                stderr = str(record.get("stderr", ""))
                if str(record.get("stderr_sha256", "")) != _text_sha256(stderr):
                    raise ValueError("materialized replay artifact stderr hash mismatch")
                command = record.get("command")
                if not isinstance(command, list) or str(
                    record.get("command_sha256", "")
                ) != _sha256_bytes(
                    "\0".join(str(part) for part in command).encode("utf-8")
                ):
                    raise ValueError("materialized replay artifact command hash mismatch")
                artifacts = record.get("artifacts")
                artifact_sha = record.get("artifact_sha256")
                hard_states = record.get("hard_state_hashes")
                if not all(
                    isinstance(value, Mapping)
                    for value in (artifacts, artifact_sha, hard_states)
                ):
                    raise _checkpoint_error("replay artifact maps are malformed")
                assert isinstance(artifacts, Mapping)
                assert isinstance(artifact_sha, Mapping)
                assert isinstance(hard_states, Mapping)
                for name in sorted(str(key) for key in artifacts):
                    bindings.append(
                        _materialized_artifact_binding(
                            root=root,
                            boundary=boundary,
                            stage_key=stage_key,
                            source_kind=f"replay_{family}_{repetition}",
                            source_row_id=case_id,
                            artifact_name=name,
                            path_value=artifacts[name],
                            expected_sha256=artifact_sha.get(name, ""),
                            expected_hard_state_id=hard_states.get(name, ""),
                        )
                    )
    bindings.sort(
        key=lambda row: (
            str(row["source_kind"]),
            str(row["source_row_id"]),
            str(row["artifact_name"]),
            str(row["relative_path"]),
        )
    )
    return bindings


def _validate_materialized_artifacts(
    payload: Mapping[str, object], *, isolation_root: Path
) -> None:
    stored = payload.get("materialized_artifact_bindings")
    if not isinstance(stored, list) or any(not isinstance(row, Mapping) for row in stored):
        raise ValueError("materialized artifact bindings are missing or malformed")
    actual = _build_materialized_artifact_bindings(
        payload, isolation_root=isolation_root
    )
    if _canonical_json(stored) != _canonical_json(actual):
        raise ValueError("materialized artifact binding set changed")


def program_result_payload(
    result: OrchestrationResult,
    *,
    program_id: str,
    program_status: str,
    limitation_kind: str,
    run_control_provenance: Mapping[str, object],
    unpublished_staging_cleanup: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Serialize exactly one program result with explicit denominator metadata."""

    program = str(program_id)
    if program_status not in {"complete", "coverage_limitation"}:
        raise ValueError("program checkpoint status must be complete or coverage_limitation")
    if set(result.profile_rows) != {program}:
        raise ValueError("program checkpoint must contain exactly one program profile family")
    tables = _tables_from_result(result)
    pair_stage_binding = _build_pair_stage_result_binding(
        result,
        program_id=program,
        program_status=program_status,
    )
    two_n_stage_bindings = _build_two_n_stage_result_bindings(
        result,
        program_id=program,
        program_status=program_status,
    )
    payload = {
        "schema_version": PROGRAM_CHECKPOINT_SCHEMA_VERSION,
        "study_manifest_id": result.study_manifest_id,
        "program_id": program,
        "program_status": program_status,
        "limitation_kind": str(limitation_kind),
        "run_control_provenance": dict(run_control_provenance),
        "profile_rows": [dict(row) for row in result.profile_rows[program]],
        "pair_views": {
            group: [dict(row) for row in result.pair_views[group]] for group in _GROUP_IDS
        },
        "two_n_results": {
            group: {
                key: [dict(row) for row in result.two_n_results[group][key]]
                for key in ("group_rows", "directional_rows", "pair_rows")
            }
            for group in _GROUP_IDS
        },
        "false_authorizations": [dict(row) for row in result.false_authorizations],
        "stage_paths": dict(result.stage_paths),
        "pair_stage_result_binding": pair_stage_binding,
        "two_n_stage_result_bindings": two_n_stage_bindings,
        "tables": tables,
        "cleanup_ledger": {},
        "unpublished_staging_cleanup": dict(
            unpublished_staging_cleanup or _empty_unpublished_staging_cleanup()
        ),
        "materialized_artifact_bindings": [],
        "summary": _summary(tables, program_status=program_status),
        "authority_granted": False,
        "proved_commute": False,
    }
    _validate_program_checkpoint_payload(payload, required_cleanup_state=None)
    return payload


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    data = (_canonical_json(payload) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def publish_program_checkpoint(
    checkpoint_dir: Path,
    payload: Mapping[str, object],
    *,
    expected_input_sha256: str,
    checkpoint_state: str,
    isolation_root: Path,
) -> None:
    """Publish an immutable version then atomically select it with active.json."""

    if not _valid_sha256(expected_input_sha256):
        raise ValueError("program checkpoint input identity must be sha256-shaped")
    if checkpoint_state not in {"planned", "complete"}:
        raise ValueError("program checkpoint state must be planned or complete")
    root = Path(isolation_root).resolve(strict=False)
    checkpoint = _require_inside(
        checkpoint_dir, root, label="program checkpoint directory"
    )
    normalized = _json_value(payload)
    if not isinstance(normalized, dict):
        raise ValueError("program checkpoint payload must be a mapping")
    if normalized.get("authority_granted") is not False or normalized.get(
        "proved_commute"
    ) is not False:
        raise ValueError("program checkpoint must remain report-only")
    _validate_program_checkpoint_payload(
        normalized, required_cleanup_state=checkpoint_state
    )
    _validate_pair_stage_result_source(normalized, isolation_root=root)
    _validate_two_n_stage_result_sources(normalized, isolation_root=root)
    if checkpoint_state == "complete":
        _validate_materialized_artifacts(normalized, isolation_root=root)
    result_data = (_canonical_json(normalized) + "\n").encode("utf-8")
    result_sha256 = _sha256_bytes(result_data)
    version_id = _sha256_bytes(
        (
            expected_input_sha256
            + "\0"
            + checkpoint_state
            + "\0"
            + result_sha256
        ).encode("utf-8")
    )
    # Keep the full digest in both markers while using the same deterministic
    # 64-bit path component convention as orchestration.  This leaves enough
    # headroom for Windows' still-common 260-character materialization limit.
    version_component = version_id[:16]
    versions = checkpoint / "versions"
    version = versions / version_component
    if not version.is_dir():
        versions.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".v.stage-", dir=versions))
        try:
            (staging / "result.json").write_bytes(result_data)
            _atomic_write_json(
                staging / "complete.json",
                {
                    "schema_version": PROGRAM_CHECKPOINT_SCHEMA_VERSION,
                    "version_id": version_id,
                    "input_sha256": expected_input_sha256,
                    "checkpoint_state": checkpoint_state,
                    "result_sha256": result_sha256,
                    "authority_granted": False,
                    "proved_commute": False,
                },
            )
            os.replace(staging, version)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise
    pointer = {
        "schema_version": PROGRAM_CHECKPOINT_POINTER_SCHEMA_VERSION,
        "version_id": version_id,
        "version_path": f"versions/{version_component}",
        "input_sha256": expected_input_sha256,
        "checkpoint_state": checkpoint_state,
        "result_sha256": result_sha256,
        "authority_granted": False,
        "proved_commute": False,
    }
    _atomic_write_json(checkpoint / "active.json", pointer)
    loaded = load_program_checkpoint(
        checkpoint,
        expected_input_sha256=expected_input_sha256,
        isolation_root=root,
    )
    if loaded != (normalized, checkpoint_state):
        raise ValueError("published program checkpoint failed self-validation")


def load_program_checkpoint(
    checkpoint_dir: Path,
    *,
    expected_input_sha256: str,
    isolation_root: Path,
) -> tuple[dict[str, object], str] | None:
    """Return only a pointer-selected, hash-valid planned or complete checkpoint."""

    checkpoint = _require_inside(
        checkpoint_dir,
        Path(isolation_root).resolve(strict=False),
        label="program checkpoint directory",
    )
    active = checkpoint / "active.json"
    if not _valid_sha256(expected_input_sha256):
        raise ValueError("program checkpoint expected input identity is malformed")
    if not active.is_file():
        return None
    try:
        pointer = json.loads(active.read_text(encoding="utf-8"))
        if not isinstance(pointer, dict):
            raise ValueError("active pointer is not an object")
        version_id = str(pointer.get("version_id", ""))
        state = str(pointer.get("checkpoint_state", ""))
        if (
            pointer.get("schema_version")
            != PROGRAM_CHECKPOINT_POINTER_SCHEMA_VERSION
            or pointer.get("input_sha256") != expected_input_sha256
            or pointer.get("authority_granted") is not False
            or pointer.get("proved_commute") is not False
            or not _valid_sha256(version_id)
            or state not in {"planned", "complete"}
            or pointer.get("version_path") != f"versions/{version_id[:16]}"
        ):
            raise ValueError("active pointer binding is invalid")
        version = checkpoint / "versions" / version_id[:16]
        complete_path = version / "complete.json"
        result_path = version / "result.json"
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        result_data = result_path.read_bytes()
        result_sha256 = _sha256_bytes(result_data)
        if (
            not isinstance(complete, dict)
            or complete.get("schema_version") != PROGRAM_CHECKPOINT_SCHEMA_VERSION
            or complete.get("version_id") != version_id
            or complete.get("input_sha256") != expected_input_sha256
            or complete.get("checkpoint_state") != state
            or complete.get("result_sha256") != result_sha256
            or pointer.get("result_sha256") != result_sha256
            or complete.get("authority_granted") is not False
            or complete.get("proved_commute") is not False
        ):
            raise ValueError("selected immutable version binding is invalid")
        expected_version_id = _sha256_bytes(
            (expected_input_sha256 + "\0" + state + "\0" + result_sha256).encode(
                "utf-8"
            )
        )
        if version_id != expected_version_id:
            raise ValueError("selected immutable version identity is invalid")
        payload = json.loads(result_data.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("selected result is not an object")
        _validate_program_checkpoint_payload(
            payload, required_cleanup_state=state
        )
        _validate_pair_stage_result_source(
            payload, isolation_root=Path(isolation_root).resolve(strict=False)
        )
        _validate_two_n_stage_result_sources(
            payload, isolation_root=Path(isolation_root).resolve(strict=False)
        )
        if state == "complete":
            _validate_materialized_artifacts(
                payload, isolation_root=Path(isolation_root).resolve(strict=False)
            )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(
            f"program checkpoint active selection is invalid: {checkpoint}: {error}"
        ) from error
    if (
        payload.get("authority_granted") is not False
        or payload.get("proved_commute") is not False
    ):
        raise ValueError("program checkpoint selected payload violates report-only semantics")
    return payload, state


def _payload_cleanup_inputs(
    payload: Mapping[str, object],
) -> tuple[
    list[Mapping[str, object]],
    dict[str, list[Mapping[str, object]]],
    list[Mapping[str, object]],
]:
    pair_views = payload.get("pair_views")
    two_n = payload.get("two_n_results")
    false_authorizations = payload.get("false_authorizations")
    if (
        not isinstance(pair_views, Mapping)
        or not isinstance(two_n, Mapping)
        or not isinstance(false_authorizations, list)
        or any(not isinstance(row, Mapping) for row in false_authorizations)
    ):
        raise ValueError("program checkpoint cleanup inputs are malformed")
    uall = pair_views.get("Uall")
    if not isinstance(uall, list) or any(not isinstance(row, Mapping) for row in uall):
        raise ValueError("program checkpoint Uall pair rows are malformed")
    directionals: dict[str, list[Mapping[str, object]]] = {}
    for group in _GROUP_IDS:
        group_value = two_n.get(group)
        rows = group_value.get("directional_rows") if isinstance(group_value, Mapping) else None
        if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
            raise ValueError("program checkpoint directional rows are malformed")
        directionals[group] = list(rows)
    return list(uall), directionals, list(false_authorizations)


def _with_cleanup(
    payload: Mapping[str, object],
    *,
    pair_rows: Sequence[Mapping[str, object]],
    directional_rows_by_group: Mapping[str, Sequence[Mapping[str, object]]],
    ledger: Mapping[str, object],
) -> dict[str, object]:
    output = json.loads(_canonical_json(payload))
    pair_views = output["pair_views"]
    two_n = output["two_n_results"]
    assert isinstance(pair_views, dict) and isinstance(two_n, dict)
    cleaned_pairs: list[dict[str, object]] = []
    pair_by_identity: dict[tuple[str, str, str], dict[str, object]] = {}
    for raw in pair_rows:
        row = dict(raw)
        if str(row.get("fail_closed_reason", "")).startswith(
            "runtime_budget_exceeded"
        ):
            row["artifact_available"] = "false"
            row["artifact_materialized"] = "false"
            row["cleanup_status"] = "retained_runtime_budget_exceeded"
        if (
            str(row.get("artifact_materialized", "")).lower() == "false"
            and str(row.get("cleanup_status", "")) == "reclaimed_nonwitness"
        ):
            row["ab_output_path"] = ""
            row["ba_output_path"] = ""
        identity = _pair_identity(row, label="cleanup Uall pair")
        if identity in pair_by_identity:
            raise _checkpoint_error("cleanup Uall pair identity is ambiguous")
        pair_by_identity[identity] = row
        cleaned_pairs.append(row)
    synchronized_views: dict[str, list[dict[str, object]]] = {}
    for group in _GROUP_IDS:
        rows: list[dict[str, object]] = []
        for raw in pair_views[group]:
            row = dict(raw)
            identity = _pair_identity(row, label=f"cleanup {group} pair view")
            source = pair_by_identity.get(identity)
            if source is None:
                raise _checkpoint_error(
                    f"cleanup {group} pair view has no canonical Uall identity"
                )
            for field in _PAIR_CLEANUP_SYNC_FIELDS:
                row[field] = source.get(field, "")
            rows.append(row)
        synchronized_views[group] = rows
    pair_views.update(synchronized_views)

    cleaned_directionals: dict[str, list[dict[str, object]]] = {}
    for group in _GROUP_IDS:
        group_value = two_n[group]
        assert isinstance(group_value, dict)
        rows = []
        for raw in directional_rows_by_group[group]:
            row = dict(raw)
            if str(row.get("fail_closed_reason", "")).startswith(
                "runtime_budget_exceeded"
            ):
                row["second_output_materialized"] = "false"
                row["cleanup_status"] = "retained_runtime_budget_exceeded"
            if (
                str(row.get("second_output_materialized", "")).lower() == "false"
                and str(row.get("cleanup_status", "")) == "reclaimed_nonwitness"
            ):
                row["second_output_path"] = ""
            rows.append(row)
        cleaned_directionals[group] = rows
        group_value["directional_rows"] = rows

    normalized_ledger = json.loads(_canonical_json(ledger))
    if isinstance(normalized_ledger, dict) and isinstance(
        normalized_ledger.get("entries"), list
    ):
        affected = {
            str(row.get("row_id", "")): row
            for row in (
                *cleaned_pairs,
                *(
                    row
                    for group in _GROUP_IDS
                    for row in cleaned_directionals[group]
                ),
            )
        }
        for entry in normalized_ledger["entries"]:
            if not isinstance(entry, dict):
                continue
            source = affected.get(str(entry.get("source_row_id", "")))
            if source is None:
                continue
            entry["row_cleanup_status"] = str(source.get("cleanup_status", ""))
            materialized_field = (
                "artifact_materialized"
                if str(entry.get("artifact_kind", "")) == "pair_ab_ba"
                else "second_output_materialized"
            )
            entry["artifact_materialized"] = str(
                source.get(materialized_field, "")
            )
            if str(source.get("cleanup_status", "")) == (
                "retained_runtime_budget_exceeded"
            ):
                entry["retention_reason"] = "retained_runtime_budget_exceeded"
    output["cleanup_ledger"] = normalized_ledger
    result = OrchestrationResult(
        out_dir=Path("."),
        study_manifest_id=str(output["study_manifest_id"]),
        profile_rows={
            str(output["program_id"]): tuple(
                dict(row) for row in output["profile_rows"]
            )
        },
        pair_views={
            group: tuple(dict(row) for row in pair_views[group]) for group in _GROUP_IDS
        },
        two_n_results={
            group: {
                key: tuple(dict(row) for row in two_n[group][key])
                for key in ("group_rows", "directional_rows", "pair_rows")
            }
            for group in _GROUP_IDS
        },
        false_authorizations=tuple(
            dict(row) for row in output["false_authorizations"]
        ),
        stage_paths=dict(output["stage_paths"]),
    )
    tables = _tables_from_result(result)
    output["tables"] = tables
    output["summary"] = _summary(
        tables, program_status=str(output["program_status"])
    )
    return output


def finalize_program_evidence(
    checkpoint_dir: Path,
    payload: Mapping[str, object],
    *,
    expected_input_sha256: str,
    isolation_root: Path,
) -> dict[str, object]:
    """Resume or perform the planned->cleanup->complete per-program protocol."""

    loaded = load_program_checkpoint(
        checkpoint_dir,
        expected_input_sha256=expected_input_sha256,
        isolation_root=isolation_root,
    )
    if loaded is not None and loaded[1] == "complete":
        return loaded[0]
    if loaded is None:
        _validate_program_checkpoint_payload(payload, required_cleanup_state=None)
    manifest = str(payload.get("study_manifest_id", ""))
    working: Mapping[str, object]
    if loaded is not None:
        working = loaded[0]
        if str(working.get("study_manifest_id", "")) != manifest:
            raise ValueError("planned checkpoint manifest changed during cleanup resume")
        planned_ledger = working.get("cleanup_ledger")
        if not isinstance(planned_ledger, Mapping) or planned_ledger.get(
            "cleanup_state"
        ) != "planned":
            raise ValueError("planned checkpoint cleanup ledger is malformed")
    else:
        working = dict(payload)
        pair_rows, directionals, false_authorizations = _payload_cleanup_inputs(working)
        planned = plan_intermediate_artifact_cleanup(
            isolation_root=isolation_root,
            study_manifest_id=manifest,
            pair_rows=pair_rows,
            directional_rows_by_group=directionals,
            false_authorizations=false_authorizations,
        )
        working = _with_cleanup(
            working,
            pair_rows=planned.pair_rows,
            directional_rows_by_group=planned.directional_rows_by_group,
            ledger=planned.ledger,
        )
        publish_program_checkpoint(
            checkpoint_dir,
            working,
            expected_input_sha256=expected_input_sha256,
            checkpoint_state="planned",
            isolation_root=isolation_root,
        )
        planned_ledger = planned.ledger

    working = _complete_unpublished_staging_cleanup(
        working, isolation_root=isolation_root
    )
    pair_rows, directionals, false_authorizations = _payload_cleanup_inputs(working)
    completed = compact_intermediate_artifacts(
        isolation_root=isolation_root,
        study_manifest_id=manifest,
        pair_rows=pair_rows,
        directional_rows_by_group=directionals,
        false_authorizations=false_authorizations,
        planned_ledger=planned_ledger,
    )
    if not cleanup_journals_resolved(
        isolation_root=isolation_root, ledger=completed.ledger
    ):
        raise RuntimeError("program cleanup journal remains prepared; planned checkpoint is recoverable")
    output = _with_cleanup(
        working,
        pair_rows=completed.pair_rows,
        directional_rows_by_group=completed.directional_rows_by_group,
        ledger=completed.ledger,
    )
    output["materialized_artifact_bindings"] = _build_materialized_artifact_bindings(
        output, isolation_root=isolation_root
    )
    publish_program_checkpoint(
        checkpoint_dir,
        output,
        expected_input_sha256=expected_input_sha256,
        checkpoint_state="complete",
        isolation_root=isolation_root,
    )
    return output


def result_from_program_payload(
    payload: Mapping[str, object], *, out_dir: Path
) -> OrchestrationResult:
    """Rehydrate a hash-validated checkpoint without invoking any runner."""

    _validate_program_checkpoint_payload(payload, required_cleanup_state="complete")
    pair_views = payload.get("pair_views")
    two_n = payload.get("two_n_results")
    profile_rows = payload.get("profile_rows")
    false_authorizations = payload.get("false_authorizations")
    stage_paths = payload.get("stage_paths")
    program = str(payload.get("program_id", ""))
    if (
        not program
        or not isinstance(profile_rows, list)
        or not isinstance(pair_views, Mapping)
        or not isinstance(two_n, Mapping)
        or not isinstance(false_authorizations, list)
        or not isinstance(stage_paths, Mapping)
    ):
        raise ValueError("program checkpoint result payload is malformed")
    return OrchestrationResult(
        out_dir=Path(out_dir).resolve(strict=False),
        study_manifest_id=str(payload.get("study_manifest_id", "")),
        profile_rows={program: tuple(dict(row) for row in profile_rows)},
        pair_views={
            group: tuple(dict(row) for row in pair_views[group]) for group in _GROUP_IDS
        },
        two_n_results={
            group: {
                key: tuple(dict(row) for row in two_n[group][key])
                for key in ("group_rows", "directional_rows", "pair_rows")
            }
            for group in _GROUP_IDS
        },
        false_authorizations=tuple(dict(row) for row in false_authorizations),
        stage_paths={str(key): str(value) for key, value in stage_paths.items()},
    )


def combine_program_results(
    results: Sequence[OrchestrationResult],
) -> OrchestrationResult:
    """Deterministically combine completed/limited one-program checkpoints."""

    if not results:
        raise ValueError("at least one per-program result is required")
    ordered: list[tuple[str, OrchestrationResult]] = []
    seen_programs: set[str] = set()
    manifest = results[0].study_manifest_id
    out_dir = results[0].out_dir.resolve(strict=False)
    for result in results:
        if result.study_manifest_id != manifest:
            raise ValueError("per-program results have different study manifests")
        if result.out_dir.resolve(strict=False) != out_dir:
            raise ValueError("per-program results have different output roots")
        if len(result.profile_rows) != 1:
            raise ValueError("each combined checkpoint must contain exactly one program")
        program = next(iter(result.profile_rows))
        if program in seen_programs:
            raise ValueError(f"duplicate per-program checkpoint: {program}")
        seen_programs.add(program)
        ordered.append((program, result))
    ordered.sort(key=lambda item: item[0])

    profiles: dict[str, tuple[dict[str, object], ...]] = {}
    pair_views: dict[str, list[dict[str, object]]] = {group: [] for group in _GROUP_IDS}
    two_n: dict[str, dict[str, list[dict[str, object]]]] = {
        group: {key: [] for key in ("group_rows", "directional_rows", "pair_rows")}
        for group in _GROUP_IDS
    }
    false_authorizations: list[dict[str, object]] = []
    stage_paths: dict[str, str] = {}
    for program, result in ordered:
        profiles[program] = tuple(dict(row) for row in result.profile_rows[program])
        for group in _GROUP_IDS:
            pair_views[group].extend(dict(row) for row in result.pair_views[group])
            for key in ("group_rows", "directional_rows", "pair_rows"):
                two_n[group][key].extend(
                    dict(row) for row in result.two_n_results[group][key]
                )
        false_authorizations.extend(dict(row) for row in result.false_authorizations)
        for key, value in result.stage_paths.items():
            text_key, text_value = str(key), str(value)
            if text_key in stage_paths and stage_paths[text_key] != text_value:
                raise ValueError(f"conflicting per-program stage path: {text_key}")
            stage_paths[text_key] = text_value
    return OrchestrationResult(
        out_dir=out_dir,
        study_manifest_id=manifest,
        profile_rows=profiles,
        pair_views={group: tuple(pair_views[group]) for group in _GROUP_IDS},
        two_n_results={
            group: {key: tuple(two_n[group][key]) for key in two_n[group]}
            for group in _GROUP_IDS
        },
        false_authorizations=tuple(
            sorted(false_authorizations, key=lambda row: str(row.get("case_id", "")))
        ),
        stage_paths=dict(sorted(stage_paths.items())),
    )
