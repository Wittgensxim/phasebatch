"""Manifest-bound prepare phase for the isolated expanded pair/2N study.

This module deliberately stops before pair execution and before the advisor 2N
evaluation.  It only freezes tool, program, action, and preflight identities;
all evidence it writes is report-only and keeps correctness authority off.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import yaml

from .manifest import (
    DEFAULT_SELECTION_SEED,
    FORMAL_PROGRAM_TARGET,
    FORMAL_SAMPLING_FRAME_SCHEMA_VERSION,
    FORMAL_SELECTION_RULE_ID,
    FORMAL_SOURCE_INVENTORY_COUNT,
    FORMAL_SOURCE_POSITIONS,
    FrozenProgramManifest,
    ProgramRecord,
    build_study_manifest,
    canonical_sha256,
    freeze_formal_program_manifest,
    freeze_program_manifest,
    normalize_program_id,
    normalize_program_relative_path,
    require_formal_program_boundary,
    require_study_manifest,
    stable_rank,
    validate_program_source_paths,
)
from .pass_universe import (
    ActionRecord,
    PassInventoryRow,
    build_nested_groups,
    join_preflight_results,
    load_frozen_policy,
    load_u14_actions,
    parse_function_pass_inventory,
    validate_u14_binding,
)
from .schema import (
    PASS_GROUP_FIELDS,
    PASS_INVENTORY_FIELDS,
    PASS_PREFLIGHT_FIELDS,
    PROGRAM_MANIFEST_FIELDS,
    canonical_row_id,
    write_csv,
)


_REQUIRED_TOOL_NAMES = ("opt", "clang", "worker", "merge_helper")
_ROOT_FILE_SUFFIX = ".ll"
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
_FORBIDDEN_PREPARE_ARTIFACT_PREFIXES = ("pair_", "advisor_2n_")
_CANDIDATE_EXCLUSION_SCHEMA = (
    "advisor-pair-scale-2n/candidate-identity-exclusions-v1"
)
_CANDIDATE_EXCLUSION_FIELDS = frozenset(
    {
        "program_id",
        "relative_path",
        "source_path",
        "source_sha256",
        "source_size_bytes",
        "stable_rank",
        "canonical_program_id",
        "canonical_relative_path",
        "canonical_source_path",
        "canonical_source_sha256",
        "canonical_stable_rank",
        "reason",
    }
)


@dataclass(frozen=True)
class RunResult:
    """Small protocol result used by injected prepare-time command runners."""

    success: bool
    output_path: Path
    hard_state_id: str = ""
    timed_out: bool = False
    stderr: str = ""
    command: tuple[str, ...] = ()


@dataclass(frozen=True)
class PrepareDependencies:
    """The complete side-effect boundary for the prepare phase.

    Production wiring is intentionally deferred to the standalone CLI.  Tests
    and future CLI code inject the existing Worker/compiler operations through
    these callables without importing any authority-bearing Phasebatch module.
    """

    compile_source: Callable[[ProgramRecord, Path], RunResult]
    print_passes: Callable[[], str]
    run_single: Callable[[Path, ActionRecord, Path], RunResult]
    verify_ir: Callable[[Path], bool]
    tool_records: Mapping[str, Mapping[str, object]]
    fixed_programs: Sequence[ProgramRecord]
    candidate_programs: Sequence[ProgramRecord]
    single_source_root: Path
    llvm_commit: str
    target: str
    hard_state_policy: Mapping[str, object]
    comparator: Mapping[str, object]
    artifact_policy: Mapping[str, object]
    candidate_identity_exclusions: Sequence[Mapping[str, object]] = ()


@dataclass(frozen=True)
class PrepareResult:
    """Immutable completion summary; it intentionally contains no experiment rows."""

    out_dir: Path
    study_manifest_path: Path
    prepare_complete_path: Path
    program_count: int
    group_sizes: dict[str, int]
    scale_gate: str
    study_manifest_id: str


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _safe_component(value: str) -> str:
    text = str(value).strip()
    if not text or text in {".", ".."} or any(mark in text for mark in ("/", "\\")):
        raise ValueError(f"unsafe output path component: {value!r}")
    return text


def _inside_output(out_dir: Path, path: Path) -> Path:
    root = out_dir.resolve()
    target = path.resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"prepare output must remain inside out_dir: {target}") from error
    return target


def _write_json(out_dir: Path, relative: str, value: object) -> Path:
    target = _inside_output(out_dir, out_dir / relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_canonical_json(value) + "\n", encoding="utf-8", newline="\n")
    return target


def _normalise_candidate_identity_exclusions(
    rows: Sequence[Mapping[str, object]],
    *,
    verify_sources: bool,
) -> tuple[dict[str, object], ...]:
    normalized: list[dict[str, object]] = []
    seen_relative: set[str] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or set(raw) != _CANDIDATE_EXCLUSION_FIELDS:
            raise ValueError(
                f"candidate identity exclusion {index} has invalid fields"
            )
        row = {str(key): value for key, value in raw.items()}
        try:
            program_id = normalize_program_id(
                row["program_id"], field="program_id"
            )
            canonical_program_id = normalize_program_id(
                row["canonical_program_id"], field="canonical_program_id"
            )
            if not isinstance(row["relative_path"], str) or not isinstance(
                row["canonical_relative_path"], str
            ):
                raise ValueError("relative paths must be strings")
            relative = normalize_program_relative_path(row["relative_path"])
            canonical_relative = normalize_program_relative_path(
                row["canonical_relative_path"]
            )
        except ValueError as error:
            raise ValueError(
                f"candidate identity exclusion {index} identity is invalid: {error}"
            ) from error
        if (
            row["relative_path"] != relative
            or row["canonical_relative_path"] != canonical_relative
            or relative == canonical_relative
            or relative in seen_relative
        ):
            raise ValueError("candidate identity exclusions contain invalid paths")
        row["program_id"] = program_id
        row["canonical_program_id"] = canonical_program_id
        row["relative_path"] = relative
        row["canonical_relative_path"] = canonical_relative
        seen_relative.add(relative)
        if not isinstance(row["source_path"], str) or not isinstance(
            row["canonical_source_path"], str
        ):
            raise ValueError(
                "candidate identity exclusion source paths must be canonical strings"
            )
        source = Path(row["source_path"])
        canonical_source = Path(row["canonical_source_path"])
        if (
            not source.is_absolute()
            or not canonical_source.is_absolute()
            or row["source_path"] != str(source.resolve())
            or row["canonical_source_path"] != str(canonical_source.resolve())
            or source.resolve() == canonical_source.resolve()
        ):
            raise ValueError("candidate identity exclusion source paths must be absolute")
        if not isinstance(row["source_sha256"], str) or not isinstance(
            row["canonical_source_sha256"], str
        ):
            raise ValueError("candidate identity exclusion source SHA mismatch")
        source_sha = row["source_sha256"]
        canonical_sha = row["canonical_source_sha256"]
        if (
            len(source_sha) != 64
            or any(character not in "0123456789abcdef" for character in source_sha)
            or source_sha != canonical_sha
        ):
            raise ValueError("candidate identity exclusion source SHA mismatch")
        size = row["source_size_bytes"]
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("candidate identity exclusion source size is invalid")
        if row["reason"] != "duplicate_source_sha256":
            raise ValueError("candidate identity exclusion reason is invalid")
        if not isinstance(row["stable_rank"], str) or row[
            "stable_rank"
        ] != stable_rank(DEFAULT_SELECTION_SEED, relative):
            raise ValueError("candidate identity exclusion stable rank mismatch")
        if not isinstance(row["canonical_stable_rank"], str) or row[
            "canonical_stable_rank"
        ] != stable_rank(
            DEFAULT_SELECTION_SEED, canonical_relative
        ):
            raise ValueError("candidate identity exclusion canonical rank mismatch")
        if verify_sources:
            if not source.is_file() or not canonical_source.is_file():
                raise ValueError("candidate identity exclusion source is missing")
            if source.stat().st_size != size or _sha256_file(source) != source_sha:
                raise ValueError("candidate identity exclusion source identity drift")
            if (
                canonical_source.stat().st_size != size
                or _sha256_file(canonical_source) != canonical_sha
            ):
                raise ValueError("candidate identity exclusion representative drift")
        normalized.append(row)
    return tuple(
        sorted(
            normalized,
            key=lambda row: (str(row["stable_rank"]), str(row["relative_path"])),
        )
    )


def _candidate_identity_exclusion_document(
    rows: Sequence[Mapping[str, object]],
    *,
    candidate_inventory_count: int,
    candidate_reserve_count: int,
    verify_sources: bool,
) -> dict[str, object]:
    normalized = _normalise_candidate_identity_exclusions(
        rows, verify_sources=verify_sources
    )
    if (
        candidate_inventory_count < candidate_reserve_count
        or candidate_inventory_count - candidate_reserve_count != len(normalized)
    ):
        raise ValueError("candidate identity exclusion partition mismatch")
    payload: dict[str, object] = {
        "schema_version": _CANDIDATE_EXCLUSION_SCHEMA,
        "candidate_inventory_count": candidate_inventory_count,
        "candidate_reserve_count": candidate_reserve_count,
        "exclusion_count": len(normalized),
        "exclusions": list(normalized),
        "exclusions_sha256": canonical_sha256(list(normalized)),
    }
    payload["document_sha256"] = canonical_sha256(payload)
    return payload


def _validate_candidate_identity_exclusion_document(
    value: object,
    *,
    verify_sources: bool,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("candidate identity exclusions must be a mapping")
    expected = {
        "schema_version",
        "candidate_inventory_count",
        "candidate_reserve_count",
        "exclusion_count",
        "exclusions",
        "exclusions_sha256",
        "document_sha256",
    }
    if set(value) != expected or value.get("schema_version") != _CANDIDATE_EXCLUSION_SCHEMA:
        raise ValueError("candidate identity exclusions schema mismatch")
    unsigned = {key: item for key, item in value.items() if key != "document_sha256"}
    if value.get("document_sha256") != canonical_sha256(unsigned):
        raise ValueError("candidate identity exclusions document self-hash mismatch")
    exclusions = value.get("exclusions")
    if not isinstance(exclusions, list):
        raise ValueError("candidate identity exclusions rows are missing")
    try:
        inventory_count = int(value.get("candidate_inventory_count", -1))
        reserve_count = int(value.get("candidate_reserve_count", -1))
    except (TypeError, ValueError) as error:
        raise ValueError("candidate identity exclusion counts are invalid") from error
    rebuilt = _candidate_identity_exclusion_document(
        exclusions,
        candidate_inventory_count=inventory_count,
        candidate_reserve_count=reserve_count,
        verify_sources=verify_sources,
    )
    if rebuilt != dict(value):
        raise ValueError("candidate identity exclusions are non-canonical")
    return rebuilt


def _result_value(result: object, field: str, default: object) -> object:
    if isinstance(result, Mapping):
        return result.get(field, default)
    return getattr(result, field, default)


def _require_successful_output(
    result: object,
    requested_output: Path,
    out_dir: Path,
    *,
    operation: str,
) -> tuple[Path, str]:
    """Validate a callback result without allowing it to redirect output I/O."""

    if bool(_result_value(result, "timed_out", False)):
        raise ValueError(f"{operation} timed out")
    if not bool(_result_value(result, "success", False)):
        raise ValueError(f"{operation} failed")
    returned = Path(_result_value(result, "output_path", requested_output))
    requested = _inside_output(out_dir, requested_output)
    returned = _inside_output(out_dir, returned)
    if returned != requested:
        raise ValueError(f"{operation} returned an unexpected output path")
    if not requested.is_file():
        raise ValueError(f"{operation} missing root IR/output: {requested}")
    digest = str(_result_value(result, "hard_state_id", "")).strip() or _sha256_file(requested)
    return requested, digest


def _validate_tool_records(
    records: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    if set(records) != set(_REQUIRED_TOOL_NAMES):
        missing = sorted(set(_REQUIRED_TOOL_NAMES) - set(records))
        unexpected = sorted(set(records) - set(_REQUIRED_TOOL_NAMES))
        raise ValueError(f"tool records mismatch: missing={missing}, unexpected={unexpected}")
    normalized: dict[str, dict[str, object]] = {}
    for name in _REQUIRED_TOOL_NAMES:
        record = records[name]
        if not isinstance(record, Mapping):
            raise ValueError(f"{name} tool record must be a mapping")
        path = Path(str(record.get("path", ""))).resolve()
        expected_hash = str(record.get("sha256", "")).strip()
        if not path.is_file():
            raise ValueError(f"{name} tool is missing: {path}")
        actual_hash = _sha256_file(path)
        if len(expected_hash) != 64 or actual_hash != expected_hash:
            raise ValueError(f"{name} hash mismatch")
        normalized[name] = {
            "path": path.as_posix(),
            "sha256": actual_hash,
            "size_bytes": path.stat().st_size,
        }
    return normalized


def _program_record_from_manifest(row: Mapping[str, object]) -> ProgramRecord:
    """Parse one exact program row; no partial/alternate identity is accepted."""

    expected_fields = {
        "program_id",
        "source_path",
        "relative_path",
        "program_family",
        "source_sha256",
        "source_size_bytes",
        "compile_command",
        "compile_command_sha256",
        "compile_status",
        "compile_stderr_sha256",
        "root_ir_path",
        "root_ir_sha256",
        "root_hard_state_id",
        "target",
        "data_layout",
        "preflight_status",
        "selection_class",
        "selection_order",
        "reserve_rank",
        "replacement_for_program_id",
    }
    if set(row) != expected_fields or not isinstance(row.get("compile_command"), list):
        raise ValueError("program record shape is not canonical")
    record = ProgramRecord(
        program_id=str(row["program_id"]),
        source_path=str(row["source_path"]),
        relative_path=str(row["relative_path"]),
        program_family=str(row["program_family"]),
        source_sha256=str(row["source_sha256"]),
        source_size_bytes=row["source_size_bytes"],
        compile_command=tuple(str(item) for item in row["compile_command"]),
        compile_status=str(row["compile_status"]),
        compile_stderr_sha256=str(row["compile_stderr_sha256"]),
        root_ir_path=str(row["root_ir_path"]),
        root_ir_sha256=str(row["root_ir_sha256"]),
        root_hard_state_id=str(row["root_hard_state_id"]),
        target=str(row["target"]),
        data_layout=str(row["data_layout"]),
        preflight_status=str(row["preflight_status"]),
        selection_class=str(row["selection_class"]),
        selection_order=row["selection_order"],
        reserve_rank=row["reserve_rank"],
        replacement_for_program_id=str(row["replacement_for_program_id"]),
    )
    if record.as_manifest_record() != dict(row):
        raise ValueError("program record is not canonical")
    return record


def _load_reference_manifest(
    path: Path,
) -> tuple[tuple[Mapping[str, object], ...], tuple[ProgramRecord, ...]]:
    """Load exact root-only U14 and fixed-50 identities from one manifest."""

    if not path.is_file():
        raise FileNotFoundError(f"existing 50-program manifest is missing: {path}")
    raw = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = yaml.safe_load(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("existing 50-program manifest must be a mapping")
    pass_config = payload.get("pass_config")
    if isinstance(pass_config, Mapping):
        actions = pass_config.get("actions")
    else:
        actions = payload.get("actions")
    if not isinstance(actions, list):
        raise ValueError("existing 50-program manifest is missing pass_config.actions")
    if not all(isinstance(action, Mapping) for action in actions):
        raise ValueError("existing U14 actions must be mappings")
    program_rows = payload.get("program_manifest", payload.get("programs"))
    if not isinstance(program_rows, list) or len(program_rows) != 50:
        raise ValueError("fixed 50 program identity mismatch: program_manifest requires 50 rows")
    try:
        programs = tuple(
            _program_record_from_manifest(row)
            for row in program_rows
            if isinstance(row, Mapping)
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"fixed 50 program identity mismatch: {error}") from error
    if len(programs) != 50:
        raise ValueError("fixed 50 program identity mismatch: invalid program record")
    return tuple(dict(action) for action in actions), programs


def _validate_root_hashes(programs: Sequence[ProgramRecord]) -> None:
    for program in programs:
        root = Path(program.root_ir_path)
        if not root.is_file() or not program.root_ir_sha256:
            raise ValueError(f"fixed 50 program identity mismatch: missing root IR for {program.program_id}")
        if _sha256_file(root) != program.root_ir_sha256:
            raise ValueError(f"fixed 50 program identity mismatch: root hash drift for {program.program_id}")


def _validate_formal_source_inventory_identity(
    reference: Sequence[ProgramRecord],
    fixed: Sequence[ProgramRecord],
    single_source_root: Path,
) -> None:
    if len(reference) != 50 or len(fixed) != 50:
        raise ValueError("fixed 50 program identity mismatch: expected exactly 50 programs")
    try:
        validate_program_source_paths(reference, single_source_root)
        _validate_root_hashes(reference)
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(f"fixed 50 program identity mismatch: {error}") from error
    reference_rows = [program.as_manifest_record() for program in reference]
    fixed_rows = [program.as_manifest_record() for program in fixed]
    if reference_rows != fixed_rows:
        raise ValueError("fixed 50 program identity mismatch")


def _freeze_roots(
    programs: Sequence[ProgramRecord],
    dependencies: PrepareDependencies,
    out_dir: Path,
) -> tuple[ProgramRecord, ...]:
    frozen: list[ProgramRecord] = []
    for record in programs:
        program_id = _safe_component(record.program_id)
        requested = _inside_output(out_dir, out_dir / "roots" / f"{program_id}{_ROOT_FILE_SUFFIX}")
        requested.parent.mkdir(parents=True, exist_ok=True)
        result = dependencies.compile_source(record, requested)
        output, hard_state_id = _require_successful_output(
            result, requested, out_dir, operation=f"compile root IR for {record.program_id}"
        )
        if not dependencies.verify_ir(output):
            raise ValueError(f"root IR verifier failed: {record.program_id}")
        frozen.append(
            replace(
                record,
                root_ir_path=str(output),
                root_ir_sha256=_sha256_file(output),
                root_hard_state_id=hard_state_id,
                compile_status="success",
                preflight_status="success",
            )
        )
    return tuple(frozen)


def _rebase_artifact_path(value: str | Path, staging_dir: Path, final_out_dir: Path) -> str:
    """Describe staged artifacts by their deterministic eventual published path."""

    staged_path = Path(value).resolve()
    try:
        relative = staged_path.relative_to(staging_dir.resolve())
    except ValueError as error:
        raise ValueError(f"staged artifact escapes prepare staging directory: {staged_path}") from error
    return str((final_out_dir / relative).resolve())


def _rebase_program_roots(
    programs: Sequence[ProgramRecord], staging_dir: Path, final_out_dir: Path
) -> tuple[ProgramRecord, ...]:
    return tuple(
        replace(
            program,
            root_ir_path=_rebase_artifact_path(
                program.root_ir_path, staging_dir, final_out_dir
            ),
        )
        for program in programs
    )


def _rebase_preflight_rows(
    rows: Sequence[Mapping[str, object]], staging_dir: Path, final_out_dir: Path
) -> list[dict[str, object]]:
    rebased: list[dict[str, object]] = []
    for raw in rows:
        row = dict(raw)
        row["output_path"] = _rebase_artifact_path(
            str(row["output_path"]), staging_dir, final_out_dir
        )
        rebased.append(row)
    return rebased


def _actions_for_inventory(
    core_actions: Sequence[ActionRecord], inventory: Sequence[PassInventoryRow]
) -> tuple[ActionRecord, ...]:
    core_by_pipeline = {action.pipeline: action for action in core_actions}
    candidates: list[ActionRecord] = list(core_actions)
    for index, row in enumerate(inventory, start=len(core_actions)):
        if not row.policy_candidate or row.pipeline in core_by_pipeline:
            continue
        candidates.append(
            ActionRecord.for_function_candidate(
                name=row.name,
                pipeline=row.pipeline,
                config_index=index,
            )
        )
    if len({action.action_id for action in candidates}) != len(candidates):
        raise ValueError("duplicate action IDs in preflight candidate universe")
    return tuple(candidates)


def _run_preflight(
    *,
    actions: Sequence[ActionRecord],
    programs: Sequence[ProgramRecord],
    policy: Mapping[str, object],
    dependencies: PrepareDependencies,
    out_dir: Path,
    preflight_program_ids: Sequence[str] | None = None,
) -> tuple[list[dict[str, object]], dict[str, tuple[str, ...]]]:
    preflight_ids = (
        tuple(str(item) for item in preflight_program_ids)
        if preflight_program_ids is not None
        else tuple(str(item) for item in policy["preflight_programs"])
    )
    if not preflight_ids or len(preflight_ids) != len(set(preflight_ids)):
        raise ValueError("preflight program IDs must be a non-empty unique sequence")
    program_by_id = {program.program_id: program for program in programs}
    missing_programs = sorted(set(preflight_ids) - set(program_by_id))
    if missing_programs:
        raise ValueError(f"frozen preflight programs missing from formal corpus: {missing_programs}")
    raw_rows: list[dict[str, object]] = []
    for action in actions:
        for program_id in preflight_ids:
            program = program_by_id[program_id]
            root_ir = Path(program.root_ir_path)
            if not root_ir.is_file():
                raise ValueError(f"missing root IR: {program_id}")
            for repetition in (1, 2):
                requested = _inside_output(
                    out_dir,
                    out_dir
                    / "preflight"
                    / action.action_id
                    / _safe_component(program_id)
                    / f"repeat-{repetition}.ll",
                )
                requested.parent.mkdir(parents=True, exist_ok=True)
                result = dependencies.run_single(root_ir, action, requested)
                timed_out = bool(_result_value(result, "timed_out", False))
                successful = bool(_result_value(result, "success", False))
                output = Path(_result_value(result, "output_path", requested))
                output_sha256 = ""
                hard_state_id = ""
                verifier_status = "not_run"
                if timed_out:
                    execution_status = "timeout"
                elif not successful:
                    execution_status = "error"
                else:
                    try:
                        materialized, hard_state_id = _require_successful_output(
                            result,
                            requested,
                            out_dir,
                            operation=f"preflight {action.name}/{program_id}/{repetition}",
                        )
                    except ValueError:
                        execution_status = "error"
                    else:
                        output_sha256 = _sha256_file(materialized)
                        verifier_status = (
                            "success" if dependencies.verify_ir(materialized) else "invalid"
                        )
                        execution_status = "success" if verifier_status == "success" else "invalid"
                raw_rows.append(
                    {
                        "program_id": program_id,
                        "action_id": action.action_id,
                        "repetition": repetition,
                        "execution_status": execution_status,
                        "verifier_status": verifier_status,
                        "output_hard_state_id": hard_state_id,
                        "output_sha256": output_sha256,
                        "output_path": str(output),
                        "command": list(_result_value(result, "command", ())),
                        "stderr": str(_result_value(result, "stderr", "")),
                    }
                )
    decisions = join_preflight_results(
        actions,
        raw_rows,
        preflight_ids,
        repeats=int(policy["preflight_repeats"]),
    )
    reasons_by_action = {
        decision.action_id: decision.rejection_reasons for decision in decisions
    }
    eligible_by_action = {decision.action_id: decision.eligible for decision in decisions}
    for row in raw_rows:
        reasons = reasons_by_action[row["action_id"]]
        row["eligible"] = eligible_by_action[row["action_id"]]
        row["deterministic"] = "true" if eligible_by_action[row["action_id"]] else "false"
        row["exclusion_reason"] = ";".join(reasons)
    return raw_rows, reasons_by_action


def _inventory_records(
    inventory: Sequence[PassInventoryRow], actions: Sequence[ActionRecord]
) -> list[dict[str, object]]:
    action_by_pipeline = {action.pipeline: action for action in actions}
    rows: list[dict[str, object]] = []
    for item in inventory:
        action = action_by_pipeline.get(item.pipeline)
        rows.append(
            {
                "name": item.name,
                "pipeline": item.pipeline,
                "registry_section": item.registry_section,
                "policy_candidate": item.policy_candidate,
                "policy_reason": item.policy_reason,
                "action": action.as_manifest_record() if action is not None else None,
            }
        )
    return rows


def _group_records(
    groups: Mapping[str, Sequence[str]], *, seed: str
) -> dict[str, object]:
    rows: dict[str, object] = {}
    for group_id in ("U14", "U30", "Uall"):
        action_ids = tuple(groups[group_id])
        rows[group_id] = {
            "group_id": group_id,
            "group_sha256": canonical_sha256(list(action_ids)),
            "group_size": len(action_ids),
            "action_ids": list(action_ids),
            "selection_seed": seed,
            "selection_method": (
                "exact_u14" if group_id == "U14" else "frozen_sha256_rank"
            ),
        }
    return rows


def _base_evidence(row_id: str, study_manifest_id: str) -> dict[str, object]:
    return {
        "row_id": row_id,
        "study_manifest_id": study_manifest_id,
        "authority_granted": False,
        "proved_commute": False,
    }


def _write_prepare_csvs(
    *,
    out_dir: Path,
    study_manifest_id: str,
    programs: Sequence[ProgramRecord],
    inventory: Sequence[PassInventoryRow],
    actions: Sequence[ActionRecord],
    preflight_rows: Sequence[Mapping[str, object]],
    groups: Mapping[str, Sequence[str]],
    seed: str,
) -> None:
    program_rows: list[dict[str, object]] = []
    for program in programs:
        row = _base_evidence(canonical_row_id("program", program.program_id), study_manifest_id)
        row.update(
            {
                "program_id": program.program_id,
                "selection_order": program.selection_order or "",
                "selection_class": program.selection_class,
                "source_path": program.source_path,
                "relative_path": program.relative_path,
                "program_family": program.program_family,
                "source_sha256": program.source_sha256,
                "source_size_bytes": program.source_size_bytes,
                "compile_command": _canonical_json(list(program.compile_command)),
                "compile_command_sha256": program.compile_command_sha256,
                "compile_status": program.compile_status,
                "compile_stderr_sha256": program.compile_stderr_sha256,
                "root_ir_path": program.root_ir_path,
                "root_ir_sha256": program.root_ir_sha256,
                "root_hard_state_id": program.root_hard_state_id,
                "target": program.target,
                "data_layout": program.data_layout,
                "reserve_rank": program.reserve_rank or "",
                "replacement_for_program_id": program.replacement_for_program_id,
            }
        )
        program_rows.append(row)
    write_csv(out_dir / "program_manifest.csv", PROGRAM_MANIFEST_FIELDS, program_rows, isolation_root=out_dir)

    action_by_pipeline = {action.pipeline: action for action in actions}
    inventory_rows: list[dict[str, object]] = []
    for item in inventory:
        action = action_by_pipeline.get(item.pipeline)
        row = _base_evidence(
            canonical_row_id("inventory", item.name, item.pipeline), study_manifest_id
        )
        row.update(
            {
                "action_id": action.action_id if action is not None else "",
                "action_sha256": action.action_id if action is not None else "",
                "canonical_action_record": action.canonical_json if action is not None else "",
                "name": item.name,
                "pipeline": item.pipeline,
                "registry_section": item.registry_section,
                "parameter_binding": item.pipeline,
                "policy_candidate": str(item.policy_candidate).lower(),
                "policy_reason": item.policy_reason,
            }
        )
        inventory_rows.append(row)
    write_csv(out_dir / "pass_inventory.csv", PASS_INVENTORY_FIELDS, inventory_rows, isolation_root=out_dir)

    preflight_csv_rows: list[dict[str, object]] = []
    for raw in preflight_rows:
        row = _base_evidence(
            canonical_row_id(
                "preflight", raw["action_id"], raw["program_id"], raw["repetition"]
            ),
            study_manifest_id,
        )
        command = _canonical_json(list(raw["command"]))
        stderr = str(raw["stderr"])
        row.update(
            {
                "program_id": raw["program_id"],
                "action_id": raw["action_id"],
                "repetition": raw["repetition"],
                "execution_status": raw["execution_status"],
                "verifier_status": raw["verifier_status"],
                "output_hard_state_id": raw["output_hard_state_id"],
                "output_sha256": raw["output_sha256"],
                "deterministic": raw["deterministic"],
                "eligible": str(raw["eligible"]).lower(),
                "exclusion_reason": raw["exclusion_reason"],
                "logical_pass_applications": 1,
                "physical_pass_invocations": 1,
                "artifact_id": "",
                "command_sha256": _sha256_bytes(command.encode("utf-8")),
                "stderr_sha256": _sha256_bytes(stderr.encode("utf-8")),
            }
        )
        preflight_csv_rows.append(row)
    write_csv(out_dir / "pass_preflight.csv", PASS_PREFLIGHT_FIELDS, preflight_csv_rows, isolation_root=out_dir)

    group_csv_rows: list[dict[str, object]] = []
    for group_id in ("U14", "U30", "Uall"):
        action_ids = tuple(groups[group_id])
        group_hash = canonical_sha256(list(action_ids))
        for index, action_id in enumerate(action_ids, start=1):
            row = _base_evidence(canonical_row_id("group", group_id, action_id), study_manifest_id)
            row.update(
                {
                    "group_id": group_id,
                    "group_sha256": group_hash,
                    "group_size": len(action_ids),
                    "action_id": action_id,
                    "action_order": index,
                    "selection_method": "exact_u14" if group_id == "U14" else "frozen_sha256_rank",
                    "selection_rank": index if group_id != "U14" else "",
                    "selection_seed": seed if group_id != "U14" else "",
                }
            )
            group_csv_rows.append(row)
    write_csv(out_dir / "pass_groups.csv", PASS_GROUP_FIELDS, group_csv_rows, isolation_root=out_dir)


def _write_program_json(
    out_dir: Path, frozen: FrozenProgramManifest, programs: Sequence[ProgramRecord]
) -> Path:
    selected_by_source = {program.source_sha256: program for program in programs}
    reserve_records: list[dict[str, object]] = []
    for reserve in frozen.reserve_order:
        selected = selected_by_source.get(reserve.source_sha256)
        row = (
            selected.as_manifest_record()
            if selected is not None
            else reserve.as_manifest_record()
        )
        row["reserve_rank"] = reserve.reserve_rank
        row["root_ir_materialized"] = selected is not None
        if selected is None:
            # Compile-preflight bytes live only in the private prepare
            # workspace.  Preserve their hashes/status/command, but never
            # publish a dangling temporary path as a durable artifact.
            row["root_ir_path"] = ""
        reserve_records.append(row)
    return _write_json(
        out_dir,
        "program_manifest.json",
        {
            "programs": [program.as_manifest_record() for program in programs],
            "reserve_order": reserve_records,
            "preflight_ledger": [entry.__dict__ for entry in frozen.preflight_ledger],
            "target": frozen.target,
            "selection_seed": frozen.selection_seed,
            "per_category_cap": frozen.per_category_cap,
            "max_source_bytes": frozen.max_source_bytes,
        },
    )


def _self_hashed_document(value: Mapping[str, object]) -> dict[str, object]:
    document = dict(value)
    document["document_sha256"] = canonical_sha256(document)
    return document


def _formal_sampling_frame_document(
    source_inventory: Sequence[ProgramRecord],
    selected_programs: Sequence[ProgramRecord],
) -> dict[str, object]:
    require_formal_program_boundary(
        source_inventory,
        target=FORMAL_PROGRAM_TARGET,
    )
    if len(selected_programs) != FORMAL_PROGRAM_TARGET:
        raise ValueError("formal sampling frame requires exactly 10 selected programs")
    ordered_source = sorted(
        source_inventory,
        key=lambda row: row.selection_order or 0,
    )
    source_rows = [row.as_manifest_record() for row in ordered_source]
    selected_rows = [row.as_manifest_record() for row in selected_programs]
    return _self_hashed_document(
        {
            "schema_version": FORMAL_SAMPLING_FRAME_SCHEMA_VERSION,
            "source_inventory_count": FORMAL_SOURCE_INVENTORY_COUNT,
            "selection_rule_id": FORMAL_SELECTION_RULE_ID,
            "source_positions": list(FORMAL_SOURCE_POSITIONS),
            "source_programs": source_rows,
            "source_programs_sha256": canonical_sha256(source_rows),
            "selected_programs": selected_rows,
            "selected_programs_sha256": canonical_sha256(selected_rows),
        }
    )


def _completion_hashes(out_dir: Path) -> dict[str, str]:
    return {
        path.relative_to(out_dir).as_posix(): _sha256_file(path)
        for path in sorted(out_dir.rglob("*"))
        if path.is_file() and path.name != "prepare_complete.json"
    }


def _output_target(out_dir: Path) -> Path:
    """Accept only the smoke/formal roots or their subtrees of this experiment."""

    target = Path(out_dir).resolve()
    experiment_root = EXPERIMENT_ROOT.resolve()
    try:
        relative = target.relative_to(experiment_root / "output")
    except ValueError as error:
        raise ValueError(
            "out_dir must be inside the isolated experiment output/(smoke|formal) subtree"
        ) from error
    if not relative.parts or relative.parts[0] not in {"smoke", "formal"}:
        raise ValueError(
            "out_dir must be inside the isolated experiment output/(smoke|formal) subtree"
        )
    return target


def _tree_hashes(out_dir: Path) -> dict[str, str]:
    return {
        path.relative_to(out_dir).as_posix(): _sha256_file(path)
        for path in sorted(out_dir.rglob("*"))
        if path.is_file()
    }


def _validate_existing_prepare_state(out_dir: Path) -> None:
    """Reject user/experimental content unless it is a self-hashed prepare tree."""

    forbidden = sorted(
        path.relative_to(out_dir).as_posix()
        for path in out_dir.rglob("*")
        if path.is_file()
        and path.name.startswith(_FORBIDDEN_PREPARE_ARTIFACT_PREFIXES)
    )
    if forbidden:
        raise ValueError(f"existing out_dir contains forbidden pair/2N artifact: {forbidden[0]}")
    completion_path = out_dir / "prepare_complete.json"
    manifest_path = out_dir / "study_manifest.json"
    if not completion_path.is_file() or not manifest_path.is_file():
        raise ValueError("existing out_dir is not a hash-validated complete prepare state")
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("existing out_dir is not a hash-validated complete prepare state") from error
    if not isinstance(completion, Mapping) or not isinstance(manifest, Mapping):
        raise ValueError("existing out_dir is not a hash-validated complete prepare state")
    completion_document_sha256 = completion.get("document_sha256")
    completion_unsigned = {
        str(key): value
        for key, value in completion.items()
        if key != "document_sha256"
    }
    if (
        not isinstance(completion_document_sha256, str)
        or canonical_sha256(completion_unsigned) != completion_document_sha256
    ):
        raise ValueError("existing out_dir is not a hash-validated complete prepare state")
    expected_hashes = completion.get("files_sha256")
    if not isinstance(expected_hashes, Mapping):
        raise ValueError("existing out_dir is not a hash-validated complete prepare state")
    actual_without_completion = {
        relative: digest
        for relative, digest in _tree_hashes(out_dir).items()
        if relative != "prepare_complete.json"
    }
    normalized_expected = {str(name): str(digest) for name, digest in expected_hashes.items()}
    if normalized_expected != actual_without_completion:
        raise ValueError("existing out_dir is not a hash-validated complete prepare state")
    if completion.get("study_manifest_id") != manifest.get("study_manifest_id"):
        raise ValueError("existing out_dir is not a hash-validated complete prepare state")
    if completion.get("authority_granted") is not False or completion.get("proved_commute") is not False:
        raise ValueError("existing out_dir is not a hash-validated complete prepare state")
    try:
        require_study_manifest(manifest, manifest)
    except (OSError, ValueError, TypeError) as error:
        raise ValueError("existing out_dir is not a hash-validated complete prepare state") from error


def _staging_dir(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{target.name}.prepare-staging-", dir=target.parent))


def _prepare_study_into(
    *,
    repo_root: Path,
    out_dir: Path,
    published_out_dir: Path,
    existing_50_manifest: Path,
    pass_policy: Path,
    dependencies: PrepareDependencies,
    core_passes: Path | None = None,
    program_target: int = FORMAL_PROGRAM_TARGET,
    jobs: int = 1,
    timeout_s: int | float = 1,
) -> PrepareResult:
    """Freeze the approved formal-10 midpoint sample before evidence.

    Every write is rooted at ``out_dir``.  Tool, LLVM, source, action, root,
    and repeat-preflight mismatches fail before a pair/2N CSV can exist.
    """

    del repo_root  # Kept in the public boundary for CLI provenance and tests.
    output = Path(out_dir).resolve()
    policy = load_frozen_policy(pass_policy)
    if dependencies.llvm_commit != policy["llvm_commit"]:
        raise ValueError("LLVM commit mismatch")
    tools = _validate_tool_records(dependencies.tool_records)
    if dependencies.target != "x86_64-w64-windows-gnu":
        raise ValueError("target mismatch")
    try:
        output_kind = published_out_dir.resolve().relative_to(
            EXPERIMENT_ROOT.resolve() / "output"
        ).parts[0]
    except (ValueError, IndexError) as error:
        raise ValueError("published output must be inside isolated output/(smoke|formal)") from error
    expected_target = 3 if output_kind == "smoke" else FORMAL_PROGRAM_TARGET
    if output_kind not in {"smoke", "formal"} or program_target != expected_target:
        raise ValueError(
            "study program target must be exactly 3 (smoke) or 10 (formal)"
        )
    if type(jobs) is not int or isinstance(jobs, bool) or jobs < 1:
        raise ValueError("jobs must be a positive integer")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    fixed = tuple(dependencies.fixed_programs)
    candidates = tuple(dependencies.candidate_programs)
    if output_kind == "formal" and candidates:
        raise ValueError("formal midpoint scope forbids candidate programs")
    if output_kind == "formal" and dependencies.candidate_identity_exclusions:
        raise ValueError("formal midpoint scope forbids candidate exclusions")
    if any(program.target != dependencies.target for program in (*fixed, *candidates)):
        raise ValueError("program target mismatch")
    expected_u14, reference_programs = _load_reference_manifest(Path(existing_50_manifest))
    if program_target == FORMAL_PROGRAM_TARGET:
        _validate_formal_source_inventory_identity(
            reference_programs,
            fixed,
            dependencies.single_source_root,
        )
    elif len(fixed) != 3:
        raise ValueError("smoke study requires exactly 3 deterministic fixed programs")
    # The scan may contain exact inventory copies of the fixed 50.  Validate
    # each source universe independently, then let Task 3's formal guard
    # accept only its explicitly defined copy relation.
    validate_program_source_paths(fixed, dependencies.single_source_root)
    validate_program_source_paths(candidates, dependencies.single_source_root)
    frozen = (
        freeze_formal_program_manifest(fixed, candidates)
        if program_target == FORMAL_PROGRAM_TARGET
        else freeze_program_manifest(fixed, (), target=3)
    )
    candidate_inventory_count = dependencies.artifact_policy.get(
        "candidate_inventory_count",
        len(frozen.reserve_order) + len(dependencies.candidate_identity_exclusions),
    )
    if (
        not isinstance(candidate_inventory_count, int)
        or isinstance(candidate_inventory_count, bool)
    ):
        raise ValueError("candidate inventory accounting mismatch")
    exclusion_document = _candidate_identity_exclusion_document(
        dependencies.candidate_identity_exclusions,
        candidate_inventory_count=candidate_inventory_count,
        candidate_reserve_count=len(frozen.reserve_order),
        verify_sources=True,
    )
    candidate_identity_exclusion_count = len(
        exclusion_document["exclusions"]
    )
    roots = _freeze_roots(frozen.programs, dependencies, output)
    if len(roots) != program_target:
        label = "formal" if program_target == FORMAL_PROGRAM_TARGET else "smoke"
        raise ValueError(f"{label} study requires exactly {program_target} frozen roots")

    # The complete candidate/reserve order is durable inside the atomic
    # prepare staging tree before the first pass invocation is allowed.  A
    # prepare failure never publishes the staging tree, but no pair/pass
    # outcome can influence this already-written selection.
    manifest_roots = _rebase_program_roots(roots, output, published_out_dir)
    _write_program_json(output, frozen, manifest_roots)
    formal_sampling_frame_sha256 = ""
    if output_kind == "formal":
        sampling_frame_path = _write_json(
            output,
            "formal_sampling_frame.json",
            _formal_sampling_frame_document(fixed, manifest_roots),
        )
        formal_sampling_frame_sha256 = _sha256_file(sampling_frame_path)
    exclusion_path = _write_json(
        output,
        "candidate_identity_exclusions.json",
        exclusion_document,
    )

    inventory = parse_function_pass_inventory(dependencies.print_passes(), policy)
    selected_core_path = Path(core_passes) if core_passes is not None else Path("configs/core_passes_v1.yaml")
    core_actions = load_u14_actions(selected_core_path)
    validate_u14_binding(core_actions, inventory, expected_u14)
    actions = _actions_for_inventory(core_actions, inventory)
    preflight_rows, reasons_by_action = _run_preflight(
        actions=actions,
        programs=roots,
        policy=policy,
        dependencies=dependencies,
        out_dir=output,
        preflight_program_ids=tuple(program.program_id for program in roots),
    )
    manifest_preflight_rows = _rebase_preflight_rows(
        preflight_rows, output, published_out_dir
    )
    core_ids = {action.action_id for action in core_actions}
    failed_u14 = {
        action_id: reasons_by_action[action_id]
        for action_id in core_ids
        if reasons_by_action[action_id]
    }
    if failed_u14:
        detail = "; ".join(
            f"{action_id}:{','.join(reasons)}"
            for action_id, reasons in sorted(failed_u14.items())
        )
        raise ValueError(f"U14 preflight failed: {detail}")
    eligible_additions = [
        action
        for action in actions
        if action.action_id not in core_ids and not reasons_by_action[action.action_id]
    ]
    groups = build_nested_groups(core_actions, eligible_additions, seed=str(policy["u30_seed"]))
    group_sizes = {group_id: len(action_ids) for group_id, action_ids in groups.items()}
    scale_gate = (
        "eligible_pass_count_below_60"
        if group_sizes["Uall"] < 60
        else "eligible_pass_count_at_least_60"
    )

    inventory_json = _write_json(output, "pass_inventory.json", _inventory_records(inventory, actions))
    preflight_json = _write_json(output, "pass_preflight.json", manifest_preflight_rows)
    group_records = _group_records(groups, seed=str(policy["u30_seed"]))
    groups_json = _write_json(output, "pass_groups.json", group_records)
    for group_id, group_record in group_records.items():
        _write_json(output, f"groups/{group_id}.json", group_record)

    manifest = build_study_manifest(
        programs=manifest_roots,
        pass_policy=Path(pass_policy),
        pass_inventory=json.loads(inventory_json.read_text(encoding="utf-8")),
        pass_preflight=json.loads(preflight_json.read_text(encoding="utf-8")),
        pass_groups=json.loads(groups_json.read_text(encoding="utf-8")),
        llvm_commit=dependencies.llvm_commit,
        target=dependencies.target,
        tools=tools,
        hard_state_policy=dependencies.hard_state_policy,
        comparator=dependencies.comparator,
        jobs=jobs,
        timeout_s=timeout_s,
        artifact_policy={
            **dict(dependencies.artifact_policy),
            "formal_program_count": (
                FORMAL_PROGRAM_TARGET if program_target == FORMAL_PROGRAM_TARGET else 0
            ),
            "fixed_program_count": (
                FORMAL_PROGRAM_TARGET
                if program_target == FORMAL_PROGRAM_TARGET
                else len(fixed)
            ),
            "formal_source_inventory_count": (
                FORMAL_SOURCE_INVENTORY_COUNT
                if program_target == FORMAL_PROGRAM_TARGET
                else 0
            ),
            "formal_selection_rule_id": (
                FORMAL_SELECTION_RULE_ID
                if program_target == FORMAL_PROGRAM_TARGET
                else ""
            ),
            "formal_source_positions": (
                list(FORMAL_SOURCE_POSITIONS)
                if program_target == FORMAL_PROGRAM_TARGET
                else []
            ),
            "formal_sampling_frame_sha256": formal_sampling_frame_sha256,
            "candidate_reserve_count": len(frozen.reserve_order),
            "candidate_inventory_count": candidate_inventory_count,
            "candidate_identity_exclusion_count": candidate_identity_exclusion_count,
            "candidate_identity_exclusions_sha256": _sha256_file(exclusion_path),
            "selection_seed": frozen.selection_seed,
        },
    )
    manifest_path = _write_json(output, "study_manifest.json", manifest)
    _write_prepare_csvs(
        out_dir=output,
        study_manifest_id=str(manifest["study_manifest_id"]),
        programs=manifest_roots,
        inventory=inventory,
        actions=actions,
        preflight_rows=manifest_preflight_rows,
        groups=groups,
        seed=str(policy["u30_seed"]),
    )
    completion_record = {
            "schema_version": "advisor-pair-scale-2n/prepare-v1",
            "study_manifest_id": manifest["study_manifest_id"],
            "authority_granted": False,
            "proved_commute": False,
            "program_count": len(roots),
            "formal_program_count": (
                FORMAL_PROGRAM_TARGET if program_target == FORMAL_PROGRAM_TARGET else 0
            ),
            "fixed_program_count": (
                FORMAL_PROGRAM_TARGET
                if program_target == FORMAL_PROGRAM_TARGET
                else len(fixed)
            ),
            "formal_source_inventory_count": (
                FORMAL_SOURCE_INVENTORY_COUNT
                if program_target == FORMAL_PROGRAM_TARGET
                else 0
            ),
            "formal_selection_rule_id": (
                FORMAL_SELECTION_RULE_ID
                if program_target == FORMAL_PROGRAM_TARGET
                else ""
            ),
            "formal_source_positions": (
                list(FORMAL_SOURCE_POSITIONS)
                if program_target == FORMAL_PROGRAM_TARGET
                else []
            ),
            "formal_sampling_frame_sha256": formal_sampling_frame_sha256,
            "candidate_reserve_count": len(frozen.reserve_order),
            "candidate_inventory_count": candidate_inventory_count,
            "candidate_identity_exclusion_count": candidate_identity_exclusion_count,
            "candidate_identity_exclusions_sha256": _sha256_file(exclusion_path),
            "selection_seed": frozen.selection_seed,
            "group_sizes": group_sizes,
            "scale_gate": scale_gate,
            "files_sha256": _completion_hashes(output),
        }
    completion_path = _write_json(
        output,
        "prepare_complete.json",
        _self_hashed_document(completion_record),
    )
    return PrepareResult(
        out_dir=output,
        study_manifest_path=manifest_path,
        prepare_complete_path=completion_path,
        program_count=len(roots),
        group_sizes=group_sizes,
        scale_gate=scale_gate,
        study_manifest_id=str(manifest["study_manifest_id"]),
    )


def prepare_study(
    *,
    repo_root: Path,
    out_dir: Path,
    existing_50_manifest: Path,
    pass_policy: Path,
    dependencies: PrepareDependencies,
    core_passes: Path | None = None,
    program_target: int = FORMAL_PROGRAM_TARGET,
    jobs: int = 1,
    timeout_s: int | float = 1,
) -> PrepareResult:
    """Prepare atomically in an approved isolated output subtree.

    A pre-existing output is never overwritten.  It is returned only when a
    freshly staged prepare tree is byte-identical to an already self-validated
    one; unknown, incomplete, stale, pair, or 2N content is rejected first.
    """

    target = _output_target(Path(out_dir))
    exists = target.exists()
    if exists and not target.is_dir():
        raise ValueError("existing out_dir is not a directory")
    if exists:
        # The required isolated directory skeleton may be created before the
        # first prepare command.  An empty leaf contains no user evidence and
        # can safely become the atomic-publish target; any file/child remains
        # a collision unless it is a self-hashed completed prepare tree.
        if not any(target.iterdir()):
            target.rmdir()
            exists = False
        else:
            _validate_existing_prepare_state(target)
    staging = _staging_dir(target)
    try:
        staged = _prepare_study_into(
            repo_root=repo_root,
            out_dir=staging,
            published_out_dir=target,
            existing_50_manifest=existing_50_manifest,
            pass_policy=pass_policy,
            dependencies=dependencies,
            core_passes=core_passes,
            program_target=program_target,
            jobs=jobs,
            timeout_s=timeout_s,
        )
        if exists:
            if _tree_hashes(target) != _tree_hashes(staging):
                raise ValueError(
                    "existing out_dir differs from the hash-validated current prepare state"
                )
            shutil.rmtree(staging)
            return replace(
                staged,
                out_dir=target,
                study_manifest_path=target / "study_manifest.json",
                prepare_complete_path=target / "prepare_complete.json",
            )
        os.replace(staging, target)
        return replace(
            staged,
            out_dir=target,
            study_manifest_path=target / "study_manifest.json",
            prepare_complete_path=target / "prepare_complete.json",
        )
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
