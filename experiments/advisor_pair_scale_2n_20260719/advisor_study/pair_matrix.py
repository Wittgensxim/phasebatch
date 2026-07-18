"""Root-state single-pass profiles and a complete, report-only AB/BA oracle.

This module is deliberately isolated from Phasebatch's production pair tester.
In particular, successful no-op actions are kept as first-round outputs: the
advisor's 2N experiment must be able to observe a no-op becoming active after
other first-round changes.  The functions accept injected runners, verifier,
effect extractor, and comparator so production execution APIs can be adapted
at the CLI boundary without granting any authority to this study.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from itertools import combinations
import os
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

from .pass_universe import ActionRecord
from .schema import canonical_row_id


_SUCCESS = "success"
_STAGE_STATUSES = frozenset(("success", "invalid", "error", "timeout"))
_TRUSTED_HARD_COMPARATOR_TIERS = frozenset(("hard", "hard_state", "worker_hash"))
_TRUSTED_WORKER_HASH_SOURCES = frozenset(("verified_worker", "worker_verified"))


@dataclass(frozen=True)
class HardStateEquality:
    """Comparator output for a pair of successful ordered executions.

    ``can_hard_fold`` names only equality of the observed hard states.  It is
    intentionally not a production authorization and callers always receive
    ``authority_granted=false`` and ``proved_commute=false`` evidence rows.
    """

    can_hard_fold: bool
    tier: str
    reason: str = ""
    trusted_hard_comparator: bool = False


@dataclass(frozen=True)
class _Stage:
    raw: object | None
    status: str
    verifier_status: str
    output_path: Path
    hard_state_id: str
    output_sha256: str
    stderr: str
    command: tuple[str, ...]
    invoked: bool
    fail_closed_reason: str
    worker_hash_verified: bool


@dataclass(frozen=True)
class _ProfileArtifact:
    path: Path
    available: bool
    actual_sha256: str
    reason: str


Runner = Callable[[Path, ActionRecord, Path], object]
Verifier = Callable[[Path], bool]
EffectExtractor = Callable[[Path, Path, ActionRecord], Mapping[str, object]]
Comparator = Callable[[object, object], object | None]


def complete_pair_oracle_cost(n: int) -> dict[str, int]:
    """Return the pair-oracle pass-application baseline for ``n`` actions.

    First-round outputs are reused once per action.  The ordered second stage
    contains one application for every ordered distinct action pair.
    """

    if type(n) is not int or n < 0:
        raise ValueError("n must be a non-negative integer")
    second_stage = n * (n - 1)
    return {
        "logical_first_round_applications": n,
        "logical_second_stage_applications": second_stage,
        "logical_total_pass_applications": n + second_stage,
    }


def pair_matrix_execution_cost(
    profiles: Sequence[Mapping[str, object]],
    pair_rows: Sequence[Mapping[str, object]],
    *,
    configured_action_count: int | None = None,
) -> dict[str, int]:
    """Separate idealized logical work from observed physical invocations."""

    configured_n = len(profiles) if configured_action_count is None else configured_action_count
    baseline = complete_pair_oracle_cost(configured_n)
    first_physical = sum(
        _as_nonnegative_int(row.get("physical_pass_invocations", 0))
        for row in profiles
    )
    second_physical = sum(
        _as_nonnegative_int(row.get("second_stage_physical_pass_applications", row.get("second_stage_physical_pass_invocations", 0)))
        for row in pair_rows
    )
    return {
        **baseline,
        "physical_first_round_invocations": first_physical,
        "physical_second_stage_invocations": second_physical,
        "physical_total_pass_invocations": first_physical + second_physical,
    }


def profile_single_passes(
    *,
    root_ir: Path,
    actions: Sequence[ActionRecord],
    out_dir: Path,
    run_single: Runner,
    verify_ir: Verifier | None = None,
    root_hard_state_id: str = "",
    extract_observed_effect: EffectExtractor | None = None,
    study_manifest_id: str = "",
) -> list[dict[str, object]]:
    """Profile every configured root action, including no-ops and failures.

    The result contains one row per supplied action in canonical action-ID
    order.  A successful no-op remains a successful row with its materialized
    output retained for the later direct-merge experiment.
    """

    if not root_ir.is_file():
        raise ValueError(f"root IR is missing: {root_ir}")
    _require_unique_actions(actions)
    root_digest = root_hard_state_id or _sha256_file(root_ir)
    root_output_sha256 = _sha256_file(root_ir)
    rows: list[dict[str, object]] = []
    for action in sorted(actions, key=lambda item: item.action_id):
        output = out_dir / action.action_id / "first.ll"
        stage = _invoke(
            run_single,
            root_ir,
            action,
            output,
            verify_ir=verify_ir,
            output_root=out_dir,
        )
        activity_status = "unknown"
        activity_evidence = "unavailable"
        changed_functions: tuple[str, ...] = ()
        changed_blocks: tuple[str, ...] = ()
        changed_module_regions: tuple[str, ...] = ()
        effect_available = False
        fail_reason = stage.fail_closed_reason
        if stage.status == _SUCCESS:
            activity_status, activity_evidence = _activity_from_trusted_evidence(
                stage,
                root_hard_state_id=root_digest,
                root_output_sha256=root_output_sha256,
            )
            if activity_status == "unknown":
                fail_reason = _join_reasons(
                    fail_reason,
                    "activity_state_unavailable",
                )
            try:
                effect = (
                    extract_observed_effect(root_ir, stage.output_path, action)
                    if extract_observed_effect is not None
                    else {
                        "changed_functions": (),
                        "changed_blocks": (),
                        "changed_module_regions": (),
                    }
                )
                changed_functions = _canonical_values(effect.get("changed_functions", ()))
                changed_blocks = _canonical_values(effect.get("changed_blocks", ()))
                changed_module_regions = _canonical_values(
                    effect.get("changed_module_regions", ())
                )
                effect_available = True
            except Exception as error:  # A failed extractor must never authorize.
                fail_reason = _join_reasons(
                    fail_reason, f"observed_effect_extraction_failed:{type(error).__name__}"
                )
        row_id = canonical_row_id(
            "single-pass",
            study_manifest_id,
            str(root_ir.resolve()),
            action.action_id,
        )
        rows.append(
            {
                "row_id": row_id,
                "study_manifest_id": study_manifest_id,
                "action_id": action.action_id,
                "action_name": action.name,
                "execution_status": stage.status,
                "root_hard_state_id": root_digest,
                "output_hard_state_id": stage.hard_state_id,
                "output_path": str(stage.output_path),
                "output_sha256": stage.output_sha256,
                "worker_hash_verified": _bool_text(stage.worker_hash_verified),
                "activity_status": activity_status,
                "activity_evidence": activity_evidence,
                "changed_functions": changed_functions,
                "changed_blocks": changed_blocks,
                "changed_module_regions": changed_module_regions,
                "changed_functions_json": _canonical_json(changed_functions),
                "changed_blocks_json": _canonical_json(changed_blocks),
                "changed_module_regions_json": _canonical_json(changed_module_regions),
                "observed_effect_available": effect_available,
                "verifier_status": stage.verifier_status,
                "logical_pass_applications": 1,
                "physical_pass_invocations": 1,
                "cache_reused": "false",
                "artifact_available": _bool_text(stage.status == _SUCCESS and stage.output_path.is_file()),
                "artifact_materialized": _bool_text(stage.output_path.is_file()),
                "artifact_id": canonical_row_id("artifact", row_id, stage.output_sha256),
                "fail_closed_reason": fail_reason,
                "command": list(stage.command),
                "command_sha256": _sha256_text("\0".join(stage.command)),
                "stderr": stage.stderr,
                "stderr_sha256": _sha256_text(stage.stderr),
                "authority_granted": "false",
                "proved_commute": "false",
            }
        )
    return rows


def run_complete_pair_matrix(
    *,
    root_ir: Path,
    profiles: Sequence[Mapping[str, object]],
    actions: Mapping[str, ActionRecord],
    out_dir: Path,
    profile_artifact_root: Path,
    run_second: Runner,
    compare: Comparator,
    verify_ir: Verifier | None = None,
    jobs: int = 1,
    timeout: float | int | None = None,
    study_manifest_id: str = "",
    program_id: str = "",
    group_id: str = "",
) -> list[dict[str, object]]:
    """Run both orders for every pair with successful root profiles.

    Pair tasks may execute concurrently, but ``Executor.map`` publishes rows
    in the canonical ``combinations`` order.  Each task keeps AB then BA
    sequential; profiling, 2N, checkpointing, and cleanup remain outside this
    bounded parallel section.
    """

    if not root_ir.is_file():
        raise ValueError(f"root IR is missing: {root_ir}")
    if type(jobs) is not int or jobs < 1:
        raise ValueError("jobs must be a positive integer")
    if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
        raise ValueError("timeout must be positive when provided")
    _validate_profile_action_binding(profiles, actions)

    ordered_profiles = sorted(profiles, key=lambda row: _text(row.get("action_id")))
    pair_inputs = tuple(combinations(ordered_profiles, 2))

    def run_pair(
        pair: tuple[Mapping[str, object], Mapping[str, object]],
    ) -> dict[str, object]:
        profile_a, profile_b = pair
        action_a_id = _text(profile_a.get("action_id"))
        action_b_id = _text(profile_b.get("action_id"))
        action_a = actions[action_a_id]
        action_b = actions[action_b_id]
        pair_id = canonical_row_id(
            "pair",
            study_manifest_id,
            program_id,
            group_id,
            action_a_id,
            action_b_id,
        )
        artifact_a = _validate_profile_artifact(
            profile_a,
            profile_artifact_root=profile_artifact_root,
        )
        artifact_b = _validate_profile_artifact(
            profile_b,
            profile_artifact_root=profile_artifact_root,
        )
        parent_a = artifact_a.path
        parent_b = artifact_b.path
        first_profiles_successful = (
            _text(profile_a.get("execution_status")) == _SUCCESS
            and _text(profile_b.get("execution_status")) == _SUCCESS
        )
        first_artifacts_available = (
            first_profiles_successful and artifact_a.available and artifact_b.available
        )
        pair_dir = out_dir / pair_id
        if first_artifacts_available:
            ab = _invoke(
                run_second,
                parent_a,
                action_b,
                pair_dir / "AB.ll",
                verify_ir=verify_ir,
                output_root=out_dir,
            )
            ba = _invoke(
                run_second,
                parent_b,
                action_a,
                pair_dir / "BA.ll",
                verify_ir=verify_ir,
                output_root=out_dir,
            )
        else:
            reason = _join_reasons(artifact_a.reason, artifact_b.reason)
            ab = _not_run(pair_dir / "AB.ll", reason)
            ba = _not_run(pair_dir / "BA.ll", reason)

        equality: HardStateEquality | None = None
        comparator_reason = ""
        if ab.status == _SUCCESS and ba.status == _SUCCESS:
            if (
                ab.worker_hash_verified
                and ba.worker_hash_verified
                and ab.hard_state_id
                and ab.hard_state_id == ba.hard_state_id
            ):
                equality = HardStateEquality(
                    can_hard_fold=True,
                    tier="worker_hash",
                    trusted_hard_comparator=True,
                )
            else:
                try:
                    equality = _normalize_equality(compare(ab.raw, ba.raw))
                except Exception as error:  # Comparator failure is an unknown row.
                    comparator_reason = f"comparator_failed:{type(error).__name__}"

        dynamic_result = (
            classify_pair(ab, ba, equality)
            if first_artifacts_available
            else _first_round_terminal_result(profile_a, profile_b)
        )
        relation = _observed_relation(profile_a, profile_b)
        artifact_available = (
            first_artifacts_available
            and ab.status == _SUCCESS
            and ba.status == _SUCCESS
            and ab.output_path.is_file()
            and ba.output_path.is_file()
        )
        if not artifact_available:
            _discard_partial_pair_outputs((ab.output_path, ba.output_path), pair_dir)
        fail_reason = _join_reasons(
            artifact_a.reason,
            artifact_b.reason,
            _profile_fail_reason(profile_a),
            _profile_fail_reason(profile_b),
            ab.fail_closed_reason,
            ba.fail_closed_reason,
            comparator_reason,
            "" if dynamic_result in {"commute", "order_sensitive"} else f"pair_{dynamic_result}",
        )
        commands = (*ab.command, "--then--", *ba.command)
        stderrs = _join_reasons(ab.stderr, ba.stderr)
        return {
                "row_id": pair_id,
                "study_manifest_id": study_manifest_id,
                "group_id": group_id,
                "program_id": program_id,
                "action_a_id": action_a_id,
                "action_b_id": action_b_id,
                "root_activity_class": _root_activity_class(profile_a, profile_b),
                "observed_relation": relation,
                "a_status": _text(profile_a.get("execution_status")),
                "a_hard_state_id": _text(profile_a.get("output_hard_state_id")),
                "a_output_sha256": artifact_a.actual_sha256,
                "a_verifier_status": _text(profile_a.get("verifier_status")),
                "b_status": _text(profile_b.get("execution_status")),
                "b_hard_state_id": _text(profile_b.get("output_hard_state_id")),
                "b_output_sha256": artifact_b.actual_sha256,
                "b_verifier_status": _text(profile_b.get("verifier_status")),
                "ab_status": ab.status,
                "ab_hard_state_id": ab.hard_state_id,
                "ab_output_path": str(ab.output_path),
                "ab_output_sha256": ab.output_sha256,
                "ab_verifier_status": ab.verifier_status,
                "ab_stderr_sha256": _sha256_text(ab.stderr),
                "ba_status": ba.status,
                "ba_hard_state_id": ba.hard_state_id,
                "ba_output_path": str(ba.output_path),
                "ba_output_sha256": ba.output_sha256,
                "ba_verifier_status": ba.verifier_status,
                "ba_stderr_sha256": _sha256_text(ba.stderr),
                "dynamic_result": dynamic_result,
                "second_stage_logical_pass_applications": 2 if first_artifacts_available else 0,
                "second_stage_physical_pass_invocations": int(ab.invoked) + int(ba.invoked),
                "total_logical_pass_applications": 4 if first_artifacts_available else 2,
                "total_physical_pass_invocations": int(ab.invoked) + int(ba.invoked),
                "reused_single_pass_outputs": "true",
                "artifact_available": _bool_text(artifact_available),
                "artifact_materialized": _bool_text(artifact_available),
                "artifact_id": canonical_row_id("artifact", pair_id, ab.output_sha256, ba.output_sha256),
                "fail_closed_reason": fail_reason,
                "command": list(commands),
                "command_sha256": _sha256_text("\0".join(commands)),
                "stderr": stderrs,
                "stderr_sha256": _sha256_text(stderrs),
                "authority_granted": "false",
                "proved_commute": "false",
            }

    if jobs == 1:
        return [run_pair(pair) for pair in pair_inputs]
    with ThreadPoolExecutor(
        max_workers=jobs,
        thread_name_prefix="advisor-uall-pair",
    ) as executor:
        return list(executor.map(run_pair, pair_inputs))


def reclaim_equal_pair_artifacts(
    rows: Sequence[Mapping[str, object]],
    *,
    pair_artifact_root: Path,
    protected_paths: Sequence[Path | str] = (),
) -> list[dict[str, object]]:
    """Reclaim only redundant, verified equal AB/BA IR evidence.

    Pair rows retain their independent hard-state/output digests and execution
    provenance even after the two identical materializations are removed.
    Anything terminal, non-commuting, false-authorizing, unverified, outside
    the pair stage, or referenced by another row is preserved fail-closed.
    ``protected_paths`` lets a later witness/replay stage reserve artifacts
    explicitly before this bounded reclamation is attempted.
    """

    root = pair_artifact_root.resolve(strict=False)
    normalized = [dict(row) for row in rows]
    path_pairs = [
        _pair_artifact_paths(row, root)
        for row in normalized
    ]
    references: dict[Path, int] = {}
    for left, right in path_pairs:
        for path in (left, right):
            if _is_relative_to(path, root):
                references[path] = references.get(path, 0) + 1
    protected = {
        Path(value).resolve(strict=False)
        for value in protected_paths
    }

    for row, (ab_path, ba_path) in zip(normalized, path_pairs, strict=True):
        reason = _pair_retention_reason(row, ab_path, ba_path, root, references, protected)
        if reason:
            row["retention_status"] = reason
            continue
        bindings = _preunlink_reclaim_bindings(row, ab_path, ba_path)
        if bindings is None or not _reclaim_bindings_current(bindings):
            row["retention_status"] = "retained_toctou_drift"
            continue
        fully_reclaimed, actual_paths, reclamation_status, reclaimed_paths = _reclaim_via_quarantine(
            bindings, ab_path.parent
        )
        if not fully_reclaimed:
            # A race can replace an original after its last pre-move check.
            # The moved bytes are evidence, so never delete them; record their
            # real location instead of recreating or overwriting an original.
            row["retention_status"] = (
                "retained_quarantine_delete_failed"
                if reclamation_status == "quarantine_delete_failed"
                else "retained_toctou_drift"
            )
            row["reclamation_status"] = reclamation_status
            row["reclamation_artifact_paths"] = actual_paths
            if reclaimed_paths:
                row["reclamation_reclaimed_paths"] = reclaimed_paths
            row["artifact_available"] = "true"
            row["artifact_materialized"] = "true"
            continue
        row["artifact_available"] = "true"
        row["artifact_materialized"] = "false"
        row["retention_status"] = "reclaimed_equal_unreferenced"
    return normalized


def _discard_partial_pair_outputs(paths: Sequence[Path], pair_dir: Path) -> None:
    """Keep an unmaterialized terminal pair physically unmaterialized."""

    root = pair_dir.resolve(strict=False)
    for raw_path in paths:
        supplied = Path(raw_path)
        resolved = supplied.resolve(strict=False)
        if not _is_relative_to(resolved, root):
            if supplied.exists():
                raise ValueError("partial pair artifact escaped its pair directory")
            continue
        for attempt in range(7):
            try:
                supplied.unlink(missing_ok=True)
            except OSError:
                if attempt == 6:
                    break
                time.sleep(0.01 * (2**attempt))
            if not supplied.exists():
                break
        if supplied.exists():
            raise RuntimeError("partial pair artifact could not be reclaimed")


def classify_pair(
    ab: object,
    ba: object,
    equality: HardStateEquality | None,
) -> str:
    """Return the closed dynamic outcome without strengthening its meaning."""

    ab_status = _stage_status(ab)
    ba_status = _stage_status(ba)
    if ab_status == "timeout" or ba_status == "timeout":
        return "timeout"
    if ab_status != _SUCCESS or ba_status != _SUCCESS:
        return "failed"
    if equality is None or equality.tier == "failed":
        return "unknown"
    return "commute" if equality.can_hard_fold else "order_sensitive"


def _first_round_terminal_result(
    profile_a: Mapping[str, object], profile_b: Mapping[str, object]
) -> str:
    """Classify a full-matrix row whose second stage was correctly not run."""

    statuses = {
        _text(profile_a.get("execution_status")),
        _text(profile_b.get("execution_status")),
    }
    if "timeout" in statuses:
        return "timeout"
    if any(status and status != _SUCCESS for status in statuses):
        return "failed"
    return "unknown"


def _pair_artifact_paths(row: Mapping[str, object], root: Path) -> tuple[Path, Path]:
    """Read explicit pair paths only; a missing provenance path is retained."""

    missing = root / "__unrecorded_pair_artifact__"
    ab_raw, ba_raw = _text(row.get("ab_output_path")), _text(row.get("ba_output_path"))
    return (
        Path(ab_raw).resolve(strict=False) if ab_raw else missing / "AB.ll",
        Path(ba_raw).resolve(strict=False) if ba_raw else missing / "BA.ll",
    )


def _pair_retention_reason(
    row: Mapping[str, object],
    ab_path: Path,
    ba_path: Path,
    root: Path,
    references: Mapping[Path, int],
    protected: set[Path],
) -> str:
    if _as_bool(row.get("false_authorization", False)):
        return "retained_false_authorization"
    if (
        _text(row.get("ab_status")) != _SUCCESS
        or _text(row.get("ba_status")) != _SUCCESS
        or _text(row.get("ab_verifier_status")) != _SUCCESS
        or _text(row.get("ba_verifier_status")) != _SUCCESS
    ):
        return "retained_terminal_status"
    if _text(row.get("dynamic_result")) != "commute":
        return "retained_noncommuting"
    if not _as_bool(row.get("artifact_available")) or not _as_bool(row.get("artifact_materialized")):
        return "retained_not_materialized"
    if not _pair_provenance_complete(row):
        return "retained_unverified_provenance"
    if not _pair_paths_are_canonical(row, ab_path, ba_path, root):
        return "retained_noncanonical_path"
    ab_sha, ba_sha = _text(row.get("ab_output_sha256")), _text(row.get("ba_output_sha256"))
    if not _is_sha256(ab_sha) or ab_sha != ba_sha:
        return "retained_hash_mismatch"
    if _text(row.get("ab_hard_state_id")) != _text(row.get("ba_hard_state_id")):
        return "retained_state_mismatch"
    if (
        not _is_relative_to(ab_path, root)
        or not _is_relative_to(ba_path, root)
        or ab_path == ba_path
    ):
        return "retained_outside_or_aliased_path"
    if (
        ab_path in protected
        or ba_path in protected
        or references.get(ab_path, 0) != 1
        or references.get(ba_path, 0) != 1
    ):
        return "retained_referenced"
    if (
        not ab_path.is_file()
        or not ba_path.is_file()
        or _sha256_file(ab_path) != ab_sha
        or _sha256_file(ba_path) != ba_sha
    ):
        return "retained_artifact_drift"
    return ""


def _pair_paths_are_canonical(
    row: Mapping[str, object], ab_path: Path, ba_path: Path, root: Path
) -> bool:
    row_id = _text(row.get("row_id"))
    if not row_id or Path(row_id).name != row_id or not all(
        character.isalnum() or character in "-_" for character in row_id
    ):
        return False
    expected_ab = (root / row_id / "AB.ll").resolve(strict=False)
    expected_ba = (root / row_id / "BA.ll").resolve(strict=False)
    return (
        _text(row.get("ab_output_path")) == str(expected_ab)
        and _text(row.get("ba_output_path")) == str(expected_ba)
        and ab_path == expected_ab
        and ba_path == expected_ba
    )


def _preunlink_reclaim_bindings(
    row: Mapping[str, object], ab_path: Path, ba_path: Path
) -> tuple[tuple[Path, tuple[int, int, int, int], str], ...] | None:
    ab = _bind_reclaim_target(ab_path, _text(row.get("ab_output_sha256")))
    ba = _bind_reclaim_target(ba_path, _text(row.get("ba_output_sha256")))
    if ab is None or ba is None:
        return None
    return (ab, ba)


def _bind_reclaim_target(
    path: Path, expected_sha256: str
) -> tuple[Path, tuple[int, int, int, int], str] | None:
    """Bind file identity and bytes so replacement before unlink fails closed."""

    before = _file_identity(path)
    if before is None or _sha256_file(path) != expected_sha256:
        return None
    after = _file_identity(path)
    if after is None or before != after:
        return None
    return path, after, expected_sha256


def _reclaim_bindings_current(
    bindings: Sequence[tuple[Path, tuple[int, int, int, int], str]]
) -> bool:
    """Perform the final identity/hash check immediately before deletion."""

    return all(
        _file_identity(path) == identity and _sha256_file(path) == expected_sha256
        for path, identity, expected_sha256 in bindings
    )


def _file_identity(path: Path) -> tuple[int, int, int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _reclaim_via_quarantine(
    bindings: Sequence[tuple[Path, tuple[int, int, int, int], str]],
    pair_directory: Path,
) -> tuple[bool, dict[str, str], str, dict[str, str]]:
    """Atomically move candidates, then verify moved evidence before unlink.

    The quarantine is a unique child of the same pair directory, therefore an
    ``os.replace`` is an in-volume rename rather than a cross-volume copy.  If
    a source changes after the final source check, its moved replacement is
    deliberately retained in quarantine and reported to the caller.
    """

    quarantine = pair_directory / ".reclaim-quarantine"
    quarantine.mkdir(parents=False, exist_ok=True)
    names = ("AB", "BA")
    actual_paths = {name: str(binding[0]) for name, binding in zip(names, bindings, strict=True)}
    moved: list[tuple[Path, tuple[int, int, int, int], str]] = []
    reclaimed_paths: dict[str, str] = {}
    for name, (source, identity, expected_sha256) in zip(names, bindings, strict=True):
        target = quarantine / f"{name}-{uuid4().hex}.ll"
        try:
            os.replace(source, target)
        except OSError:
            return False, _existing_reclamation_paths(actual_paths), "quarantine_move_failed", reclaimed_paths
        actual_paths[name] = str(target)
        moved_binding = (target, identity, expected_sha256)
        moved.append(moved_binding)
        if not _reclaim_bindings_current((moved_binding,)):
            return False, _existing_reclamation_paths(actual_paths), "quarantined_drift", reclaimed_paths
    # Recheck the complete moved pair so a race in the second move cannot
    # cause deletion of a previously moved first artifact.
    if not _reclaim_bindings_current(moved):
        return False, _existing_reclamation_paths(actual_paths), "quarantined_drift", reclaimed_paths
    for name, (target, _identity, _expected_sha256) in zip(names, moved, strict=True):
        try:
            target.unlink()
        except OSError:
            return (
                False,
                _existing_reclamation_paths(actual_paths),
                "quarantine_delete_failed",
                reclaimed_paths,
            )
        reclaimed_paths[name] = str(target)
    try:
        quarantine.rmdir()
    except OSError:
        # An empty directory is harmless and contains no experiment evidence.
        pass
    return True, {}, "reclaimed", reclaimed_paths


def _existing_reclamation_paths(paths: Mapping[str, str]) -> dict[str, str]:
    """Do not report a quarantine path after its file was successfully deleted."""

    return {
        name: value
        for name, value in paths.items()
        if Path(value).is_file()
    }


def _pair_provenance_complete(row: Mapping[str, object]) -> bool:
    required_text = (
        "row_id",
        "study_manifest_id",
        "program_id",
        "group_id",
        "action_a_id",
        "action_b_id",
    )
    return (
        all(_text(row.get(name)) for name in required_text)
        and _is_sha256(_text(row.get("command_sha256")))
        and _is_sha256(_text(row.get("stderr_sha256")))
        and _is_sha256(_text(row.get("ab_hard_state_id")))
        and _is_sha256(_text(row.get("ba_hard_state_id")))
    )


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value.lower())


def _invoke(
    runner: Runner,
    parent: Path,
    action: ActionRecord,
    requested_output: Path,
    *,
    verify_ir: Verifier | None,
    output_root: Path,
) -> _Stage:
    try:
        raw = runner(parent, action, requested_output)
    except Exception as error:
        return _Stage(
            raw=None,
            status="error",
            verifier_status="not_run",
            output_path=requested_output,
            hard_state_id="",
            output_sha256="",
            stderr="",
            command=(),
            invoked=True,
            fail_closed_reason=f"runner_failed:{type(error).__name__}",
            worker_hash_verified=False,
        )
    return _stage_from_raw(
        raw,
        requested_output,
        verify_ir=verify_ir,
        output_root=output_root,
    )


def _stage_from_raw(
    raw: object,
    requested_output: Path,
    *,
    verify_ir: Verifier | None,
    output_root: Path,
) -> _Stage:
    status = _stage_status(raw)
    reported_output_path = _path_field(raw, "output_path", requested_output)
    output_path = requested_output
    stderr = _text(_field(raw, "stderr", ""))
    command = _command_tuple(_field(raw, "command", ()))
    verifier_status = "not_run"
    fail_reason = ""
    requested_resolved = requested_output.resolve()
    output_root_resolved = output_root.resolve()
    if not _is_relative_to(requested_resolved, output_root_resolved):
        status = "error"
        fail_reason = "requested_output_outside_artifact_root"
    elif reported_output_path.resolve() != requested_resolved:
        status = "error"
        fail_reason = "output_path_redirected"
    if status == _SUCCESS:
        if not output_path.is_file():
            status = "error"
            fail_reason = "missing_output_artifact"
        elif verify_ir is None:
            verifier_status = _text(_field(raw, "verifier_status", "success")) or "success"
            if verifier_status != "success":
                status = "invalid" if verifier_status == "invalid" else "error"
                fail_reason = f"verifier_{verifier_status}"
        else:
            try:
                valid = bool(verify_ir(output_path))
            except Exception as error:
                status = "error"
                verifier_status = "unknown"
                fail_reason = f"verifier_failed:{type(error).__name__}"
            else:
                verifier_status = "success" if valid else "invalid"
                if not valid:
                    status = "invalid"
                    fail_reason = "verifier_invalid"
    hard_state_id = _text(_field(raw, "hard_state_id", ""))
    output_sha256 = _sha256_file(output_path) if output_path.is_file() else ""
    if status == _SUCCESS and not hard_state_id:
        hard_state_id = output_sha256
    if status != _SUCCESS and not fail_reason:
        fail_reason = f"stage_{status}"
    return _Stage(
        raw=raw,
        status=status,
        verifier_status=verifier_status,
        output_path=output_path,
        hard_state_id=hard_state_id,
        output_sha256=output_sha256,
        stderr=stderr,
        command=command,
        invoked=True,
        fail_closed_reason=fail_reason,
        worker_hash_verified=_worker_hash_verified(raw),
    )


def _not_run(path: Path, reason: str) -> _Stage:
    return _Stage(
        raw=None,
        status="not_run",
        verifier_status="not_run",
        output_path=path,
        hard_state_id="",
        output_sha256="",
        stderr="",
        command=(),
        invoked=False,
        fail_closed_reason=reason,
        worker_hash_verified=False,
    )


def _stage_status(value: object) -> str:
    if isinstance(value, _Stage):
        return value.status
    explicit = _text(_field(value, "execution_status", _field(value, "status", "")))
    if explicit in _STAGE_STATUSES:
        return explicit
    if bool(_field(value, "timed_out", False)):
        return "timeout"
    return _SUCCESS if bool(_field(value, "success", False)) else "error"


def _normalize_equality(value: object | None) -> HardStateEquality | None:
    if value is None:
        return None
    if isinstance(value, HardStateEquality):
        if (
            value.tier not in _TRUSTED_HARD_COMPARATOR_TIERS
            or not value.trusted_hard_comparator
        ):
            return None
        return value
    tier = _text(_field(value, "tier", ""))
    if tier not in _TRUSTED_HARD_COMPARATOR_TIERS:
        return None
    fold = _field(value, "can_hard_fold", None)
    trusted = _field(value, "trusted_hard_comparator", None)
    if not isinstance(fold, bool) or trusted is not True:
        return None
    return HardStateEquality(
        can_hard_fold=fold,
        tier=tier,
        reason=_text(_field(value, "reason", "")),
        trusted_hard_comparator=True,
    )


def _observed_relation(
    profile_a: Mapping[str, object], profile_b: Mapping[str, object]
) -> str:
    effect_a = _profile_effect(profile_a)
    effect_b = _profile_effect(profile_b)
    if effect_a is None or effect_b is None:
        return "observed_unknown"
    changed_a = set().union(*effect_a) if effect_a else set()
    changed_b = set().union(*effect_b) if effect_b else set()
    return "observed_overlap" if changed_a & changed_b else "observed_disjoint"


def _profile_effect(profile: Mapping[str, object]) -> tuple[tuple[str, ...], ...] | None:
    if not _as_bool(profile.get("observed_effect_available", True)):
        return None
    values: list[tuple[str, ...]] = []
    for plain_key, json_key in (
        ("changed_functions", "changed_functions_json"),
        ("changed_blocks", "changed_blocks_json"),
        ("changed_module_regions", "changed_module_regions_json"),
    ):
        if plain_key in profile:
            values.append(_canonical_values(profile[plain_key]))
        elif json_key in profile:
            values.append(_canonical_values(profile[json_key]))
        else:
            return None
    return tuple(values)


def _root_activity_class(profile_a: Mapping[str, object], profile_b: Mapping[str, object]) -> str:
    activity_a = _text(profile_a.get("activity_status"))
    activity_b = _text(profile_b.get("activity_status"))
    if activity_a == "active" and activity_b == "active":
        return "active_active"
    if {activity_a, activity_b} == {"active", "no_op"}:
        return "active_noop"
    if activity_a == "no_op" and activity_b == "no_op":
        return "noop_noop"
    return "unknown"


def _validate_profile_action_binding(
    profiles: Sequence[Mapping[str, object]], actions: Mapping[str, ActionRecord]
) -> None:
    seen: set[str] = set()
    for profile in profiles:
        action_id = _text(profile.get("action_id"))
        if not action_id:
            raise ValueError("single-pass profile requires action_id")
        if action_id in seen:
            raise ValueError(f"duplicate single-pass profile action_id: {action_id}")
        if action_id not in actions:
            raise ValueError(f"single-pass profile action is absent from action map: {action_id}")
        if actions[action_id].action_id != action_id:
            raise ValueError(f"action map identity mismatch: {action_id}")
        seen.add(action_id)


def _require_unique_actions(actions: Sequence[ActionRecord]) -> None:
    action_ids = [action.action_id for action in actions]
    if len(action_ids) != len(set(action_ids)):
        raise ValueError("duplicate action IDs in root profiling")


def _profile_output(profile: Mapping[str, object]) -> Path:
    raw = _text(profile.get("output_path"))
    return Path(raw) if raw else Path("__missing_single_pass_output__.ll")


def _profile_output_hash(profile: Mapping[str, object]) -> str:
    explicit = _text(profile.get("output_sha256"))
    return explicit or _sha256_file(_profile_output(profile))


def _validate_profile_artifact(
    profile: Mapping[str, object],
    *,
    profile_artifact_root: Path,
) -> _ProfileArtifact:
    """Re-hash a persisted first-round artifact before it becomes a parent.

    A profile is evidence, not a capability: a successful status does not
    permit a second-stage invocation unless its materialized bytes still match
    the profile digest exactly.
    """

    path = _profile_output(profile)
    controlled_root = profile_artifact_root.resolve()
    if not _is_relative_to(path.resolve(), controlled_root):
        return _ProfileArtifact(
            path,
            False,
            "",
            "profile_output_outside_artifact_root",
        )
    expected_sha256 = _text(profile.get("output_sha256"))
    if not _as_bool(profile.get("artifact_available", True)):
        return _ProfileArtifact(path, False, "", "profile_artifact_unavailable")
    if not path.is_file():
        return _ProfileArtifact(path, False, "", "missing_first_round_artifact")
    actual_sha256 = _sha256_file(path)
    if not expected_sha256:
        return _ProfileArtifact(
            path,
            False,
            actual_sha256,
            "missing_profile_output_sha256",
        )
    if actual_sha256 != expected_sha256:
        return _ProfileArtifact(
            path,
            False,
            actual_sha256,
            "profile_output_sha256_mismatch",
        )
    return _ProfileArtifact(path, True, actual_sha256, "")


def _activity_from_trusted_evidence(
    stage: _Stage,
    *,
    root_hard_state_id: str,
    root_output_sha256: str,
) -> tuple[str, str]:
    """Classify root activity without trusting a bare reported state ID."""

    if (
        stage.worker_hash_verified
        and stage.hard_state_id
        and root_hard_state_id
    ):
        return (
            "no_op" if stage.hard_state_id == root_hard_state_id else "active",
            "verified_hard_state_id",
        )
    if stage.output_sha256 and root_output_sha256:
        return (
            "no_op" if stage.output_sha256 == root_output_sha256 else "active",
            "output_sha256",
        )
    return "unknown", "unavailable"


def _profile_fail_reason(profile: Mapping[str, object]) -> str:
    return _text(profile.get("fail_closed_reason"))


def _field(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _path_field(value: object, name: str, default: Path) -> Path:
    raw = _field(value, name, default)
    if isinstance(raw, Path):
        return raw
    text = _text(raw)
    return Path(text) if text else default


def _command_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return ()


def _canonical_values(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = (value,)
        value = decoded
    if isinstance(value, Mapping):
        values = [f"{key}:{item}" for key, item in value.items()]
    elif isinstance(value, Sequence) or isinstance(value, set):
        values = [str(item) for item in value]
    else:
        values = [str(value)]
    return tuple(sorted(set(values)))


def _canonical_json(values: tuple[str, ...]) -> str:
    return json.dumps(list(values), ensure_ascii=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _text(value: object) -> str:
    return "" if value is None else str(value)


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() == "true"


def _worker_hash_verified(raw: object) -> bool:
    """Accept only an explicit protocol assertion or a closed trusted source."""

    if _field(raw, "worker_hash_verified", False) is True:
        return True
    return _text(_field(raw, "hard_state_source", "")) in _TRUSTED_WORKER_HASH_SOURCES


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _as_nonnegative_int(value: object) -> int:
    try:
        converted = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, converted)


def _join_reasons(*reasons: str) -> str:
    return ";".join(reason for reason in reasons if reason)
