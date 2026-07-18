"""Deterministic, evidence-only materialization for the advisor study.

This module is deliberately a reporting boundary: it reads already frozen
evidence rows, fails closed on identity or authority violations, and writes
only aggregate tables, Chinese advisor materials, and fixed-size figures.
It does not execute transformations, construct batches, or grant authority.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Mapping, Sequence

from .report import (
    build_pair_confusion,
    build_pair_metric_rows,
    derive_group_pair_rows,
    pair_oracle_cost,
    ratio,
)
from .schema import (
    PAIR_METRICS_FIELDS,
    STATUS_VALUES_BY_TABLE,
    TABLE_FIELDS,
    canonical_row_id,
    require_authority_off,
    write_csv,
)


FORBIDDEN_CLAIMS = (
    "2N proves soundness",
    "static proof",
    "N! dynamic baseline",
    "semantic equivalence",
    "no counterexample proves commute",
)

_GROUP_ORDER = ("U14", "U30", "Uall")
_SOURCE_INDEX_FIELDS = (
    "row_id",
    "study_manifest_id",
    "source_table",
    "source_row_id",
    "source_kind",
    "authority_granted",
    "proved_commute",
)
_GATE_FUNNEL_FIELDS = (
    "row_id",
    "study_manifest_id",
    "group_id",
    "gate",
    "program_count",
    "program_rate",
    "source_row_ids",
    "authority_granted",
    "proved_commute",
)
_FALSE_AUTHORIZATION_FIELDS = (
    "row_id",
    "study_manifest_id",
    "group_id",
    "program_id",
    "action_a_id",
    "action_b_id",
    "two_n_pair_status",
    "dynamic_result",
    "validation_status",
    "stable_false_authorization",
    "worker_replay_status",
    "external_opt_replay_status",
    "two_n_replay_status",
    "source_row_ids",
    "authority_granted",
    "proved_commute",
)
_EXECUTION_COST_FIELDS = (
    "row_id",
    "study_manifest_id",
    "group_id",
    "method",
    "scope",
    "program_count",
    "configured_n",
    "logical_first_round_applications",
    "logical_second_stage_applications",
    "logical_total_pass_applications",
    "physical_first_round_invocations",
    "physical_second_stage_invocations",
    "physical_pass_invocations",
    "cache_reuse_count",
    "content_addressed_reuse_event_count",
    "reuse_event_source_row_ids",
    "merge_helper_calls",
    "merge_construction_time_ms",
    "parse_time_ms",
    "verifier_time_ms",
    "worker_time_ms",
    "replay_time_ms",
    "total_wall_time_ms",
    "provenance",
    "source_row_ids",
    "authority_granted",
    "proved_commute",
)
_LIMITATION_FIELDS = (
    "row_id",
    "study_manifest_id",
    "group_id",
    "limitation_kind",
    "case_count",
    "denominator_count",
    "source_row_ids",
    "denominator_source_row_ids",
    "interpretation_zh",
    "authority_granted",
    "proved_commute",
)

_DERIVED_TABLE_FIELDS: Mapping[str, tuple[str, ...]] = {
    "pair_pass_family_metrics.csv": PAIR_METRICS_FIELDS,
    "two_n_gate_funnel.csv": _GATE_FUNNEL_FIELDS,
    "two_n_false_authorizations.csv": _FALSE_AUTHORIZATION_FIELDS,
    "execution_costs.csv": _EXECUTION_COST_FIELDS,
    "study_limitations.csv": _LIMITATION_FIELDS,
    "source_index.csv": _SOURCE_INDEX_FIELDS,
}

_BASE_TABLE_ORDER = tuple(TABLE_FIELDS)
_DERIVED_TABLE_ORDER = (
    "pair_pass_family_metrics.csv",
    "two_n_gate_funnel.csv",
    "two_n_false_authorizations.csv",
    "execution_costs.csv",
    "study_limitations.csv",
    "source_index.csv",
)
_REPORT_FILES = (
    "README.md",
    "expanded_pair_and_2n_report.md",
    "advisor_talking_points.md",
    "advisor_q_and_a.md",
    "evidence_manifest.json",
)
_FIGURE_STEMS = (
    "01_pair_precision_coverage",
    "02_2n_gate_applicability",
    "03_2n_pair_agreement",
    "04_execution_cost",
)

_NUMERIC_REQUIRED_FIELDS: Mapping[str, frozenset[str]] = {
    "program_manifest.csv": frozenset({"selection_order", "source_size_bytes", "reserve_rank"}),
    "pass_preflight.csv": frozenset({"repetition", "logical_pass_applications", "physical_pass_invocations"}),
    "pass_groups.csv": frozenset({"group_size", "action_order", "selection_rank"}),
    "single_pass_observations.csv": frozenset({"logical_pass_applications", "physical_pass_invocations", "wall_time_ms"}),
    "pair_observations.csv": frozenset({"second_stage_logical_pass_applications", "second_stage_physical_pass_invocations", "total_logical_pass_applications", "total_physical_pass_invocations", "wall_time_ms"}),
    "pair_dynamic_confusion.csv": frozenset({"pair_count", "program_count"}),
    "advisor_2n_group_results.csv": frozenset({"configured_n", "successful_n", "active_n", "no_op_n", "failed_n", "timeout_n", "directional_authorized_count", "directional_unavailable_count", "logical_pass_applications", "physical_pass_invocations", "merge_helper_calls", "merge_construction_time_ms", "parse_time_ms", "verifier_time_ms", "worker_time_ms", "replay_time_ms", "wall_time_ms"}),
    "advisor_2n_directional_results.csv": frozenset({"logical_pass_applications", "physical_pass_invocations", "merge_helper_calls", "wall_time_ms"}),
    "advisor_2n_pair_validation.csv": frozenset({"replay_time_ms"}),
    "artifact_index.csv": frozenset({"size_bytes"}),
}
REQUIRED_AGGREGATE_FILES = tuple(
    (*_BASE_TABLE_ORDER, *_DERIVED_TABLE_ORDER, *_REPORT_FILES,
     *(f"{stem}.svg" for stem in _FIGURE_STEMS),
     *(f"{stem}.png" for stem in _FIGURE_STEMS))
)


@dataclass(frozen=True)
class AggregateResult:
    """Complete deterministic aggregate materialization result."""

    out_dir: Path
    aggregate_dir: Path
    figures_dir: Path
    files: tuple[Path, ...]


def validate_claim_text(text: str) -> None:
    """Reject wording that would strengthen empirical evidence into proof."""

    folded = str(text).casefold()
    for phrase in FORBIDDEN_CLAIMS:
        if phrase.casefold() in folded:
            raise ValueError(f"forbidden unqualified claim: {phrase}")


def build_failure_ledger_rows(
    *,
    preflight_rows: Sequence[Mapping[str, object]],
    group_rows: Sequence[Mapping[str, object]],
    pair_rows: Sequence[Mapping[str, object]],
    study_manifest_id: str,
) -> list[dict[str, object]]:
    """Project canonical causal terminal evidence into a typed ledger.

    Directional and validation rows propagate one group gate, while U14/U30
    pair views derive from Uall.  This ledger therefore records preflight
    failures, one row per 2N group gate, and canonical Uall pair terminals.
    """

    manifest = _nonempty(study_manifest_id, "study_manifest_id")
    rows: list[dict[str, object]] = []
    seen_sources: set[tuple[str, str]] = set()

    def append(
        *, source: Mapping[str, object], stage: str, failure_kind: str,
        status: str, reason_field: str, group_id: str = "", action_id: str = "",
        pair_id: str = "",
    ) -> None:
        source_id = _nonempty(source.get("row_id"), f"{stage} source row_id")
        source_key = (stage, source_id)
        if source_key in seen_sources:
            raise ValueError(f"duplicate failure source row: {stage}:{source_id}")
        rows.append(
            {
                "row_id": canonical_row_id("failure-ledger", manifest, stage, source_id),
                "study_manifest_id": manifest,
                "stage": stage,
                "group_id": group_id,
                "program_id": _text(source.get("program_id")),
                "action_id": action_id,
                "pair_id": pair_id,
                "failure_kind": failure_kind,
                "status": status,
                "reason": _text(source.get(reason_field)) or failure_kind,
                "command": "",
                "command_sha256": _text(source.get("command_sha256")),
                "stderr_path": "",
                "stderr_sha256": _text(source.get("stderr_sha256")),
                "artifact_id": _text(source.get("artifact_id")),
                "source_row_ids": _canonical_json([source_id]),
                "authority_granted": "false",
                "proved_commute": "false",
            }
        )
        seen_sources.add(source_key)

    for preflight in sorted(preflight_rows, key=lambda row: _text(row.get("row_id"))):
        status = _text(preflight.get("execution_status"))
        if status == "success":
            continue
        if status not in {"error", "timeout", "invalid"}:
            raise ValueError(f"unsupported terminal preflight status: {status}")
        append(
            source=preflight,
            stage="pass_preflight",
            failure_kind=f"pass_preflight_{status}",
            status=status,
            reason_field="exclusion_reason",
            action_id=_text(preflight.get("action_id")),
        )

    for group in sorted(group_rows, key=lambda row: _text(row.get("row_id"))):
        failure_kind, status = _two_n_group_failure(group)
        if not failure_kind:
            continue
        append(
            source=group,
            stage="advisor_2n_group_gate",
            failure_kind=failure_kind,
            status=status,
            reason_field="fail_closed_reason",
            group_id=_text(group.get("group_id")),
        )

    for pair in sorted(pair_rows, key=lambda row: _text(row.get("row_id"))):
        if _text(pair.get("group_id")) != "Uall":
            continue
        dynamic = _text(pair.get("dynamic_result"))
        if dynamic not in {"failed", "timeout", "unknown"}:
            continue
        source_id = _nonempty(pair.get("row_id"), "terminal pair row_id")
        failure_kind = _pair_failure_kind(pair, dynamic)
        status = (
            "timeout" if dynamic == "timeout"
            else "unresolved" if dynamic == "unknown"
            else "error" if failure_kind.endswith("_error")
            else "invalid" if failure_kind.endswith("_invalid_ir")
            else "failed"
        )
        append(
            source=pair,
            stage="pair_oracle",
            failure_kind=failure_kind,
            status=status,
            reason_field="fail_closed_reason",
            group_id="Uall",
            pair_id=source_id,
        )
    return sorted(rows, key=lambda row: (_text(row["stage"]), _text(row["source_row_ids"])))


def _two_n_group_failure(group: Mapping[str, object]) -> tuple[str, str]:
    round1 = _text(group.get("round1_status"))
    disjoint = _text(group.get("first_round_disjoint_status"))
    merge = _text(group.get("all_n_merge_status"))
    second = _text(group.get("all_n_second_round_status"))
    authorization = _text(group.get("group_authorization_status"))
    if round1 == "timeout":
        return "round1_precondition_timeout", "timeout"
    if round1 != "complete":
        return "round1_precondition_failed", "unavailable"
    if disjoint == "overlap":
        return "overlapping_patch_region", "unavailable"
    if disjoint != "disjoint":
        return "patch_region_unknown", "unresolved"
    if merge == "timeout":
        return "direct_merge_timeout", "timeout"
    if merge == "merge_invalid":
        return "merged_ir_invalid", "invalid"
    if merge != "complete":
        return "direct_merge_undefined", "unavailable"
    if second == "timeout":
        return "second_round_timeout", "timeout"
    if second != "complete":
        return "second_round_failed", "failed"
    if authorization in {"group_precondition_unavailable", "unknown"}:
        return "group_authorization_unavailable", "unavailable"
    return "", ""


def _pair_failure_kind(pair: Mapping[str, object], dynamic: str) -> str:
    endpoint_statuses = {_text(pair.get("a_status")), _text(pair.get("b_status"))}
    ordered_statuses = {_text(pair.get("ab_status")), _text(pair.get("ba_status"))}
    verifier_statuses = {
        _text(pair.get("ab_verifier_status")),
        _text(pair.get("ba_verifier_status")),
    }
    if "timeout" in endpoint_statuses:
        return "first_round_timeout"
    if "invalid" in endpoint_statuses:
        return "first_round_invalid_ir"
    if "error" in endpoint_statuses:
        return "first_round_error"
    if "timeout" in ordered_statuses:
        return "ab_ba_timeout"
    if "invalid" in ordered_statuses or "invalid" in verifier_statuses:
        return "ab_ba_invalid_ir"
    if "error" in ordered_statuses:
        return "ab_ba_error"
    if dynamic == "unknown":
        return "comparator_uncertainty"
    return "ab_ba_failed"


def build_artifact_index_rows(
    *,
    program_rows: Sequence[Mapping[str, object]],
    single_rows: Sequence[Mapping[str, object]],
    pair_rows: Sequence[Mapping[str, object]],
    directional_rows: Sequence[Mapping[str, object]],
    materialized_artifact_bindings: Sequence[Mapping[str, object]],
    study_manifest_id: str,
    isolation_root: Path,
) -> list[dict[str, object]]:
    """Index logical artifacts with validated checkpoint bytes taking priority."""

    manifest = _nonempty(study_manifest_id, "study_manifest_id")
    root = Path(isolation_root).resolve(strict=False)
    bindings: dict[tuple[str, str], dict[str, Mapping[str, object]]] = {}
    for original in materialized_artifact_bindings:
        if not isinstance(original, Mapping):
            raise ValueError("materialized artifact binding must be a mapping")
        binding = dict(original)
        source_kind = _nonempty(binding.get("source_kind"), "artifact binding source_kind")
        source_id = _nonempty(binding.get("source_row_id"), "artifact binding source_row_id")
        artifact_name = _nonempty(binding.get("artifact_name"), "artifact binding artifact_name")
        key = (source_kind, source_id)
        by_name = bindings.setdefault(key, {})
        if artifact_name in by_name:
            raise ValueError(f"duplicate materialized artifact binding: {key}:{artifact_name}")
        relative = _relative_artifact_path(binding.get("relative_path"), root)
        physical = root / Path(relative)
        if not physical.is_file():
            raise ValueError(f"materialized artifact binding is missing: {relative}")
        digest = _nonempty(binding.get("sha256"), "materialized artifact sha256")
        if _sha256(physical) != digest:
            raise ValueError(f"materialized artifact binding hash mismatch: {relative}")
        binding["relative_path"] = relative
        binding["size_bytes"] = physical.stat().st_size
        by_name[artifact_name] = binding

    # One index row represents one artifact slot.  Reclaimed non-witness pair
    # outputs are intentionally absent; retained-but-unmaterialized slots stay
    # visible with materialized=false.
    specs: list[tuple[str, str, Mapping[str, object], str, str, str]] = []
    for row in program_rows:
        specs.append(("program_root", "root_ir", row, "root_ir", "root_ir_path", "root_ir_sha256"))
    for row in single_rows:
        source_id = _text(row.get("row_id"))
        if _text(row.get("group_id")) == "Uall" and (
            _text(row.get("artifact_id")) or ("profile", source_id) in bindings
        ):
            specs.append(("profile", "single_pass_output", row, "output", "", ""))
    for row in pair_rows:
        source_id = _text(row.get("row_id"))
        retained = _text(row.get("cleanup_status")).startswith("retained_")
        if _text(row.get("group_id")) == "Uall" and (
            retained or ("pair", source_id) in bindings
        ):
            specs.extend(
                (
                    ("pair", "pair_ab_output", row, "AB", "ab_output_path", "ab_output_sha256"),
                    ("pair", "pair_ba_output", row, "BA", "ba_output_path", "ba_output_sha256"),
                )
            )
    for row in directional_rows:
        source_id = _text(row.get("row_id"))
        physical = bindings.get(("two_n_directional", source_id), {})
        if _text(row.get("merged_input_path")) or _text(row.get("merged_input_sha256")) or "merged_input" in physical:
            specs.append(("two_n_directional", "two_n_merged_input", row, "merged_input", "merged_input_path", "merged_input_sha256"))
        if (
            _text(row.get("cleanup_status")).startswith("retained_")
            or _text(row.get("second_output_path"))
            or _text(row.get("second_output_sha256"))
            or "second_round_output" in physical
        ):
            specs.append(("two_n_directional", "two_n_second_round_output", row, "second_round_output", "second_output_path", "second_output_sha256"))

    rows: list[dict[str, object]] = []
    consumed_bindings: set[tuple[str, str, str]] = set()
    seen_slots: set[tuple[str, str, str]] = set()
    for source_kind, artifact_kind, source, artifact_name, path_field, sha_field in sorted(
        specs, key=lambda item: (item[0], _text(item[2].get("row_id")), item[3])
    ):
        source_id = _nonempty(source.get("row_id"), "artifact source row_id")
        slot = (source_kind, source_id, artifact_name)
        if slot in seen_slots:
            raise ValueError(f"duplicate artifact slot: {slot}")
        physical = bindings.get((source_kind, source_id), {}).get(artifact_name)
        if physical is not None:
            relative_path = _text(physical.get("relative_path"))
            digest = _text(physical.get("sha256"))
            size_bytes = int(physical.get("size_bytes", 0))
            materialized = "true"
            available = "true"
            component_source = "checkpoint_materialized_binding"
            consumed_bindings.add(slot)
        else:
            claimed_materialized = (
                _text(source.get("artifact_materialized")).lower() == "true"
                if source_kind in {"profile", "pair"}
                else _text(source.get("second_output_materialized")).lower() == "true"
                if source_kind == "two_n_directional" and artifact_name == "second_round_output"
                else False
            )
            if claimed_materialized:
                raise ValueError(
                    "canonical row claims a materialized artifact without a checkpoint binding: "
                    f"{source_kind}:{source_id}:{artifact_name}"
                )
            relative_path = _relative_artifact_path(source.get(path_field, ""), root)
            digest = _text(source.get(sha_field, ""))
            size_bytes = 0
            materialized = "false"
            available = (
                "true"
                if digest
                else _text(source.get("artifact_available")).lower()
                if source_kind in {"profile", "pair"}
                else "false"
            )
            component_source = "canonical_row"
            if source_kind == "program_root":
                path = root / Path(relative_path)
                if not path.is_file() or _sha256(path) != digest:
                    raise ValueError(f"root IR artifact binding mismatch: {relative_path}")
                size_bytes = path.stat().st_size
                materialized = available = "true"
                component_source = "program_manifest"
        parent_id = _text(source.get("artifact_id")) or canonical_row_id(
            "artifact", manifest, source_kind, source_id
        )
        # Preserve the canonical source artifact identity verbatim so callers
        # can join source evidence to this index.  row_id and artifact_name in
        # provenance disambiguate multi-file AB/BA bundles.
        artifact_id = parent_id
        provenance = {
            "source_kind": source_kind,
            "source_table": (
                "program_manifest.csv" if source_kind == "program_root"
                else "single_pass_observations.csv" if source_kind == "profile"
                else "pair_observations.csv" if source_kind == "pair"
                else "advisor_2n_directional_results.csv"
            ),
            "cleanup_status": _text(source.get("cleanup_status")),
            "artifact_name": artifact_name,
            "component_source": component_source,
        }
        rows.append(
            {
                "row_id": canonical_row_id("artifact-index", manifest, *slot),
                "study_manifest_id": manifest,
                "artifact_id": artifact_id,
                "artifact_kind": artifact_kind,
                "stage": (
                    "program_prepare" if source_kind == "program_root"
                    else "single_pass" if source_kind == "profile"
                    else "pair_oracle" if source_kind == "pair"
                    else "advisor_2n"
                ),
                "group_id": _text(source.get("group_id")),
                "program_id": _text(source.get("program_id")),
                "action_id": _text(source.get("action_id")),
                "pair_id": source_id if source_kind == "pair" else "",
                "relative_path": relative_path,
                "sha256": digest,
                "size_bytes": size_bytes,
                "available": available if available in {"true", "false"} else "false",
                "materialized": materialized,
                "provenance": _canonical_json(provenance),
                "source_row_ids": _canonical_json([source_id]),
                "authority_granted": "false",
                "proved_commute": "false",
            }
        )
        seen_slots.add(slot)

    # Replay witnesses and any other checkpoint-validated physical artifact
    # have no compact canonical row.  Preserve them directly rather than drop
    # a validated materialization.
    for (source_kind, source_id), by_name in sorted(bindings.items()):
        for artifact_name, physical in sorted(by_name.items()):
            slot = (source_kind, source_id, artifact_name)
            if slot in consumed_bindings:
                continue
            rows.append(
                {
                    "row_id": canonical_row_id("artifact-index", manifest, *slot),
                    "study_manifest_id": manifest,
                    "artifact_id": canonical_row_id("artifact-slot", manifest, *slot),
                    "artifact_kind": "replay_artifact" if source_kind.startswith("replay_") else "checkpoint_artifact",
                    "stage": "false_authorization_replay" if source_kind.startswith("replay_") else "checkpoint",
                    "group_id": "",
                    "program_id": "",
                    "action_id": "",
                    "pair_id": source_id if source_kind.startswith("replay_") else "",
                    "relative_path": _text(physical.get("relative_path")),
                    "sha256": _text(physical.get("sha256")),
                    "size_bytes": int(physical.get("size_bytes", 0)),
                    "available": "true",
                    "materialized": "true",
                    "provenance": _canonical_json({"source_kind": source_kind, "artifact_name": artifact_name, "component_source": "checkpoint_materialized_binding"}),
                    "source_row_ids": _canonical_json([source_id]),
                    "authority_granted": "false",
                    "proved_commute": "false",
                }
            )
    return sorted(rows, key=lambda row: (_text(row["artifact_kind"]), _text(row["row_id"])))


def _relative_artifact_path(value: object, root: Path) -> str:
    text = _text(value)
    if not text:
        return ""
    supplied = Path(text)
    candidate = supplied if supplied.is_absolute() else root / supplied
    resolved = candidate.resolve(strict=False)
    _inside(root, resolved)
    return resolved.relative_to(root).as_posix()


def _common_relative_path(paths: Sequence[str]) -> str:
    if not paths:
        return ""
    split = [Path(path).parts for path in paths]
    common: list[str] = []
    for parts in zip(*split):
        if len(set(parts)) != 1:
            break
        common.append(parts[0])
    if len(paths) == 1:
        return Path(paths[0]).as_posix()
    return Path(*common).as_posix() if common else ""


def materialize_aggregate(
    *,
    out_dir: Path,
    study_manifest_id: str,
    group_actions: Mapping[str, Sequence[str]],
    tables: Mapping[str, Sequence[Mapping[str, object]]],
    materialized_artifact_bindings: Sequence[Mapping[str, object]] = (),
    isolation_root: Path | None = None,
    program_count: int | None = None,
    formal_scope_user_override: str = "",
) -> AggregateResult:
    """Write the complete aggregate/report inventory from frozen evidence.

    ``tables`` accepts only raw study tables registered in :mod:`schema`.
    Missing raw tables are represented by empty deterministic CSVs.  U14/U30
    views are derived only from frozen exact action IDs, never from a label or
    a result-dependent pass substitution.
    """

    manifest = _nonempty(study_manifest_id, "study_manifest_id")
    if program_count is not None and (
        not isinstance(program_count, int)
        or isinstance(program_count, bool)
        or program_count < 1
    ):
        raise ValueError("program_count metadata must be a positive integer when present")
    formal_scope_user_override = str(formal_scope_user_override).strip()
    if formal_scope_user_override:
        raise ValueError("formal_scope_user_override is not accepted at the aggregate boundary")
    root = (isolation_root or Path(__file__).resolve().parents[1]).resolve()
    target = Path(out_dir).resolve()
    _validate_output_target(root, target)
    groups = _validate_groups(group_actions)
    raw = _validate_raw_tables(tables, manifest)
    if isinstance(materialized_artifact_bindings, (str, bytes)) or not isinstance(
        materialized_artifact_bindings, Sequence
    ) or any(not isinstance(row, Mapping) for row in materialized_artifact_bindings):
        raise ValueError("materialized_artifact_bindings must be a sequence of mappings")
    artifact_bindings = tuple(dict(row) for row in materialized_artifact_bindings)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(tempfile.mkdtemp(
        prefix=f".{target.name}.aggregate-staging-", dir=target.parent
    ))
    stage_target = stage_root / target.name
    try:
        built = _build_aggregate(
            stage_target,
            root,
            target,
            manifest,
            groups,
            raw,
            artifact_bindings,
            program_count=program_count,
            formal_scope_user_override=formal_scope_user_override,
        )
        _validate_bundle_inventory(stage_target, built.files)
        target.mkdir(parents=True, exist_ok=True)
        _publish_bundle(stage_target, target)
        final_files = tuple(
            target / path.relative_to(stage_target) for path in built.files
        )
        return AggregateResult(
            target,
            target / "aggregate",
            target / "figures",
            tuple(sorted(final_files, key=lambda path: path.as_posix())),
        )
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def _build_aggregate(
    target: Path,
    root: Path,
    artifact_root: Path,
    manifest: str,
    groups: Mapping[str, tuple[str, ...]],
    raw: Mapping[str, list[dict[str, object]]],
    materialized_artifact_bindings: Sequence[Mapping[str, object]],
    *,
    program_count: int | None,
    formal_scope_user_override: str,
) -> AggregateResult:
    """Build a complete candidate bundle under a private sibling staging root."""

    _inside(root, artifact_root)
    aggregate_dir = target / "aggregate"
    figures_dir = target / "figures"
    _inside(root, aggregate_dir)
    _inside(root, figures_dir)
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    pair_rows = _derive_pair_views(raw["pair_observations.csv"], groups, manifest)
    single_rows = _derive_single_views(raw["single_pass_observations.csv"], groups, manifest)
    base_rows = dict(raw)
    base_rows["pair_observations.csv"] = pair_rows
    base_rows["single_pass_observations.csv"] = single_rows

    pair_confusion: list[dict[str, object]] = []
    pair_metrics: list[dict[str, object]] = []
    pair_pass_family: list[dict[str, object]] = []
    for group_id in _GROUP_ORDER:
        selected_pairs = [row for row in pair_rows if _text(row.get("group_id")) == group_id]
        selected_singles = [row for row in single_rows if _text(row.get("group_id")) == group_id]
        action_ids = groups[group_id]
        for view in ("all_successful", "active_active"):
            pair_confusion.extend(
                build_pair_confusion(
                    selected_pairs,
                    group=group_id,
                    activity_view=view,
                    study_manifest_id=manifest,
                )
            )
            metric_rows = build_pair_metric_rows(
                selected_pairs,
                group=group_id,
                activity_view=view,
                configured_action_count=len(action_ids),
                configured_action_ids=action_ids,
                single_pass_rows=selected_singles,
                study_manifest_id=manifest,
            )
            for row in metric_rows:
                if row["aggregation_scope"] in {"program_micro", "program_macro"}:
                    pair_metrics.append(row)
                else:
                    pair_pass_family.append(row)

    group_rows = list(raw["advisor_2n_group_results.csv"])
    directional_rows = list(raw["advisor_2n_directional_results.csv"])
    validation_rows = list(raw["advisor_2n_pair_validation.csv"])
    failure_rows = _unique_rows(
        [
            *raw["failure_ledger.csv"],
            *build_failure_ledger_rows(
                preflight_rows=raw["pass_preflight.csv"],
                group_rows=group_rows,
                pair_rows=raw["pair_observations.csv"],
                study_manifest_id=manifest,
            ),
        ],
        "failure ledger",
    )
    artifact_rows = _unique_rows(
        [
            *raw["artifact_index.csv"],
            *build_artifact_index_rows(
                program_rows=raw["program_manifest.csv"],
                single_rows=raw["single_pass_observations.csv"],
                pair_rows=raw["pair_observations.csv"],
                directional_rows=directional_rows,
                materialized_artifact_bindings=materialized_artifact_bindings,
                study_manifest_id=manifest,
                isolation_root=artifact_root,
            ),
        ],
        "artifact index",
    )
    two_n_metrics = _two_n_metrics(groups, group_rows, directional_rows, validation_rows, manifest)
    gate_funnel = _gate_funnel(two_n_metrics, manifest)
    false_authorizations = _false_authorizations(validation_rows, manifest)
    execution_costs = _execution_costs(
        groups,
        raw["pair_observations.csv"],
        raw["single_pass_observations.csv"],
        group_rows,
        validation_rows,
        artifact_rows,
        manifest,
    )
    limitations = _limitations(groups, group_rows, directional_rows, validation_rows, manifest)

    base_rows["pair_dynamic_confusion.csv"] = pair_confusion
    base_rows["pair_metrics_by_group.csv"] = pair_metrics
    base_rows["advisor_2n_metrics_by_group.csv"] = two_n_metrics
    base_rows["failure_ledger.csv"] = failure_rows
    base_rows["artifact_index.csv"] = artifact_rows

    written: list[Path] = []
    for table_name in _BASE_TABLE_ORDER:
        path = aggregate_dir / table_name
        write_csv(path, TABLE_FIELDS[table_name], base_rows[table_name], isolation_root=root)
        written.append(path)
    extras = {
        "pair_pass_family_metrics.csv": pair_pass_family,
        "two_n_gate_funnel.csv": gate_funnel,
        "two_n_false_authorizations.csv": false_authorizations,
        "execution_costs.csv": execution_costs,
        "study_limitations.csv": limitations,
        "source_index.csv": _source_index(raw, manifest),
    }
    for table_name in _DERIVED_TABLE_ORDER:
        path = aggregate_dir / table_name
        _write_fixed_csv(path, _DERIVED_TABLE_FIELDS[table_name], extras[table_name], root)
        written.append(path)

    figure_data = _figure_data(pair_metrics, two_n_metrics, validation_rows, execution_costs)
    for index, stem in enumerate(_FIGURE_STEMS):
        title, labels, series = figure_data[index]
        svg_path = figures_dir / f"{stem}.svg"
        png_path = figures_dir / f"{stem}.png"
        _write_svg(svg_path, title, labels, series, root)
        _write_png(png_path, title, labels, series, root)
        written.extend((svg_path, png_path))

    reports = _reports(
        manifest,
        pair_metrics,
        two_n_metrics,
        false_authorizations,
        program_count=program_count,
        formal_scope_user_override=formal_scope_user_override,
    )
    for filename, content in reports.items():
        validate_claim_text(content)
        path = aggregate_dir / filename
        _write_text(path, content, root)
        written.append(path)

    evidence_path = aggregate_dir / "evidence_manifest.json"
    inventory = [
        {
            "relative_path": path.relative_to(target).as_posix(),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(written, key=lambda item: item.relative_to(target).as_posix())
    ]
    evidence = {
        "schema_version": "advisor-pair-scale-aggregate-v1",
        "study_manifest_id": manifest,
        "program_count": program_count,
        "formal_scope_user_override": formal_scope_user_override,
        "authority_granted": False,
        "proved_commute": False,
        "aggregate_inventory_sha256": hashlib.sha256(
            _canonical_json(inventory).encode("utf-8")
        ).hexdigest(),
        "files": inventory,
    }
    _write_text(evidence_path, _canonical_json(evidence) + "\n", root)
    written.append(evidence_path)
    required_by_name = {path.name for path in written}
    if required_by_name != set(REQUIRED_AGGREGATE_FILES):
        raise AssertionError("aggregate inventory is incomplete or contains an unexpected file")
    return AggregateResult(target, aggregate_dir, figures_dir, tuple(sorted(written, key=lambda p: p.as_posix())))


def _validate_groups(group_actions: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    if set(group_actions) != set(_GROUP_ORDER):
        raise ValueError("group_actions must contain exactly U14, U30, and Uall")
    frozen: dict[str, tuple[str, ...]] = {}
    for group_id in _GROUP_ORDER:
        values = tuple(_nonempty(action, f"{group_id} action_id") for action in group_actions[group_id])
        if len(values) < 2 or len(set(values)) != len(values):
            raise ValueError(f"{group_id} must contain unique exact action IDs")
        frozen[group_id] = values
    if not set(frozen["U14"]).issubset(frozen["U30"]) or not set(frozen["U30"]).issubset(frozen["Uall"]):
        raise ValueError("pass groups must remain nested U14 subset U30 subset Uall")
    return frozen


def _validate_output_target(root: Path, target: Path) -> None:
    """Allow aggregate publication only under the two frozen output roots."""

    _inside(root, target)
    relative = target.relative_to(root)
    if len(relative.parts) < 2 or relative.parts[0] != "output" or relative.parts[1] not in {"smoke", "formal"}:
        raise ValueError("aggregate out_dir must remain under output/smoke or output/formal")


def _validate_bundle_inventory(stage_target: Path, files: Sequence[Path]) -> None:
    expected = {
        *(f"aggregate/{name}" for name in (*_BASE_TABLE_ORDER, *_DERIVED_TABLE_ORDER, *_REPORT_FILES)),
        *(f"figures/{stem}.svg" for stem in _FIGURE_STEMS),
        *(f"figures/{stem}.png" for stem in _FIGURE_STEMS),
    }
    actual = {path.relative_to(stage_target).as_posix() for path in files}
    if actual != expected:
        raise ValueError("fixed aggregate inventory mismatch before publish")
    manifest_path = stage_target / "aggregate" / "evidence_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_inventory = payload.get("aggregate_inventory_sha256")
    entries = payload.get("files")
    if not isinstance(expected_inventory, str) or len(expected_inventory) != 64 or not isinstance(entries, list):
        raise ValueError("evidence manifest lacks a validated aggregate inventory hash")
    if hashlib.sha256(_canonical_json(entries).encode("utf-8")).hexdigest() != expected_inventory:
        raise ValueError("evidence manifest aggregate inventory hash mismatch")
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("evidence manifest entry is malformed")
        relative = _nonempty(entry.get("relative_path"), "evidence relative_path")
        path = stage_target / relative
        if not path.is_file() or _sha256(path) != _nonempty(entry.get("sha256"), "evidence sha256"):
            raise ValueError("evidence manifest file hash mismatch")


def _publish_bundle(stage_target: Path, target: Path) -> None:
    """Swap fully validated aggregate/figure directories with rollback.

    The only visible mutations happen after validation.  If either directory
    move fails, any old completed directory is moved back before the error is
    surfaced.  Staging and backup roots are siblings of the frozen output root
    and are always removed by the caller.
    """

    backup_root = Path(tempfile.mkdtemp(prefix=f".{target.name}.aggregate-backup-", dir=target.parent))
    moved_old: list[str] = []
    published: list[str] = []
    try:
        for name in ("aggregate", "figures"):
            source = stage_target / name
            destination = target / name
            if destination.exists():
                os.replace(destination, backup_root / name)
                moved_old.append(name)
            os.replace(source, destination)
            published.append(name)
    except Exception:
        for name in reversed(published):
            destination = target / name
            staged = stage_target / name
            if destination.exists():
                os.replace(destination, staged)
        for name in reversed(moved_old):
            backup = backup_root / name
            if backup.exists():
                os.replace(backup, target / name)
        raise
    finally:
        shutil.rmtree(backup_root, ignore_errors=True)


def _validate_raw_tables(
    tables: Mapping[str, Sequence[Mapping[str, object]]], manifest: str
) -> dict[str, list[dict[str, object]]]:
    unexpected = set(tables) - set(TABLE_FIELDS)
    if unexpected:
        raise ValueError(f"unregistered raw evidence table(s): {sorted(unexpected)}")
    result: dict[str, list[dict[str, object]]] = {name: [] for name in _BASE_TABLE_ORDER}
    for table_name, rows in tables.items():
        if not isinstance(rows, Sequence):
            raise ValueError(f"{table_name} rows must be a sequence")
        seen: set[str] = set()
        expected_fields = set(TABLE_FIELDS[table_name])
        for original in rows:
            if not isinstance(original, Mapping):
                raise ValueError(f"{table_name} evidence row must be a mapping")
            row = dict(original)
            missing = sorted(expected_fields - set(row))
            extra = sorted(set(row) - expected_fields)
            if missing:
                raise ValueError(f"{table_name} missing required fields: {missing}")
            if extra:
                raise ValueError(f"{table_name} has unregistered fields: {extra}")
            require_authority_off(row)
            if _text(row.get("study_manifest_id")) != manifest:
                raise ValueError(f"{table_name} study_manifest_id mismatch")
            row_id = _nonempty(row.get("row_id"), f"{table_name} row_id")
            if row_id in seen:
                raise ValueError(f"duplicate row_id in {table_name}: {row_id}")
            for field, statuses in STATUS_VALUES_BY_TABLE.get(table_name, {}).items():
                if _text(row.get(field)) not in statuses:
                    raise ValueError(f"{table_name} invalid {field}")
            for field in _numeric_required_fields_for_row(table_name, row):
                _required_integer(row, field)
            if table_name == "advisor_2n_pair_validation.csv":
                _validate_false_authorization_literals(row)
            if table_name in {"single_pass_observations.csv", "pair_observations.csv"} and _text(row.get("group_id")) != "Uall":
                raise ValueError(f"{table_name} raw evidence must be the shared Uall profile/pair matrix")
            if table_name in {"single_pass_observations.csv", "pair_observations.csv"} and _text(row.get("cache_reused")).lower() not in {"true", "false"}:
                raise ValueError(f"{table_name} cache_reused must be true or false")
            seen.add(row_id)
            result[table_name].append(row)
    return result


def _numeric_required_fields_for_row(table_name: str, row: Mapping[str, object]) -> frozenset[str]:
    """Return per-table numeric requirements, preserving fixed-program semantics."""

    fields = _NUMERIC_REQUIRED_FIELDS.get(table_name, frozenset())
    # A frozen fixed selection has no reserve ordinal.  The column remains in
    # the stable manifest schema, while a reserve replacement must still carry
    # a finite rank for auditability.
    if table_name == "program_manifest.csv" and _text(row.get("selection_class")) == "fixed":
        return fields - {"reserve_rank"}
    if (
        table_name == "pass_groups.csv"
        and _text(row.get("group_id")) == "U14"
        and _text(row.get("selection_method")) == "exact_u14"
    ):
        return fields - {"selection_rank"}
    return fields


def _validate_false_authorization_literals(row: Mapping[str, object]) -> None:
    false_authorization = _text(row.get("false_authorization")).lower()
    stable = _text(row.get("stable_false_authorization")).lower()
    if false_authorization not in {"true", "false"} or stable not in {"true", "false"}:
        raise ValueError("false_authorization and stable_false_authorization must be true or false")
    if _text(row.get("validation_status")) == "false_authorization" and false_authorization != "true":
        raise ValueError("false_authorization validation requires false_authorization=true")
    if stable == "true":
        if false_authorization != "true" or _text(row.get("validation_status")) != "false_authorization":
            raise ValueError("stable_false_authorization=true requires a false_authorization validation")
        replay_statuses = (
            _text(row.get("worker_replay_status")),
            _text(row.get("external_opt_replay_status")),
            _text(row.get("two_n_replay_status")),
        )
        if any(status != "stable" for status in replay_statuses):
            raise ValueError("stable_false_authorization=true requires all replay statuses stable")


def _derive_pair_views(
    rows: Sequence[Mapping[str, object]], groups: Mapping[str, Sequence[str]], manifest: str
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for group_id in _GROUP_ORDER:
        derived = derive_group_pair_rows(rows, group=group_id, action_ids=groups[group_id], study_manifest_id=manifest)
        for row in derived:
            source_id = _nonempty(row.get("row_id"), "pair source row_id")
            copy = dict(row)
            copy["group_id"] = group_id
            copy["row_id"] = canonical_row_id(
                "pair_observation", manifest, group_id, source_id,
                _text(copy.get("program_id")), _text(copy.get("action_a_id")), _text(copy.get("action_b_id")),
            )
            result.append(copy)
    return _unique_rows(result, "derived pair")


def _derive_single_views(
    rows: Sequence[Mapping[str, object]], groups: Mapping[str, Sequence[str]], manifest: str
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for group_id in _GROUP_ORDER:
        allowed = set(groups[group_id])
        for row in rows:
            action_id = _text(row.get("action_id"))
            if action_id not in allowed:
                continue
            source_id = _nonempty(row.get("row_id"), "single-pass source row_id")
            copy = dict(row)
            copy["group_id"] = group_id
            copy["row_id"] = canonical_row_id(
                "single_pass_observation", manifest, group_id, source_id,
                _text(copy.get("program_id")), action_id,
            )
            result.append(copy)
    return _unique_rows(result, "derived single-pass")


def _two_n_metrics(
    groups: Mapping[str, Sequence[str]], group_rows: Sequence[Mapping[str, object]],
    directional_rows: Sequence[Mapping[str, object]], validation_rows: Sequence[Mapping[str, object]], manifest: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group_id in _GROUP_ORDER:
        group_source = _group_rows(group_rows, group_id)
        directional = _group_rows(directional_rows, group_id)
        pairs = _group_rows(validation_rows, group_id)
        program_ids = {_text(row.get("program_id")) for row in group_source if _text(row.get("program_id"))}
        configured = [_integer(row, "configured_n") for row in group_source]
        available = [row for row in directional if _text(row.get("directional_status")) in {"authorized_all_others", "rejected_effect_changed"}]
        authorized = [row for row in directional if _text(row.get("directional_status")) == "authorized_all_others"]
        rejected = [row for row in directional if _text(row.get("directional_status")) == "rejected_effect_changed"]
        pair_authorized = [row for row in pairs if _text(row.get("two_n_pair_status")) == "both_directions_authorized"]
        agreement = [row for row in pairs if _text(row.get("validation_status")) == "agree"]
        false_auth = [row for row in pairs if _text(row.get("validation_status")) == "false_authorization"]
        unavailable = [row for row in pairs if _text(row.get("validation_status")) in {"unavailable", "ground_truth_failed", "ground_truth_timeout", "ground_truth_unknown", "unknown"}]
        source_ids = _source_ids((*group_source, *directional, *pairs))
        rows.append({
            "row_id": canonical_row_id("advisor_2n_metrics", manifest, group_id),
            "study_manifest_id": manifest,
            "group_id": group_id,
            "program_count": len(program_ids),
            "configured_n_min": min(configured) if configured else len(groups[group_id]),
            "configured_n_max": max(configured) if configured else len(groups[group_id]),
            "successful_n_min": _min_or_zero(group_source, "successful_n"),
            "successful_n_max": _max_or_zero(group_source, "successful_n"),
            "active_n_total": _sum(group_source, "active_n"),
            "no_op_n_total": _sum(group_source, "no_op_n"),
            "failed_n_total": _sum(group_source, "failed_n"),
            "timeout_n_total": _sum(group_source, "timeout_n"),
            "round1_success_programs": _count(group_source, "round1_status", "complete"),
            "round1_success_rate": ratio(_count(group_source, "round1_status", "complete"), len(program_ids)),
            "disjoint_merge_applicable_programs": _count(group_source, "first_round_disjoint_status", "disjoint"),
            "disjoint_merge_applicability_rate": ratio(_count(group_source, "first_round_disjoint_status", "disjoint"), len(program_ids)),
            "all_n_merge_valid_programs": _count(group_source, "all_n_merge_status", "complete"),
            "all_n_merge_valid_rate": ratio(_count(group_source, "all_n_merge_status", "complete"), len(program_ids)),
            "all_n_second_round_complete_programs": _count(group_source, "all_n_second_round_status", "complete"),
            "all_n_second_round_completion_rate": ratio(_count(group_source, "all_n_second_round_status", "complete"), len(program_ids)),
            "complete_group_approval_programs": _count(group_source, "group_authorization_status", "authorized"),
            "complete_group_authorization_rate": ratio(_count(group_source, "group_authorization_status", "authorized"), len(program_ids)),
            "directional_authorized_count": len(authorized),
            "rejection_count": len(rejected),
            "directional_available_count": len(available),
            "pair_authorized_count": len(pair_authorized),
            "pair_agreement_count": len(agreement),
            "false_authorization_count": len(false_auth),
            "unavailable_count": len(unavailable),
            "directional_coverage": ratio(len(authorized), len(directional)),
            "directional_availability": ratio(len(available), len(directional)),
            "pair_coverage": ratio(len(pair_authorized), len(pairs)),
            "pair_agreement_rate": ratio(len(agreement), len(pairs)),
            "logical_pass_applications": _sum(group_source, "logical_pass_applications"),
            "physical_pass_invocations": _sum(group_source, "physical_pass_invocations"),
            "merge_helper_calls": _sum(group_source, "merge_helper_calls"),
            "merge_construction_time_ms": _sum(group_source, "merge_construction_time_ms"),
            "parse_time_ms": _sum(group_source, "parse_time_ms"),
            "verifier_time_ms": _sum(group_source, "verifier_time_ms"),
            "worker_time_ms": _sum(group_source, "worker_time_ms"),
            "replay_time_ms": _sum(pairs, "replay_time_ms"),
            "wall_time_ms": _sum(group_source, "wall_time_ms"),
            "source_row_ids": source_ids,
            "authority_granted": "false",
            "proved_commute": "false",
        })
    return rows


def _gate_funnel(rows: Sequence[Mapping[str, object]], manifest: str) -> list[dict[str, object]]:
    stages = (
        ("round1_success", "round1_success_programs"),
        ("disjoint_merge_applicable", "disjoint_merge_applicable_programs"),
        ("all_n_merge_valid", "all_n_merge_valid_programs"),
        ("all_n_second_round_complete", "all_n_second_round_complete_programs"),
        ("complete_group_authorized", "complete_group_approval_programs"),
    )
    output: list[dict[str, object]] = []
    for row in rows:
        for gate, field in stages:
            output.append({
                "row_id": canonical_row_id("two_n_gate", manifest, row["group_id"], gate),
                "study_manifest_id": manifest,
                "group_id": row["group_id"],
                "gate": gate,
                "program_count": row[field],
                "program_rate": ratio(_integer(row, field), _integer(row, "program_count")),
                "source_row_ids": row["source_row_ids"],
                "authority_granted": "false",
                "proved_commute": "false",
            })
    return output


def _false_authorizations(rows: Sequence[Mapping[str, object]], manifest: str) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        if _text(row.get("validation_status")) != "false_authorization":
            continue
        source_id = _nonempty(row.get("row_id"), "2N pair row_id")
        output.append({
            "row_id": canonical_row_id("two_n_false_authorization", manifest, source_id),
            "study_manifest_id": manifest,
            "group_id": _text(row.get("group_id")),
            "program_id": _text(row.get("program_id")),
            "action_a_id": _text(row.get("action_a_id")),
            "action_b_id": _text(row.get("action_b_id")),
            "two_n_pair_status": _text(row.get("two_n_pair_status")),
            "dynamic_result": _text(row.get("dynamic_result")),
            "validation_status": "false_authorization",
            "stable_false_authorization": _text(row.get("stable_false_authorization")),
            "worker_replay_status": _text(row.get("worker_replay_status")),
            "external_opt_replay_status": _text(row.get("external_opt_replay_status")),
            "two_n_replay_status": _text(row.get("two_n_replay_status")),
            "source_row_ids": source_id,
            "authority_granted": "false",
            "proved_commute": "false",
        })
    return output


def _execution_costs(
    groups: Mapping[str, Sequence[str]],
    raw_pairs: Sequence[Mapping[str, object]],
    raw_singles: Sequence[Mapping[str, object]],
    group_rows: Sequence[Mapping[str, object]],
    validation_rows: Sequence[Mapping[str, object]],
    artifacts: Sequence[Mapping[str, object]],
    manifest: str,
) -> list[dict[str, object]]:
    """Separate group-local logical views from deduplicated physical work."""

    result: list[dict[str, object]] = []
    reuse_count, reuse_ids = _reuse_events(artifacts)
    program_ids = {
        _text(row.get("program_id"))
        for row in (*raw_pairs, *raw_singles, *group_rows)
        if _text(row.get("program_id"))
    }
    total_programs = len(program_ids)
    for group_id in _GROUP_ORDER:
        n = len(groups[group_id])
        group_programs = {
            _text(row.get("program_id"))
            for row in (*raw_pairs, *group_rows)
            if _text(row.get("program_id"))
        }
        count = len(group_programs)
        pair_cost = pair_oracle_cost(n)
        result.extend((
            _cost_row(
                manifest, group_id, "pair_oracle", "per_group_logical_view", count, n,
                pair_cost["logical_first_round_applications"] * count,
                pair_cost["logical_second_stage_applications"] * count,
                physical_first="", physical_second="", physical_total="", cache_reuse="", reuse_event_count="", reuse_event_ids="", helper_calls=0,
                merge_time=0, parse_time=0, verifier_time=0, worker_time=0, replay_time=0, total_wall=0,
                source_ids="", provenance="logical_formula:N+N(N-1);frozen_group_manifest",
            ),
            _cost_row(
                manifest, group_id, "advisor_2n", "per_group_logical_view", count, n,
                n * count, n * count,
                physical_first="", physical_second="", physical_total="", cache_reuse="", reuse_event_count="", reuse_event_ids="", helper_calls=0,
                merge_time=0, parse_time=0, verifier_time=0, worker_time=0, replay_time=0, total_wall=0,
                source_ids="", provenance="logical_formula:N+N;frozen_group_manifest",
            ),
        ))

    # The Uall profile family is run once.  U14/U30 are only exact-ID views,
    # so neither may add another physical first-round charge here.
    unique_profiles = _unique_profile_rows(raw_singles)
    pair_second = _sum(raw_pairs, "second_stage_physical_pass_invocations")
    pair_first = _sum(unique_profiles, "physical_pass_invocations")
    pair_worker = _sum(raw_singles, "wall_time_ms") + _sum(raw_pairs, "wall_time_ms")
    result.append(_cost_row(
        manifest, "Uall", "pair_oracle", "experiment_wide_actual", total_programs, len(groups["Uall"]),
        pair_oracle_cost(len(groups["Uall"]))["logical_first_round_applications"] * total_programs,
        pair_oracle_cost(len(groups["Uall"]))["logical_second_stage_applications"] * total_programs,
        physical_first=pair_first, physical_second=pair_second, physical_total=pair_first + pair_second,
        cache_reuse=_sum_boolean(raw_singles, "cache_reused") + _sum_boolean(raw_pairs, "cache_reused"),
        reuse_event_count=reuse_count, reuse_event_ids=reuse_ids,
        helper_calls=0, merge_time=0, parse_time=0, verifier_time=0, worker_time=pair_worker, replay_time=0,
        total_wall=pair_worker, source_ids=_source_ids((*raw_singles, *raw_pairs)),
        provenance="single_pass_observations.csv:unique_Uall_profiles;pair_observations.csv:ordered_second_stage",
    ))

    # Each advisor group performs distinct merges/second rounds, but all reuse
    # the same Uall profile artifacts.  Subtract the group-local first-round
    # profile charge before summing its actual second-stage physical work.
    advisor_second = 0
    advisor_helper = 0
    advisor_merge = advisor_parse = advisor_verifier = advisor_worker = 0
    for group_id in _GROUP_ORDER:
        rows = _group_rows(group_rows, group_id)
        actual_second = sum(max(0, _integer(row, "physical_pass_invocations") - _group_first_for_program(raw_singles, set(groups[group_id]), _text(row.get("program_id")))) for row in rows)
        advisor_second += actual_second
        advisor_helper += _sum(rows, "merge_helper_calls")
        advisor_merge += _sum(rows, "merge_construction_time_ms")
        advisor_parse += _sum(rows, "parse_time_ms")
        advisor_verifier += _sum(rows, "verifier_time_ms")
        advisor_worker += _sum(rows, "worker_time_ms")
    replay_time = _sum(validation_rows, "replay_time_ms")
    # End-to-end group wall time is the frozen measurement.  Component clocks
    # may overlap, so they are reported separately and are never re-summed.
    total_wall = _sum(raw_singles, "wall_time_ms") + _sum(group_rows, "wall_time_ms") + replay_time
    result.append(_cost_row(
        manifest, "all_groups", "advisor_2n", "experiment_wide_actual", total_programs, len(groups["Uall"]),
        len(groups["Uall"]) * total_programs,
        sum(len(groups[group]) * total_programs for group in _GROUP_ORDER),
        physical_first=pair_first, physical_second=advisor_second, physical_total=pair_first + advisor_second,
        cache_reuse=_sum_boolean(raw_singles, "cache_reused"), reuse_event_count=reuse_count, reuse_event_ids=reuse_ids, helper_calls=advisor_helper,
        merge_time=advisor_merge, parse_time=advisor_parse, verifier_time=advisor_verifier,
        worker_time=advisor_worker, replay_time=replay_time, total_wall=total_wall,
        source_ids=_source_ids((*raw_singles, *group_rows, *validation_rows)),
        provenance="single_pass_observations.csv:unique_Uall_profiles;advisor_2n_group_results.csv:second_round_only;advisor_2n_pair_validation.csv:replay",
    ))
    return result


def _cost_row(
    manifest: str, group_id: str, method: str, scope: str, program_count: int, configured_n: int,
    logical_first: int, logical_second: int, *, physical_first: int | str, physical_second: int | str,
    physical_total: int | str, cache_reuse: int | str, reuse_event_count: int | str, reuse_event_ids: str, helper_calls: int, merge_time: int,
    parse_time: int, verifier_time: int, worker_time: int, replay_time: int, total_wall: int,
    source_ids: str, provenance: str,
) -> dict[str, object]:
    return {
        "row_id": canonical_row_id("execution_cost", manifest, group_id, method, scope),
        "study_manifest_id": manifest,
        "group_id": group_id,
        "method": method,
        "scope": scope,
        "program_count": program_count,
        "configured_n": configured_n,
        "logical_first_round_applications": logical_first,
        "logical_second_stage_applications": logical_second,
        "logical_total_pass_applications": logical_first + logical_second,
        "physical_first_round_invocations": physical_first,
        "physical_second_stage_invocations": physical_second,
        "physical_pass_invocations": physical_total,
        "cache_reuse_count": cache_reuse,
        "content_addressed_reuse_event_count": reuse_event_count,
        "reuse_event_source_row_ids": reuse_event_ids,
        "merge_helper_calls": helper_calls,
        "merge_construction_time_ms": merge_time,
        "parse_time_ms": parse_time,
        "verifier_time_ms": verifier_time,
        "worker_time_ms": worker_time,
        "replay_time_ms": replay_time,
        "total_wall_time_ms": total_wall,
        "provenance": provenance,
        "source_row_ids": source_ids,
        "authority_granted": "false",
        "proved_commute": "false",
    }


def _unique_profile_rows(rows: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    unique: dict[tuple[str, str], Mapping[str, object]] = {}
    for row in rows:
        key = (_text(row.get("program_id")), _text(row.get("action_id")))
        if not all(key):
            raise ValueError("single-pass physical provenance requires program_id and action_id")
        if key in unique:
            raise ValueError("duplicate raw Uall profile in physical provenance")
        unique[key] = row
    return [unique[key] for key in sorted(unique)]


def _reuse_events(rows: Sequence[Mapping[str, object]]) -> tuple[int, str]:
    """Read durable content-addressed view-reuse events from artifact index."""

    events = [row for row in rows if _text(row.get("artifact_kind")) == "reuse_event"]
    ids: list[str] = []
    for row in events:
        artifact_id = _nonempty(row.get("artifact_id"), "reuse event artifact_id")
        _nonempty(row.get("sha256"), "reuse event sha256")
        provenance = _nonempty(row.get("provenance"), "reuse event provenance")
        if "content_addressed" not in provenance:
            raise ValueError("reuse event provenance must identify content_addressed reuse")
        _nonempty(row.get("source_row_ids"), "reuse event source_row_ids")
        ids.append(artifact_id)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate content-addressed reuse event artifact_id")
    return len(ids), ",".join(sorted(ids))


def _group_first_for_program(rows: Sequence[Mapping[str, object]], actions: set[str], program_id: str) -> int:
    return sum(
        _integer(row, "physical_pass_invocations")
        for row in rows
        if _text(row.get("program_id")) == program_id and _text(row.get("action_id")) in actions
    )


def _limitations(
    groups: Mapping[str, Sequence[str]], group_rows: Sequence[Mapping[str, object]],
    directional_rows: Sequence[Mapping[str, object]], validation_rows: Sequence[Mapping[str, object]], manifest: str,
) -> list[dict[str, object]]:
    definitions = (
        (
            "runtime_budget_exceeded",
            "程序级 wall-time budget 超限；该程序保留在冻结分母中，仅构成 applicability/coverage limitation。",
        ),
        ("round1_precondition", "第一轮存在 failed、timeout 或 invalid 前提限制，仅说明适用性受限。"),
        ("first_round_overlap", "完整第一轮 patch family 非严格不重叠，仅说明直接合并不可用。"),
        ("direct_merge_unavailable", "结构化直接合并未定义或无效，仅说明覆盖范围受限。"),
        ("second_round_incomplete", "合并后的第二轮未完成，仅说明该程序/组不可适用。"),
        ("ground_truth_unavailable", "AB/BA ground truth 未决、失败或超时，不能用于授权结论。"),
    )
    output: list[dict[str, object]] = []
    for group_id in _GROUP_ORDER:
        source = _group_rows(group_rows, group_id)
        directional = _group_rows(directional_rows, group_id)
        validation = _group_rows(validation_rows, group_id)
        cases = {
            "runtime_budget_exceeded": [
                row
                for row in source
                if _text(row.get("fail_closed_reason")).startswith(
                    "runtime_budget_exceeded"
                )
            ],
            "round1_precondition": [
                row for row in source if _text(row.get("round1_status")) != "complete"
            ],
            "first_round_overlap": [
                row
                for row in source
                if _text(row.get("first_round_disjoint_status")) == "overlap"
            ],
            "direct_merge_unavailable": [
                row
                for row in source
                if _text(row.get("all_n_merge_status"))
                in {"direct_merge_not_defined", "merge_invalid", "timeout", "unknown"}
            ]
            + [
                row
                for row in directional
                if _text(row.get("directional_status")) == "direct_merge_not_defined"
            ],
            "second_round_incomplete": [
                row
                for row in source
                if _text(row.get("all_n_second_round_status")) != "complete"
            ],
            "ground_truth_unavailable": [
                row
                for row in validation
                if _text(row.get("validation_status"))
                in {
                    "ground_truth_failed",
                    "ground_truth_timeout",
                    "ground_truth_unknown",
                    "unavailable",
                    "unknown",
                }
            ],
        }
        denominators = {
            "runtime_budget_exceeded": source,
            "round1_precondition": source,
            "first_round_overlap": source,
            "direct_merge_unavailable": [*source, *directional],
            "second_round_incomplete": source,
            "ground_truth_unavailable": validation,
        }
        # ``groups`` is a frozen-input binding, not a source of observed case
        # counts.  Requiring the key makes accidental partial derivation fail.
        if group_id not in groups:
            raise ValueError(f"limitation derivation is missing frozen group {group_id}")
        for kind, interpretation in definitions:
            case_rows = cases[kind]
            denominator_rows = denominators[kind]
            output.append({
                "row_id": canonical_row_id("study_limitation", manifest, group_id, kind),
                "study_manifest_id": manifest,
                "group_id": group_id,
                "limitation_kind": kind,
                "case_count": len(case_rows),
                "denominator_count": len(denominator_rows),
                "source_row_ids": _source_ids(case_rows),
                "denominator_source_row_ids": _source_ids(denominator_rows),
                "interpretation_zh": interpretation,
                "authority_granted": "false",
                "proved_commute": "false",
            })
    return output


def _source_index(raw: Mapping[str, Sequence[Mapping[str, object]]], manifest: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for table_name in _BASE_TABLE_ORDER:
        for source in raw[table_name]:
            source_id = _nonempty(source.get("row_id"), "source row_id")
            rows.append({
                "row_id": canonical_row_id("source_index", manifest, table_name, source_id),
                "study_manifest_id": manifest,
                "source_table": table_name,
                "source_row_id": source_id,
                "source_kind": "raw_evidence",
                "authority_granted": "false",
                "proved_commute": "false",
            })
    return rows


def _reports(
    manifest: str, pair_metrics: Sequence[Mapping[str, object]], two_n_metrics: Sequence[Mapping[str, object]],
    false_authorizations: Sequence[Mapping[str, object]],
    *,
    program_count: int | None,
    formal_scope_user_override: str,
) -> dict[str, str]:
    pair_summary = _group_metric_lines(pair_metrics, ("precision", "coverage", "recall"))
    two_n_summary = _group_metric_lines(two_n_metrics, ("directional_coverage", "pair_coverage", "pair_agreement_rate"))
    stable = sum(
        _text(row.get("stable_false_authorization")).lower() == "true"
        for row in false_authorizations
    )
    program_label = "未声明" if program_count is None else str(program_count)
    scope_note = f"本次冻结范围元数据为 program_count={program_label}。"
    readme = f"""# 扩大 pair 与导师 2N 实验\n\n本目录包含冻结 manifest `{manifest}` 的原始证据汇总。{scope_note} 所有行保持 `authority_granted=false` 与 `proved_commute=false`。\n\n请先阅读 `expanded_pair_and_2n_report.md`，再依据 `evidence_manifest.json` 与 `source_index.csv` 回溯原始证据。\n"""
    report = f"""# 扩大 pair 与导师 2N 结果报告\n\n## 范围元数据\n\n{scope_note}\n\n## 结论边界\n\n本报告只给出实证证据，不给出定理、证明或授权。稳定的 `false_authorization` 会反驳所测试规则/域中的经验可靠性；合并或前提失败仅表示适用性和覆盖范围有限。即使观测 effect 精度较高，也不能把 observed effects 当作 authority。覆盖率低时仍需要残余动态 AB/BA。未观测到违反只构成经验性证据，不构成定理。\n\n## Pair 观察\n\n{pair_summary}\n\n## 导师 2N 门槛\n\n{two_n_summary}\n\n## 稳定反例\n\n稳定 `false_authorization` 数：{stable}。只有 Worker AB/BA、external `opt` 与 2N 重放均稳定的记录才应计入该数。\n\n## 成本口径\n\n完整 pair oracle 的逻辑工作是每程序 `N + N(N-1)` 次 pass application；它不是排列枚举。导师 2N 的理想上界按每程序最多 `N` 次第一轮与 `N` 次第二轮记录，实际物理调用、缓存复用、merge helper 和 wall-clock 见 `execution_costs.csv`。\n"""
    talking = f"""# 汇报话术\n\n- {scope_note}\n- 我们固定了相同 {program_label} 个程序与 U14、U30、Uall 三个 pass 组；Uall 按实际合格数量报告。\n- pair oracle 完整保留 all-successful 与 active-active 视图，并记录 commute、order_sensitive、failed、timeout、unknown。\n- 导师 2N 只在完整第一轮 patch family 严格非重叠时使用结构化直接合并；它没有顺序执行其余 pass，也没有文本 best-effort merge。\n- 稳定反例反驳经验可靠性；门槛失败是适用性限制，不是正确性结论。\n- 所有输出都是 evidence-only，未产生 certificate、batch/search authority 或 PROVED_COMMUTE。\n"""
    qa = """# 答辩问答\n\n## 为什么没有把 observed_disjoint 当作授权？\n\n它只是经验预测；AB/BA 的动态结果才是本实验的 ground truth 观察。\n\n## 合并失败是否说明 2N 不正确？\n\n不必然。严格直接合并无定义、第一轮重叠或第二轮失败首先是 applicability/coverage limitation。\n\n## 没有看到反例能否说明方法正确？\n\n不能。它只增加在冻结程序、pass 组和工具版本上的实证证据。\n\n## 为什么需要完整 pair oracle？\n\n它提供 all-successful AB/BA 对照、失败/超时桶以及对 2N 方向性结果的独立交叉检查。\n"""
    return {
        "README.md": readme,
        "expanded_pair_and_2n_report.md": report,
        "advisor_talking_points.md": talking,
        "advisor_q_and_a.md": qa,
    }


def _group_metric_lines(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> str:
    lines: list[str] = []
    for group_id in _GROUP_ORDER:
        candidates = [row for row in rows if _text(row.get("group_id")) == group_id]
        if not candidates:
            lines.append(f"- {group_id}: 无可用行。")
            continue
        row = next((item for item in candidates if _text(item.get("aggregation_scope")) == "program_micro"), candidates[0])
        values = "；".join(f"{field}={_text(row.get(field))}" for field in fields)
        lines.append(f"- {group_id}: {values}。")
    return "\n".join(lines)


def _figure_data(
    pair_metrics: Sequence[Mapping[str, object]], two_n_metrics: Sequence[Mapping[str, object]],
    validation_rows: Sequence[Mapping[str, object]], execution_costs: Sequence[Mapping[str, object]],
) -> tuple[tuple[str, tuple[str, ...], tuple[tuple[str, tuple[float, ...]], ...]], ...]:
    metric_by_group: dict[str, Mapping[str, object]] = {}
    for row in pair_metrics:
        if _text(row.get("aggregation_scope")) == "program_micro" and _text(row.get("denominator_scope")) == "all_successful":
            metric_by_group[_text(row.get("group_id"))] = row
    metric_labels = tuple(_GROUP_ORDER)
    precision = tuple(_decimal(metric_by_group.get(group, {}).get("precision", 0)) for group in metric_labels)
    coverage = tuple(_decimal(metric_by_group.get(group, {}).get("coverage", 0)) for group in metric_labels)
    two_n_by_group = {_text(row.get("group_id")): row for row in two_n_metrics}
    funnel_fields = (
        "round1_success_programs", "disjoint_merge_applicable_programs", "all_n_second_round_complete_programs", "complete_group_approval_programs",
    )
    funnel_labels = ("第一轮成功", "严格合并可用", "第二轮完成", "组授权")
    funnel = tuple(
        (group, tuple(float(_integer(two_n_by_group.get(group, {}), field)) for field in funnel_fields))
        for group in _GROUP_ORDER
    )
    validation_labels = ("AB/BA一致", "false_authorization", "不可用")
    agreement_by_group = []
    for group in _GROUP_ORDER:
        rows = _group_rows(validation_rows, group)
        agreement = sum(_text(row.get("validation_status")) == "agree" for row in rows)
        false_auth = sum(_text(row.get("validation_status")) == "false_authorization" for row in rows)
        agreement_by_group.append((group, (float(agreement), float(false_auth), float(len(rows) - agreement - false_auth))))
    cost_rows = [
        row for row in execution_costs
        if _text(row.get("scope")) in {"per_group_logical_view", "experiment_wide_actual"}
    ]
    method_zh = {"pair_oracle": "完整Pair", "advisor_2n": "导师2N"}
    scope_zh = {"per_group_logical_view": "组逻辑", "experiment_wide_actual": "实验实际"}
    group_zh = {"all_groups": "全部组"}
    cost_labels = tuple(
        "\n".join((
            group_zh.get(_text(row.get("group_id")), _text(row.get("group_id"))),
            method_zh.get(_text(row.get("method")), _text(row.get("method"))),
            scope_zh.get(_text(row.get("scope")), _text(row.get("scope"))),
        ))
        for row in cost_rows
    )
    logical = tuple(float(_integer(row, "logical_total_pass_applications")) for row in cost_rows)
    physical = tuple(_optional_float(row.get("physical_pass_invocations")) for row in cost_rows)
    wall = tuple(_optional_float(row.get("total_wall_time_ms")) for row in cost_rows)
    return (
        ("Pair 预测精度与覆盖率", metric_labels, (("精度", precision), ("覆盖率", coverage))),
        ("导师 2N 门槛适用性", funnel_labels, funnel),
        ("导师 2N 与 AB/BA 对照", validation_labels, tuple(agreement_by_group)),
        ("执行成本（分面：逻辑 / 物理 / 墙钟）", cost_labels, (("逻辑工作", logical), ("物理调用", physical), ("墙钟毫秒", wall))),
    )


def _write_svg(
    path: Path, title: str, labels: Sequence[str], series: Sequence[tuple[str, Sequence[float]]], root: Path,
) -> None:
    if title.startswith("执行成本（分面"):
        _write_cost_svg(path, title, labels, series, root)
        return
    width, height = 1200, 720
    left, bottom, top = 110, 610, 95
    plot_width, plot_height = 1010, bottom - top
    colors = ("#0B7285", "#F08C00", "#1971C2", "#D9480F", "#2B8A3E")
    max_value = max((value for _name, values in series for value in values), default=0.0)
    scale = plot_height / max(max_value, 1.0)
    group_width = plot_width / max(len(labels), 1)
    bar_width = max(12.0, min(54.0, group_width / max(len(series) + 1, 2)))
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;fill:#17202A}</style>',
        f'<text x="{width / 2}" y="48" text-anchor="middle" font-size="30" font-weight="700">{_xml(title)}</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{left + plot_width}" y2="{bottom}" stroke="#343A40" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#343A40" stroke-width="2"/>',
    ]
    for tick in range(5):
        y = bottom - plot_height * tick / 4
        value = max_value * tick / 4
        elements.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#DEE2E6"/>')
        elements.append(f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" font-size="15">{value:.1f}</text>')
    for index, label in enumerate(labels):
        center = left + group_width * (index + 0.5)
        elements.append(f'<text x="{center:.2f}" y="{bottom + 32}" text-anchor="middle" font-size="16">{_xml(label)}</text>')
        for s_index, (_name, values) in enumerate(series):
            value = values[index] if index < len(values) else 0.0
            h = max(0.0, value * scale)
            x = center - (len(series) - 1) * bar_width / 2 + s_index * bar_width
            elements.append(f'<rect x="{x:.2f}" y="{bottom - h:.2f}" width="{bar_width - 3:.2f}" height="{h:.2f}" fill="{colors[s_index % len(colors)]}"/>')
    for index, (name, _values) in enumerate(series):
        x = left + index * 180
        elements.append(f'<rect x="{x}" y="{height - 54}" width="18" height="18" fill="{colors[index % len(colors)]}"/>')
        elements.append(f'<text x="{x + 26}" y="{height - 39}" font-size="16">{_xml(name)}</text>')
    elements.append("</svg>\n")
    _write_text(path, "\n".join(elements), root)


def _write_png(
    path: Path, title: str, labels: Sequence[str], series: Sequence[tuple[str, Sequence[float]]], root: Path,
) -> None:
    if title.startswith("执行成本（分面"):
        _write_cost_png(path, title, labels, series, root)
        return
    _inside(root, path)
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as error:  # pragma: no cover - environment prerequisite
        raise RuntimeError("Pillow is required for deterministic PNG figures") from error
    width, height = 1200, 720
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if not font_path.is_file():  # pragma: no cover - Windows study environment
        raise RuntimeError("Chinese figure font is unavailable: C:\\Windows\\Fonts\\msyh.ttc")
    title_font = ImageFont.truetype(str(font_path), 30)
    label_font = ImageFont.truetype(str(font_path), 16)
    left, bottom, top = 110, 610, 95
    plot_width, plot_height = 1010, bottom - top
    colors = ((11, 114, 133), (240, 140, 0), (25, 113, 194), (217, 72, 15), (43, 138, 62))
    draw.text((width // 2, 25), title, font=title_font, fill=(23, 32, 42), anchor="ma")
    draw.line((left, top, left, bottom), fill=(52, 58, 64), width=2)
    draw.line((left, bottom, left + plot_width, bottom), fill=(52, 58, 64), width=2)
    max_value = max((value for _name, values in series for value in values), default=0.0)
    scale = plot_height / max(max_value, 1.0)
    group_width = plot_width / max(len(labels), 1)
    bar_width = max(12.0, min(54.0, group_width / max(len(series) + 1, 2)))
    for tick in range(5):
        y = bottom - plot_height * tick / 4
        value = max_value * tick / 4
        draw.line((left, y, left + plot_width, y), fill=(222, 226, 230), width=1)
        draw.text((left - 12, y), f"{value:.1f}", font=label_font, fill=(23, 32, 42), anchor="rm")
    for index, label in enumerate(labels):
        center = left + group_width * (index + 0.5)
        draw.text((center, bottom + 30), label, font=label_font, fill=(23, 32, 42), anchor="ma")
        for s_index, (_name, values) in enumerate(series):
            value = values[index] if index < len(values) else 0.0
            h = max(0.0, value * scale)
            x = center - (len(series) - 1) * bar_width / 2 + s_index * bar_width
            draw.rectangle((x, bottom - h, x + bar_width - 3, bottom), fill=colors[s_index % len(colors)])
    for index, (name, _values) in enumerate(series):
        x = left + index * 180
        draw.rectangle((x, height - 54, x + 18, height - 36), fill=colors[index % len(colors)])
        draw.text((x + 26, height - 45), name, font=label_font, fill=(23, 32, 42), anchor="lm")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False, compress_level=9)


def _write_cost_svg(
    path: Path, title: str, labels: Sequence[str], series: Sequence[tuple[str, Sequence[float]]], root: Path,
) -> None:
    """Draw three independently scaled cost facets; never mix their units."""

    width, height, top, bottom = 1200, 720, 100, 550
    colors = ("#0B7285", "#F08C00", "#1971C2")
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;fill:#17202A}</style>',
        f'<text x="600" y="48" text-anchor="middle" font-size="28" font-weight="700">{_xml(title)}</text>',
    ]
    for index, (name, values) in enumerate(series):
        left, panel_width = 55 + index * 385, 330
        right = left + panel_width
        max_value = max((value for value in values if value is not None), default=0.0)
        scale = (bottom - top) / max(max_value, 1.0)
        elements.extend((
            f'<text x="{left + panel_width / 2}" y="78" text-anchor="middle" font-size="20" font-weight="700">{_xml(name)}</text>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#343A40" stroke-width="2"/>',
            f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#343A40" stroke-width="2"/>',
        ))
        for tick in range(5):
            y = bottom - (bottom - top) * tick / 4
            elements.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#DEE2E6"/>')
            elements.append(f'<text x="{left - 6}" y="{y + 4:.2f}" text-anchor="end" font-size="11">{max_value * tick / 4:.1f}</text>')
        width_per = panel_width / max(len(labels), 1)
        bar_width = max(4.0, width_per * 0.58)
        for label_index, label in enumerate(labels):
            value = values[label_index] if label_index < len(values) else 0.0
            label_x = left + width_per * label_index + width_per / 2
            label_lines = str(label).split("\n")
            elements.append(f'<text x="{label_x:.2f}" y="{bottom + 14}" text-anchor="middle" font-size="10">')
            for line_index, line in enumerate(label_lines):
                dy = "0" if line_index == 0 else "11"
                elements.append(f'<tspan x="{label_x:.2f}" dy="{dy}">{_xml(line)}</tspan>')
            elements.append("</text>")
            if value is None:
                elements.append(f'<text x="{label_x:.2f}" y="{bottom - 6}" text-anchor="middle" font-size="10">N/A</text>')
                continue
            x = left + width_per * label_index + (width_per - bar_width) / 2
            h = max(0.0, value * scale)
            elements.append(f'<rect x="{x:.2f}" y="{bottom - h:.2f}" width="{bar_width:.2f}" height="{h:.2f}" fill="{colors[index % len(colors)]}"/>')
    elements.append("</svg>\n")
    _write_text(path, "\n".join(elements), root)


def _write_cost_png(
    path: Path, title: str, labels: Sequence[str], series: Sequence[tuple[str, Sequence[float]]], root: Path,
) -> None:
    _inside(root, path)
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as error:  # pragma: no cover - environment prerequisite
        raise RuntimeError("Pillow is required for deterministic PNG figures") from error
    width, height, top, bottom = 1200, 720, 100, 550
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if not font_path.is_file():  # pragma: no cover - Windows study environment
        raise RuntimeError("Chinese figure font is unavailable: C:\\Windows\\Fonts\\msyh.ttc")
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.truetype(str(font_path), 28)
    label_font = ImageFont.truetype(str(font_path), 11)
    panel_font = ImageFont.truetype(str(font_path), 20)
    colors = ((11, 114, 133), (240, 140, 0), (25, 113, 194))
    draw.text((600, 25), title, font=title_font, fill=(23, 32, 42), anchor="ma")
    for index, (name, values) in enumerate(series):
        left, panel_width = 55 + index * 385, 330
        right = left + panel_width
        max_value = max((value for value in values if value is not None), default=0.0)
        scale = (bottom - top) / max(max_value, 1.0)
        draw.text((left + panel_width / 2, 78), name, font=panel_font, fill=(23, 32, 42), anchor="ms")
        draw.line((left, top, left, bottom), fill=(52, 58, 64), width=2)
        draw.line((left, bottom, right, bottom), fill=(52, 58, 64), width=2)
        for tick in range(5):
            y = bottom - (bottom - top) * tick / 4
            draw.line((left, y, right, y), fill=(222, 226, 230), width=1)
            draw.text((left - 6, y), f"{max_value * tick / 4:.1f}", font=label_font, fill=(23, 32, 42), anchor="rm")
        width_per = panel_width / max(len(labels), 1)
        bar_width = max(4.0, width_per * 0.58)
        for label_index, label in enumerate(labels):
            value = values[label_index] if label_index < len(values) else 0.0
            label_x = left + width_per * label_index + width_per / 2
            draw.multiline_text(
                (label_x, bottom + 10),
                str(label),
                font=label_font,
                fill=(23, 32, 42),
                anchor="ma",
                align="center",
                spacing=0,
            )
            if value is None:
                draw.text((label_x, bottom - 6), "N/A", font=label_font, fill=(23, 32, 42), anchor="ms")
                continue
            x = left + width_per * label_index + (width_per - bar_width) / 2
            h = max(0.0, value * scale)
            draw.rectangle((x, bottom - h, x + bar_width, bottom), fill=colors[index % len(colors)])
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False, compress_level=9)


def _write_fixed_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]], root: Path) -> None:
    _inside(root, path)
    if len(fields) != len(set(fields)) or fields[:2] != ("row_id", "study_manifest_id"):
        raise ValueError("derived CSV schema must start with deterministic evidence identity")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for original in rows:
        require_authority_off(original)
        row_id = _nonempty(original.get("row_id"), "derived row_id")
        _nonempty(original.get("study_manifest_id"), "derived study_manifest_id")
        if row_id in seen:
            raise ValueError(f"duplicate derived row_id: {row_id}")
        seen.add(row_id)
        normalized.append({field: _text(original.get(field)) for field in fields})
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(sorted(normalized, key=lambda row: (row["row_id"], *(row[field] for field in fields[1:]))))
    _write_text(path, stream.getvalue(), root)


def _write_text(path: Path, content: str, root: Path) -> None:
    _inside(root, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def _inside(root: Path, path: Path) -> None:
    try:
        path.resolve().relative_to(root)
    except ValueError as error:
        raise ValueError(f"aggregate target escapes isolation root: {path}") from error


def _group_rows(rows: Sequence[Mapping[str, object]], group_id: str) -> list[Mapping[str, object]]:
    return [row for row in rows if _text(row.get("group_id")) == group_id]


def _unique_rows(rows: Sequence[dict[str, object]], label: str) -> list[dict[str, object]]:
    seen: set[str] = set()
    output: list[dict[str, object]] = []
    for row in rows:
        row_id = _nonempty(row.get("row_id"), f"{label} row_id")
        if row_id in seen:
            raise ValueError(f"duplicate {label} row_id: {row_id}")
        seen.add(row_id)
        output.append(row)
    return output


def _source_ids(rows: Iterable[Mapping[str, object]]) -> str:
    return ",".join(sorted({_text(row.get("row_id")) for row in rows if _text(row.get("row_id"))}))


def _count(rows: Iterable[Mapping[str, object]], field: str, value: str) -> int:
    return sum(_text(row.get(field)) == value for row in rows)


def _sum(rows: Iterable[Mapping[str, object]], field: str) -> int:
    return sum(_integer(row, field) for row in rows)


def _sum_boolean(rows: Iterable[Mapping[str, object]], field: str) -> int:
    return sum(_text(row.get(field)).lower() == "true" for row in rows)


def _min_or_zero(rows: Sequence[Mapping[str, object]], field: str) -> int:
    return min((_integer(row, field) for row in rows), default=0)


def _max_or_zero(rows: Sequence[Mapping[str, object]], field: str) -> int:
    return max((_integer(row, field) for row in rows), default=0)


def _integer(row: Mapping[str, object], field: str) -> int:
    raw = row.get(field, 0)
    if raw in (None, ""):
        return 0
    try:
        value = Decimal(str(raw))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{field} must be a finite non-negative integer") from error
    if not value.is_finite() or value < 0 or value != value.to_integral_value():
        raise ValueError(f"{field} must be a finite non-negative integer")
    return int(value)


def _required_integer(row: Mapping[str, object], field: str) -> int:
    if row.get(field) in (None, ""):
        raise ValueError(f"{field} must be a finite non-negative integer")
    return _integer(row, field)


def _decimal(value: object) -> float:
    if value in (None, ""):
        return 0.0
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return 0.0
    return float(result) if result.is_finite() and result >= 0 else 0.0


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return _decimal(value)


def _nonempty(value: object, name: str) -> str:
    text = _text(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _text(value: object) -> str:
    return "" if value is None else str(value)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _xml(value: str) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
