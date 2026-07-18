"""Closed vocabularies and deterministic CSV schemas for the isolated study.

This module is deliberately independent of Phasebatch correctness-authority
code.  Rows produced by the study are evidence only and must keep both
authority columns explicitly false.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping


OBSERVED_BUCKETS = (
    "observed_disjoint",
    "observed_overlap",
    "observed_unknown",
)

DYNAMIC_BUCKETS = (
    "commute",
    "order_sensitive",
    "failed",
    "timeout",
    "unknown",
)

TWO_N_DIRECTIONAL_STATUSES = (
    "authorized_all_others",
    "rejected_effect_changed",
    "round1_precondition_failed",
    "direct_merge_not_defined",
    "merge_invalid",
    "second_round_failed",
    "timeout",
    "unknown",
)

SINGLE_PASS_STATUSES = (
    "success",
    "invalid",
    "error",
    "timeout",
)

ACTIVITY_STATUSES = (
    "active",
    "no_op",
    "unknown",
)

ROOT_ACTIVITY_CLASSES = (
    "active_active",
    "active_noop",
    "noop_noop",
    "unknown",
)

TWO_N_PAIR_STATUSES = (
    "both_directions_authorized",
    "one_direction_only",
    "both_rejected",
    "group_precondition_unavailable",
)

PAIR_STAGE_STATUSES = (
    *SINGLE_PASS_STATUSES,
    "not_run",
    "unknown",
)

VERIFIER_STATUSES = (
    "success",
    "invalid",
    "not_run",
    "unknown",
)

TWO_N_ROUND1_STATUSES = (
    "complete",
    "round1_precondition_failed",
    "timeout",
    "unknown",
)

TWO_N_DISJOINT_STATUSES = (
    "disjoint",
    "overlap",
    "unknown",
)

TWO_N_MERGE_STATUSES = (
    "complete",
    "direct_merge_not_defined",
    "merge_invalid",
    "timeout",
    "unknown",
)

TWO_N_SECOND_ROUND_STATUSES = (
    "complete",
    "second_round_failed",
    "timeout",
    "unknown",
)

TWO_N_GROUP_AUTHORIZATION_STATUSES = (
    "authorized",
    "rejected",
    "group_precondition_unavailable",
    "unknown",
)

TWO_N_PAIR_VALIDATION_STATUSES = (
    "agree",
    "false_authorization",
    "ground_truth_failed",
    "ground_truth_timeout",
    "ground_truth_unknown",
    "unavailable",
    "unknown",
)

REPLAY_STATUSES = (
    "not_required",
    "stable",
    "nondeterministic",
    "mismatch",
    "failed",
    "timeout",
    "unavailable",
    "unknown",
)

FAILURE_LEDGER_STATUSES = (
    "error",
    "failed",
    "invalid",
    "timeout",
    "unavailable",
    "nondeterministic",
    "unresolved",
    "unknown",
)


def _evidence_fields(*fields: str) -> tuple[str, ...]:
    return (
        "row_id",
        "study_manifest_id",
        *fields,
        "authority_granted",
        "proved_commute",
    )


PROGRAM_MANIFEST_FIELDS = _evidence_fields(
    "program_id",
    "selection_order",
    "selection_class",
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
    "reserve_rank",
    "replacement_for_program_id",
)


PASS_INVENTORY_FIELDS = _evidence_fields(
    "action_id",
    "action_sha256",
    "canonical_action_record",
    "name",
    "pipeline",
    "registry_section",
    "parameter_binding",
    "policy_candidate",
    "policy_reason",
)

PASS_PREFLIGHT_FIELDS = _evidence_fields(
    "program_id",
    "action_id",
    "repetition",
    "execution_status",
    "verifier_status",
    "output_hard_state_id",
    "output_sha256",
    "deterministic",
    "eligible",
    "exclusion_reason",
    "logical_pass_applications",
    "physical_pass_invocations",
    "artifact_id",
    "command_sha256",
    "stderr_sha256",
)

PASS_GROUP_FIELDS = _evidence_fields(
    "group_id",
    "group_sha256",
    "group_size",
    "action_id",
    "action_order",
    "selection_method",
    "selection_rank",
    "selection_seed",
)

SINGLE_PASS_OBSERVATION_FIELDS = _evidence_fields(
    "group_id",
    "program_id",
    "action_id",
    "execution_status",
    "root_hard_state_id",
    "output_hard_state_id",
    "activity_status",
    "changed_functions_json",
    "changed_blocks_json",
    "changed_module_regions_json",
    "verifier_status",
    "logical_pass_applications",
    "physical_pass_invocations",
    "cache_reused",
    "artifact_available",
    "artifact_materialized",
    "artifact_id",
    "fail_closed_reason",
    "command_sha256",
    "stderr_sha256",
    "wall_time_ms",
)

PAIR_OBSERVATION_FIELDS = _evidence_fields(
    "group_id",
    "program_id",
    "action_a_id",
    "action_b_id",
    "root_activity_class",
    "observed_relation",
    "program_family",
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
    "ab_output_path",
    "ab_output_sha256",
    "ab_verifier_status",
    "ab_stderr_sha256",
    "ba_status",
    "ba_hard_state_id",
    "ba_output_path",
    "ba_output_sha256",
    "ba_verifier_status",
    "ba_stderr_sha256",
    "dynamic_result",
    "second_stage_logical_pass_applications",
    "second_stage_physical_pass_invocations",
    "total_logical_pass_applications",
    "total_physical_pass_invocations",
    "cache_reused",
    "artifact_available",
    "artifact_materialized",
    "cleanup_status",
    "artifact_id",
    "fail_closed_reason",
    "command_sha256",
    "stderr_sha256",
    "wall_time_ms",
)

PAIR_DYNAMIC_CONFUSION_FIELDS = _evidence_fields(
    "group_id",
    "denominator_scope",
    "observed_relation",
    "dynamic_result",
    "pair_count",
    "program_count",
    "source_row_ids",
)

PAIR_METRICS_FIELDS = _evidence_fields(
    "group_id",
    "denominator_scope",
    "aggregation_scope",
    "program_family",
    "pass_pair_id",
    "action_a_id",
    "action_b_id",
    "program_count",
    "pair_row_count",
    "successful_pair_count",
    "observed_disjoint_count",
    "false_authorization_count",
    "commuting_pair_count",
    "observed_disjoint_commuting_count",
    "precision",
    "false_authorization_rate",
    "coverage",
    "recall",
    "count_weighted_precision",
    "count_weighted_coverage",
    "logical_cost_weighted_precision",
    "logical_cost_weighted_coverage",
    "physical_cost_weighted_precision",
    "physical_cost_weighted_coverage",
    "wall_time_weighted_precision",
    "wall_time_weighted_coverage",
    "program_macro_precision_mean",
    "program_macro_coverage_mean",
    "program_macro_recall_mean",
    "failed_count",
    "timeout_count",
    "unknown_count",
    "logical_pass_applications",
    "physical_pass_invocations",
    "wall_time_ms",
    "source_row_ids",
)

ADVISOR_2N_GROUP_FIELDS = _evidence_fields(
    "group_id",
    "program_id",
    "configured_n",
    "successful_n",
    "active_n",
    "no_op_n",
    "failed_n",
    "timeout_n",
    "round1_status",
    "first_round_disjoint_status",
    "all_n_merge_status",
    "all_n_second_round_status",
    "group_authorization_status",
    "directional_authorized_count",
    "directional_unavailable_count",
    "logical_pass_applications",
    "physical_pass_invocations",
    "merge_helper_calls",
    "merge_construction_time_ms",
    "parse_time_ms",
    "verifier_time_ms",
    "worker_time_ms",
    "replay_time_ms",
    "fail_closed_reason",
    "source_row_ids",
    "wall_time_ms",
)

ADVISOR_2N_DIRECTIONAL_FIELDS = _evidence_fields(
    "group_id",
    "program_id",
    "action_id",
    "directional_status",
    "first_round_status",
    "first_round_effect_sha256",
    "merged_input_status",
    "merged_input_hard_state_id",
    "merged_input_sha256",
    "merged_input_path",
    "second_round_status",
    "second_round_effect_sha256",
    "second_output_path",
    "second_output_sha256",
    "second_output_materialized",
    "other_contributions_preserved",
    "verifier_status",
    "logical_pass_applications",
    "physical_pass_invocations",
    "merge_helper_calls",
    "artifact_id",
    "cleanup_status",
    "fail_closed_reason",
    "command_sha256",
    "stderr_sha256",
    "wall_time_ms",
)

ADVISOR_2N_PAIR_FIELDS = _evidence_fields(
    "group_id",
    "program_id",
    "action_a_id",
    "action_b_id",
    "action_a_directional_status",
    "action_b_directional_status",
    "two_n_pair_status",
    "pair_observation_row_id",
    "dynamic_result",
    "validation_status",
    "false_authorization",
    "stable_false_authorization",
    "worker_replay_status",
    "external_opt_replay_status",
    "two_n_replay_status",
    "replay_artifact_id",
    "replay_time_ms",
    "fail_closed_reason",
    "source_row_ids",
)

ADVISOR_2N_METRICS_FIELDS = _evidence_fields(
    "group_id",
    "program_count",
    "configured_n_min",
    "configured_n_max",
    "successful_n_min",
    "successful_n_max",
    "active_n_total",
    "no_op_n_total",
    "failed_n_total",
    "timeout_n_total",
    "round1_success_programs",
    "round1_success_rate",
    "disjoint_merge_applicable_programs",
    "disjoint_merge_applicability_rate",
    "all_n_merge_valid_programs",
    "all_n_merge_valid_rate",
    "all_n_second_round_complete_programs",
    "all_n_second_round_completion_rate",
    "complete_group_approval_programs",
    "complete_group_authorization_rate",
    "directional_authorized_count",
    "rejection_count",
    "directional_available_count",
    "pair_authorized_count",
    "pair_agreement_count",
    "false_authorization_count",
    "unavailable_count",
    "directional_coverage",
    "directional_availability",
    "pair_coverage",
    "pair_agreement_rate",
    "logical_pass_applications",
    "physical_pass_invocations",
    "merge_helper_calls",
    "merge_construction_time_ms",
    "parse_time_ms",
    "verifier_time_ms",
    "worker_time_ms",
    "replay_time_ms",
    "wall_time_ms",
    "source_row_ids",
)

FAILURE_LEDGER_FIELDS = _evidence_fields(
    "stage",
    "group_id",
    "program_id",
    "action_id",
    "pair_id",
    "failure_kind",
    "status",
    "reason",
    "command",
    "command_sha256",
    "stderr_path",
    "stderr_sha256",
    "artifact_id",
    "source_row_ids",
)

ARTIFACT_INDEX_FIELDS = _evidence_fields(
    "artifact_id",
    "artifact_kind",
    "stage",
    "group_id",
    "program_id",
    "action_id",
    "pair_id",
    "relative_path",
    "sha256",
    "size_bytes",
    "available",
    "materialized",
    "provenance",
    "source_row_ids",
)


TABLE_FIELDS = MappingProxyType(
    {
        "program_manifest.csv": PROGRAM_MANIFEST_FIELDS,
        "pass_inventory.csv": PASS_INVENTORY_FIELDS,
        "pass_preflight.csv": PASS_PREFLIGHT_FIELDS,
        "pass_groups.csv": PASS_GROUP_FIELDS,
        "single_pass_observations.csv": SINGLE_PASS_OBSERVATION_FIELDS,
        "pair_observations.csv": PAIR_OBSERVATION_FIELDS,
        "pair_dynamic_confusion.csv": PAIR_DYNAMIC_CONFUSION_FIELDS,
        "pair_metrics_by_group.csv": PAIR_METRICS_FIELDS,
        "advisor_2n_group_results.csv": ADVISOR_2N_GROUP_FIELDS,
        "advisor_2n_directional_results.csv": ADVISOR_2N_DIRECTIONAL_FIELDS,
        "advisor_2n_pair_validation.csv": ADVISOR_2N_PAIR_FIELDS,
        "advisor_2n_metrics_by_group.csv": ADVISOR_2N_METRICS_FIELDS,
        "failure_ledger.csv": FAILURE_LEDGER_FIELDS,
        "artifact_index.csv": ARTIFACT_INDEX_FIELDS,
    }
)


STATUS_VALUES_BY_TABLE = MappingProxyType(
    {
        "program_manifest.csv": MappingProxyType(
            {
                "compile_status": PAIR_STAGE_STATUSES,
            }
        ),
        "pass_preflight.csv": MappingProxyType(
            {
                "execution_status": SINGLE_PASS_STATUSES,
                "verifier_status": VERIFIER_STATUSES,
            }
        ),
        "single_pass_observations.csv": MappingProxyType(
            {
                "execution_status": SINGLE_PASS_STATUSES,
                "activity_status": ACTIVITY_STATUSES,
                "verifier_status": VERIFIER_STATUSES,
            }
        ),
        "pair_observations.csv": MappingProxyType(
            {
                "root_activity_class": ROOT_ACTIVITY_CLASSES,
                "observed_relation": OBSERVED_BUCKETS,
                "a_status": PAIR_STAGE_STATUSES,
                "b_status": PAIR_STAGE_STATUSES,
                "ab_status": PAIR_STAGE_STATUSES,
                "ba_status": PAIR_STAGE_STATUSES,
                "ab_verifier_status": VERIFIER_STATUSES,
                "ba_verifier_status": VERIFIER_STATUSES,
                "dynamic_result": DYNAMIC_BUCKETS,
            }
        ),
        "pair_dynamic_confusion.csv": MappingProxyType(
            {
                "observed_relation": OBSERVED_BUCKETS,
                "dynamic_result": DYNAMIC_BUCKETS,
            }
        ),
        "advisor_2n_group_results.csv": MappingProxyType(
            {
                "round1_status": TWO_N_ROUND1_STATUSES,
                "first_round_disjoint_status": TWO_N_DISJOINT_STATUSES,
                "all_n_merge_status": TWO_N_MERGE_STATUSES,
                "all_n_second_round_status": TWO_N_SECOND_ROUND_STATUSES,
                "group_authorization_status": TWO_N_GROUP_AUTHORIZATION_STATUSES,
            }
        ),
        "advisor_2n_directional_results.csv": MappingProxyType(
            {
                "directional_status": TWO_N_DIRECTIONAL_STATUSES,
                "first_round_status": PAIR_STAGE_STATUSES,
                "merged_input_status": TWO_N_MERGE_STATUSES,
                "second_round_status": PAIR_STAGE_STATUSES,
                "verifier_status": VERIFIER_STATUSES,
            }
        ),
        "advisor_2n_pair_validation.csv": MappingProxyType(
            {
                "action_a_directional_status": TWO_N_DIRECTIONAL_STATUSES,
                "action_b_directional_status": TWO_N_DIRECTIONAL_STATUSES,
                "two_n_pair_status": TWO_N_PAIR_STATUSES,
                "dynamic_result": DYNAMIC_BUCKETS,
                "validation_status": TWO_N_PAIR_VALIDATION_STATUSES,
                "worker_replay_status": REPLAY_STATUSES,
                "external_opt_replay_status": REPLAY_STATUSES,
                "two_n_replay_status": REPLAY_STATUSES,
            }
        ),
        "failure_ledger.csv": MappingProxyType(
            {
                "status": FAILURE_LEDGER_STATUSES,
            }
        ),
    }
)

ISOLATION_ROOT = Path(__file__).resolve().parents[1]

# Descriptive aliases make downstream modules readable while preserving one
# canonical tuple object for each table.
PASS_GROUPS_FIELDS = PASS_GROUP_FIELDS
PAIR_CONFUSION_FIELDS = PAIR_DYNAMIC_CONFUSION_FIELDS
TWO_N_GROUP_FIELDS = ADVISOR_2N_GROUP_FIELDS
TWO_N_DIRECTIONAL_FIELDS = ADVISOR_2N_DIRECTIONAL_FIELDS
TWO_N_PAIR_VALIDATION_FIELDS = ADVISOR_2N_PAIR_FIELDS


def canonical_row_id(prefix: str, *parts: object) -> str:
    """Return a stable, order-sensitive row identifier."""

    payload = json.dumps(
        [str(part) for part in parts],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{prefix}-{digest}"


def require_authority_off(row: Mapping[str, object]) -> None:
    """Reject any row that does not explicitly keep study authority off."""

    if str(row.get("authority_granted", "")).lower() != "false":
        raise ValueError("advisor study must keep authority_granted=false")
    if str(row.get("proved_commute", "")).lower() != "false":
        raise ValueError("advisor study must keep proved_commute=false")


def write_csv(
    path: Path,
    fields: tuple[str, ...],
    rows: Iterable[Mapping[str, object]],
    *,
    isolation_root: Path | None = None,
) -> None:
    """Write deterministic UTF-8/LF CSV bytes using a frozen field order."""

    if len(fields) != len(set(fields)):
        raise ValueError("CSV fields must be unique")

    root = (isolation_root or ISOLATION_ROOT).resolve()
    target = path.resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"CSV target must remain inside isolation root: {root}"
        ) from error

    expected_fields = TABLE_FIELDS.get(path.name)
    if expected_fields is None:
        raise ValueError(f"unregistered CSV table: {path.name}")
    if fields != expected_fields:
        raise ValueError(f"CSV fields do not match frozen schema for {path.name}")

    status_rules = STATUS_VALUES_BY_TABLE.get(path.name, {})

    normalized: list[dict[str, str]] = []
    requires_authority_off = {
        "authority_granted",
        "proved_commute",
    }.issubset(fields)
    seen_row_ids: set[str] = set()
    for row in rows:
        if requires_authority_off:
            require_authority_off(row)
        row_id = str(row.get("row_id", ""))
        study_manifest_id = str(row.get("study_manifest_id", ""))
        if not row_id.strip():
            raise ValueError("evidence row requires non-empty row_id")
        if not study_manifest_id.strip():
            raise ValueError("evidence row requires non-empty study_manifest_id")
        if row_id in seen_row_ids:
            raise ValueError(f"duplicate row_id: {row_id}")
        seen_row_ids.add(row_id)
        for field, allowed in status_rules.items():
            value = str(row.get(field, ""))
            if value not in allowed:
                allowed_text = ", ".join(allowed)
                raise ValueError(
                    f"invalid {field}={value!r} for {path.name}; "
                    f"expected one of: {allowed_text}"
                )
        normalized.append({field: str(row.get(field, "")) for field in fields})

    if "row_id" in fields:
        ordered = sorted(
            normalized,
            key=lambda row: (
                row["row_id"],
                *(row[field] for field in fields if field != "row_id"),
            ),
        )
    else:
        ordered = sorted(
            normalized,
            key=lambda row: tuple(row[field] for field in fields),
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(ordered)
