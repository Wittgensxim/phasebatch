"""Deterministic, evidence-only aggregation for the expanded pair oracle.

This module deliberately does not grant authority.  ``observed_disjoint`` is
only an empirical prediction which is checked against the AB/BA outcome.
"""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from typing import Iterable, Mapping, Sequence

from .schema import DYNAMIC_BUCKETS, OBSERVED_BUCKETS, canonical_row_id


_ALL_SUCCESSFUL = "all_successful"
_ACTIVE_ACTIVE = "active_active"
_KNOWN_DYNAMIC = frozenset({"commute", "order_sensitive"})


def ratio(numerator: int | Decimal, denominator: int | Decimal) -> str:
    """Return a fixed-six-decimal ratio, or blank when the denominator is zero."""

    denominator_decimal = Decimal(denominator)
    if not denominator_decimal:
        return ""
    value = Decimal(numerator) / denominator_decimal
    return format(value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN), "f")


def pair_oracle_cost(n: int) -> dict[str, int]:
    """Logical AB/BA work: N roots plus N(N-1) ordered second stages."""

    if type(n) is not int or n < 0:
        raise ValueError("configured pass count must be a non-negative integer")
    second_stage = n * (n - 1)
    return {
        "logical_first_round_applications": n,
        "logical_second_stage_applications": second_stage,
        "logical_total_pass_applications": n + second_stage,
    }


def derive_group_pair_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    group: str,
    action_ids: Iterable[str],
    study_manifest_id: str | None = None,
) -> list[dict[str, object]]:
    """Derive a nested-group view only from exact frozen action identities.

    Rows from a larger source group which contain either a non-member action are
    omitted.  A row already labelled as this target group must itself be
    internally consistent; otherwise the manifest/group join fails closed.
    """

    frozen_ids = tuple(str(action_id) for action_id in action_ids)
    if not group:
        raise ValueError("group must be non-empty")
    if not frozen_ids or any(not action_id for action_id in frozen_ids):
        raise ValueError("action_ids must be non-empty")
    if len(set(frozen_ids)) != len(frozen_ids):
        raise ValueError("duplicate action identity in group action_ids")
    _validate_manifest(rows, study_manifest_id)
    allowed = frozenset(frozen_ids)
    result: list[dict[str, object]] = []
    for row in rows:
        a, b = _action_pair(row)
        declared_group = _text(row.get("group_id"))
        in_group = a in allowed and b in allowed
        if declared_group == group and not in_group:
            raise ValueError("target group row contains action outside frozen group")
        if in_group:
            derived = dict(row)
            derived["group_id"] = group
            result.append(derived)
    return result


def build_pair_confusion(
    rows: Sequence[Mapping[str, object]],
    *,
    group: str,
    activity_view: str,
    study_manifest_id: str | None = None,
) -> list[dict[str, object]]:
    """Return all 3 x 5 observed/dynamic buckets, including zero-count rows."""

    selected, manifest = _select_view(rows, activity_view, study_manifest_id)
    counts: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in selected:
        relation, dynamic = _buckets(row)
        counts[(relation, dynamic)].append(row)

    confusion: list[dict[str, object]] = []
    for relation in OBSERVED_BUCKETS:
        for dynamic in DYNAMIC_BUCKETS:
            source = sorted(counts[(relation, dynamic)], key=_row_id)
            source_ids = tuple(_row_id(row) for row in source)
            confusion.append(
                {
                    "row_id": canonical_row_id("pair_confusion", manifest, group, activity_view, relation, dynamic),
                    "study_manifest_id": manifest,
                    "group_id": group,
                    "denominator_scope": activity_view,
                    "observed_relation": relation,
                    "dynamic_result": dynamic,
                    "pair_count": len(source),
                    "program_count": len({_text(row.get("program_id")) for row in source}),
                    "source_row_ids": ",".join(source_ids),
                    "authority_granted": "false",
                    "proved_commute": "false",
                }
            )
    return confusion


def build_pair_metrics(
    rows: Sequence[Mapping[str, object]],
    *,
    group: str,
    activity_view: str,
    study_manifest_id: str | None = None,
    single_pass_rows: Sequence[Mapping[str, object]] | None = None,
    configured_action_count: int | None = None,
    configured_action_ids: Iterable[str] | None = None,
) -> dict[str, object]:
    """Calculate a single micro, family, pair, or program metric aggregate.

    Coverage is observed-disjoint rows divided by AB/BA-decidable pair rows.
    Failed, timed-out, and unknown AB/BA results remain explicitly counted but
    are not silently regarded as a correct positive prediction.
    """

    selected, manifest = _select_view(rows, activity_view, study_manifest_id)
    physical = _physical_cost(
        selected,
        single_pass_rows=single_pass_rows,
        study_manifest_id=manifest,
        frozen_action_ids=configured_action_ids,
        configured_action_count=configured_action_count,
    )
    return _metrics_for_selected(
        selected,
        group=group,
        activity_view=activity_view,
        configured_action_count=configured_action_count,
        physical_cost=physical,
    )


def build_pair_metric_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    group: str,
    activity_view: str,
    configured_action_count: int,
    study_manifest_id: str | None = None,
    single_pass_rows: Sequence[Mapping[str, object]] | None = None,
    configured_action_ids: Iterable[str] | None = None,
) -> list[dict[str, object]]:
    """Emit deterministic micro, macro, program-family, and pass-pair strata."""

    selected, manifest = _select_view(rows, activity_view, study_manifest_id)
    # Never infer N from the rows surviving a particular view.  A frozen group
    # still contains failed/no-op/filtered actions, and pair-oracle cost is
    # defined for that complete group rather than for a visible subgraph.
    _configured_action_count(selected, configured_action_count)
    physical = _physical_cost(
        selected,
        single_pass_rows=single_pass_rows,
        study_manifest_id=manifest,
        frozen_action_ids=configured_action_ids,
        configured_action_count=configured_action_count,
    )
    micro = _metrics_for_selected(
        selected,
        group=group,
        activity_view=activity_view,
        configured_action_count=configured_action_count,
        physical_cost=physical,
    )
    per_program = [
        _metrics_for_selected(
            program_rows,
            group=group,
            activity_view=activity_view,
            configured_action_count=configured_action_count,
            physical_cost=physical,
        )
        for _program, program_rows in _strata(selected, lambda row: _text(row.get("program_id")))
    ]
    macro_precision = _mean_nonblank(row["precision"] for row in per_program)
    macro_coverage = _mean_nonblank(row["coverage"] for row in per_program)
    macro_recall = _mean_nonblank(row["recall"] for row in per_program)
    common = {
        "study_manifest_id": manifest,
        "group_id": group,
        "denominator_scope": activity_view,
        "program_macro_precision_mean": macro_precision,
        "program_macro_coverage_mean": macro_coverage,
        "program_macro_recall_mean": macro_recall,
        "authority_granted": "false",
        "proved_commute": "false",
    }
    result = [_metric_row(micro, common, "program_micro")]
    # A distinct macro row records the same totals while documenting the
    # program-equal means.  Per-program inputs remain available as source IDs.
    result.append(_metric_row(micro, common, "program_macro"))
    for family, family_rows in _strata(selected, lambda row: _text(row.get("program_family"))):
        result.append(_metric_row(_metrics_for_selected(family_rows, group=group, activity_view=activity_view, configured_action_count=configured_action_count, physical_cost=physical), common, "program_family", program_family=family))
    for pair_id, pair_rows in _strata(selected, _pass_pair_id):
        a, b = pair_id.split(":", 1)
        result.append(_metric_row(_metrics_for_selected(pair_rows, group=group, activity_view=activity_view, configured_action_count=configured_action_count, physical_cost=physical), common, "pass_pair", pass_pair_id=pair_id, action_a_id=a, action_b_id=b))
    return result


def _metric_row(
    metrics: Mapping[str, object], common: Mapping[str, object], aggregation_scope: str,
    *, program_family: str = "", pass_pair_id: str = "", action_a_id: str = "", action_b_id: str = "",
) -> dict[str, object]:
    row = dict(metrics)
    row.update(common)
    row.update(
        {
            "row_id": canonical_row_id("pair_metrics", common["study_manifest_id"], common["group_id"], common["denominator_scope"], aggregation_scope, program_family, pass_pair_id),
            "aggregation_scope": aggregation_scope,
            "program_family": program_family,
            "pass_pair_id": pass_pair_id,
            "action_a_id": action_a_id,
            "action_b_id": action_b_id,
        }
    )
    return row


def _metrics_for_selected(
    rows: Sequence[Mapping[str, object]], *, group: str, activity_view: str,
    configured_action_count: int | None = None,
    physical_cost: Mapping[str, object] | None = None,
) -> dict[str, object]:
    valid = [row for row in rows if _text(row.get("dynamic_result")) in _KNOWN_DYNAMIC]
    disjoint = [row for row in valid if _text(row.get("observed_relation")) == "observed_disjoint"]
    false_authorizations = [row for row in disjoint if _text(row.get("dynamic_result")) == "order_sensitive"]
    commuting = [row for row in valid if _text(row.get("dynamic_result")) == "commute"]
    disjoint_commuting = [row for row in disjoint if _text(row.get("dynamic_result")) == "commute"]
    logical = _weights(rows, "total_logical_pass_applications")
    physical = _weights(rows, "total_physical_pass_invocations")
    wall = _weights(rows, "wall_time_ms")
    action_count = _configured_action_count(rows, configured_action_count)
    oracle_cost = pair_oracle_cost(action_count)
    physical_cost = physical_cost or _physical_cost(
        rows,
        single_pass_rows=None,
        study_manifest_id=None,
        frozen_action_ids=None,
        configured_action_count=configured_action_count,
    )
    return {
        "group_id": group,
        "denominator_scope": activity_view,
        "program_count": len({_text(row.get("program_id")) for row in rows}),
        "pair_row_count": len(rows),
        "successful_pair_count": len(valid),
        "observed_disjoint_count": len(disjoint),
        "false_authorization_count": len(false_authorizations),
        "commuting_pair_count": len(commuting),
        "observed_disjoint_commuting_count": len(disjoint_commuting),
        "precision": ratio(len(disjoint_commuting), len(disjoint)),
        "false_authorization_rate": ratio(len(false_authorizations), len(disjoint)),
        "coverage": ratio(len(disjoint), len(valid)),
        "recall": ratio(len(disjoint_commuting), len(commuting)),
        "commute_recall": ratio(len(disjoint_commuting), len(commuting)),
        "count_weighted_precision": ratio(len(disjoint_commuting), len(disjoint)),
        "count_weighted_coverage": ratio(len(disjoint), len(valid)),
        "logical_cost_weighted_precision": _weighted_ratio(disjoint_commuting, disjoint, logical),
        "logical_cost_weighted_coverage": _weighted_ratio(disjoint, valid, logical),
        "physical_cost_weighted_precision": _weighted_ratio(disjoint_commuting, disjoint, physical),
        "physical_cost_weighted_coverage": _weighted_ratio(disjoint, valid, physical),
        "wall_time_weighted_precision": _weighted_ratio(disjoint_commuting, disjoint, wall),
        "wall_time_weighted_coverage": _weighted_ratio(disjoint, valid, wall),
        # Compatibility aliases for the approved Task-6 smoke assertion.
        "cost_weighted_precision": _weighted_ratio(disjoint_commuting, disjoint, logical),
        "cost_weighted_coverage": _weighted_ratio(disjoint, valid, logical),
        "failed_count": sum(_text(row.get("dynamic_result")) == "failed" for row in rows),
        "timeout_count": sum(_text(row.get("dynamic_result")) == "timeout" for row in rows),
        "unknown_count": sum(_text(row.get("dynamic_result")) == "unknown" for row in rows),
        "logical_first_round_applications": oracle_cost["logical_first_round_applications"],
        "logical_second_stage_applications": oracle_cost["logical_second_stage_applications"],
        "logical_pass_applications": oracle_cost["logical_total_pass_applications"],
        "physical_first_round_invocations": physical_cost["physical_first_round_invocations"],
        "physical_second_stage_invocations": physical_cost["physical_second_stage_invocations"],
        "physical_pass_invocations": physical_cost["physical_pass_invocations"],
        "physical_cost_complete": physical_cost["physical_cost_complete"],
        "wall_time_ms": sum(wall.values()),
        "source_row_ids": ",".join(sorted(_row_id(row) for row in rows)),
    }


def _select_view(
    rows: Sequence[Mapping[str, object]], activity_view: str, study_manifest_id: str | None
) -> tuple[list[Mapping[str, object]], str]:
    if activity_view not in {_ALL_SUCCESSFUL, _ACTIVE_ACTIVE}:
        raise ValueError("activity_view must be all_successful or active_active")
    manifest = _validate_manifest(rows, study_manifest_id)
    selected: list[Mapping[str, object]] = []
    for row in rows:
        _buckets(row)
        if _text(row.get("a_status")) != "success" or _text(row.get("b_status")) != "success":
            continue
        if activity_view == _ACTIVE_ACTIVE and _text(row.get("root_activity_class")) != _ACTIVE_ACTIVE:
            continue
        selected.append(row)
    return selected, manifest


def _validate_manifest(rows: Sequence[Mapping[str, object]], expected: str | None) -> str:
    _validate_authority_off(rows)
    manifests = {_text(row.get("study_manifest_id")) for row in rows}
    if not rows:
        if expected:
            return expected
        raise ValueError("study_manifest_id required for empty evidence")
    if "" in manifests or len(manifests) != 1:
        raise ValueError("study_manifest_id mismatch in pair evidence")
    manifest = next(iter(manifests))
    if expected is not None and manifest != expected:
        raise ValueError("study_manifest_id mismatch")
    return manifest


def _validate_authority_off(rows: Sequence[Mapping[str, object]]) -> None:
    """Refuse evidence that asserts any correctness authority or proof."""

    for row in rows:
        for field_name in ("authority_granted", "proved_commute"):
            value = row.get(field_name)
            if value is False or value == "false":
                continue
            raise ValueError(f"{field_name} must be explicitly false")


def _configured_action_count(
    rows: Sequence[Mapping[str, object]], configured_action_count: int | None
) -> int:
    inferred = len({action for row in rows for action in _action_pair(row)})
    if configured_action_count is None:
        return inferred
    if type(configured_action_count) is not int or configured_action_count < 0:
        raise ValueError("configured_action_count must be a non-negative integer")
    if configured_action_count < inferred:
        raise ValueError("configured_action_count is smaller than observed action set")
    return configured_action_count


def _physical_cost(
    rows: Sequence[Mapping[str, object]], *,
    single_pass_rows: Sequence[Mapping[str, object]] | None,
    study_manifest_id: str | None,
    frozen_action_ids: Iterable[str] | None,
    configured_action_count: int | None,
) -> dict[str, object]:
    """Keep first-round reuse separate from ordered second-stage executions."""

    second_stage = sum(
        _integer_field(
            row,
            "second_stage_physical_pass_invocations",
            fallback="total_physical_pass_invocations",
        )
        for row in rows
    )
    if single_pass_rows is None:
        return {
            "physical_first_round_invocations": "",
            "physical_second_stage_invocations": second_stage,
            "physical_pass_invocations": "",
            "physical_cost_complete": "false",
        }
    _validate_manifest(single_pass_rows, study_manifest_id)
    action_set = _freeze_action_ids(frozen_action_ids, configured_action_count)
    if action_set is None:
        raise ValueError("configured_action_ids required with single_pass_rows")
    observed_action_ids = {action for row in rows for action in _action_pair(row)}
    if not observed_action_ids.issubset(action_set):
        raise ValueError("pair evidence action outside frozen action group")
    pair_program_ids = {_text(row.get("program_id")) for row in rows}
    if rows and (not pair_program_ids or "" in pair_program_ids):
        raise ValueError("pair evidence missing program_id")
    if not pair_program_ids:
        pair_program_ids = {
            _text(profile.get("program_id"))
            for profile in single_pass_rows
            if _text(profile.get("program_id"))
        }
    if not pair_program_ids:
        # Backward-compatible support for a one-program direct caller that
        # predates explicit program provenance in profile fixtures.
        pair_program_ids = {"__single_program__"}
    first_profiles: dict[tuple[str, str], Mapping[str, object]] = {}
    for profile in single_pass_rows:
        action_id = _text(profile.get("action_id"))
        if not action_id:
            raise ValueError("single-pass profile missing action_id")
        if action_id not in action_set:
            raise ValueError("single-pass profile action outside frozen action group")
        program_id = _text(profile.get("program_id"))
        if not program_id:
            if len(pair_program_ids) != 1:
                raise ValueError("single-pass profile missing program_id")
            program_id = next(iter(pair_program_ids))
        if program_id not in pair_program_ids:
            # A caller can pass the full Uall profile family when computing a
            # program stratum.  Only profiles for the selected pair programs
            # contribute to that physical-cost denominator.
            continue
        key = (program_id, action_id)
        if key in first_profiles:
            raise ValueError("duplicate single-pass profile program/action identity")
        first_profiles[key] = profile
    expected_profile_keys = {
        (program_id, action_id)
        for program_id in pair_program_ids
        for action_id in action_set
    }
    if set(first_profiles) != expected_profile_keys:
        return {
            "physical_first_round_invocations": "",
            "physical_second_stage_invocations": second_stage,
            "physical_pass_invocations": "",
            "physical_cost_complete": "false",
        }
    first_stage = sum(
        _integer_field(profile, "physical_pass_invocations")
        for profile in first_profiles.values()
    )
    return {
        "physical_first_round_invocations": first_stage,
        "physical_second_stage_invocations": second_stage,
        "physical_pass_invocations": first_stage + second_stage,
        "physical_cost_complete": "true",
    }


def _freeze_action_ids(
    action_ids: Iterable[str] | None, configured_action_count: int | None
) -> frozenset[str] | None:
    if action_ids is None:
        return None
    frozen = tuple(str(action_id) for action_id in action_ids)
    if not frozen or any(not action_id for action_id in frozen):
        raise ValueError("configured_action_ids must be non-empty exact action IDs")
    if len(set(frozen)) != len(frozen):
        raise ValueError("configured_action_ids contains duplicate action IDs")
    if configured_action_count is not None and len(frozen) != configured_action_count:
        raise ValueError("configured_action_ids does not match configured_action_count")
    return frozenset(frozen)


def _integer_field(
    row: Mapping[str, object], field: str, *, fallback: str | None = None
) -> int:
    raw = row.get(field)
    if raw is None and fallback is not None:
        raw = row.get(fallback, 0)
    try:
        value = Decimal(str(raw if raw is not None else 0))
    except (InvalidOperation, ValueError):
        raise ValueError(f"{field} must be a finite non-negative integer for {_row_label(row)}") from None
    if not value.is_finite() or value < 0 or value != value.to_integral_value():
        raise ValueError(f"{field} must be a finite non-negative integer for {_row_label(row)}")
    return int(value)


def _buckets(row: Mapping[str, object]) -> tuple[str, str]:
    relation = _text(row.get("observed_relation"))
    dynamic = _text(row.get("dynamic_result"))
    if relation not in OBSERVED_BUCKETS:
        raise ValueError(f"unknown observed relation: {relation}")
    if dynamic not in DYNAMIC_BUCKETS:
        raise ValueError(f"unknown dynamic result: {dynamic}")
    return relation, dynamic


def _action_pair(row: Mapping[str, object]) -> tuple[str, str]:
    a, b = _text(row.get("action_a_id")), _text(row.get("action_b_id"))
    if not a or not b or a == b:
        raise ValueError("pair row requires two distinct exact action ids")
    return a, b


def _pass_pair_id(row: Mapping[str, object]) -> str:
    return ":".join(sorted(_action_pair(row)))


def _strata(rows: Sequence[Mapping[str, object]], key) -> list[tuple[str, list[Mapping[str, object]]]]:
    buckets: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        buckets[key(row)].append(row)
    return [(value, buckets[value]) for value in sorted(buckets)]


def _weights(rows: Sequence[Mapping[str, object]], field: str) -> dict[int, Decimal]:
    result: dict[int, Decimal] = {}
    for index, row in enumerate(rows):
        if field in {
            "total_logical_pass_applications",
            "total_physical_pass_invocations",
        }:
            value = Decimal(_integer_field(row, field))
        else:
            value = _nonnegative_decimal(row, field)
        result[id(row)] = value
    return result


def _nonnegative_decimal(row: Mapping[str, object], field: str) -> Decimal:
    try:
        value = Decimal(str(row.get(field, 0)))
    except (InvalidOperation, ValueError):
        raise ValueError(f"{field} must be finite and non-negative for {_row_label(row)}") from None
    if not value.is_finite() or value < 0:
        raise ValueError(f"{field} must be finite and non-negative for {_row_label(row)}")
    return value


def _weighted_ratio(
    numerator_rows: Sequence[Mapping[str, object]], denominator_rows: Sequence[Mapping[str, object]], weights: Mapping[int, Decimal]
) -> str:
    return ratio(sum((weights[id(row)] for row in numerator_rows), Decimal(0)), sum((weights[id(row)] for row in denominator_rows), Decimal(0)))


def _mean_nonblank(values: Iterable[object]) -> str:
    parsed = [Decimal(str(value)) for value in values if value != ""]
    return ratio(sum(parsed, Decimal(0)), len(parsed))


def _text(value: object) -> str:
    return "" if value is None else str(value)


def _row_id(row: Mapping[str, object]) -> str:
    value = _text(row.get("row_id"))
    if not value:
        raise ValueError("pair row missing row_id")
    return value


def _row_label(row: Mapping[str, object]) -> str:
    row_id = _text(row.get("row_id"))
    if row_id:
        return row_id
    action_id = _text(row.get("action_id"))
    return action_id or "evidence row"
