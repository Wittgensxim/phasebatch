from __future__ import annotations

from decimal import Decimal

import pytest

from advisor_study.report import (
    build_pair_confusion,
    build_pair_metric_rows,
    build_pair_metrics,
    derive_group_pair_rows,
    pair_oracle_cost,
    ratio,
)


def _pair(
    *,
    row_id: str,
    group: str = "Uall",
    program: str = "p1",
    family: str = "Benchmarks",
    action_a: str = "a",
    action_b: str = "b",
    relation: str = "observed_disjoint",
    dynamic: str = "commute",
    activity: str = "active_active",
    cost: int = 4,
    physical_cost: int = 2,
    wall_time_ms: int = 10,
    manifest: str = "manifest",
) -> dict[str, object]:
    return {
        "row_id": row_id,
        "study_manifest_id": manifest,
        "group_id": group,
        "program_id": program,
        "program_family": family,
        "action_a_id": action_a,
        "action_b_id": action_b,
        "root_activity_class": activity,
        "a_status": "success",
        "b_status": "success",
        "observed_relation": relation,
        "dynamic_result": dynamic,
        "total_logical_pass_applications": cost,
        "total_physical_pass_invocations": physical_cost,
        "wall_time_ms": wall_time_ms,
        "authority_granted": "false",
        "proved_commute": "false",
    }


def test_pair_metrics_use_declared_denominators_and_fixed_decimal_format() -> None:
    metrics = build_pair_metrics(
        [
            _pair(row_id="1", dynamic="commute", cost=4),
            _pair(row_id="2", dynamic="order_sensitive", cost=6),
            _pair(row_id="3", relation="observed_overlap", dynamic="commute", cost=10),
        ],
        group="U14",
        activity_view="all_successful",
        study_manifest_id="manifest",
    )

    assert metrics["precision"] == "0.500000"
    assert metrics["false_authorization_count"] == 1
    assert metrics["false_authorization_rate"] == "0.500000"
    assert metrics["coverage"] == "0.666667"
    assert metrics["recall"] == "0.500000"
    assert metrics["commute_recall"] == "0.500000"
    assert metrics["count_weighted_coverage"] == "0.666667"
    assert metrics["logical_cost_weighted_coverage"] == "0.500000"
    assert metrics["cost_weighted_coverage"] == "0.500000"
    assert ratio(Decimal(1), Decimal(3)) == "0.333333"
    assert ratio(0, 0) == ""


def test_confusion_is_complete_zero_filled_and_views_never_mix() -> None:
    rows = [
        _pair(row_id="aa", relation="observed_disjoint", dynamic="commute"),
        _pair(
            row_id="an",
            relation="observed_overlap",
            dynamic="failed",
            activity="active_noop",
        ),
    ]

    all_rows = build_pair_confusion(
        rows, group="U14", activity_view="all_successful", study_manifest_id="manifest"
    )
    active_rows = build_pair_confusion(
        rows, group="U14", activity_view="active_active", study_manifest_id="manifest"
    )

    assert len(all_rows) == len(active_rows) == 15
    assert all(row["denominator_scope"] == "all_successful" for row in all_rows)
    assert all(row["authority_granted"] == "false" for row in all_rows)
    assert all(row["proved_commute"] == "false" for row in all_rows)
    all_counts = {(r["observed_relation"], r["dynamic_result"]): r["pair_count"] for r in all_rows}
    active_counts = {(r["observed_relation"], r["dynamic_result"]): r["pair_count"] for r in active_rows}
    assert all_counts[("observed_disjoint", "commute")] == 1
    assert all_counts[("observed_overlap", "failed")] == 1
    assert active_counts[("observed_disjoint", "commute")] == 1
    assert active_counts[("observed_overlap", "failed")] == 0
    assert all_counts[("observed_unknown", "timeout")] == 0


def test_metric_rows_include_micro_macro_family_and_pair_strata() -> None:
    rows = [
        _pair(row_id="p1ab", program="p1", family="A", dynamic="commute"),
        _pair(
            row_id="p1ac",
            program="p1",
            family="A",
            action_b="c",
            relation="observed_overlap",
            dynamic="commute",
        ),
        _pair(
            row_id="p2ab",
            program="p2",
            family="B",
            dynamic="order_sensitive",
        ),
    ]

    metric_rows = build_pair_metric_rows(
        rows,
        group="U14",
        activity_view="all_successful",
        study_manifest_id="manifest",
        configured_action_count=3,
    )
    scopes = [row["aggregation_scope"] for row in metric_rows]
    assert scopes[0] == "program_micro"
    assert set(scopes) >= {"program_micro", "program_macro", "program_family", "pass_pair"}
    micro = metric_rows[0]
    assert micro["program_count"] == 2
    assert micro["program_macro_precision_mean"] == "0.500000"
    assert micro["program_macro_coverage_mean"] == "0.750000"
    assert micro["program_macro_recall_mean"] == "0.500000"
    family_rows = [r for r in metric_rows if r["aggregation_scope"] == "program_family"]
    assert [r["program_family"] for r in family_rows] == ["A", "B"]
    pair_rows = [r for r in metric_rows if r["aggregation_scope"] == "pass_pair"]
    assert [r["pass_pair_id"] for r in pair_rows] == ["a:b", "a:c"]


def test_group_views_are_derived_from_exact_action_ids_not_source_group_labels() -> None:
    rows = [
        _pair(row_id="ab", group="Uall", action_a="a", action_b="b"),
        _pair(row_id="ac", group="Uall", action_a="a", action_b="c"),
        _pair(row_id="bc", group="Uall", action_a="b", action_b="c"),
    ]

    derived = derive_group_pair_rows(
        rows,
        group="U14",
        action_ids=("b", "a"),
        study_manifest_id="manifest",
    )

    assert [row["row_id"] for row in derived] == ["ab"]
    assert derived[0]["group_id"] == "U14"
    assert derive_group_pair_rows(
        rows, group="U30", action_ids=("a", "b", "c"), study_manifest_id="manifest"
    ) == [
        {**row, "group_id": "U30"} for row in rows
    ]


def test_identity_and_group_joins_fail_closed() -> None:
    row = _pair(row_id="wrong", manifest="wrong-manifest")
    with pytest.raises(ValueError, match="study_manifest_id"):
        build_pair_confusion(
            [row], group="U14", activity_view="all_successful", study_manifest_id="manifest"
        )
    with pytest.raises(ValueError, match="duplicate action"):
        derive_group_pair_rows(
            [_pair(row_id="x")],
            group="U14",
            action_ids=("a", "a"),
            study_manifest_id="manifest",
        )
    with pytest.raises(ValueError, match="outside"):
        derive_group_pair_rows(
            [_pair(row_id="x", group="U14", action_a="a", action_b="outside")],
            group="U14",
            action_ids=("a", "b"),
            study_manifest_id="manifest",
        )


def test_pair_cost_is_n_plus_n_times_n_minus_one_never_factorial() -> None:
    assert pair_oracle_cost(0) == {
        "logical_first_round_applications": 0,
        "logical_second_stage_applications": 0,
        "logical_total_pass_applications": 0,
    }
    assert pair_oracle_cost(3) == {
        "logical_first_round_applications": 3,
        "logical_second_stage_applications": 6,
        "logical_total_pass_applications": 9,
    }
    assert pair_oracle_cost(30)["logical_total_pass_applications"] == 900


def test_aggregate_rejects_authority_claims_instead_of_overwriting_them() -> None:
    row = _pair(row_id="authority")
    row["authority_granted"] = "true"

    with pytest.raises(ValueError, match="authority_granted"):
        build_pair_metrics(
            [row], group="U14", activity_view="all_successful", study_manifest_id="manifest"
        )


def test_group_logical_cost_is_unique_first_round_plus_ordered_second_stage() -> None:
    rows = [
        _pair(row_id="ab", action_a="a", action_b="b", cost=4),
        _pair(row_id="ac", action_a="a", action_b="c", cost=4),
        _pair(row_id="bc", action_a="b", action_b="c", cost=4),
    ]
    profiles = [
        {"study_manifest_id": "manifest", "action_id": action, "physical_pass_invocations": 1,
         "authority_granted": "false", "proved_commute": "false"}
        for action in ("a", "b", "c")
    ]

    metrics = build_pair_metrics(
        rows,
        group="U14",
        activity_view="all_successful",
        study_manifest_id="manifest",
        single_pass_rows=profiles,
        configured_action_ids=("a", "b", "c"),
    )

    assert metrics["logical_first_round_applications"] == 3
    assert metrics["logical_second_stage_applications"] == 6
    assert metrics["logical_pass_applications"] == 9
    assert metrics["physical_first_round_invocations"] == 3
    assert metrics["physical_second_stage_invocations"] == 6
    assert metrics["physical_pass_invocations"] == 9
    assert metrics["physical_cost_complete"] == "true"


def test_physical_cost_uses_program_and_action_identity_for_shared_passes() -> None:
    rows = [_pair(row_id=f"{program}-ab", program=program) for program in ("p1", "p2", "p3")]
    profiles = [
        {
            "study_manifest_id": "manifest",
            "program_id": program,
            "action_id": action,
            "physical_pass_invocations": 1,
            "authority_granted": "false",
            "proved_commute": "false",
        }
        for program in ("p1", "p2", "p3")
        for action in ("a", "b")
    ]

    metrics = build_pair_metrics(
        rows,
        group="U14",
        activity_view="all_successful",
        study_manifest_id="manifest",
        single_pass_rows=profiles,
        configured_action_ids=("a", "b"),
    )

    assert metrics["physical_first_round_invocations"] == 6
    assert metrics["physical_second_stage_invocations"] == 6
    assert metrics["physical_pass_invocations"] == 12
    duplicate = [*profiles, dict(profiles[0])]
    with pytest.raises(ValueError, match="duplicate single-pass profile program/action identity"):
        build_pair_metrics(
            rows,
            group="U14",
            activity_view="all_successful",
            study_manifest_id="manifest",
            single_pass_rows=duplicate,
            configured_action_ids=("a", "b"),
        )


def test_metric_rows_keep_frozen_group_n_when_active_view_has_only_two_actions() -> None:
    # The frozen U14 is N=3, even though only a/b form an active-active
    # observed pair and c had no eligible pair row after a failed first round.
    rows = [_pair(row_id="ab", action_a="a", action_b="b", activity="active_active")]

    metric_rows = build_pair_metric_rows(
        rows,
        group="U14",
        activity_view="active_active",
        study_manifest_id="manifest",
        configured_action_count=3,
    )

    assert {row["aggregation_scope"] for row in metric_rows} >= {
        "program_micro", "program_macro", "program_family", "pass_pair"
    }
    assert {row["logical_pass_applications"] for row in metric_rows} == {9}
    assert {row["logical_first_round_applications"] for row in metric_rows} == {3}
    assert {row["logical_second_stage_applications"] for row in metric_rows} == {6}


def test_physical_cost_requires_profiles_for_the_full_frozen_action_group() -> None:
    rows = [_pair(row_id="ab", action_a="a", action_b="b")]
    profiles = [
        {"study_manifest_id": "manifest", "action_id": action, "physical_pass_invocations": 1,
         "authority_granted": "false", "proved_commute": "false"}
        for action in ("a", "b")
    ]

    metrics = build_pair_metrics(
        rows,
        group="U14",
        activity_view="active_active",
        study_manifest_id="manifest",
        configured_action_count=3,
        configured_action_ids=("a", "b", "c"),
        single_pass_rows=profiles,
    )

    assert metrics["logical_pass_applications"] == 9
    assert metrics["physical_cost_complete"] == "false"
    assert metrics["physical_first_round_invocations"] == ""
    assert metrics["physical_pass_invocations"] == ""


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("total_logical_pass_applications", "1.5"),
        ("total_physical_pass_invocations", "NaN"),
        ("second_stage_physical_pass_invocations", "Infinity"),
    ],
)
def test_invocation_counts_reject_fractional_or_nonfinite_values(
    field: str, value: str
) -> None:
    row = _pair(row_id="bad-count")
    row[field] = value

    with pytest.raises(ValueError, match="finite non-negative integer"):
        build_pair_metrics(
            [row], group="U14", activity_view="all_successful", study_manifest_id="manifest"
        )
