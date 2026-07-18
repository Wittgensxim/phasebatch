from __future__ import annotations

import csv
from pathlib import Path

import pytest

from advisor_study.schema import (
    ADVISOR_2N_DIRECTIONAL_FIELDS,
    ADVISOR_2N_GROUP_FIELDS,
    ADVISOR_2N_METRICS_FIELDS,
    ADVISOR_2N_PAIR_FIELDS,
    ARTIFACT_INDEX_FIELDS,
    DYNAMIC_BUCKETS,
    FAILURE_LEDGER_STATUSES,
    FAILURE_LEDGER_FIELDS,
    OBSERVED_BUCKETS,
    PAIR_DYNAMIC_CONFUSION_FIELDS,
    PAIR_METRICS_FIELDS,
    PAIR_OBSERVATION_FIELDS,
    PASS_GROUP_FIELDS,
    PASS_INVENTORY_FIELDS,
    PASS_PREFLIGHT_FIELDS,
    PROGRAM_MANIFEST_FIELDS,
    REPLAY_STATUSES,
    ROOT_ACTIVITY_CLASSES,
    SINGLE_PASS_OBSERVATION_FIELDS,
    STATUS_VALUES_BY_TABLE,
    TABLE_FIELDS,
    TWO_N_DIRECTIONAL_STATUSES,
    canonical_row_id,
    require_authority_off,
    write_csv,
)


EXPECTED_TABLE_NAMES = {
    "program_manifest.csv",
    "pass_inventory.csv",
    "pass_preflight.csv",
    "pass_groups.csv",
    "single_pass_observations.csv",
    "pair_observations.csv",
    "pair_dynamic_confusion.csv",
    "pair_metrics_by_group.csv",
    "advisor_2n_group_results.csv",
    "advisor_2n_directional_results.csv",
    "advisor_2n_pair_validation.csv",
    "advisor_2n_metrics_by_group.csv",
    "failure_ledger.csv",
    "artifact_index.csv",
}

EXPECTED_STATUS_COLUMNS = {
    "program_manifest.csv": {"compile_status"},
    "pass_preflight.csv": {"execution_status", "verifier_status"},
    "single_pass_observations.csv": {
        "execution_status",
        "activity_status",
        "verifier_status",
    },
    "pair_observations.csv": {
        "root_activity_class",
        "observed_relation",
        "a_status",
        "b_status",
        "ab_status",
        "ba_status",
        "ab_verifier_status",
        "ba_verifier_status",
        "dynamic_result",
    },
    "pair_dynamic_confusion.csv": {
        "observed_relation",
        "dynamic_result",
    },
    "advisor_2n_group_results.csv": {
        "round1_status",
        "first_round_disjoint_status",
        "all_n_merge_status",
        "all_n_second_round_status",
        "group_authorization_status",
    },
    "advisor_2n_directional_results.csv": {
        "directional_status",
        "first_round_status",
        "merged_input_status",
        "second_round_status",
        "verifier_status",
    },
    "advisor_2n_pair_validation.csv": {
        "action_a_directional_status",
        "action_b_directional_status",
        "two_n_pair_status",
        "dynamic_result",
        "validation_status",
        "worker_replay_status",
        "external_opt_replay_status",
        "two_n_replay_status",
    },
    "failure_ledger.csv": {"status"},
}


def test_study_status_vocabularies_are_closed() -> None:
    assert OBSERVED_BUCKETS == (
        "observed_disjoint",
        "observed_overlap",
        "observed_unknown",
    )
    assert DYNAMIC_BUCKETS == (
        "commute",
        "order_sensitive",
        "failed",
        "timeout",
        "unknown",
    )
    assert TWO_N_DIRECTIONAL_STATUSES == (
        "authorized_all_others",
        "rejected_effect_changed",
        "round1_precondition_failed",
        "direct_merge_not_defined",
        "merge_invalid",
        "second_round_failed",
        "timeout",
        "unknown",
    )
    assert ROOT_ACTIVITY_CLASSES == (
        "active_active",
        "active_noop",
        "noop_noop",
        "unknown",
    )


def test_status_registry_is_closed_by_table_and_field() -> None:
    assert REPLAY_STATUSES == (
        "not_required",
        "stable",
        "nondeterministic",
        "mismatch",
        "failed",
        "timeout",
        "unavailable",
        "unknown",
    )
    assert FAILURE_LEDGER_STATUSES == (
        "error",
        "failed",
        "invalid",
        "timeout",
        "unavailable",
        "nondeterministic",
        "unresolved",
        "unknown",
    )
    assert {
        table: set(columns)
        for table, columns in STATUS_VALUES_BY_TABLE.items()
    } == EXPECTED_STATUS_COLUMNS
    assert STATUS_VALUES_BY_TABLE["pair_observations.csv"][
        "root_activity_class"
    ] == ROOT_ACTIVITY_CLASSES
    assert STATUS_VALUES_BY_TABLE["pair_observations.csv"][
        "observed_relation"
    ] == OBSERVED_BUCKETS
    assert STATUS_VALUES_BY_TABLE["pair_observations.csv"][
        "dynamic_result"
    ] == DYNAMIC_BUCKETS
    assert STATUS_VALUES_BY_TABLE["advisor_2n_directional_results.csv"][
        "directional_status"
    ] == TWO_N_DIRECTIONAL_STATUSES


def test_row_id_is_deterministic_and_part_order_is_significant() -> None:
    left = canonical_row_id("pair", "manifest", "program", "A", "B")
    right = canonical_row_id("pair", "manifest", "program", "A", "B")
    reversed_actions = canonical_row_id(
        "pair", "manifest", "program", "B", "A"
    )

    assert left == right
    assert left.startswith("pair-")
    assert len(left) == len("pair-") + 64
    assert left != reversed_actions


@pytest.mark.parametrize(
    "row",
    [
        {"authority_granted": "false", "proved_commute": "false"},
        {"authority_granted": False, "proved_commute": False},
        {"authority_granted": "FALSE", "proved_commute": "FALSE"},
    ],
)
def test_authority_off_accepts_only_explicit_false_values(
    row: dict[str, object],
) -> None:
    require_authority_off(row)


@pytest.mark.parametrize(
    "row",
    [
        {},
        {"authority_granted": "true", "proved_commute": "false"},
        {"authority_granted": "false", "proved_commute": "true"},
        {"authority_granted": "0", "proved_commute": "false"},
    ],
)
def test_authority_off_fails_closed(row: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        require_authority_off(row)


def test_table_registry_and_evidence_columns_are_fixed() -> None:
    assert set(TABLE_FIELDS) == EXPECTED_TABLE_NAMES
    for fields in TABLE_FIELDS.values():
        assert isinstance(fields, tuple)
        assert fields[0] == "row_id"
        assert "study_manifest_id" in fields
        assert fields[-2:] == ("authority_granted", "proved_commute")
        assert len(fields) == len(set(fields))

    assert PROGRAM_MANIFEST_FIELDS[:6] == (
        "row_id",
        "study_manifest_id",
        "program_id",
        "selection_order",
        "selection_class",
        "source_path",
    )
    assert PASS_INVENTORY_FIELDS[2:7] == (
        "action_id",
        "action_sha256",
        "canonical_action_record",
        "name",
        "pipeline",
    )
    assert PAIR_OBSERVATION_FIELDS[2:8] == (
        "group_id",
        "program_id",
        "action_a_id",
        "action_b_id",
        "root_activity_class",
        "observed_relation",
    )
    assert ADVISOR_2N_PAIR_FIELDS[-10:-2] == (
        "stable_false_authorization",
        "worker_replay_status",
        "external_opt_replay_status",
        "two_n_replay_status",
        "replay_artifact_id",
        "replay_time_ms",
        "fail_closed_reason",
        "source_row_ids",
    )


def test_pair_metrics_schema_covers_required_weightings_and_strata() -> None:
    required = {
        "aggregation_scope",
        "program_family",
        "pass_pair_id",
        "count_weighted_coverage",
        "logical_cost_weighted_coverage",
        "physical_cost_weighted_coverage",
        "wall_time_weighted_coverage",
        "logical_pass_applications",
        "physical_pass_invocations",
        "wall_time_ms",
        "program_macro_precision_mean",
        "program_macro_coverage_mean",
        "program_macro_recall_mean",
    }
    assert required.issubset(PAIR_METRICS_FIELDS)


def test_two_n_metrics_schema_covers_rejections_and_stage_timing() -> None:
    required = {
        "rejection_count",
        "merge_construction_time_ms",
        "parse_time_ms",
        "verifier_time_ms",
        "worker_time_ms",
        "wall_time_ms",
    }
    assert required.issubset(ADVISOR_2N_METRICS_FIELDS)


def test_deterministic_writer_uses_fixed_order_and_canonical_row_sort(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pair_dynamic_confusion.csv"
    rows = [
        {
            "row_id": "confusion-z",
            "study_manifest_id": "manifest",
            "group_id": "U14",
            "denominator_scope": "all_successful",
            "observed_relation": "observed_overlap",
            "dynamic_result": "commute",
            "pair_count": 2,
            "program_count": 1,
            "source_row_ids": "pair-z",
            "authority_granted": "false",
            "proved_commute": "false",
        },
        {
            "row_id": "confusion-a",
            "study_manifest_id": "manifest",
            "group_id": "U14",
            "denominator_scope": "all_successful",
            "observed_relation": "observed_disjoint",
            "dynamic_result": "commute",
            "pair_count": 3,
            "program_count": 1,
            "source_row_ids": "pair-a",
            "authority_granted": "false",
            "proved_commute": "false",
        },
    ]

    write_csv(
        path,
        PAIR_DYNAMIC_CONFUSION_FIELDS,
        rows,
        isolation_root=tmp_path,
    )
    first_bytes = path.read_bytes()
    write_csv(
        path,
        PAIR_DYNAMIC_CONFUSION_FIELDS,
        list(reversed(rows)),
        isolation_root=tmp_path,
    )

    assert path.read_bytes() == first_bytes
    assert b"\r\n" not in first_bytes
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == PAIR_DYNAMIC_CONFUSION_FIELDS
        assert [row["row_id"] for row in reader] == [
            "confusion-a",
            "confusion-z",
        ]


def test_writer_rejects_authority_on_evidence(tmp_path: Path) -> None:
    row = {field: "" for field in PAIR_DYNAMIC_CONFUSION_FIELDS}
    row.update(
        {
            "row_id": "confusion-a",
            "study_manifest_id": "manifest",
            "authority_granted": "true",
            "proved_commute": "false",
        }
    )

    with pytest.raises(ValueError, match="authority_granted=false"):
        write_csv(
            tmp_path / "pair_dynamic_confusion.csv",
            PAIR_DYNAMIC_CONFUSION_FIELDS,
            [row],
            isolation_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("observed_relation", "disjoint"),
        ("dynamic_result", "probably_commutes"),
    ],
)
def test_writer_rejects_status_outside_closed_vocabulary(
    tmp_path: Path,
    field: str,
    invalid_value: str,
) -> None:
    row = {column: "" for column in PAIR_DYNAMIC_CONFUSION_FIELDS}
    row.update(
        {
            "row_id": "confusion-a",
            "study_manifest_id": "manifest",
            "group_id": "U14",
            "denominator_scope": "all_successful",
            "observed_relation": "observed_disjoint",
            "dynamic_result": "commute",
            "pair_count": "1",
            "program_count": "1",
            "source_row_ids": "pair-a",
            "authority_granted": "false",
            "proved_commute": "false",
            field: invalid_value,
        }
    )

    with pytest.raises(ValueError, match=field):
        write_csv(
            tmp_path / "pair_dynamic_confusion.csv",
            PAIR_DYNAMIC_CONFUSION_FIELDS,
            [row],
            isolation_root=tmp_path,
        )


def test_writer_rejects_target_outside_isolation_root(tmp_path: Path) -> None:
    isolation_root = tmp_path / "isolated"
    outside = tmp_path / "escaped" / "pair_dynamic_confusion.csv"
    row = {
        "row_id": "confusion-a",
        "study_manifest_id": "manifest",
        "group_id": "U14",
        "denominator_scope": "all_successful",
        "observed_relation": "observed_disjoint",
        "dynamic_result": "commute",
        "pair_count": "1",
        "program_count": "1",
        "source_row_ids": "pair-a",
        "authority_granted": "false",
        "proved_commute": "false",
    }

    with pytest.raises(ValueError, match="isolation root"):
        write_csv(
            outside,
            PAIR_DYNAMIC_CONFUSION_FIELDS,
            [row],
            isolation_root=isolation_root,
        )
    assert not outside.exists()


def test_writer_rejects_unregistered_table_name(tmp_path: Path) -> None:
    row = {
        "row_id": "row-a",
        "study_manifest_id": "manifest",
        "authority_granted": "false",
        "proved_commute": "false",
    }
    with pytest.raises(ValueError, match="unregistered CSV table"):
        write_csv(
            tmp_path / "unregistered.csv",
            (
                "row_id",
                "study_manifest_id",
                "authority_granted",
                "proved_commute",
            ),
            [row],
            isolation_root=tmp_path,
        )


@pytest.mark.parametrize(("field", "value"), [("row_id", ""), ("row_id", "  "), ("study_manifest_id", "")])
def test_writer_rejects_empty_evidence_identity(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    row = {
        "row_id": "row-a",
        "study_manifest_id": "manifest",
        "authority_granted": "false",
        "proved_commute": "false",
        field: value,
    }
    with pytest.raises(ValueError, match=field):
        write_csv(
            tmp_path / "artifact_index.csv",
            ARTIFACT_INDEX_FIELDS,
            [row],
            isolation_root=tmp_path,
        )


def test_writer_rejects_duplicate_row_ids(tmp_path: Path) -> None:
    base = {
        "row_id": "row-a",
        "study_manifest_id": "manifest",
        "authority_granted": "false",
        "proved_commute": "false",
    }
    with pytest.raises(ValueError, match="duplicate row_id"):
        write_csv(
            tmp_path / "artifact_index.csv",
            ARTIFACT_INDEX_FIELDS,
            [base, dict(base)],
            isolation_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("table_name", "status_field"),
    [
        (table_name, status_field)
        for table_name, status_fields in EXPECTED_STATUS_COLUMNS.items()
        for status_field in sorted(status_fields)
    ],
)
def test_every_registered_status_column_rejects_invalid_value(
    tmp_path: Path,
    table_name: str,
    status_field: str,
) -> None:
    fields = TABLE_FIELDS[table_name]
    row = {field: "" for field in fields}
    row.update(
        {
            "row_id": "row-a",
            "study_manifest_id": "manifest",
            "authority_granted": "false",
            "proved_commute": "false",
        }
    )
    for field, allowed in STATUS_VALUES_BY_TABLE[table_name].items():
        row[field] = allowed[0]
    row[status_field] = "__invalid_status__"

    with pytest.raises(ValueError, match=status_field):
        write_csv(
            tmp_path / table_name,
            fields,
            [row],
            isolation_root=tmp_path,
        )
