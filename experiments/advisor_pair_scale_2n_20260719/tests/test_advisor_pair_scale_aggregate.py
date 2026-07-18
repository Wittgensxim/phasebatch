from __future__ import annotations

import hashlib
import json
from pathlib import Path
import csv

import pytest
import advisor_study.aggregate as aggregate_module

from advisor_study.aggregate import (
    FORBIDDEN_CLAIMS,
    REQUIRED_AGGREGATE_FILES,
    materialize_aggregate,
    validate_claim_text,
)
from advisor_study.schema import STATUS_VALUES_BY_TABLE, TABLE_FIELDS


def _complete_row(table_name: str, **values: object) -> dict[str, object]:
    """Provide a fully shaped raw row; aggregate must reject partial rows."""

    row: dict[str, object] = {field: "" for field in TABLE_FIELDS[table_name]}
    for field, statuses in STATUS_VALUES_BY_TABLE.get(table_name, {}).items():
        row[field] = statuses[0]
    for field in row:
        if (
            field.endswith(("_n", "_count", "_calls", "_applications", "_invocations", "_ms"))
            or field in {"selection_order", "selection_rank", "group_size", "source_size_bytes", "reserve_rank", "repetition"}
        ):
            row[field] = 0
        if field in {"cache_reused", "artifact_available", "artifact_materialized", "deterministic", "eligible", "available", "materialized"}:
            row[field] = "false"
    row.update(
        {
            "row_id": f"row-{table_name}",
            "study_manifest_id": "manifest-1",
            "authority_granted": "false",
            "proved_commute": "false",
        }
    )
    row.update(values)
    return row


def _pair(row_id: str = "pair-uall") -> dict[str, object]:
    return _complete_row(
        "pair_observations.csv",
        row_id=row_id,
        group_id="Uall",
        program_id="p1",
        program_family="Benchmarks",
        action_a_id="a",
        action_b_id="b",
        root_activity_class="active_active",
        a_status="success",
        b_status="success",
        ab_status="success",
        ba_status="success",
        observed_relation="observed_disjoint",
        dynamic_result="order_sensitive",
        second_stage_physical_pass_invocations=2,
        total_logical_pass_applications=4,
        total_physical_pass_invocations=4,
        wall_time_ms=17,
    )


def _single(action_id: str) -> dict[str, object]:
    return _complete_row(
        "single_pass_observations.csv",
        row_id=f"single-{action_id}",
        group_id="Uall",
        program_id="p1",
        action_id=action_id,
        execution_status="success",
        activity_status="active",
        verifier_status="success",
        physical_pass_invocations=1,
        wall_time_ms=3,
    )


def _two_n_group() -> dict[str, object]:
    return _complete_row(
        "advisor_2n_group_results.csv",
        row_id="two-n-group",
        group_id="U14",
        program_id="p1",
        configured_n=2,
        successful_n=2,
        active_n=2,
        no_op_n=0,
        failed_n=0,
        timeout_n=0,
        round1_status="complete",
        first_round_disjoint_status="disjoint",
        all_n_merge_status="complete",
        all_n_second_round_status="complete",
        group_authorization_status="authorized",
        directional_authorized_count=2,
        directional_unavailable_count=0,
        logical_pass_applications=4,
        physical_pass_invocations=4,
        merge_helper_calls=2,
        merge_construction_time_ms=5,
        parse_time_ms=1,
        verifier_time_ms=2,
        worker_time_ms=4,
        wall_time_ms=18,
    )


def _two_n_directional(action_id: str) -> dict[str, object]:
    return _complete_row(
        "advisor_2n_directional_results.csv",
        row_id=f"directional-{action_id}",
        group_id="U14",
        program_id="p1",
        action_id=action_id,
        directional_status="authorized_all_others",
        first_round_status="success",
        merged_input_status="complete",
        second_round_status="success",
        verifier_status="success",
        logical_pass_applications=2,
        physical_pass_invocations=2,
        merge_helper_calls=1,
        wall_time_ms=9,
    )


def _two_n_pair() -> dict[str, object]:
    return _complete_row(
        "advisor_2n_pair_validation.csv",
        row_id="two-n-pair",
        group_id="U14",
        program_id="p1",
        action_a_id="a",
        action_b_id="b",
        action_a_directional_status="authorized_all_others",
        action_b_directional_status="authorized_all_others",
        two_n_pair_status="both_directions_authorized",
        pair_observation_row_id="pair-uall",
        dynamic_result="order_sensitive",
        validation_status="false_authorization",
        false_authorization="true",
        stable_false_authorization="true",
        worker_replay_status="stable",
        external_opt_replay_status="stable",
        two_n_replay_status="stable",
    )


def _reuse_event() -> dict[str, object]:
    return _complete_row(
        "artifact_index.csv",
        row_id="reuse-event-row",
        artifact_id="reuse-event-1",
        artifact_kind="reuse_event",
        relative_path="raw/reuse/reuse-event-1.json",
        sha256="a" * 64,
        size_bytes=1,
        available="true",
        materialized="true",
        provenance="content_addressed_view_reuse",
        source_row_ids="single-a,single-b",
    )


def _all_outputs(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_raw_validation_allows_fixed_manifest_without_reserve_rank_but_keeps_pair_fields_strict() -> None:
    fixed_program = _complete_row(
        "program_manifest.csv",
        selection_class="fixed",
        reserve_rank="",
        selection_order=1,
        source_size_bytes=1,
    )
    pair = _pair()

    validated = aggregate_module._validate_raw_tables(
        {"program_manifest.csv": [fixed_program], "pair_observations.csv": [pair]}, "manifest-1"
    )

    assert validated["program_manifest.csv"][0]["reserve_rank"] == ""
    assert "reserve_rank" not in validated["pair_observations.csv"][0]
    malformed_pair = dict(pair)
    del malformed_pair["action_b_id"]
    with pytest.raises(ValueError, match="pair_observations.csv missing required fields"):
        aggregate_module._validate_raw_tables({"pair_observations.csv": [malformed_pair]}, "manifest-1")
    reserve_program = dict(fixed_program)
    reserve_program["selection_class"] = "reserve"
    with pytest.raises(ValueError, match="reserve_rank must be a finite non-negative integer"):
        aggregate_module._validate_raw_tables({"program_manifest.csv": [reserve_program]}, "manifest-1")


def test_raw_validation_allows_exact_u14_without_selection_rank_but_requires_ranked_selection() -> None:
    exact_u14 = _complete_row(
        "pass_groups.csv",
        group_id="U14",
        group_size=14,
        action_id="a",
        action_order=1,
        selection_method="exact_u14",
        selection_rank="",
    )

    validated = aggregate_module._validate_raw_tables({"pass_groups.csv": [exact_u14]}, "manifest-1")

    assert validated["pass_groups.csv"][0]["selection_rank"] == ""
    ranked = dict(exact_u14)
    ranked["selection_method"] = "frozen_sha256_rank"
    with pytest.raises(ValueError, match="selection_rank must be a finite non-negative integer"):
        aggregate_module._validate_raw_tables({"pass_groups.csv": [ranked]}, "manifest-1")


def test_materialize_aggregate_writes_fixed_inventory_metrics_provenance_and_figures(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "isolated" / "output" / "smoke"
    result = materialize_aggregate(
        out_dir=out_dir,
        isolation_root=tmp_path / "isolated",
        study_manifest_id="manifest-1",
        group_actions={"U14": ("a", "b"), "U30": ("a", "b"), "Uall": ("a", "b")},
        tables={
            "pair_observations.csv": [_pair()],
            "single_pass_observations.csv": [_single("a"), _single("b")],
            "advisor_2n_group_results.csv": [_two_n_group()],
            "advisor_2n_directional_results.csv": [_two_n_directional("a"), _two_n_directional("b")],
            "advisor_2n_pair_validation.csv": [_two_n_pair()],
            "artifact_index.csv": [_reuse_event()],
        },
    )

    aggregate = out_dir / "aggregate"
    figures = out_dir / "figures"
    assert set(path.name for path in result.files) == set(REQUIRED_AGGREGATE_FILES)
    assert all(path.is_file() for path in result.files)
    assert json.loads((aggregate / "evidence_manifest.json").read_text(encoding="utf-8"))["study_manifest_id"] == "manifest-1"
    assert len((aggregate / "pair_dynamic_confusion.csv").read_text(encoding="utf-8").splitlines()) == 1 + 3 * 2 * 15
    assert "false_authorization" in (aggregate / "two_n_false_authorizations.csv").read_text(encoding="utf-8")
    assert "实证证据" in (aggregate / "expanded_pair_and_2n_report.md").read_text(encoding="utf-8")
    assert "稳定 `false_authorization` 数：1" in (aggregate / "expanded_pair_and_2n_report.md").read_text(encoding="utf-8")
    assert "row_id" in (aggregate / "source_index.csv").read_text(encoding="utf-8")
    assert "precision" in (aggregate / "pair_metrics_by_group.csv").read_text(encoding="utf-8")
    assert "disjoint_merge_applicable" in (aggregate / "two_n_gate_funnel.csv").read_text(encoding="utf-8")
    assert "no-op" not in (aggregate / "study_limitations.csv").read_text(encoding="utf-8")
    with (aggregate / "advisor_2n_metrics_by_group.csv").open(encoding="utf-8", newline="") as handle:
        metrics = {row["group_id"]: row for row in csv.DictReader(handle)}
    assert metrics["U14"]["configured_n_min"] == "2"
    assert metrics["U14"]["round1_success_programs"] == "1"
    assert metrics["U14"]["directional_authorized_count"] == "2"
    assert metrics["U14"]["false_authorization_count"] == "1"
    assert metrics["U14"]["merge_construction_time_ms"] == "5"
    with (aggregate / "execution_costs.csv").open(encoding="utf-8", newline="") as handle:
        costs = list(csv.DictReader(handle))
    pair_cost = next(row for row in costs if row["group_id"] == "U14" and row["method"] == "pair_oracle")
    assert pair_cost["logical_total_pass_applications"] == "4"
    assert pair_cost["logical_second_stage_applications"] == "2"
    for stem in (
        "01_pair_precision_coverage",
        "02_2n_gate_applicability",
        "03_2n_pair_agreement",
        "04_execution_cost",
    ):
        assert "中文" not in stem
        assert "精度" in (figures / f"{stem}.svg").read_text(encoding="utf-8") or stem != "01_pair_precision_coverage"
        assert (figures / f"{stem}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    agreement_svg = (figures / "03_2n_pair_agreement.svg").read_text(encoding="utf-8")
    assert all(group in agreement_svg for group in ("U14", "U30", "Uall"))
    cost_svg = (figures / "04_execution_cost.svg").read_text(encoding="utf-8")
    assert "N/A" in cost_svg
    assert "完整Pair" in cost_svg and "导师2N" in cost_svg
    assert "<tspan" in cost_svg
    assert "rotate(45" not in cost_svg


def test_materialize_aggregate_is_byte_deterministic_and_rejects_authority_or_escape(
    tmp_path: Path,
) -> None:
    root = tmp_path / "isolated"
    kwargs = {
        "out_dir": root / "output" / "formal",
        "isolation_root": root,
        "study_manifest_id": "manifest-1",
        "group_actions": {"U14": ("a", "b"), "U30": ("a", "b"), "Uall": ("a", "b")},
        "tables": {"pair_observations.csv": [_pair()]},
    }
    materialize_aggregate(**kwargs)
    first = _all_outputs(kwargs["out_dir"])
    materialize_aggregate(**kwargs)
    assert _all_outputs(kwargs["out_dir"]) == first

    bad = _pair()
    bad["authority_granted"] = "true"
    with pytest.raises(ValueError, match="authority_granted"):
        materialize_aggregate(
            out_dir=root / "output" / "formal",
            isolation_root=root,
            study_manifest_id="manifest-1",
            group_actions={"U14": ("a", "b"), "U30": ("a", "b"), "Uall": ("a", "b")},
            tables={"pair_observations.csv": [bad]},
        )


def test_formal_ten_program_report_is_dynamic_and_has_no_old_scope_claim(
    tmp_path: Path,
) -> None:
    root = tmp_path / "isolated"
    result = materialize_aggregate(
        out_dir=root / "output" / "formal",
        isolation_root=root,
        study_manifest_id="manifest-1",
        group_actions={"U14": ("a", "b"), "U30": ("a", "b"), "Uall": ("a", "b")},
        tables={"pair_observations.csv": [_pair()]},
        program_count=10,
    )

    report = (result.aggregate_dir / "advisor_talking_points.md").read_text(encoding="utf-8")
    detailed_report = (result.aggregate_dir / "expanded_pair_and_2n_report.md").read_text(encoding="utf-8")
    evidence = json.loads((result.aggregate_dir / "evidence_manifest.json").read_text(encoding="utf-8"))
    assert "10 个程序" in report
    assert "50 个程序" not in report
    assert "100 个程序" not in report
    assert "existing root-only fixed50" not in report
    assert "用户后续范围覆盖" not in detailed_report
    assert evidence["program_count"] == 10
    assert evidence["formal_scope_user_override"] == ""


def test_aggregate_rejects_non_output_target_and_partial_or_extra_raw_rows(tmp_path: Path) -> None:
    root = tmp_path / "isolated"
    common = {
        "isolation_root": root,
        "study_manifest_id": "manifest-1",
        "group_actions": {"U14": ("a", "b"), "U30": ("a", "b"), "Uall": ("a", "b")},
    }
    with pytest.raises(ValueError, match="output/(smoke|formal)"):
        materialize_aggregate(out_dir=root / "docs" / "aggregate", tables={"pair_observations.csv": [_pair()]}, **common)
    partial = _pair()
    partial.pop("ab_status")
    with pytest.raises(ValueError, match="missing required fields"):
        materialize_aggregate(out_dir=root / "output" / "smoke", tables={"pair_observations.csv": [partial]}, **common)
    extra = _pair()
    extra["unregistered_detail"] = "not allowed"
    with pytest.raises(ValueError, match="unregistered fields"):
        materialize_aggregate(out_dir=root / "output" / "smoke", tables={"pair_observations.csv": [extra]}, **common)
    missing_count = _pair()
    missing_count["total_physical_pass_invocations"] = ""
    with pytest.raises(ValueError, match="total_physical_pass_invocations"):
        materialize_aggregate(out_dir=root / "output" / "smoke", tables={"pair_observations.csv": [missing_count]}, **common)


@pytest.mark.parametrize("field", ["worker_replay_status", "external_opt_replay_status", "two_n_replay_status"])
def test_stable_false_authorization_requires_all_replay_witnesses_stable(
    tmp_path: Path, field: str,
) -> None:
    invalid = _two_n_pair()
    invalid[field] = "failed"
    with pytest.raises(ValueError, match="stable_false_authorization"):
        materialize_aggregate(
            out_dir=tmp_path / "isolated" / "output" / "formal",
            isolation_root=tmp_path / "isolated",
            study_manifest_id="manifest-1",
            group_actions={"U14": ("a", "b"), "U30": ("a", "b"), "Uall": ("a", "b")},
            tables={"advisor_2n_pair_validation.csv": [invalid]},
        )


def test_aggregate_stages_then_publishes_and_preserves_previous_complete_on_build_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "isolated"
    out_dir = root / "output" / "formal"
    old_aggregate = out_dir / "aggregate"
    old_figures = out_dir / "figures"
    old_aggregate.mkdir(parents=True)
    old_figures.mkdir(parents=True)
    (old_aggregate / "README.md").write_text("old complete", encoding="utf-8")
    (old_aggregate / "pair_observations.csv").write_text("old aggregate", encoding="utf-8")
    (old_figures / "old.png").write_bytes(b"old")
    monkeypatch.setattr(aggregate_module, "_write_png", lambda *_args: (_ for _ in ()).throw(RuntimeError("injected figure failure")))

    with pytest.raises(RuntimeError, match="injected figure failure"):
        materialize_aggregate(
            out_dir=out_dir,
            isolation_root=root,
            study_manifest_id="manifest-1",
            group_actions={"U14": ("a", "b"), "U30": ("a", "b"), "Uall": ("a", "b")},
            tables={"pair_observations.csv": [_pair()]},
        )

    assert (old_aggregate / "README.md").read_text(encoding="utf-8") == "old complete"
    assert (old_aggregate / "pair_observations.csv").read_text(encoding="utf-8") == "old aggregate"
    assert (old_figures / "old.png").read_bytes() == b"old"
    assert not list((root / "output").glob(".formal.aggregate-staging-*"))


def test_png_is_valid_fixed_size_and_costs_keep_components_and_shared_first_round_provenance(
    tmp_path: Path,
) -> None:
    from PIL import Image

    root = tmp_path / "isolated"
    result = materialize_aggregate(
        out_dir=root / "output" / "formal",
        isolation_root=root,
        study_manifest_id="manifest-1",
        group_actions={"U14": ("a", "b"), "U30": ("a", "b"), "Uall": ("a", "b")},
        tables={
            "pair_observations.csv": [_pair()],
            "single_pass_observations.csv": [_single("a"), _single("b")],
            "advisor_2n_group_results.csv": [_two_n_group()],
            "advisor_2n_directional_results.csv": [_two_n_directional("a"), _two_n_directional("b")],
            "advisor_2n_pair_validation.csv": [_two_n_pair()],
            "artifact_index.csv": [_reuse_event()],
        },
    )
    for image_path in result.figures_dir.glob("*.png"):
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            assert image.size == (1200, 720)
    with (result.aggregate_dir / "execution_costs.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        costs = list(reader)
    assert {"physical_first_round_invocations", "physical_second_stage_invocations", "merge_construction_time_ms", "parse_time_ms", "verifier_time_ms", "worker_time_ms", "replay_time_ms", "total_wall_time_ms", "provenance"}.issubset(fields)
    actual_pair = next(row for row in costs if row["scope"] == "experiment_wide_actual" and row["method"] == "pair_oracle")
    assert actual_pair["physical_first_round_invocations"] == "2"
    assert actual_pair["physical_second_stage_invocations"] == "2"
    assert actual_pair["physical_pass_invocations"] == "4"
    assert "single_pass_observations.csv" in actual_pair["provenance"]
    assert actual_pair["content_addressed_reuse_event_count"] == "1"
    assert actual_pair["reuse_event_source_row_ids"] == "reuse-event-1"
    actual_two_n = next(row for row in costs if row["scope"] == "experiment_wide_actual" and row["method"] == "advisor_2n")
    assert actual_two_n["logical_first_round_applications"] == "2"
    assert actual_two_n["logical_second_stage_applications"] == "6"
    assert actual_two_n["logical_total_pass_applications"] == "8"
    assert actual_two_n["total_wall_time_ms"] == "24"
    assert actual_two_n["merge_construction_time_ms"] == "5"
    assert actual_two_n["replay_time_ms"] == "0"
    evidence = json.loads((result.aggregate_dir / "evidence_manifest.json").read_text(encoding="utf-8"))
    assert len(evidence["aggregate_inventory_sha256"]) == 64
    with pytest.raises(ValueError, match="escapes isolation root"):
        materialize_aggregate(
            out_dir=tmp_path / "outside",
            isolation_root=root,
            study_manifest_id="manifest-1",
            group_actions={"U14": ("a", "b"), "U30": ("a", "b"), "Uall": ("a", "b")},
            tables={"pair_observations.csv": [_pair()]},
        )


@pytest.mark.parametrize("phrase", FORBIDDEN_CLAIMS)
def test_claim_discipline_rejects_unqualified_forbidden_claims(phrase: str) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        validate_claim_text(f"结论：{phrase}。")


def test_terminal_uall_pairs_project_once_into_typed_failure_ledger() -> None:
    preflight = _complete_row(
        "pass_preflight.csv",
        row_id="preflight-error",
        program_id="probe",
        action_id="candidate-x",
        repetition=1,
        execution_status="error",
        verifier_status="not_run",
        eligible="false",
        exclusion_reason="pass_parse_failure",
        command_sha256="7" * 64,
        stderr_sha256="8" * 64,
    )
    group_gate = _two_n_group()
    group_gate.update(
        {
            "row_id": "two-n-group-unavailable",
            "first_round_disjoint_status": "overlap",
            "all_n_merge_status": "direct_merge_not_defined",
            "all_n_second_round_status": "unknown",
            "group_authorization_status": "group_precondition_unavailable",
            "fail_closed_reason": "first-round patches overlap",
        }
    )
    failed = _pair("pair-failed")
    failed.update(
        {
            "ab_status": "error",
            "ab_verifier_status": "not_run",
            "dynamic_result": "failed",
            "artifact_id": "artifact-failed",
            "fail_closed_reason": "stage_error;pair_failed",
            "command_sha256": "1" * 64,
            "stderr_sha256": "2" * 64,
        }
    )
    timed_out = _pair("pair-timeout")
    timed_out.update(
        {
            "ba_status": "timeout",
            "ba_verifier_status": "not_run",
            "dynamic_result": "timeout",
            "artifact_id": "artifact-timeout",
            "fail_closed_reason": "stage_timeout;pair_timeout",
            "command_sha256": "3" * 64,
            "stderr_sha256": "4" * 64,
        }
    )
    unknown = _pair("pair-unknown")
    unknown.update(
        {
            "dynamic_result": "unknown",
            "artifact_id": "artifact-unknown",
            "fail_closed_reason": "comparator_uncertainty",
            "command_sha256": "5" * 64,
            "stderr_sha256": "6" * 64,
        }
    )
    derived_duplicate = dict(failed, row_id="pair-failed-u14", group_id="U14")

    rows = aggregate_module.build_failure_ledger_rows(
        preflight_rows=[preflight],
        group_rows=[group_gate],
        pair_rows=[unknown, derived_duplicate, timed_out, failed],
        study_manifest_id="manifest-1",
    )

    by_source = {
        json.loads(str(row["source_row_ids"]))[0]: row for row in rows
    }
    assert set(by_source) == {
        "preflight-error",
        "two-n-group-unavailable",
        "pair-failed",
        "pair-timeout",
        "pair-unknown",
    }
    assert by_source["preflight-error"]["failure_kind"] == "pass_preflight_error"
    assert by_source["two-n-group-unavailable"]["failure_kind"] == "overlapping_patch_region"
    assert by_source["pair-failed"]["failure_kind"] == "ab_ba_error"
    assert by_source["pair-timeout"]["failure_kind"] == "ab_ba_timeout"
    assert by_source["pair-unknown"]["failure_kind"] == "comparator_uncertainty"
    assert by_source["pair-failed"]["status"] == "error"
    assert by_source["pair-timeout"]["status"] == "timeout"
    assert by_source["pair-unknown"]["status"] == "unresolved"
    assert by_source["pair-failed"]["group_id"] == "Uall"
    assert all(row["authority_granted"] == "false" for row in rows)
    assert all(row["proved_commute"] == "false" for row in rows)
    assert rows == aggregate_module.build_failure_ledger_rows(
        preflight_rows=[preflight],
        group_rows=[group_gate],
        pair_rows=list(reversed([unknown, derived_duplicate, timed_out, failed])),
        study_manifest_id="manifest-1",
    )


def test_artifact_index_prefers_checkpoint_binding_and_keeps_cleaned_logical_pair(
    tmp_path: Path,
) -> None:
    root = tmp_path / "isolated"
    profile_path = root / "raw" / "profiles" / "a.ll"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_bytes(b"profile bytes")
    profile_sha = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    profile = _single("a")
    profile.update(
        {
            "artifact_id": "artifact-profile-a",
            "artifact_available": "false",
            "artifact_materialized": "true",
        }
    )
    cleaned = _pair("pair-cleaned")
    cleaned.update(
        {
            "artifact_id": "artifact-pair-cleaned",
            "artifact_available": "false",
            "artifact_materialized": "false",
            "cleanup_status": "retained_terminal_failure_witness",
            "ab_output_path": str(root / "raw" / "pairs" / "pair-cleaned" / "AB.ll"),
            "ab_output_sha256": "a" * 64,
            "ba_output_path": str(root / "raw" / "pairs" / "pair-cleaned" / "BA.ll"),
            "ba_output_sha256": "",
        }
    )
    binding = {
        "source_kind": "profile",
        "source_row_id": "single-a",
        "artifact_name": "output",
        "stage_key": "p1:profiles",
        "stage_relative_path": "raw/profiles",
        "relative_path": "raw/profiles/a.ll",
        "sha256": profile_sha,
        "hard_state_id": "c" * 64,
    }

    rows = aggregate_module.build_artifact_index_rows(
        program_rows=[],
        single_rows=[profile],
        pair_rows=[cleaned],
        directional_rows=[],
        materialized_artifact_bindings=[binding],
        study_manifest_id="manifest-1",
        isolation_root=root,
    )

    assert len(rows) == 3
    by_kind = {row["artifact_kind"]: row for row in rows}
    physical = by_kind["single_pass_output"]
    assert physical["artifact_id"] == "artifact-profile-a"
    assert physical["relative_path"] == "raw/profiles/a.ll"
    assert physical["sha256"] == profile_sha
    assert physical["size_bytes"] == len(b"profile bytes")
    assert physical["available"] == "true"
    assert physical["materialized"] == "true"
    logical_ab = by_kind["pair_ab_output"]
    logical_ba = by_kind["pair_ba_output"]
    assert logical_ab["artifact_id"] == logical_ba["artifact_id"] == "artifact-pair-cleaned"
    assert logical_ab["relative_path"] == "raw/pairs/pair-cleaned/AB.ll"
    assert logical_ba["relative_path"] == "raw/pairs/pair-cleaned/BA.ll"
    assert logical_ab["size_bytes"] == logical_ba["size_bytes"] == 0
    assert logical_ab["available"] == "true"
    assert logical_ba["available"] == "false"
    assert logical_ab["materialized"] == logical_ba["materialized"] == "false"
    provenance = json.loads(str(logical_ab["provenance"]))
    assert provenance["cleanup_status"] == "retained_terminal_failure_witness"
    assert provenance["artifact_name"] == "AB"
    assert all(row["source_row_ids"] in {'["single-a"]', '["pair-cleaned"]'} for row in rows)
    assert all(row["authority_granted"] == "false" for row in rows)
    assert all(row["proved_commute"] == "false" for row in rows)
    assert rows == aggregate_module.build_artifact_index_rows(
        program_rows=[],
        single_rows=[profile],
        pair_rows=[cleaned],
        directional_rows=[],
        materialized_artifact_bindings=[dict(reversed(tuple(binding.items())))],
        study_manifest_id="manifest-1",
        isolation_root=root,
    )


def test_artifact_index_formal_shape_keeps_root_and_unmaterialized_2n_slot(
    tmp_path: Path,
) -> None:
    root = tmp_path / "isolated"
    root_ir = root / "output" / "formal" / "roots" / "p1.ll"
    root_ir.parent.mkdir(parents=True)
    root_ir.write_bytes(b"root ir")
    root_sha = hashlib.sha256(root_ir.read_bytes()).hexdigest()
    program = _complete_row(
        "program_manifest.csv",
        row_id="program-p1",
        program_id="p1",
        selection_order=1,
        selection_class="fixed",
        source_size_bytes=1,
        reserve_rank="",
        root_ir_path=str(root_ir),
        root_ir_sha256=root_sha,
    )
    directional = _two_n_directional("a")
    directional.update(
        {
            "row_id": "directional-unavailable",
            "artifact_id": "",
            "directional_status": "direct_merge_not_defined",
            "merged_input_status": "direct_merge_not_defined",
            "merged_input_path": "",
            "merged_input_sha256": "",
            "second_round_status": "unknown",
            "second_output_path": "",
            "second_output_sha256": "",
            "second_output_materialized": "false",
            "cleanup_status": "retained_terminal_failure_witness",
        }
    )
    reclaimed = _pair("pair-reclaimed")
    reclaimed.update(
        {
            "artifact_id": "artifact-reclaimed",
            "artifact_available": "true",
            "artifact_materialized": "false",
            "cleanup_status": "reclaimed_nonwitness",
        }
    )

    rows = aggregate_module.build_artifact_index_rows(
        program_rows=[program],
        single_rows=[],
        pair_rows=[reclaimed],
        directional_rows=[directional],
        materialized_artifact_bindings=[],
        study_manifest_id="manifest-1",
        isolation_root=root,
    )

    assert [row["artifact_kind"] for row in rows] == ["root_ir", "two_n_second_round_output"]
    root_row, two_n_row = rows
    assert root_row["relative_path"] == "output/formal/roots/p1.ll"
    assert root_row["sha256"] == root_sha
    assert root_row["size_bytes"] == len(b"root ir")
    assert root_row["materialized"] == "true"
    assert two_n_row["relative_path"] == ""
    assert two_n_row["sha256"] == ""
    assert two_n_row["size_bytes"] == 0
    assert two_n_row["available"] == "false"
    assert two_n_row["materialized"] == "false"


def test_materialize_rebases_checkpoint_binding_relative_to_formal_output(
    tmp_path: Path,
) -> None:
    root = tmp_path / "isolated"
    out_dir = root / "output" / "formal"
    artifact = out_dir / "raw" / "profiles" / "a.ll"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"profile")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    profile = _single("a")
    profile.update(
        {
            "artifact_id": "artifact-profile-a",
            "artifact_available": "true",
            "artifact_materialized": "true",
        }
    )

    result = materialize_aggregate(
        out_dir=out_dir,
        isolation_root=root,
        study_manifest_id="manifest-1",
        group_actions={"U14": ("a", "b"), "U30": ("a", "b"), "Uall": ("a", "b")},
        tables={"single_pass_observations.csv": [profile]},
        materialized_artifact_bindings=[
            {
                "source_kind": "profile",
                "source_row_id": "single-a",
                "artifact_name": "output",
                "stage_key": "p1:profiles",
                "stage_relative_path": "raw/profiles",
                "relative_path": "raw/profiles/a.ll",
                "sha256": digest,
                "hard_state_id": "d" * 64,
            }
        ],
    )

    with (result.aggregate_dir / "artifact_index.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["relative_path"] == "raw/profiles/a.ll"
    assert rows[0]["materialized"] == "true"
