from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing
from pathlib import Path

import pytest

from advisor_study import cli
from advisor_study import orchestration as orchestration_module
from advisor_study.program_runtime import (
    combine_program_results,
    finalize_program_evidence,
    load_program_checkpoint,
    program_checkpoint_input_sha256,
    program_checkpoint_path,
    program_result_payload,
    publish_program_checkpoint,
    runtime_budget_limited_result,
)
from advisor_study.run_control import (
    DEFAULT_PROGRAM_WALL_TIME_BUDGET_S,
    add_runtime_budget_skip,
    ensure_run_control,
    load_run_control,
)


def _concurrent_runtime_skip_writer(
    out_dir: str, program_id: str, ready: object, start: object, results: object
) -> None:
    ready.set()  # type: ignore[attr-defined]
    start.wait(15)  # type: ignore[attr-defined]
    try:
        add_runtime_budget_skip(
            Path(out_dir),
            study_manifest_id="manifest-concurrent",
            program_ids=("p1", "p2"),
            program_id=program_id,
            observed_wall_time_s=600,
        )
    except Exception as error:  # pragma: no cover - asserted in parent process.
        results.put(f"{type(error).__name__}:{error}")  # type: ignore[attr-defined]
    else:
        results.put("ok")  # type: ignore[attr-defined]


def _actions() -> dict[str, object]:
    class Action:
        def __init__(self, action_id: str) -> None:
            self.action_id = action_id

    return {name: Action(name) for name in ("A", "B", "C")}


def _groups(actions: dict[str, object]) -> dict[str, tuple[object, ...]]:
    return {
        "U14": (actions["A"], actions["B"]),
        "U30": (actions["A"], actions["B"], actions["C"]),
        "Uall": tuple(actions.values()),
    }


def _provenance() -> dict[str, object]:
    control: dict[str, object] = {
        "schema_version": "advisor-pair-scale-2n/run-control-v1",
        "study_manifest_id": "manifest-1",
        "program_wall_time_budget_s": 600,
        "enforcement_mode": "external_program_boundary_skip",
        "frozen_program_ids": ["p1"],
        "skip_programs": [
            {
                "program_id": "p1",
                "limitation_kind": "runtime_budget_exceeded",
                "reason": "budget exceeded",
                "observed_wall_time_s": 731.25,
            }
        ],
        "authority_granted": False,
        "proved_commute": False,
    }
    control["control_id"] = hashlib.sha256(
        json.dumps(
            control, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    control_file_sha256 = hashlib.sha256(
        (
            json.dumps(
                control, ensure_ascii=True, sort_keys=True, separators=(",", ":")
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    return {
        "program_id": "p1",
        "decision": "skip",
        "control_id": control["control_id"],
        "control_file_sha256": control_file_sha256,
        "program_wall_time_budget_s": 600,
        "observed_wall_time_s": 731.25,
        "limitation_kind": "runtime_budget_exceeded",
        "reason": "budget exceeded",
        "enforcement_mode": "external_program_boundary_skip",
        "control_payload": control,
    }


def _runtime_payload(tmp_path: Path, *, program_id: str = "p1") -> dict[str, object]:
    result = runtime_budget_limited_result(
        out_dir=tmp_path / "output" / "formal",
        study_manifest_id="manifest-1",
        program_id=program_id,
        program_family="family",
        groups=_groups(_actions()),
        provenance=_provenance(),
    )
    return program_result_payload(
        result,
        program_id=program_id,
        program_status="coverage_limitation",
        limitation_kind="runtime_budget_exceeded",
        run_control_provenance=_provenance(),
    )


@pytest.mark.parametrize(
    "mutation",
    ("string_budget", "string_observed", "below_budget", "out_of_frozen_skip"),
)
def test_checkpoint_runtime_fallback_reuses_strict_run_control_validation(
    tmp_path: Path, mutation: str
) -> None:
    provenance = copy.deepcopy(_provenance())
    control = provenance["control_payload"]
    assert isinstance(control, dict)
    skips = control["skip_programs"]
    assert isinstance(skips, list) and isinstance(skips[0], dict)
    if mutation == "string_budget":
        control["program_wall_time_budget_s"] = "600"
        provenance["program_wall_time_budget_s"] = "600"
    elif mutation == "string_observed":
        skips[0]["observed_wall_time_s"] = "731.25"
        provenance["observed_wall_time_s"] = "731.25"
    elif mutation == "below_budget":
        skips[0]["observed_wall_time_s"] = 599
        provenance["observed_wall_time_s"] = 599
    else:
        skips.append(
            {
                "program_id": "p2",
                "limitation_kind": "runtime_budget_exceeded",
                "reason": "out of frozen set",
                "observed_wall_time_s": 731.25,
            }
        )
    control_without_id = {
        key: value for key, value in control.items() if key != "control_id"
    }
    control["control_id"] = hashlib.sha256(
        json.dumps(
            control_without_id,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    provenance["control_id"] = control["control_id"]
    provenance["control_file_sha256"] = hashlib.sha256(
        (
            json.dumps(
                control, ensure_ascii=True, sort_keys=True, separators=(",", ":")
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    result = runtime_budget_limited_result(
        out_dir=tmp_path / "output" / "formal",
        study_manifest_id="manifest-1",
        program_id="p1",
        program_family="family",
        groups=_groups(_actions()),
        provenance=provenance,
    )

    with pytest.raises(ValueError, match="runtime fallback.*control|run control"):
        program_result_payload(
            result,
            program_id="p1",
            program_status="coverage_limitation",
            limitation_kind="runtime_budget_exceeded",
            run_control_provenance=provenance,
        )
def test_default_control_is_explicit_self_hashed_and_report_only(tmp_path: Path) -> None:
    control = ensure_run_control(
        tmp_path,
        study_manifest_id="manifest-1",
        program_ids=("p1", "p2"),
    )

    assert control.program_wall_time_budget_s == DEFAULT_PROGRAM_WALL_TIME_BUDGET_S == 600
    assert control.enforcement_mode == "external_program_boundary_skip"
    assert control.decision_for("p1").decision == "execute"
    assert control.program_ids == ("p1", "p2")
    with pytest.raises(ValueError, match="frozen program"):
        control.decision_for("not-frozen")
    raw = json.loads((tmp_path / "run_control.json").read_text(encoding="utf-8"))
    assert raw["frozen_program_ids"] == ["p1", "p2"]
    assert len(raw["control_id"]) == 64
    assert raw["authority_granted"] is False
    assert raw["proved_commute"] is False


def test_control_tampering_and_unknown_program_fail_closed(tmp_path: Path) -> None:
    ensure_run_control(tmp_path, study_manifest_id="manifest-1", program_ids=("p1",))
    path = tmp_path / "run_control.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["program_wall_time_budget_s"] = 601
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="control_id"):
        load_run_control(path, study_manifest_id="manifest-1", program_ids=("p1",))

    path.unlink()
    ensure_run_control(tmp_path, study_manifest_id="manifest-1", program_ids=("p1",))
    with pytest.raises(ValueError, match="frozen program"):
        add_runtime_budget_skip(
            tmp_path,
            study_manifest_id="manifest-1",
            program_ids=("p1",),
            program_id="not-frozen",
            observed_wall_time_s=600,
        )


def test_concurrent_runtime_skip_updates_are_kernel_locked_and_never_lose_an_entry(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "output" / "formal"
    ensure_run_control(
        out_dir,
        study_manifest_id="manifest-concurrent",
        program_ids=("p1", "p2"),
    )
    context = multiprocessing.get_context("spawn")
    ready = (context.Event(), context.Event())
    start = context.Event()
    results = context.Queue()
    writers = [
        context.Process(
            target=_concurrent_runtime_skip_writer,
            args=(str(out_dir), program_id, ready[index], start, results),
        )
        for index, program_id in enumerate(("p1", "p2"))
    ]
    for writer in writers:
        writer.start()
    try:
        assert all(event.wait(15) for event in ready)
        start.set()
        for writer in writers:
            writer.join(15)
    finally:
        for writer in writers:
            if writer.is_alive():
                writer.terminate()
                writer.join(5)

    assert [writer.exitcode for writer in writers] == [0, 0]
    assert sorted(results.get(timeout=5) for _ in writers) == ["ok", "ok"]
    control = load_run_control(
        out_dir / "run_control.json",
        study_manifest_id="manifest-concurrent",
        program_ids=("p1", "p2"),
    )
    assert set(control.skip_programs) == {"p1", "p2"}


def test_atomic_skip_update_is_auditable_and_read_at_program_boundary(tmp_path: Path) -> None:
    first = ensure_run_control(
        tmp_path, study_manifest_id="manifest-1", program_ids=("p1", "p2")
    )
    assert first.decision_for("p2").decision == "execute"

    updated = add_runtime_budget_skip(
        tmp_path,
        study_manifest_id="manifest-1",
        program_ids=("p1", "p2"),
        program_id="p2",
        observed_wall_time_s=731.25,
        reason="external monitor exceeded the frozen program budget",
    )

    decision = load_run_control(
        tmp_path / "run_control.json",
        study_manifest_id="manifest-1",
        program_ids=("p1", "p2"),
    ).decision_for("p2")
    assert decision.decision == "skip"
    assert decision.limitation_kind == "runtime_budget_exceeded"
    assert decision.observed_wall_time_s == 731.25
    assert decision.control_id == updated.control_id != first.control_id
    assert not list(tmp_path.glob(".run_control.*.tmp"))


def test_runtime_budget_skip_retains_full_denominators_and_never_authorizes() -> None:
    actions = _actions()
    groups = _groups(actions)
    provenance = _provenance()

    result = runtime_budget_limited_result(
        out_dir=Path("output/formal"),
        study_manifest_id="manifest-1",
        program_id="p1",
        program_family="family",
        groups=groups,
        provenance=provenance,
    )

    assert len(result.profile_rows["p1"]) == 3
    assert len(result.pair_views["Uall"]) == 3
    assert len(result.pair_views["U14"]) == 1
    assert len(result.pair_views["U30"]) == 3
    assert len(result.two_n_results["Uall"]["directional_rows"]) == 3
    assert len(result.two_n_results["Uall"]["pair_rows"]) == 3
    assert all(row["execution_status"] == "timeout" for row in result.profile_rows["p1"])
    assert all(row["dynamic_result"] == "timeout" for row in result.pair_views["Uall"])
    assert all(
        row["group_authorization_status"] == "group_precondition_unavailable"
        for group in result.two_n_results.values()
        for row in group["group_rows"]
    )
    assert all(
        row["false_authorization"] == "false"
        and row["authority_granted"] == "false"
        and row["proved_commute"] == "false"
        for group in result.two_n_results.values()
        for row in group["pair_rows"]
    )
    assert result.false_authorizations == ()


def test_runtime_budget_rows_pass_existing_complete_raw_coverage_gate() -> None:
    actions = _actions()
    groups = _groups(actions)
    result = runtime_budget_limited_result(
        out_dir=Path("output/formal"),
        study_manifest_id="manifest-1",
        program_id="p1",
        program_family="family",
        groups=groups,
        provenance=_provenance(),
    )
    tables = {
        "single_pass_observations.csv": [
            cli._project_row("single_pass_observations.csv", row)
            for row in result.profile_rows["p1"]
        ],
        "pair_observations.csv": [
            cli._project_row("pair_observations.csv", row)
            for row in result.pair_views["Uall"]
        ],
        "advisor_2n_group_results.csv": [
            cli._project_row("advisor_2n_group_results.csv", row)
            for group in ("U14", "U30", "Uall")
            for row in result.two_n_results[group]["group_rows"]
        ],
        "advisor_2n_directional_results.csv": [
            cli._project_row("advisor_2n_directional_results.csv", row)
            for group in ("U14", "U30", "Uall")
            for row in result.two_n_results[group]["directional_rows"]
        ],
        "advisor_2n_pair_validation.csv": [
            cli._project_row("advisor_2n_pair_validation.csv", row)
            for group in ("U14", "U30", "Uall")
            for row in result.two_n_results[group]["pair_rows"]
        ],
    }
    frozen = cli.FrozenPhase(
        out_dir=Path("output/formal"),
        manifest_path=Path("output/formal/study_manifest.json"),
        study_manifest_id="manifest-1",
        program_count=1,
        program_ids=("p1",),
        groups={
            group: tuple(action.action_id for action in configured)
            for group, configured in groups.items()
        },
        jobs=1,
        timeout_s=3,
    )

    cli._validate_raw_coverage(tables, frozen)


def test_runtime_and_checkpoint_identity_reject_u14_not_nested_in_u30() -> None:
    actions = _actions()
    non_nested = {
        "U14": (actions["A"], actions["B"]),
        "U30": (actions["A"], actions["C"]),
        "Uall": tuple(actions.values()),
    }
    with pytest.raises(ValueError, match="U14.*U30.*Uall"):
        runtime_budget_limited_result(
            out_dir=Path("output/formal"),
            study_manifest_id="manifest-1",
            program_id="p1",
            program_family="family",
            groups=non_nested,
            provenance=_provenance(),
        )
    with pytest.raises(ValueError, match="U14.*U30.*Uall"):
        program_checkpoint_input_sha256(
            study_manifest_id="manifest-1",
            program_id="p1",
            root_ir_sha256="d" * 64,
            group_action_ids={
                group: tuple(action.action_id for action in configured)
                for group, configured in non_nested.items()
            },
            runner_semantics_id="semantics-v1",
        )


def test_program_checkpoint_is_versioned_hash_valid_and_complete_is_reusable(
    tmp_path: Path,
) -> None:
    skeleton = {
        "schema_version": "advisor-pair-scale-2n/program-checkpoint-v1",
        "study_manifest_id": "manifest-1",
        "program_id": "p1",
        "program_status": "coverage_limitation",
        "limitation_kind": "runtime_budget_exceeded",
        "tables": {"single_pass_observations.csv": [{"program_id": "p1"}]},
        "cleanup_ledger": {"cleanup_state": "complete"},
        "summary": {
            "program_denominator": 1,
            "completed_program_count": 0,
            "coverage_limitation_program_count": 1,
        },
        "authority_granted": False,
        "proved_commute": False,
    }
    checkpoint = tmp_path / "checkpoint"
    with pytest.raises(ValueError, match="checkpoint"):
        publish_program_checkpoint(
            checkpoint,
            skeleton,
            expected_input_sha256="a" * 64,
            checkpoint_state="complete",
            isolation_root=tmp_path,
        )

    payload = _runtime_payload(tmp_path)
    completed = finalize_program_evidence(
        checkpoint,
        payload,
        expected_input_sha256="a" * 64,
        isolation_root=tmp_path,
    )

    loaded = load_program_checkpoint(
        checkpoint,
        expected_input_sha256="a" * 64,
        isolation_root=tmp_path,
    )
    assert loaded == (completed, "complete")
    pointer = json.loads((checkpoint / "active.json").read_text(encoding="utf-8"))
    version = checkpoint / pointer["version_path"]
    assert (version / "result.json").is_file()
    assert (version / "complete.json").is_file()

    # Rehashing only the payload is insufficient: the immutable version's
    # completion hashes and active pointer binding are both checked.
    (version / "result.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint"):
        load_program_checkpoint(
            checkpoint,
            expected_input_sha256="a" * 64,
            isolation_root=tmp_path,
        )


def test_program_finalize_publishes_planned_before_safe_cleanup_and_keeps_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import advisor_study.program_runtime as runtime

    actions = _actions()
    groups = _groups(actions)
    result = runtime_budget_limited_result(
        out_dir=tmp_path / "output" / "formal",
        study_manifest_id="manifest-1",
        program_id="p1",
        program_family="family",
        groups=groups,
        provenance=_provenance(),
    )
    payload = program_result_payload(
        result,
        program_id="p1",
        program_status="coverage_limitation",
        limitation_kind="runtime_budget_exceeded",
        run_control_provenance=_provenance(),
    )
    checkpoint = tmp_path / "output" / "formal" / "raw" / "program-checkpoints" / "p1"
    original = runtime.compact_intermediate_artifacts
    observed_states: list[str] = []

    def observing_cleanup(**kwargs: object):
        loaded = load_program_checkpoint(
            checkpoint,
            expected_input_sha256="a" * 64,
            isolation_root=tmp_path / "output" / "formal",
        )
        assert loaded is not None
        observed_states.append(loaded[1])
        return original(**kwargs)

    monkeypatch.setattr(runtime, "compact_intermediate_artifacts", observing_cleanup)
    completed = finalize_program_evidence(
        checkpoint,
        payload,
        expected_input_sha256="a" * 64,
        isolation_root=tmp_path / "output" / "formal",
    )

    assert observed_states == ["planned"]
    assert completed["cleanup_ledger"]["cleanup_state"] == "complete"
    assert completed["summary"]["program_denominator"] == 1
    assert len(completed["tables"]["pair_observations.csv"]) == 3
    assert len(completed["tables"]["advisor_2n_pair_validation.csv"]) == 7
    assert all(
        row["cleanup_status"] == "retained_runtime_budget_exceeded"
        for row in completed["tables"]["pair_observations.csv"]
    )
    assert all(
        row["cleanup_status"] == "retained_runtime_budget_exceeded"
        for row in completed["tables"]["advisor_2n_directional_results.csv"]
    )
    assert load_program_checkpoint(
        checkpoint,
        expected_input_sha256="a" * 64,
        isolation_root=tmp_path / "output" / "formal",
    )[1] == "complete"


def test_complete_program_checkpoint_resumes_without_rerunning_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import advisor_study.program_runtime as runtime

    checkpoint = tmp_path / "output" / "formal" / "raw" / "program-checkpoints" / "p1"
    payload = _runtime_payload(tmp_path)
    completed = finalize_program_evidence(
        checkpoint,
        payload,
        expected_input_sha256="a" * 64,
        isolation_root=tmp_path / "output" / "formal",
    )
    monkeypatch.setattr(
        runtime,
        "compact_intermediate_artifacts",
        lambda **_kwargs: pytest.fail("completed program cleanup was rerun"),
    )

    assert finalize_program_evidence(
        checkpoint,
        payload,
        expected_input_sha256="a" * 64,
        isolation_root=tmp_path / "output" / "formal",
    ) == completed


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_table",
        "wrong_row_counts",
        "zero_denominator",
        "incomplete_cleanup",
        "missing_group",
        "authority_on",
        "false_authorization_without_witness",
    ),
)
def test_complete_checkpoint_rejects_coverage_summary_cleanup_and_witness_tampering(
    tmp_path: Path, mutation: str
) -> None:
    source_checkpoint = tmp_path / "source"
    payload = _runtime_payload(tmp_path)
    completed = finalize_program_evidence(
        source_checkpoint,
        payload,
        expected_input_sha256="a" * 64,
        isolation_root=tmp_path,
    )
    altered = copy.deepcopy(completed)
    if mutation == "missing_table":
        altered["tables"].pop("pair_observations.csv")
    elif mutation == "wrong_row_counts":
        altered["summary"]["row_counts"]["pair_observations.csv"] += 1
    elif mutation == "zero_denominator":
        altered["summary"]["program_denominator"] = 0
    elif mutation == "incomplete_cleanup":
        altered["cleanup_ledger"]["cleanup_state"] = "planned"
    elif mutation == "missing_group":
        altered["pair_views"].pop("U14")
    elif mutation == "authority_on":
        altered["tables"]["pair_observations.csv"][0]["authority_granted"] = "true"
    else:
        altered["two_n_results"]["Uall"]["pair_rows"][0]["false_authorization"] = "true"
        altered["tables"]["advisor_2n_pair_validation.csv"][0]["false_authorization"] = "true"
    with pytest.raises(ValueError, match="checkpoint"):
        publish_program_checkpoint(
            tmp_path / f"tampered-{mutation}",
            altered,
            expected_input_sha256="b" * 64,
            checkpoint_state="complete",
            isolation_root=tmp_path,
        )


def test_complete_checkpoint_revalidates_materialized_profile_artifact_on_every_load(
    tmp_path: Path,
) -> None:
    result = runtime_budget_limited_result(
        out_dir=tmp_path / "output" / "formal",
        study_manifest_id="manifest-1",
        program_id="p1",
        program_family="family",
        groups=_groups(_actions()),
        provenance=_provenance(),
    )
    artifact = tmp_path / "output" / "formal" / "raw" / "profile" / "A.ll"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("define i32 @f() { ret i32 0 }\n", encoding="utf-8")
    row = result.profile_rows["p1"][0]
    row.update(
        {
            "execution_status": "success",
            "activity_status": "active",
            "output_path": str(artifact),
            "output_sha256": cli._sha256_file(artifact),
            "output_hard_state_id": cli._phasebatch_hard_state_id(artifact),
            "artifact_available": "true",
            "artifact_materialized": "true",
        }
    )
    for group in ("U14", "U30", "Uall"):
        group_row = result.two_n_results[group]["group_rows"][0]
        group_row.update(
            {
                "successful_n": 1,
                "active_n": 1,
                "timeout_n": group_row["configured_n"] - 1,
            }
        )
        action_a = next(
            directional
            for directional in result.two_n_results[group]["directional_rows"]
            if directional["action_id"] == "A"
        )
        action_a["first_round_status"] = "success"
    result.stage_paths["p1:profiles"] = "raw/profile"
    pair_stage = result.out_dir / "raw" / "bound-stage-fixture" / "pairs"
    orchestration_module._run_or_reuse_stage(
        pair_stage,
        lambda _directory: {
            "rows": [dict(value) for value in result.pair_views["Uall"]]
        },
        expected_input_sha256=hashlib.sha256(b"bound-pair-fixture").hexdigest(),
        isolation_root=tmp_path,
    )
    result.stage_paths["p1:pairs:Uall"] = pair_stage.relative_to(
        result.out_dir
    ).as_posix()
    for group in ("U14", "U30", "Uall"):
        stage = result.out_dir / "raw" / "bound-stage-fixture" / "two-n" / group
        group_result = result.two_n_results[group]
        orchestration_module._run_or_reuse_stage(
            stage,
            lambda _directory, value=group_result: {
                key: [dict(row) for row in value[key]]
                for key in ("group_rows", "directional_rows", "pair_rows")
            },
            expected_input_sha256=hashlib.sha256(
                f"bound-two-n-fixture\0{group}".encode("utf-8")
            ).hexdigest(),
            isolation_root=tmp_path,
        )
        result.stage_paths[f"p1:two_n:{group}"] = stage.relative_to(
            result.out_dir
        ).as_posix()
    payload = program_result_payload(
        result,
        program_id="p1",
        program_status="coverage_limitation",
        limitation_kind="runtime_budget_exceeded",
        run_control_provenance=_provenance(),
    )
    checkpoint = tmp_path / "output" / "formal" / "raw" / "program-checkpoints" / "p1"
    finalize_program_evidence(
        checkpoint,
        payload,
        expected_input_sha256="f" * 64,
        isolation_root=tmp_path / "output" / "formal",
    )
    artifact.write_text("define i32 @f() { ret i32 1 }\n", encoding="utf-8")

    with pytest.raises(ValueError, match="materialized.*artifact"):
        load_program_checkpoint(
            checkpoint,
            expected_input_sha256="f" * 64,
            isolation_root=tmp_path / "output" / "formal",
        )


def test_bidirectional_false_authorization_requires_one_complete_case_per_endpoint(
    tmp_path: Path,
) -> None:
    import advisor_study.program_runtime as runtime

    payload = _runtime_payload(tmp_path)
    pair = payload["two_n_results"]["Uall"]["pair_rows"][0]
    pair.update(
        {
            "action_a_directional_status": "authorized_all_others",
            "action_b_directional_status": "authorized_all_others",
            "false_authorization": "true",
        }
    )
    table_pair = next(
        row
        for row in payload["tables"]["advisor_2n_pair_validation.csv"]
        if row["row_id"] == pair["row_id"]
    )
    table_pair.update(pair)
    payload["summary"]["table_sha256"]["advisor_2n_pair_validation.csv"] = hashlib.sha256(
        json.dumps(
            payload["tables"]["advisor_2n_pair_validation.csv"],
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    payload["false_authorizations"] = [
        {
            "case_id": "one-collapsed-case",
            "case": {
                "study_manifest_id": "manifest-1",
                "program_id": "p1",
                "group_id": "Uall",
                "authorized_action_id": pair["action_a_id"],
                "advisor_pair_row": dict(pair),
            },
        }
    ]

    with pytest.raises(
        ValueError,
        match="2N pair derived fields|authorized endpoint|replay family",
    ):
        runtime._validate_program_checkpoint_payload(
            payload, required_cleanup_state=None
        )


def test_finalize_rejects_empty_complete_skeleton_before_cleanup(tmp_path: Path) -> None:
    skeleton = {
        "schema_version": "advisor-pair-scale-2n/program-checkpoint-v1",
        "study_manifest_id": "manifest-1",
        "program_id": "p1",
        "program_status": "complete",
        "limitation_kind": "",
        "run_control_provenance": {},
        "profile_rows": [],
        "pair_views": {"U14": [], "U30": [], "Uall": []},
        "two_n_results": {
            group: {"group_rows": [], "directional_rows": [], "pair_rows": []}
            for group in ("U14", "U30", "Uall")
        },
        "false_authorizations": [],
        "stage_paths": {},
        "tables": {name: [] for name in (
            "single_pass_observations.csv",
            "pair_observations.csv",
            "advisor_2n_group_results.csv",
            "advisor_2n_directional_results.csv",
            "advisor_2n_pair_validation.csv",
        )},
        "cleanup_ledger": {"cleanup_state": "complete"},
        "summary": {
            "program_denominator": 1,
            "completed_program_count": 1,
            "coverage_limitation_program_count": 0,
            "row_counts": {},
            "table_sha256": {},
        },
        "authority_granted": False,
        "proved_commute": False,
    }
    with pytest.raises(ValueError, match="checkpoint"):
        finalize_program_evidence(
            tmp_path / "skeleton",
            skeleton,
            expected_input_sha256="a" * 64,
            isolation_root=tmp_path,
        )


def test_cleanup_updates_every_group_view_by_canonical_pair_identity(tmp_path: Path) -> None:
    import advisor_study.program_runtime as runtime

    payload = _runtime_payload(tmp_path)
    compact_pairs = copy.deepcopy(payload["pair_views"]["Uall"])
    target = next(
        row
        for row in compact_pairs
        if (row["action_a_id"], row["action_b_id"]) == ("A", "B")
    )
    target.update(
        {
            "ab_output_path": "",
            "ba_output_path": "",
            "artifact_available": "false",
            "artifact_materialized": "false",
            "cleanup_status": "reclaimed_nonwitness",
            "fail_closed_reason": "",
        }
    )
    directionals = {
        group: payload["two_n_results"][group]["directional_rows"]
        for group in ("U14", "U30", "Uall")
    }
    updated = runtime._with_cleanup(
        payload,
        pair_rows=compact_pairs,
        directional_rows_by_group=directionals,
        ledger={
            "cleanup_state": "complete",
            "study_manifest_id": "manifest-1",
            "authority_granted": False,
            "proved_commute": False,
            "entries": [],
            "summary": {},
        },
    )

    for group in ("U14", "U30", "Uall"):
        matching = next(
            row
            for row in updated["pair_views"][group]
            if (row["action_a_id"], row["action_b_id"]) == ("A", "B")
        )
        assert matching["artifact_materialized"] == "false"
        assert matching["ab_output_path"] == ""
        assert matching["ba_output_path"] == ""
        assert matching["cleanup_status"] == "reclaimed_nonwitness"


def test_checkpoint_identity_binds_runner_semantics_and_combines_without_row_loss(
    tmp_path: Path,
) -> None:
    first_identity = program_checkpoint_input_sha256(
        study_manifest_id="manifest-1",
        program_id="p1",
        root_ir_sha256="d" * 64,
        group_action_ids={"U14": ("A", "B"), "U30": ("A", "B", "C"), "Uall": ("A", "B", "C")},
        runner_semantics_id="worker-canonical-and-compare-v2",
    )
    second_identity = program_checkpoint_input_sha256(
        study_manifest_id="manifest-1",
        program_id="p1",
        root_ir_sha256="d" * 64,
        group_action_ids={"U14": ("A", "B"), "U30": ("A", "B", "C"), "Uall": ("A", "B", "C")},
        runner_semantics_id="worker-canonical-and-compare-v3",
    )
    assert first_identity != second_identity
    checkpoint = program_checkpoint_path(
        tmp_path / "output" / "formal", first_identity
    )
    assert checkpoint.parent.name == "program-checkpoints"
    assert checkpoint.name == first_identity[:16]

    actions = _actions()
    groups = _groups(actions)
    provenance = _provenance()
    results = tuple(
        runtime_budget_limited_result(
            out_dir=tmp_path / "output" / "formal",
            study_manifest_id="manifest-1",
            program_id=program_id,
            program_family="family",
            groups=groups,
            provenance=provenance,
        )
        for program_id in ("p1", "p2")
    )

    combined = combine_program_results(results)
    assert set(combined.profile_rows) == {"p1", "p2"}
    assert len(combined.pair_views["Uall"]) == 6
    assert len(combined.two_n_results["Uall"]["group_rows"]) == 2
    assert combined.false_authorizations == ()
