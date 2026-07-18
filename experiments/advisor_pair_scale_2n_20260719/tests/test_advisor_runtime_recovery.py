from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from advisor_study.orchestration import OrchestrationDependencies, run_study_orchestration
from advisor_study import program_runtime
from advisor_study.program_runtime import (
    finalize_program_evidence,
    load_program_checkpoint,
    plan_unpublished_staging_cleanup,
    program_result_payload,
    recover_runtime_budget_limited_program,
    runtime_budget_limited_result,
)
from advisor_study.schema import canonical_row_id


class _Action:
    def __init__(self, action_id: str) -> None:
        self.action_id = action_id


def _groups() -> dict[str, tuple[_Action, ...]]:
    actions = {name: _Action(name) for name in ("A", "B", "C")}
    return {
        "U14": (actions["A"], actions["B"]),
        "U30": tuple(actions.values()),
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
                "reason": "test supervisor timeout",
                "observed_wall_time_s": 611,
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
        "observed_wall_time_s": 611,
        "limitation_kind": "runtime_budget_exceeded",
        "reason": "test supervisor timeout",
        "enforcement_mode": "external_program_boundary_skip",
        "control_payload": control,
    }


def test_partial_stage_recovery_reuses_real_profiles_and_types_only_missing_stages(
    tmp_path: Path,
) -> None:
    manifest = "c" * 64
    out_dir = tmp_path / "output" / "formal"
    root = tmp_path / "root.ll"
    root.write_text("define i32 @f() { ret i32 0 }\n", encoding="utf-8")
    groups = _groups()
    profile_calls = 0

    def real_profiles(
        _root: Path, actions: tuple[object, ...], directory: Path
    ) -> list[dict[str, object]]:
        nonlocal profile_calls
        profile_calls += 1
        rows: list[dict[str, object]] = []
        for action in actions:
            action_id = str(getattr(action, "action_id"))
            status = "error" if action_id == "B" else "success"
            output = directory / action_id / "first.ll"
            digest = ""
            if status == "success":
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(root.read_text(encoding="utf-8"), encoding="utf-8")
                digest = hashlib.sha256(output.read_bytes()).hexdigest()
            rows.append(
                {
                    "row_id": canonical_row_id("profile", manifest, "p1", action_id),
                    "study_manifest_id": manifest,
                    "program_id": "p1",
                    "group_id": "Uall",
                    "action_id": action_id,
                    "execution_status": status,
                    "activity_status": "no_op" if action_id == "C" else "active",
                    "output_path": str(output) if status == "success" else "",
                    "output_sha256": digest,
                    "output_hard_state_id": digest,
                    "artifact_available": "true" if status == "success" else "false",
                    "artifact_materialized": "true" if status == "success" else "false",
                    "authority_granted": "false",
                    "proved_commute": "false",
                }
            )
        return rows

    dependencies = OrchestrationDependencies(
        profile_uall=real_profiles,
        run_uall_pairs=lambda *_args: (_ for _ in ()).throw(
            RuntimeError("injected timeout after profile publication")
        ),
        run_group_two_n=lambda *_args: pytest.fail("2N ran before injected timeout"),
        replay_worker=lambda *_args: pytest.fail("replay ran before false authorization"),
        replay_external_opt=lambda *_args: pytest.fail("replay ran before false authorization"),
        replay_two_n=lambda *_args: pytest.fail("replay ran before false authorization"),
    )
    with pytest.raises(RuntimeError, match="injected timeout"):
        run_study_orchestration(
            out_dir=out_dir,
            isolation_root=tmp_path,
            study_manifest_id=manifest,
            programs={"p1": root},
            groups=groups,
            dependencies=dependencies,
        )
    profile_complete = next((out_dir / "raw").rglob("profiles/complete.json"))
    profile_mtime = profile_complete.stat().st_mtime_ns
    replay_calls: list[str] = []

    recovered = recover_runtime_budget_limited_program(
        out_dir=out_dir,
        isolation_root=tmp_path,
        study_manifest_id=manifest,
        program_id="p1",
        root_ir=root,
        program_family="synthetic",
        groups=groups,
        provenance=_provenance(),
        replay_worker=lambda *_args: replay_calls.append("worker") or {},
        replay_external_opt=lambda *_args: replay_calls.append("external") or {},
        replay_two_n=lambda *_args: replay_calls.append("two_n") or {},
    )

    assert profile_calls == 1
    assert profile_complete.stat().st_mtime_ns == profile_mtime
    profiles = {row["action_id"]: row for row in recovered.profile_rows["p1"]}
    assert {key: row["execution_status"] for key, row in profiles.items()} == {
        "A": "success",
        "B": "error",
        "C": "success",
    }
    pairs = {
        (row["action_a_id"], row["action_b_id"]): row
        for row in recovered.pair_views["Uall"]
    }
    assert pairs[("A", "C")]["dynamic_result"] == "timeout"
    assert pairs[("A", "C")]["a_status"] == profiles["A"]["execution_status"]
    assert pairs[("A", "C")]["b_status"] == profiles["C"]["execution_status"]
    assert pairs[("A", "B")]["dynamic_result"] == "failed"
    assert pairs[("A", "B")]["b_status"] == profiles["B"]["execution_status"]
    assert "pair_precondition_failed" in str(pairs[("A", "B")]["fail_closed_reason"])
    assert all(
        str(row["fail_closed_reason"]).startswith("runtime_budget_exceeded")
        for group in recovered.two_n_results.values()
        for row in group["group_rows"]
    )
    for group in recovered.two_n_results.values():
        group_row = group["group_rows"][0]
        assert group_row["round1_status"] == "round1_precondition_failed"
        assert group_row["all_n_merge_status"] == "unknown"
        assert group_row["all_n_second_round_status"] == "unknown"
        assert all(
            row["directional_status"] == "round1_precondition_failed"
            and row["merged_input_status"] == "unknown"
            and row["second_round_status"] == "not_run"
            for row in group["directional_rows"]
        )
        assert all(
            row["two_n_pair_status"] == "group_precondition_unavailable"
            and row["action_a_directional_status"]
            == "round1_precondition_failed"
            and row["action_b_directional_status"]
            == "round1_precondition_failed"
            for row in group["pair_rows"]
        )
    assert replay_calls == []


def _limited_payload(tmp_path: Path, *, staging_plan: dict[str, object]) -> dict[str, object]:
    groups = _groups()
    result = runtime_budget_limited_result(
        out_dir=tmp_path / "output" / "formal",
        study_manifest_id="manifest-1",
        program_id="p1",
        program_family="synthetic",
        groups=groups,
        provenance=_provenance(),
    )
    stage_paths = {
        str(entry["stage_key"]): str(entry["final_stage_relative_path"])
        for entry in staging_plan["entries"]
    }
    result = replace(result, stage_paths=stage_paths)
    payload = program_result_payload(
        result,
        program_id="p1",
        program_status="coverage_limitation",
        limitation_kind="runtime_budget_exceeded",
        run_control_provenance=_provenance(),
        unpublished_staging_cleanup=staging_plan,
    )
    return payload


def test_unpublished_staging_is_planned_before_exact_delete_and_recorded_complete(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "output" / "formal"
    final_stage = out_dir / "raw" / "manifest" / "p" / "program" / "profiles"
    orphan = final_stage.parent / ".profiles.stage-interrupted"
    artifact = orphan / "nested" / "partial.ll"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("partial\n", encoding="utf-8")
    plan = plan_unpublished_staging_cleanup(
        isolation_root=out_dir,
        stage_paths={"p1:profiles": final_stage.relative_to(out_dir).as_posix()},
    )
    assert plan["cleanup_state"] == "planned"
    assert plan["entries"][0]["relative_path"].endswith(".profiles.stage-interrupted")
    assert plan["entries"][0]["files"][0]["sha256"] == hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    checkpoint = out_dir / "raw" / "program-checkpoints" / "p1"

    completed = finalize_program_evidence(
        checkpoint,
        _limited_payload(tmp_path, staging_plan=plan),
        expected_input_sha256="d" * 64,
        isolation_root=out_dir,
    )

    assert not orphan.exists()
    assert completed["unpublished_staging_cleanup"]["cleanup_state"] == "complete"
    assert completed["unpublished_staging_cleanup"]["entries"][0]["status"] == "reclaimed"


def test_unpublished_staging_hash_drift_is_rejected_after_planned_checkpoint(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "output" / "formal"
    final_stage = out_dir / "raw" / "m" / "p" / "x" / "profiles"
    orphan = final_stage.parent / ".profiles.stage-interrupted"
    artifact = orphan / "partial.ll"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("before\n", encoding="utf-8")
    plan = plan_unpublished_staging_cleanup(
        isolation_root=out_dir,
        stage_paths={"p1:profiles": final_stage.relative_to(out_dir).as_posix()},
    )
    artifact.write_text("tampered\n", encoding="utf-8")
    checkpoint = out_dir / "raw" / "program-checkpoints" / "p1"

    with pytest.raises(ValueError, match="unpublished staging.*drift"):
        finalize_program_evidence(
            checkpoint,
            _limited_payload(tmp_path, staging_plan=plan),
            expected_input_sha256="e" * 64,
            isolation_root=out_dir,
        )

    assert artifact.is_file()
    loaded = load_program_checkpoint(
        checkpoint,
        expected_input_sha256="e" * 64,
        isolation_root=out_dir,
    )
    assert loaded is not None and loaded[1] == "planned"


def test_forged_unpublished_staging_plan_cannot_target_non_stage_directory(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "output" / "formal"
    final_stage = out_dir / "raw" / "m" / "p" / "x" / "profiles"
    orphan = final_stage.parent / ".profiles.stage-interrupted"
    artifact = orphan / "partial.ll"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("planned\n", encoding="utf-8")
    plan = plan_unpublished_staging_cleanup(
        isolation_root=out_dir,
        stage_paths={"p1:profiles": final_stage.relative_to(out_dir).as_posix()},
    )
    victim = out_dir / "raw" / "must-not-delete"
    victim_file = victim / "valuable.ll"
    victim.mkdir(parents=True)
    victim_file.write_text("valuable\n", encoding="utf-8")

    entry = plan["entries"][0]
    entry["relative_path"] = victim.relative_to(out_dir).as_posix()
    identity = {
        "stage_key": entry["stage_key"],
        "final_stage_relative_path": entry["final_stage_relative_path"],
        "relative_path": entry["relative_path"],
        "reason": entry["reason"],
        "directories": entry["directories"],
        "files": entry["files"],
    }
    entry["cleanup_id"] = hashlib.sha256(
        program_runtime._canonical_json(identity).encode("utf-8")
    ).hexdigest()
    plan["cleanup_sha256"] = program_runtime._unpublished_cleanup_sha256(plan)

    with pytest.raises(ValueError, match="unpublished staging.*stage"):
        finalize_program_evidence(
            out_dir / "raw" / "program-checkpoints" / "p1-forged",
            _limited_payload(tmp_path, staging_plan=plan),
            expected_input_sha256="f" * 64,
            isolation_root=out_dir,
        )

    assert victim_file.read_text(encoding="utf-8") == "valuable\n"
