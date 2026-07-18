from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from advisor_study import cli
from advisor_study.orchestration import (
    OrchestrationDependencies,
    run_study_orchestration,
)
from advisor_study.direct_merge import derive_two_n_pair_fields
from advisor_study.program_runtime import (
    finalize_program_evidence,
    load_program_checkpoint,
    program_result_payload,
    runtime_budget_limited_result,
)
from advisor_study.schema import canonical_row_id


class _Action:
    def __init__(self, action_id: str) -> None:
        self.action_id = action_id


def _sha(value: str | bytes) -> str:
    data = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()


def _ir(value: int) -> str:
    return f"define i32 @f() {{\nentry:\n  ret i32 {value}\n}}\n"


def _write_ir(path: Path, value: int) -> tuple[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_ir(value), encoding="utf-8", newline="\n")
    return cli._sha256_file(path), cli._phasebatch_hard_state_id(path)


def _transport(command: list[str], stderr: str = "") -> dict[str, object]:
    return {
        "command": command,
        "command_sha256": _sha("\0".join(command)),
        "stderr": stderr,
        "stderr_sha256": _sha(stderr),
    }


def _dependencies(manifest: str, program: str) -> OrchestrationDependencies:
    def profile(
        _root: Path, actions: tuple[object, ...], directory: Path
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for index, action in enumerate(actions, start=1):
            action_id = str(getattr(action, "action_id"))
            output = directory / action_id / "first.ll"
            digest, hard = _write_ir(output, index * 10)
            rows.append(
                {
                    "row_id": canonical_row_id("profile", manifest, program, action_id),
                    "study_manifest_id": manifest,
                    "program_id": program,
                    "group_id": "Uall",
                    "action_id": action_id,
                    "execution_status": "success",
                    "activity_status": "active",
                    "output_path": str(output),
                    "output_sha256": digest,
                    "output_hard_state_id": hard,
                    "artifact_available": "true",
                    "artifact_materialized": "true",
                    "verifier_status": "success",
                    "logical_pass_applications": 1,
                    "physical_pass_invocations": 1,
                    "cache_reused": "false",
                    "authority_granted": "false",
                    "proved_commute": "false",
                    **_transport(["fake-worker", action_id, str(output)]),
                }
            )
        return rows

    def pairs(
        _root: Path,
        profiles: list[dict[str, object]],
        actions: dict[str, object],
        directory: Path,
    ) -> list[dict[str, object]]:
        profile_by_action = {str(row["action_id"]): row for row in profiles}
        rows: list[dict[str, object]] = []
        ordered = sorted(actions)
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                pair_dir = directory / f"{left}-{right}"
                ab = pair_dir / "AB.ll"
                ba = pair_dir / "BA.ll"
                ab_sha, ab_hard = _write_ir(ab, 111)
                ba_sha, ba_hard = _write_ir(ba, 222)
                rows.append(
                    {
                        "row_id": canonical_row_id("pair", manifest, program, left, right),
                        "study_manifest_id": manifest,
                        "program_id": program,
                        "group_id": "Uall",
                        "action_a_id": left,
                        "action_b_id": right,
                        "a_status": "success",
                        "b_status": "success",
                        "a_hard_state_id": profile_by_action[left]["output_hard_state_id"],
                        "a_output_sha256": profile_by_action[left]["output_sha256"],
                        "a_verifier_status": "success",
                        "b_hard_state_id": profile_by_action[right]["output_hard_state_id"],
                        "b_output_sha256": profile_by_action[right]["output_sha256"],
                        "b_verifier_status": "success",
                        "ab_status": "success",
                        "ab_output_path": str(ab),
                        "ab_output_sha256": ab_sha,
                        "ab_hard_state_id": ab_hard,
                        "ab_verifier_status": "success",
                        "ab_stderr_sha256": _sha(""),
                        "ba_status": "success",
                        "ba_output_path": str(ba),
                        "ba_output_sha256": ba_sha,
                        "ba_hard_state_id": ba_hard,
                        "ba_verifier_status": "success",
                        "ba_stderr_sha256": _sha(""),
                        "dynamic_result": "order_sensitive",
                        "observed_relation": "observed_disjoint",
                        "root_activity_class": "active_active",
                        "artifact_available": "true",
                        "artifact_materialized": "true",
                        "cleanup_status": "not_eligible",
                        "artifact_id": canonical_row_id("pair-artifact", left, right),
                        "authority_granted": "false",
                        "proved_commute": "false",
                        **_transport(["fake-pair", left, right]),
                    }
                )
        return rows

    def two_n(
        _root: Path,
        group_id: str,
        actions: dict[str, object],
        profiles: list[dict[str, object]],
        directory: Path,
        pair_view: list[dict[str, object]],
    ) -> dict[str, object]:
        authorized = group_id == "U14"
        directional_status = (
            "authorized_all_others" if authorized else "rejected_effect_changed"
        )
        directionals: list[dict[str, object]] = []
        for action_id in sorted(actions):
            merged = directory / action_id / "merged_input.ll"
            second = directory / action_id / "second_round.ll"
            merged_sha, merged_hard = _write_ir(merged, 30)
            second_sha, _second_hard = _write_ir(second, 40)
            directionals.append(
                {
                    "row_id": canonical_row_id(
                        "directional", manifest, program, group_id, action_id
                    ),
                    "study_manifest_id": manifest,
                    "program_id": program,
                    "group_id": group_id,
                    "action_id": action_id,
                    "directional_status": directional_status,
                    "first_round_status": "success",
                    "first_round_effect_sha256": _sha(f"first-{group_id}-{action_id}"),
                    "merged_input_status": "complete",
                    "merged_input_path": str(merged),
                    "merged_input_sha256": merged_sha,
                    "merged_input_hard_state_id": merged_hard,
                    "second_round_status": "success",
                    "second_round_effect_sha256": _sha(f"second-{group_id}-{action_id}"),
                    "second_output_path": str(second),
                    "second_output_sha256": second_sha,
                    "second_output_materialized": "true",
                    "other_contributions_preserved": "true",
                    "verifier_status": "success",
                    "cleanup_status": "not_eligible",
                    "artifact_id": canonical_row_id(
                        "directional-artifact", group_id, action_id
                    ),
                    "authority_granted": "false",
                    "proved_commute": "false",
                }
            )
        pair_rows: list[dict[str, object]] = []
        for observed in pair_view:
            left = str(observed["action_a_id"])
            right = str(observed["action_b_id"])
            is_false = authorized and (left, right) == ("A", "B")
            pair_rows.append(
                {
                    "row_id": canonical_row_id(
                        "two-n-pair", manifest, program, group_id, left, right
                    ),
                    "study_manifest_id": manifest,
                    "program_id": program,
                    "group_id": group_id,
                    "action_a_id": left,
                    "action_b_id": right,
                    "action_a_directional_status": directional_status,
                    "action_b_directional_status": directional_status,
                    "two_n_pair_status": (
                        "both_directions_authorized" if authorized else "both_rejected"
                    ),
                    "pair_observation_row_id": observed["row_id"],
                    "dynamic_result": observed["dynamic_result"],
                    "validation_status": (
                        "false_authorization"
                        if is_false
                        else "agree" if authorized else "unavailable"
                    ),
                    "false_authorization": "true" if is_false else "false",
                    "stable_false_authorization": "false",
                    "worker_replay_status": "not_required",
                    "external_opt_replay_status": "not_required",
                    "two_n_replay_status": "not_required",
                    "source_row_ids": str(observed["row_id"]),
                    "authority_granted": "false",
                    "proved_commute": "false",
                }
            )
        return {
            "group_row": {
                "row_id": canonical_row_id("two-n-group", manifest, program, group_id),
                "study_manifest_id": manifest,
                "program_id": program,
                "group_id": group_id,
                "configured_n": len(actions),
                "successful_n": len(actions),
                "active_n": len(actions),
                "no_op_n": 0,
                "failed_n": 0,
                "timeout_n": 0,
                "round1_status": "complete",
                "first_round_disjoint_status": "disjoint",
                "all_n_merge_status": "complete",
                "all_n_second_round_status": "complete",
                "group_authorization_status": "authorized" if authorized else "rejected",
                "directional_authorized_count": len(actions) if authorized else 0,
                "directional_unavailable_count": 0 if authorized else len(actions),
                "authority_granted": "false",
                "proved_commute": "false",
            },
            "directional_rows": directionals,
            "pair_rows": pair_rows,
        }

    def replay(kind: str):
        def run(
            case: dict[str, object], _repetition: int, directory: Path
        ) -> dict[str, object]:
            artifacts: dict[str, str] = {}
            artifact_sha: dict[str, str] = {}
            hard_states: dict[str, str] = {}
            values = {"S": 0, "A": 10, "B": 20, "AB": 111, "BA": 222, "merged_input": 30}
            if kind == "two_n":
                values["second_round_output"] = 40
            for name, value in values.items():
                path = directory / f"{name}.ll"
                digest, hard = _write_ir(path, value)
                artifacts[name] = str(path)
                artifact_sha[name] = digest
                hard_states[name] = hard
            stages = {
                name: {
                    "execution_status": "success",
                    "verifier_status": "success",
                    "hard_state_id": hard_states[name],
                    "output_sha256": artifact_sha[name],
                    "command_sha256": _sha(f"stage-command-{name}"),
                    "stderr_sha256": _sha(""),
                    "error_fingerprint": _sha(f"stage-success-{name}"),
                }
                for name in ("A", "B", "AB", "BA")
            }
            two_n_result: dict[str, str] = {}
            if kind == "two_n":
                two_n_result = {
                    str(key): str(value)
                    for key, value in dict(case["expected_two_n"]).items()
                }
                two_n_result["second_output_hard_state_id"] = hard_states[
                    "second_round_output"
                ]
            return {
                "status": "success",
                "hard_state_hashes": hard_states,
                "artifact_sha256": artifact_sha,
                "artifacts": artifacts,
                "stderr": "",
                "command": [f"fake-{kind}-replay"],
                "two_n_result": two_n_result,
                "stage_results": stages,
                "merge_status": "complete",
                "merge_error_fingerprint": "",
            }

        return run

    return OrchestrationDependencies(
        profile_uall=profile,
        run_uall_pairs=pairs,
        run_group_two_n=two_n,
        replay_worker=replay("worker"),
        replay_external_opt=replay("external_opt"),
        replay_two_n=replay("two_n"),
    )


@pytest.mark.parametrize("artifact_change", ("delete", "tamper"))
def test_production_checkpoint_chain_revalidates_real_false_auth_replay_artifact(
    tmp_path: Path, artifact_change: str
) -> None:
    manifest = "a" * 64
    program = "p1"
    out_dir = tmp_path / "output" / "formal"
    root = tmp_path / "root.ll"
    _write_ir(root, 0)
    actions = {name: _Action(name) for name in ("A", "B", "C")}
    groups = {
        "U14": (actions["A"], actions["B"]),
        "U30": tuple(actions.values()),
        "Uall": tuple(actions.values()),
    }
    result = run_study_orchestration(
        out_dir=out_dir,
        isolation_root=tmp_path,
        study_manifest_id=manifest,
        programs={program: root},
        groups=groups,
        dependencies=_dependencies(manifest, program),
    )
    assert len(result.false_authorizations) == 2
    payload = program_result_payload(
        result,
        program_id=program,
        program_status="complete",
        limitation_kind="",
        run_control_provenance={"decision": "execute"},
    )
    checkpoint = out_dir / "raw" / "program-checkpoints" / "production-chain"
    complete = finalize_program_evidence(
        checkpoint,
        payload,
        expected_input_sha256="b" * 64,
        isolation_root=out_dir,
    )
    loaded = load_program_checkpoint(
        checkpoint,
        expected_input_sha256="b" * 64,
        isolation_root=out_dir,
    )
    assert loaded == (complete, "complete")
    witness = complete["false_authorizations"][0]
    artifact = Path(witness["worker"][0]["artifacts"]["AB"])
    if artifact_change == "delete":
        artifact.unlink()
    else:
        artifact.write_text(_ir(999), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="materialized.*artifact"):
        load_program_checkpoint(
            checkpoint,
            expected_input_sha256="b" * 64,
            isolation_root=out_dir,
        )


def test_partial_pair_delete_failure_keeps_full_checkpoint_recoverably_planned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = "d" * 64
    program = "p1"
    out_dir = tmp_path / "output" / "formal"
    root = tmp_path / "root.ll"
    _write_ir(root, 0)
    actions = {name: _Action(name) for name in ("A", "B")}
    groups = {group: tuple(actions.values()) for group in ("U14", "U30", "Uall")}
    base = _dependencies(manifest, program)

    def commuting_pairs(
        root_ir: Path,
        profiles: list[dict[str, object]],
        action_map: dict[str, object],
        directory: Path,
    ) -> list[dict[str, object]]:
        rows = base.run_uall_pairs(root_ir, profiles, action_map, directory)
        for row in rows:
            ab = Path(str(row["ab_output_path"]))
            ba = Path(str(row["ba_output_path"]))
            ba.write_bytes(ab.read_bytes())
            row["ba_output_sha256"] = cli._sha256_file(ba)
            row["ba_hard_state_id"] = cli._phasebatch_hard_state_id(ba)
            row["dynamic_result"] = "commute"
        return rows

    def coherent_two_n(
        root_ir: Path,
        group_id: str,
        action_map: dict[str, object],
        profiles: list[dict[str, object]],
        directory: Path,
        pair_view: list[dict[str, object]],
    ) -> dict[str, object]:
        value = base.run_group_two_n(
            root_ir, group_id, action_map, profiles, directory, pair_view
        )
        directional = {
            str(row["action_id"]): str(row["directional_status"])
            for row in value["directional_rows"]
        }
        observed = {
            (str(row["action_a_id"]), str(row["action_b_id"])): row
            for row in pair_view
        }
        for row in value["pair_rows"]:
            key = (str(row["action_a_id"]), str(row["action_b_id"]))
            row.update(
                derive_two_n_pair_fields(
                    directional[key[0]],
                    directional[key[1]],
                    observed[key]["dynamic_result"],
                    observation_available=True,
                )
            )
        return value

    result = run_study_orchestration(
        out_dir=out_dir,
        isolation_root=tmp_path,
        study_manifest_id=manifest,
        programs={program: root},
        groups=groups,
        dependencies=replace(
            base,
            run_uall_pairs=commuting_pairs,
            run_group_two_n=coherent_two_n,
        ),
    )
    assert result.false_authorizations == ()
    payload = program_result_payload(
        result,
        program_id=program,
        program_status="complete",
        limitation_kind="",
        run_control_provenance={"decision": "execute"},
    )
    checkpoint = out_dir / "raw" / "program-checkpoints" / "partial-delete"
    original_unlink = Path.unlink
    ba_delete_attempts = 0

    def permanently_lock_ba(path: Path, *args: object, **kwargs: object) -> None:
        nonlocal ba_delete_attempts
        if "cleanup-quarantine" in path.parts and path.name == "BA.ll":
            ba_delete_attempts += 1
            raise PermissionError("injected permanent BA delete lock")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", permanently_lock_ba)
    for _attempt in range(2):
        with pytest.raises(RuntimeError, match="partial pair cleanup.*recoverable"):
            finalize_program_evidence(
                checkpoint,
                payload,
                expected_input_sha256="e" * 64,
                isolation_root=out_dir,
            )
        loaded = load_program_checkpoint(
            checkpoint,
            expected_input_sha256="e" * 64,
            isolation_root=out_dir,
        )
        assert loaded is not None and loaded[1] == "planned"
        markers = [
            json.loads(path.read_text(encoding="utf-8"))["checkpoint_state"]
            for path in checkpoint.rglob("complete.json")
        ]
        assert markers and set(markers) == {"planned"}

    assert ba_delete_attempts == 2
    monkeypatch.setattr(Path, "unlink", original_unlink)
    completed = finalize_program_evidence(
        checkpoint,
        payload,
        expected_input_sha256="e" * 64,
        isolation_root=out_dir,
    )
    pair_row = completed["pair_views"]["Uall"][0]
    assert pair_row["artifact_materialized"] == "false"
    assert pair_row["ab_output_path"] == ""
    assert pair_row["ba_output_path"] == ""
    loaded = load_program_checkpoint(
        checkpoint,
        expected_input_sha256="e" * 64,
        isolation_root=out_dir,
    )
    assert loaded == (completed, "complete")


def test_complete_checkpoint_rejects_successful_profile_hidden_by_false_materialized_flag(
    tmp_path: Path,
) -> None:
    manifest = "a" * 64
    program = "p1"
    out_dir = tmp_path / "output" / "formal"
    root = tmp_path / "root.ll"
    _write_ir(root, 0)
    actions = {name: _Action(name) for name in ("A", "B", "C")}
    groups = {
        "U14": (actions["A"], actions["B"]),
        "U30": tuple(actions.values()),
        "Uall": tuple(actions.values()),
    }
    result = run_study_orchestration(
        out_dir=out_dir,
        isolation_root=tmp_path,
        study_manifest_id=manifest,
        programs={program: root},
        groups=groups,
        dependencies=_dependencies(manifest, program),
    )
    result.profile_rows[program][0]["artifact_materialized"] = "false"
    with pytest.raises(ValueError, match="materialized artifact.*flag"):
        payload = program_result_payload(
            result,
            program_id=program,
            program_status="complete",
            limitation_kind="",
            run_control_provenance={"decision": "execute"},
        )
        finalize_program_evidence(
            out_dir / "raw" / "program-checkpoints" / "false-flag",
            payload,
            expected_input_sha256="c" * 64,
            isolation_root=out_dir,
        )


def test_complete_checkpoint_rejects_existing_pair_path_with_false_materialized_flag(
    tmp_path: Path,
) -> None:
    manifest = "a" * 64
    program = "p1"
    out_dir = tmp_path / "output" / "formal"
    root = tmp_path / "root.ll"
    _write_ir(root, 0)
    actions = {name: _Action(name) for name in ("A", "B", "C")}
    groups = {
        "U14": (actions["A"], actions["B"]),
        "U30": tuple(actions.values()),
        "Uall": tuple(actions.values()),
    }
    result = run_study_orchestration(
        out_dir=out_dir,
        isolation_root=tmp_path,
        study_manifest_id=manifest,
        programs={program: root},
        groups=groups,
        dependencies=_dependencies(manifest, program),
    )
    for rows in result.pair_views.values():
        for row in rows:
            if (row["action_a_id"], row["action_b_id"]) == ("A", "B"):
                row["artifact_materialized"] = "false"
    payload = program_result_payload(
        result,
        program_id=program,
        program_status="complete",
        limitation_kind="",
        run_control_provenance={"decision": "execute"},
    )

    with pytest.raises(ValueError, match="materialized artifact flag is false"):
        finalize_program_evidence(
            out_dir / "raw" / "program-checkpoints" / "false-pair-flag",
            payload,
            expected_input_sha256="e" * 64,
            isolation_root=out_dir,
        )


def test_checkpoint_rejects_false_authorization_pair_first_round_cross_binding_drift(
    tmp_path: Path,
) -> None:
    manifest = "a" * 64
    program = "p1"
    out_dir = tmp_path / "output" / "formal"
    root = tmp_path / "root.ll"
    _write_ir(root, 0)
    actions = {name: _Action(name) for name in ("A", "B", "C")}
    groups = {
        "U14": (actions["A"], actions["B"]),
        "U30": tuple(actions.values()),
        "Uall": tuple(actions.values()),
    }
    result = run_study_orchestration(
        out_dir=out_dir,
        isolation_root=tmp_path,
        study_manifest_id=manifest,
        programs={program: root},
        groups=groups,
        dependencies=_dependencies(manifest, program),
    )
    payload = program_result_payload(
        result,
        program_id=program,
        program_status="complete",
        limitation_kind="",
        run_control_provenance={"decision": "execute"},
    )
    payload["false_authorizations"][0]["case"]["pair_observation"][
        "a_hard_state_id"
    ] = "f" * 64

    with pytest.raises(ValueError, match="first-round.*profile"):
        finalize_program_evidence(
            out_dir / "raw" / "program-checkpoints" / "cross-binding-drift",
            payload,
            expected_input_sha256="f" * 64,
            isolation_root=out_dir,
        )


@pytest.mark.parametrize("shared_evidence", ("artifact_path", "command"))
def test_false_authorization_replay_families_require_independent_evidence(
    tmp_path: Path, shared_evidence: str
) -> None:
    manifest = "a" * 64
    program = "p1"
    out_dir = tmp_path / "output" / "formal"
    root = tmp_path / "root.ll"
    _write_ir(root, 0)
    actions = {name: _Action(name) for name in ("A", "B", "C")}
    groups = {
        "U14": (actions["A"], actions["B"]),
        "U30": tuple(actions.values()),
        "Uall": tuple(actions.values()),
    }
    result = run_study_orchestration(
        out_dir=out_dir,
        isolation_root=tmp_path,
        study_manifest_id=manifest,
        programs={program: root},
        groups=groups,
        dependencies=_dependencies(manifest, program),
    )
    witness = result.false_authorizations[0]
    for repetition in range(2):
        worker = witness["worker"][repetition]
        external = witness["external_opt"][repetition]
        if shared_evidence == "artifact_path":
            external["artifacts"] = dict(worker["artifacts"])
            external["artifact_sha256"] = dict(worker["artifact_sha256"])
            external["hard_state_hashes"] = dict(worker["hard_state_hashes"])
        else:
            external["command"] = list(worker["command"])
            external["command_sha256"] = worker["command_sha256"]
    with pytest.raises(ValueError, match="replay families.*independent"):
        payload = program_result_payload(
            result,
            program_id=program,
            program_status="complete",
            limitation_kind="",
            run_control_provenance={"decision": "execute"},
        )
        finalize_program_evidence(
            out_dir / "raw" / "program-checkpoints" / f"shared-{shared_evidence}",
            payload,
            expected_input_sha256="d" * 64,
            isolation_root=out_dir,
        )


def test_checkpoint_recomputes_false_authorization_from_directionals_and_pair(
    tmp_path: Path,
) -> None:
    manifest = "a" * 64
    program = "p1"
    out_dir = tmp_path / "output" / "formal"
    root = tmp_path / "root.ll"
    _write_ir(root, 0)
    actions = {name: _Action(name) for name in ("A", "B", "C")}
    groups = {
        "U14": (actions["A"], actions["B"]),
        "U30": tuple(actions.values()),
        "Uall": tuple(actions.values()),
    }
    result = run_study_orchestration(
        out_dir=out_dir,
        isolation_root=tmp_path,
        study_manifest_id=manifest,
        programs={program: root},
        groups=groups,
        dependencies=_dependencies(manifest, program),
    )
    target = next(
        row
        for row in result.two_n_results["U14"]["pair_rows"]
        if (row["action_a_id"], row["action_b_id"]) == ("A", "B")
    )
    target["validation_status"] = "agree"
    target["false_authorization"] = "false"
    tampered = replace(result, false_authorizations=())

    with pytest.raises(ValueError, match="2N pair derived fields|2N replay writeback"):
        program_result_payload(
            tampered,
            program_id=program,
            program_status="complete",
            limitation_kind="",
            run_control_provenance={"decision": "execute"},
        )


def test_checkpoint_rejects_filtered_pair_view_semantic_drift_from_uall(
    tmp_path: Path,
) -> None:
    manifest = "a" * 64
    program = "p1"
    out_dir = tmp_path / "output" / "formal"
    root = tmp_path / "root.ll"
    _write_ir(root, 0)
    actions = {name: _Action(name) for name in ("A", "B", "C")}
    groups = {
        "U14": (actions["A"], actions["B"]),
        "U30": tuple(actions.values()),
        "Uall": tuple(actions.values()),
    }
    result = run_study_orchestration(
        out_dir=out_dir,
        isolation_root=tmp_path,
        study_manifest_id=manifest,
        programs={program: root},
        groups=groups,
        dependencies=_dependencies(manifest, program),
    )
    u14_view = next(
        row
        for row in result.pair_views["U14"]
        if (row["action_a_id"], row["action_b_id"]) == ("A", "B")
    )
    u14_view["dynamic_result"] = "commute"
    pair_row = next(
        row
        for row in result.two_n_results["U14"]["pair_rows"]
        if (row["action_a_id"], row["action_b_id"]) == ("A", "B")
    )
    pair_row["dynamic_result"] = "commute"
    pair_row["validation_status"] = "agree"
    pair_row["false_authorization"] = "false"
    tampered = replace(result, false_authorizations=())

    with pytest.raises(
        ValueError, match="filtered pair view.*Uall|2N replay writeback"
    ):
        program_result_payload(
            tampered,
            program_id=program,
            program_status="complete",
            limitation_kind="",
            run_control_provenance={"decision": "execute"},
        )


def test_checkpoint_binds_pair_semantics_to_original_uall_pair_stage(
    tmp_path: Path,
) -> None:
    manifest = "a" * 64
    program = "p1"
    out_dir = tmp_path / "output" / "formal"
    root = tmp_path / "root.ll"
    _write_ir(root, 0)
    actions = {name: _Action(name) for name in ("A", "B", "C")}
    groups = {
        "U14": (actions["A"], actions["B"]),
        "U30": tuple(actions.values()),
        "Uall": tuple(actions.values()),
    }
    result = run_study_orchestration(
        out_dir=out_dir,
        isolation_root=tmp_path,
        study_manifest_id=manifest,
        programs={program: root},
        groups=groups,
        dependencies=_dependencies(manifest, program),
    )
    for group in ("U14", "U30", "Uall"):
        observed = next(
            row
            for row in result.pair_views[group]
            if (row["action_a_id"], row["action_b_id"]) == ("A", "B")
        )
        observed["dynamic_result"] = "commute"
        validation = next(
            row
            for row in result.two_n_results[group]["pair_rows"]
            if (row["action_a_id"], row["action_b_id"]) == ("A", "B")
        )
        validation.update(
            {
                "dynamic_result": "commute",
                "validation_status": "agree" if group == "U14" else "unavailable",
                "false_authorization": "false",
                "stable_false_authorization": "false",
                "worker_replay_status": "not_required",
                "external_opt_replay_status": "not_required",
                "two_n_replay_status": "not_required",
            }
        )
    tampered = replace(result, false_authorizations=())

    with pytest.raises(ValueError, match="pair stage result binding"):
        program_result_payload(
            tampered,
            program_id=program,
            program_status="complete",
            limitation_kind="",
            run_control_provenance={"decision": "execute"},
        )


@pytest.mark.parametrize("field", ("ab_output_sha256", "artifact_id"))
def test_checkpoint_pair_stage_binding_keeps_noncleanup_artifact_identity(
    tmp_path: Path, field: str
) -> None:
    manifest = "a" * 64
    program = "p1"
    out_dir = tmp_path / "output" / "formal"
    root = tmp_path / "root.ll"
    _write_ir(root, 0)
    actions = {name: _Action(name) for name in ("A", "B", "C")}
    groups = {
        "U14": (actions["A"], actions["B"]),
        "U30": tuple(actions.values()),
        "Uall": tuple(actions.values()),
    }
    result = run_study_orchestration(
        out_dir=out_dir,
        isolation_root=tmp_path,
        study_manifest_id=manifest,
        programs={program: root},
        groups=groups,
        dependencies=_dependencies(manifest, program),
    )
    for group in ("U30", "Uall"):
        row = next(
            value
            for value in result.pair_views[group]
            if (value["action_a_id"], value["action_b_id"]) == ("A", "C")
        )
        row[field] = "f" * 64

    with pytest.raises(ValueError, match="pair stage result binding"):
        program_result_payload(
            result,
            program_id=program,
            program_status="complete",
            limitation_kind="",
            run_control_provenance={"decision": "execute"},
        )


def test_coverage_checkpoint_cannot_erase_completed_stage_history_as_runtime_fallback(
    tmp_path: Path,
) -> None:
    manifest = "a" * 64
    program = "p1"
    out_dir = tmp_path / "output" / "formal"
    root = tmp_path / "root.ll"
    _write_ir(root, 0)
    actions = {name: _Action(name) for name in ("A", "B", "C")}
    groups = {
        "U14": (actions["A"], actions["B"]),
        "U30": tuple(actions.values()),
        "Uall": tuple(actions.values()),
    }
    completed = run_study_orchestration(
        out_dir=out_dir,
        isolation_root=tmp_path,
        study_manifest_id=manifest,
        programs={program: root},
        groups=groups,
        dependencies=_dependencies(manifest, program),
    )
    provenance = {
        "control_id": "c" * 64,
        "control_file_sha256": "f" * 64,
        "program_wall_time_budget_s": 600,
        "observed_wall_time_s": 600,
        "limitation_kind": "runtime_budget_exceeded",
        "reason": "typed supervisor budget",
    }
    downgraded = runtime_budget_limited_result(
        out_dir=out_dir,
        study_manifest_id=manifest,
        program_id=program,
        program_family="synthetic",
        groups=groups,
        provenance=provenance,
    )

    original_profiles = {
        str(row["action_id"]): row for row in completed.profile_rows[program]
    }
    for row in downgraded.profile_rows[program]:
        row["row_id"] = original_profiles[str(row["action_id"])]["row_id"]
    original_pairs = {
        (str(row["action_a_id"]), str(row["action_b_id"])): row
        for row in completed.pair_views["Uall"]
    }
    for group in ("U14", "U30", "Uall"):
        for row in downgraded.pair_views[group]:
            identity = (str(row["action_a_id"]), str(row["action_b_id"]))
            row["row_id"] = original_pairs[identity]["row_id"]
        original_group = completed.two_n_results[group]
        limited_group = downgraded.two_n_results[group]
        limited_group["group_rows"][0]["row_id"] = original_group["group_rows"][0][
            "row_id"
        ]
        original_directionals = {
            str(row["action_id"]): row for row in original_group["directional_rows"]
        }
        for row in limited_group["directional_rows"]:
            row["row_id"] = original_directionals[str(row["action_id"])]["row_id"]
        original_validation = {
            (str(row["action_a_id"]), str(row["action_b_id"])): row
            for row in original_group["pair_rows"]
        }
        for row in limited_group["pair_rows"]:
            identity = (str(row["action_a_id"]), str(row["action_b_id"]))
            pair_row_id = original_pairs[identity]["row_id"]
            row["row_id"] = original_validation[identity]["row_id"]
            row["pair_observation_row_id"] = pair_row_id
            row["source_row_ids"] = pair_row_id

    with pytest.raises(
        ValueError, match="canonical runtime fallback|stage result binding"
    ):
        program_result_payload(
            downgraded,
            program_id=program,
            program_status="coverage_limitation",
            limitation_kind="runtime_budget_exceeded",
            run_control_provenance=provenance,
        )


def test_coverage_checkpoint_cannot_hide_completed_group_false_authorization(
    tmp_path: Path,
) -> None:
    manifest = "a" * 64
    program = "p1"
    out_dir = tmp_path / "output" / "formal"
    root = tmp_path / "root.ll"
    _write_ir(root, 0)
    actions = {name: _Action(name) for name in ("A", "B", "C")}
    groups = {
        "U14": (actions["A"], actions["B"]),
        "U30": tuple(actions.values()),
        "Uall": tuple(actions.values()),
    }
    result = run_study_orchestration(
        out_dir=out_dir,
        isolation_root=tmp_path,
        study_manifest_id=manifest,
        programs={program: root},
        groups=groups,
        dependencies=_dependencies(manifest, program),
    )
    target = next(
        row
        for row in result.two_n_results["U14"]["pair_rows"]
        if (row["action_a_id"], row["action_b_id"]) == ("A", "B")
    )
    target["validation_status"] = "agree"
    target["false_authorization"] = "false"
    tampered = replace(result, false_authorizations=())

    with pytest.raises(ValueError, match="2N pair derived fields|2N replay writeback"):
        program_result_payload(
            tampered,
            program_id=program,
            program_status="coverage_limitation",
            limitation_kind="runtime_budget_exceeded",
            run_control_provenance={"decision": "skip"},
        )


def test_executed_coverage_checkpoint_requires_runtime_control_skip_provenance(
    tmp_path: Path,
) -> None:
    manifest = "a" * 64
    program = "p1"
    out_dir = tmp_path / "output" / "formal"
    root = tmp_path / "root.ll"
    _write_ir(root, 0)
    actions = {name: _Action(name) for name in ("A", "B", "C")}
    groups = {
        "U14": (actions["A"], actions["B"]),
        "U30": tuple(actions.values()),
        "Uall": tuple(actions.values()),
    }
    result = run_study_orchestration(
        out_dir=out_dir,
        isolation_root=tmp_path,
        study_manifest_id=manifest,
        programs={program: root},
        groups=groups,
        dependencies=_dependencies(manifest, program),
    )

    with pytest.raises(ValueError, match="runtime fallback provenance|control"):
        program_result_payload(
            result,
            program_id=program,
            program_status="coverage_limitation",
            limitation_kind="runtime_budget_exceeded",
            run_control_provenance={"decision": "execute"},
        )


def test_checkpoint_binds_directional_authorization_to_original_two_n_stage(
    tmp_path: Path,
) -> None:
    manifest = "a" * 64
    program = "p1"
    out_dir = tmp_path / "output" / "formal"
    root = tmp_path / "root.ll"
    _write_ir(root, 0)
    actions = {name: _Action(name) for name in ("A", "B", "C")}
    groups = {
        "U14": (actions["A"], actions["B"]),
        "U30": tuple(actions.values()),
        "Uall": tuple(actions.values()),
    }
    result = run_study_orchestration(
        out_dir=out_dir,
        isolation_root=tmp_path,
        study_manifest_id=manifest,
        programs={program: root},
        groups=groups,
        dependencies=_dependencies(manifest, program),
    )
    for row in result.two_n_results["U14"]["directional_rows"]:
        row["directional_status"] = "rejected_effect_changed"
    pair = result.two_n_results["U14"]["pair_rows"][0]
    pair.update(
        {
            "action_a_directional_status": "rejected_effect_changed",
            "action_b_directional_status": "rejected_effect_changed",
            "two_n_pair_status": "both_rejected",
            "validation_status": "unavailable",
            "false_authorization": "false",
            "stable_false_authorization": "false",
            "worker_replay_status": "not_required",
            "external_opt_replay_status": "not_required",
            "two_n_replay_status": "not_required",
        }
    )
    group = result.two_n_results["U14"]["group_rows"][0]
    group.update(
        {
            "group_authorization_status": "rejected",
            "directional_authorized_count": 0,
            "directional_unavailable_count": 2,
        }
    )
    tampered = replace(result, false_authorizations=())

    with pytest.raises(ValueError, match="2N stage result binding|2N replay writeback"):
        program_result_payload(
            tampered,
            program_id=program,
            program_status="complete",
            limitation_kind="",
            run_control_provenance={"decision": "execute"},
        )


def test_checkpoint_revalidates_bound_two_n_stage_result_bytes(
    tmp_path: Path,
) -> None:
    manifest = "a" * 64
    program = "p1"
    out_dir = tmp_path / "output" / "formal"
    root = tmp_path / "root.ll"
    _write_ir(root, 0)
    actions = {name: _Action(name) for name in ("A", "B", "C")}
    groups = {
        "U14": (actions["A"], actions["B"]),
        "U30": tuple(actions.values()),
        "Uall": tuple(actions.values()),
    }
    result = run_study_orchestration(
        out_dir=out_dir,
        isolation_root=tmp_path,
        study_manifest_id=manifest,
        programs={program: root},
        groups=groups,
        dependencies=_dependencies(manifest, program),
    )
    payload = program_result_payload(
        result,
        program_id=program,
        program_status="complete",
        limitation_kind="",
        run_control_provenance={"decision": "execute"},
    )
    stage_result = (
        out_dir
        / payload["two_n_stage_result_bindings"]["U14"]["stage_path"]
        / "result.json"
    )
    stage_result.write_text(
        stage_result.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="2N stage source.*hash"):
        finalize_program_evidence(
            out_dir / "raw" / "program-checkpoints" / "stage-source-tamper",
            payload,
            expected_input_sha256="9" * 64,
            isolation_root=out_dir,
        )


def test_checkpoint_revalidates_bound_pair_stage_result_bytes(
    tmp_path: Path,
) -> None:
    manifest = "a" * 64
    program = "p1"
    out_dir = tmp_path / "output" / "formal"
    root = tmp_path / "root.ll"
    _write_ir(root, 0)
    actions = {name: _Action(name) for name in ("A", "B", "C")}
    groups = {
        "U14": (actions["A"], actions["B"]),
        "U30": tuple(actions.values()),
        "Uall": tuple(actions.values()),
    }
    result = run_study_orchestration(
        out_dir=out_dir,
        isolation_root=tmp_path,
        study_manifest_id=manifest,
        programs={program: root},
        groups=groups,
        dependencies=_dependencies(manifest, program),
    )
    payload = program_result_payload(
        result,
        program_id=program,
        program_status="complete",
        limitation_kind="",
        run_control_provenance={"decision": "execute"},
    )
    stage_result = (
        out_dir
        / payload["pair_stage_result_binding"]["stage_path"]
        / "result.json"
    )
    stage_result.write_text(
        stage_result.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="pair stage source.*hash"):
        finalize_program_evidence(
            out_dir / "raw" / "program-checkpoints" / "pair-source-tamper",
            payload,
            expected_input_sha256="8" * 64,
            isolation_root=out_dir,
        )


def test_checkpoint_rejects_synchronized_replay_stage_artifact_binding_drift(
    tmp_path: Path,
) -> None:
    manifest = "a" * 64
    program = "p1"
    out_dir = tmp_path / "output" / "formal"
    root = tmp_path / "root.ll"
    _write_ir(root, 0)
    actions = {name: _Action(name) for name in ("A", "B", "C")}
    groups = {
        "U14": (actions["A"], actions["B"]),
        "U30": tuple(actions.values()),
        "Uall": tuple(actions.values()),
    }
    result = run_study_orchestration(
        out_dir=out_dir,
        isolation_root=tmp_path,
        study_manifest_id=manifest,
        programs={program: root},
        groups=groups,
        dependencies=_dependencies(manifest, program),
    )
    for record in result.false_authorizations[0]["worker"]:
        record["stage_results"]["A"]["hard_state_id"] = "f" * 64

    with pytest.raises(ValueError, match="replay.*evidence binding"):
        program_result_payload(
            result,
            program_id=program,
            program_status="complete",
            limitation_kind="",
            run_control_provenance={"decision": "execute"},
        )


def test_checkpoint_recomputes_replay_family_status_against_expected_pair_stages(
    tmp_path: Path,
) -> None:
    manifest = "a" * 64
    program = "p1"
    out_dir = tmp_path / "output" / "formal"
    root = tmp_path / "root.ll"
    _write_ir(root, 0)
    actions = {name: _Action(name) for name in ("A", "B", "C")}
    groups = {
        "U14": (actions["A"], actions["B"]),
        "U30": tuple(actions.values()),
        "Uall": tuple(actions.values()),
    }
    result = run_study_orchestration(
        out_dir=out_dir,
        isolation_root=tmp_path,
        study_manifest_id=manifest,
        programs={program: root},
        groups=groups,
        dependencies=_dependencies(manifest, program),
    )
    witness = result.false_authorizations[0]
    for family in ("worker", "external_opt", "two_n"):
        for record in witness[family]:
            for field in (
                "stage_results",
                "artifacts",
                "artifact_sha256",
                "hard_state_hashes",
            ):
                values = record[field]
                values["AB"], values["BA"] = values["BA"], values["AB"]

    with pytest.raises(ValueError, match="replay family status.*recomputed"):
        program_result_payload(
            result,
            program_id=program,
            program_status="complete",
            limitation_kind="",
            run_control_provenance={"decision": "execute"},
        )


def test_checkpoint_recomputes_pair_replay_writeback_from_bound_witnesses(
    tmp_path: Path,
) -> None:
    manifest = "a" * 64
    program = "p1"
    out_dir = tmp_path / "output" / "formal"
    root = tmp_path / "root.ll"
    _write_ir(root, 0)
    actions = {name: _Action(name) for name in ("A", "B", "C")}
    groups = {
        "U14": (actions["A"], actions["B"]),
        "U30": tuple(actions.values()),
        "Uall": tuple(actions.values()),
    }
    result = run_study_orchestration(
        out_dir=out_dir,
        isolation_root=tmp_path,
        study_manifest_id=manifest,
        programs={program: root},
        groups=groups,
        dependencies=_dependencies(manifest, program),
    )
    pair = next(
        row
        for row in result.two_n_results["U14"]["pair_rows"]
        if (row["action_a_id"], row["action_b_id"]) == ("A", "B")
    )
    pair.update(
        {
            "stable_false_authorization": "false",
            "worker_replay_status": "not_required",
            "external_opt_replay_status": "not_required",
            "two_n_replay_status": "not_required",
            "replay_artifact_id": "",
            "replay_time_ms": 0,
        }
    )

    with pytest.raises(ValueError, match="2N.*replay writeback|2N stage result binding"):
        program_result_payload(
            result,
            program_id=program,
            program_status="complete",
            limitation_kind="",
            run_control_provenance={"decision": "execute"},
        )
