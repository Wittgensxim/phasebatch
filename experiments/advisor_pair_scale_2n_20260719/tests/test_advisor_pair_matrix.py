from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
import threading
import time

import pytest
import advisor_study.pair_matrix as pair_matrix

from advisor_study.pair_matrix import (
    HardStateEquality,
    complete_pair_oracle_cost,
    profile_single_passes,
    reclaim_equal_pair_artifacts,
    run_complete_pair_matrix,
)
from advisor_study.pass_universe import ActionRecord


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _action(name: str, index: int) -> ActionRecord:
    return ActionRecord.for_function_candidate(
        name=name,
        pipeline=name,
        config_index=index,
    )


@dataclass(frozen=True)
class _Run:
    success: bool
    output_path: Path
    hard_state_id: str = ""
    timed_out: bool = False
    stderr: str = ""
    command: tuple[str, ...] = ()


def _successful(path: Path, *, hard_state_id: str | None = None) -> _Run:
    return _Run(
        success=True,
        output_path=path,
        hard_state_id=hard_state_id or _sha256(path),
        command=("fake-worker",),
    )


def _profile(
    action: ActionRecord,
    output: Path,
    *,
    active: bool,
    changed_functions: tuple[str, ...] = (),
    status: str = "success",
    hard_state_id: str = "",
) -> dict[str, object]:
    return {
        "action_id": action.action_id,
        "action_name": action.name,
        "execution_status": status,
        "output_path": str(output),
        "output_hard_state_id": hard_state_id or _sha256(output),
        "output_sha256": _sha256(output),
        "activity_status": "active" if active else "no_op",
        "changed_functions": changed_functions,
        "changed_blocks": (),
        "changed_module_regions": (),
        "observed_effect_available": True,
        "verifier_status": "success",
        "physical_pass_invocations": 1,
        "artifact_available": True,
        "artifact_materialized": True,
    }


def test_single_pass_profiling_retains_success_noop_error_timeout_and_invalid(
    tmp_path: Path,
) -> None:
    root = tmp_path / "S.ll"
    root.write_text("root\n", encoding="utf-8")
    root_hash = _sha256(root)
    actions = tuple(_action(name, index) for index, name in enumerate("ABCDE"))

    def run_single(parent: Path, action: ActionRecord, output: Path) -> _Run:
        output.parent.mkdir(parents=True, exist_ok=True)
        if action.name == "C":
            return _Run(False, output, timed_out=True, stderr="deadline")
        if action.name == "E":
            return _Run(False, output, stderr="worker-error")
        if action.name == "B":
            output.write_bytes(parent.read_bytes())
        else:
            output.write_text(f"{action.name}\n", encoding="utf-8")
        return _successful(output)

    profiles = profile_single_passes(
        root_ir=root,
        actions=actions,
        out_dir=tmp_path / "profiles",
        run_single=run_single,
        verify_ir=lambda path: path.read_text(encoding="utf-8") != "D\n",
        root_hard_state_id=root_hash,
        extract_observed_effect=lambda _root, _output, action: {
            "changed_functions": (action.name,),
            "changed_blocks": (),
            "changed_module_regions": (),
        },
        study_manifest_id="manifest",
    )

    by_name = {str(row["action_name"]): row for row in profiles}
    assert set(by_name) == {"A", "B", "C", "D", "E"}
    assert by_name["A"]["execution_status"] == "success"
    assert by_name["A"]["activity_status"] == "active"
    assert by_name["B"]["execution_status"] == "success"
    assert by_name["B"]["activity_status"] == "no_op"
    assert by_name["C"]["execution_status"] == "timeout"
    assert by_name["D"]["execution_status"] == "invalid"
    assert by_name["E"]["execution_status"] == "error"
    assert all(row["authority_granted"] == "false" for row in profiles)
    assert all(row["proved_commute"] == "false" for row in profiles)


def test_complete_matrix_keeps_noops_reuses_single_outputs_and_records_ab_ba(
    tmp_path: Path,
) -> None:
    root = tmp_path / "S.ll"
    root.write_text("root\n", encoding="utf-8")
    actions = tuple(_action(name, index) for index, name in enumerate("ABC"))
    outputs = {action.name: tmp_path / f"{action.name}.ll" for action in actions}
    outputs["A"].write_text("A\n", encoding="utf-8")
    outputs["B"].write_bytes(root.read_bytes())
    outputs["C"].write_bytes(root.read_bytes())
    profiles = [
        _profile(actions[0], outputs["A"], active=True, changed_functions=("f",)),
        _profile(actions[1], outputs["B"], active=False),
        _profile(actions[2], outputs["C"], active=False),
    ]
    calls: list[tuple[str, str]] = []

    def run_second(parent: Path, action: ActionRecord, output: Path) -> _Run:
        output.parent.mkdir(parents=True, exist_ok=True)
        calls.append((parent.stem, action.name))
        output.write_text(f"{parent.stem}-{action.name}\n", encoding="utf-8")
        return _successful(output, hard_state_id=f"{parent.stem}-{action.name}")

    rows = run_complete_pair_matrix(
        root_ir=root,
        profiles=profiles,
        actions={action.action_id: action for action in actions},
        out_dir=tmp_path / "pairs",
        profile_artifact_root=tmp_path,
        run_second=run_second,
        compare=lambda _left, _right: HardStateEquality(
            can_hard_fold=False, tier="hard", trusted_hard_comparator=True
        ),
        study_manifest_id="manifest",
    )

    assert len(rows) == 3
    assert {row["root_activity_class"] for row in rows} == {
        "active_noop",
        "noop_noop",
    }
    assert len(calls) == 6
    assert all(row["reused_single_pass_outputs"] == "true" for row in rows)
    assert all(row["a_status"] == "success" for row in rows)
    assert all(row["b_status"] == "success" for row in rows)
    assert all(row["ab_status"] == "success" for row in rows)
    assert all(row["ba_status"] == "success" for row in rows)
    assert all(row["dynamic_result"] == "order_sensitive" for row in rows)
    assert all(row["observed_relation"] == "observed_disjoint" for row in rows)
    assert all(row["total_logical_pass_applications"] == 4 for row in rows)
    assert all(row["total_physical_pass_invocations"] == 2 for row in rows)
    assert all(row["authority_granted"] == "false" for row in rows)
    assert all(row["proved_commute"] == "false" for row in rows)


def test_complete_matrix_jobs_two_runs_pairs_concurrently_and_preserves_canonical_rows(
    tmp_path: Path,
) -> None:
    root = tmp_path / "S.ll"
    root.write_text("root\n", encoding="utf-8")
    actions = tuple(_action(name, index) for index, name in enumerate("ABCD"))
    outputs = {action.name: tmp_path / f"{action.name}.ll" for action in actions}
    for action in actions:
        outputs[action.name].write_text(f"{action.name}\n", encoding="utf-8")
    profiles = [
        _profile(action, outputs[action.name], active=True, changed_functions=(action.name,))
        for action in actions
    ]
    action_map = {action.action_id: action for action in actions}

    def deterministic_run(parent: Path, action: ActionRecord, output: Path) -> _Run:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{parent.name}:{action.name}:{output.name}\n", encoding="utf-8")
        return _Run(
            True,
            output,
            hard_state_id=_sha256(output),
            command=("fake-worker", parent.name, action.name),
        )

    sequential = run_complete_pair_matrix(
        root_ir=root,
        profiles=profiles,
        actions=action_map,
        out_dir=tmp_path / "pairs",
        profile_artifact_root=tmp_path,
        run_second=deterministic_run,
        compare=lambda _left, _right: HardStateEquality(
            False, "hard", trusted_hard_comparator=True
        ),
        jobs=1,
        study_manifest_id="manifest",
        program_id="program",
        group_id="Uall",
    )

    lock = threading.Lock()
    first_two = threading.Barrier(2, timeout=5)
    call_count = 0
    active = 0
    max_active = 0

    def concurrent_run(parent: Path, action: ActionRecord, output: Path) -> _Run:
        nonlocal call_count, active, max_active
        with lock:
            call_count += 1
            ordinal = call_count
            active += 1
            max_active = max(max_active, active)
        try:
            if ordinal <= 2:
                first_two.wait()
            time.sleep(0.005)
            return deterministic_run(parent, action, output)
        finally:
            with lock:
                active -= 1

    parallel = run_complete_pair_matrix(
        root_ir=root,
        profiles=profiles,
        actions=action_map,
        out_dir=tmp_path / "pairs",
        profile_artifact_root=tmp_path,
        run_second=concurrent_run,
        compare=lambda _left, _right: HardStateEquality(
            False, "hard", trusted_hard_comparator=True
        ),
        jobs=2,
        study_manifest_id="manifest",
        program_id="program",
        group_id="Uall",
    )

    ordered_ids = sorted(action_map)
    assert max_active >= 2
    assert parallel == sequential
    assert [
        (row["action_a_id"], row["action_b_id"]) for row in parallel
    ] == list(combinations(ordered_ids, 2))


def test_parallel_matrix_preserves_typed_timeout_and_runner_exception_rows(
    tmp_path: Path,
) -> None:
    root = tmp_path / "S.ll"
    root.write_text("root\n", encoding="utf-8")
    actions = tuple(_action(name, index) for index, name in enumerate("ABC"))
    outputs = {action.name: tmp_path / f"{action.name}.ll" for action in actions}
    for action in actions:
        outputs[action.name].write_text(f"{action.name}\n", encoding="utf-8")
    profiles = [
        _profile(action, outputs[action.name], active=True, changed_functions=(action.name,))
        for action in actions
    ]

    def terminal_run(_parent: Path, action: ActionRecord, output: Path) -> _Run:
        if action.name == "B":
            return _Run(False, output, timed_out=True, stderr="deadline")
        if action.name == "C":
            raise RuntimeError("injected runner failure")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("success\n", encoding="utf-8")
        return _successful(output)

    common = {
        "root_ir": root,
        "profiles": profiles,
        "actions": {action.action_id: action for action in actions},
        "out_dir": tmp_path / "pairs",
        "profile_artifact_root": tmp_path,
        "run_second": terminal_run,
        "compare": lambda _left, _right: HardStateEquality(
            True, "hard", trusted_hard_comparator=True
        ),
        "study_manifest_id": "manifest",
        "program_id": "program",
        "group_id": "Uall",
    }
    sequential = run_complete_pair_matrix(**common, jobs=1)
    parallel = run_complete_pair_matrix(**common, jobs=2)

    assert parallel == sequential
    assert {row["dynamic_result"] for row in parallel} == {"failed", "timeout"}
    assert any(
        "runner_failed:RuntimeError" in str(row["fail_closed_reason"])
        for row in parallel
    )
    assert any(row["ab_status"] == "timeout" or row["ba_status"] == "timeout" for row in parallel)


def test_complete_matrix_rejects_boolean_jobs(tmp_path: Path) -> None:
    root = tmp_path / "S.ll"
    root.write_text("root\n", encoding="utf-8")
    actions = (_action("A", 0), _action("B", 1))
    outputs = (tmp_path / "A.ll", tmp_path / "B.ll")
    for output in outputs:
        output.write_text(output.stem, encoding="utf-8")
    profiles = [
        _profile(action, output, active=True)
        for action, output in zip(actions, outputs)
    ]

    with pytest.raises(ValueError, match="jobs must be a positive integer"):
        run_complete_pair_matrix(
            root_ir=root,
            profiles=profiles,
            actions={action.action_id: action for action in actions},
            out_dir=tmp_path / "pairs",
            profile_artifact_root=tmp_path,
            run_second=lambda _parent, _action, output: _Run(False, output),
            compare=lambda _left, _right: None,
            jobs=True,
        )


def test_discard_partial_pair_output_retries_transient_windows_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair_dir = tmp_path / "pairs" / "pair-transient"
    artifact = pair_dir / "AB.ll"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("partial\n", encoding="utf-8")
    original_unlink = Path.unlink
    attempts = 0

    def transient_lock(path: Path, *args: object, **kwargs: object) -> None:
        nonlocal attempts
        if path == artifact and attempts < 2:
            attempts += 1
            raise PermissionError("injected transient Windows lock")
        attempts += 1
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", transient_lock)
    pair_matrix._discard_partial_pair_outputs((artifact,), pair_dir)

    assert attempts == 3
    assert not artifact.exists()


def test_discard_partial_pair_output_fails_closed_on_permanent_windows_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair_dir = tmp_path / "pairs" / "pair-permanent"
    artifact = pair_dir / "BA.ll"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("partial\n", encoding="utf-8")
    attempts = 0

    def permanent_lock(path: Path, *args: object, **kwargs: object) -> None:
        nonlocal attempts
        if path == artifact:
            attempts += 1
            raise PermissionError("injected permanent Windows lock")
        Path.unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", permanent_lock)
    with pytest.raises(RuntimeError, match="partial pair artifact could not be reclaimed"):
        pair_matrix._discard_partial_pair_outputs((artifact,), pair_dir)

    assert attempts == 7
    assert artifact.is_file()


def test_parallel_partial_pair_cleanup_is_isolated_to_each_canonical_pair_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "S.ll"
    root.write_text("root\n", encoding="utf-8")
    actions = tuple(_action(name, index) for index, name in enumerate("ABCD"))
    outputs = {action.name: tmp_path / f"{action.name}.ll" for action in actions}
    for action in actions:
        outputs[action.name].write_text(f"{action.name}\n", encoding="utf-8")
    profiles = [
        _profile(action, outputs[action.name], active=True, changed_functions=(action.name,))
        for action in actions
    ]
    pair_root = tmp_path / "pairs"
    sentinel = pair_root / "must-remain.ll"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("sentinel\n", encoding="utf-8")
    original_unlink = Path.unlink
    first_wave = threading.Barrier(3, timeout=5)
    observed_threads: set[int] = set()
    observed_unlinks: list[tuple[Path, str]] = []
    observation_lock = threading.Lock()

    def synchronized_unlink(path: Path, *args: object, **kwargs: object) -> None:
        wait = False
        if path.parent.parent == pair_root and path.name in {"AB.ll", "BA.ll"}:
            with observation_lock:
                observed_unlinks.append((path.parent, path.name))
                thread_id = threading.get_ident()
                if thread_id not in observed_threads:
                    observed_threads.add(thread_id)
                    wait = True
        if wait:
            first_wave.wait()
        original_unlink(path, *args, **kwargs)

    def partial_run(_parent: Path, _action: ActionRecord, output: Path) -> _Run:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"partial:{output.name}\n", encoding="utf-8")
        return _Run(output.name == "AB.ll", output, stderr="injected BA failure")

    monkeypatch.setattr(Path, "unlink", synchronized_unlink)
    rows = run_complete_pair_matrix(
        root_ir=root,
        profiles=profiles,
        actions={action.action_id: action for action in actions},
        out_dir=pair_root,
        profile_artifact_root=tmp_path,
        run_second=partial_run,
        compare=lambda _left, _right: None,
        jobs=3,
        study_manifest_id="manifest",
        program_id="program",
        group_id="Uall",
    )

    pair_directories = {path for path, _name in observed_unlinks}
    assert len(observed_threads) == 3
    assert len(pair_directories) == len(rows) == 6
    assert all({name for path, name in observed_unlinks if path == directory} == {"AB.ll", "BA.ll"} for directory in pair_directories)
    assert all(not (directory / "AB.ll").exists() and not (directory / "BA.ll").exists() for directory in pair_directories)
    assert sentinel.read_text(encoding="utf-8") == "sentinel\n"


@pytest.mark.parametrize(
    ("ab", "ba", "equality", "expected"),
    [
        (
            _Run(False, Path("AB.ll"), timed_out=True),
            _Run(True, Path("BA.ll")),
            HardStateEquality(True, "hard", trusted_hard_comparator=True),
            "timeout",
        ),
        (
            _Run(False, Path("AB.ll")),
            _Run(True, Path("BA.ll")),
            HardStateEquality(True, "hard", trusted_hard_comparator=True),
            "failed",
        ),
        (
            _Run(False, Path("AB.ll"), stderr="AB failed"),
            _Run(False, Path("BA.ll"), stderr="BA failed"),
            HardStateEquality(True, "hard", trusted_hard_comparator=True),
            "failed",
        ),
        (
            _Run(True, Path("AB.ll")),
            _Run(True, Path("BA.ll")),
            HardStateEquality(True, "failed", trusted_hard_comparator=True),
            "unknown",
        ),
        (
            _Run(True, Path("AB.ll")),
            _Run(True, Path("BA.ll")),
            HardStateEquality(True, "hard", trusted_hard_comparator=True),
            "commute",
        ),
    ],
)
def test_complete_matrix_classifies_timeout_failure_unknown_and_commute(
    tmp_path: Path,
    ab: _Run,
    ba: _Run,
    equality: HardStateEquality,
    expected: str,
) -> None:
    root = tmp_path / "S.ll"
    root.write_text("root\n", encoding="utf-8")
    actions = (_action("A", 0), _action("B", 1))
    a_output = tmp_path / "A.ll"
    b_output = tmp_path / "B.ll"
    a_output.write_text("A\n", encoding="utf-8")
    b_output.write_text("B\n", encoding="utf-8")
    profiles = [
        _profile(actions[0], a_output, active=True, changed_functions=("f",)),
        _profile(actions[1], b_output, active=True, changed_functions=("g",)),
    ]
    runs = iter((ab, ba))

    def run_second(_parent: Path, _action: ActionRecord, output: Path) -> _Run:
        raw = next(runs)
        if raw.success:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(f"{output.stem}\n", encoding="utf-8")
        return replace(raw, output_path=output)

    row = run_complete_pair_matrix(
        root_ir=root,
        profiles=profiles,
        actions={action.action_id: action for action in actions},
        out_dir=tmp_path / "pairs",
        profile_artifact_root=tmp_path,
        run_second=run_second,
        compare=lambda _left, _right: equality,
        study_manifest_id="manifest",
    )[0]

    assert row["root_activity_class"] == "active_active"
    assert row["observed_relation"] == "observed_disjoint"
    assert row["dynamic_result"] == expected
    assert row["ab_stderr_sha256"] == hashlib.sha256(
        ab.stderr.encode("utf-8")
    ).hexdigest()
    assert row["ba_stderr_sha256"] == hashlib.sha256(
        ba.stderr.encode("utf-8")
    ).hexdigest()
    if expected in {"timeout", "failed"}:
        assert row["artifact_available"] == "false"
        assert row["fail_closed_reason"]
        # A row advertised as unmaterialized must not leave a successful or
        # partial directional output behind for checkpoint validation.
        assert not Path(str(row["ab_output_path"])).exists()
        assert not Path(str(row["ba_output_path"])).exists()
    if expected == "unknown":
        assert row["artifact_available"] == "true"
        assert "pair_unknown" in str(row["fail_closed_reason"])


def test_matrix_keeps_terminal_first_round_pairs_without_second_stage_execution(
    tmp_path: Path,
) -> None:
    root = tmp_path / "S.ll"
    root.write_text("root\n", encoding="utf-8")
    actions = (_action("A", 0), _action("B", 1), _action("C", 2))
    paths = [tmp_path / f"{action.name}.ll" for action in actions]
    for path in paths:
        path.write_text(path.stem, encoding="utf-8")
    profiles = [
        _profile(actions[0], paths[0], active=True, changed_functions=("f",)),
        _profile(actions[1], paths[1], active=True, changed_functions=("f",)),
        _profile(actions[2], paths[2], active=False, status="timeout"),
    ]
    profiles[0]["observed_effect_available"] = False
    calls = 0

    def run_second(_parent: Path, _action: ActionRecord, output: Path) -> _Run:
        nonlocal calls
        calls += 1
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("second", encoding="utf-8")
        return _successful(output)

    rows = run_complete_pair_matrix(
        root_ir=root,
        profiles=profiles,
        actions={action.action_id: action for action in actions},
        out_dir=tmp_path / "pairs",
        profile_artifact_root=tmp_path,
        run_second=run_second,
        compare=lambda _left, _right: HardStateEquality(
            True, "hard", trusted_hard_comparator=True
        ),
        study_manifest_id="manifest",
    )

    assert len(rows) == 3
    assert calls == 2
    by_pair = {(str(row["action_a_id"]), str(row["action_b_id"])): row for row in rows}
    first_pair = tuple(sorted((actions[0].action_id, actions[1].action_id)))
    assert by_pair[first_pair]["observed_relation"] == "observed_unknown"
    assert by_pair[first_pair]["dynamic_result"] == "commute"
    for pair in (
        tuple(sorted((actions[0].action_id, actions[2].action_id))),
        tuple(sorted((actions[1].action_id, actions[2].action_id))),
    ):
        row = by_pair[pair]
        assert row["a_status"] == "success" or row["b_status"] == "success"
        assert "timeout" in {row["a_status"], row["b_status"]}
        assert row["ab_status"] == row["ba_status"] == "not_run"
        assert row["dynamic_result"] == "timeout"
        assert row["artifact_available"] == "false"
        assert row["artifact_materialized"] == "false"


def _retention_row(
    root: Path,
    name: str,
    *,
    ab_text: str = "; equal\n",
    ba_text: str = "; equal\n",
    **overrides: object,
) -> dict[str, object]:
    pair_dir = root / name
    pair_dir.mkdir(parents=True, exist_ok=True)
    ab = pair_dir / "AB.ll"
    ba = pair_dir / "BA.ll"
    ab.write_text(ab_text, encoding="utf-8", newline="\n")
    ba.write_text(ba_text, encoding="utf-8", newline="\n")
    digest_ab, digest_ba = _sha256(ab), _sha256(ba)
    row: dict[str, object] = {
        "row_id": name,
        "study_manifest_id": "a" * 64,
        "program_id": "program",
        "group_id": "Uall",
        "action_a_id": "A",
        "action_b_id": "B",
        "ab_status": "success",
        "ba_status": "success",
        "ab_verifier_status": "success",
        "ba_verifier_status": "success",
        "ab_hard_state_id": digest_ab,
        "ba_hard_state_id": digest_ba,
        "ab_output_sha256": digest_ab,
        "ba_output_sha256": digest_ba,
        "ab_output_path": str(ab),
        "ba_output_path": str(ba),
        "dynamic_result": "commute",
        "artifact_available": "true",
        "artifact_materialized": "true",
        "false_authorization": "false",
        "command_sha256": hashlib.sha256(b"command").hexdigest(),
        "stderr_sha256": hashlib.sha256(b"").hexdigest(),
    }
    row.update(overrides)
    return row


def test_retention_reclaims_only_verified_equal_unreferenced_pair_ir(tmp_path: Path) -> None:
    row = _retention_row(tmp_path, "equal")
    retained = reclaim_equal_pair_artifacts([row], pair_artifact_root=tmp_path)

    assert not Path(str(row["ab_output_path"])).exists()
    assert not Path(str(row["ba_output_path"])).exists()
    assert retained[0]["artifact_available"] == "true"
    assert retained[0]["artifact_materialized"] == "false"
    assert retained[0]["retention_status"] == "reclaimed_equal_unreferenced"
    assert retained[0]["ab_output_sha256"] == retained[0]["ba_output_sha256"]
    assert retained[0]["ab_hard_state_id"] == retained[0]["ba_hard_state_id"]


def test_retention_never_deletes_mismatch_terminal_or_false_authorization_evidence(
    tmp_path: Path,
) -> None:
    mismatch = _retention_row(tmp_path, "mismatch", ba_text="; different\n")
    terminal = _retention_row(tmp_path, "terminal", ba_status="error", dynamic_result="failed")
    false_auth = _retention_row(tmp_path, "false-auth", false_authorization="true")
    retained = reclaim_equal_pair_artifacts(
        [mismatch, terminal, false_auth], pair_artifact_root=tmp_path
    )

    assert all(Path(str(row["ab_output_path"])).is_file() for row in (mismatch, terminal, false_auth))
    assert all(Path(str(row["ba_output_path"])).is_file() for row in (mismatch, terminal, false_auth))
    assert all(row["artifact_materialized"] == "true" for row in retained)
    assert {str(row["retention_status"]) for row in retained} == {
        "retained_hash_mismatch",
        "retained_terminal_status",
        "retained_false_authorization",
    }


def test_retention_keeps_pair_ir_referenced_by_another_row(tmp_path: Path) -> None:
    first = _retention_row(tmp_path, "first")
    second = _retention_row(tmp_path, "second")
    second["ab_output_path"] = first["ab_output_path"]
    second["ab_output_sha256"] = first["ab_output_sha256"]
    second["ab_hard_state_id"] = first["ab_hard_state_id"]
    retained = reclaim_equal_pair_artifacts([first, second], pair_artifact_root=tmp_path)

    assert Path(str(first["ab_output_path"])).is_file()
    assert Path(str(first["ba_output_path"])).is_file()
    assert Path(str(second["ba_output_path"])).is_file()
    assert {row["retention_status"] for row in retained} == {
        "retained_referenced",
        "retained_noncanonical_path",
    }


def test_retention_rechecks_identity_immediately_before_delete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    row = _retention_row(tmp_path, "toctou")
    ab = Path(str(row["ab_output_path"]))
    original_replace = pair_matrix.os.replace
    replaced = False

    def replace_after_final_check(source: str | Path, destination: str | Path) -> None:
        nonlocal replaced
        if Path(source) == ab and not replaced:
            ab.write_text("; replacement raced final check\n", encoding="utf-8", newline="\n")
            replaced = True
        original_replace(source, destination)

    monkeypatch.setattr(pair_matrix.os, "replace", replace_after_final_check)
    retained = reclaim_equal_pair_artifacts([row], pair_artifact_root=tmp_path)

    assert replaced
    assert not ab.exists()
    assert Path(str(row["ba_output_path"])).is_file()
    assert retained[0]["artifact_materialized"] == "true"
    assert retained[0]["retention_status"] == "retained_toctou_drift"
    actual = Path(str(retained[0]["reclamation_artifact_paths"]["AB"]))
    assert actual.is_file()
    assert actual.read_text(encoding="utf-8") == "; replacement raced final check\n"
    assert actual.parent.name == ".reclaim-quarantine"


def test_retention_records_only_surviving_quarantine_artifact_after_partial_delete_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = _retention_row(tmp_path, "partial-delete")
    original_unlink = Path.unlink

    def fail_only_quarantined_ba(path: Path, *args: object, **kwargs: object) -> None:
        if path.parent.name == ".reclaim-quarantine" and path.name.startswith("BA-"):
            raise OSError("injected BA delete failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_only_quarantined_ba)
    retained = reclaim_equal_pair_artifacts([row], pair_artifact_root=tmp_path)
    result = retained[0]

    assert result["retention_status"] == "retained_quarantine_delete_failed"
    assert result["artifact_materialized"] == "true"
    assert set(result["reclamation_artifact_paths"]) == {"BA"}
    assert Path(str(result["reclamation_artifact_paths"]["BA"])).is_file()
    assert set(result["reclamation_reclaimed_paths"]) == {"AB"}
    assert not Path(str(result["reclamation_reclaimed_paths"]["AB"])).exists()
    assert result["retention_status"] != "reclaimed_equal_unreferenced"


@pytest.mark.parametrize("kind", ("wrong_stage", "alias", "protected"))
def test_retention_requires_canonical_pair_paths_and_honors_protection(
    tmp_path: Path,
    kind: str,
) -> None:
    row = _retention_row(tmp_path, f"canonical-{kind}")
    protected_paths: list[Path] = []
    if kind == "wrong_stage":
        wrong = tmp_path / str(row["row_id"]) / "nested" / "AB.ll"
        wrong.parent.mkdir(parents=True, exist_ok=True)
        wrong.write_bytes(Path(str(row["ab_output_path"])).read_bytes())
        row["ab_output_path"] = str(wrong)
    elif kind == "alias":
        row["ab_output_path"] = str(
            tmp_path / str(row["row_id"]) / ".." / str(row["row_id"]) / "AB.ll"
        )
    else:
        protected_paths.append(Path(str(row["ab_output_path"])))

    retained = reclaim_equal_pair_artifacts(
        [row], pair_artifact_root=tmp_path, protected_paths=protected_paths
    )

    assert Path(str(row["ba_output_path"])).is_file()
    if kind != "wrong_stage":
        assert Path(str(row["ab_output_path"])).is_file()
    assert retained[0]["artifact_materialized"] == "true"
    assert retained[0]["retention_status"] == (
        "retained_referenced" if kind == "protected" else "retained_noncanonical_path"
    )


def test_pair_cost_uses_n_plus_n_times_n_minus_one_not_factorial() -> None:
    assert complete_pair_oracle_cost(0) == {
        "logical_first_round_applications": 0,
        "logical_second_stage_applications": 0,
        "logical_total_pass_applications": 0,
    }
    assert complete_pair_oracle_cost(3) == {
        "logical_first_round_applications": 3,
        "logical_second_stage_applications": 6,
        "logical_total_pass_applications": 9,
    }
    assert complete_pair_oracle_cost(30)["logical_total_pass_applications"] == 900


def test_profile_rejects_runner_output_redirect_outside_requested_artifact_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "S.ll"
    root.write_text("root\n", encoding="utf-8")
    action = _action("A", 0)
    redirected = tmp_path / "outside" / "A.ll"

    def redirected_runner(_parent: Path, _action: ActionRecord, requested: Path) -> _Run:
        requested.parent.mkdir(parents=True, exist_ok=True)
        requested.write_text("requested\n", encoding="utf-8")
        redirected.parent.mkdir(parents=True, exist_ok=True)
        redirected.write_text("redirected\n", encoding="utf-8")
        return _successful(redirected)

    row = profile_single_passes(
        root_ir=root,
        actions=(action,),
        out_dir=tmp_path / "profiles",
        run_single=redirected_runner,
        study_manifest_id="manifest",
    )[0]

    assert row["execution_status"] == "error"
    assert row["output_path"] == str(tmp_path / "profiles" / action.action_id / "first.ll")
    assert row["artifact_available"] == "false"
    assert "output_path_redirected" in str(row["fail_closed_reason"])


def test_pair_matrix_fails_closed_on_profile_artifact_hash_drift_without_second_runs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "S.ll"
    root.write_text("root\n", encoding="utf-8")
    actions = (_action("A", 0), _action("B", 1))
    outputs = (tmp_path / "A.ll", tmp_path / "B.ll")
    outputs[0].write_text("A\n", encoding="utf-8")
    outputs[1].write_text("B\n", encoding="utf-8")
    profiles = [
        _profile(actions[0], outputs[0], active=True, changed_functions=("f",)),
        _profile(actions[1], outputs[1], active=True, changed_functions=("g",)),
    ]
    profiles[0]["output_sha256"] = "0" * 64
    calls = 0

    def run_second(_parent: Path, _action: ActionRecord, _output: Path) -> _Run:
        nonlocal calls
        calls += 1
        raise AssertionError("hash-drift first round must not start a second run")

    row = run_complete_pair_matrix(
        root_ir=root,
        profiles=profiles,
        actions={action.action_id: action for action in actions},
        out_dir=tmp_path / "pairs",
        profile_artifact_root=tmp_path,
        run_second=run_second,
        compare=lambda _left, _right: HardStateEquality(
            True, "hard", trusted_hard_comparator=True
        ),
        study_manifest_id="manifest",
    )[0]

    assert calls == 0
    assert row["ab_status"] == "not_run"
    assert row["ba_status"] == "not_run"
    assert row["dynamic_result"] == "unknown"
    assert "profile_output_sha256_mismatch" in str(row["fail_closed_reason"])


def test_shared_unverified_worker_hash_and_untrusted_comparator_never_commute(
    tmp_path: Path,
) -> None:
    root = tmp_path / "S.ll"
    root.write_text("root\n", encoding="utf-8")
    actions = (_action("A", 0), _action("B", 1))
    outputs = (tmp_path / "A.ll", tmp_path / "B.ll")
    outputs[0].write_text("A\n", encoding="utf-8")
    outputs[1].write_text("B\n", encoding="utf-8")
    profiles = [
        _profile(actions[0], outputs[0], active=True, changed_functions=("f",)),
        _profile(actions[1], outputs[1], active=True, changed_functions=("g",)),
    ]
    compare_calls = 0

    def run_second(_parent: Path, _action: ActionRecord, output: Path) -> _Run:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(output.stem, encoding="utf-8")
        return _Run(True, output, hard_state_id="shared-unverified")

    def malformed_comparator(_left: object, _right: object) -> HardStateEquality:
        nonlocal compare_calls
        compare_calls += 1
        return HardStateEquality(
            True, "unrecognized-tier", trusted_hard_comparator=True
        )

    row = run_complete_pair_matrix(
        root_ir=root,
        profiles=profiles,
        actions={action.action_id: action for action in actions},
        out_dir=tmp_path / "pairs",
        profile_artifact_root=tmp_path,
        run_second=run_second,
        compare=malformed_comparator,
        study_manifest_id="manifest",
    )[0]

    assert compare_calls == 1
    assert row["dynamic_result"] == "unknown"
    assert "pair_unknown" in str(row["fail_closed_reason"])


def test_persisted_profile_artifact_outside_controlled_root_is_not_a_parent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "S.ll"
    root.write_text("root\n", encoding="utf-8")
    controlled = tmp_path / "controlled-profiles"
    external = tmp_path / "external-profiles"
    actions = (_action("A", 0), _action("B", 1))
    outputs = (external / "A.ll", external / "B.ll")
    external.mkdir()
    outputs[0].write_text("A\n", encoding="utf-8")
    outputs[1].write_text("B\n", encoding="utf-8")
    profiles = [
        _profile(actions[0], outputs[0], active=True, changed_functions=("f",)),
        _profile(actions[1], outputs[1], active=True, changed_functions=("g",)),
    ]
    calls = 0

    def run_second(_parent: Path, _action: ActionRecord, _output: Path) -> _Run:
        nonlocal calls
        calls += 1
        raise AssertionError("outside profile artifact must not become a parent")

    row = run_complete_pair_matrix(
        root_ir=root,
        profiles=profiles,
        actions={action.action_id: action for action in actions},
        out_dir=tmp_path / "pairs",
        profile_artifact_root=controlled,
        run_second=run_second,
        compare=lambda _left, _right: HardStateEquality(
            True, "hard", trusted_hard_comparator=True
        ),
        study_manifest_id="manifest",
    )[0]

    assert calls == 0
    assert row["ab_status"] == "not_run"
    assert row["ba_status"] == "not_run"
    assert row["dynamic_result"] == "unknown"
    assert "profile_output_outside_artifact_root" in str(row["fail_closed_reason"])


def test_unverified_raw_hash_cannot_false_classify_changed_output_as_noop(
    tmp_path: Path,
) -> None:
    root = tmp_path / "S.ll"
    root.write_text("root\n", encoding="utf-8")
    root_hash = _sha256(root)
    action = _action("A", 0)

    def spoofed_hash_runner(_parent: Path, _action: ActionRecord, output: Path) -> _Run:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("actually changed\n", encoding="utf-8")
        return _Run(True, output, hard_state_id=root_hash)

    row = profile_single_passes(
        root_ir=root,
        actions=(action,),
        out_dir=tmp_path / "profiles",
        run_single=spoofed_hash_runner,
        root_hard_state_id=root_hash,
        study_manifest_id="manifest",
    )[0]

    assert row["activity_status"] == "active"
    assert row["activity_evidence"] == "output_sha256"
    assert row["activity_status"] != "no_op"
