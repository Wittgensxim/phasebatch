from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import multiprocessing
from pathlib import Path

import pytest

from advisor_study import aggregate, cli
from advisor_study import orchestration as orchestration_module
from advisor_study import program_runtime as program_runtime_module
from advisor_study.program_runtime import (
    load_program_checkpoint,
    program_checkpoint_input_sha256,
    program_checkpoint_path,
    runtime_budget_limited_result,
)
from advisor_study.run_control import add_runtime_budget_skip, ensure_run_control


class _Action:
    def __init__(self, action_id: str) -> None:
        self.action_id = action_id


def _hold_study_writer_lock(out_dir: str, ready: object, release: object) -> None:
    with cli._study_run_writer_lock(Path(out_dir)):
        ready.set()  # type: ignore[attr-defined]
        release.wait(15)  # type: ignore[attr-defined]


def _groups() -> dict[str, tuple[_Action, ...]]:
    actions = {name: _Action(name) for name in ("A", "B", "C")}
    return {
        "U14": (actions["A"], actions["B"]),
        "U30": tuple(actions.values()),
        "Uall": tuple(actions.values()),
    }


def _frozen(tmp_path: Path, program_ids: tuple[str, ...]) -> cli.FrozenPhase:
    groups = _groups()
    return cli.FrozenPhase(
        out_dir=tmp_path / "output" / "formal",
        manifest_path=tmp_path / "output" / "formal" / "study_manifest.json",
        study_manifest_id="manifest-1",
        program_count=len(program_ids),
        program_ids=program_ids,
        groups={
            group: tuple(action.action_id for action in actions)
            for group, actions in groups.items()
        },
        jobs=1,
        timeout_s=3,
    )


def _roots(tmp_path: Path, program_ids: tuple[str, ...]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for program_id in program_ids:
        root = tmp_path / "roots" / f"{program_id}.ll"
        root.parent.mkdir(parents=True, exist_ok=True)
        root.write_text(f"; {program_id}\n", encoding="utf-8")
        roots[program_id] = root
    return roots


def _synthetic_result(
    frozen: cli.FrozenPhase, groups: dict[str, tuple[_Action, ...]], program_id: str
):
    result = runtime_budget_limited_result(
        out_dir=frozen.out_dir,
        study_manifest_id=frozen.study_manifest_id,
        program_id=program_id,
        program_family="synthetic",
        groups=groups,
        provenance={
            "control_id": "c" * 64,
            "control_file_sha256": "f" * 64,
            "program_wall_time_budget_s": 600,
            "observed_wall_time_s": 600,
            "limitation_kind": "runtime_budget_exceeded",
            "reason": "synthetic runner fixture",
        },
    )
    # These tests exercise complete checkpoint/reuse plumbing, not the skip
    # path. Convert the structural scaffold into a completed run whose
    # individual pass attempts timed out, without retaining skip provenance.
    complete_reason = "synthetic completed runner recorded a pass timeout"
    for row in result.profile_rows[program_id]:
        row["fail_closed_reason"] = complete_reason
        row["logical_pass_applications"] = 1
        row["physical_pass_invocations"] = 1
    for group in ("U14", "U30", "Uall"):
        for row in result.pair_views[group]:
            row["fail_closed_reason"] = complete_reason
            row["cleanup_status"] = "not_eligible"
        group_row = result.two_n_results[group]["group_rows"][0]
        group_row["fail_closed_reason"] = complete_reason
        group_row["logical_pass_applications"] = len(groups[group])
        group_row["physical_pass_invocations"] = len(groups[group])
        for row in result.two_n_results[group]["directional_rows"]:
            row["fail_closed_reason"] = complete_reason
            row["cleanup_status"] = "not_eligible"
            row["logical_pass_applications"] = 1
            row["physical_pass_invocations"] = 1
        for row in result.two_n_results[group]["pair_rows"]:
            row["validation_status"] = "unavailable"
            row["fail_closed_reason"] = complete_reason

    stage_paths: dict[str, str] = {}
    isolation_root = frozen.out_dir.parents[1]
    pair_stage_dir = frozen.out_dir / "synthetic-pairs" / program_id / "Uall"
    orchestration_module._run_or_reuse_stage(
        pair_stage_dir,
        lambda _directory: {
            "rows": [dict(row) for row in result.pair_views["Uall"]]
        },
        expected_input_sha256=hashlib.sha256(
            f"{program_id}\0Uall\0synthetic-pairs-complete".encode("utf-8")
        ).hexdigest(),
        isolation_root=isolation_root,
    )
    stage_paths[f"{program_id}:pairs:Uall"] = pair_stage_dir.relative_to(
        frozen.out_dir
    ).as_posix()
    for group in ("U14", "U30", "Uall"):
        stage_dir = frozen.out_dir / "synthetic-two-n" / program_id / group
        group_result = result.two_n_results[group]
        orchestration_module._run_or_reuse_stage(
            stage_dir,
            lambda _directory, value=group_result: {
                key: [dict(row) for row in value[key]]
                for key in ("group_rows", "directional_rows", "pair_rows")
            },
            expected_input_sha256=hashlib.sha256(
                f"{program_id}\0{group}\0synthetic-complete".encode("utf-8")
            ).hexdigest(),
            isolation_root=isolation_root,
        )
        stage_paths[f"{program_id}:two_n:{group}"] = stage_dir.relative_to(
            frozen.out_dir
        ).as_posix()
    return replace(result, stage_paths=stage_paths)


def test_program_loop_checkpoints_before_next_and_second_run_uses_zero_runners(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    frozen = _frozen(tmp_path, ("p2", "p1"))
    groups = _groups()
    roots = _roots(tmp_path, frozen.program_ids)
    calls: list[str] = []

    def run_program(program_id: str, _root: Path):
        if program_id == "p2":
            p1_input = program_checkpoint_input_sha256(
                study_manifest_id=frozen.study_manifest_id,
                program_id="p1",
                root_ir_sha256=cli._sha256_file(roots["p1"]),
                group_action_ids=frozen.groups,
                runner_semantics_id=cli.RAW_EXECUTION_SEMANTICS_REVISION,
            )
            assert load_program_checkpoint(
                program_checkpoint_path(frozen.out_dir, p1_input),
                expected_input_sha256=p1_input,
                isolation_root=frozen.out_dir,
            )[1] == "complete"
        calls.append(program_id)
        return _synthetic_result(frozen, groups, program_id)

    combined, index = cli._run_program_checkpoints(
        frozen=frozen,
        programs=roots,
        program_families={program_id: "synthetic" for program_id in roots},
        groups=groups,
        run_program=run_program,
    )

    assert calls == ["p1", "p2"]
    assert set(combined.profile_rows) == {"p1", "p2"}
    assert [row["program_id"] for row in index["entries"]] == ["p1", "p2"]
    first_events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["status"] for event in first_events] == [
        "start", "complete", "start", "complete"
    ]
    assert all(event["program_wall_time_budget_s"] == 600 for event in first_events)
    assert all(event["utc"] for event in first_events)
    assert first_events[-1]["category_counts"]["pair_dynamic_result"]["timeout"] == 3

    calls.clear()
    _combined_again, index_again = cli._run_program_checkpoints(
        frozen=frozen,
        programs=roots,
        program_families={program_id: "synthetic" for program_id in roots},
        groups=groups,
        run_program=run_program,
    )
    assert calls == []
    assert index_again == index
    second_events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["status"] for event in second_events] == ["reused", "reused"]
    current = json.loads(
        (frozen.out_dir / "logs" / "current_program.json").read_text(encoding="utf-8")
    )
    assert current["program_id"] == "p2"
    assert current["status"] == "reused"
    assert not (frozen.out_dir / "raw" / "current_program.json").exists()


def test_typed_skip_keeps_full_pair_and_2n_denominators(tmp_path: Path) -> None:
    frozen = _frozen(tmp_path, ("p1",))
    groups = _groups()
    roots = _roots(tmp_path, frozen.program_ids)
    ensure_run_control(
        frozen.out_dir,
        study_manifest_id=frozen.study_manifest_id,
        program_ids=frozen.program_ids,
    )
    add_runtime_budget_skip(
        frozen.out_dir,
        study_manifest_id=frozen.study_manifest_id,
        program_ids=frozen.program_ids,
        program_id="p1",
        observed_wall_time_s=600,
    )

    combined, index = cli._run_program_checkpoints(
        frozen=frozen,
        programs=roots,
        program_families={"p1": "synthetic"},
        groups=groups,
        run_program=lambda *_args: pytest.fail("typed skip invoked the runner"),
    )

    assert index["entries"][0]["program_status"] == "coverage_limitation"
    assert len(combined.profile_rows["p1"]) == 3
    assert len(combined.pair_views["Uall"]) == 3
    assert len(combined.two_n_results["U14"]["pair_rows"]) == 1
    assert len(combined.two_n_results["U30"]["pair_rows"]) == 3
    assert len(combined.two_n_results["Uall"]["pair_rows"]) == 3
    assert all(
        row["group_authorization_status"] == "group_precondition_unavailable"
        for group in combined.two_n_results.values()
        for row in group["group_rows"]
    )


def test_checkpoint_revision_change_invalidates_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frozen = _frozen(tmp_path, ("p1",))
    groups = _groups()
    roots = _roots(tmp_path, frozen.program_ids)
    calls: list[str] = []

    def run_program(program_id: str, _root: Path):
        calls.append(program_id)
        return _synthetic_result(frozen, groups, program_id)

    cli._run_program_checkpoints(
        frozen=frozen,
        programs=roots,
        program_families={"p1": "synthetic"},
        groups=groups,
        run_program=run_program,
    )
    monkeypatch.setattr(
        cli,
        "RAW_EXECUTION_SEMANTICS_REVISION",
        cli.RAW_EXECUTION_SEMANTICS_REVISION + "-changed",
    )
    cli._run_program_checkpoints(
        frozen=frozen,
        programs=roots,
        program_families={"p1": "synthetic"},
        groups=groups,
        run_program=run_program,
    )
    assert calls == ["p1", "p1"]


def test_program_runtime_version_change_invalidates_progress_before_index_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _frozen(tmp_path, ("p1",))
    groups = _groups()
    roots = _roots(tmp_path, frozen.program_ids)
    calls: list[str] = []

    def run_program(program_id: str, _root: Path):
        calls.append(program_id)
        return _synthetic_result(frozen, groups, program_id)

    monkeypatch.setattr(
        program_runtime_module, "PROGRAM_RUNTIME_IMPLEMENTATION_VERSION", 2
    )
    first = cli._run_program_checkpoints(
        frozen=frozen,
        programs=roots,
        program_families={"p1": "synthetic"},
        groups=groups,
        run_program=run_program,
    )[1]
    assert first["program_runtime_implementation_version"] == 2

    monkeypatch.setattr(
        program_runtime_module, "PROGRAM_RUNTIME_IMPLEMENTATION_VERSION", 3
    )
    second = cli._run_program_checkpoints(
        frozen=frozen,
        programs=roots,
        program_families={"p1": "synthetic"},
        groups=groups,
        run_program=run_program,
    )[1]

    assert calls == ["p1", "p1"]
    assert second["program_runtime_implementation_version"] == 3
    assert second["entries"][0]["program_runtime_implementation_version"] == 3


def test_incremental_progress_index_expands_after_each_program_and_is_stable_on_reuse(
    tmp_path: Path,
) -> None:
    frozen = _frozen(tmp_path, ("p2", "p1"))
    groups = _groups()
    roots = _roots(tmp_path, frozen.program_ids)

    def interrupted(program_id: str, _root: Path):
        if program_id == "p2":
            raise RuntimeError("injected program boundary interruption")
        return _synthetic_result(frozen, groups, program_id)

    with pytest.raises(RuntimeError, match="injected program boundary"):
        cli._run_program_checkpoints(
            frozen=frozen,
            programs=roots,
            program_families={program_id: "synthetic" for program_id in roots},
            groups=groups,
            run_program=interrupted,
        )
    progress_path = frozen.out_dir / "raw" / "program_checkpoint_progress.json"
    partial = json.loads(progress_path.read_text(encoding="utf-8"))
    assert partial["completed_program_count"] == 1
    assert [entry["program_id"] for entry in partial["entries"]] == ["p1"]
    supplied_id = partial.pop("progress_id")
    assert supplied_id == cli._sha256_bytes(
        json.dumps(partial, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    )

    _combined, complete = cli._run_program_checkpoints(
        frozen=frozen,
        programs=roots,
        program_families={program_id: "synthetic" for program_id in roots},
        groups=groups,
        run_program=lambda program_id, _root: _synthetic_result(
            frozen, groups, program_id
        ),
    )
    assert complete["completed_program_count"] == 2
    before = progress_path.read_bytes()
    before_mtime = progress_path.stat().st_mtime_ns
    _combined_again, complete_again = cli._run_program_checkpoints(
        frozen=frozen,
        programs=roots,
        program_families={program_id: "synthetic" for program_id in roots},
        groups=groups,
        run_program=lambda *_args: pytest.fail("complete progress invoked runner"),
    )
    assert complete_again == complete
    assert progress_path.read_bytes() == before
    assert progress_path.stat().st_mtime_ns == before_mtime


def test_complete_run_writer_lock_rejects_a_second_run_before_checkpoint_write(
    tmp_path: Path,
) -> None:
    frozen = _frozen(tmp_path, ("p1",))
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    holder = context.Process(
        target=_hold_study_writer_lock,
        args=(str(frozen.out_dir), ready, release),
    )
    holder.start()
    try:
        assert ready.wait(15), "child process did not acquire the writer lock"
        with pytest.raises(RuntimeError, match="study run writer lock"):
            cli._run_frozen(frozen)
    finally:
        release.set()
        holder.join(15)
        if holder.is_alive():
            holder.terminate()
            holder.join(5)

    assert holder.exitcode == 0
    assert not (frozen.out_dir / "raw" / "program-checkpoints").exists()


def test_complete_cli_reuse_starts_no_worker_or_helper_and_preserves_raw_mtimes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import advisor_study.direct_merge as direct_merge

    frozen = _frozen(tmp_path, ("p1",))
    groups = _groups()
    roots = _roots(tmp_path, frozen.program_ids)
    cli._run_program_checkpoints(
        frozen=frozen,
        programs=roots,
        program_families={"p1": "synthetic"},
        groups=groups,
        run_program=lambda program_id, _root: _synthetic_result(
            frozen, groups, program_id
        ),
    )
    actions = {action.action_id: action for action in groups["Uall"]}
    manifest = {
        "tools": {
            "opt": {"path": str(tmp_path / "opt.exe")},
            "merge_helper": {"path": str(tmp_path / "merge-helper.exe")},
            "worker": {"path": str(tmp_path / "worker.exe")},
        },
        "program_manifest": [
            {
                "program_id": "p1",
                "root_ir_path": str(roots["p1"]),
                "root_hard_state_id": "a" * 64,
                "program_family": "synthetic",
            }
        ],
    }
    resource_calls = {"worker": 0, "helper": 0}

    class UnexpectedWorker:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            resource_calls["worker"] += 1
            pytest.fail("complete checkpoint reuse constructed Worker")

    class UnexpectedMergeClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            resource_calls["helper"] += 1
            pytest.fail("complete checkpoint reuse constructed merge helper")

    monkeypatch.setattr(cli, "_manifest_for_frozen", lambda _frozen: manifest)
    monkeypatch.setattr(cli, "_load_actions", lambda _out_dir: actions)
    monkeypatch.setattr(cli, "_phasebatch_hard_state_id", lambda _path: "a" * 64)
    monkeypatch.setattr(cli, "_WorkerRunner", UnexpectedWorker)
    monkeypatch.setattr(direct_merge, "DirectMergeClient", UnexpectedMergeClient)

    cli._run_frozen(frozen)
    raw_mtimes = {
        path.relative_to(frozen.out_dir / "raw").as_posix(): path.stat().st_mtime_ns
        for path in (frozen.out_dir / "raw").rglob("*")
        if path.is_file()
    }
    cli._run_frozen(frozen)

    assert resource_calls == {"worker": 0, "helper": 0}
    assert raw_mtimes == {
        path.relative_to(frozen.out_dir / "raw").as_posix(): path.stat().st_mtime_ns
        for path in (frozen.out_dir / "raw").rglob("*")
        if path.is_file()
    }


def test_active_handoff_binds_checkpoint_index_and_rejects_selected_pointer_tamper(
    tmp_path: Path,
) -> None:
    frozen = _frozen(tmp_path, ("p1",))
    groups = _groups()
    roots = _roots(tmp_path, frozen.program_ids)
    combined, index = cli._run_program_checkpoints(
        frozen=frozen,
        programs=roots,
        program_families={"p1": "synthetic"},
        groups=groups,
        run_program=lambda program_id, _root: _synthetic_result(frozen, groups, program_id),
    )
    rows = cli._raw_rows_from_compacted_result(combined)
    ledger = cli._cleanup_ledger_from_checkpoint_index(
        frozen=frozen, checkpoint_index=index
    )
    cli._publish_cleanup_handoff(
        frozen.out_dir,
        rows,
        frozen.study_manifest_id,
        ledger,
        checkpoint_index=index,
    )
    artifact_bindings: list[dict[str, object]] = []
    assert cli._raw_rows_from_complete(
        frozen,
        expected_checkpoint_index=index,
        expected_rows=rows,
        materialized_artifact_bindings_out=artifact_bindings,
    ) == rows
    payloads = cli._validate_program_checkpoint_index(frozen, index)
    assert artifact_bindings == [
        dict(binding)
        for program_id in sorted(payloads)
        for binding in payloads[program_id]["materialized_artifact_bindings"]
    ]

    entry = index["entries"][0]
    active = Path(entry["checkpoint_dir"]) / "active.json"
    pointer = json.loads(active.read_text(encoding="utf-8"))
    pointer["result_sha256"] = "0" * 64
    active.write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint"):
        cli._raw_rows_from_complete(
            frozen,
            expected_checkpoint_index=index,
            expected_rows=rows,
        )


def test_runtime_limitation_rows_include_zero_category_and_explicit_denominator() -> None:
    base = {
        "study_manifest_id": "manifest-1",
        "authority_granted": "false",
        "proved_commute": "false",
    }
    group_rows = [
        {
            **base,
            "row_id": f"group-{group}",
            "group_id": group,
            "program_id": "p1",
            "round1_status": "complete",
            "first_round_disjoint_status": "disjoint",
            "all_n_merge_status": "complete",
            "all_n_second_round_status": "complete",
            "fail_closed_reason": "",
        }
        for group in ("U14", "U30", "Uall")
    ]
    rows = aggregate._limitations(
        {"U14": ("A",), "U30": ("A",), "Uall": ("A",)},
        group_rows,
        (),
        (),
        "manifest-1",
    )
    runtime = [row for row in rows if row["limitation_kind"] == "runtime_budget_exceeded"]
    assert len(runtime) == 3
    assert all(row["case_count"] == 0 for row in runtime)
    assert all(row["denominator_count"] == 1 for row in runtime)
    assert all(row["denominator_source_row_ids"] for row in runtime)

    limited = copy.deepcopy(group_rows)
    limited[0]["fail_closed_reason"] = "runtime_budget_exceeded;budget_s=600"
    rows = aggregate._limitations(
        {"U14": ("A",), "U30": ("A",), "Uall": ("A",)},
        limited,
        (),
        (),
        "manifest-1",
    )
    by_group = {
        row["group_id"]: row
        for row in rows
        if row["limitation_kind"] == "runtime_budget_exceeded"
    }
    assert by_group["U14"]["case_count"] == 1
    assert by_group["U30"]["case_count"] == 0
    assert by_group["Uall"]["case_count"] == 0
