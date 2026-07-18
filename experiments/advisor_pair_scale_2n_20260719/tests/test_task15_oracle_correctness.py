from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import advisor_study.cli as cli
import advisor_study.orchestration as orchestration
from advisor_study.pass_universe import ActionRecord


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module(path: Path, return_value: int = 0) -> None:
    path.write_text(
        "; ModuleID = 'fixture'\n"
        'source_filename = "fixture.c"\n'
        'target triple = "x86_64-w64-windows-gnu"\n\n'
        f"define i32 @f() {{\nentry:\n  ret i32 {return_value}\n}}\n",
        encoding="utf-8",
        newline="\n",
    )


def test_phasebatch_hard_state_is_path_independent_and_real_comparator_distinguishes_states(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left" / "same.ll"
    right = tmp_path / "right" / "same.ll"
    different = tmp_path / "different.ll"
    left.parent.mkdir()
    right.parent.mkdir()
    _module(left, 0)
    _module(right, 0)
    _module(different, 1)
    opt = Path("E:/llvm/build/bin/opt.exe")
    llvm_diff = opt.parent / "llvm-diff.exe"
    if not opt.is_file() or not llvm_diff.is_file():
        pytest.skip("frozen LLVM tools are unavailable")

    assert cli._phasebatch_hard_state_id(left) == cli._phasebatch_hard_state_id(right)
    equal = cli._compare_phasebatch_hard_states(
        {"output_path": str(left)}, {"output_path": str(right)}, opt=opt, timeout_s=10
    )
    unequal = cli._compare_phasebatch_hard_states(
        {"output_path": str(left)}, {"output_path": str(different)}, opt=opt, timeout_s=10
    )

    assert equal is not None and equal.trusted_hard_comparator and equal.can_hard_fold
    assert unequal is not None and unequal.trusted_hard_comparator and not unequal.can_hard_fold


def test_real_worker_reports_phasebatch_noop_and_rejects_targetmachine_only_action(
    tmp_path: Path,
) -> None:
    worker_path = Path("E:/PO2/worker/build/phasebatch-worker.exe")
    if not worker_path.is_file():
        pytest.skip("frozen Worker is unavailable")
    root = tmp_path / "root.ll"
    _module(root)
    runner = cli._WorkerRunner(worker_path, timeout_s=15)
    try:
        noop = runner.apply(
            root,
            SimpleNamespace(pipeline="dce"),
            tmp_path / "noop" / "out.ll",
        )
        targetmachine = {
            pipeline: runner.apply(
                root,
                SimpleNamespace(pipeline=pipeline),
                tmp_path / "targetmachine" / pipeline / "out.ll",
            )
            for pipeline in (
                "complex-deinterleaving",
                "interleaved-load-combine",
                "typepromotion",
                "interleaved-access",
            )
        }
    finally:
        runner.close()

    assert noop["success"] is True
    assert noop["hard_state_id"] == cli._phasebatch_hard_state_id(root)
    assert noop["hard_state_source"] == "phasebatch_hard_state_policy"
    assert all(result["success"] is False for result in targetmachine.values())
    assert all("TargetMachine" in str(result["stderr"]) for result in targetmachine.values())


def test_prepare_cli_wires_eligibility_preflight_to_worker_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FrozenRecord:
        selection_class = "fixed"
        root_ir_path = "unused.ll"

        def __init__(self, program_id: str) -> None:
            self.program_id = program_id

        def as_manifest_record(self) -> dict[str, str]:
            return {"program_id": self.program_id}

    class FakeWorker:
        def __init__(self, _path: Path, *, timeout_s: float) -> None:
            self.timeout_s = timeout_s
            self.closed = False

        def apply(self, _parent: Path, _action: object, _output: Path) -> object:
            raise AssertionError("prepare_study stub does not execute preflight")

        def close(self) -> None:
            self.closed = True

    actions = tuple(
        ActionRecord.for_function_candidate(
            name=f"p{index}", pipeline=f"p{index}", config_index=index
        )
        for index in range(14)
    )
    records = tuple(FrozenRecord(f"program-{index}") for index in range(50))
    workers: list[FakeWorker] = []
    captured: dict[str, object] = {}

    def fake_worker(path: Path, *, timeout_s: float) -> FakeWorker:
        value = FakeWorker(path, timeout_s=timeout_s)
        workers.append(value)
        return value

    def fake_prepare_study(**kwargs: object) -> SimpleNamespace:
        captured["dependencies"] = kwargs["dependencies"]
        return SimpleNamespace(
            study_manifest_path=tmp_path / "output" / "smoke" / "study_manifest.json",
            study_manifest_id="a" * 64,
            program_count=3,
            group_sizes={"U14": 14, "U30": 30, "Uall": 70},
            scale_gate="eligible_pass_count_at_least_60",
        )

    monkeypatch.setattr(cli, "_require_output_kind", lambda _path: "smoke")
    monkeypatch.setattr(cli, "load_frozen_policy", lambda _path: {"llvm_commit": "commit"})
    monkeypatch.setattr(cli, "_source_entries", lambda *_args: tuple(range(50)))
    monkeypatch.setattr(cli, "load_u14_actions", lambda _path: actions)
    monkeypatch.setattr(
        cli, "_existing_root_only_records", lambda *_args, **_kwargs: (records, {})
    )
    monkeypatch.setattr(cli, "_tool_record", lambda path, name: {"path": str(path), "sha256": name})
    monkeypatch.setattr(
        cli,
        "_run_process",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="commit", stderr=""),
    )
    monkeypatch.setattr(cli, "_WorkerRunner", fake_worker)
    monkeypatch.setattr(cli, "prepare_study", fake_prepare_study)
    output = tmp_path / "output" / "smoke"
    source_manifest = tmp_path / "selected.yaml"
    source_manifest.write_text("programs: []\n", encoding="utf-8", newline="\n")
    args = SimpleNamespace(
        output=output,
        smoke_programs=3,
        jobs=2,
        timeout=7.0,
        pass_policy=tmp_path / "policy.json",
        source_manifest=source_manifest,
        single_source_root=tmp_path / "SingleSource",
        core_passes=tmp_path / "core.yaml",
        opt=tmp_path / "opt.exe",
        clang=tmp_path / "clang.exe",
        worker=tmp_path / "worker.exe",
        merge_helper=tmp_path / "merge.exe",
    )

    cli._prepare_frozen(args)

    dependencies = captured["dependencies"]
    assert workers and getattr(dependencies.run_single, "__self__", None) is workers[0]
    assert workers[0].closed is True
    assert "def run_opt" not in inspect.getsource(cli._prepare_frozen)


def test_legacy_stage_completion_is_not_reused_after_execution_semantics_revision(
    tmp_path: Path,
) -> None:
    root = tmp_path / "isolated"
    stage = root / "raw" / "stage"
    stage.mkdir(parents=True)
    result = stage / "result.json"
    result.write_text('{"value":"legacy"}\n', encoding="utf-8", newline="\n")
    input_digest = "b" * 64
    (stage / "complete.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "input_sha256": input_digest,
                "full_digest_identity": input_digest,
                "files": {"result.json": _sha256(result)},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    calls: list[str] = []

    def produce(_directory: Path) -> dict[str, str]:
        calls.append("current")
        return {"value": "current"}

    payload, reused = orchestration._run_or_reuse_stage(
        stage,
        produce,
        expected_input_sha256=input_digest,
        isolation_root=root,
    )

    marker = json.loads((stage / "complete.json").read_text(encoding="utf-8"))
    assert payload == {"value": "current"}
    assert reused is False
    assert calls == ["current"]
    assert marker["schema_version"] == orchestration.STAGE_COMPLETION_SCHEMA_VERSION
    assert marker["raw_execution_semantics_revision"] == (
        orchestration.RAW_EXECUTION_SEMANTICS_REVISION
    )
