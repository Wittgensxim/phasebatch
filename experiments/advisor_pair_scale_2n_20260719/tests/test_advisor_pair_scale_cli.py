"""Frozen CLI boundary tests for the isolated advisor study.

These tests intentionally exercise the parser and the manifest gate without
running LLVM.  The execution adapters are separately covered by the prepare
and orchestration suites; the command boundary must reject invalid input
*before* it can invoke one of those adapters.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from itertools import combinations
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

import advisor_study.cli as cli
from advisor_study.direct_merge import DirectMergeClient, evaluate_group_2n
from advisor_study.manifest import ProgramRecord, build_study_manifest
from advisor_study.orchestration import _run_replay_family, _validate_replay_record
from advisor_study.pair_matrix import (
    HardStateEquality,
    profile_single_passes,
    run_complete_pair_matrix,
)
from advisor_study.pass_universe import ActionRecord


_FORMAL_SOURCE_POSITIONS = (3, 8, 13, 18, 23, 28, 33, 38, 43, 48)
_FORMAL_SELECTION_RULE_ID = "systematic_midpoint_fixed50_n10_v1"


@pytest.fixture(autouse=True)
def _isolated_experiment_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "EXPERIMENT_ROOT", tmp_path / "experiment")


def _digest(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _write_self_hashed_json(path: Path, payload: dict[str, object]) -> None:
    unsigned = dict(payload)
    unsigned.pop("document_sha256", None)
    unsigned["document_sha256"] = cli.canonical_sha256(unsigned)
    path.write_text(json.dumps(unsigned, sort_keys=True), encoding="utf-8")


def _resign_study_manifest(path: Path, manifest: dict[str, object]) -> str:
    programs = manifest["program_manifest"]
    assert isinstance(programs, list)
    manifest["program_manifest_sha256"] = cli.canonical_sha256(programs)
    identity = {
        key: value
        for key, value in manifest.items()
        if key not in {"study_manifest_id", "study_manifest_sha256"}
    }
    manifest_id = cli.canonical_sha256(identity)
    manifest["study_manifest_id"] = manifest_id
    manifest["study_manifest_sha256"] = manifest_id
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest_id


def _replay_stage_results(
    hard_state_hashes: dict[str, str], artifact_sha256: dict[str, str]
) -> dict[str, dict[str, str]]:
    return {
        name: {
            "execution_status": "success",
            "verifier_status": "success",
            "hard_state_id": hard_state_hashes[name],
            "output_sha256": artifact_sha256[name],
            "command_sha256": _digest(f"command-{name}"),
            "stderr_sha256": _digest(""),
            "error_fingerprint": _digest(f"success-{name}"),
        }
        for name in ("A", "B", "AB", "BA")
    }


def _program(root: Path, index: int) -> ProgramRecord:
    family = f"Family{index // 5:02d}"
    source = root / "llvm" / "SingleSource" / "Benchmarks" / family / f"p{index}.c"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(f"int main(void) {{ return {index}; }}\n", encoding="utf-8")
    root_ir = root / "roots" / f"p{index}.ll"
    root_ir.parent.mkdir(parents=True, exist_ok=True)
    root_ir.write_text(f"; ModuleID = 'p{index}'\n", encoding="utf-8")
    return ProgramRecord(
        program_id=f"p{index}",
        source_path=str(source.resolve()),
        relative_path=f"SingleSource/Benchmarks/{family}/p{index}.c",
        program_family=f"SingleSource/Benchmarks/{family}",
        source_sha256=_digest(source.read_bytes()),
        source_size_bytes=source.stat().st_size,
        compile_command=("clang", "-S", str(source)),
        compile_status="success",
        compile_stderr_sha256="",
        root_ir_path=str(root_ir.resolve()),
        root_ir_sha256=_digest(root_ir.read_bytes()),
        root_hard_state_id=cli._phasebatch_hard_state_id(root_ir),
        target="x86_64-w64-windows-gnu",
        data_layout="e-m:w-p:64:64",
        preflight_status="success",
        selection_class="fixed",
        selection_order=index + 1,
    )


def _actions(count: int) -> tuple[ActionRecord, ...]:
    return tuple(
        ActionRecord.for_function_candidate(
            name=f"pass-{index}", pipeline=f"pass-{index}", config_index=index
        )
        for index in range(count)
    )


def _write_frozen_manifest(
    tmp_path: Path,
    *,
    program_count: int = 3,
    u30_extra_count: int = 16,
    nested: bool = True,
    output_kind: str = "smoke",
) -> Path:
    """Build a self-validating, smoke-sized frozen manifest and sidecars."""

    if output_kind not in {"smoke", "formal"}:
        raise ValueError("test fixture output_kind must be smoke or formal")
    out_dir = tmp_path / "experiment" / "output" / output_kind
    out_dir.mkdir(parents=True)
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    tools: dict[str, dict[str, object]] = {}
    for name in ("opt", "clang", "worker", "merge_helper"):
        path = tools_dir / f"{name}.exe"
        path.write_text(name, encoding="utf-8")
        tools[name] = {"path": str(path), "sha256": _digest(path.read_bytes())}
    llvm_diff = tools_dir / "llvm-diff.exe"
    llvm_diff.write_text("llvm-diff", encoding="utf-8")
    actions = _actions(31)
    u14 = tuple(action.action_id for action in actions[:14])
    u30 = tuple(action.action_id for action in actions[: 14 + u30_extra_count])
    uall = tuple(action.action_id for action in actions)
    if not nested:
        u30 = (actions[-1].action_id,) + u30[1:]
    groups = {
        name: {
            "group_id": name,
            "group_size": len(ids),
            "action_ids": list(ids),
        }
        for name, ids in (("U14", u14), ("U30", u30), ("Uall", uall))
    }
    exclusion_payload = {
        "schema_version": "advisor-pair-scale-2n/candidate-identity-exclusions-v1",
        "candidate_inventory_count": 0,
        "candidate_reserve_count": 0,
        "exclusion_count": 0,
        "exclusions": [],
        "exclusions_sha256": cli.canonical_sha256([]),
    }
    exclusion_payload["document_sha256"] = cli.canonical_sha256(exclusion_payload)
    exclusion_path = out_dir / "candidate_identity_exclusions.json"
    exclusion_path.write_text(
        json.dumps(exclusion_payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    source_programs: tuple[ProgramRecord, ...] = ()
    if output_kind == "formal":
        source_programs = tuple(_program(tmp_path, index) for index in range(50))
        source_indexes = [position - 1 for position in _FORMAL_SOURCE_POSITIONS]
        source_indexes.extend(
            index for index in range(50) if index not in source_indexes
        )
        programs = tuple(
            replace(
                source_programs[source_index],
                selection_order=selection_order,
            )
            for selection_order, source_index in enumerate(
                source_indexes[:program_count], start=1
            )
        )
    else:
        programs = tuple(_program(tmp_path, index) for index in range(program_count))
    reserve_rows = []
    program_sidecar = out_dir / "program_manifest.json"
    program_sidecar.write_text(
        json.dumps(
            {
                "programs": [program.as_manifest_record() for program in programs],
                "reserve_order": reserve_rows,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    sampling_frame_path: Path | None = None
    sampling_frame_sha256 = ""
    if output_kind == "formal":
        source_rows = [program.as_manifest_record() for program in source_programs]
        expected_selected = [
            replace(
                source_programs[source_position - 1],
                selection_order=selection_order,
            ).as_manifest_record()
            for selection_order, source_position in enumerate(
                _FORMAL_SOURCE_POSITIONS,
                start=1,
            )
        ]
        frame: dict[str, object] = {
            "schema_version": "advisor-pair-scale-2n/formal-sampling-frame-v1",
            "source_inventory_count": 50,
            "selection_rule_id": _FORMAL_SELECTION_RULE_ID,
            "source_positions": list(_FORMAL_SOURCE_POSITIONS),
            "source_programs": source_rows,
            "source_programs_sha256": cli.canonical_sha256(source_rows),
            "selected_programs": expected_selected,
            "selected_programs_sha256": cli.canonical_sha256(expected_selected),
        }
        sampling_frame_path = out_dir / "formal_sampling_frame.json"
        _write_self_hashed_json(sampling_frame_path, frame)
        sampling_frame_sha256 = _digest(sampling_frame_path.read_bytes())
    manifest = build_study_manifest(
        programs=programs,
        pass_policy={"policy": "frozen"},
        pass_inventory={"actions": [action.as_manifest_record() for action in actions]},
        pass_preflight={"complete": True},
        pass_groups=groups,
        llvm_commit="a" * 40,
        target="x86_64-w64-windows-gnu",
        tools=tools,
        hard_state_policy={
            "policy_id": "phasebatch-hard-state-v1-debug-insensitive",
            "implementation": "phasebatch.ir_equivalence.hard_state_hash",
            "raw_execution_semantics_revision": cli.RAW_EXECUTION_SEMANTICS_REVISION,
        },
        comparator={
            "comparator_id": (
                "phasebatch-hard-state-v1-debug-insensitive@phasebatch.ir_equivalence.v2"
            ),
            "implementation": "phasebatch.ir_equivalence.compare_hard_states",
            "comparator_version": "phasebatch.ir_equivalence.v2",
            "llvm_diff": {
                "path": str(llvm_diff),
                "sha256": _digest(llvm_diff.read_bytes()),
            },
            "raw_execution_semantics_revision": cli.RAW_EXECUTION_SEMANTICS_REVISION,
        },
        jobs=2,
        timeout_s=3,
        artifact_policy=(
            {
                "retained": True,
                "formal_program_count": 10,
                "fixed_program_count": 10,
                "formal_source_inventory_count": 50,
                "formal_selection_rule_id": _FORMAL_SELECTION_RULE_ID,
                "formal_source_positions": list(_FORMAL_SOURCE_POSITIONS),
                "formal_sampling_frame_sha256": sampling_frame_sha256,
                "candidate_reserve_count": 0,
                "candidate_inventory_count": 0,
                "candidate_identity_exclusion_count": 0,
                "candidate_identity_exclusions_sha256": _digest(exclusion_path.read_bytes()),
                "selection_seed": 0,
                "raw_execution_semantics_revision": cli.RAW_EXECUTION_SEMANTICS_REVISION,
            }
            if output_kind == "formal"
            else {
                "retained": True,
                "raw_execution_semantics_revision": cli.RAW_EXECUTION_SEMANTICS_REVISION,
            }
        ),
    )
    manifest_path = out_dir / "study_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    (out_dir / "pass_groups.json").write_text(json.dumps(groups, sort_keys=True), encoding="utf-8")
    (out_dir / "pass_inventory.json").write_text(
        json.dumps([{"action": action.as_manifest_record()} for action in actions], sort_keys=True),
        encoding="utf-8",
    )
    hashed_paths = [
        manifest_path,
        out_dir / "pass_groups.json",
        out_dir / "pass_inventory.json",
        exclusion_path,
        program_sidecar,
    ]
    if sampling_frame_path is not None:
        hashed_paths.append(sampling_frame_path)
    files = {
        path.relative_to(out_dir).as_posix(): _digest(path.read_bytes())
        for path in hashed_paths
    }
    completion: dict[str, object] = {
        "schema_version": "advisor-pair-scale-2n/prepare-v1",
        "study_manifest_id": manifest["study_manifest_id"],
        "authority_granted": False,
        "proved_commute": False,
        "program_count": program_count,
        "formal_program_count": 10 if output_kind == "formal" else 0,
        "fixed_program_count": 10 if output_kind == "formal" else program_count,
        "formal_source_inventory_count": 50 if output_kind == "formal" else 0,
        "formal_selection_rule_id": (
            _FORMAL_SELECTION_RULE_ID if output_kind == "formal" else ""
        ),
        "formal_source_positions": (
            list(_FORMAL_SOURCE_POSITIONS) if output_kind == "formal" else []
        ),
        "formal_sampling_frame_sha256": sampling_frame_sha256,
        "candidate_reserve_count": 0,
        "candidate_inventory_count": 0,
        "candidate_identity_exclusion_count": 0,
        "candidate_identity_exclusions_sha256": _digest(exclusion_path.read_bytes()),
        "selection_seed": 0,
        "group_sizes": {name: value["group_size"] for name, value in groups.items()},
        "scale_gate": "eligible_pass_count_below_60",
        "files_sha256": files,
    }
    _write_self_hashed_json(out_dir / "prepare_complete.json", completion)
    return manifest_path


def test_cli_rejects_unknown_and_forbidden_authority_arguments() -> None:
    with pytest.raises(SystemExit, match="2"):
        cli.main(["prepare", "--batch"])
    with pytest.raises(SystemExit, match="2"):
        cli.main(["prepare", "--definitely-unknown"])


def test_worker_runner_pool_pins_one_runner_per_thread_and_closes_all(
    tmp_path: Path,
) -> None:
    created: list[object] = []
    first_calls = threading.Barrier(2, timeout=5)

    class FakeRunner:
        def __init__(self, _path: Path, *, timeout_s: float) -> None:
            self.timeout_s = timeout_s
            self.thread_ids: set[int] = set()
            self.active = False
            self.closed = 0
            created.append(self)

        def apply(self, _parent: Path, _action: object, output: Path) -> dict[str, object]:
            assert not self.active
            self.active = True
            self.thread_ids.add(threading.get_ident())
            try:
                if output.name == "first.ll":
                    first_calls.wait()
                return {"success": True, "output_path": str(output)}
            finally:
                self.active = False

        def close(self) -> None:
            self.closed += 1

    pool = cli._WorkerRunnerPool(
        tmp_path / "worker.exe",
        timeout_s=3.0,
        jobs=2,
        runner_factory=FakeRunner,
    )

    def invoke(index: int) -> None:
        pool.apply(Path("S.ll"), object(), tmp_path / str(index) / "first.ll")
        pool.apply(Path("S.ll"), object(), tmp_path / str(index) / "second.ll")

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(invoke, range(2)))
    pool.release_thread_bindings()
    pool.close()

    assert len(created) == 2
    assert all(len(runner.thread_ids) == 1 for runner in created)  # type: ignore[attr-defined]
    assert len(set().union(*(runner.thread_ids for runner in created))) == 2  # type: ignore[attr-defined]
    assert all(runner.closed == 1 for runner in created)  # type: ignore[attr-defined]


def test_successful_worker_stderr_is_request_local_and_pair_rows_are_jobs_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeProcess:
        def __init__(self, historical_stderr: str) -> None:
            self.stderr_text = historical_stderr
            self.closed = 0

        def request(self, *_args: object, **_kwargs: object) -> object:
            return SimpleNamespace(payload={"status": "ok"})

        def close(self) -> None:
            self.closed += 1

    def make_runner(historical_stderr: str) -> cli._WorkerRunner:
        runner = object.__new__(cli._WorkerRunner)
        runner._process = FakeProcess(historical_stderr)
        runner._timeout_s = 3.0
        runner._worker_path = tmp_path / "worker.exe"

        def request(operation: str, **payload: object) -> dict[str, object]:
            if operation == "load":
                return {"module_handle": "loaded-parent"}
            if operation == "apply":
                output = Path(str(payload["materialize_path"]))
                output.write_text(
                    f"{Path(str(payload['parent_handle'])).name}:{payload['pipeline']}:{output.name}\n",
                    encoding="utf-8",
                )
                return {
                    "module_handle": f"apply-{threading.get_ident()}",
                    "canonical_hash": "c" * 64,
                }
            raise AssertionError(f"unexpected operation: {operation}")

        runner._request = request
        return runner

    monkeypatch.setattr(
        cli,
        "_phasebatch_hard_state_id",
        lambda path: _digest(Path(path).read_bytes()),
    )
    root = tmp_path / "S.ll"
    root.write_text("root\n", encoding="utf-8")
    actions = tuple(
        ActionRecord.for_function_candidate(
            name=name,
            pipeline=name,
            config_index=index,
        )
        for index, name in enumerate("ABC")
    )
    profiles: list[dict[str, object]] = []
    for action in actions:
        output = tmp_path / f"{action.name}.ll"
        output.write_text(f"{action.name}\n", encoding="utf-8")
        profiles.append(
            {
                "action_id": action.action_id,
                "execution_status": "success",
                "output_path": str(output),
                "output_hard_state_id": _digest(output.read_bytes()),
                "output_sha256": _digest(output.read_bytes()),
                "verifier_status": "success",
                "activity_status": "active",
                "observed_effect_available": True,
                "changed_functions": (action.name,),
                "changed_blocks": (),
                "changed_module_regions": (),
            }
        )
    common = {
        "root_ir": root,
        "profiles": profiles,
        "actions": {action.action_id: action for action in actions},
        "out_dir": tmp_path / "pairs",
        "profile_artifact_root": tmp_path,
        "verify_ir": lambda _path: True,
        "compare": lambda _left, _right: HardStateEquality(
            False, "hard", trusted_hard_comparator=True
        ),
        "study_manifest_id": "manifest",
        "program_id": "program",
        "group_id": "Uall",
    }
    sequential_runner = make_runner("historical stderr from jobs1")
    sequential = run_complete_pair_matrix(
        **common,
        run_second=sequential_runner.apply,
        jobs=1,
    )

    pool_runners = iter(
        (
            make_runner("unrelated history from pool runner A"),
            make_runner("different history from pool runner B"),
        )
    )
    pool = cli._WorkerRunnerPool(
        tmp_path / "worker.exe",
        timeout_s=3.0,
        jobs=2,
        runner_factory=lambda *_args, **_kwargs: next(pool_runners),
    )
    parallel = run_complete_pair_matrix(
        **common,
        run_second=pool.apply,
        jobs=2,
    )
    pool.release_thread_bindings()
    pool.close()
    sequential_runner.close()

    assert [(row["stderr"], row["stderr_sha256"]) for row in parallel] == [
        (row["stderr"], row["stderr_sha256"]) for row in sequential
    ]
    assert all(row["stderr"] == "" for row in sequential)
    assert all(row["stderr_sha256"] == _digest("") for row in sequential)


@pytest.mark.parametrize(
    ("error_type", "expected_status"),
    [("timeout", "timeout"), ("error", "error")],
)
def test_worker_failure_keeps_current_exception_text(
    tmp_path: Path,
    error_type: str,
    expected_status: str,
) -> None:
    from phasebatch.opt_worker import WorkerError, WorkerTimeoutError

    runner = object.__new__(cli._WorkerRunner)
    runner._process = SimpleNamespace(
        stderr_text="unrelated historical stderr",
        request=lambda *_args, **_kwargs: None,
        close=lambda: None,
    )
    runner._timeout_s = 3.0
    runner._worker_path = tmp_path / "worker.exe"
    current = f"current {error_type} detail"

    def fail_request(_operation: str, **_payload: object) -> object:
        exception = WorkerTimeoutError if error_type == "timeout" else WorkerError
        raise exception(current)

    runner._request = fail_request
    result = runner.apply(
        tmp_path / "S.ll",
        SimpleNamespace(pipeline="instcombine"),
        tmp_path / "output.ll",
    )

    assert result["execution_status"] == expected_status
    assert result["stderr"] == current


def test_worker_runner_pool_rejects_boolean_jobs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="jobs must be a positive integer"):
        cli._WorkerRunnerPool(
            tmp_path / "worker.exe",
            timeout_s=3.0,
            jobs=True,
            runner_factory=lambda *_args, **_kwargs: object(),
        )


def test_worker_runner_pool_attempts_every_close_after_one_close_failure(
    tmp_path: Path,
) -> None:
    created: list[object] = []

    class FakeRunner:
        def __init__(self, _path: Path, *, timeout_s: float) -> None:
            self.index = len(created)
            self.timeout_s = timeout_s
            self.closed = 0
            created.append(self)

        def apply(self, *_args: object) -> object:
            return None

        def close(self) -> None:
            self.closed += 1
            if self.index == 0:
                raise RuntimeError("injected close failure")

    pool = cli._WorkerRunnerPool(
        tmp_path / "worker.exe",
        timeout_s=3.0,
        jobs=2,
        runner_factory=FakeRunner,
    )

    with pytest.raises(RuntimeError, match="injected close failure"):
        pool.close()

    assert len(created) == 2
    assert all(runner.closed == 1 for runner in created)  # type: ignore[attr-defined]


def test_worker_runtime_jobs_one_keeps_direct_single_runner_path(
    tmp_path: Path,
) -> None:
    created: list[object] = []

    class FakeRunner:
        def __init__(self, _path: Path, *, timeout_s: float) -> None:
            self.timeout_s = timeout_s
            self.closed = 0
            created.append(self)

        def apply(self, *_args: object) -> object:
            return None

        def close(self) -> None:
            self.closed += 1

    primary, pool = cli._create_worker_runners(
        tmp_path / "worker.exe",
        timeout_s=4.0,
        jobs=1,
        runner_factory=FakeRunner,
    )

    assert pool is None
    assert created == [primary]
    assert primary.timeout_s == 4.0
    primary.close()
    assert primary.closed == 1


def test_cli_scans_complete_singlesource_inventory_and_preflights_candidates(
    tmp_path: Path,
) -> None:
    single = tmp_path / "suite" / "SingleSource"
    for relative, payload in (
        ("A/fixed.c", "int fixed(void) { return 0; }\n"),
        ("A/extra.c", "int extra(void) { return 1; }\n"),
        ("B/fail.c", "int fail(void) { return 2; }\n"),
        ("C/large.c", "x" * 201),
        ("D/extra-copy.c", "int extra(void) { return 1; }\n"),
    ):
        source = single / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(payload, encoding="utf-8")
    fixed = _program(tmp_path, 0)
    fixed_source = single / "A" / "fixed.c"
    fixed = ProgramRecord(
        **{
            **fixed.__dict__,
            "program_id": "fixed",
            "source_path": str(fixed_source.resolve()),
            "relative_path": "SingleSource/A/fixed.c",
            "program_family": "SingleSource/A",
            "source_sha256": _digest(fixed_source.read_bytes()),
            "source_size_bytes": fixed_source.stat().st_size,
        }
    )
    calls: list[str] = []

    def compile_source(source: Path, output: Path) -> cli.RunResult:
        calls.append(source.name)
        if source.name == "fail.c":
            return cli.RunResult(False, output, stderr="compile failed", command=("clang", str(source)))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            'target datalayout = "e-m:w-p:64:64"\n'
            'target triple = "x86_64-w64-windows-gnu"\n'
            'define i32 @f() { ret i32 0 }\n',
            encoding="utf-8",
        )
        return cli.RunResult(
            True,
            output,
            hard_state_id=_digest(output.read_bytes()),
            command=("clang", str(source)),
        )

    entries = cli._scan_single_source_inventory(single)
    preflight = cli._preflight_candidate_records(
        entries,
        fixed_programs=(fixed,),
        single_source_root=single,
        preflight_root=tmp_path / "preflight",
        compile_source=compile_source,
        verify_ir=lambda path: path.is_file(),
        max_source_bytes=200,
    )

    assert [row.relative_path for row in entries] == [
        "SingleSource/A/extra.c",
        "SingleSource/A/fixed.c",
        "SingleSource/B/fail.c",
        "SingleSource/C/large.c",
        "SingleSource/D/extra-copy.c",
    ]
    records = preflight.records
    by_path = {row.relative_path: row for row in records}
    duplicate_paths = {"SingleSource/A/extra.c", "SingleSource/D/extra-copy.c"}
    retained_duplicate_paths = duplicate_paths & set(by_path)
    assert len(retained_duplicate_paths) == 1
    retained_duplicate_path = next(iter(retained_duplicate_paths))
    assert by_path[retained_duplicate_path].preflight_status == "success"
    assert by_path["SingleSource/B/fail.c"].preflight_status == "compile_failed"
    assert by_path["SingleSource/C/large.c"].preflight_status == "source_too_large"
    assert len(preflight.exclusions) == 1
    exclusion = preflight.exclusions[0]
    assert exclusion["relative_path"] in duplicate_paths
    assert exclusion["canonical_relative_path"] in duplicate_paths
    assert exclusion["relative_path"] != exclusion["canonical_relative_path"]
    assert exclusion["source_sha256"] == exclusion["canonical_source_sha256"]
    assert exclusion["stable_rank"] == cli.stable_rank(0, exclusion["relative_path"])
    assert exclusion["reason"] == "duplicate_source_sha256"
    assert set(calls) == {Path(retained_duplicate_path).name, "fail.c"}


def test_inventory_scan_rejects_resolved_tree_escape_portably(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    single = tmp_path / "suite" / "SingleSource"
    single.mkdir(parents=True)
    outside = tmp_path / "outside.c"
    outside.write_text("int outside(void) { return 0; }\n", encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "_walk_single_source_paths",
        lambda _root: (outside,),
    )

    with pytest.raises(ValueError, match="escapes resolved SingleSource root"):
        cli._scan_single_source_inventory(single)


def test_candidate_preflight_rejects_outside_source_before_read_or_compile(
    tmp_path: Path,
) -> None:
    single = tmp_path / "suite" / "SingleSource"
    single.mkdir(parents=True)
    outside = tmp_path / "outside.c"
    outside.write_text("int outside(void) { return 0; }\n", encoding="utf-8")
    calls: list[Path] = []

    with pytest.raises(ValueError, match="escapes resolved SingleSource root"):
        cli._preflight_candidate_records(
            (
                cli._SourceInventoryEntry(
                    program_id="outside",
                    source_path=outside,
                    relative_path="SingleSource/Outside/outside.c",
                ),
            ),
            fixed_programs=(),
            single_source_root=single,
            preflight_root=tmp_path / "preflight",
            compile_source=lambda source, output: (
                calls.append(source) or cli.RunResult(False, output)
            ),
            verify_ir=lambda _path: False,
        )

    assert calls == []


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("relative_path", "SingleSource/../outside.c"),
        ("program_id", ""),
        ("canonical_program_id", ""),
        ("canonical_relative_path", "SingleSource\\A\\canonical.c"),
        ("canonical_source_sha256", 7),
    ],
)
def test_resigned_malformed_candidate_exclusion_document_fails_closed(
    tmp_path: Path,
    field: str,
    bad_value: object,
) -> None:
    source = tmp_path / "suite" / "SingleSource" / "B" / "duplicate.c"
    canonical = tmp_path / "suite" / "SingleSource" / "A" / "canonical.c"
    for path in (source, canonical):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("int same(void) { return 0; }\n", encoding="utf-8")
    digest = _digest(source.read_bytes())
    row: dict[str, object] = {
        "program_id": "duplicate",
        "relative_path": "SingleSource/B/duplicate.c",
        "source_path": str(source.resolve()),
        "source_sha256": digest,
        "source_size_bytes": source.stat().st_size,
        "stable_rank": cli.stable_rank(0, "SingleSource/B/duplicate.c"),
        "canonical_program_id": "canonical",
        "canonical_relative_path": "SingleSource/A/canonical.c",
        "canonical_source_path": str(canonical.resolve()),
        "canonical_source_sha256": digest,
        "canonical_stable_rank": cli.stable_rank(0, "SingleSource/A/canonical.c"),
        "reason": "duplicate_source_sha256",
    }
    row[field] = bad_value
    if field == "relative_path":
        row["stable_rank"] = cli.stable_rank(0, str(bad_value))
    if field == "canonical_relative_path":
        row["canonical_stable_rank"] = cli.stable_rank(0, str(bad_value))
    unsigned: dict[str, object] = {
        "schema_version": "advisor-pair-scale-2n/candidate-identity-exclusions-v1",
        "candidate_inventory_count": 2,
        "candidate_reserve_count": 1,
        "exclusion_count": 1,
        "exclusions": [row],
        "exclusions_sha256": cli.canonical_sha256([row]),
    }
    document = {**unsigned, "document_sha256": cli.canonical_sha256(unsigned)}

    with pytest.raises(ValueError, match="candidate identity exclusion"):
        cli._validate_candidate_identity_exclusion_document(
            document, verify_sources=True
        )


@pytest.mark.parametrize("drift_field", ("triple", "data_layout"))
def test_fixed_root_target_identity_is_parsed_and_cross_checked(
    tmp_path: Path,
    drift_field: str,
) -> None:
    root_ir = tmp_path / "S.ll"
    root_ir.write_text(
        'target datalayout = "e-m:w-p:64:64"\n'
        'target triple = "x86_64-w64-windows-gnu"\n'
        'define i32 @main() { ret i32 0 }\n',
        encoding="utf-8",
    )
    manifest = {
        "target": {
            "triple": "x86_64-w64-windows-gnu",
            "data_layout": "e-m:w-p:64:64",
        }
    }

    assert cli._bound_root_target_identity(manifest, root_ir) == (
        "x86_64-w64-windows-gnu",
        "e-m:w-p:64:64",
    )

    drifted = json.loads(json.dumps(manifest))
    drifted["target"][drift_field] = "drifted"
    with pytest.raises(ValueError, match="target identity mismatch"):
        cli._bound_root_target_identity(drifted, root_ir)


def test_frozen_phase_gate_accepts_uall_outside_reporting_range(tmp_path: Path) -> None:
    # Uall=30 is reportable.  The CLI must never pad it to 60 or reject it.
    manifest = _write_frozen_manifest(tmp_path)
    frozen = cli.load_frozen_phase(manifest, phase="run")

    assert frozen.out_dir == manifest.parent
    assert frozen.program_count == 3
    assert len(frozen.groups["Uall"]) == 31
    assert frozen.jobs == 2
    assert frozen.timeout_s == 3.0


def test_frozen_phase_gate_accepts_only_systematic_midpoint_formal_ten_scope(
    tmp_path: Path,
) -> None:
    manifest = _write_frozen_manifest(
        tmp_path,
        program_count=10,
        output_kind="formal",
    )

    frozen = cli.load_frozen_phase(manifest, phase="run")

    assert frozen.program_count == 10
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    scope = raw["artifact_policy"]["value"]
    assert scope["formal_program_count"] == 10
    assert scope["fixed_program_count"] == 10
    assert scope["formal_source_inventory_count"] == 50
    assert scope["formal_selection_rule_id"] == _FORMAL_SELECTION_RULE_ID
    assert scope["formal_source_positions"] == list(_FORMAL_SOURCE_POSITIONS)
    assert scope["candidate_reserve_count"] == 0
    assert scope["candidate_inventory_count"] == 0
    assert scope["candidate_identity_exclusion_count"] == 0
    assert "formal_scope_user_override" not in scope
    selected = sorted(raw["program_manifest"], key=lambda row: row["selection_order"])
    assert [row["program_id"] for row in selected] == [
        "p2", "p7", "p12", "p17", "p22", "p27", "p32", "p37", "p42", "p47"
    ]
    assert [row["selection_order"] for row in selected] == list(range(1, 11))
    assert len({row["program_family"] for row in selected}) == 10


def test_formal_sampling_frame_is_read_once_for_hash_and_json_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _write_frozen_manifest(
        tmp_path,
        program_count=10,
        output_kind="formal",
    )
    frame_path = manifest.parent / "formal_sampling_frame.json"
    frozen_bytes = frame_path.read_bytes()
    reads: list[Path] = []

    def read_frame_once(path: Path) -> bytes:
        reads.append(Path(path))
        return frozen_bytes

    monkeypatch.setattr(
        cli,
        "_read_formal_sampling_frame_bytes",
        read_frame_once,
        raising=False,
    )

    assert cli.load_frozen_phase(manifest, phase="run").program_count == 10
    assert reads == [frame_path]


@pytest.mark.parametrize(
    ("output_kind", "program_count"),
    (("smoke", 3), ("formal", 10)),
)
@pytest.mark.parametrize("mutation", ("extra", "missing"))
def test_prepare_completion_rejects_resigned_extra_or_missing_fields(
    tmp_path: Path,
    output_kind: str,
    program_count: int,
    mutation: str,
) -> None:
    manifest = _write_frozen_manifest(
        tmp_path,
        program_count=program_count,
        output_kind=output_kind,
    )
    completion_path = manifest.parent / "prepare_complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if mutation == "extra":
        completion["unexpected_field"] = "resigned"
    else:
        completion.pop("scale_gate")
    _write_self_hashed_json(completion_path, completion)

    with pytest.raises(ValueError, match="prepare completion field set"):
        cli.load_frozen_phase(manifest, phase="run")


@pytest.mark.parametrize(
    ("field", "value", "remove", "match"),
    [
        ("formal_source_inventory_count", 49, False, "source inventory"),
        ("formal_source_inventory_count", None, True, "source inventory"),
        ("formal_selection_rule_id", "wrong-rule", False, "selection rule"),
        ("formal_selection_rule_id", None, True, "selection rule"),
        ("formal_source_positions", [3, 8, 13, 18, 23, 28, 33, 38, 43, 47], False, "source positions"),
        ("formal_source_positions", None, True, "source positions"),
    ],
)
def test_formal_phase_gate_rejects_missing_or_wrong_systematic_selection_scope(
    tmp_path: Path,
    field: str,
    value: object,
    remove: bool,
    match: str,
) -> None:
    manifest_path = _write_frozen_manifest(
        tmp_path,
        program_count=10,
        output_kind="formal",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scope = manifest["artifact_policy"]["value"]
    if remove:
        scope.pop(field)
    else:
        scope[field] = value
    manifest["artifact_policy"]["sha256"] = cli.canonical_sha256(scope)
    identity = {
        key: item
        for key, item in manifest.items()
        if key not in {"study_manifest_id", "study_manifest_sha256"}
    }
    manifest_id = cli.canonical_sha256(identity)
    manifest["study_manifest_id"] = manifest_id
    manifest["study_manifest_sha256"] = manifest_id
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    completion_path = manifest_path.parent / "prepare_complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["study_manifest_id"] = manifest_id
    completion["files_sha256"][manifest_path.name] = _digest(manifest_path.read_bytes())
    _write_self_hashed_json(completion_path, completion)

    with pytest.raises(ValueError, match=match):
        cli.load_frozen_phase(manifest_path, phase="run")


@pytest.mark.parametrize(
    "drift_field",
    (
        "program_id",
        "source_path",
        "relative_path",
        "program_family",
        "source_sha256",
        "root_ir_path",
        "root_ir_sha256",
        "root_hard_state_id",
    ),
)
def test_formal_gate_rejects_resigned_selected_program_identity_drift(
    tmp_path: Path,
    drift_field: str,
) -> None:
    manifest_path = _write_frozen_manifest(
        tmp_path,
        program_count=10,
        output_kind="formal",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    program_sidecar_path = manifest_path.parent / "program_manifest.json"
    program_sidecar = json.loads(program_sidecar_path.read_text(encoding="utf-8"))
    manifest_row = next(
        row for row in manifest["program_manifest"] if row["selection_order"] == 1
    )
    sidecar_row = next(
        row for row in program_sidecar["programs"] if row["selection_order"] == 1
    )

    changes: dict[str, object]
    if drift_field == "program_id":
        changes = {"program_id": "replacement-program"}
    elif drift_field == "source_path":
        replacement_source = tmp_path / "llvm" / "SingleSource" / "replacement.c"
        replacement_source.write_text("int replacement(void) { return 0; }\n", encoding="utf-8")
        changes = {"source_path": str(replacement_source.resolve())}
    elif drift_field == "relative_path":
        changes = {
            "relative_path": (
                f"{manifest_row['program_family']}/replacement-program.c"
            )
        }
    elif drift_field == "program_family":
        replacement_family = "SingleSource/Benchmarks/ReplacementFamily"
        changes = {
            "program_family": replacement_family,
            "relative_path": f"{replacement_family}/replacement-program.c",
        }
    elif drift_field == "source_sha256":
        changes = {"source_sha256": "b" * 64}
    elif drift_field == "root_ir_path":
        replacement_root = tmp_path / "roots" / "replacement-program.ll"
        replacement_root.write_text("; replacement root\n", encoding="utf-8")
        changes = {"root_ir_path": str(replacement_root.resolve())}
    elif drift_field == "root_ir_sha256":
        changes = {"root_ir_sha256": "c" * 64}
    else:
        changes = {"root_hard_state_id": "d" * 64}
    manifest_row.update(changes)
    sidecar_row.update(changes)

    manifest_id = _resign_study_manifest(manifest_path, manifest)
    program_sidecar_path.write_text(
        json.dumps(program_sidecar, sort_keys=True),
        encoding="utf-8",
    )
    completion_path = manifest_path.parent / "prepare_complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["study_manifest_id"] = manifest_id
    completion["files_sha256"][manifest_path.name] = _digest(
        manifest_path.read_bytes()
    )
    completion["files_sha256"][program_sidecar_path.name] = _digest(
        program_sidecar_path.read_bytes()
    )
    _write_self_hashed_json(completion_path, completion)

    with pytest.raises(ValueError, match="sampling frame|program binding"):
        cli.load_frozen_phase(manifest_path, phase="run")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("formal_program_count", 9),
        ("formal_program_count", False),
        ("fixed_program_count", 9),
        ("fixed_program_count", False),
        ("formal_source_inventory_count", 49),
        ("formal_source_inventory_count", False),
        ("formal_selection_rule_id", "wrong-rule"),
        ("formal_selection_rule_id", False),
        ("formal_source_positions", [3, 8, 13, 18, 23, 28, 33, 38, 43, 47]),
        ("formal_source_positions", False),
        ("candidate_reserve_count", 1),
        ("candidate_reserve_count", False),
        ("candidate_inventory_count", 1),
        ("candidate_inventory_count", False),
        ("candidate_identity_exclusion_count", 1),
        ("candidate_identity_exclusion_count", False),
        ("selection_seed", 1),
        ("selection_seed", False),
    ],
)
def test_formal_gate_rejects_resigned_prepare_completion_scope_drift(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    manifest_path = _write_frozen_manifest(
        tmp_path,
        program_count=10,
        output_kind="formal",
    )
    completion_path = manifest_path.parent / "prepare_complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion[field] = value
    _write_self_hashed_json(completion_path, completion)

    with pytest.raises(ValueError, match="prepare completion formal scope mismatch"):
        cli.load_frozen_phase(manifest_path, phase="run")


def test_formal_phase_gate_rejects_boolean_selection_seed_after_resigning(
    tmp_path: Path,
) -> None:
    manifest_path = _write_frozen_manifest(
        tmp_path,
        program_count=10,
        output_kind="formal",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scope = manifest["artifact_policy"]["value"]
    scope["selection_seed"] = False
    manifest["artifact_policy"]["sha256"] = cli.canonical_sha256(scope)
    identity = {
        key: value
        for key, value in manifest.items()
        if key not in {"study_manifest_id", "study_manifest_sha256"}
    }
    manifest_id = cli.canonical_sha256(identity)
    manifest["study_manifest_id"] = manifest_id
    manifest["study_manifest_sha256"] = manifest_id
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    completion_path = manifest_path.parent / "prepare_complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["study_manifest_id"] = manifest_id
    completion["files_sha256"][manifest_path.name] = _digest(manifest_path.read_bytes())
    _write_self_hashed_json(completion_path, completion)

    with pytest.raises(ValueError, match="selection_seed=0 as an integer"):
        cli.load_frozen_phase(manifest_path, phase="run")


def test_formal_phase_gate_validates_candidate_exclusion_self_hash(
    tmp_path: Path,
) -> None:
    manifest = _write_frozen_manifest(
        tmp_path,
        program_count=10,
        output_kind="formal",
    )
    sidecar = manifest.parent / "candidate_identity_exclusions.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["document_sha256"] = "0" * 64
    sidecar.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    completion_path = manifest.parent / "prepare_complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["files_sha256"][sidecar.name] = _digest(sidecar.read_bytes())
    _write_self_hashed_json(completion_path, completion)

    with pytest.raises(ValueError, match="candidate identity exclusions"):
        cli.load_frozen_phase(manifest, phase="run")


def test_formal_gate_rejects_resigned_extension_program_binding_drift(
    tmp_path: Path,
) -> None:
    manifest_path = _write_frozen_manifest(
        tmp_path,
        program_count=10,
        output_kind="formal",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["program_manifest"][0]["selection_class"] = "extension"
    manifest["program_manifest_sha256"] = cli.canonical_sha256(
        manifest["program_manifest"]
    )
    identity = {
        key: value
        for key, value in manifest.items()
        if key not in {"study_manifest_id", "study_manifest_sha256"}
    }
    manifest_id = cli.canonical_sha256(identity)
    manifest["study_manifest_id"] = manifest_id
    manifest["study_manifest_sha256"] = manifest_id
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    completion_path = manifest_path.parent / "prepare_complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["study_manifest_id"] = manifest_id
    completion["files_sha256"][manifest_path.name] = _digest(manifest_path.read_bytes())
    _write_self_hashed_json(completion_path, completion)

    with pytest.raises(ValueError, match="only existing fixed programs"):
        cli.load_frozen_phase(manifest_path, phase="run")


@pytest.mark.parametrize(
    ("kwargs", "phase", "match"),
    [
        ({"program_count": 9, "output_kind": "formal"}, "run", "exactly 10"),
        ({"program_count": 11, "output_kind": "formal"}, "run", "exactly 10"),
        ({"program_count": 50, "output_kind": "formal"}, "run", "exactly 10"),
        ({"nested": False}, "run", "U30 requires exactly 16"),
        ({"u30_extra_count": 15}, "run", "U30 requires exactly 16"),
    ],
)
def test_frozen_phase_gate_rejects_bad_formal_or_group_freeze(
    tmp_path: Path, kwargs: dict[str, object], phase: str, match: str
) -> None:
    manifest = _write_frozen_manifest(tmp_path, **kwargs)
    with pytest.raises(ValueError, match=match):
        cli.load_frozen_phase(manifest, phase=phase)


def test_run_and_summarize_accept_only_the_frozen_manifest_and_phase_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _write_frozen_manifest(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(cli, "_run_frozen", lambda frozen: calls.append(f"run:{frozen.study_manifest_id}"))
    monkeypatch.setattr(cli, "_summarize_frozen", lambda frozen: calls.append(f"summarize:{frozen.study_manifest_id}"))

    assert cli.main(["run", "--manifest", str(manifest)]) == 0
    assert cli.main(["summarize", "--manifest", str(manifest)]) == 0
    assert calls == [f"run:{cli.load_frozen_phase(manifest, phase='run').study_manifest_id}", f"summarize:{cli.load_frozen_phase(manifest, phase='summarize').study_manifest_id}"]


def test_phase_gate_rejects_tampered_completion_hash_and_missing_tools(tmp_path: Path) -> None:
    manifest = _write_frozen_manifest(tmp_path)
    complete_path = manifest.parent / "prepare_complete.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete["files_sha256"]["pass_groups.json"] = "0" * 64
    _write_self_hashed_json(complete_path, complete)
    with pytest.raises(ValueError, match="hash-validated"):
        cli.load_frozen_phase(manifest, phase="run")

    complete["files_sha256"]["pass_groups.json"] = _digest((manifest.parent / "pass_groups.json").read_bytes())
    _write_self_hashed_json(complete_path, complete)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    Path(raw["tools"]["worker"]["path"]).unlink()
    with pytest.raises(ValueError, match="worker tool is missing"):
        cli.load_frozen_phase(manifest, phase="run")


def test_phase_gate_rejects_a_portable_audit_copy_outside_the_experiment_root(tmp_path: Path) -> None:
    manifest = _write_frozen_manifest(tmp_path)
    copied = tmp_path / "audit-copy" / "output" / "smoke"
    copied.mkdir(parents=True)
    for path in manifest.parent.iterdir():
        (copied / path.name).write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="isolated experiment output"):
        cli.load_frozen_phase(copied / "study_manifest.json", phase="run")


def test_raw_gate_rejects_missing_required_raw_tables(tmp_path: Path) -> None:
    manifest = _write_frozen_manifest(tmp_path)
    frozen = cli.load_frozen_phase(manifest, phase="run")
    raw_dir = frozen.out_dir / "raw"
    raw_dir.mkdir()
    rows = raw_dir / "study_rows.json"
    rows.write_text(json.dumps({
        "study_manifest_id": frozen.study_manifest_id,
        "raw_execution_semantics_revision": cli.RAW_EXECUTION_SEMANTICS_REVISION,
        "authority_granted": False,
        "proved_commute": False,
        "tables": {},
    }), encoding="utf-8")
    (raw_dir / "complete.json").write_text(json.dumps({
        "schema_version": cli._RAW_COMPLETION_SCHEMA,
        "raw_execution_semantics_revision": cli.RAW_EXECUTION_SEMANTICS_REVISION,
        "study_manifest_id": frozen.study_manifest_id,
        "authority_granted": False,
        "proved_commute": False,
        "files_sha256": {"study_rows.json": _digest(rows.read_bytes())},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="required tables"):
        cli._raw_rows_from_complete(frozen)


def _complete_raw_coverage_tables(frozen: cli.FrozenPhase) -> dict[str, list[dict[str, object]]]:
    """Build the exact raw cartesian coverage expected by the CLI gate."""

    tables: dict[str, list[dict[str, object]]] = {
        "single_pass_observations.csv": [],
        "pair_observations.csv": [],
        "advisor_2n_group_results.csv": [],
        "advisor_2n_directional_results.csv": [],
        "advisor_2n_pair_validation.csv": [],
    }
    for program_id in frozen.program_ids:
        for action_id in frozen.groups["Uall"]:
            tables["single_pass_observations.csv"].append(
                {
                    "program_id": program_id,
                    "group_id": "Uall",
                    "action_id": action_id,
                    "execution_status": "success",
                }
            )
        for action_a_id, action_b_id in combinations(sorted(frozen.groups["Uall"]), 2):
            tables["pair_observations.csv"].append(
                {
                    "program_id": program_id,
                    "group_id": "Uall",
                    "action_a_id": action_a_id,
                    "action_b_id": action_b_id,
                    "a_status": "success",
                    "b_status": "success",
                    "ab_status": "success",
                    "ba_status": "success",
                    "dynamic_result": "unknown",
                }
            )
        for group_id, action_ids in frozen.groups.items():
            tables["advisor_2n_group_results.csv"].append(
                {"program_id": program_id, "group_id": group_id}
            )
            for action_id in action_ids:
                tables["advisor_2n_directional_results.csv"].append(
                    {"program_id": program_id, "group_id": group_id, "action_id": action_id}
                )
            for action_a_id, action_b_id in combinations(sorted(action_ids), 2):
                tables["advisor_2n_pair_validation.csv"].append(
                    {
                        "program_id": program_id,
                        "group_id": group_id,
                        "action_a_id": action_a_id,
                        "action_b_id": action_b_id,
                    }
                )
    return tables


def _complete_raw_coverage_tables_with_terminal_pair(
    frozen: cli.FrozenPhase, *, endpoint_status: str = "error"
) -> dict[str, list[dict[str, object]]]:
    """Keep a terminal first-round action in the full Uall pair matrix."""

    tables = _complete_raw_coverage_tables(frozen)
    terminal_action = frozen.groups["Uall"][0]
    for row in tables["single_pass_observations.csv"]:
        if str(row["action_id"]) == terminal_action:
            row["execution_status"] = endpoint_status
    for row in tables["pair_observations.csv"]:
        a_terminal = str(row["action_a_id"]) == terminal_action
        b_terminal = str(row["action_b_id"]) == terminal_action
        row["a_status"] = endpoint_status if a_terminal else "success"
        row["b_status"] = endpoint_status if b_terminal else "success"
        if a_terminal or b_terminal:
            row["ab_status"] = "not_run"
            row["ba_status"] = "not_run"
            row["dynamic_result"] = "timeout" if endpoint_status == "timeout" else "failed"
    return tables


def test_raw_coverage_accepts_full_uall_pair_matrix_with_terminal_rows(tmp_path: Path) -> None:
    frozen = cli.load_frozen_phase(_write_frozen_manifest(tmp_path), phase="run")
    tables = _complete_raw_coverage_tables_with_terminal_pair(frozen)

    cli._validate_raw_coverage(tables, frozen)


@pytest.mark.parametrize("mutation", ("missing", "replaced", "malformed"))
def test_raw_coverage_rejects_terminal_pair_loss_replacement_or_malformed_state(
    tmp_path: Path, mutation: str
) -> None:
    frozen = cli.load_frozen_phase(_write_frozen_manifest(tmp_path), phase="run")
    tables = _complete_raw_coverage_tables_with_terminal_pair(frozen, endpoint_status="timeout")
    terminal_action = frozen.groups["Uall"][0]
    terminal_rows = [
        row for row in tables["pair_observations.csv"]
        if terminal_action in {str(row["action_a_id"]), str(row["action_b_id"])}
    ]
    assert terminal_rows
    target = terminal_rows[0]
    if mutation == "missing":
        tables["pair_observations.csv"].remove(target)
        match = "Uall pair evidence exact set mismatch"
    elif mutation == "replaced":
        replacement = next(
            row for row in tables["pair_observations.csv"]
            if terminal_action not in {str(row["action_a_id"]), str(row["action_b_id"])}
        )
        target["action_a_id"] = replacement["action_a_id"]
        target["action_b_id"] = replacement["action_b_id"]
        match = "Uall pair evidence exact set mismatch"
    else:
        target["ab_status"] = "success"
        match = "terminal AB/BA status mismatch"

    with pytest.raises(ValueError, match=match):
        cli._validate_raw_coverage(tables, frozen)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("ab_status", "not_run", "successful AB/BA status mismatch"),
        ("ba_status", "not_run", "successful AB/BA status mismatch"),
        ("dynamic_result", "failed", "successful pair dynamic result mismatch"),
    ),
)
def test_raw_coverage_rejects_unattempted_or_inconsistent_successful_pair(
    tmp_path: Path, field: str, value: str, match: str
) -> None:
    frozen = cli.load_frozen_phase(_write_frozen_manifest(tmp_path), phase="run")
    tables = _complete_raw_coverage_tables(frozen)
    tables["pair_observations.csv"][0][field] = value

    with pytest.raises(ValueError, match=match):
        cli._validate_raw_coverage(tables, frozen)


def test_raw_coverage_accepts_invalid_attempted_successful_pair(tmp_path: Path) -> None:
    frozen = cli.load_frozen_phase(_write_frozen_manifest(tmp_path), phase="run")
    tables = _complete_raw_coverage_tables(frozen)
    row = tables["pair_observations.csv"][0]
    row["ab_status"] = "invalid"
    row["dynamic_result"] = "failed"

    cli._validate_raw_coverage(tables, frozen)


def test_raw_coverage_rejects_invalid_attempted_successful_pair_with_wrong_dynamic_result(
    tmp_path: Path,
) -> None:
    frozen = cli.load_frozen_phase(_write_frozen_manifest(tmp_path), phase="run")
    tables = _complete_raw_coverage_tables(frozen)
    row = tables["pair_observations.csv"][0]
    row["ab_status"] = "invalid"
    row["dynamic_result"] = "unknown"

    with pytest.raises(ValueError, match="successful pair dynamic result mismatch"):
        cli._validate_raw_coverage(tables, frozen)


@pytest.mark.parametrize(
    ("table_name", "group_id", "mutation", "match"),
    [
        ("pair_observations.csv", "Uall", "self", "Uall pair evidence exact set mismatch"),
        ("pair_observations.csv", "Uall", "reversed", "Uall pair evidence exact set mismatch"),
        ("pair_observations.csv", "Uall", "duplicate", "Uall pair evidence exact set mismatch"),
        ("advisor_2n_pair_validation.csv", "U14", "self", "2N pair evidence exact set mismatch"),
        ("advisor_2n_pair_validation.csv", "U14", "reversed", "2N pair evidence exact set mismatch"),
        ("advisor_2n_pair_validation.csv", "U14", "duplicate", "2N pair evidence exact set mismatch"),
    ],
)
def test_raw_coverage_rejects_noncanonical_pair_substitution_that_hides_a_missing_pair(
    tmp_path: Path, table_name: str, group_id: str, mutation: str, match: str
) -> None:
    """Self, reversed, and duplicate pairs cannot hide a removed pair."""

    frozen = cli.load_frozen_phase(_write_frozen_manifest(tmp_path), phase="run")
    tables = _complete_raw_coverage_tables(frozen)
    scoped = [
        row
        for row in tables[table_name]
        if str(row["group_id"]) == group_id and str(row["program_id"]) == frozen.program_ids[0]
    ]
    target = scoped[0]
    if mutation == "self":
        target["action_b_id"] = target["action_a_id"]
    elif mutation == "reversed":
        target["action_a_id"], target["action_b_id"] = target["action_b_id"], target["action_a_id"]
    else:
        target["action_a_id"] = scoped[1]["action_a_id"]
        target["action_b_id"] = scoped[1]["action_b_id"]

    with pytest.raises(ValueError, match=match):
        cli._validate_raw_coverage(tables, frozen)


def test_raw_complete_rejects_rehashed_pair_deletion(tmp_path: Path) -> None:
    """Rehashing a tampered raw file cannot bypass exact pair coverage."""

    frozen = cli.load_frozen_phase(_write_frozen_manifest(tmp_path), phase="run")
    tables = _complete_raw_coverage_tables(frozen)
    for table_name, rows in tables.items():
        for index, row in enumerate(rows):
            row.update(
                {
                    "row_id": f"{table_name}:{index}",
                    "study_manifest_id": frozen.study_manifest_id,
                    "authority_granted": "false",
                    "proved_commute": "false",
                }
            )
    cli._write_raw_completion(frozen.out_dir, tables, frozen.study_manifest_id)
    assert cli._raw_rows_from_complete(frozen)["pair_observations.csv"]

    tables["pair_observations.csv"].pop()
    cli._write_raw_completion(frozen.out_dir, tables, frozen.study_manifest_id)
    with pytest.raises(ValueError, match="Uall pair evidence exact set mismatch"):
        cli._raw_rows_from_complete(frozen)


def test_raw_reader_applies_cleanup_defaults_only_to_legacy_handoff_without_mutating_hashes(
    tmp_path: Path,
) -> None:
    frozen = cli.load_frozen_phase(_write_frozen_manifest(tmp_path), phase="run")
    tables = _complete_raw_coverage_tables(frozen)
    for table_name, rows in tables.items():
        for index, row in enumerate(rows):
            row.update({
                "row_id": f"{table_name}:{index}",
                "study_manifest_id": frozen.study_manifest_id,
                "authority_granted": "false",
                "proved_commute": "false",
            })
    cli._write_raw_completion(frozen.out_dir, tables, frozen.study_manifest_id)
    raw_dir = frozen.out_dir / "raw"
    rows_path = raw_dir / "study_rows.json"
    before = rows_path.read_bytes()

    loaded = cli._raw_rows_from_complete(frozen)

    assert loaded["advisor_2n_directional_results.csv"][0]["second_output_sha256"] == ""
    assert rows_path.read_bytes() == before


def test_raw_reader_rejects_new_cleanup_handoff_missing_persisted_cleanup_field(
    tmp_path: Path,
) -> None:
    frozen = cli.load_frozen_phase(_write_frozen_manifest(tmp_path), phase="run")
    tables = _complete_raw_coverage_tables(frozen)
    tables["advisor_2n_directional_results.csv"][0]["second_output_sha256"] = "a" * 64
    for table_name, rows in tables.items():
        for index, row in enumerate(rows):
            row.update({
                "row_id": f"{table_name}:{index}",
                "study_manifest_id": frozen.study_manifest_id,
                "authority_granted": "false",
                "proved_commute": "false",
            })
    for row in tables["pair_observations.csv"]:
        row.update({
            "ab_output_path": "", "ab_output_sha256": "",
            "ba_output_path": "", "ba_output_sha256": "",
            "ab_verifier_status": "not_run", "ba_verifier_status": "not_run",
            "artifact_materialized": "false", "cleanup_status": "retained_unmaterialized",
        })
    for row in tables["advisor_2n_directional_results.csv"]:
        row.setdefault("second_output_sha256", "")
        row.update({
            "merged_input_path": "", "second_output_path": "",
            "second_output_materialized": "false", "cleanup_status": "retained_unmaterialized",
        })
    cli._write_raw_completion(
        frozen.out_dir,
        tables,
        frozen.study_manifest_id,
        cleanup_ledger=_retained_cleanup_ledger(frozen, tables),
    )
    # New cleanup hand-offs are immutable/versioned; tamper only with the
    # selected test hand-off, then rehash it to exercise semantic validation.
    rows_path = cli._raw_handoff_dir(frozen.out_dir / "raw", frozen) / "study_rows.json"
    raw = json.loads(rows_path.read_text(encoding="utf-8"))
    raw["tables"]["advisor_2n_directional_results.csv"][0].pop("second_output_sha256")
    rows_path.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
    complete_path = rows_path.parent / "complete.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete["files_sha256"]["study_rows.json"] = _digest(rows_path.read_bytes())
    complete_path.write_text(json.dumps(complete, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="new cleanup raw evidence missing"):
        cli._raw_rows_from_complete(frozen)


@pytest.mark.parametrize("failed_name", ("cleanup_ledger.json", "complete.json"))
def test_cleanup_publication_failure_never_calls_deletion_and_preserves_old_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_name: str,
) -> None:
    """Ledger/complete write failures happen before the cleanup callback."""

    out_dir = tmp_path / "experiment" / "output" / "smoke"
    cli._write_raw_completion(out_dir, {}, "old-manifest")
    raw_dir = out_dir / "raw"
    old_complete = (raw_dir / "complete.json").read_bytes()
    artifact = tmp_path / "dynamic.ll"
    artifact.write_text("must remain\n", encoding="utf-8")
    called = False
    original = cli._write_handoff_file

    def fail_selected_write(path: Path, content: str) -> None:
        if path.name == failed_name:
            raise OSError(f"injected {failed_name} publish failure")
        original(path, content)

    def destructive_cleanup() -> str:
        nonlocal called
        called = True
        artifact.unlink()
        return "unreachable"

    monkeypatch.setattr(cli, "_write_handoff_file", fail_selected_write)
    with pytest.raises(OSError, match="injected"):
        cli._execute_after_planned_cleanup_handoff(
            out_dir=out_dir,
            rows={
                "single_pass_observations.csv": [],
                "pair_observations.csv": [],
                "advisor_2n_group_results.csv": [],
                "advisor_2n_directional_results.csv": [],
                "advisor_2n_pair_validation.csv": [],
            },
            manifest_id="new-manifest",
            planned_ledger={
                "cleanup_state": "planned",
                "entries": [],
                "summary": {
                    "reclaimed_file_count": 0, "reclaimed_bytes": 0,
                    "retained_file_count": 0, "retained_bytes": 0,
                    "planned_file_count": 0, "planned_bytes": 0,
                },
            },
            execute_cleanup=destructive_cleanup,
        )

    assert called is False
    assert artifact.is_file()
    assert (raw_dir / "complete.json").read_bytes() == old_complete
    assert not (raw_dir / cli._ACTIVE_RAW_HANDOFF_FILE).exists()


def test_versioned_raw_handoff_requires_active_pointer_and_ledger(tmp_path: Path) -> None:
    frozen = cli.load_frozen_phase(_write_frozen_manifest(tmp_path), phase="run")
    raw = frozen.out_dir / "raw"
    versioned = raw / "handoffs" / "candidate"
    versioned.mkdir(parents=True)
    with pytest.raises(ValueError, match="require an active pointer"):
        cli._raw_handoff_dir(raw, frozen)
    pointer = {
        "schema_version": cli._ACTIVE_RAW_HANDOFF_SCHEMA,
        "raw_execution_semantics_revision": cli.RAW_EXECUTION_SEMANTICS_REVISION,
        "study_manifest_id": frozen.study_manifest_id,
        "handoff_dir": "handoffs/candidate",
        "authority_granted": False,
        "proved_commute": False,
    }
    (raw / cli._ACTIVE_RAW_HANDOFF_FILE).write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(ValueError, match="missing cleanup ledger"):
        cli._raw_handoff_dir(raw, frozen)


def _retained_cleanup_ledger(
    frozen: cli.FrozenPhase, tables: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    """Build a complete no-materialization ledger for reader tamper tests."""

    entries: list[dict[str, object]] = []
    for kind, table_name, artifacts, materialized_field in (
        (
            "pair_ab_ba", "pair_observations.csv",
            (("AB", "ab_output_path", "ab_output_sha256"), ("BA", "ba_output_path", "ba_output_sha256")),
            "artifact_materialized",
        ),
        (
            "two_n_second_round", "advisor_2n_directional_results.csv",
            (("second_round", "second_output_path", "second_output_sha256"),),
            "second_output_materialized",
        ),
    ):
        for row in tables[table_name]:
            identity = {
                "study_manifest_id": frozen.study_manifest_id,
                "artifact_kind": kind,
                "source_row_id": str(row["row_id"]),
                "group_id": str(row.get("group_id", "")),
                "program_id": str(row.get("program_id", "")),
                "action_id": str(row.get("action_id", "")),
            }
            cleanup_id = _digest(json.dumps(identity, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
            files = [
                {
                    "name": name,
                    "original_path": str(row[path_field]),
                    "actual_path": str(row[path_field]),
                    "quarantine_path": "",
                    "sha256": str(row[sha_field]),
                    "size_bytes": 0,
                    "reclaimed": False,
                }
                for name, path_field, sha_field in artifacts
            ]
            entries.append({
                **identity,
                "cleanup_id": cleanup_id,
                "cleanup_status": "retained",
                "row_cleanup_status": str(row["cleanup_status"]),
                "retention_reason": "retained_unmaterialized",
                "artifact_materialized": str(row.get(materialized_field, "")),
                "file_count": len(files),
                "size_bytes": 0,
                "reclaimed_file_count": 0,
                "reclaimed_bytes": 0,
                "retained_file_count": len(files),
                "retained_bytes": 0,
                "planned_file_count": 0,
                "planned_bytes": 0,
                "artifacts": files,
            })
    entries.sort(key=lambda entry: str(entry["cleanup_id"]))
    return {
        "schema_version": "advisor-pair-scale-cleanup-v2",
        "cleanup_state": "complete",
        "study_manifest_id": frozen.study_manifest_id,
        "authority_granted": False,
        "proved_commute": False,
        "protected_pair_row_ids": [],
        "protected_directionals": [],
        "entries": entries,
        "summary": {
            "reclaimed_file_count": 0,
            "reclaimed_bytes": 0,
            "retained_file_count": sum(int(entry["retained_file_count"]) for entry in entries),
            "retained_bytes": 0,
            "planned_file_count": 0,
            "planned_bytes": 0,
        },
    }


@pytest.mark.parametrize("mutation", ("summary", "duplicate_id", "actual_path"))
def test_new_cleanup_reader_rejects_rehashed_ledger_binding_or_summary_tampering(
    tmp_path: Path, mutation: str,
) -> None:
    frozen = cli.load_frozen_phase(_write_frozen_manifest(tmp_path), phase="run")
    tables = _complete_raw_coverage_tables(frozen)
    for table_name, rows in tables.items():
        for index, row in enumerate(rows):
            row.update({
                "row_id": f"{table_name}:{index}",
                "study_manifest_id": frozen.study_manifest_id,
                "authority_granted": "false",
                "proved_commute": "false",
            })
    for row in tables["pair_observations.csv"]:
        row.update({
            "ab_output_path": "", "ab_output_sha256": "",
            "ba_output_path": "", "ba_output_sha256": "",
            "ab_verifier_status": "not_run", "ba_verifier_status": "not_run",
            "artifact_materialized": "false", "cleanup_status": "retained_unmaterialized",
        })
    for row in tables["advisor_2n_directional_results.csv"]:
        row.update({
            "merged_input_path": "", "second_output_path": "", "second_output_sha256": "",
            "second_output_materialized": "false", "cleanup_status": "retained_unmaterialized",
        })
    ledger = _retained_cleanup_ledger(frozen, tables)
    cli._write_raw_completion(frozen.out_dir, tables, frozen.study_manifest_id, cleanup_ledger=ledger)
    active = cli._raw_handoff_dir(frozen.out_dir / "raw", frozen)
    ledger_path = active / "cleanup_ledger.json"
    altered = json.loads(ledger_path.read_text(encoding="utf-8"))
    if mutation == "summary":
        altered["summary"]["retained_file_count"] += 1
        match = "summary"
    elif mutation == "duplicate_id":
        altered["entries"][1]["cleanup_id"] = altered["entries"][0]["cleanup_id"]
        match = "unique"
    else:
        altered["entries"][0]["artifacts"][0]["actual_path"] = "unexpected.ll"
        match = "path"
    ledger_path.write_text(json.dumps(altered, sort_keys=True), encoding="utf-8")
    complete_path = active / "complete.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete["files_sha256"]["cleanup_ledger.json"] = _digest(ledger_path.read_bytes())
    complete_path.write_text(json.dumps(complete, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        cli._raw_rows_from_complete(frozen)


def test_prepare_rejects_existing_output_collision_before_execution(tmp_path: Path) -> None:
    out_dir = tmp_path / "experiment" / "output" / "smoke"
    out_dir.mkdir(parents=True)
    (out_dir / "foreign.txt").write_text("collision", encoding="utf-8")
    with pytest.raises(ValueError, match="hash-validated complete prepare state"):
        cli.validate_prepare_output(out_dir)


def test_cli_replay_dependency_factory_captures_two_complete_replays_per_family(
    tmp_path: Path,
) -> None:
    """Exercise the CLI injection seam without starting Worker, opt, or LLVM.

    Orchestration owns the mandated two repetitions.  The test proves that the
    CLI-provided callback set reaches that capture boundary for every replay
    family and that each record retains its required artifacts, hashes, stderr,
    and command identity.
    """

    artifact_names = ("S", "A", "B", "AB", "BA", "merged_input")
    calls: dict[str, list[int]] = {"worker": [], "external_opt": [], "two_n": []}

    def factory(defaults: dict[str, object]) -> dict[str, object]:
        assert set(defaults) == set(calls)

        def callback(family: str):
            def run(_case: dict[str, object], repetition: int, directory: Path) -> dict[str, object]:
                calls[family].append(repetition)
                directory.mkdir(parents=True, exist_ok=True)
                artifacts: dict[str, str] = {}
                hashes: dict[str, str] = {}
                names = (
                    (*artifact_names, "second_round_output")
                    if family == "two_n"
                    else artifact_names
                )
                for name in names:
                    path = directory / f"{name}.ll"
                    path.write_text(f"{family}:{repetition}:{name}\n", encoding="utf-8")
                    artifacts[name] = str(path)
                    hashes[name] = _digest(path.read_bytes())
                two_n_result = (
                    {
                        "merged_input_sha256": hashes["merged_input"],
                        "merged_input_hard_state_id": hashes["merged_input"],
                        "second_output_sha256": hashes["second_round_output"],
                        "second_output_hard_state_id": hashes[
                            "second_round_output"
                        ],
                    }
                    if family == "two_n"
                    else {}
                )
                return {
                    "status": "success",
                    "hard_state_hashes": hashes,
                    "artifact_sha256": hashes,
                    "artifacts": artifacts,
                    "stderr": f"{family} stderr {repetition}",
                    "command": ("synthetic-replay", family, str(repetition)),
                    "two_n_result": two_n_result,
                    "stage_results": _replay_stage_results(hashes, hashes),
                }

            return run

        return {family: callback(family) for family in calls}

    callbacks = cli._build_replay_dependencies(
        {family: (lambda _case, _repetition, _directory: {}) for family in calls},
        factory,
    )
    callback_keys = {"worker": "worker", "external_opt": "external_opt", "two_n": "two_n"}
    for family, callback_key in callback_keys.items():
        records = _run_replay_family(
            family, callbacks[callback_key], {"case_id": "synthetic"}, tmp_path / "capture"
        )
        assert len(records) == 2
        for repetition, record in enumerate(records, start=1):
            assert record["command"] == ["synthetic-replay", family, str(repetition)]
            assert record["stderr"] == f"{family} stderr {repetition}"
            expected_names = set(artifact_names)
            if family == "two_n":
                expected_names.add("second_round_output")
            assert set(record["artifacts"]) == expected_names
            assert set(record["hard_state_hashes"]) == expected_names
            assert set(record["artifact_sha256"]) == expected_names
            repeat_dir = tmp_path / "capture" / family / f"repeat-{repetition}"
            assert (repeat_dir / "record.json").is_file()
            assert (repeat_dir / "stderr.txt").is_file()
            assert (repeat_dir / "command.txt").is_file()
    assert calls == {family: [1, 2] for family in calls}


@pytest.mark.parametrize("family", ("worker", "external_opt"))
def test_pair_only_replay_record_does_not_require_or_materialize_merge_evidence(
    tmp_path: Path, family: str
) -> None:
    root = tmp_path / "root.ll"
    root.write_text(
        "define i32 @f() {\nentry:\n  ret i32 0\n}\n",
        encoding="utf-8",
        newline="\n",
    )

    def runner(parent: Path, action: object, output: Path) -> dict[str, object]:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            parent.read_text(encoding="utf-8") + f"; {action.name}\n",
            encoding="utf-8",
            newline="\n",
        )
        return {
            "success": True,
            "execution_status": "success",
            "verifier_status": "success",
            "output_path": str(output),
            "hard_state_id": cli._phasebatch_hard_state_id(output),
            "stderr": "",
            "command": ("synthetic-runner", action.name),
        }

    record = cli._build_pair_only_replay_record(
        root=root,
        left=SimpleNamespace(name="A"),
        right=SimpleNamespace(name="B"),
        directory=tmp_path / "replay",
        runner=runner,
        family=family,
        repetition=1,
    )
    normalized = _validate_replay_record(record, tmp_path / "replay")

    assert normalized["status"] == "success"
    assert normalized["merge_status"] == "not_applicable"
    assert normalized["merge_error_fingerprint"] == ""
    assert "merged_input" not in normalized["artifacts"]


def test_real_worker_helper_small_group_round_trip_and_failure(tmp_path: Path) -> None:
    """Exercise the real isolated Worker/helper path without a full matrix."""

    experiment = Path(__file__).resolve().parents[1]
    worker_path = Path(r"E:\PO2\worker\build\phasebatch-worker.exe")
    helper_path = experiment / "build" / "merge_helper" / "phasebatch-2n-merge.exe"
    if not worker_path.is_file() or not helper_path.is_file():
        pytest.skip("real isolated Worker/helper prerequisites are unavailable")
    root = experiment / "tests" / "fixtures" / "advisor_2n_merge" / "base.ll"
    root_copy = tmp_path / "root.ll"
    root_copy.write_bytes(root.read_bytes())
    actions = tuple(
        ActionRecord.for_function_candidate(name=name, pipeline=name, config_index=index)
        for index, name in enumerate(("instcombine", "simplifycfg"))
    )
    runner = cli._WorkerRunner(worker_path, timeout_s=10)
    try:
        bad = runner.apply(root_copy, ActionRecord.for_function_candidate(name="bad", pipeline="not-a-pass", config_index=9), tmp_path / "bad.ll")
        assert bad["success"] is False
        with DirectMergeClient((str(helper_path),), timeout_s=10) as helper:
            assert helper.ping()["protocol_version"] == 1
            profiles = profile_single_passes(
                root_ir=root_copy,
                actions=actions,
                out_dir=tmp_path / "profiles",
                run_single=runner.apply,
                verify_ir=None,
                study_manifest_id="manifest-real-worker-helper",
            )
            assert all(row["execution_status"] == "success" for row in profiles)
            outcome = evaluate_group_2n(
                root_ir=root_copy,
                group_id="U14",
                program_id="fixture",
                study_manifest_id="manifest-real-worker-helper",
                actions={action.action_id: action for action in actions},
                profiles=profiles,
                merge_client=helper,
                out_dir=tmp_path / "two_n",
                run_second=runner.apply,
                verify_ir=None,
                pair_observations=(),
            )
            assert len(outcome.directional_rows) == 2
            assert outcome.group_row["authority_granted"] == "false"
            assert outcome.group_row["proved_commute"] == "false"
    finally:
        runner.close()


def test_pair_replay_never_runs_a_second_stage_from_root_after_first_stage_failure(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root.ll"
    root.write_text("; root\n", encoding="utf-8")
    left = ActionRecord.for_function_candidate(
        name="left", pipeline="left", config_index=1
    )
    right = ActionRecord.for_function_candidate(
        name="right", pipeline="right", config_index=2
    )
    calls: list[tuple[str, str]] = []

    def runner(parent: Path, action: ActionRecord, output: Path) -> dict[str, object]:
        calls.append((parent.name, action.action_id))
        if action.action_id == left.action_id:
            return {
                "success": False,
                "execution_status": "error",
                "verifier_status": "not_run",
                "output_path": str(output),
                "stderr": "left failed",
                "command": ("runner", left.action_id),
            }
        output.write_text(f"; {action.action_id}\n", encoding="utf-8")
        return {
            "success": True,
            "execution_status": "success",
            "verifier_status": "success",
            "output_path": str(output),
            "hard_state_id": _digest(action.action_id),
            "stderr": "",
            "command": ("runner", action.action_id),
        }

    _paths, stages = cli._run_replay_pair_stages(
        root=root,
        left=left,
        right=right,
        directory=tmp_path / "replay",
        runner=runner,
    )

    assert stages["A"]["execution_status"] == "error"
    assert stages["AB"]["execution_status"] == "not_run"
    assert ("S.ll", right.action_id) in calls[:2]
    assert all(
        parent != "S.ll"
        for parent, action in calls[2:]
    )


def test_two_n_replay_missing_full_group_merge_discards_pair_only_merge(
    tmp_path: Path,
) -> None:
    pair_merge = tmp_path / "merged_input.ll"
    pair_merge.write_text("; pair-only merge\n", encoding="utf-8")
    digest = _digest(pair_merge.read_bytes())
    base: dict[str, object] = {
        "status": "success",
        "stderr": "",
        "merge_status": "complete",
        "merge_error_fingerprint": "",
        "artifacts": {"merged_input": str(pair_merge)},
        "artifact_sha256": {"merged_input": digest},
        "hard_state_hashes": {"merged_input": _digest("pair-only-hard")},
    }
    directional = {
        "merged_input_path": str(tmp_path / "missing-full-group-merge.ll"),
        "merged_input_sha256": _digest("full-group-merge"),
        "merged_input_hard_state_id": _digest("full-group-hard"),
        "second_output_path": str(tmp_path / "missing-second-output.ll"),
        "second_output_sha256": _digest("second-output"),
    }

    identities = cli._bind_two_n_replay_artifacts(base, directional, tmp_path)

    assert identities is None
    assert base["status"] == "error"
    assert "merged_input" not in base["artifacts"]
    assert "merged_input" not in base["artifact_sha256"]
    assert "merged_input" not in base["hard_state_hashes"]


def test_two_n_replay_binds_exact_full_group_merge_and_second_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    two_n_dir = tmp_path / "two_n" / "A"
    two_n_dir.mkdir(parents=True)
    merged = two_n_dir / "merged_input.ll"
    second = two_n_dir / "second_round.ll"
    merged.write_text("; full-group merge\n", encoding="utf-8")
    second.write_text("; second output\n", encoding="utf-8")
    merged_hard = _digest("full-group-hard")
    second_hard = _digest("second-hard")
    monkeypatch.setattr(
        cli,
        "_phasebatch_hard_state_id",
        lambda path: merged_hard if Path(path) == merged else second_hard,
    )
    pair_merge = tmp_path / "merged_input.ll"
    pair_merge.write_text("; pair-only merge\n", encoding="utf-8")
    base: dict[str, object] = {
        "status": "success",
        "stderr": "",
        "merge_status": "complete",
        "merge_error_fingerprint": "",
        "artifacts": {"merged_input": str(pair_merge)},
        "artifact_sha256": {"merged_input": _digest(pair_merge.read_bytes())},
        "hard_state_hashes": {"merged_input": _digest("pair-only-hard")},
    }
    directional = {
        "merged_input_path": str(merged),
        "merged_input_sha256": _digest(merged.read_bytes()),
        "merged_input_hard_state_id": merged_hard,
        "second_output_path": str(second),
        "second_output_sha256": _digest(second.read_bytes()),
    }

    identities = cli._bind_two_n_replay_artifacts(base, directional, tmp_path)

    assert identities == {
        "merged_input_sha256": _digest(merged.read_bytes()),
        "merged_input_hard_state_id": merged_hard,
        "second_output_sha256": _digest(second.read_bytes()),
        "second_output_hard_state_id": second_hard,
    }
    assert Path(base["artifacts"]["merged_input"]).read_bytes() == merged.read_bytes()
    assert (
        Path(base["artifacts"]["second_round_output"]).read_bytes()
        == second.read_bytes()
    )


def test_external_replay_stage_runs_independent_verifier_and_reports_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "S.ll"
    output = tmp_path / "A.ll"
    parent.write_text("; root\n", encoding="utf-8")
    action = SimpleNamespace(pipeline="instcombine")
    verifier_calls: list[Path] = []

    def fake_run(command: tuple[str, ...], *, timeout_s: float) -> SimpleNamespace:
        assert timeout_s == 5.0
        output.write_text("; invalid output\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="")

    def fake_verify(_opt: Path, _timeout_s: float, path: Path) -> bool:
        verifier_calls.append(path)
        return False

    monkeypatch.setattr(cli, "_run_process", fake_run)
    monkeypatch.setattr(cli, "_verify_with_opt", fake_verify)

    result = cli._external_replay_apply(
        Path("opt"), 5.0, parent, action, output
    )

    assert verifier_calls == [output]
    assert result["success"] is False
    assert result["execution_status"] == "invalid"
    assert result["verifier_status"] == "invalid"


def test_run_frozen_injects_replay_callbacks_twice_and_retains_complete_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI seam must exercise each supplied replay family exactly twice."""

    from advisor_study.orchestration import _run_replay_family
    import advisor_study.direct_merge as direct_merge
    import advisor_study.orchestration as orchestration

    manifest = _write_frozen_manifest(tmp_path)
    frozen = cli.load_frozen_phase(manifest, phase="run")
    calls: dict[str, list[int]] = {name: [] for name in ("worker", "external_opt", "two_n")}
    captured: dict[str, list[dict[str, object]]] = {}
    created_workers: list[object] = []

    class FakeWorker:
        def __init__(self, _path: Path, *, timeout_s: float) -> None:
            self.timeout_s = timeout_s
            self.closed = 0
            created_workers.append(self)

        def apply(self, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("fake orchestration must not execute the Worker")

        def close(self) -> None:
            self.closed += 1

    class FakeMergeClient:
        def __init__(self, _command: tuple[str, ...], *, timeout_s: float) -> None:
            self.timeout_s = timeout_s

        def __enter__(self) -> "FakeMergeClient":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def ping(self) -> dict[str, int]:
            return {"protocol_version": 1}

    def replay_factory(defaults: object) -> dict[str, object]:
        assert isinstance(defaults, dict)
        assert set(defaults) == {"worker", "external_opt", "two_n"}
        assert all(callable(callback) for callback in defaults.values())

        def replay(family: str):
            def callback(_case: dict[str, object], repetition: int, directory: Path) -> dict[str, object]:
                calls[family].append(repetition)
                artifacts: dict[str, str] = {}
                for name in ("S", "A", "B", "AB", "BA", "merged_input"):
                    path = directory / f"{name}.ll"
                    path.write_text(f"; {family} replay {repetition} {name}\n", encoding="utf-8")
                    artifacts[name] = str(path)
                hard_state_hashes = {
                    name: _digest(f"{family}-state-{repetition}-{name}")
                    for name in artifacts
                }
                artifact_sha256 = {
                    name: _digest(Path(path).read_bytes())
                    for name, path in artifacts.items()
                }
                return {
                    "status": "success",
                    "hard_state_hashes": hard_state_hashes,
                    "artifact_sha256": artifact_sha256,
                    "artifacts": artifacts,
                    "stderr": f"{family} stderr {repetition}",
                    "command": ("injected-replay", family, str(repetition)),
                    "two_n_result": {},
                    "stage_results": _replay_stage_results(
                        hard_state_hashes, artifact_sha256
                    ),
                }

            return callback

        return {name: replay(name) for name in calls}

    def fake_orchestration(*, dependencies: object, out_dir: Path, **_kwargs: object) -> object:
        case = {
            "program_id": "p0",
            "group_id": "U14",
            "advisor_pair_row": {
                "action_a_id": frozen.groups["U14"][0],
                "action_b_id": frozen.groups["U14"][1],
            },
        }
        for family, callback in (
            ("worker", dependencies.replay_worker),  # type: ignore[attr-defined]
            ("external_opt", dependencies.replay_external_opt),  # type: ignore[attr-defined]
            ("two_n", dependencies.replay_two_n),  # type: ignore[attr-defined]
        ):
            captured[family] = _run_replay_family(family, callback, case, out_dir / "replay")
        empty_two_n = {group: {"group_rows": (), "directional_rows": (), "pair_rows": ()} for group in ("U14", "U30", "Uall")}
        return SimpleNamespace(
            profile_rows={},
            pair_views={"Uall": ()},
            two_n_results=empty_two_n,
            false_authorizations=(),
        )

    monkeypatch.setattr(cli, "_WorkerRunner", FakeWorker)
    monkeypatch.setattr(direct_merge, "DirectMergeClient", FakeMergeClient)
    monkeypatch.setattr(orchestration, "run_study_orchestration", fake_orchestration)
    # This seam deliberately returns no matrix rows; raw coverage is exercised
    # by its own complete-table tests, while this test isolates replay wiring.
    # The production checkpoint loop now rejects such an intentionally empty
    # result, so invoke the one-program callback through a narrow fake loop and
    # bypass only the unrelated checkpoint/handoff validation in this test.
    def fake_program_checkpoints(**kwargs: object) -> tuple[object, dict[str, object]]:
        run_program = kwargs["run_program"]
        programs = kwargs["programs"]
        assert callable(run_program) and isinstance(programs, dict)
        return run_program("p0", programs["p0"]), {}

    monkeypatch.setattr(cli, "_run_program_checkpoints", fake_program_checkpoints)
    monkeypatch.setattr(cli, "_raw_rows_from_compacted_result", lambda _result: {})
    monkeypatch.setattr(cli, "_cleanup_ledger_from_checkpoint_index", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(cli, "_publish_cleanup_handoff", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "_raw_rows_from_complete", lambda _frozen, **_kwargs: {})

    cli._run_frozen(frozen, replay_dependency_factory=replay_factory)

    assert len(created_workers) == frozen.jobs
    assert all(worker.closed == 1 for worker in created_workers)
    assert calls == {"worker": [1, 2], "external_opt": [1, 2], "two_n": [1, 2]}
    for family, records in captured.items():
        assert len(records) == 2
        for repetition, record in enumerate(records, start=1):
            assert record["stderr"] == f"{family} stderr {repetition}"
            assert record["command"] == ["injected-replay", family, str(repetition)]
            assert set(record["artifacts"]) == {"S", "A", "B", "AB", "BA", "merged_input"}
            for name, path in record["artifacts"].items():
                artifact = Path(path)
                assert artifact.is_file()
                assert record["artifact_sha256"][name] == _digest(artifact.read_bytes())
                assert len(record["hard_state_hashes"][name]) == 64
            record_path = frozen.out_dir / "replay" / family / f"repeat-{repetition}" / "record.json"
            persisted = json.loads(record_path.read_text(encoding="utf-8"))
            assert persisted["command"] == record["command"]
            assert persisted["stderr"] == record["stderr"]
            assert persisted["artifact_sha256"] == record["artifact_sha256"]
