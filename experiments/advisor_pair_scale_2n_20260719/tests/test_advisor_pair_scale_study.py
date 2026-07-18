from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

import advisor_study.orchestration as orchestration_module
import advisor_study.program_runtime as program_runtime_module
import advisor_study.study as study_module
from advisor_study.manifest import ProgramRecord
from advisor_study.pass_universe import load_u14_actions
from advisor_study.study import PrepareDependencies, RunResult, prepare_study
from advisor_study.orchestration import (
    OrchestrationDependencies,
    _normalize_two_n_result,
    run_study_orchestration,
)
from advisor_study.direct_merge import Advisor2NGroupResult


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
POLICY_PATH = EXPERIMENT_ROOT / "configs" / "advisor_pair_scale_pass_policy_v1.json"
LLVM_COMMIT = "aac212f0bc9acbc40a8a2e9638f4b7496c25d0b2"
TARGET = "x86_64-w64-windows-gnu"


def _sha256(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def test_normalize_typed_2n_result_with_mappingproxy_preserves_status_and_wall_time() -> None:
    typed = Advisor2NGroupResult(
        group_row=MappingProxyType({
            "group_id": "U14",
            "execution_status": "complete",
            "wall_time_ms": 17,
            "nested": MappingProxyType({"status": "success"}),
        }),
        directional_rows=(
            MappingProxyType({"action_id": "A", "directional_status": "authorized_all_others", "wall_time_ms": 3}),
        ),
        pair_rows=(
            MappingProxyType({"action_a_id": "A", "action_b_id": "B", "two_n_pair_status": "not_authorized", "wall_time_ms": 5}),
        ),
    )

    normalized = _normalize_two_n_result(typed)

    assert normalized["group_rows"] == [{
        "group_id": "U14",
        "execution_status": "complete",
        "wall_time_ms": 17,
        "nested": {"status": "success"},
    }]
    assert normalized["directional_rows"][0]["directional_status"] == "authorized_all_others"
    assert normalized["directional_rows"][0]["wall_time_ms"] == 3
    assert normalized["pair_rows"][0]["two_n_pair_status"] == "not_authorized"
    assert normalized["pair_rows"][0]["wall_time_ms"] == 5


def _program(
    root: Path,
    *,
    program_id: str,
    family: str,
    sequence: int,
    selection_class: str,
) -> ProgramRecord:
    source = root / "llvm" / "SingleSource" / family / f"{program_id}.c"
    source.parent.mkdir(parents=True, exist_ok=True)
    source_bytes = f"int main(void) {{ return {sequence}; }}\n".encode("utf-8")
    source.write_bytes(source_bytes)
    seeded_root = root / "seed_roots" / f"{program_id}.ll"
    seeded_root.parent.mkdir(parents=True, exist_ok=True)
    seeded_root.write_bytes(b"; root " + program_id.encode("utf-8") + b"\n")
    return ProgramRecord(
        program_id=program_id,
        source_path=str(source.resolve()),
        relative_path=f"SingleSource/{family}/{program_id}.c",
        program_family=f"SingleSource/{family}",
        source_sha256=_sha256(source_bytes),
        source_size_bytes=len(source_bytes),
        compile_command=("clang", "-S", program_id),
        compile_status="success",
        compile_stderr_sha256="",
        root_ir_path=str(seeded_root.resolve()),
        root_ir_sha256=_sha256(seeded_root.read_bytes()),
        root_hard_state_id=_sha256(seeded_root.read_bytes()),
        target=TARGET,
        data_layout="e-m:w-p:64:64",
        preflight_status="success",
        selection_class=selection_class,
        selection_order=sequence + 1 if selection_class == "fixed" else None,
    )


def _program_universe(root: Path) -> tuple[tuple[ProgramRecord, ...], tuple[ProgramRecord, ...]]:
    fixed_names = ("20021219-1", "crc8.be", "fannkuch", "ffbench", "queens")
    fixed = tuple(
        _program(
            root,
            program_id=fixed_names[index] if index < len(fixed_names) else f"fixed-{index:02d}",
            family=f"Fixed{index // 5}",
            sequence=index,
            selection_class="fixed",
        )
        for index in range(50)
    )
    candidates = tuple(
        _program(
            root,
            program_id=f"candidate-{index:02d}",
            family=f"Extension{index // 5}",
            sequence=100 + index,
            selection_class="candidate",
        )
        for index in range(50)
    )
    return fixed, candidates


def _function_registry(policy: dict[str, object]) -> str:
    candidates = policy["candidate_pipelines"]
    assert isinstance(candidates, list)
    return "Function passes:\n" + "\n".join(f"  {name}" for name in candidates) + "\n"


@pytest.fixture(autouse=True)
def _isolated_output_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(study_module, "EXPERIMENT_ROOT", tmp_path / "experiment", raising=False)


def _output_dir() -> Path:
    return study_module.EXPERIMENT_ROOT / "output" / "formal" / "run"


def _write_reference_manifest(path: Path, fixed_programs: tuple[ProgramRecord, ...]) -> None:
    actions = load_u14_actions(REPO_ROOT / "configs" / "core_passes_v1.yaml")
    path.write_text(
        json.dumps(
            {
                "pass_config": {"actions": [action.as_manifest_record() for action in actions]},
                "program_manifest": [program.as_manifest_record() for program in fixed_programs],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _successful_prepare_dependencies(
    root: Path,
    *,
    allowed_extra_actions: int | None = None,
    llvm_commit: str = LLVM_COMMIT,
    verifier_result: bool = True,
    preflight_verifier_result: bool | None = None,
    unstable_action: str | None = None,
    tool_hash_mismatch: str | None = None,
) -> PrepareDependencies:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    fixed, _candidates = _program_universe(root)
    tools_dir = root / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    tool_records: dict[str, dict[str, object]] = {}
    for name in ("opt", "clang", "worker", "merge_helper"):
        tool = tools_dir / f"{name}.exe"
        content = f"tool:{name}".encode("utf-8")
        tool.write_bytes(content)
        tool_records[name] = {"path": str(tool), "sha256": _sha256(content)}
    if tool_hash_mismatch is not None:
        tool_records[tool_hash_mismatch]["sha256"] = "0" * 64

    core_names = {action.name for action in load_u14_actions(REPO_ROOT / "configs" / "core_passes_v1.yaml")}
    extras = [
        str(name)
        for name in policy["candidate_pipelines"]
        if str(name) not in core_names
    ]
    permitted_extras = set(extras if allowed_extra_actions is None else extras[:allowed_extra_actions])

    def compile_source(record: ProgramRecord, output: Path) -> RunResult:
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(record.root_ir_path, output)
        digest = _sha256(output.read_bytes())
        return RunResult(success=True, output_path=output, hard_state_id=digest)

    def run_single(root_ir: Path, action: object, output: Path) -> RunResult:
        name = str(getattr(action, "name"))
        if name not in core_names and name not in permitted_extras:
            return RunResult(success=False, output_path=output, stderr="excluded-for-test")
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = root_ir.read_bytes() + b"\n; action=" + name.encode("utf-8")
        if unstable_action == name and output.name == "repeat-2.ll":
            payload += b"-different"
        output.write_bytes(payload)
        return RunResult(
            success=True,
            output_path=output,
            hard_state_id=_sha256(payload),
        )

    def verify_ir(path: Path) -> bool:
        if path.parent.name == "roots":
            return verifier_result
        return verifier_result if preflight_verifier_result is None else preflight_verifier_result

    return PrepareDependencies(
        compile_source=compile_source,
        print_passes=lambda: _function_registry(policy),
        run_single=run_single,
        verify_ir=verify_ir,
        tool_records=tool_records,
        fixed_programs=fixed,
        candidate_programs=(),
        single_source_root=root / "llvm" / "SingleSource",
        llvm_commit=llvm_commit,
        target=TARGET,
        hard_state_policy={"schema": "hard-v1"},
        comparator={"schema": "comparator-v1"},
        artifact_policy={"retain_roots": True},
    )


def _prepare(
    tmp_path: Path,
    dependencies: PrepareDependencies,
    *,
    core_copy: bool = True,
    out_dir: Path | None = None,
) -> object:
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True, exist_ok=True)
    core = repo / "configs" / "core_passes_v1.yaml"
    if core_copy:
        shutil.copyfile(REPO_ROOT / "configs" / "core_passes_v1.yaml", core)
    reference = repo / "fixed.json"
    _write_reference_manifest(reference, tuple(dependencies.fixed_programs))
    selected_out_dir = _output_dir() if out_dir is None else out_dir
    relative = selected_out_dir.resolve().relative_to(
        study_module.EXPERIMENT_ROOT.resolve() / "output"
    )
    program_target = 3 if relative.parts[0] == "smoke" else 10
    selected_dependencies = (
        replace(dependencies, fixed_programs=tuple(dependencies.fixed_programs[:3]))
        if program_target == 3
        else dependencies
    )
    return prepare_study(
        repo_root=repo,
        out_dir=selected_out_dir,
        existing_50_manifest=reference,
        pass_policy=POLICY_PATH,
        dependencies=selected_dependencies,
        core_passes=core,
        program_target=program_target,
    )


def test_prepare_freezes_formal_inputs_before_pair_or_two_n_rows(tmp_path: Path) -> None:
    result = _prepare(tmp_path, _successful_prepare_dependencies(tmp_path))
    out_dir = _output_dir()

    assert result.program_count == 10
    assert result.group_sizes == {"U14": 14, "U30": 30, "Uall": 77}
    assert result.scale_gate == "eligible_pass_count_at_least_60"
    assert (out_dir / "study_manifest.json").is_file()
    assert (out_dir / "prepare_complete.json").is_file()
    assert (out_dir / "program_manifest.csv").is_file()
    exclusion_path = out_dir / "candidate_identity_exclusions.json"
    assert exclusion_path.is_file()
    exclusion_payload = json.loads(exclusion_path.read_text(encoding="utf-8"))
    document_sha256 = exclusion_payload.pop("document_sha256")
    assert document_sha256 == study_module.canonical_sha256(exclusion_payload)
    assert exclusion_payload["exclusions_sha256"] == study_module.canonical_sha256(
        exclusion_payload["exclusions"]
    )
    frozen_programs = json.loads((out_dir / "program_manifest.json").read_text(encoding="utf-8"))
    assert frozen_programs["target"] == 10
    assert len(frozen_programs["programs"]) == 10
    assert frozen_programs["reserve_order"] == []
    selected = sorted(
        frozen_programs["programs"], key=lambda row: row["selection_order"]
    )
    assert [row["program_id"] for row in selected] == [
        "fannkuch",
        "fixed-07",
        "fixed-12",
        "fixed-17",
        "fixed-22",
        "fixed-27",
        "fixed-32",
        "fixed-37",
        "fixed-42",
        "fixed-47",
    ]
    assert [row["selection_order"] for row in selected] == list(range(1, 11))
    assert len({row["program_family"] for row in selected}) == 10
    assert (out_dir / "pass_inventory.csv").is_file()
    assert (out_dir / "pass_preflight.csv").is_file()
    assert (out_dir / "pass_groups.csv").is_file()
    assert not (out_dir / "pair_observations.csv").exists()
    assert not (out_dir / "advisor_2n_group_results.csv").exists()
    manifest = json.loads((out_dir / "study_manifest.json").read_text(encoding="utf-8"))
    assert manifest["authority_granted"] is False
    assert manifest["proved_commute"] is False
    scope = manifest["artifact_policy"]["value"]
    assert scope["formal_program_count"] == 10
    assert "formal_scope_user_override" not in scope
    assert scope["fixed_program_count"] == 10
    assert scope["formal_source_inventory_count"] == 50
    assert scope["formal_selection_rule_id"] == "systematic_midpoint_fixed50_n10_v1"
    assert scope["formal_source_positions"] == [3, 8, 13, 18, 23, 28, 33, 38, 43, 48]
    assert scope["candidate_reserve_count"] == 0
    assert scope["candidate_inventory_count"] == 0
    assert scope["candidate_identity_exclusion_count"] == 0
    assert scope["candidate_identity_exclusions_sha256"] == _sha256(
        exclusion_path.read_bytes()
    )
    completion = json.loads((out_dir / "prepare_complete.json").read_text(encoding="utf-8"))
    assert completion["formal_source_inventory_count"] == 50
    assert completion["formal_selection_rule_id"] == "systematic_midpoint_fixed50_n10_v1"
    assert completion["formal_source_positions"] == [3, 8, 13, 18, 23, 28, 33, 38, 43, 48]


def test_prepare_persists_program_candidate_order_before_first_pass_result(
    tmp_path: Path,
) -> None:
    dependencies = _successful_prepare_dependencies(tmp_path)
    observed: list[bool] = []

    def run_single(root_ir: Path, action: object, output: Path) -> RunResult:
        staging_root = next(
            parent for parent in output.parents if ".prepare-staging-" in parent.name
        )
        observed.append((staging_root / "program_manifest.json").is_file())
        return dependencies.run_single(root_ir, action, output)

    _prepare(tmp_path, replace(dependencies, run_single=run_single))

    assert observed
    assert all(observed)


def test_prepare_formal_rejects_candidate_extensions_before_pair_or_2n_rows(
    tmp_path: Path,
) -> None:
    dependencies = _successful_prepare_dependencies(tmp_path)
    _fixed, candidates = _program_universe(tmp_path)
    compile_calls: list[str] = []

    def forbidden_compile(record: ProgramRecord, output: Path) -> RunResult:
        compile_calls.append(record.program_id)
        return dependencies.compile_source(record, output)

    with pytest.raises(ValueError, match="forbids candidate programs"):
        _prepare(
            tmp_path,
            replace(
                dependencies,
                candidate_programs=candidates[:1],
                compile_source=forbidden_compile,
            ),
        )

    assert compile_calls == []

    assert not (_output_dir() / "pair_observations.csv").exists()
    assert not (_output_dir() / "advisor_2n_group_results.csv").exists()


def test_prepare_does_not_pad_uall_below_sixty(tmp_path: Path) -> None:
    result = _prepare(
        tmp_path,
        _successful_prepare_dependencies(tmp_path, allowed_extra_actions=40),
    )
    assert result.group_sizes == {"U14": 14, "U30": 30, "Uall": 54}
    assert result.scale_gate == "eligible_pass_count_below_60"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"llvm_commit": "0" * 40}, "LLVM commit mismatch"),
        ({"tool_hash_mismatch": "worker"}, "worker hash mismatch"),
        ({"tool_hash_mismatch": "merge_helper"}, "merge_helper hash mismatch"),
    ],
)
def test_prepare_fails_closed_on_identity_mismatch(
    tmp_path: Path, kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _prepare(tmp_path, _successful_prepare_dependencies(tmp_path, **kwargs))


def test_prepare_fails_closed_when_compilation_does_not_materialize_root_ir(tmp_path: Path) -> None:
    dependencies = _successful_prepare_dependencies(tmp_path)

    def missing_root(record: ProgramRecord, output: Path) -> RunResult:
        return RunResult(success=True, output_path=output, hard_state_id="x")

    with pytest.raises(ValueError, match="missing root IR"):
        _prepare(tmp_path, replace(dependencies, compile_source=missing_root))


@pytest.mark.parametrize(
    ("dependencies", "match"),
    [
        (lambda root: _successful_prepare_dependencies(root, unstable_action="adce"), "unstable_hard_hash"),
        (lambda root: _successful_prepare_dependencies(root, preflight_verifier_result=False), "U14 preflight failed"),
    ],
)
def test_prepare_fails_closed_on_unstable_or_invalid_u14_preflight(
    tmp_path: Path, dependencies: object, match: str
) -> None:
    assert callable(dependencies)
    with pytest.raises(ValueError, match=match):
        _prepare(tmp_path, dependencies(tmp_path))


def test_prepare_fails_closed_on_u14_identity_drift(tmp_path: Path) -> None:
    dependencies = _successful_prepare_dependencies(tmp_path)
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True, exist_ok=True)
    core = repo / "configs" / "core_passes_v1.yaml"
    original = (REPO_ROOT / "configs" / "core_passes_v1.yaml").read_text(encoding="utf-8")
    core.write_text(original.replace("pipeline: mem2reg", "pipeline: sroa", 1), encoding="utf-8")
    reference = repo / "fixed.json"
    _write_reference_manifest(reference, tuple(dependencies.fixed_programs))
    with pytest.raises(ValueError, match="U14 action identity drift"):
        prepare_study(
            repo_root=repo,
            out_dir=_output_dir(),
            existing_50_manifest=reference,
            pass_policy=POLICY_PATH,
            dependencies=dependencies,
            core_passes=core,
        )


def test_prepare_rejects_outside_isolated_output_subtrees(tmp_path: Path) -> None:
    dependencies = _successful_prepare_dependencies(tmp_path)
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True, exist_ok=True)
    core = repo / "configs" / "core_passes_v1.yaml"
    shutil.copyfile(REPO_ROOT / "configs" / "core_passes_v1.yaml", core)
    reference = repo / "fixed.json"
    _write_reference_manifest(reference, tuple(dependencies.fixed_programs))

    with pytest.raises(ValueError, match=r"out_dir must be inside.*output/\(smoke\|formal\)"):
        prepare_study(
            repo_root=repo,
            out_dir=study_module.EXPERIMENT_ROOT / "docs" / "forbidden",
            existing_50_manifest=reference,
            pass_policy=POLICY_PATH,
            dependencies=dependencies,
            core_passes=core,
        )


@pytest.mark.parametrize("leaf", ["smoke", "formal"])
def test_prepare_allows_exact_isolated_output_roots(tmp_path: Path, leaf: str) -> None:
    out_dir = study_module.EXPERIMENT_ROOT / "output" / leaf
    result = _prepare(
        tmp_path,
        _successful_prepare_dependencies(tmp_path),
        out_dir=out_dir,
    )
    assert result.out_dir == out_dir.resolve()
    assert (out_dir / "prepare_complete.json").is_file()


def test_prepare_publishes_into_a_precreated_empty_smoke_output_root(tmp_path: Path) -> None:
    """The prescribed empty output/smoke skeleton is a safe first publish."""

    out_dir = study_module.EXPERIMENT_ROOT / "output" / "smoke"
    out_dir.mkdir(parents=True)

    result = _prepare(
        tmp_path,
        _successful_prepare_dependencies(tmp_path),
        out_dir=out_dir,
    )

    assert result.out_dir == out_dir.resolve()
    assert (out_dir / "prepare_complete.json").is_file()
    assert (out_dir / "study_manifest.json").is_file()


@pytest.mark.parametrize("bad_name", ["pair_observations.csv", "unknown-user-file.txt"])
def test_prepare_rejects_existing_output_without_overwriting_it(
    tmp_path: Path, bad_name: str
) -> None:
    dependencies = _successful_prepare_dependencies(tmp_path)
    out_dir = _output_dir()
    out_dir.mkdir(parents=True)
    preserved = out_dir / bad_name
    preserved.write_text("user-owned", encoding="utf-8")

    with pytest.raises(ValueError, match="existing out_dir"):
        _prepare(tmp_path, dependencies)

    assert preserved.read_text(encoding="utf-8") == "user-owned"


def test_prepare_rejects_existing_hash_invalid_prepare_state_without_overwrite(tmp_path: Path) -> None:
    dependencies = _successful_prepare_dependencies(tmp_path)
    result = _prepare(tmp_path, dependencies)
    inventory = result.out_dir / "pass_inventory.csv"
    original = inventory.read_text(encoding="utf-8")
    inventory.write_text(original + "tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="existing out_dir"):
        _prepare(tmp_path, dependencies)

    assert inventory.read_text(encoding="utf-8") == original + "tampered\n"


def test_prepare_reuses_byte_identical_hash_validated_output_without_rewriting(tmp_path: Path) -> None:
    dependencies = _successful_prepare_dependencies(tmp_path)
    first = _prepare(tmp_path, dependencies)
    completion = first.prepare_complete_path
    first_hash = _sha256(completion.read_bytes())
    first_mtime_ns = completion.stat().st_mtime_ns

    second = _prepare(tmp_path, dependencies)

    assert second == first
    assert _sha256(completion.read_bytes()) == first_hash
    assert completion.stat().st_mtime_ns == first_mtime_ns


def test_prepare_rejects_fixed_fifty_manifest_program_identity_drift(tmp_path: Path) -> None:
    dependencies = _successful_prepare_dependencies(tmp_path)
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True, exist_ok=True)
    core = repo / "configs" / "core_passes_v1.yaml"
    shutil.copyfile(REPO_ROOT / "configs" / "core_passes_v1.yaml", core)
    reference = repo / "fixed.json"
    _write_reference_manifest(reference, tuple(dependencies.fixed_programs))
    payload = json.loads(reference.read_text(encoding="utf-8"))
    payload["program_manifest"][0]["root_ir_sha256"] = "0" * 64
    reference.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="fixed 50 program identity mismatch"):
        prepare_study(
            repo_root=repo,
            out_dir=_output_dir(),
            existing_50_manifest=reference,
            pass_policy=POLICY_PATH,
            dependencies=dependencies,
            core_passes=core,
        )


class _OrchestrationAction:
    def __init__(self, action_id: str) -> None:
        self.action_id = action_id


def _orchestration_replay_record(
    directory: Path,
    tag: str,
    *,
    two_n_result: dict[str, str] | None = None,
    hard_state_tag: str = "stable",
    terminal_stage: str = "",
    terminal_fingerprint_tag: str = "",
    terminal_stages: dict[str, str] | None = None,
    stage_hard_state_ids: dict[str, str] | None = None,
) -> dict[str, object]:
    artifacts: dict[str, str] = {}
    for name in ("S", "A", "B", "AB", "BA", "merged_input"):
        artifact = directory / f"{name}.ll"
        artifact.write_text(f"; {tag} {name}\n", encoding="utf-8", newline="\n")
        artifacts[name] = str(artifact)
    hard_state_hashes = {
        name: _sha256(f"{hard_state_tag}-state-{name}") for name in artifacts
    }
    hard_state_hashes.update(stage_hard_state_ids or {})
    artifact_sha256 = {
        name: _sha256(Path(path).read_bytes()) for name, path in artifacts.items()
    }
    stage_results = {
        name: {
            "execution_status": "success",
            "verifier_status": "success",
            "hard_state_id": hard_state_hashes[name],
            "output_sha256": artifact_sha256[name],
            "command_sha256": _sha256(f"command-{name}"),
            "stderr_sha256": _sha256(""),
            "error_fingerprint": _sha256(f"success-{name}"),
        }
        for name in ("A", "B", "AB", "BA")
    }
    configured_terminal_stages = dict(terminal_stages or {})
    if terminal_stage:
        configured_terminal_stages[terminal_stage] = (
            terminal_fingerprint_tag or "stable terminal failure"
        )
    for name, fingerprint_tag in configured_terminal_stages.items():
        terminal_stderr_sha256 = _sha256(fingerprint_tag)
        stage_results[name] = {
            **stage_results[name],
            "execution_status": "error",
            "verifier_status": "not_run",
            "hard_state_id": "",
            "output_sha256": "",
            "stderr_sha256": terminal_stderr_sha256,
            "error_fingerprint": orchestration_module._terminal_error_fingerprint(
                "error", "not_run", terminal_stderr_sha256
            ),
        }
        artifacts.pop(name)
        artifact_sha256.pop(name)
        hard_state_hashes.pop(name)
    normalized_two_n = dict(two_n_result or {})
    if normalized_two_n:
        normalized_two_n.setdefault(
            "second_output_hard_state_id", _sha256("stable-second-output")
        )
        if normalized_two_n.get("directional_status"):
            first_effect = normalized_two_n["first_round_effect_sha256"]
            second_effect = normalized_two_n["second_round_effect_sha256"]
            merged = Path(artifacts["merged_input"])
            merged.write_text(
                f"; merged-input {first_effect}\n", encoding="utf-8", newline="\n"
            )
            artifact_sha256["merged_input"] = _sha256(merged.read_bytes())
            hard_state_hashes["merged_input"] = normalized_two_n[
                "merged_input_hard_state_id"
            ]
            second = directory / "second_round_output.ll"
            second.write_text(
                f"; second-output {second_effect}\n",
                encoding="utf-8",
                newline="\n",
            )
            artifacts["second_round_output"] = str(second)
            artifact_sha256["second_round_output"] = _sha256(second.read_bytes())
            hard_state_hashes["second_round_output"] = normalized_two_n[
                "second_output_hard_state_id"
            ]
    return {
        "status": "success",
        "hard_state_hashes": hard_state_hashes,
        "artifact_sha256": artifact_sha256,
        "artifacts": artifacts,
        "stderr": f"{tag} stderr",
        "command": ["replay", tag],
        "two_n_result": normalized_two_n,
        "stage_results": stage_results,
        "merge_status": "complete",
        "merge_error_fingerprint": "",
    }


def _orchestration_dependencies(calls: dict[str, list[object]]) -> OrchestrationDependencies:
    def profile(root: Path, actions: tuple[object, ...], directory: Path) -> list[dict[str, object]]:
        calls["profile"].append(tuple(action.action_id for action in actions))
        rows: list[dict[str, object]] = []
        for action in actions:
            output = directory / action.action_id / "first.ll"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(f"; first {action.action_id}\n", encoding="utf-8")
            rows.append(
                {
                    "action_id": action.action_id,
                    "execution_status": "success",
                    "verifier_status": "success",
                    "output_path": str(output),
                    "output_sha256": _sha256(output.read_bytes()),
                    "output_hard_state_id": _sha256(f"first-{action.action_id}"),
                    "command": ["worker", action.action_id],
                    "stderr": "",
                }
            )
        return rows

    def pairs(
        _root: Path,
        profiles: list[dict[str, object]],
        actions: dict[str, object],
        directory: Path,
    ) -> list[dict[str, object]]:
        calls["pair"].append(tuple(sorted(actions)))
        profile_by_action = {str(row["action_id"]): row for row in profiles}
        rows: list[dict[str, object]] = []
        ids = sorted(actions)
        for index, left in enumerate(ids):
            for right in ids[index + 1 :]:
                pair_dir = directory / f"{left}-{right}"
                pair_dir.mkdir(parents=True, exist_ok=True)
                (pair_dir / "AB.ll").write_text("; AB\n", encoding="utf-8")
                (pair_dir / "BA.ll").write_text("; BA\n", encoding="utf-8")
                rows.append(
                    {
                        "row_id": f"pair-{left}-{right}",
                        "study_manifest_id": "manifest-1",
                        "program_id": "program-1",
                        "group_id": "Uall",
                        "action_a_id": left,
                        "action_b_id": right,
                        "a_status": "success",
                        "a_hard_state_id": profile_by_action[left]["output_hard_state_id"],
                        "a_output_sha256": profile_by_action[left]["output_sha256"],
                        "a_verifier_status": "success",
                        "b_status": "success",
                        "b_hard_state_id": profile_by_action[right]["output_hard_state_id"],
                        "b_output_sha256": profile_by_action[right]["output_sha256"],
                        "b_verifier_status": "success",
                        "dynamic_result": "order_sensitive" if (left, right) == ("A", "B") else "commute",
                        "ab_status": "success",
                        "ab_verifier_status": "success",
                        "ba_status": "success",
                        "ba_verifier_status": "success",
                        "ab_hard_state_id": _sha256("stable-state-AB"),
                        "ab_output_sha256": _sha256((pair_dir / "AB.ll").read_bytes()),
                        "ab_stderr_sha256": _sha256(""),
                        "ba_hard_state_id": _sha256("stable-state-BA"),
                        "ba_output_sha256": _sha256((pair_dir / "BA.ll").read_bytes()),
                        "ba_stderr_sha256": _sha256(""),
                        "command": ["worker", left, right],
                        "command_sha256": _sha256(
                            "\0".join(("worker", left, right))
                        ),
                        "stderr": "",
                        "stderr_sha256": _sha256(""),
                    }
                )
        return rows

    def two_n(
        _root: Path,
        group_id: str,
        actions: dict[str, object],
        _profiles: list[dict[str, object]],
        _directory: Path,
        pair_rows: list[dict[str, object]],
    ) -> dict[str, object]:
        calls["two_n"].append((group_id, tuple(sorted(actions)), tuple(row["row_id"] for row in pair_rows)))
        pair_results = [
            {
                "row_id": f"advisor-{group_id}-{row['row_id']}",
                "group_id": group_id,
                "action_a_id": row["action_a_id"],
                "action_b_id": row["action_b_id"],
                "pair_observation_row_id": row["row_id"],
                "two_n_pair_status": "both_directions_authorized",
                "action_a_directional_status": "authorized_all_others",
                "action_b_directional_status": "authorized_all_others",
                "false_authorization": "true" if group_id == "U14" and row["row_id"] == "pair-A-B" else "false",
                "authority_granted": "false",
                "proved_commute": "false",
            }
            for row in pair_rows
        ]
        return {
            "group_row": {
                "study_manifest_id": "manifest-1",
                "program_id": "program-1",
                "group_id": group_id,
            },
            "directional_rows": [
                {
                    "study_manifest_id": "manifest-1",
                    "program_id": "program-1",
                    "group_id": group_id,
                    "action_id": action_id,
                    "merged_input_status": "not_run",
                    "directional_status": "authorized_all_others",
                    "first_round_effect_sha256": _sha256(
                        f"{group_id}-{action_id}-first-effect"
                    ),
                    "second_round_effect_sha256": _sha256(
                        f"{group_id}-{action_id}-second-effect"
                    ),
                    "merged_input_sha256": _sha256(
                        (
                            "; merged-input "
                            + _sha256(f"{group_id}-{action_id}-first-effect")
                            + "\n"
                        ).encode("utf-8")
                    ),
                    "merged_input_hard_state_id": _sha256(
                        "stable-state-merged_input"
                    ),
                    "second_output_sha256": _sha256(
                        (
                            "; second-output "
                            + _sha256(f"{group_id}-{action_id}-second-effect")
                            + "\n"
                        ).encode("utf-8")
                    ),
                }
                for action_id in sorted(actions)
            ],
            "pair_rows": [
                {
                    **row,
                    "study_manifest_id": "manifest-1",
                    "program_id": "program-1",
                    "dynamic_result": next(
                        source["dynamic_result"]
                        for source in pair_rows
                        if source["row_id"] == row["pair_observation_row_id"]
                    ),
                }
                for row in pair_results
            ],
        }

    def replay(kind: str):
        def runner(case: dict[str, object], repetition: int, directory: Path) -> dict[str, object]:
            calls[kind].append(repetition)
            expected_stages = case["expected_pair_stages"]
            return _orchestration_replay_record(
                directory,
                "stable",
                two_n_result=dict(case["expected_two_n"]) if kind == "replay_two_n" else {},
                stage_hard_state_ids={
                    name: str(stage["hard_state_id"])
                    for name, stage in expected_stages.items()
                    if str(stage.get("execution_status", "")) == "success"
                },
            )

        return runner

    return OrchestrationDependencies(
        profile_uall=profile,
        run_uall_pairs=pairs,
        run_group_two_n=two_n,
        replay_worker=replay("worker"),
        replay_external_opt=replay("opt"),
        replay_two_n=replay("replay_two_n"),
    )


def _run_orchestration(tmp_path: Path, calls: dict[str, list[object]]):
    root = tmp_path / "root.ll"
    root.write_text("; root\n", encoding="utf-8")
    actions = {name: _OrchestrationAction(name) for name in ("A", "B", "C")}
    return run_study_orchestration(
        out_dir=tmp_path / "output" / "smoke",
        isolation_root=tmp_path,
        study_manifest_id="manifest-1",
        programs={"program-1": root},
        groups={"U14": (actions["A"], actions["B"]), "U30": tuple(actions.values()), "Uall": tuple(actions.values())},
        dependencies=_orchestration_dependencies(calls),
    )


def test_orchestration_profiles_uall_once_derives_pair_views_and_runs_three_independent_two_n_groups(
    tmp_path: Path,
) -> None:
    calls: dict[str, list[object]] = {name: [] for name in ("profile", "pair", "two_n", "worker", "opt", "replay_two_n")}
    result = _run_orchestration(tmp_path, calls)

    assert calls["profile"] == [("A", "B", "C")]
    assert calls["pair"] == [("A", "B", "C")]
    assert [entry[:2] for entry in calls["two_n"]] == [("U14", ("A", "B")), ("U30", ("A", "B", "C")), ("Uall", ("A", "B", "C"))]
    assert [(row["action_a_id"], row["action_b_id"]) for row in result.pair_views["U14"]] == [("A", "B")]
    assert [(row["action_a_id"], row["action_b_id"]) for row in result.pair_views["U30"]] == [("A", "B"), ("A", "C"), ("B", "C")]
    assert all((result.out_dir / relative / "complete.json").is_file() for relative in result.stage_paths.values())
    assert len(result.false_authorizations) == 2
    assert {
        row["case"]["authorized_action_id"] for row in result.false_authorizations
    } == {"A", "B"}
    assert all(
        row["stable_false_authorization"] == "true"
        for row in result.false_authorizations
    )
    replayed_pair = next(
        row
        for row in result.two_n_results["U14"]["pair_rows"]
        if row["action_a_id"] == "A" and row["action_b_id"] == "B"
    )
    assert replayed_pair["stable_false_authorization"] == "true"
    assert replayed_pair["worker_replay_status"] == "stable"
    assert replayed_pair["external_opt_replay_status"] == "stable"
    assert replayed_pair["two_n_replay_status"] == "stable"
    assert replayed_pair["replay_artifact_id"]
    replay_sources = json.loads(replayed_pair["source_row_ids"])
    assert {witness["case_id"] for witness in result.false_authorizations}.issubset(
        replay_sources
    )
    worker_repeat = result.false_authorizations[0]["worker"][0]
    assert worker_repeat["hard_state_hashes"]["AB"] != worker_repeat["artifact_sha256"]["AB"]
    assert calls["worker"] == calls["opt"] == calls["replay_two_n"] == [1, 2, 1, 2]


def test_orchestration_reuses_hash_valid_complete_evidence_and_recomputes_invalid_stage(
    tmp_path: Path,
) -> None:
    calls: dict[str, list[object]] = {name: [] for name in ("profile", "pair", "two_n", "worker", "opt", "replay_two_n")}
    first = _run_orchestration(tmp_path, calls)
    profile_complete = first.out_dir / first.stage_paths["program-1:profiles"] / "complete.json"
    before = profile_complete.read_bytes()

    second = _run_orchestration(tmp_path, calls)
    assert second == first
    assert calls["profile"] == [("A", "B", "C")]
    assert profile_complete.read_bytes() == before

    profile_complete.write_text("{\"files\":{}}\n", encoding="utf-8")
    _run_orchestration(tmp_path, calls)
    assert calls["profile"] == [("A", "B", "C"), ("A", "B", "C")]


def test_orchestration_marks_unstable_false_authorization_nonstable_and_preserves_replay_evidence(
    tmp_path: Path,
) -> None:
    calls: dict[str, list[object]] = {name: [] for name in ("profile", "pair", "two_n", "worker", "opt", "replay_two_n")}
    dependencies = _orchestration_dependencies(calls)

    def unstable_worker(_case: dict[str, object], repetition: int, directory: Path) -> dict[str, object]:
        record = _orchestration_replay_record(
            directory,
            f"worker-{repetition}",
            hard_state_tag=f"worker-{repetition}",
        )
        calls["worker"].append(repetition)
        return record

    dependencies = replace(dependencies, replay_worker=unstable_worker)
    root = tmp_path / "root.ll"
    root.write_text("; root\n", encoding="utf-8")
    actions = {name: _OrchestrationAction(name) for name in ("A", "B", "C")}
    result = run_study_orchestration(
        out_dir=tmp_path / "output" / "smoke",
        isolation_root=tmp_path,
        study_manifest_id="manifest-1",
        programs={"program-1": root},
        groups={"U14": (actions["A"], actions["B"]), "U30": tuple(actions.values()), "Uall": tuple(actions.values())},
        dependencies=dependencies,
    )

    witness = result.false_authorizations[0]
    assert witness["stable_false_authorization"] == "false"
    assert witness["replay_status"] == "nondeterministic"
    replay_dir = result.out_dir / witness["replay_relative_path"]
    assert (replay_dir / "worker" / "repeat-1" / "S.ll").is_file()
    assert (replay_dir / "worker" / "repeat-2" / "stderr.txt").is_file()
    replayed_pair = next(
        row
        for row in result.two_n_results["U14"]["pair_rows"]
        if row["action_a_id"] == "A" and row["action_b_id"] == "B"
    )
    assert replayed_pair["stable_false_authorization"] == "false"
    assert replayed_pair["worker_replay_status"] == "nondeterministic"
    assert replayed_pair["external_opt_replay_status"] == "stable"
    assert replayed_pair["two_n_replay_status"] == "stable"


def test_orchestration_two_n_effect_or_second_output_drift_is_written_back_nondeterministic(
    tmp_path: Path,
) -> None:
    calls: dict[str, list[object]] = {
        name: []
        for name in ("profile", "pair", "two_n", "worker", "opt", "replay_two_n")
    }
    dependencies = _orchestration_dependencies(calls)

    def drifting_two_n(
        case: dict[str, object], repetition: int, directory: Path
    ) -> dict[str, object]:
        expected = dict(case["expected_two_n"])
        expected["second_round_effect_sha256"] = _sha256(f"effect-{repetition}")
        expected["second_output_sha256"] = _sha256(
            (
                "; second-output "
                + expected["second_round_effect_sha256"]
                + "\n"
            ).encode("utf-8")
        )
        expected["second_output_hard_state_id"] = _sha256(
            f"hard-state-{repetition}"
        )
        calls["replay_two_n"].append(repetition)
        return _orchestration_replay_record(
            directory,
            f"two-n-{repetition}",
            two_n_result=expected,
        )

    dependencies = replace(dependencies, replay_two_n=drifting_two_n)
    root = tmp_path / "root.ll"
    root.write_text("; root\n", encoding="utf-8")
    actions = {name: _OrchestrationAction(name) for name in ("A", "B", "C")}
    result = run_study_orchestration(
        out_dir=tmp_path / "output" / "smoke",
        isolation_root=tmp_path,
        study_manifest_id="manifest-1",
        programs={"program-1": root},
        groups={
            "U14": (actions["A"], actions["B"]),
            "U30": tuple(actions.values()),
            "Uall": tuple(actions.values()),
        },
        dependencies=dependencies,
    )

    replayed_pair = next(
        row
        for row in result.two_n_results["U14"]["pair_rows"]
        if row["action_a_id"] == "A" and row["action_b_id"] == "B"
    )
    assert replayed_pair["stable_false_authorization"] == "false"
    assert replayed_pair["two_n_replay_status"] == "nondeterministic"


def test_orchestration_accepts_exact_one_sided_terminal_pair_for_replay() -> None:
    profiles = [
        {
            "action_id": action_id,
            "execution_status": "success",
            "verifier_status": "success",
            "output_hard_state_id": _sha256(f"first-{action_id}"),
            "output_sha256": _sha256(f"first-bytes-{action_id}"),
        }
        for action_id in ("A", "B")
    ]
    row = {
        "row_id": "pair-A-B",
        "study_manifest_id": "manifest-1",
        "program_id": "program-1",
        "action_a_id": "A",
        "action_b_id": "B",
        "dynamic_result": "failed",
        "a_status": "success",
        "a_verifier_status": "success",
        "a_hard_state_id": profiles[0]["output_hard_state_id"],
        "a_output_sha256": profiles[0]["output_sha256"],
        "b_status": "success",
        "b_verifier_status": "success",
        "b_hard_state_id": profiles[1]["output_hard_state_id"],
        "b_output_sha256": profiles[1]["output_sha256"],
        "ab_status": "success",
        "ab_verifier_status": "success",
        "ab_hard_state_id": _sha256("AB"),
        "ab_output_sha256": _sha256("AB-bytes"),
        "ab_stderr_sha256": _sha256(""),
        "ba_status": "error",
        "ba_verifier_status": "not_run",
        "ba_hard_state_id": "",
        "ba_output_sha256": "",
        "ba_stderr_sha256": _sha256("terminal-error"),
        "command_sha256": _sha256("command"),
        "stderr_sha256": _sha256("terminal-error"),
    }

    orchestration_module._validate_original_pair(
        row,
        study_manifest_id="manifest-1",
        program_id="program-1",
        action_a_id="A",
        action_b_id="B",
        profiles=profiles,
    )


def test_orchestration_stably_replays_one_sided_terminal_false_authorization(
    tmp_path: Path,
) -> None:
    calls: dict[str, list[object]] = {
        name: []
        for name in ("profile", "pair", "two_n", "worker", "opt", "replay_two_n")
    }
    dependencies = _orchestration_dependencies(calls)
    original_pairs = dependencies.run_uall_pairs

    def terminal_pairs(*args: object) -> list[dict[str, object]]:
        rows = original_pairs(*args)  # type: ignore[arg-type]
        target = next(
            row
            for row in rows
            if row["action_a_id"] == "A" and row["action_b_id"] == "B"
        )
        target.update(
            {
                "dynamic_result": "failed",
                "ba_status": "error",
                "ba_verifier_status": "not_run",
                "ba_hard_state_id": "",
                "ba_output_sha256": "",
                "ba_stderr_sha256": _sha256("stable terminal failure"),
                "command_sha256": _sha256("pair-command"),
                "stderr_sha256": _sha256("stable terminal failure"),
            }
        )
        return rows

    def replay(kind: str):
        def runner(
            case: dict[str, object], repetition: int, directory: Path
        ) -> dict[str, object]:
            calls[kind].append(repetition)
            expected_stages = case["expected_pair_stages"]
            return _orchestration_replay_record(
                directory,
                "stable-terminal",
                terminal_stage="BA",
                two_n_result=(
                    dict(case["expected_two_n"])
                    if kind == "replay_two_n"
                    else {}
                ),
                stage_hard_state_ids={
                    name: str(stage["hard_state_id"])
                    for name, stage in expected_stages.items()
                    if str(stage.get("execution_status", "")) == "success"
                },
            )

        return runner

    dependencies = replace(
        dependencies,
        run_uall_pairs=terminal_pairs,
        replay_worker=replay("worker"),
        replay_external_opt=replay("opt"),
        replay_two_n=replay("replay_two_n"),
    )
    root = tmp_path / "root.ll"
    root.write_text("; root\n", encoding="utf-8")
    actions = {name: _OrchestrationAction(name) for name in ("A", "B", "C")}
    result = run_study_orchestration(
        out_dir=tmp_path / "output" / "smoke",
        isolation_root=tmp_path,
        study_manifest_id="manifest-1",
        programs={"program-1": root},
        groups={
            "U14": (actions["A"], actions["B"]),
            "U30": tuple(actions.values()),
            "Uall": tuple(actions.values()),
        },
        dependencies=dependencies,
    )

    pair = next(
        row
        for row in result.two_n_results["U14"]["pair_rows"]
        if row["action_a_id"] == "A" and row["action_b_id"] == "B"
    )
    assert pair["dynamic_result"] == "failed"
    assert pair["stable_false_authorization"] == "true"
    assert pair["worker_replay_status"] == "stable"
    assert all(
        "BA" not in witness["worker"][0]["artifacts"]
        for witness in result.false_authorizations
    )


def test_terminal_diagnostic_and_partial_output_drift_is_semantically_stable_and_retained(
    tmp_path: Path,
) -> None:
    def runner(
        _case: dict[str, object], repetition: int, directory: Path
    ) -> dict[str, object]:
        raw = _orchestration_replay_record(
            directory,
            "same-terminal-semantics",
            terminal_stage="BA",
            terminal_fingerprint_tag=f"terminal-diagnostic-{repetition}",
        )
        partial = directory / "BA.ll"
        partial.write_text(
            f"; partial terminal output {repetition}\n",
            encoding="utf-8",
            newline="\n",
        )
        partial_sha = _sha256(partial.read_bytes())
        raw["artifacts"]["BA"] = str(partial)  # type: ignore[index]
        raw["artifact_sha256"]["BA"] = partial_sha  # type: ignore[index]
        raw["stage_results"]["BA"]["output_sha256"] = partial_sha  # type: ignore[index]
        raw["stage_results"]["BA"]["command_sha256"] = _sha256(  # type: ignore[index]
            f"terminal-command-{repetition}"
        )
        raw["stderr"] = f"transport diagnostic {repetition}"
        raw["command"] = ["replay", f"terminal-repeat-{repetition}"]
        return raw

    replay_root = tmp_path / "replay"
    records = orchestration_module._run_replay_family(
        "worker", runner, {}, replay_root
    )

    case = {
        "expected_pair_stages": {
            "AB": {
                "execution_status": "success",
                "verifier_status": "success",
                "hard_state_id": records[0]["stage_results"]["AB"][
                    "hard_state_id"
                ],
            },
            "BA": {
                "execution_status": "error",
                "verifier_status": "not_run",
                "hard_state_id": "",
                "error_fingerprint": orchestration_module._terminal_error_fingerprint(
                    "error", "not_run", _sha256("original diagnostic")
                ),
            },
        }
    }

    assert (
        records[0]["stage_results"]["BA"]["stderr_sha256"]
        != records[1]["stage_results"]["BA"]["stderr_sha256"]
    )
    assert (
        records[0]["stage_results"]["BA"]["error_fingerprint"]
        != records[1]["stage_results"]["BA"]["error_fingerprint"]
    )
    assert (
        records[0]["stage_results"]["BA"]["output_sha256"]
        != records[1]["stage_results"]["BA"]["output_sha256"]
    )
    for repetition, record in enumerate(records, start=1):
        repeat_dir = replay_root / "worker" / f"repeat-{repetition}"
        assert (repeat_dir / "stderr.txt").read_text(encoding="utf-8") == record[
            "stderr"
        ]
        assert (repeat_dir / "command.txt").read_text(encoding="utf-8") == "\0".join(
            record["command"]
        )
    assert (
        orchestration_module._replay_family_status(
            records, case, require_two_n=False
        )
        == "stable"
    )
    assert program_runtime_module._checkpoint_replay_signature(
        records[0]
    ) == program_runtime_module._checkpoint_replay_signature(records[1])
    program_runtime_module._validate_checkpoint_replay_family(
        {"worker": records, "family_statuses": {"worker": "stable"}},
        family="worker",
    )


def test_terminal_replay_semantics_ignore_original_diagnostic_fingerprint(
    tmp_path: Path,
) -> None:
    records = []
    for repetition in (1, 2):
        directory = tmp_path / f"repeat-{repetition}"
        directory.mkdir()
        records.append(
            orchestration_module._validate_replay_record(
                _orchestration_replay_record(
                    directory,
                    "same-terminal-replay",
                    terminal_stage="BA",
                    terminal_fingerprint_tag="same-but-wrong-terminal",
                ),
                directory,
            )
        )
    case = {
        "expected_pair_stages": {
            "AB": {
                "execution_status": "success",
                "verifier_status": "success",
                "hard_state_id": records[0]["stage_results"]["AB"][
                    "hard_state_id"
                ],
            },
            "BA": {
                "execution_status": "error",
                "verifier_status": "not_run",
                "hard_state_id": "",
                "error_fingerprint": orchestration_module._terminal_error_fingerprint(
                    "error", "not_run", _sha256("original-terminal")
                ),
            },
        }
    }

    assert (
        orchestration_module._replay_family_status(
            records, case, require_two_n=False
        )
        == "stable"
    )


def test_two_n_artifact_byte_drift_is_stable_but_effect_drift_is_not(
    tmp_path: Path,
) -> None:
    semantic_two_n = {
        "two_n_pair_status": "authorized",
        "action_a_directional_status": "authorized_all_others",
        "action_b_directional_status": "unavailable",
        "directional_status": "authorized_all_others",
        "first_round_effect_sha256": _sha256("first-effect"),
        "second_round_effect_sha256": _sha256("second-effect"),
        "merged_input_hard_state_id": _sha256("merged-hard-state"),
        "second_output_hard_state_id": _sha256("second-hard-state"),
    }

    def runner(
        _case: dict[str, object], repetition: int, directory: Path
    ) -> dict[str, object]:
        raw = _orchestration_replay_record(
            directory,
            "same-two-n-stage-bytes",
            two_n_result={
                **semantic_two_n,
                "merged_input_sha256": _sha256("placeholder-merged"),
                "second_output_sha256": _sha256("placeholder-second"),
            },
        )
        for artifact_name, result_field in (
            ("merged_input", "merged_input_sha256"),
            ("second_round_output", "second_output_sha256"),
        ):
            artifact = Path(raw["artifacts"][artifact_name])  # type: ignore[index]
            artifact.write_text(
                f"; semantically ignored artifact bytes {artifact_name} {repetition}\n",
                encoding="utf-8",
                newline="\n",
            )
            digest = _sha256(artifact.read_bytes())
            raw["artifact_sha256"][artifact_name] = digest  # type: ignore[index]
            raw["two_n_result"][result_field] = digest  # type: ignore[index]
        return raw

    records = orchestration_module._run_replay_family(
        "two_n", runner, {}, tmp_path / "two-n-replay"
    )
    expected = {
        **semantic_two_n,
        "merged_input_sha256": _sha256("original-merged-bytes"),
        "second_output_sha256": _sha256("original-second-bytes"),
    }
    case = {"expected_pair_stages": {}, "expected_two_n": expected}

    assert records[0]["two_n_result"]["merged_input_sha256"] != records[1][
        "two_n_result"
    ]["merged_input_sha256"]
    assert records[0]["two_n_result"]["second_output_sha256"] != records[1][
        "two_n_result"
    ]["second_output_sha256"]
    assert (
        orchestration_module._replay_family_status(
            records, case, require_two_n=True, family="two_n"
        )
        == "stable"
    )
    effect_drift = json.loads(json.dumps(records))
    effect_drift[1]["two_n_result"]["second_round_effect_sha256"] = _sha256(
        "different-second-effect"
    )
    assert (
        orchestration_module._replay_family_status(
            effect_drift, case, require_two_n=True, family="two_n"
        )
        == "nondeterministic"
    )


def test_replay_record_rejects_stage_claim_not_bound_to_persisted_artifact(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "repeat"
    directory.mkdir()
    raw = _orchestration_replay_record(directory, "binding")
    raw["stage_results"]["AB"]["output_sha256"] = _sha256(  # type: ignore[index]
        "different-output"
    )

    with pytest.raises(ValueError, match="AB.*artifact|artifact.*AB"):
        orchestration_module._validate_replay_record(raw, directory)


def test_replay_record_rejects_terminal_fingerprint_not_bound_to_stage_evidence(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "repeat"
    directory.mkdir()
    raw = _orchestration_replay_record(
        directory, "binding", terminal_stage="BA"
    )
    raw["stage_results"]["BA"]["error_fingerprint"] = _sha256(  # type: ignore[index]
        "forged-terminal-fingerprint"
    )

    with pytest.raises(ValueError, match="BA.*fingerprint|fingerprint.*BA"):
        orchestration_module._validate_replay_record(raw, directory)


def test_replay_record_rejects_two_n_claim_without_bound_second_output_artifact(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "repeat"
    directory.mkdir()
    raw = _orchestration_replay_record(
        directory,
        "binding",
        two_n_result={
            "second_output_sha256": _sha256("claimed-second-output"),
            "second_output_hard_state_id": _sha256("claimed-second-hard-state"),
        },
    )

    with pytest.raises(ValueError, match="second_round_output|second output"):
        orchestration_module._validate_replay_record(raw, directory)


def test_bidirectional_writeback_requires_both_direction_cases_stable() -> None:
    pair = {
        "row_id": "advisor-U14-pair-A-B",
        "action_a_id": "A",
        "action_b_id": "B",
        "action_a_directional_status": "authorized_all_others",
        "action_b_directional_status": "authorized_all_others",
        "source_row_ids": "[]",
    }
    stable_families = {
        "worker": "stable",
        "external_opt": "stable",
        "two_n": "stable",
    }
    witnesses = [
        {
            "case_id": "case-A",
            "case": {"authorized_action_id": "A"},
            "stable_false_authorization": "true",
            "family_statuses": stable_families,
            "replay_time_ms": 3,
        },
        {
            "case_id": "case-B",
            "case": {"authorized_action_id": "B"},
            "stable_false_authorization": "false",
            "family_statuses": {
                **stable_families,
                "worker": "nondeterministic",
            },
            "replay_time_ms": 5,
        },
    ]

    orchestration_module._writeback_replay_results(pair, witnesses)

    assert pair["stable_false_authorization"] == "false"
    assert pair["worker_replay_status"] == "nondeterministic"
    assert pair["external_opt_replay_status"] == "stable"
    assert pair["two_n_replay_status"] == "stable"
    assert pair["replay_time_ms"] == 8
    assert json.loads(pair["source_row_ids"]) == ["case-A", "case-B"]


def test_orchestration_same_two_n_drift_from_original_is_mismatch(
    tmp_path: Path,
) -> None:
    calls: dict[str, list[object]] = {
        name: []
        for name in ("profile", "pair", "two_n", "worker", "opt", "replay_two_n")
    }
    dependencies = _orchestration_dependencies(calls)

    def mismatching_two_n(
        case: dict[str, object], repetition: int, directory: Path
    ) -> dict[str, object]:
        replayed = dict(case["expected_two_n"])
        replayed["second_round_effect_sha256"] = _sha256("same-wrong-effect")
        replayed["second_output_sha256"] = _sha256(
            (
                "; second-output "
                + replayed["second_round_effect_sha256"]
                + "\n"
            ).encode("utf-8")
        )
        calls["replay_two_n"].append(repetition)
        return _orchestration_replay_record(
            directory, "stable-wrong", two_n_result=replayed
        )

    dependencies = replace(dependencies, replay_two_n=mismatching_two_n)
    root = tmp_path / "root.ll"
    root.write_text("; root\n", encoding="utf-8")
    actions = {name: _OrchestrationAction(name) for name in ("A", "B", "C")}
    result = run_study_orchestration(
        out_dir=tmp_path / "output" / "smoke",
        isolation_root=tmp_path,
        study_manifest_id="manifest-1",
        programs={"program-1": root},
        groups={
            "U14": (actions["A"], actions["B"]),
            "U30": tuple(actions.values()),
            "Uall": tuple(actions.values()),
        },
        dependencies=dependencies,
    )
    pair = next(
        row
        for row in result.two_n_results["U14"]["pair_rows"]
        if row["action_a_id"] == "A" and row["action_b_id"] == "B"
    )
    assert pair["stable_false_authorization"] == "false"
    assert pair["two_n_replay_status"] == "mismatch"


def test_orchestration_rejects_u14_not_nested_in_u30() -> None:
    actions = {name: _OrchestrationAction(name) for name in ("A", "B", "C")}

    with pytest.raises(ValueError, match="U14.*U30"):
        orchestration_module._validate_groups(
            {
                "U14": (actions["A"], actions["B"]),
                "U30": (actions["A"], actions["C"]),
                "Uall": tuple(actions.values()),
            }
        )


def test_raw_execution_semantics_revision_is_v6() -> None:
    assert "raw-v6" in orchestration_module.RAW_EXECUTION_SEMANTICS_REVISION


def test_orchestration_rejects_pair_first_round_identity_not_bound_to_profile(
    tmp_path: Path,
) -> None:
    calls: dict[str, list[object]] = {
        name: []
        for name in ("profile", "pair", "two_n", "worker", "opt", "replay_two_n")
    }
    dependencies = _orchestration_dependencies(calls)
    original_pairs = dependencies.run_uall_pairs

    def cross_bound_to_wrong_profile(*args: object) -> list[dict[str, object]]:
        rows = original_pairs(*args)  # type: ignore[arg-type]
        rows[0]["a_hard_state_id"] = "0" * 64
        return rows

    dependencies = replace(
        dependencies, run_uall_pairs=cross_bound_to_wrong_profile
    )
    root = tmp_path / "root.ll"
    root.write_text("; root\n", encoding="utf-8", newline="\n")
    actions = {name: _OrchestrationAction(name) for name in ("A", "B", "C")}

    with pytest.raises(ValueError, match="first-round.*profile"):
        run_study_orchestration(
            out_dir=tmp_path / "output" / "formal",
            isolation_root=tmp_path,
            study_manifest_id="manifest-1",
            programs={"program-1": root},
            groups={
                "U14": (actions["A"], actions["B"]),
                "U30": tuple(actions.values()),
                "Uall": tuple(actions.values()),
            },
            dependencies=dependencies,
        )
    assert calls["worker"] == calls["opt"] == calls["replay_two_n"] == []


def test_orchestration_mandatorily_replays_both_terminal_orders_with_family_local_stderr(
    tmp_path: Path,
) -> None:
    calls: dict[str, list[object]] = {
        name: []
        for name in ("profile", "pair", "two_n", "worker", "opt", "replay_two_n")
    }
    dependencies = _orchestration_dependencies(calls)
    original_pairs = dependencies.run_uall_pairs

    def both_terminal(*args: object) -> list[dict[str, object]]:
        rows = original_pairs(*args)  # type: ignore[arg-type]
        target = next(row for row in rows if row["row_id"] == "pair-A-B")
        target.update(
            {
                "dynamic_result": "failed",
                "ab_status": "error",
                "ab_verifier_status": "not_run",
                "ab_hard_state_id": "",
                "ab_output_sha256": "",
                "ab_stderr_sha256": _sha256("worker-ab-terminal"),
                "ba_status": "error",
                "ba_verifier_status": "not_run",
                "ba_hard_state_id": "",
                "ba_output_sha256": "",
                "ba_stderr_sha256": _sha256("worker-ba-terminal"),
                "stderr": "worker-ab-terminal\nworker-ba-terminal",
                "stderr_sha256": _sha256(
                    "worker-ab-terminal\nworker-ba-terminal"
                ),
            }
        )
        return rows

    def replay(family: str):
        def run(
            case: dict[str, object], repetition: int, directory: Path
        ) -> dict[str, object]:
            calls[family].append(repetition)
            prefix = "external" if family == "opt" else "worker"
            expected_stages = case["expected_pair_stages"]
            return _orchestration_replay_record(
                directory,
                f"{family}-double-terminal",
                terminal_stages={
                    "AB": f"{prefix}-ab-terminal",
                    "BA": f"{prefix}-ba-terminal",
                },
                stage_hard_state_ids={
                    name: str(stage["hard_state_id"])
                    for name, stage in expected_stages.items()
                    if str(stage.get("execution_status", "")) == "success"
                },
                two_n_result=(
                    dict(case["expected_two_n"])
                    if family == "replay_two_n"
                    else {}
                ),
            )

        return run

    dependencies = replace(
        dependencies,
        run_uall_pairs=both_terminal,
        replay_worker=replay("worker"),
        replay_external_opt=replay("opt"),
        replay_two_n=replay("replay_two_n"),
    )
    root = tmp_path / "root.ll"
    root.write_text("; root\n", encoding="utf-8", newline="\n")
    actions = {name: _OrchestrationAction(name) for name in ("A", "B", "C")}
    result = run_study_orchestration(
        out_dir=tmp_path / "output" / "formal",
        isolation_root=tmp_path,
        study_manifest_id="manifest-1",
        programs={"program-1": root},
        groups={
            "U14": (actions["A"], actions["B"]),
            "U30": tuple(actions.values()),
            "Uall": tuple(actions.values()),
        },
        dependencies=dependencies,
    )

    assert len(result.false_authorizations) == 2
    pair = result.two_n_results["U14"]["pair_rows"][0]
    assert pair["stable_false_authorization"] == "true"
    assert pair["worker_replay_status"] == "stable"
    assert pair["external_opt_replay_status"] == "stable"
    assert pair["two_n_replay_status"] == "stable"
    assert calls["worker"] == calls["opt"] == calls["replay_two_n"] == [
        1,
        2,
        1,
        2,
    ]


def test_orchestration_rejects_non_smoke_or_formal_output_before_any_runner(tmp_path: Path) -> None:
    calls: dict[str, list[object]] = {name: [] for name in ("profile", "pair", "two_n", "worker", "opt", "replay_two_n")}
    root = tmp_path / "root.ll"
    root.write_text("; root\n", encoding="utf-8", newline="\n")
    actions = {name: _OrchestrationAction(name) for name in ("A", "B")}

    with pytest.raises(ValueError, match="output/smoke or output/formal"):
        run_study_orchestration(
            out_dir=tmp_path / "docs",
            isolation_root=tmp_path,
            study_manifest_id="manifest-1",
            programs={"program-1": root},
            groups={"U14": tuple(actions.values()), "U30": tuple(actions.values()), "Uall": tuple(actions.values())},
            dependencies=_orchestration_dependencies(calls),
        )

    assert not any(calls.values())


def test_orchestration_invalidates_all_downstream_stages_when_reprofiled_content_changes(
    tmp_path: Path,
) -> None:
    calls: dict[str, list[object]] = {name: [] for name in ("profile", "pair", "two_n", "worker", "opt", "replay_two_n")}
    dependencies = _orchestration_dependencies(calls)
    changed = {"value": False}
    original_profile = dependencies.profile_uall

    def altered_profile(root: Path, actions: tuple[object, ...], directory: Path) -> list[dict[str, object]]:
        rows = original_profile(root, actions, directory)
        if changed["value"]:
            first = Path(rows[0]["output_path"])
            first.write_text("; changed first round\n", encoding="utf-8", newline="\n")
            rows[0]["output_sha256"] = _sha256(first.read_bytes())
            rows[0]["output_hard_state_id"] = _sha256("changed-first-round")
        return rows

    dependencies = replace(dependencies, profile_uall=altered_profile)
    root = tmp_path / "root.ll"
    root.write_text("; root\n", encoding="utf-8", newline="\n")
    actions = {name: _OrchestrationAction(name) for name in ("A", "B", "C")}
    kwargs = {
        "out_dir": tmp_path / "output" / "smoke",
        "isolation_root": tmp_path,
        "study_manifest_id": "manifest-1",
        "programs": {"program-1": root},
        "groups": {"U14": (actions["A"], actions["B"]), "U30": tuple(actions.values()), "Uall": tuple(actions.values())},
        "dependencies": dependencies,
    }
    first = run_study_orchestration(**kwargs)
    profile_result = first.out_dir / first.stage_paths["program-1:profiles"] / "result.json"
    profile_result.write_text("{}\n", encoding="utf-8", newline="\n")
    changed["value"] = True
    run_study_orchestration(**kwargs)

    assert len(calls["profile"]) == 2
    assert len(calls["pair"]) == 2
    assert len(calls["two_n"]) == 6
    assert calls["worker"] == calls["opt"] == calls["replay_two_n"] == [
        1, 2, 1, 2, 1, 2, 1, 2
    ]


def test_orchestration_rejects_false_authorization_without_exact_ab_ba_binding(tmp_path: Path) -> None:
    calls: dict[str, list[object]] = {name: [] for name in ("profile", "pair", "two_n", "worker", "opt", "replay_two_n")}
    dependencies = _orchestration_dependencies(calls)
    original_pairs = dependencies.run_uall_pairs

    def wrong_manifest(*args: object) -> list[dict[str, object]]:
        rows = original_pairs(*args)  # type: ignore[arg-type]
        rows[0]["study_manifest_id"] = "another-manifest"
        return rows

    dependencies = replace(dependencies, run_uall_pairs=wrong_manifest)
    root = tmp_path / "root.ll"
    root.write_text("; root\n", encoding="utf-8", newline="\n")
    actions = {name: _OrchestrationAction(name) for name in ("A", "B", "C")}
    with pytest.raises(ValueError, match="study_manifest_id"):
        run_study_orchestration(
            out_dir=tmp_path / "output" / "formal",
            isolation_root=tmp_path,
            study_manifest_id="manifest-1",
            programs={"program-1": root},
            groups={"U14": (actions["A"], actions["B"]), "U30": tuple(actions.values()), "Uall": tuple(actions.values())},
            dependencies=dependencies,
        )


def test_orchestration_publishes_final_artifact_paths_not_staging_paths(tmp_path: Path) -> None:
    calls: dict[str, list[object]] = {name: [] for name in ("profile", "pair", "two_n", "worker", "opt", "replay_two_n")}
    result = _run_orchestration(tmp_path, calls)

    profile = result.profile_rows["program-1"][0]
    assert Path(str(profile["output_path"])).is_file()
    assert ".stage-" not in str(profile["output_path"])
    profile_record = result.out_dir / result.stage_paths["program-1:profiles"] / _sha256(str(profile["action_id"]))[:16] / "profile.json"
    assert ".stage-" not in profile_record.read_text(encoding="utf-8")


def test_orchestration_rebinds_command_hash_after_staging_path_publication(
    tmp_path: Path,
) -> None:
    final = tmp_path / "output" / "smoke" / "raw" / "profiles"

    def produce(staging: Path) -> dict[str, object]:
        output = staging / "action" / "first.ll"
        output.parent.mkdir(parents=True)
        output.write_text("; first\n", encoding="utf-8", newline="\n")
        command = ["worker", "materialize", str(output)]
        return {
            "rows": [
                {
                    "command": command,
                    "command_sha256": _sha256("\0".join(command)),
                    "output_path": str(output),
                }
            ]
        }

    payload, reused = orchestration_module._run_or_reuse_stage(
        final,
        produce,
        expected_input_sha256="a" * 64,
        isolation_root=tmp_path,
    )

    row = payload["rows"][0]
    assert reused is False
    assert row["command"] == [
        "worker",
        "materialize",
        str(final / "action" / "first.ll"),
    ]
    assert row["command_sha256"] == _sha256("\0".join(row["command"]))


def test_orchestration_rejects_invalid_staged_command_hash_before_rebase(
    tmp_path: Path,
) -> None:
    final = tmp_path / "output" / "smoke" / "raw" / "profiles"

    def produce(staging: Path) -> dict[str, object]:
        output = staging / "action" / "first.ll"
        output.parent.mkdir(parents=True)
        output.write_text("; first\n", encoding="utf-8", newline="\n")
        return {
            "command": ["worker", "materialize", str(output)],
            "command_sha256": "0" * 64,
            "output_path": str(output),
        }

    with pytest.raises(ValueError, match="staged command hash mismatch"):
        orchestration_module._run_or_reuse_stage(
            final,
            produce,
            expected_input_sha256="a" * 64,
            isolation_root=tmp_path,
        )

    assert not final.exists()


def test_rebase_staging_paths_handles_large_nested_pair_evidence_with_two_resolves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / ".pairs.stage-123"
    final = tmp_path / "pairs"
    staging.mkdir()
    evidence = staging / "evidence" / "pair-observations.json"
    evidence.parent.mkdir()

    old = str(staging.resolve(strict=False))
    new = str(final.resolve(strict=False))
    rows = [
        {
            "row_id": f"pair-{index:04d}",
            "ab_output_path": str(staging / "pairs" / f"pair-{index:04d}" / "AB.ll"),
            "ba_output_path": str(staging / "pairs" / f"pair-{index:04d}" / "BA.ll"),
            "artifacts": {
                "merged_input_path": str(staging / "merged" / f"pair-{index:04d}" / "input.ll"),
                "nested": [
                    str(staging / "stderr" / f"pair-{index:04d}.txt"),
                    {"command_log_path": str(staging / "commands" / f"pair-{index:04d}.txt")},
                ],
            },
        }
        for index in range(512)
    ]
    near_prefix = f"{old}-sibling\\must-not-rebase.ll"
    payload = {
        "pair_rows": rows,
        "near_prefix": near_prefix,
        "unrelated": "not a path",
    }
    evidence.write_text(
        json.dumps({"ab_output_path": rows[0]["ab_output_path"], "near_prefix": near_prefix}),
        encoding="utf-8",
    )

    original_resolve = orchestration_module.Path.resolve
    resolve_calls = 0

    def counted_resolve(path: Path, *args: object, **kwargs: object) -> Path:
        nonlocal resolve_calls
        resolve_calls += 1
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(orchestration_module.Path, "resolve", counted_resolve)
    prefixes = orchestration_module._rebase_path_prefixes(staging, final)
    rebased = orchestration_module._rebase_staging_paths(payload, staging, final, prefixes=prefixes)
    orchestration_module._rebase_json_evidence_paths(staging, final, prefixes=prefixes)

    assert resolve_calls == 2
    assert rebased["pair_rows"][0]["ab_output_path"] == new + rows[0]["ab_output_path"][len(old) :]
    assert rebased["pair_rows"][-1]["artifacts"]["nested"][1]["command_log_path"] == (
        new + rows[-1]["artifacts"]["nested"][1]["command_log_path"][len(old) :]
    )
    assert rebased["near_prefix"] == near_prefix
    assert rebased["unrelated"] == "not a path"
    rebased_evidence = json.loads(evidence.read_text(encoding="utf-8"))
    assert rebased_evidence["ab_output_path"] == new + rows[0]["ab_output_path"][len(old) :]
    assert rebased_evidence["near_prefix"] == near_prefix


def test_orchestration_failed_stage_never_publishes_completion_marker(tmp_path: Path) -> None:
    calls: dict[str, list[object]] = {name: [] for name in ("profile", "pair", "two_n", "worker", "opt", "replay_two_n")}
    dependencies = _orchestration_dependencies(calls)

    def fail_profile(_root: Path, _actions: tuple[object, ...], _directory: Path) -> list[dict[str, object]]:
        raise RuntimeError("forced profile failure")

    root = tmp_path / "root.ll"
    root.write_text("; root\n", encoding="utf-8", newline="\n")
    actions = {name: _OrchestrationAction(name) for name in ("A", "B", "C")}
    with pytest.raises(RuntimeError, match="forced profile failure"):
        run_study_orchestration(
            out_dir=tmp_path / "output" / "formal",
            isolation_root=tmp_path,
            study_manifest_id="manifest-1",
            programs={"program-1": root},
            groups={"U14": (actions["A"], actions["B"]), "U30": tuple(actions.values()), "Uall": tuple(actions.values())},
            dependencies=replace(dependencies, profile_uall=fail_profile),
        )

    assert not list((tmp_path / "output" / "formal").rglob("complete.json"))


def test_orchestration_stage_evidence_records_status_command_hashes_artifacts_and_stderr(
    tmp_path: Path,
) -> None:
    calls: dict[str, list[object]] = {name: [] for name in ("profile", "pair", "two_n", "worker", "opt", "replay_two_n")}
    result = _run_orchestration(tmp_path, calls)
    profile = result.profile_rows["program-1"][0]
    evidence_paths = [
        result.out_dir / result.stage_paths["program-1:profiles"] / _sha256(str(profile["action_id"]))[:16] / "profile.json",
        result.out_dir / result.stage_paths["program-1:view:U14"] / "view_evidence.json",
        result.out_dir / result.stage_paths["program-1:two_n:U14"] / "two_n_evidence.json",
    ]
    required = {
        "status",
        "command",
        "command_sha256",
        "stderr",
        "stderr_sha256",
        "artifact_available",
        "artifact_materialized",
    }
    for evidence_path in evidence_paths:
        record = json.loads(evidence_path.read_text(encoding="utf-8"))
        assert required.issubset(record)


def test_orchestration_rejects_incomplete_group_2n_typed_rows(tmp_path: Path) -> None:
    calls: dict[str, list[object]] = {name: [] for name in ("profile", "pair", "two_n", "worker", "opt", "replay_two_n")}
    dependencies = _orchestration_dependencies(calls)
    original_two_n = dependencies.run_group_two_n

    def incomplete_two_n(*args: object) -> dict[str, object]:
        result = original_two_n(*args)  # type: ignore[arg-type]
        result["directional_rows"] = result["directional_rows"][:-1]
        return result

    root = tmp_path / "root.ll"
    root.write_text("; root\n", encoding="utf-8", newline="\n")
    actions = {name: _OrchestrationAction(name) for name in ("A", "B", "C")}
    with pytest.raises(ValueError, match="directional rows must cover"):
        run_study_orchestration(
            out_dir=tmp_path / "output" / "smoke",
            isolation_root=tmp_path,
            study_manifest_id="manifest-1",
            programs={"program-1": root},
            groups={"U14": (actions["A"], actions["B"]), "U30": tuple(actions.values()), "Uall": tuple(actions.values())},
            dependencies=replace(dependencies, run_group_two_n=incomplete_two_n),
        )
