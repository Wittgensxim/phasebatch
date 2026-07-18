from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from advisor_study.manifest import (
    ProgramRecord,
    build_study_manifest,
    canonical_sha256,
    extend_program_manifest,
    freeze_formal_program_manifest,
    freeze_program_manifest,
    require_formal_program_boundary,
    require_study_manifest,
    select_extension_candidates,
    stable_rank,
    validate_program_source_paths,
)


def _sha256(value: str | bytes) -> str:
    data = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _program(
    name: str,
    category: str,
    *,
    compile_status: str = "success",
    preflight_status: str = "success",
    source_sha256: str | None = None,
    relative_path: str | None = None,
    source_path: str | None = None,
    source_size_bytes: int = 128,
    program_family: str | None = None,
) -> ProgramRecord:
    relative = relative_path or f"SingleSource/{category}/{name}.c"
    return ProgramRecord(
        program_id=name,
        source_path=source_path or f"E:/llvm-test-suite/{relative}",
        relative_path=relative,
        program_family=program_family or "/".join(relative.split("/")[:-1]),
        source_sha256=source_sha256 or _sha256(f"source:{name}"),
        source_size_bytes=source_size_bytes,
        compile_command=("clang.exe", "-S", relative, "-o", f"{name}.ll"),
        compile_status=compile_status,
        compile_stderr_sha256="",
        root_ir_path=f"E:/roots/{name}.ll",
        root_ir_sha256=_sha256(f"root:{name}"),
        root_hard_state_id=f"hard:{name}",
        target="x86_64-w64-windows-gnu",
        data_layout="e-m:w-p270:32:32-p271:32:32-p272:64:64",
        preflight_status=preflight_status,
    )


def test_existing_50_are_retained_and_extension_is_deterministic() -> None:
    fixed = [
        replace(
            _program(f"fixed-{index}", f"Fixed{index}"),
            selection_class="fixed",
            selection_order=index + 1,
        )
        for index in range(50)
    ]
    candidates = [
        _program(f"extra-{index}", f"Extra{index}") for index in range(80)
    ]

    first = extend_program_manifest(fixed, candidates, target=100, seed=0)
    second = extend_program_manifest(
        fixed, list(reversed(candidates)), target=100, seed=0
    )

    assert first == second
    assert len(first) == 100
    assert set(fixed) <= set(first)
    assert [row.relative_path for row in first] == sorted(
        row.relative_path for row in first
    )
    assert all("dynamic_relation" not in row.__dataclass_fields__ for row in first)


def test_stable_rank_is_seeded_and_selection_has_no_result_labels() -> None:
    assert stable_rank(0, "SingleSource/A/x.c") == (
        "e815195f934b74608962282c7a092e61b93e1df20b6469540a1eb6856119f47e"
    )
    assert stable_rank(7, "SingleSource/Z/f.c") == (
        "340c22b36a2d46aabc8bb440a6e79f955fec5e53065ce770f88c307ba3254986"
    )

    fixed = [_program(f"fixed-{index}", f"Fixed{index}") for index in range(50)]
    candidates = [_program(f"extra-{index}", f"Extra{index}") for index in range(60)]
    selected = extend_program_manifest(fixed, candidates, target=100, seed=0)

    assert all("pair" not in name and "two_n" not in name for name in ProgramRecord.__dataclass_fields__)
    assert all(row.preflight_status == "success" for row in selected)


def test_extension_continues_after_fixed_category_cap() -> None:
    fixed = [_program(f"fixed-{index}", "Full") for index in range(5)]
    candidates = [
        _program("must-skip", "Full"),
        _program("reserve-a", "A"),
        _program("reserve-b", "B"),
    ]

    selected = extend_program_manifest(
        fixed,
        candidates,
        target=7,
        seed=0,
        per_category_cap=5,
    )

    assert {row.program_id for row in selected} == {
        "fixed-0",
        "fixed-1",
        "fixed-2",
        "fixed-3",
        "fixed-4",
        "reserve-a",
        "reserve-b",
    }


def test_program_family_is_derived_from_relative_path_and_cannot_bypass_cap() -> None:
    with pytest.raises(ValueError, match="program_family must equal source parent"):
        _program("forged", "Full", program_family="NotTheSourceParent")

    fixed = [_program(f"fixed-{index}", "Full") for index in range(5)]
    candidate = _program("candidate", "Full")
    with pytest.raises(ValueError, match="needed 6 programs, selected 5"):
        freeze_program_manifest(fixed, [candidate], target=6, seed=0)


def test_extension_selection_continues_global_rank_without_category_bootstrap() -> None:
    fixed = [_program(f"fixed-{index}", "AlreadyFull") for index in range(5)]
    candidates = [
        _program("full-ignored", "AlreadyFull"),
        *[_program(f"a-{index}", "A") for index in range(3)],
        *[_program(f"b-{index}", "B") for index in range(3)],
        *[_program(f"c-{index}", "C") for index in range(3)],
        _program("d-only", "D"),
    ]
    actual = select_extension_candidates(
        fixed, candidates, needed=4, seed=0, per_category_cap=5
    )
    assert [row.program_id for row in actual] == ["c-1", "b-0", "a-0", "c-0"]


def test_formal_boundary_systematically_selects_ten_from_fixed_fifty() -> None:
    fixed = [
        replace(
            _program(f"fixed-{index}", f"Fixed{index}"),
            selection_class="fixed",
            selection_order=index + 1,
        )
        for index in range(50)
    ]
    candidates = [_program(f"extra-{index}", f"Extra{index}") for index in range(60)]
    require_formal_program_boundary(fixed, target=10)
    frozen = freeze_formal_program_manifest(fixed, (), seed=0)
    selected = sorted(frozen.programs, key=lambda row: row.selection_order or 0)
    assert len(selected) == 10
    assert [row.program_id for row in selected] == [
        "fixed-2",
        "fixed-7",
        "fixed-12",
        "fixed-17",
        "fixed-22",
        "fixed-27",
        "fixed-32",
        "fixed-37",
        "fixed-42",
        "fixed-47",
    ]
    assert [row.selection_order for row in selected] == list(range(1, 11))
    assert len({row.program_family for row in selected}) == 10
    assert frozen.reserve_order == ()
    assert all(row.selection_class == "fixed" for row in frozen.programs)

    with pytest.raises(ValueError, match="source inventory requires exactly 50"):
        require_formal_program_boundary(fixed[:-1], target=10)
    for invalid_target in (9, 11, 50):
        with pytest.raises(ValueError, match="target=10"):
            require_formal_program_boundary(fixed, target=invalid_target)
    with pytest.raises(ValueError, match="forbids candidate"):
        freeze_formal_program_manifest(fixed, candidates, seed=0)


def test_formal_midpoint_selection_requires_ten_distinct_program_families() -> None:
    fixed = [
        replace(
            _program(f"fixed-{index}", f"Fixed{index}"),
            selection_class="fixed",
            selection_order=index + 1,
        )
        for index in range(50)
    ]
    duplicate_family = fixed[2].program_family
    duplicate_relative_path = f"{duplicate_family}/fixed-7.c"
    fixed[7] = replace(
        fixed[7],
        relative_path=duplicate_relative_path,
        source_path=f"E:/llvm-test-suite/{duplicate_relative_path}",
        program_family=duplicate_family,
    )

    with pytest.raises(ValueError, match="10 distinct program_family"):
        freeze_formal_program_manifest(fixed, (), seed=0)


def test_freeze_retains_failures_and_uses_only_frozen_reserve_order() -> None:
    fixed = [_program("fixed-a", "Fixed"), _program("fixed-b", "FixedB")]
    candidates = [
        _program(
            "compile-failure",
            "Failure",
            compile_status="failed",
            preflight_status="compile_failed",
        ),
        _program(
            "root-ir-failure",
            "Failure2",
            preflight_status="root_ir_failed",
        ),
        _program("reserve-a", "A"),
        _program("reserve-b", "B"),
    ]

    frozen = freeze_program_manifest(fixed, candidates, target=4, seed=0)

    assert len(frozen.programs) == 4
    assert {row.program_id for row in frozen.programs} == {
        "fixed-a",
        "fixed-b",
        "reserve-a",
        "reserve-b",
    }
    assert len(frozen.reserve_order) == len(candidates)
    assert [row.reserve_rank for row in frozen.reserve_order] == list(
        range(1, len(candidates) + 1)
    )
    assert {
        entry.program_id: entry.preflight_status for entry in frozen.preflight_ledger
    } == {
        "compile-failure": "compile_failed",
        "root-ir-failure": "root_ir_failed",
    }


def test_freeze_fails_when_too_few_valid_rows_and_ledgers_large_sources() -> None:
    fixed = [_program("fixed", "Fixed")]
    only_valid = _program("only-valid", "Valid")
    with pytest.raises(ValueError, match="needed 3 programs, selected 2"):
        freeze_program_manifest(fixed, [only_valid], target=3, seed=0)

    oversized = _program("oversized", "Large", source_size_bytes=200_001)
    frozen = freeze_program_manifest(
        fixed,
        [oversized, only_valid],
        target=2,
        seed=0,
        max_source_bytes=200_000,
    )
    assert {row.program_id for row in frozen.programs} == {"fixed", "only-valid"}
    assert {(row.program_id, row.reason) for row in frozen.preflight_ledger} == {
        ("oversized", "source_too_large")
    }


def test_extension_rejects_duplicate_source_hash_and_source_hash_drift() -> None:
    fixed = [_program("fixed", "Fixed")]
    duplicate = _program(
        "duplicate",
        "Candidate",
        source_sha256=fixed[0].source_sha256,
    )
    with pytest.raises(ValueError, match="duplicate source_sha256"):
        extend_program_manifest(fixed, [duplicate], target=2, seed=0)

    drifted = _program(
        "fixed-drifted",
        "Fixed",
        relative_path=fixed[0].relative_path,
        source_sha256=_sha256("changed source"),
    )
    with pytest.raises(ValueError, match="source hash drift"):
        extend_program_manifest(fixed, [drifted], target=2, seed=0)


def test_windows_source_spelling_is_a_fixed_inventory_copy_on_any_host() -> None:
    fixed = _program("fixed", "Fixed")
    copied = _program(
        "inventory-copy",
        "Fixed",
        source_sha256=fixed.source_sha256,
        relative_path=fixed.relative_path,
        source_path=fixed.source_path.replace("/", "\\"),
    )
    frozen = freeze_program_manifest([fixed], [copied], target=1, seed=0)
    assert frozen.programs == (fixed,)


def test_source_path_validation_rejects_escape_and_source_hash_drift(
    tmp_path: Path,
) -> None:
    single_source = tmp_path / "suite" / "SingleSource"
    source = single_source / "UnitTests" / "sample.c"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"int main(void) { return 0; }\n")
    record = _program(
        "sample",
        "UnitTests",
        source_path=str(source),
        relative_path="SingleSource/UnitTests/sample.c",
        source_sha256=_sha256(source.read_bytes()),
        source_size_bytes=source.stat().st_size,
    )

    validate_program_source_paths([record], single_source)

    outside = tmp_path / "outside.c"
    outside.write_bytes(source.read_bytes())
    escaped = replace(record, source_path=str(outside))
    with pytest.raises(ValueError, match="escapes SingleSource root"):
        validate_program_source_paths([escaped], single_source)

    source.write_bytes(b"int main(void) { return 1; }\n")
    with pytest.raises(ValueError, match="source hash drift"):
        validate_program_source_paths([record], single_source)


def test_study_identity_is_canonical_and_requires_exact_match() -> None:
    programs = (_program("b", "B"), _program("a", "A"))
    kwargs = {
        "programs": programs,
        "pass_policy": {"schema": 1, "policy_sha256": "policy"},
        "pass_inventory": {"actions": ["a", "b"]},
        "pass_preflight": {"eligible": ["a", "b"]},
        "pass_groups": {"U14": ["a"]},
        "llvm_commit": "aac212f0bc9acbc40a8a2e9638f4b7496c25d0b2",
        "target": "x86_64-w64-windows-gnu",
        "tools": {
            "opt": {"path": "E:/llvm/build/bin/opt.exe", "sha256": _sha256("opt")},
            "clang": {"path": "E:/llvm/build/bin/clang.exe", "sha256": _sha256("clang")},
            "worker": {"path": "E:/PO2/worker/build/phasebatch-worker.exe", "sha256": _sha256("worker")},
            "merge_helper": {"path": "E:/PO2/experiments/helper.exe", "sha256": _sha256("merge_helper")},
        },
        "hard_state_policy": {"schema": "hard-v1"},
        "comparator": {"schema": "comparator-v1"},
        "jobs": 8,
        "timeout_s": 15,
        "artifact_policy": {"retain_roots": True},
    }

    first = build_study_manifest(**kwargs)
    second = build_study_manifest(**{**kwargs, "programs": tuple(reversed(programs))})

    assert first == second
    assert first["authority_granted"] is False
    assert first["proved_commute"] is False
    assert first["study_manifest_id"] == first["study_manifest_sha256"]
    require_study_manifest(first, json.loads(json.dumps(first)))

    drifted = json.loads(json.dumps(first))
    drifted["tools"]["worker"]["sha256"] = _sha256("other-worker")
    with pytest.raises(ValueError, match="study manifest identity mismatch"):
        require_study_manifest(first, drifted)

    invalid_tool_hash = {**kwargs, "tools": {**kwargs["tools"]}}
    invalid_tool_hash["tools"]["opt"] = {
        **invalid_tool_hash["tools"]["opt"],
        "sha256": "not-a-sha256",
    }
    with pytest.raises(ValueError, match="tools.opt.sha256"):
        build_study_manifest(**invalid_tool_hash)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda manifest: manifest.pop("tools"), "missing"),
        (lambda manifest: manifest.__setitem__("unexpected", "value"), "unexpected"),
        (lambda manifest: manifest.__setitem__("execution", []), "execution"),
        (lambda manifest: manifest["tools"].__setitem__("opt", []), "tools.opt"),
    ],
)
def test_require_study_manifest_rejects_self_hashed_malformed_shapes(
    mutate: object,
    match: str,
) -> None:
    programs = (_program("a", "A"),)
    valid = build_study_manifest(
        programs=programs,
        pass_policy={"schema": 1},
        pass_inventory={"actions": ["a"]},
        pass_preflight={"eligible": ["a"]},
        pass_groups={"U14": ["a"]},
        llvm_commit="aac212f0bc9acbc40a8a2e9638f4b7496c25d0b2",
        target="x86_64-w64-windows-gnu",
        tools={
            name: {"path": f"E:/{name}.exe", "sha256": _sha256(name)}
            for name in ("opt", "clang", "worker", "merge_helper")
        },
        hard_state_policy={"schema": "hard-v1"},
        comparator={"schema": "comparator-v1"},
        jobs=1,
        timeout_s=1,
        artifact_policy={"retain_roots": True},
    )
    malformed = json.loads(json.dumps(valid))
    assert callable(mutate)
    mutate(malformed)
    identity = {
        key: value
        for key, value in malformed.items()
        if key not in {"study_manifest_id", "study_manifest_sha256"}
    }
    malformed["study_manifest_id"] = canonical_sha256(identity)
    malformed["study_manifest_sha256"] = malformed["study_manifest_id"]
    with pytest.raises(ValueError, match=match):
        require_study_manifest(malformed, malformed)
