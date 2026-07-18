"""Black-box contract tests for the isolated LLVM patch-inspection helper.

The helper is intentionally separate from the production Worker.  These tests
exercise only JSON-lines requests and fixtures under this experiment directory.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
HELPER = EXPERIMENT_ROOT / "build" / "merge_helper" / "phasebatch-2n-merge.exe"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "advisor_2n_merge"


def _require_helper() -> Path:
    assert HELPER.is_file(), f"merge helper missing: {HELPER}"
    return HELPER


def _request(request: dict[str, object]) -> dict[str, object]:
    helper = _require_helper()
    completed = subprocess.run(
        [str(helper)],
        input=json.dumps(request, sort_keys=True) + "\n",
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert len(lines) == 1, completed.stdout
    return json.loads(lines[0])


def _inspect(output: str) -> dict[str, object]:
    return _request(
        {
            "request_id": 41,
            "op": "inspect_patch",
            "base_path": str(FIXTURES / "base.ll"),
            "output_path": str(FIXTURES / output),
        }
    )


def _assert_rejected(reply: dict[str, object], *, request_id: int = 41) -> None:
    assert reply["request_id"] == request_id
    assert reply["status"] == "error"
    assert isinstance(reply.get("error_kind"), str)
    assert isinstance(reply.get("error_message"), str)


def test_ping_advertises_the_inspection_protocol() -> None:
    reply = _request({"request_id": 7, "op": "ping"})

    assert reply["request_id"] == 7
    assert reply["status"] == "ok"
    assert reply["protocol_version"] == 1
    assert reply["operations"] == ["ping", "inspect_patch", "merge", "compare_effect"]
    assert isinstance(reply["llvm_version"], str)


@pytest.mark.parametrize(
    ("payload", "expected_request_id"),
    [
        ({"op": "ping"}, -1),
        ({"request_id": "7", "op": "ping"}, -1),
        ({"request_id": True, "op": "ping"}, -1),
        ({"request_id": 7}, 7),
        ({"request_id": 7, "op": ""}, 7),
        ({"request_id": 7, "op": 9}, 7),
    ],
)
def test_json_line_requests_require_an_integer_id_and_nonempty_string_operation(
    payload: dict[str, object], expected_request_id: int
) -> None:
    reply = _request(payload)

    assert reply["request_id"] == expected_request_id
    assert reply["status"] == "error"
    assert reply["error_kind"] == "invalid_request"
    assert isinstance(reply["error_message"], str)
    assert reply["error_message"]


def test_inspect_patch_reports_a_typed_single_function_patch() -> None:
    reply = _inspect("change_f.ll")

    assert reply["request_id"] == 41
    assert reply["status"] == "ok"
    assert reply["changed_functions"] == ["f"]
    assert reply["base_skeleton_hash"] == reply["output_skeleton_hash"]
    assert reply["base_symbol_inventory_hash"] == reply["output_symbol_inventory_hash"]
    assert len(reply["base_module_hash"]) == 64
    assert len(reply["output_module_hash"]) == 64
    assert len(reply["patch_hash"]) == 64
    assert reply["patch_record"] == {
        "schema_version": 1,
        "changed_functions": [
            {
                "name": "f",
                "base_isolated_hash": reply["changed_function_hashes"][0]["base_isolated_hash"],
                "output_isolated_hash": reply["changed_function_hashes"][0]["output_isolated_hash"],
            }
        ],
    }
    assert reply["changed_function_hashes"][0]["name"] == "f"
    assert reply["changed_function_hashes"][0]["base_isolated_hash"] != reply[
        "changed_function_hashes"
    ][0]["output_isolated_hash"]


def test_inspect_patch_is_deterministic_and_retains_noop_as_empty_patch() -> None:
    first = _inspect("change_g.ll")
    second = _inspect("change_g.ll")
    noop = _inspect("base.ll")

    assert first["patch_hash"] == second["patch_hash"]
    assert first["changed_functions"] == ["g"]
    assert noop["status"] == "ok"
    assert noop["changed_functions"] == []
    assert noop["changed_function_hashes"] == []


def test_inspect_patch_detects_the_same_changed_function_for_later_overlap_gate() -> None:
    left = _inspect("change_f.ll")
    right = _inspect("change_f_again.ll")

    assert left["changed_functions"] == ["f"]
    assert right["changed_functions"] == ["f"]
    assert left["patch_hash"] != right["patch_hash"]


@pytest.mark.parametrize(
    "output",
    [
        "module_attribute_change.ll",
    ],
)
def test_inspect_patch_rejects_module_level_changes(output: str) -> None:
    _assert_rejected(_inspect(output))


@pytest.mark.parametrize(
    ("label", "mutator"),
    [
        (
            "signature",
            lambda text: text.replace("define i32 @f(i32 %x)", "define i64 @f(i64 %x)")
            .replace("%sum = add nsw i32 %x, 1", "%sum = add nsw i64 %x, 1")
            .replace("ret i32 %sum", "ret i64 %sum"),
        ),
        ("linkage", lambda text: text.replace("define i32 @f", "define internal i32 @f")),
        ("function_attribute", lambda text: text.replace("define i32 @f(i32 %x)", "define i32 @f(i32 %x) noinline")),
        ("global", lambda text: text.replace("@shared = global i32 7", "@shared = global i32 9")),
        (
            "fresh_symbol",
            lambda text: text.replace(
                "declare i32 @external(i32)", "declare i32 @external(i32)\ndeclare i32 @fresh(i32)"
            ),
        ),
        ("target", lambda text: text.replace("x86_64-pc-windows-msvc", "x86_64-pc-linux-gnu")),
        ("datalayout", lambda text: text.replace("n8:16:32:64-S128", "n8:16:32:64-S64")),
        ("section", lambda text: text.replace("define i32 @f(i32 %x)", 'define i32 @f(i32 %x) section ".text.advisor"')),
        ("gc", lambda text: text.replace("define i32 @f(i32 %x)", 'define i32 @f(i32 %x) gc "statepoint-example"')),
    ],
)
def test_inspect_patch_rejects_non_body_and_fresh_identity_changes(
    tmp_path: Path, label: str, mutator
) -> None:
    output = tmp_path / f"{label}.ll"
    output.write_text(mutator((FIXTURES / "base.ll").read_text(encoding="utf-8")), encoding="utf-8")

    reply = _request(
        {
            "request_id": 41,
            "op": "inspect_patch",
            "base_path": str(FIXTURES / "base.ll"),
            "output_path": str(output),
        }
    )
    _assert_rejected(reply)


@pytest.mark.parametrize(
    ("label", "append", "expected_kind"),
    [
        ("alias", "\n@f_alias = alias i32 (i32), ptr @f\n", "symbol_inventory_changed"),
        (
            "ifunc",
            "\ndefine ptr @resolver() {\nentry:\n  ret ptr @f\n}\n@ifunc = ifunc i32 (i32), ptr @resolver\n",
            "symbol_inventory_changed",
        ),
        ("module_flags", "\n!llvm.module.flags = !{!0}\n!0 = !{i32 1, !\"advisor\", i32 1}\n", "module_skeleton_changed"),
        ("comdat", "", "function_identity_changed"),
        ("personality", "", "function_identity_changed"),
    ],
)
def test_inspect_patch_rejects_remaining_forbidden_ir_identity_changes(
    tmp_path: Path, label: str, append: str, expected_kind: str
) -> None:
    text = (FIXTURES / "base.ll").read_text(encoding="utf-8") + append
    if label == "comdat":
        text = text.replace("define i32 @f(i32 %x)", "define i32 @f(i32 %x) comdat($f)")
        text = text.replace("@shared = global i32 7, align 4", "@shared = global i32 7, align 4\n$f = comdat any")
    elif label == "personality":
        text = text.replace("define i32 @f(i32 %x)", "define i32 @f(i32 %x) personality ptr @external")

    output = tmp_path / f"{label}.ll"
    output.write_text(text, encoding="utf-8")
    reply = _request(
        {
            "request_id": 41,
            "op": "inspect_patch",
            "base_path": str(FIXTURES / "base.ll"),
            "output_path": str(output),
        }
    )

    _assert_rejected(reply)
    assert reply["error_kind"] == expected_kind


def test_inspect_patch_rejects_parse_or_verifier_failure(tmp_path: Path) -> None:
    output = tmp_path / "invalid.ll"
    output.write_text("define i32 @f( { ret i32 0 }\n", encoding="utf-8")

    reply = _request(
        {
            "request_id": 41,
            "op": "inspect_patch",
            "base_path": str(FIXTURES / "base.ll"),
            "output_path": str(output),
        }
    )
    _assert_rejected(reply)
    assert reply["error_kind"] == "parse_failed"


def test_inspect_patch_reports_a_typed_verification_failure() -> None:
    reply = _inspect("verifier_failure.ll")

    _assert_rejected(reply)
    assert reply["error_kind"] == "verification_failed"


def _merge(
    tmp_path: Path,
    outputs: list[Path],
    *,
    output_name: str = "merged.ll",
) -> tuple[dict[str, object], Path]:
    merged_path = tmp_path / output_name
    reply = _request(
        {
            "request_id": 73,
            "op": "merge",
            "base_path": str(FIXTURES / "base.ll"),
            "output_paths": [str(path) for path in outputs],
            "merged_path": str(merged_path),
        }
    )
    return reply, merged_path


def test_merge_directly_combines_disjoint_whole_function_patches_deterministically(
    tmp_path: Path,
) -> None:
    forward, forward_path = _merge(
        tmp_path,
        [FIXTURES / "change_f.ll", FIXTURES / "change_g.ll"],
        output_name="forward.ll",
    )
    reverse, reverse_path = _merge(
        tmp_path,
        [FIXTURES / "change_g.ll", FIXTURES / "change_f.ll"],
        output_name="reverse.ll",
    )

    assert forward["request_id"] == 73
    assert forward["status"] == "ok"
    assert forward["merged_functions"] == ["f", "g"]
    assert forward["contributed_functions"] == ["f", "g"]
    assert forward["output_module_hash"] == reverse["output_module_hash"]
    assert forward["output_skeleton_hash"] == forward["base_skeleton_hash"]
    assert forward_path.read_bytes() == reverse_path.read_bytes()

    inspected = _request(
        {
            "request_id": 74,
            "op": "inspect_patch",
            "base_path": str(FIXTURES / "base.ll"),
            "output_path": str(forward_path),
        }
    )
    assert inspected["status"] == "ok"
    assert inspected["changed_functions"] == ["f", "g"]


def test_merge_rejects_overlapping_or_unsupported_patches(tmp_path: Path) -> None:
    overlap, _ = _merge(
        tmp_path,
        [FIXTURES / "change_f.ll", FIXTURES / "change_f_again.ll"],
        output_name="overlap.ll",
    )
    _assert_rejected(overlap, request_id=73)
    assert overlap["error_kind"] == "overlapping_function_patch"

    fresh = tmp_path / "fresh_symbol.ll"
    fresh.write_text(
        (FIXTURES / "base.ll")
        .read_text(encoding="utf-8")
        .replace(
            "declare i32 @external(i32)",
            "declare i32 @external(i32)\ndeclare i32 @fresh(i32)",
        ),
        encoding="utf-8",
    )
    unsupported, _ = _merge(tmp_path, [fresh], output_name="unsupported.ll")
    _assert_rejected(unsupported, request_id=73)
    assert unsupported["error_kind"] == "patch_not_mergeable"


def test_merge_requires_an_explicit_output_path_and_never_uses_text_merge() -> None:
    reply = _request(
        {
            "request_id": 75,
            "op": "merge",
            "base_path": str(FIXTURES / "base.ll"),
            "output_paths": [str(FIXTURES / "change_f.ll")],
        }
    )

    _assert_rejected(reply, request_id=75)
    assert reply["error_kind"] == "invalid_request"


def test_merge_rejects_canonical_path_aliases_of_any_input(tmp_path: Path) -> None:
    base_input = tmp_path / "base_input.ll"
    patch_input = tmp_path / "patch_input.ll"
    base_input.write_bytes((FIXTURES / "base.ll").read_bytes())
    patch_input.write_bytes((FIXTURES / "change_f.ll").read_bytes())
    base_before = base_input.read_bytes()
    patch_before = patch_input.read_bytes()

    base_alias = str(base_input.parent) + "\\.\\" + base_input.name
    patch_alias = str(patch_input.parent) + "\\.\\" + patch_input.name
    for request_id, merged_path in ((79, base_alias), (80, patch_alias)):
        reply = _request(
            {
                "request_id": request_id,
                "op": "merge",
                "base_path": str(base_input),
                "output_paths": [str(patch_input)],
                "merged_path": merged_path,
            }
        )
        _assert_rejected(reply, request_id=request_id)
        assert reply["error_kind"] == "invalid_request"

    assert base_input.read_bytes() == base_before
    assert patch_input.read_bytes() == patch_before


def test_merge_records_the_exact_inspected_patch_family_before_cloning(
    tmp_path: Path,
) -> None:
    inspected_f = _inspect("change_f.ll")
    inspected_g = _inspect("change_g.ll")
    reply, _ = _merge(
        tmp_path,
        [FIXTURES / "change_g.ll", FIXTURES / "change_f.ll"],
        output_name="recorded.ll",
    )

    assert reply["status"] == "ok"
    assert reply["input_patch_hashes"] == sorted(
        [inspected_f["patch_hash"], inspected_g["patch_hash"]]
    )
    assert len(reply["input_output_module_hashes"]) == 2


def test_compare_effect_requires_exact_patch_identity_and_preserves_contributions(
    tmp_path: Path,
) -> None:
    merged_g, merged_g_path = _merge(
        tmp_path, [FIXTURES / "change_g.ll"], output_name="merged_g.ll"
    )
    assert merged_g["status"] == "ok"
    merged_fg, merged_fg_path = _merge(
        tmp_path,
        [FIXTURES / "change_f.ll", FIXTURES / "change_g.ll"],
        output_name="merged_fg.ll",
    )
    assert merged_fg["status"] == "ok"

    reply = _request(
        {
            "request_id": 76,
            "op": "compare_effect",
            "first_base_path": str(FIXTURES / "base.ll"),
            "first_output_path": str(FIXTURES / "change_f.ll"),
            "second_base_path": str(merged_g_path),
            "second_output_path": str(merged_fg_path),
            "protected_functions": ["g"],
        }
    )

    assert reply["status"] == "ok"
    assert reply["same_effect"] is True
    assert reply["first_changed_functions"] == ["f"]
    assert reply["second_changed_functions"] == ["f"]
    assert reply["protected_functions"] == ["g"]
    assert reply["protected_functions_preserved"] is True
    assert reply["first_patch_hash"] == reply["second_patch_hash"]


def test_compare_effect_detects_noop_becoming_active_and_lost_contributions(
    tmp_path: Path,
) -> None:
    merged_g, merged_g_path = _merge(
        tmp_path, [FIXTURES / "change_g.ll"], output_name="second_base.ll"
    )
    assert merged_g["status"] == "ok"
    merged_fg, merged_fg_path = _merge(
        tmp_path,
        [FIXTURES / "change_f.ll", FIXTURES / "change_g.ll"],
        output_name="second_output.ll",
    )
    assert merged_fg["status"] == "ok"

    newly_active = _request(
        {
            "request_id": 77,
            "op": "compare_effect",
            "first_base_path": str(FIXTURES / "base.ll"),
            "first_output_path": str(FIXTURES / "base.ll"),
            "second_base_path": str(merged_g_path),
            "second_output_path": str(merged_fg_path),
            "protected_functions": ["g"],
        }
    )
    assert newly_active["status"] == "ok"
    assert newly_active["same_effect"] is False

    changed_g = tmp_path / "changed_g_again.ll"
    changed_g.write_text(
        merged_fg_path.read_text(encoding="utf-8").replace(
            "mul nsw i32 %x, 4", "mul nsw i32 %x, 6"
        ),
        encoding="utf-8",
    )
    lost_contribution = _request(
        {
            "request_id": 78,
            "op": "compare_effect",
            "first_base_path": str(FIXTURES / "base.ll"),
            "first_output_path": str(FIXTURES / "change_f.ll"),
            "second_base_path": str(merged_g_path),
            "second_output_path": str(changed_g),
            "protected_functions": ["g"],
        }
    )
    assert lost_contribution["status"] == "ok"
    assert lost_contribution["same_effect"] is False
    assert lost_contribution["protected_functions_preserved"] is False


def test_compare_effect_requires_the_complete_merged_contribution_family(
    tmp_path: Path,
) -> None:
    merged_g, merged_g_path = _merge(
        tmp_path, [FIXTURES / "change_g.ll"], output_name="complete_base.ll"
    )
    assert merged_g["status"] == "ok"
    merged_fg, merged_fg_path = _merge(
        tmp_path,
        [FIXTURES / "change_f.ll", FIXTURES / "change_g.ll"],
        output_name="complete_output.ll",
    )
    assert merged_fg["status"] == "ok"

    for request_id, protected in ((81, []), (82, ["f", "g"])):
        reply = _request(
            {
                "request_id": request_id,
                "op": "compare_effect",
                "first_base_path": str(FIXTURES / "base.ll"),
                "first_output_path": str(FIXTURES / "change_f.ll"),
                "second_base_path": str(merged_g_path),
                "second_output_path": str(merged_fg_path),
                "protected_functions": protected,
            }
        )
        _assert_rejected(reply, request_id=request_id)
        assert reply["error_kind"] == "protected_functions_mismatch"
