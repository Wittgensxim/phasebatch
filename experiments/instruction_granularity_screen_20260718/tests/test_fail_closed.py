from __future__ import annotations

from instruction_granularity.aggregate import merge_three_decisions
from pathlib import Path

from instruction_granularity.extractors import (
    compare_modules,
    read_validated_artifact,
    select_pair,
)
from instruction_granularity.ir import parse_module_text
from instruction_granularity.models import (
    ExtractionLevel,
    ArtifactRef,
    SelectionDecision,
    TransitionFeature,
)


BASE = r'''
define i32 @f(i32 %x) #0 {
entry:
  %v = add i32 %x, 1
  br label %exit
exit:
  ret i32 %v
}
attributes #0 = { nounwind }
'''


def _transition(*tokens: str, wildcard: str = "") -> TransitionFeature:
    return TransitionFeature(
        functions=frozenset({"f"}),
        blocks=frozenset({"f::entry"}),
        effect_tokens=frozenset(),
        instruction_tokens=frozenset(
            {("f", "entry", "compute", token) for token in tokens}
        ),
        wildcard_reasons=(wildcard,) if wildcard else (),
    )


def test_instruction_selection_requires_nonempty_disjoint_and_no_wildcard() -> None:
    assert select_pair(_transition("a"), _transition("b"), ExtractionLevel.INSTRUCTION_ONLY).status == "selected"
    assert select_pair(_transition("a"), _transition("a"), ExtractionLevel.INSTRUCTION_ONLY).status == "not_selected"
    assert select_pair(_transition(), _transition("b"), ExtractionLevel.INSTRUCTION_ONLY).status == "not_selected"
    assert select_pair(_transition("a", wildcard="parse_failed"), _transition("b"), ExtractionLevel.INSTRUCTION_ONLY).status == "unknown"


def test_function_header_and_referenced_attribute_changes_fail_closed() -> None:
    changed_header = BASE.replace("define i32 @f(i32 %x) #0", "define internal i32 @f(i32 %x) #0")
    changed_attr = BASE.replace("{ nounwind }", "{ nounwind noinline }")
    before = parse_module_text(BASE, ExtractionLevel.INSTRUCTION_ONLY)

    header_transition = compare_modules(
        before,
        parse_module_text(changed_header, ExtractionLevel.INSTRUCTION_ONLY),
        ExtractionLevel.INSTRUCTION_ONLY,
    )
    attr_transition = compare_modules(
        before,
        parse_module_text(changed_attr, ExtractionLevel.INSTRUCTION_ONLY),
        ExtractionLevel.INSTRUCTION_ONLY,
    )

    assert "function_header_changed" in header_transition.wildcard_reasons
    assert "function_attribute_changed" in attr_transition.wildcard_reasons


def test_function_block_and_cfg_instability_fail_closed() -> None:
    extra_function = BASE + "\ndefine void @g() {\nentry:\n  ret void\n}\n"
    extra_block = BASE.replace("exit:\n", "middle:\n  br label %exit\nexit:\n")
    cfg_change = BASE.replace("br label %exit", "br i1 true, label %exit, label %entry")
    before = parse_module_text(BASE, ExtractionLevel.INSTRUCTION_ONLY)

    reasons = []
    for text in (extra_function, extra_block, cfg_change):
        reasons.append(
            compare_modules(
                before,
                parse_module_text(text, ExtractionLevel.INSTRUCTION_ONLY),
                ExtractionLevel.INSTRUCTION_ONLY,
            ).wildcard_reasons
        )
    assert "functions_added_or_deleted" in reasons[0]
    assert "blocks_added_or_deleted" in reasons[1]
    assert "cfg_unstable" in reasons[2]


def test_malformed_instruction_and_unresolved_label_fail_closed() -> None:
    malformed = BASE.replace("%v = add i32 %x, 1", "%v = call i32 @g(")
    unresolved = BASE.replace("br label %exit", "br label %missing")

    malformed_module = parse_module_text(malformed, ExtractionLevel.INSTRUCTION_ONLY)
    unresolved_module = parse_module_text(unresolved, ExtractionLevel.INSTRUCTION_ONLY)

    assert "logical_instruction_parse_failed" in malformed_module.wildcard_reasons
    assert "unresolved_block_label" in unresolved_module.wildcard_reasons


def test_injected_hash_collision_fails_closed() -> None:
    parsed = parse_module_text(
        BASE,
        ExtractionLevel.INSTRUCTION_ONLY,
        hasher=lambda _: "collision",
    )

    assert "instruction_fingerprint_collision" in parsed.wildcard_reasons
    assert parsed.collisions


def test_three_round_merge_requires_identical_known_decisions() -> None:
    selected = SelectionDecision("selected", "instruction_disjoint")
    not_selected = SelectionDecision("not_selected", "instruction_overlap")
    unknown = SelectionDecision("unknown", "artifact_hash_mismatch")

    assert merge_three_decisions([selected, selected, selected]).status == "selected"
    assert merge_three_decisions([not_selected] * 3).status == "not_selected"
    assert merge_three_decisions([selected, not_selected, selected]).status == "unknown"
    assert merge_three_decisions([selected, unknown, selected]).status == "unknown"


def test_missing_size_hash_and_cross_round_artifact_errors_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "a.ll"
    path.write_text(BASE, encoding="utf-8")
    wrong = ArtifactRef(
        path=path,
        expected_size=path.stat().st_size + 1,
        expected_sha256="0" * 64,
        consistency_errors=("hard_hash_unstable_across_sources",),
    )
    result = read_validated_artifact(wrong)
    assert result.text is None
    assert "artifact_size_mismatch" in result.error
    assert "artifact_hash_mismatch" in result.error
    assert "hard_hash_unstable_across_sources" in result.error

    missing = read_validated_artifact(
        ArtifactRef(
            path=tmp_path / "missing.ll",
            expected_size=1,
            expected_sha256="0" * 64,
        )
    )
    assert missing.text is None
    assert "artifact_read_failed" in missing.error
