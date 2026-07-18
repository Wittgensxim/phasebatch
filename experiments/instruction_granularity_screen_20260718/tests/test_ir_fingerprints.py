from __future__ import annotations

from collections import Counter

from instruction_granularity.extractors import compare_modules
from instruction_granularity.ir import (
    changed_counter_fingerprints,
    parse_module_text,
)
from instruction_granularity.models import ExtractionLevel, ExtractionTrace


MULTILINE_IR = r'''
source_filename = "debug-only.c"
define i32 @f(i32 %input, ptr %ptr) #0 !dbg !9 {
entry:
  %r = call noundef i32 @callee(
      i32 %input,
      ptr @global), !dbg !10
  switch i32 %r, label %done [
    i32 0, label %case
  ], !dbg !11
case:
  %p = phi i32 [ %r, %entry ],
                 [ 7, %case ], !dbg !12
  br label %done
done:
  ret i32 %r
}
attributes #0 = { nounwind }
!llvm.dbg.cu = !{!9}
!9 = distinct !DICompileUnit(language: DW_LANG_C99, file: !13)
!10 = !DILocation(line: 1, column: 1, scope: !9)
!11 = !DILocation(line: 2, column: 1, scope: !9)
!12 = !DILocation(line: 3, column: 1, scope: !9)
!13 = !DIFile(filename: "debug-only.c", directory: ".")
'''


def _canonical_values(module) -> list[str]:  # noqa: ANN001
    return sorted(
        value
        for values in module.canonical_instructions.values()
        for value in values
    )


def test_multiline_call_switch_phi_are_single_logical_instructions() -> None:
    parsed = parse_module_text(MULTILINE_IR, ExtractionLevel.INSTRUCTION_ONLY)

    assert parsed.wildcard_reasons == ()
    assert parsed.logical_instruction_count == 5
    assert parsed.opcodes == frozenset({"br", "call", "phi", "ret", "switch"})
    canonical = "\n".join(_canonical_values(parsed))
    assert canonical.count("call noundef i32 @callee(") == 1
    assert canonical.count("switch i32") == 1
    assert canonical.count("phi i32") == 1


def test_ssa_and_block_labels_are_alpha_normalized_but_semantics_remain() -> None:
    left = r'''
define i32 @f(i32 %arg) {
left:
  %sum = add nsw i32 %arg, 7
  %cmp = icmp eq i32 %sum, 9
  br i1 %cmp, label %yes, label %no
yes:
  ret i32 %sum
no:
  ret i32 0
}
'''
    right = r'''
define i32 @f(i32 %renamed) {
start:
  %tmp9 = add nsw i32 %renamed, 7
  %flag = icmp eq i32 %tmp9, 9
  br i1 %flag, label %take, label %leave
take:
  ret i32 %tmp9
leave:
  ret i32 0
}
'''
    parsed_left = parse_module_text(left, ExtractionLevel.INSTRUCTION_ONLY)
    parsed_right = parse_module_text(right, ExtractionLevel.INSTRUCTION_ONLY)

    assert _canonical_values(parsed_left) == _canonical_values(parsed_right)
    canonical = "\n".join(_canonical_values(parsed_left))
    assert "%sum" not in canonical
    assert "%arg," not in canonical
    assert "%yes" not in canonical
    assert "add nsw i32 %arg0, 7" in canonical
    assert "icmp eq i32 %v0, 9" in canonical
    assert "label %bb1" in canonical


def test_result_ssa_removed_and_globals_callees_constants_flags_preserved() -> None:
    parsed = parse_module_text(
        r'''
@global = external global i32
declare i32 @callee(ptr, i32)
define i32 @f(ptr %p) {
entry:
  %x = load atomic volatile i32, ptr @global acquire, align 4
  %y = call fastcc i32 @callee(ptr %p, i32 42)
  %z = icmp sgt i32 %y, -1
  ret i32 %y
}
''',
        ExtractionLevel.INSTRUCTION_ONLY,
    )
    canonical = "\n".join(_canonical_values(parsed))
    assert "%x =" not in canonical
    assert "%y =" not in canonical
    assert "%z =" not in canonical
    for token in (
        "load atomic volatile i32",
        "@global",
        "acquire",
        "align 4",
        "call fastcc i32 @callee",
        "i32 42",
        "icmp sgt i32",
        "-1",
    ):
        assert token in canonical


def test_debug_only_changes_do_not_change_instruction_fingerprints() -> None:
    without_debug = "\n".join(
        line for line in MULTILINE_IR.splitlines() if not line.lstrip().startswith("!")
    ).replace(", !dbg !10", "").replace(", !dbg !11", "").replace(", !dbg !12", "")
    left = parse_module_text(MULTILINE_IR, ExtractionLevel.INSTRUCTION_ONLY)
    right = parse_module_text(without_debug, ExtractionLevel.INSTRUCTION_ONLY)

    assert _canonical_values(left) == _canonical_values(right)


def test_counter_multiset_detects_count_change() -> None:
    fingerprint = "f" * 64
    before = Counter({fingerprint: 2})
    after = Counter({fingerprint: 1})

    assert changed_counter_fingerprints(before, after) == frozenset({fingerprint})
    assert changed_counter_fingerprints(after, after) == frozenset()


def test_pure_instruction_reorder_is_not_a_change() -> None:
    before = r'''
declare void @a()
declare void @b()
define void @f() {
entry:
  call void @a()
  call void @b()
  ret void
}
'''
    after = r'''
declare void @a()
declare void @b()
define void @f() {
entry:
  call void @b()
  call void @a()
  ret void
}
'''
    left = parse_module_text(before, ExtractionLevel.INSTRUCTION_ONLY)
    right = parse_module_text(after, ExtractionLevel.INSTRUCTION_ONLY)
    transition = compare_modules(left, right, ExtractionLevel.INSTRUCTION_ONLY)

    assert transition.instruction_tokens == frozenset()


def test_layer_trace_counts_only_paid_builders() -> None:
    expected = {
        ExtractionLevel.FUNC_ONLY: (1, 0, 0, 0),
        ExtractionLevel.BLOCK_ONLY: (1, 1, 0, 0),
        ExtractionLevel.EFFECT_ONLY: (1, 1, 1, 0),
        ExtractionLevel.INSTRUCTION_ONLY: (1, 1, 1, 1),
    }
    for level, counts in expected.items():
        trace = ExtractionTrace()
        parse_module_text(MULTILINE_IR, level, trace=trace)
        assert (
            trace.function_builds,
            trace.block_builds,
            trace.effect_builds,
            trace.instruction_builds,
        ) == counts
