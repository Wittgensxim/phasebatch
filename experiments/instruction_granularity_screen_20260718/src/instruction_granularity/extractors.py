from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import time

from .ir import changed_counter_fingerprints, parse_module_text
from .models import (
    ArtifactRead,
    ArtifactRef,
    ExtractionRun,
    ExtractionLevel,
    ExtractionTrace,
    FrozenDataset,
    LEVEL_RANK,
    ParsedModule,
    SelectionDecision,
    TransitionFeature,
)


def compare_modules(
    before: ParsedModule,
    after: ParsedModule,
    level: ExtractionLevel,
) -> TransitionFeature:
    level = ExtractionLevel(level)
    function_names = set(before.functions) | set(after.functions)
    functions = frozenset(
        name
        for name in function_names
        if before.functions.get(name) != after.functions.get(name)
    )
    if LEVEL_RANK[level] < LEVEL_RANK[ExtractionLevel.BLOCK_ONLY]:
        return TransitionFeature(
            functions=functions,
            blocks=frozenset(),
            effect_tokens=frozenset(),
            instruction_tokens=frozenset(),
        )

    block_names = set(before.blocks) | set(after.blocks)
    blocks = frozenset(
        name
        for name in block_names
        if before.blocks.get(name) != after.blocks.get(name)
    )
    if LEVEL_RANK[level] < LEVEL_RANK[ExtractionLevel.EFFECT_ONLY]:
        return TransitionFeature(
            functions=functions,
            blocks=blocks,
            effect_tokens=frozenset(),
            instruction_tokens=frozenset(),
        )

    reasons = set(before.wildcard_reasons) | set(after.wildcard_reasons)
    left_functions = set(before.functions)
    right_functions = set(after.functions)
    if left_functions != right_functions:
        reasons.add("functions_added_or_deleted")
    left_blocks = set(before.blocks)
    right_blocks = set(after.blocks)
    if left_blocks != right_blocks:
        reasons.add("blocks_added_or_deleted")

    shared_functions = left_functions & right_functions
    if any(
        before.function_headers.get(name) != after.function_headers.get(name)
        for name in shared_functions
    ):
        reasons.add("function_header_changed")

    referenced_attributes = {
        attribute_id
        for name in shared_functions
        for attribute_id in (
            before.function_attribute_references.get(name, frozenset())
            | after.function_attribute_references.get(name, frozenset())
        )
    }
    changed_attributes = {
        attribute_id
        for attribute_id in set(before.attribute_groups) | set(after.attribute_groups)
        if before.attribute_groups.get(attribute_id)
        != after.attribute_groups.get(attribute_id)
    }
    if referenced_attributes & changed_attributes:
        reasons.add("function_attribute_changed")
    if (
        before.module_structure != after.module_structure
        or changed_attributes - referenced_attributes
    ):
        reasons.add("module_structure_changed")

    effect_tokens = frozenset()
    if not reasons:
        effect_tokens = frozenset(
            key
            for key in set(before.effect_slices) | set(after.effect_slices)
            if before.effect_slices.get(key) != after.effect_slices.get(key)
        )
    if LEVEL_RANK[level] < LEVEL_RANK[ExtractionLevel.INSTRUCTION_ONLY]:
        return TransitionFeature(
            functions=functions,
            blocks=blocks,
            effect_tokens=effect_tokens,
            instruction_tokens=frozenset(),
            wildcard_reasons=tuple(sorted(reasons)),
            observed_opcodes=before.opcodes | after.opcodes,
        )

    if before.cfg_signatures != after.cfg_signatures:
        reasons.add("cfg_unstable")
    instruction_tokens: set[tuple[str, str, str, str]] = set()
    if not reasons:
        empty: Counter[str] = Counter()
        for key in set(before.instruction_counters) | set(after.instruction_counters):
            changed = changed_counter_fingerprints(
                before.instruction_counters.get(key, empty),
                after.instruction_counters.get(key, empty),
            )
            instruction_tokens.update((*key, fingerprint) for fingerprint in changed)
    return TransitionFeature(
        functions=functions,
        blocks=blocks,
        effect_tokens=effect_tokens if not reasons else frozenset(),
        instruction_tokens=frozenset(instruction_tokens),
        wildcard_reasons=tuple(sorted(reasons)),
        observed_opcodes=before.opcodes | after.opcodes,
    )


def select_pair(
    left: TransitionFeature,
    right: TransitionFeature,
    level: ExtractionLevel,
) -> SelectionDecision:
    """Apply one cumulative observed-change screen."""

    level = ExtractionLevel(level)
    function_disjoint = bool(
        left.functions
        and right.functions
        and left.functions.isdisjoint(right.functions)
    )
    if function_disjoint:
        return SelectionDecision("selected", "function_disjoint")
    if level == ExtractionLevel.FUNC_ONLY:
        return SelectionDecision("not_selected", "function_overlap_or_empty")

    block_disjoint = bool(
        left.blocks and right.blocks and left.blocks.isdisjoint(right.blocks)
    )
    if block_disjoint:
        return SelectionDecision("selected", "block_disjoint")
    if level == ExtractionLevel.BLOCK_ONLY:
        return SelectionDecision("not_selected", "block_overlap_or_empty")

    effect_disjoint = bool(
        not left.wildcard_reasons
        and not right.wildcard_reasons
        and left.effect_tokens
        and right.effect_tokens
        and left.effect_tokens.isdisjoint(right.effect_tokens)
    )
    if effect_disjoint:
        return SelectionDecision("selected", "effect_disjoint")
    if level == ExtractionLevel.EFFECT_ONLY:
        if left.wildcard_reasons or right.wildcard_reasons:
            return SelectionDecision("unknown", "effect_wildcard")
        return SelectionDecision("not_selected", "effect_overlap_or_empty")

    return select_incremental_instruction(left, right)


def select_incremental_instruction(
    left: TransitionFeature,
    right: TransitionFeature,
) -> SelectionDecision:
    if left.wildcard_reasons or right.wildcard_reasons:
        reasons = sorted(set(left.wildcard_reasons) | set(right.wildcard_reasons))
        return SelectionDecision("unknown", ";".join(reasons) or "instruction_wildcard")
    if not left.instruction_tokens or not right.instruction_tokens:
        return SelectionDecision("not_selected", "instruction_tokens_empty")
    if not left.instruction_tokens.isdisjoint(right.instruction_tokens):
        return SelectionDecision("not_selected", "instruction_overlap")
    return SelectionDecision("selected", "instruction_disjoint")


def extraction_error_feature(reason: str) -> TransitionFeature:
    return TransitionFeature(
        functions=frozenset(),
        blocks=frozenset(),
        effect_tokens=frozenset(),
        instruction_tokens=frozenset(),
        wildcard_reasons=(reason,),
    )


def read_validated_artifact(reference: ArtifactRef) -> ArtifactRead:
    """Read a frozen IR file with pre/post stat and completion hash checks."""

    errors = list(reference.consistency_errors)
    path = Path(reference.path)
    try:
        before = path.stat()
        if not path.is_file():
            errors.append("artifact_not_regular_file")
        data = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        errors.append(f"artifact_read_failed:{type(exc).__name__}")
        return ArtifactRead(text=None, error=";".join(sorted(set(errors))))
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_size != len(data)
    ):
        errors.append("artifact_changed_during_read")
    actual_hash = hashlib.sha256(data).hexdigest()
    if reference.expected_size < 0 or len(data) != reference.expected_size:
        errors.append("artifact_size_mismatch")
    if not reference.expected_sha256 or actual_hash != reference.expected_sha256:
        errors.append("artifact_hash_mismatch")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        errors.append("artifact_utf8_decode_failed")
        text = None
    return ArtifactRead(
        text=text if not errors else None,
        error=";".join(sorted(set(errors))),
        actual_size=len(data),
        actual_sha256=actual_hash,
    )


def run_extractor(
    dataset: FrozenDataset,
    source_repetition: int,
    level: ExtractionLevel,
    *,
    clock_ns=time.perf_counter_ns,
) -> ExtractionRun:
    """Run one fully independent extractor over 49/686/1,411 frozen inputs."""

    level = ExtractionLevel(level)
    try:
        attempt = next(
            item for item in dataset.attempts if item.repetition == source_repetition
        )
    except StopIteration as exc:
        raise ValueError(f"unknown source repetition: {source_repetition}") from exc

    total_start = clock_ns()
    artifact_start = total_start
    base_reads: dict[str, ArtifactRead] = {}
    output_reads: dict[tuple[str, str], ArtifactRead] = {}
    errors: list[dict[str, str]] = []
    programs = sorted(dataset.base_artifacts)
    for program in programs:
        read = read_validated_artifact(dataset.base_artifacts[program])
        base_reads[program] = read
        if read.error:
            errors.append(
                {
                    "scope": "base_artifact",
                    "source_repetition": str(source_repetition),
                    "program": program,
                    "action_id": "",
                    "path": str(dataset.base_artifacts[program].path),
                    "reason": read.error,
                }
            )
    for key in sorted(dataset.transition_keys):
        reference = attempt.outputs[key]
        read = read_validated_artifact(reference)
        output_reads[key] = read
        if read.error:
            errors.append(
                {
                    "scope": "single_pass_artifact",
                    "source_repetition": str(source_repetition),
                    "program": key[0],
                    "action_id": key[1],
                    "path": str(reference.path),
                    "reason": read.error,
                }
            )
    artifact_end = clock_ns()

    trace = ExtractionTrace()
    parse_start = artifact_end
    base_modules = {
        program: parse_module_text(read.text, level, trace=trace)
        for program, read in base_reads.items()
        if read.text is not None
    }
    output_modules = {
        key: parse_module_text(read.text, level, trace=trace)
        for key, read in output_reads.items()
        if read.text is not None
    }
    collisions = tuple(
        collision
        for module in (*base_modules.values(), *output_modules.values())
        for collision in module.collisions
    )
    parse_end = clock_ns()

    feature_start = parse_end
    features: dict[tuple[str, str], TransitionFeature] = {}
    for key in sorted(dataset.transition_keys):
        program = key[0]
        if program not in base_modules:
            reason = base_reads[program].error or "base_parse_unavailable"
            features[key] = extraction_error_feature(reason)
        elif key not in output_modules:
            reason = output_reads[key].error or "output_parse_unavailable"
            features[key] = extraction_error_feature(reason)
        else:
            features[key] = compare_modules(
                base_modules[program], output_modules[key], level
            )
    feature_end = clock_ns()

    selection_start = feature_end
    decisions = {
        pair.observation_id: select_pair(
            features[(pair.program, pair.action_a_id)],
            features[(pair.program, pair.action_b_id)],
            level,
        )
        for pair in dataset.pairs
    }
    selection_end = clock_ns()
    return ExtractionRun(
        level=level,
        source_repetition=source_repetition,
        features=features,
        decisions=decisions,
        trace=trace,
        artifact_errors=tuple(errors),
        collisions=collisions,
        artifact_read_ms=(artifact_end - artifact_start) / 1_000_000,
        parse_ms=(parse_end - parse_start) / 1_000_000,
        feature_build_ms=(feature_end - feature_start) / 1_000_000,
        pair_selection_ms=(selection_end - selection_start) / 1_000_000,
        total_extraction_ms=(selection_end - total_start) / 1_000_000,
    )
