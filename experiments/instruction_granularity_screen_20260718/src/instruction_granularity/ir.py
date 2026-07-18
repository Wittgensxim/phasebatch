from __future__ import annotations

from collections import Counter
import re
from typing import Iterable

from .deterministic_io import sha256_text
from .models import (
    EffectKey,
    ExtractionLevel,
    ExtractionTrace,
    FingerprintCollision,
    Hasher,
    LEVEL_RANK,
    ParsedModule,
)


MEMORY_INTRINSIC_PREFIXES = (
    "llvm.memcpy.",
    "llvm.memmove.",
    "llvm.memset.",
    "llvm.masked.load.",
    "llvm.masked.store.",
    "llvm.masked.gather.",
    "llvm.masked.scatter.",
    "llvm.vp.load.",
    "llvm.vp.store.",
    "llvm.vp.gather.",
    "llvm.vp.scatter.",
    "llvm.prefetch",
    "llvm.lifetime.start.",
    "llvm.lifetime.end.",
    "llvm.invariant.start.",
    "llvm.invariant.end.",
    "llvm.experimental.noalias.scope.decl",
)

_FUNCTION_RE = re.compile(
    r'^define\b.*?@(?P<name>"(?:\\.|[^"\\])*"|[-A-Za-z$._0-9]+)\s*\('
)
_LABEL_RE = re.compile(r"^([A-Za-z$._-][\w$._-]*|\d+):")
_QUOTED_LABEL_RE = re.compile(r'^(?P<name>"(?:\\.|[^"\\])*"):\s*$')
_LOCAL_RE = re.compile(r'%(?P<name>"(?:\\.|[^"\\])*"|[-A-Za-z$._0-9]+)')
_RESULT_RE = re.compile(
    r'^%(?P<name>"(?:\\.|[^"\\])*"|[-A-Za-z$._0-9]+)\s*=\s*(?P<body>.+)$'
)
_LABEL_REFERENCE_RE = re.compile(
    r'\blabel\s+%(?P<name>"(?:\\.|[^"\\])*"|[-A-Za-z$._0-9]+)'
)
_ATTRIBUTE_RE = re.compile(r"^attributes\s+#(?P<id>\d+)\s*=")
_ATTRIBUTE_REFERENCE_RE = re.compile(r"#(?P<id>\d+)(?=\s*(?:\{|$))")
_DIRECT_TARGET_RE = re.compile(
    r'@(?P<name>"(?:\\.|[^"\\])*"|[-A-Za-z$._0-9]+)\s*\('
)
_DEBUG_ATTACHMENT_RE = re.compile(r"\s*,?\s*!dbg !\d+")
_METADATA_DEFINITION_RE = re.compile(r"^!(\d+)\s*=")
_METADATA_REFERENCE_RE = re.compile(r"!(\d+)")

_PHI_OPCODES = frozenset({"phi"})
_CFG_OPCODES = frozenset(
    {
        "br",
        "switch",
        "indirectbr",
        "ret",
        "resume",
        "catchswitch",
        "catchret",
        "cleanupret",
        "unreachable",
    }
)
_MEMORY_OPCODES = frozenset(
    {"alloca", "load", "store", "fence", "atomicrmw", "cmpxchg"}
)
_CALL_OPCODES = frozenset({"call", "invoke", "callbr"})
_COMPUTE_OPCODES = frozenset(
    {
        "fneg",
        "add",
        "fadd",
        "sub",
        "fsub",
        "mul",
        "fmul",
        "udiv",
        "sdiv",
        "fdiv",
        "urem",
        "srem",
        "frem",
        "shl",
        "lshr",
        "ashr",
        "and",
        "or",
        "xor",
        "extractelement",
        "insertelement",
        "shufflevector",
        "extractvalue",
        "insertvalue",
        "trunc",
        "zext",
        "sext",
        "fptrunc",
        "fpext",
        "fptoui",
        "fptosi",
        "uitofp",
        "sitofp",
        "ptrtoint",
        "inttoptr",
        "bitcast",
        "addrspacecast",
        "icmp",
        "fcmp",
        "select",
        "freeze",
        "getelementptr",
    }
)
_TAIL_PREFIXES = frozenset({"tail", "musttail", "notail"})
_OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}", "<": ">"}
_CLOSERS = frozenset(_OPEN_TO_CLOSE.values())


def parse_module_text(
    text: str,
    level: ExtractionLevel,
    *,
    trace: ExtractionTrace | None = None,
    hasher: Hasher | None = None,
) -> ParsedModule:
    """Parse one module while paying only for the requested cumulative level."""

    level = ExtractionLevel(level)
    trace = trace if trace is not None else ExtractionTrace()
    hasher = hasher or sha256_text
    trace.function_builds += 1

    legacy_text = normalize_legacy_ir(text)
    legacy_functions, legacy_closed = _function_lines(legacy_text)
    functions = {
        name: sha256_text("\n".join(lines) + "\n")
        for name, lines in legacy_functions.items()
    }
    reasons: set[str] = set()
    if not legacy_closed:
        reasons.add("function_parse_failed")
    parsed = ParsedModule(
        level=level,
        text_sha256=sha256_text(text),
        functions=dict(sorted(functions.items())),
        wildcard_reasons=tuple(sorted(reasons)),
    )
    if LEVEL_RANK[level] < LEVEL_RANK[ExtractionLevel.BLOCK_ONLY]:
        return parsed

    trace.block_builds += 1
    blocks: dict[str, str] = {}
    for function_name, lines in legacy_functions.items():
        blocks.update(_legacy_blocks(function_name, lines))
    parsed.blocks = dict(sorted(blocks.items()))
    if LEVEL_RANK[level] < LEVEL_RANK[ExtractionLevel.EFFECT_ONLY]:
        return parsed

    trace.effect_builds += 1
    hard_text = debug_insensitive_ir(text)
    hard_functions, hard_closed = _function_lines(hard_text)
    if not hard_closed:
        reasons.add("function_parse_failed")
    parsed.function_headers = {
        name: lines[0].strip() for name, lines in hard_functions.items() if lines
    }
    parsed.attribute_groups = _attribute_groups(hard_text)
    parsed.function_attribute_references = {
        name: frozenset(
            match.group("id")
            for match in _ATTRIBUTE_REFERENCE_RE.finditer(lines[0].strip())
        )
        for name, lines in hard_functions.items()
        if lines
    }
    parsed.module_structure = _module_structure(hard_text)

    effect_slices: dict[EffectKey, str] = {}
    instruction_counters: dict[EffectKey, Counter[str]] = {}
    canonical_instructions: dict[EffectKey, tuple[str, ...]] = {}
    cfg_signatures: dict[tuple[str, str], tuple[str, tuple[str, ...]]] = {}
    opcodes: set[str] = set()
    logical_count = 0
    collision_forms: dict[str, set[str]] = {}
    collision_locations: dict[str, set[EffectKey]] = {}

    if LEVEL_RANK[level] >= LEVEL_RANK[ExtractionLevel.INSTRUCTION_ONLY]:
        trace.instruction_builds += 1

    for function_name, function_lines in hard_functions.items():
        block_rows = _effect_blocks(function_lines)
        declared_labels = {name for name, _ in block_rows}
        block_map = {
            name: f"%bb{index}" for index, (name, _) in enumerate(block_rows)
        }
        logical_by_block: dict[str, list[str]] = {}
        for block_name, physical_lines in block_rows:
            logical, valid = logical_instructions(physical_lines)
            logical_by_block[block_name] = logical
            logical_count += len(logical)
            if not valid:
                reasons.add("logical_instruction_parse_failed")
            for instruction in logical:
                references = _label_references(instruction)
                if not references.issubset(declared_labels):
                    reasons.add("unresolved_block_label")

        argument_map = _argument_alpha_map(function_lines[0] if function_lines else "")
        value_map = _value_alpha_map(logical_by_block)
        for block_name, _ in block_rows:
            logical = logical_by_block[block_name]
            by_effect: dict[str, list[str]] = {}
            canonical_by_effect: dict[str, list[str]] = {}
            terminator_opcode = ""
            successor_labels: tuple[str, ...] = ()
            for instruction in logical:
                opcode = instruction_opcode(instruction)
                if not opcode:
                    reasons.add("logical_instruction_parse_failed")
                    continue
                opcodes.add(opcode)
                effect_class = classify_instruction(instruction)
                by_effect.setdefault(effect_class, []).append(instruction)
                if opcode in _CFG_OPCODES:
                    terminator_opcode = opcode
                    successor_labels = tuple(
                        block_map[name]
                        for name in _label_references_in_order(instruction)
                        if name in block_map
                    )
                if LEVEL_RANK[level] >= LEVEL_RANK[ExtractionLevel.INSTRUCTION_ONLY]:
                    canonical = canonicalize_instruction(
                        instruction,
                        argument_map=argument_map,
                        value_map=value_map,
                        block_map=block_map,
                    )
                    canonical_by_effect.setdefault(effect_class, []).append(canonical)

            qualified = f"{function_name}::{block_name}"
            if qualified not in parsed.blocks:
                reasons.add("block_correspondence_failed")
                continue
            cfg_signatures[(function_name, block_name)] = (
                terminator_opcode,
                successor_labels,
            )
            for effect_class, values in by_effect.items():
                key = (function_name, block_name, effect_class)
                effect_slices[key] = sha256_text("\n".join(values) + "\n")
            for effect_class, values in canonical_by_effect.items():
                key = (function_name, block_name, effect_class)
                canonical_instructions[key] = tuple(values)
                counter: Counter[str] = Counter()
                for canonical in values:
                    fingerprint = hasher(canonical)
                    counter[fingerprint] += 1
                    collision_forms.setdefault(fingerprint, set()).add(canonical)
                    collision_locations.setdefault(fingerprint, set()).add(key)
                instruction_counters[key] = counter

    collisions: list[FingerprintCollision] = []
    for fingerprint in sorted(collision_forms):
        forms = tuple(sorted(collision_forms[fingerprint]))
        if len(forms) < 2:
            continue
        reasons.add("instruction_fingerprint_collision")
        locations = sorted(collision_locations[fingerprint])
        location = locations[0] if locations else ("", "", "")
        collisions.append(
            FingerprintCollision(
                fingerprint=fingerprint,
                canonical_forms=forms,
                function=location[0],
                block=location[1],
                effect_class=location[2],
            )
        )

    parsed.effect_slices = dict(sorted(effect_slices.items()))
    parsed.instruction_counters = dict(sorted(instruction_counters.items()))
    parsed.canonical_instructions = dict(sorted(canonical_instructions.items()))
    parsed.cfg_signatures = dict(sorted(cfg_signatures.items()))
    parsed.wildcard_reasons = tuple(sorted(reasons))
    parsed.opcodes = frozenset(opcodes)
    parsed.collisions = tuple(collisions)
    parsed.logical_instruction_count = logical_count
    return parsed


def normalize_legacy_ir(text: str) -> str:
    """Exact lossy normalization used by the frozen function/block study."""

    lines: list[str] = []
    previous_blank = False
    for raw_line in text.splitlines():
        line = re.sub(r"\s*,?\s*!dbg !\d+", "", raw_line.rstrip())
        stripped = line.strip()
        if stripped.startswith("; ModuleID =") or stripped.startswith(
            "source_filename ="
        ):
            continue
        if stripped.startswith("!") or stripped.startswith("attributes #"):
            continue
        if stripped.startswith(";") and not stripped.startswith("; <label>"):
            continue
        if " ;" in line and not stripped.startswith("; <label>"):
            line = line.split(" ;", 1)[0].rstrip()
        stripped = line.strip()
        if not stripped:
            if previous_blank:
                continue
            previous_blank = True
            lines.append("")
            continue
        previous_blank = False
        lines.append(line.rstrip())
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + ("\n" if lines else "")


def debug_insensitive_ir(text: str) -> str:
    debug_metadata_ids = _debug_only_metadata_ids(text)
    lines: list[str] = []
    previous_blank = False
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if _is_debug_metadata_line(stripped, debug_metadata_ids):
            continue
        line = _strip_debug_attachment(raw_line.rstrip())
        line = _strip_ir_comment(line).rstrip()
        if not line.strip():
            if not previous_blank:
                lines.append("")
            previous_blank = True
            continue
        previous_blank = False
        lines.append(line)
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + ("\n" if lines else "")


def logical_instructions(lines: Iterable[str]) -> tuple[list[str], bool]:
    logical: list[str] = []
    current: list[str] = []
    stack: list[str] = []
    valid = True
    for raw_line in lines:
        clean, stack, line_valid = _scan_delimiters(raw_line.strip(), stack)
        valid = valid and line_valid
        if clean:
            current.append(clean)
        continuation = bool(stack) or bool(clean and clean.rstrip().endswith(","))
        if current and not continuation and line_valid:
            logical.append(" ".join(current))
            current = []
    if current or stack:
        valid = False
    return logical, valid


def classify_instruction(instruction: str) -> str:
    opcode = instruction_opcode(instruction)
    if opcode in _PHI_OPCODES:
        return "phi"
    if opcode in _CFG_OPCODES:
        return "cfg"
    if opcode in _MEMORY_OPCODES:
        return "memory"
    if opcode in _CALL_OPCODES and _is_memory_intrinsic(instruction):
        return "memory"
    if opcode in _CALL_OPCODES:
        return "call"
    if opcode in _COMPUTE_OPCODES:
        return "compute"
    return "other"


def instruction_opcode(instruction: str) -> str:
    body = instruction.strip()
    match = _RESULT_RE.match(body)
    if match:
        body = match.group("body").lstrip()
    parts = body.split()
    while parts and parts[0] in _TAIL_PREFIXES:
        parts.pop(0)
    return parts[0].lower() if parts else ""


def canonicalize_instruction(
    instruction: str,
    *,
    argument_map: dict[str, str],
    value_map: dict[str, str],
    block_map: dict[str, str],
) -> str:
    match = _RESULT_RE.match(instruction.strip())
    body = match.group("body") if match else instruction.strip()

    def replace_local(local_match: re.Match[str]) -> str:
        encoded = local_match.group("name")
        decoded = _decode_identifier(encoded)
        if decoded in block_map:
            return block_map[decoded]
        if decoded in argument_map:
            return argument_map[decoded]
        if decoded in value_map:
            return value_map[decoded]
        return local_match.group(0)

    replaced = _LOCAL_RE.sub(replace_local, body)
    return _normalize_instruction_whitespace(replaced)


def changed_counter_fingerprints(
    before: Counter[str], after: Counter[str]
) -> frozenset[str]:
    return frozenset(
        fingerprint
        for fingerprint in set(before) | set(after)
        if before.get(fingerprint, 0) != after.get(fingerprint, 0)
    )


def _function_lines(text: str) -> tuple[dict[str, list[str]], bool]:
    result: dict[str, list[str]] = {}
    lines = text.splitlines()
    index = 0
    all_closed = True
    while index < len(lines):
        match = _FUNCTION_RE.match(lines[index].strip())
        if not match:
            index += 1
            continue
        name = _decode_identifier(match.group("name"))
        values = [lines[index]]
        index += 1
        closed = False
        while index < len(lines):
            values.append(lines[index])
            index += 1
            if values[-1].strip() == "}":
                closed = True
                break
        all_closed = all_closed and closed
        result[name] = values
    return result, all_closed


def _legacy_blocks(function_name: str, function_lines: list[str]) -> dict[str, str]:
    blocks: dict[str, str] = {}
    current_label = "entry"
    current_lines: list[str] = []
    for line in function_lines[1:]:
        stripped = line.strip()
        if stripped == "}":
            break
        label_name = _label_definition(stripped)
        if label_name is not None:
            if current_lines:
                blocks[f"{function_name}::{current_label}"] = sha256_text(
                    "\n".join(current_lines) + "\n"
                )
            current_label = label_name
            current_lines = [line]
            continue
        if stripped:
            current_lines.append(line)
    if current_lines:
        blocks[f"{function_name}::{current_label}"] = sha256_text(
            "\n".join(current_lines) + "\n"
        )
    return blocks


def _effect_blocks(function_lines: list[str]) -> list[tuple[str, list[str]]]:
    result: list[tuple[str, list[str]]] = []
    current_label = "entry"
    current_lines: list[str] = []
    saw_content = False
    for line in function_lines[1:]:
        stripped = line.strip()
        if stripped == "}":
            break
        label = _label_definition(stripped)
        if label is not None:
            if current_lines or saw_content:
                result.append((current_label, current_lines))
            current_label = label
            current_lines = []
            saw_content = False
            continue
        if stripped:
            current_lines.append(line)
            saw_content = True
    if current_lines or saw_content:
        result.append((current_label, current_lines))
    return result


def _argument_alpha_map(header: str) -> dict[str, str]:
    match = _FUNCTION_RE.match(header.strip())
    if not match:
        return {}
    open_index = header.find("(", match.start())
    close_index = _matching_close(header, open_index, "(", ")")
    if open_index < 0 or close_index < 0:
        return {}
    arguments = _split_top_level(header[open_index + 1 : close_index], ",")
    result: dict[str, str] = {}
    for part in arguments:
        matches = list(_LOCAL_RE.finditer(part))
        if not matches:
            continue
        candidate = matches[-1]
        suffix = part[candidate.end() :].strip()
        if suffix:
            continue
        name = _decode_identifier(candidate.group("name"))
        result.setdefault(name, f"%arg{len(result)}")
    return result


def _value_alpha_map(logical_by_block: dict[str, list[str]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for instructions in logical_by_block.values():
        for instruction in instructions:
            match = _RESULT_RE.match(instruction.strip())
            if not match:
                continue
            name = _decode_identifier(match.group("name"))
            result.setdefault(name, f"%v{len(result)}")
    return result


def _label_definition(line: str) -> str | None:
    match = _LABEL_RE.match(line)
    if match:
        return match.group(1)
    quoted = _QUOTED_LABEL_RE.match(line)
    if quoted:
        return _decode_identifier(quoted.group("name"))
    return None


def _label_references(instruction: str) -> frozenset[str]:
    return frozenset(_label_references_in_order(instruction))


def _label_references_in_order(instruction: str) -> tuple[str, ...]:
    return tuple(
        _decode_identifier(match.group("name"))
        for match in _LABEL_REFERENCE_RE.finditer(instruction)
    )


def _scan_delimiters(
    line: str, initial_stack: list[str]
) -> tuple[str, list[str], bool]:
    stack = list(initial_stack)
    output: list[str] = []
    quoted = False
    escaped = False
    valid = True
    for character in line:
        if quoted:
            output.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == ";":
            break
        output.append(character)
        if character == '"':
            quoted = True
        elif character in _OPEN_TO_CLOSE:
            stack.append(_OPEN_TO_CLOSE[character])
        elif character in _CLOSERS:
            if not stack or stack.pop() != character:
                valid = False
                stack.clear()
    if quoted:
        valid = False
    return "".join(output).strip(), stack, valid


def _is_memory_intrinsic(instruction: str) -> bool:
    match = _DIRECT_TARGET_RE.search(instruction)
    if not match:
        return False
    target = _decode_identifier(match.group("name"))
    return any(target.startswith(prefix) for prefix in MEMORY_INTRINSIC_PREFIXES)


def _attribute_groups(text: str) -> dict[str, str]:
    return {
        match.group("id"): line.strip()
        for line in text.splitlines()
        if (match := _ATTRIBUTE_RE.match(line.strip()))
    }


def _module_structure(text: str) -> str:
    values: list[str] = []
    in_function = False
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not in_function and _FUNCTION_RE.match(stripped):
            in_function = True
            continue
        if in_function:
            if stripped == "}":
                in_function = False
            continue
        if not stripped or _ATTRIBUTE_RE.match(stripped):
            continue
        values.append(stripped)
    return "\n".join(values) + ("\n" if values else "")


def _normalize_instruction_whitespace(text: str) -> str:
    pieces: list[str] = []
    start = 0
    index = 0
    while index < len(text):
        if text[index] != '"':
            index += 1
            continue
        pieces.append(_normalize_unquoted(text[start:index]))
        quote_start = index
        index += 1
        escaped = False
        while index < len(text):
            if escaped:
                escaped = False
            elif text[index] == "\\":
                escaped = True
            elif text[index] == '"':
                index += 1
                break
            index += 1
        pieces.append(text[quote_start:index])
        start = index
    pieces.append(_normalize_unquoted(text[start:]))
    return "".join(pieces).strip()


def _normalize_unquoted(text: str) -> str:
    value = re.sub(r"\s+", " ", text)
    value = re.sub(r"\s*,\s*", ", ", value)
    value = re.sub(r"\s*([\(\[\{])\s*", r"\1", value)
    value = re.sub(r"\s*([\)\]\}])", r"\1", value)
    return value


def _matching_close(text: str, start: int, opener: str, closer: str) -> int:
    if start < 0 or start >= len(text) or text[start] != opener:
        return -1
    depth = 0
    quoted = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character == opener:
            depth += 1
        elif character == closer:
            depth -= 1
            if depth == 0:
                return index
    return -1


def _split_top_level(text: str, delimiter: str) -> list[str]:
    result: list[str] = []
    start = 0
    stack: list[str] = []
    quoted = False
    escaped = False
    for index, character in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character in _OPEN_TO_CLOSE:
            stack.append(_OPEN_TO_CLOSE[character])
        elif character in _CLOSERS and stack and stack[-1] == character:
            stack.pop()
        elif character == delimiter and not stack:
            result.append(text[start:index].strip())
            start = index + 1
    result.append(text[start:].strip())
    return [value for value in result if value]


def _decode_identifier(value: str) -> str:
    if value.startswith('"') and value.endswith('"'):
        inner = value[1:-1]
        return re.sub(
            r"\\([0-9A-Fa-f]{2})",
            lambda match: chr(int(match.group(1), 16)),
            inner,
        )
    return value


def _strip_debug_attachment(line: str) -> str:
    return _rewrite_unquoted(line, lambda segment: _DEBUG_ATTACHMENT_RE.sub("", segment))


def _strip_ir_comment(line: str) -> str:
    quoted = False
    escaped = False
    for index, character in enumerate(line):
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character == ";":
            return line[:index]
    return line


def _rewrite_unquoted(text: str, rewrite) -> str:  # noqa: ANN001
    pieces: list[str] = []
    segment_start = 0
    index = 0
    while index < len(text):
        if text[index] != '"':
            index += 1
            continue
        pieces.append(rewrite(text[segment_start:index]))
        quote_start = index
        index += 1
        escaped = False
        while index < len(text):
            if escaped:
                escaped = False
            elif text[index] == "\\":
                escaped = True
            elif text[index] == '"':
                index += 1
                break
            index += 1
        pieces.append(text[quote_start:index])
        segment_start = index
    pieces.append(rewrite(text[segment_start:]))
    return "".join(pieces)


def _unquoted_text(text: str) -> str:
    return _rewrite_unquoted(text, lambda value: value)


def _is_debug_metadata_line(stripped: str, debug_ids: set[str]) -> bool:
    if stripped.startswith("!llvm.dbg"):
        return True
    match = _METADATA_DEFINITION_RE.match(stripped)
    return bool(match and match.group(1) in debug_ids)


def _debug_only_metadata_ids(text: str) -> set[str]:
    definitions: dict[str, str] = {}
    debug_roots: set[str] = set()
    non_debug_roots: set[str] = set()
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        definition = _METADATA_DEFINITION_RE.match(stripped)
        if definition:
            metadata_id = definition.group(1)
            definitions[metadata_id] = stripped
            unquoted = _unquoted_text(stripped)
            if any(
                token in unquoted
                for token in ("!DI", "DICompileUnit", "DISubprogram", "DILocation")
            ):
                debug_roots.add(metadata_id)
            continue
        references = set(_METADATA_REFERENCE_RE.findall(stripped))
        if stripped.startswith("!llvm.dbg"):
            debug_roots.update(references)
            continue
        debug_roots.update(re.findall(r"!dbg\s+!(\d+)", _unquoted_text(stripped)))
        retained = _strip_debug_attachment(stripped)
        non_debug_roots.update(_METADATA_REFERENCE_RE.findall(retained))

    def closure(roots: set[str]) -> set[str]:
        reached: set[str] = set()
        pending = list(roots)
        while pending:
            metadata_id = pending.pop()
            if metadata_id in reached:
                continue
            reached.add(metadata_id)
            definition = definitions.get(metadata_id, "")
            pending.extend(
                reference
                for reference in _METADATA_REFERENCE_RE.findall(definition)
                if reference != metadata_id and reference not in reached
            )
        return reached

    return closure(debug_roots) - closure(non_debug_roots)

