from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .normalizer import LABEL_RE, canonical_hash, count_ir_features, hash_text, normalize_ir_text


DEFINE_RE = re.compile(r"^define\b.*@([^(\s]+)\(")


@dataclass
class IRSnapshot:
    module_hash: str
    functions: dict[str, str]
    blocks: dict[str, str]
    features: dict[str, int]


def parse_ir_snapshot(path: Path) -> IRSnapshot:
    path = Path(path)
    lines = normalize_ir_text(path.read_text(encoding="utf-8")).splitlines()
    functions: dict[str, str] = {}
    blocks: dict[str, str] = {}

    index = 0
    while index < len(lines):
        line = lines[index].strip()
        match = DEFINE_RE.match(line)
        if not match:
            index += 1
            continue

        function_name = match.group(1)
        function_lines = [lines[index]]
        index += 1
        while index < len(lines):
            function_lines.append(lines[index])
            if lines[index].strip() == "}":
                index += 1
                break
            index += 1

        functions[function_name] = hash_text("\n".join(function_lines) + "\n")
        blocks.update(_parse_blocks(function_name, function_lines))

    return IRSnapshot(
        module_hash=canonical_hash(path),
        functions=functions,
        blocks=blocks,
        features=count_ir_features(path),
    )


def changed_regions(before: IRSnapshot, after: IRSnapshot) -> dict:
    before_funcs = set(before.functions)
    after_funcs = set(after.functions)
    before_blocks = set(before.blocks)
    after_blocks = set(after.blocks)

    added_functions = sorted(after_funcs - before_funcs)
    deleted_functions = sorted(before_funcs - after_funcs)
    changed_functions = sorted(
        name for name in (before_funcs & after_funcs) if before.functions[name] != after.functions[name]
    )

    added_blocks = sorted(after_blocks - before_blocks)
    deleted_blocks = sorted(before_blocks - after_blocks)
    changed_blocks = sorted(
        name for name in (before_blocks & after_blocks) if before.blocks[name] != after.blocks[name]
    )

    all_changed_functions = sorted(set(changed_functions) | set(added_functions) | set(deleted_functions))
    all_changed_blocks = sorted(set(changed_blocks) | set(added_blocks) | set(deleted_blocks))

    return {
        "changed_functions": all_changed_functions,
        "changed_blocks": all_changed_blocks,
        "added_functions": added_functions,
        "deleted_functions": deleted_functions,
        "added_blocks": added_blocks,
        "deleted_blocks": deleted_blocks,
        "funcs_changed": len(all_changed_functions),
        "blocks_changed": len(all_changed_blocks),
    }


def _parse_blocks(function_name: str, function_lines: list[str]) -> dict[str, str]:
    blocks: dict[str, str] = {}
    current_label = "entry"
    current_lines: list[str] = []
    in_body = False

    for line in function_lines[1:]:
        stripped = line.strip()
        if stripped == "}":
            break
        in_body = True
        label_match = LABEL_RE.match(stripped)
        if label_match:
            if current_lines:
                blocks[f"{function_name}::{current_label}"] = hash_text("\n".join(current_lines) + "\n")
            current_label = label_match.group(1)
            current_lines = [line]
            continue
        if in_body and stripped:
            current_lines.append(line)

    if current_lines:
        blocks[f"{function_name}::{current_label}"] = hash_text("\n".join(current_lines) + "\n")
    return blocks
