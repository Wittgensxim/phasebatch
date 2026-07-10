from __future__ import annotations

import hashlib
import re
from pathlib import Path


LABEL_RE = re.compile(r"^([A-Za-z$._-][\w$._-]*|\d+):")


def normalize_ir_text(text: str) -> str:
    lines: list[str] = []
    previous_blank = False
    for raw_line in text.splitlines():
        line = re.sub(r"\s*,?\s*!dbg !\d+", "", raw_line.rstrip())
        stripped = line.strip()

        if stripped.startswith("; ModuleID =") or stripped.startswith("source_filename ="):
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


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_hash(path: Path) -> str:
    return hash_text(normalize_ir_text(Path(path).read_text(encoding="utf-8")))


def count_ir_features(path: Path) -> dict[str, int]:
    text = normalize_ir_text(Path(path).read_text(encoding="utf-8"))
    counts = {
        "functions": 0,
        "basic_blocks": 0,
        "instructions": 0,
        "branches": 0,
        "loads": 0,
        "stores": 0,
        "calls": 0,
        "direct_calls": 0,
        "intrinsic_calls": 0,
        "indirect_calls": 0,
        "phis": 0,
        "selects": 0,
        "allocas": 0,
    }
    in_function = False
    saw_label = False
    implicit_entry = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("define ") and "@" in stripped:
            counts["functions"] += 1
            in_function = True
            saw_label = False
            implicit_entry = False
            continue
        if in_function and stripped == "}":
            in_function = False
            continue
        if not in_function or not stripped:
            continue
        if LABEL_RE.match(stripped):
            counts["basic_blocks"] += 1
            saw_label = True
            continue
        if stripped.startswith(("declare ", ";")):
            continue

        if not saw_label and not implicit_entry:
            counts["basic_blocks"] += 1
            implicit_entry = True
        counts["instructions"] += 1

        opcode = _opcode(stripped)
        if opcode == "br":
            counts["branches"] += 1
        elif opcode == "load":
            counts["loads"] += 1
        elif opcode == "store":
            counts["stores"] += 1
        elif opcode == "call":
            counts["calls"] += 1
            target = _direct_call_target(stripped)
            if target is None:
                counts["indirect_calls"] += 1
            elif target.startswith("llvm."):
                counts["intrinsic_calls"] += 1
            else:
                counts["direct_calls"] += 1
        elif opcode == "phi":
            counts["phis"] += 1
        elif opcode == "select":
            counts["selects"] += 1
        elif opcode == "alloca":
            counts["allocas"] += 1
    return counts


def _opcode(instruction: str) -> str:
    body = instruction
    if " = " in body:
        body = body.split(" = ", 1)[1].strip()
    parts = body.split()
    while parts and parts[0] in {"tail", "musttail", "notail"}:
        parts.pop(0)
    return parts[0] if parts else ""


def _direct_call_target(instruction: str) -> str | None:
    match = re.search(r"\bcall\b.*?@(?P<name>\"[^\"]+\"|[-A-Za-z$._0-9]+)\s*\(", instruction)
    if not match:
        return None
    return match.group("name").strip('"')
