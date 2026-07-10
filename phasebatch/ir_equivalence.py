from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import re
import subprocess
from pathlib import Path

from .tools import find_tool


DEBUG_ATTACHMENT_RE = re.compile(r"\s*,?\s*!dbg !\d+")
LABEL_DEFINITION_RE = re.compile(r"^([A-Za-z$._-][\w$._-]*|\d+):")
LOCAL_VALUE_RE = re.compile(r"%(?:[-A-Za-z$._][\w$._-]*|\d+)")


@dataclass(frozen=True)
class EqualityResult:
    equal: bool
    tier: str
    can_hard_fold: bool
    reason: str
    text_hash_equal: bool | None = None
    llvm_diff_equal: bool | None = None
    module_fingerprint_equal: bool | None = None
    left_hash: str = ""
    right_hash: str = ""
    error_message: str = ""


def compare_ir_equivalence(
    left: Path,
    right: Path,
    tools: dict | None = None,
    timeout: int = 10,
) -> EqualityResult:
    left = Path(left)
    right = Path(right)
    try:
        left_hash = safe_canonical_hash(left)
        right_hash = safe_canonical_hash(right)
    except OSError as exc:
        return EqualityResult(
            equal=False,
            tier="failed",
            can_hard_fold=False,
            reason="tool_failed",
            error_message=str(exc),
        )

    text_hash_equal = left_hash == right_hash
    if text_hash_equal:
        return EqualityResult(
            equal=True,
            tier="canonical_hash",
            can_hard_fold=True,
            reason="hash_equal",
            text_hash_equal=True,
            left_hash=left_hash,
            right_hash=right_hash,
        )

    completed = None
    from .opt_backend import active_opt_backend
    from .opt_worker import WorkerError

    backend = active_opt_backend()
    if backend is not None:
        try:
            llvm_diff_equal = backend.compare_paths(left, right, timeout=timeout)
        except WorkerError:
            if not backend.fallback_external:
                raise
            backend = None

    if backend is None:
        llvm_diff = _resolve_llvm_diff_tool(tools or {})
        if not llvm_diff:
            return EqualityResult(
                equal=False,
                tier="failed",
                can_hard_fold=False,
                reason="tool_failed",
                text_hash_equal=False,
                left_hash=left_hash,
                right_hash=right_hash,
                error_message="llvm-diff not found",
            )
        try:
            completed = subprocess.run(
                [llvm_diff, str(left), str(right)],
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return EqualityResult(
                equal=False,
                tier="failed",
                can_hard_fold=False,
                reason="tool_failed",
                text_hash_equal=False,
                left_hash=left_hash,
                right_hash=right_hash,
                error_message=str(exc),
            )
        llvm_diff_equal = completed.returncode == 0
    if not llvm_diff_equal:
        return EqualityResult(
            equal=False,
            tier="different",
            can_hard_fold=False,
            reason="llvm_diff_difference",
            text_hash_equal=False,
            llvm_diff_equal=False,
            left_hash=left_hash,
            right_hash=right_hash,
            error_message=(completed.stdout or completed.stderr or "").strip() if completed is not None else "",
        )

    try:
        left_fingerprint = module_safety_fingerprint(left)
        right_fingerprint = module_safety_fingerprint(right)
    except OSError as exc:
        return EqualityResult(
            equal=False,
            tier="failed",
            can_hard_fold=False,
            reason="tool_failed",
            text_hash_equal=False,
            llvm_diff_equal=True,
            left_hash=left_hash,
            right_hash=right_hash,
            error_message=str(exc),
        )

    module_fingerprint_equal = left_fingerprint == right_fingerprint
    if module_fingerprint_equal:
        return EqualityResult(
            equal=True,
            tier="structural_diff",
            can_hard_fold=True,
            reason="llvm_diff_equal_and_module_fingerprint_equal",
            text_hash_equal=False,
            llvm_diff_equal=True,
            module_fingerprint_equal=True,
            left_hash=left_hash,
            right_hash=right_hash,
        )

    return EqualityResult(
        equal=False,
        tier="different",
        can_hard_fold=False,
        reason="module_fingerprint_difference",
        text_hash_equal=False,
        llvm_diff_equal=True,
        module_fingerprint_equal=False,
        left_hash=left_hash,
        right_hash=right_hash,
    )


def safe_canonical_hash(path: Path) -> str:
    return _hash_text(safe_canonical_text(Path(path).read_text(encoding="utf-8")))


def safe_canonical_text(text: str) -> str:
    lines: list[str] = []
    previous_blank = False
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if _is_debug_metadata_line(stripped):
            continue
        line = _strip_debug_attachment(raw_line.rstrip())
        line = _strip_ir_comment(line).rstrip()
        stripped = line.strip()
        if not stripped:
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


def module_safety_fingerprint(path: Path) -> str:
    text = Path(path).read_text(encoding="utf-8")
    items: list[str] = []
    in_function = False
    current_function = ""
    function_lines: list[str] = []

    for raw_line in text.splitlines():
        line = _strip_debug_attachment(raw_line.rstrip())
        line = _strip_ir_comment(line).rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith(";") or _is_debug_metadata_line(stripped):
            continue

        if in_function:
            if stripped == "}":
                if function_lines:
                    items.append(f"function_body:{current_function}:{_hash_text(chr(10).join(function_lines))}")
                in_function = False
                current_function = ""
                function_lines = []
                continue
            function_lines.append(_normalize_function_fingerprint_line(stripped))
            continue

        if _starts_function_definition(stripped):
            current_function = _function_name(stripped)
            items.append(f"function_signature:{_normalize_function_fingerprint_line(_canonical_signature(stripped))}")
            if stripped.endswith("{"):
                in_function = True
                function_lines = []
            continue

        items.append(f"module:{stripped}")

    if in_function and function_lines:
        items.append(f"function_body:{current_function}:{_hash_text(chr(10).join(function_lines))}")

    return _hash_text("\n".join(sorted(items)) + ("\n" if items else ""))


def _resolve_llvm_diff_tool(tools: dict) -> str | None:
    direct = tools.get("llvm-diff") or tools.get("llvm_diff")
    if isinstance(direct, dict):
        direct = direct.get("path")
    if direct:
        return str(direct)
    env_direct = os.environ.get("PHASEBATCH_LLVM_DIFF")
    if env_direct:
        return env_direct
    opt = tools.get("opt")
    if isinstance(opt, dict):
        opt = opt.get("path")
    if opt:
        opt_path = Path(str(opt))
        for candidate in ("llvm-diff", "llvm-diff.exe"):
            sibling = opt_path.parent / candidate
            if sibling.exists():
                return str(sibling)
    return find_tool("llvm-diff", required=False)


def _strip_debug_attachment(line: str) -> str:
    return _rewrite_unquoted(line, lambda segment: DEBUG_ATTACHMENT_RE.sub("", segment))


def _normalize_function_fingerprint_line(stripped: str) -> str:
    if LABEL_DEFINITION_RE.match(stripped):
        return "label:"
    return _rewrite_unquoted(stripped, lambda segment: LOCAL_VALUE_RE.sub("%local", segment))


def _is_debug_metadata_line(stripped: str) -> bool:
    if stripped.startswith("!llvm.dbg"):
        return True
    if not re.match(r"!\d+\s*=", stripped):
        return False
    unquoted = _unquoted_text(stripped)
    return any(
        token in unquoted
        for token in (
            "!DI",
            "DICompileUnit",
            "DISubprogram",
            "DILocation",
            "DILocalVariable",
            "DIExpression",
            "DIFile",
            "DIBasicType",
            "DIDerivedType",
            "DISubroutineType",
            "DIGlobalVariable",
        )
    )


def _strip_ir_comment(line: str) -> str:
    in_string = False
    escaped = False
    for index, char in enumerate(line):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == ";":
            return line[:index]
    return line


def _rewrite_unquoted(text: str, rewrite) -> str:
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
            char = text[index]
            index += 1
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                break
        pieces.append(text[quote_start:index])
        segment_start = index
    pieces.append(rewrite(text[segment_start:]))
    return "".join(pieces)


def _unquoted_text(text: str) -> str:
    pieces: list[str] = []
    segment_start = 0
    index = 0
    while index < len(text):
        if text[index] != '"':
            index += 1
            continue
        pieces.append(text[segment_start:index])
        index += 1
        escaped = False
        while index < len(text):
            char = text[index]
            index += 1
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                break
        segment_start = index
    pieces.append(text[segment_start:])
    return "".join(pieces)


def _starts_function_definition(stripped: str) -> bool:
    return stripped.startswith(("define ", "declare "))


def _canonical_signature(stripped: str) -> str:
    return stripped[:-1].rstrip() if stripped.endswith("{") else stripped


def _function_name(signature: str) -> str:
    match = re.search(r"@([A-Za-z$._][\w$._-]*)", signature)
    return match.group(1) if match else signature


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
