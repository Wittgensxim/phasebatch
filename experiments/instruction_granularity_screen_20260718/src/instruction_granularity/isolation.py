from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

from .deterministic_io import canonical_json_bytes, sha256_file, write_json


@dataclass(frozen=True, slots=True)
class InventoryDiff:
    added: tuple[str, ...]
    deleted: tuple[str, ...]
    size_changed: tuple[str, ...]
    content_changed: tuple[str, ...]

    @property
    def is_clean(self) -> bool:
        return not (
            self.added or self.deleted or self.size_changed or self.content_changed
        )


def assert_within_root(path: Path, root: Path) -> Path:
    candidate = Path(path).resolve()
    boundary = Path(root).resolve()
    if candidate == boundary or boundary not in candidate.parents:
        raise ValueError(f"write path outside experiment root: {candidate}")
    return candidate


def compare_inventories(baseline: dict, final: dict) -> InventoryDiff:
    before = {_record_path(row): row for row in baseline.get("files", [])}
    after = {_record_path(row): row for row in final.get("files", [])}
    common = set(before) & set(after)
    size_changed = {
        path for path in common if int(before[path]["size"]) != int(after[path]["size"])
    }
    content_changed = {
        path
        for path in common - size_changed
        if before[path]["sha256"] != after[path]["sha256"]
    }
    return InventoryDiff(
        added=tuple(sorted(set(after) - set(before))),
        deleted=tuple(sorted(set(before) - set(after))),
        size_changed=tuple(sorted(size_changed)),
        content_changed=tuple(sorted(content_changed)),
    )


def assert_safe_source_tree(root: Path) -> tuple[str, ...]:
    """AST-audit source for process-launch APIs; ordinary text is not flagged."""

    violations: list[str] = []
    for path in sorted(Path(root).rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as exc:
            violations.append(f"{path}:parse_error:{exc}")
            continue
        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    aliases[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    aliases[alias.asname or alias.name] = f"{module}.{alias.name}"
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node.func, aliases)
            if name.startswith("subprocess.") or name in {
                "os.system",
                "os.popen",
                "asyncio.create_subprocess_exec",
                "asyncio.create_subprocess_shell",
            }:
                violations.append(f"{path}:{node.lineno}:{name}")
    return tuple(violations)


def build_inventory(root: Path, *, schema_version: str) -> dict:
    """Hash every regular file using Win32 extended paths and stable relpaths."""

    root_text = os.path.abspath(os.fspath(root))
    extended_root = _extended_path(root_text)
    if not os.path.isdir(extended_root):
        raise FileNotFoundError(root_text)
    files: list[dict[str, object]] = []
    directory_count = 0
    stack: list[tuple[str, str]] = [(extended_root, "")]
    while stack:
        directory, relative_directory = stack.pop()
        directory_count += 1
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name.casefold())
        except OSError as exc:
            raise OSError(f"inventory scandir failed: {directory}") from exc
        child_directories: list[tuple[str, str]] = []
        for entry in entries:
            relative = (
                f"{relative_directory}/{entry.name}"
                if relative_directory
                else entry.name
            )
            entry_stat = entry.stat(follow_symlinks=False)
            is_reparse_point = bool(
                getattr(entry_stat, "st_file_attributes", 0) & 0x400
            )
            if entry.is_symlink() or is_reparse_point:
                raise ValueError(f"inventory rejects symlink/reparse entry: {relative}")
            try:
                if entry.is_dir(follow_symlinks=False):
                    child_directories.append((entry.path, relative))
                    continue
                if not entry.is_file(follow_symlinks=False):
                    raise ValueError(f"inventory rejects non-regular entry: {relative}")
                before = entry.stat(follow_symlinks=False)
                digest = hashlib.sha256()
                with open(entry.path, "rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                after = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise OSError(f"inventory file read failed: {relative}") from exc
            if (
                before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
                or before.st_ino != after.st_ino
            ):
                raise RuntimeError(f"inventory concurrent file change: {relative}")
            files.append(
                {
                    "path": relative.replace("\\", "/"),
                    "size": before.st_size,
                    "mtime_ns": before.st_mtime_ns,
                    "sha256": digest.hexdigest(),
                }
            )
        stack.extend(reversed(child_directories))
    files.sort(key=lambda row: str(row["path"]).casefold())
    return {
        "schema_version": schema_version,
        "root": root_text,
        "directory_count": directory_count,
        "files": files,
    }


def build_protected_inventory(
    workspace: Path, protected_roots: tuple[str, ...]
) -> dict:
    workspace = Path(workspace).resolve()
    combined: list[dict[str, object]] = []
    roots: list[str] = []
    directory_count = 0
    for name in protected_roots:
        root = workspace / name
        inventory = build_inventory(
            root, schema_version="instruction-granularity-protected-root-v1"
        )
        roots.append(str(root))
        directory_count += int(inventory["directory_count"])
        for row in inventory["files"]:
            combined.append(
                {
                    "path": f"{name}/{row['path']}",
                    "size": row["size"],
                    "mtime_ns": row["mtime_ns"],
                    "sha256": row["sha256"],
                }
            )
    combined.sort(key=lambda row: str(row["path"]).casefold())
    return {
        "schema_version": "observed-effect-protected-inventory-v1",
        "protected_roots": roots,
        "directory_count": directory_count,
        "files": combined,
    }


def inventory_record_sha256(payload: dict) -> str:
    records = [
        {
            "path": _record_path(row).replace("\\", "/"),
            "size": int(row["size"]),
            "sha256": str(row["sha256"]),
        }
        for row in payload.get("files", [])
    ]
    records.sort(key=lambda row: row["path"].casefold())
    return hashlib.sha256(canonical_json_bytes(records)).hexdigest()


def write_and_verify_final_inventory(
    baseline_path: Path,
    final_path: Path,
    final_payload: dict,
) -> dict:
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8-sig"))
    diff = compare_inventories(baseline, final_payload)
    baseline_directory_count = baseline.get("directory_count")
    final_directory_count = final_payload.get("directory_count")
    directory_count_match = (
        True
        if baseline_directory_count is None
        else int(baseline_directory_count) == int(final_directory_count)
    )
    write_json(final_path, final_payload)
    return {
        "baseline_path": str(Path(baseline_path)),
        "final_path": str(Path(final_path)),
        "baseline_file_count": len(baseline.get("files", [])),
        "final_file_count": len(final_payload.get("files", [])),
        "baseline_directory_count": baseline_directory_count,
        "final_directory_count": final_directory_count,
        "directory_count_match": directory_count_match,
        "baseline_record_sha256": inventory_record_sha256(baseline),
        "final_record_sha256": inventory_record_sha256(final_payload),
        "baseline_container_sha256": sha256_file(Path(baseline_path)),
        "final_container_sha256": sha256_file(Path(final_path)),
        "added": list(diff.added),
        "deleted": list(diff.deleted),
        "size_changed": list(diff.size_changed),
        "content_changed": list(diff.content_changed),
        "is_clean": diff.is_clean and directory_count_match,
    }


def _call_name(node: ast.expr, aliases: dict[str, str]) -> str:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value, aliases)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _record_path(row: dict) -> str:
    for key in ("path", "relative_path", "relativePath"):
        if key in row:
            return str(row[key]).replace("\\", "/")
    raise KeyError(f"inventory record has no path field: {sorted(row)}")


def _extended_path(path: str) -> str:
    if os.name != "nt" or path.startswith("\\\\?\\"):
        return path
    if path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path.lstrip("\\")
    return "\\\\?\\" + path
