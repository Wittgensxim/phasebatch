from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_LLVM_BIN = Path("E:/llvm/build/bin")


def find_graphviz_dot(prefixes: list[Path] | None = None) -> str | None:
    roots = [Path(prefix) for prefix in prefixes] if prefixes is not None else [Path(sys.prefix)]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if prefixes is None and conda_prefix:
        conda_root = Path(conda_prefix)
        if conda_root not in roots:
            roots.append(conda_root)

    relative_candidates = (
        Path("Library/bin/graphviz/dot.exe"),
        Path("Library/bin/dot.exe"),
        Path("Library/bin/dot.bat"),
        Path("bin/dot"),
        Path("bin/dot.exe"),
    )
    for root in roots:
        for relative in relative_candidates:
            candidate = root / relative
            if candidate.is_file():
                return str(candidate)
    return shutil.which("dot")


def find_tool(name: str, required: bool = True) -> str | None:
    direct_env = os.environ.get(_tool_env_var(name))
    if direct_env:
        direct_path = Path(direct_env)
        if direct_path.exists():
            return str(direct_path)
        if required:
            raise RuntimeError(f"configured LLVM tool '{name}' was not found at {direct_path}")

    candidates = _tool_names(name)

    for root_value in (os.environ.get("PHASEBATCH_LLVM_BIN"), str(DEFAULT_LLVM_BIN)):
        if not root_value:
            continue
        root = Path(root_value)
        for candidate in candidates:
            tool = root / candidate
            if tool.exists():
                return str(tool)

    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            return found

    if required:
        raise RuntimeError(
            f"required LLVM tool '{name}' was not found; set PHASEBATCH_LLVM_BIN "
            "or add the LLVM bin directory to PATH"
        )
    return None


def run_version(tool: str) -> str:
    completed = subprocess.run(
        [tool, "--version"],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    text = (completed.stdout or completed.stderr).strip()
    return text


def collect_toolchain() -> dict:
    tools: dict[str, dict[str, str | None]] = {}
    for name, required in (("clang", True), ("opt", True), ("llc", False), ("llvm-size", False), ("llvm-diff", False)):
        path = find_tool(name, required=required)
        tools[name] = {
            "path": path,
            "version": run_version(path) if path else None,
        }

    return {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "tools": tools,
    }


def write_metadata(out_dir: Path, metadata: dict) -> None:
    from .opt_backend import opt_backend_metadata

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata)
    backend = opt_backend_metadata()
    if backend is not None:
        payload["opt_backend"] = backend
    (out_dir / "metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _tool_names(name: str) -> list[str]:
    if name.lower().endswith(".exe"):
        return [name]
    return [name, f"{name}.exe"]


def _tool_env_var(name: str) -> str:
    return f"PHASEBATCH_{name.upper().replace('-', '_')}"
