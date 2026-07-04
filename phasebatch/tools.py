from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_LLVM_BIN = Path("E:/llvm/build/bin")


def find_tool(name: str, required: bool = True) -> str | None:
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
    for name, required in (("clang", True), ("opt", True), ("llc", False), ("llvm-size", False)):
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
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _tool_names(name: str) -> list[str]:
    if name.lower().endswith(".exe"):
        return [name]
    return [name, f"{name}.exe"]
