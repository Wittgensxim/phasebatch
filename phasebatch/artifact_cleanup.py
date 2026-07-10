from __future__ import annotations

from pathlib import Path


def cleanup_ir_artifacts(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return {
            "ir_artifacts_cleaned": "true",
            "deleted_ir_artifacts": "0",
            "deleted_ir_artifact_bytes": "0",
            "deleted_empty_dirs": "0",
        }

    root = run_dir.resolve()
    deleted = 0
    deleted_bytes = 0
    for path in sorted(run_dir.rglob("*.ll")):
        if not path.is_file():
            continue
        if _has_keep_marker(path, root):
            continue
        resolved = path.resolve()
        if not _is_relative_to(resolved, root):
            raise RuntimeError(f"refusing to delete IR artifact outside run directory: {resolved}")
        deleted_bytes += path.stat().st_size
        path.unlink()
        deleted += 1
    deleted_empty_dirs = _cleanup_empty_dirs(run_dir, root)
    return {
        "ir_artifacts_cleaned": "true",
        "deleted_ir_artifacts": str(deleted),
        "deleted_ir_artifact_bytes": str(deleted_bytes),
        "deleted_empty_dirs": str(deleted_empty_dirs),
    }


def mark_ir_artifacts_kept() -> dict:
    return {
        "ir_artifacts_cleaned": "false",
        "deleted_ir_artifacts": "0",
        "deleted_ir_artifact_bytes": "0",
        "deleted_empty_dirs": "0",
    }


def _cleanup_empty_dirs(run_dir: Path, root: Path) -> int:
    deleted = 0
    dirs = [path for path in run_dir.rglob("*") if path.is_dir() and not path.is_symlink()]
    dirs.sort(key=lambda path: len(path.relative_to(run_dir).parts), reverse=True)
    for path in dirs:
        resolved = path.resolve()
        if resolved == root:
            continue
        if not _is_relative_to(resolved, root):
            raise RuntimeError(f"refusing to delete empty directory outside run directory: {resolved}")
        try:
            path.rmdir()
        except OSError:
            continue
        deleted += 1
    return deleted


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _has_keep_marker(path: Path, root: Path) -> bool:
    try:
        current = path.parent.resolve()
    except OSError:
        return False
    while _is_relative_to(current, root):
        if (current / ".keep_ir_artifacts").exists():
            return True
        if current == root:
            return False
        current = current.parent
    return False
