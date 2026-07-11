from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

from .runner import compile_c_to_ll
from .schema import RunResult


BENCHMARK_CANDIDATE_FIELDS = [
    "program",
    "relative_path",
    "absolute_path",
    "source_bytes",
    "category",
    "compile_status",
    "compile_time_ms",
    "selected",
    "skip_reason",
    "error_message",
]


def discover_advisor_benchmarks(
    *,
    test_suite_root: Path,
    out_dir: Path,
    clang: str,
    num_programs: int = 15,
    max_source_bytes: int = 200_000,
    selection_seed: int = 0,
    timeout: int = 15,
    benchmark_manifest: Path | None = None,
    compile_runner: Callable[[str, Path, Path, int], RunResult] = compile_c_to_ll,
) -> dict:
    if num_programs < 1:
        raise ValueError("num_programs must be positive")
    if max_source_bytes < 1:
        raise ValueError("max_source_bytes must be positive")

    root = Path(test_suite_root).resolve()
    single_source = root / "SingleSource"
    if not single_source.is_dir():
        raise FileNotFoundError(f"LLVM SingleSource directory not found: {single_source}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    explicit = _load_manifest(Path(benchmark_manifest), root) if benchmark_manifest else None
    entries = explicit if explicit is not None else _scan_entries(single_source, root)
    rows = _candidate_rows(entries, root)
    smoke_root = out_dir / ".benchmark_smoke"

    for index, row in enumerate(rows):
        source = Path(row["absolute_path"])
        if source.suffix.lower() != ".c":
            row.update(compile_status="skipped", skip_reason="not_c_source")
            continue
        if not source.is_file():
            row.update(compile_status="skipped", skip_reason="source_missing")
            continue
        if int(row["source_bytes"]) > max_source_bytes:
            row.update(compile_status="skipped", skip_reason="source_too_large")
            continue

        output_ll = smoke_root / f"{index:05d}_{_safe_name(row['program'])}" / "input.ll"
        try:
            result = compile_runner(clang, source, output_ll, timeout)
        except Exception as exc:  # The candidate row must survive tool failures.
            row.update(
                compile_status="failed",
                compile_time_ms="",
                skip_reason="compile_exception",
                error_message=_one_line(exc),
            )
        else:
            row["compile_time_ms"] = _format_number(result.time_ms)
            if result.success:
                row["compile_status"] = "success"
            else:
                row.update(
                    compile_status="failed",
                    skip_reason="compile_failed",
                    error_message=_one_line(result.stderr or result.failure_kind),
                )
        finally:
            _remove_smoke_artifact(output_ll, smoke_root)

    eligible = [row for row in rows if row["compile_status"] == "success"]
    if explicit is not None:
        selected_rows = eligible
    else:
        selected_rows = _select_rows(eligible, num_programs, selection_seed)
    selected_keys = {row["absolute_path"] for row in selected_rows}
    for row in rows:
        row["selected"] = _bool(row["absolute_path"] in selected_keys)
        if row["compile_status"] == "success" and row["absolute_path"] not in selected_keys:
            row["skip_reason"] = "not_selected"

    rows.sort(key=lambda row: row["relative_path"].lower())
    selected_rows.sort(key=lambda row: row["relative_path"].lower())
    _write_csv(out_dir / "benchmark_candidates.csv", BENCHMARK_CANDIDATE_FIELDS, rows)
    _write_csv(out_dir / "benchmark_selection.csv", BENCHMARK_CANDIDATE_FIELDS, selected_rows)
    manifest_path = out_dir / "selected_benchmarks.yaml"
    _write_selected_manifest(manifest_path, selected_rows)

    selected = [
        {"name": row["program"], "path": row["absolute_path"]}
        for row in selected_rows
    ]
    return {
        "candidates": len(rows),
        "compile_successes": len(eligible),
        "selected_count": len(selected),
        "selected": selected,
        "benchmark_candidates_csv": str(out_dir / "benchmark_candidates.csv"),
        "benchmark_selection_csv": str(out_dir / "benchmark_selection.csv"),
        "selected_benchmarks_yaml": str(manifest_path),
    }


def _scan_entries(single_source: Path, root: Path) -> list[dict]:
    return [
        {"name": path.stem, "path": path.resolve(), "relative_path": path.resolve().relative_to(root).as_posix()}
        for path in sorted(single_source.rglob("*.c"), key=lambda item: item.relative_to(single_source).as_posix().lower())
        if path.is_file()
    ]


def _candidate_rows(entries: list[dict], root: Path) -> list[dict]:
    names: Counter[str] = Counter()
    rows = []
    for entry in entries:
        source = Path(entry["path"]).resolve()
        relative = entry.get("relative_path") or _relative_path(source, root)
        base_name = _safe_name(str(entry.get("name") or source.stem))
        name_key = base_name.casefold()
        names[name_key] += 1
        program = base_name if names[name_key] == 1 else f"{base_name}_{names[name_key]}"
        parent = Path(relative).parent.as_posix()
        rows.append(
            {
                "program": program,
                "relative_path": str(relative).replace("\\", "/"),
                "absolute_path": str(source),
                "source_bytes": str(source.stat().st_size) if source.is_file() else "",
                "category": parent if parent not in {"", "."} else "SingleSource",
                "compile_status": "pending",
                "compile_time_ms": "",
                "selected": "false",
                "skip_reason": "",
                "error_message": "",
            }
        )
    return rows


def _select_rows(rows: list[dict], limit: int, seed: int) -> list[dict]:
    ranked = sorted(rows, key=lambda row: (_stable_rank(seed, row["relative_path"]), row["relative_path"].lower()))
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in ranked:
        by_category[row["category"]].append(row)

    selected: list[dict] = []
    selected_paths: set[str] = set()
    parent_counts: Counter[str] = Counter()
    diverse_categories = sorted(
        by_category,
        key=lambda category: (_stable_rank(seed, category), category.lower()),
    )[: min(3, limit)]
    for category in diverse_categories:
        row = by_category[category][0]
        selected.append(row)
        selected_paths.add(row["absolute_path"])
        parent_counts[row["category"]] += 1

    for row in ranked:
        if len(selected) >= limit:
            break
        if row["absolute_path"] in selected_paths or parent_counts[row["category"]] >= 5:
            continue
        selected.append(row)
        selected_paths.add(row["absolute_path"])
        parent_counts[row["category"]] += 1
    return selected


def _load_manifest(path: Path, root: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"benchmark manifest not found: {path}")
    try:
        import yaml  # type: ignore
    except ImportError:
        loaded = {"benchmarks": _parse_small_manifest(path)}
    else:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not isinstance(loaded.get("benchmarks"), list):
        raise ValueError("benchmark manifest must contain a benchmarks list")

    entries = []
    for index, item in enumerate(loaded["benchmarks"]):
        if not isinstance(item, dict):
            raise ValueError(f"benchmark manifest entry {index} must be a mapping")
        raw_path = str(item.get("path") or "").strip()
        name = str(item.get("name") or "").strip()
        if not raw_path or not name:
            raise ValueError(f"benchmark manifest entry {index} requires name and path")
        source = Path(raw_path)
        if not source.is_absolute():
            source = root / source
        source = source.resolve()
        entries.append({"name": name, "path": source, "relative_path": _relative_path(source, root)})
    return entries


def _parse_small_manifest(path: Path) -> list[dict]:
    entries = []
    current: dict[str, str] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.split("#", 1)[0].strip()
        if not stripped or stripped == "benchmarks:":
            continue
        if stripped.startswith("-"):
            if current:
                entries.append(current)
            current = {}
            stripped = stripped[1:].strip()
        if ":" in stripped and current is not None:
            key, value = stripped.split(":", 1)
            current[key.strip()] = value.strip().strip("'\"")
    if current:
        entries.append(current)
    return entries


def _write_selected_manifest(path: Path, rows: list[dict]) -> None:
    lines = ["benchmarks:"]
    for row in rows:
        lines.extend(
            [
                f"  - name: {_yaml_scalar(row['program'])}",
                f"    path: {_yaml_scalar(row['relative_path'])}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _remove_smoke_artifact(output_ll: Path, smoke_root: Path) -> None:
    output_ll.unlink(missing_ok=True)
    current = output_ll.parent
    while current != smoke_root.parent and current.exists():
        try:
            current.rmdir()
        except OSError:
            break
        if current == smoke_root:
            break
        current = current.parent


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _stable_rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "benchmark"


def _yaml_scalar(value: str) -> str:
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def _format_number(value: object) -> str:
    try:
        return f"{float(value):.3f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return ""


def _one_line(value: object) -> str:
    return " ".join(str(value or "").split())[:2000]


def _bool(value: bool) -> str:
    return "true" if value else "false"
