from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path

from .normalizer import canonical_hash
from .runner import run_opt
from .schema import PIPELINE_REPLAY_FIELDS
from .tools import collect_toolchain


def replay_optimized_pipeline(run_dir: Path, timeout: int = 10) -> dict:
    run_dir = Path(run_dir)
    root_ir = run_dir / "states" / "S0000" / "input.ll"
    final_ir = run_dir / "final.ll"
    replay_output = run_dir / "replayed_final.ll"
    optimized_pipeline = _read_pipeline(run_dir / "optimized_pipeline.txt")
    start = time.perf_counter()
    error_message = ""
    replay_hash = ""
    final_hash = ""

    if not root_ir.exists():
        row = _row(run_dir, root_ir, optimized_pipeline, replay_output, final_ir, "", "", False, "failed", f"missing root IR: {root_ir}", start)
        _write_csv(run_dir / "pipeline_replay.csv", PIPELINE_REPLAY_FIELDS, [row])
        return {**row, "pipeline_replay_csv": str(run_dir / "pipeline_replay.csv")}
    if not final_ir.exists():
        row = _row(run_dir, root_ir, optimized_pipeline, replay_output, final_ir, "", "", False, "failed", f"missing final IR: {final_ir}", start)
        _write_csv(run_dir / "pipeline_replay.csv", PIPELINE_REPLAY_FIELDS, [row])
        return {**row, "pipeline_replay_csv": str(run_dir / "pipeline_replay.csv")}

    if not optimized_pipeline:
        shutil.copyfile(root_ir, replay_output)
        replay_hash = canonical_hash(replay_output)
        final_hash = canonical_hash(final_ir)
        hashes_match = replay_hash == final_hash
        status = "success" if hashes_match else "mismatch"
        row = _row(run_dir, root_ir, optimized_pipeline, replay_output, final_ir, replay_hash, final_hash, hashes_match, status, "", start)
        _write_csv(run_dir / "pipeline_replay.csv", PIPELINE_REPLAY_FIELDS, [row])
        return {**row, "pipeline_replay_csv": str(run_dir / "pipeline_replay.csv")}

    passes = _split_pipeline(optimized_pipeline)
    try:
        opt = _opt_path(run_dir)
        result = run_opt(opt, root_ir, passes, replay_output, timeout)
    except Exception as exc:  # pragma: no cover - defensive for real tool failures.
        row = _row(run_dir, root_ir, optimized_pipeline, replay_output, final_ir, "", "", False, "failed", str(exc), start)
        _write_csv(run_dir / "pipeline_replay.csv", PIPELINE_REPLAY_FIELDS, [row])
        return {**row, "pipeline_replay_csv": str(run_dir / "pipeline_replay.csv")}

    if not result.success or not replay_output.exists():
        error_message = (result.stderr or result.failure_kind or "opt failed").strip()
        final_hash = canonical_hash(final_ir)
        row = _row(run_dir, root_ir, optimized_pipeline, replay_output, final_ir, "", final_hash, False, "failed", error_message, start)
        _write_csv(run_dir / "pipeline_replay.csv", PIPELINE_REPLAY_FIELDS, [row])
        return {**row, "pipeline_replay_csv": str(run_dir / "pipeline_replay.csv")}

    replay_hash = canonical_hash(replay_output)
    final_hash = canonical_hash(final_ir)
    hashes_match = replay_hash == final_hash
    status = "success" if hashes_match else "mismatch"
    row = _row(run_dir, root_ir, optimized_pipeline, replay_output, final_ir, replay_hash, final_hash, hashes_match, status, "", start)
    _write_csv(run_dir / "pipeline_replay.csv", PIPELINE_REPLAY_FIELDS, [row])
    return {**row, "pipeline_replay_csv": str(run_dir / "pipeline_replay.csv")}


def update_replay_status_artifacts(run_dir: Path, replay_result: dict | None, replay_verified: str) -> None:
    run_dir = Path(run_dir)
    _update_chosen_path_summary(run_dir / "chosen_path_summary.csv", replay_verified)
    _update_optimize_summary(run_dir / "optimize_summary.md", replay_result, replay_verified)


def _row(
    run_dir: Path,
    root_ir: Path,
    optimized_pipeline: str,
    replay_output: Path,
    final_ir: Path,
    replay_hash: str,
    final_hash: str,
    hashes_match: bool,
    replay_status: str,
    error_message: str,
    start: float,
) -> dict:
    return {
        "program": run_dir.name,
        "root_ir_path": str(root_ir),
        "optimized_pipeline": optimized_pipeline,
        "replay_output_path": str(replay_output),
        "final_ir_path": str(final_ir),
        "replay_hash": replay_hash,
        "final_hash": final_hash,
        "hashes_match": _bool(hashes_match),
        "replay_status": replay_status,
        "error_message": error_message,
        "time_ms": f"{(time.perf_counter() - start) * 1000:.3f}",
    }


def _read_pipeline(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _split_pipeline(value: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in value.replace("\n", ""):
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
        if char == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _opt_path(run_dir: Path) -> str:
    metadata = _read_json(run_dir / "metadata.json")
    opt = ((metadata.get("tools") or {}).get("opt") or {}).get("path")
    if opt:
        return str(opt)
    collected = collect_toolchain()
    opt = ((collected.get("tools") or {}).get("opt") or {}).get("path")
    if not opt:
        raise RuntimeError("could not find opt for pipeline replay")
    return str(opt)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _update_chosen_path_summary(path: Path, replay_verified: str) -> None:
    rows, fieldnames = _read_csv_with_fields(path)
    if not rows or not fieldnames:
        return
    if "replay_verified" not in fieldnames:
        fieldnames.append("replay_verified")
    for row in rows:
        row["replay_verified"] = replay_verified
    _write_csv(path, fieldnames, rows)


def _update_optimize_summary(path: Path, replay_result: dict | None, replay_verified: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Optimize Batches Summary\n"
    marker = "## Final Pipeline Replay Verification"
    existing = existing.split(marker, 1)[0].rstrip()
    lines = [
        existing,
        "",
        marker,
        "",
        f"- replay_verified: {replay_verified}",
    ]
    if replay_result:
        lines.extend(
            [
                f"- replay_status: {replay_result.get('replay_status', '')}",
                f"- hashes_match: {replay_result.get('hashes_match', '')}",
                f"- replay_hash: {replay_result.get('replay_hash', '')}",
                f"- final_hash: {replay_result.get('final_hash', '')}",
                f"- replayed_final_ll: {replay_result.get('replay_output_path', '')}",
            ]
        )
        if replay_result.get("hashes_match") != "true":
            lines.append("- WARNING: final pipeline replay did not reproduce final.ll.")
        error = replay_result.get("error_message", "")
        if error:
            lines.append(f"- error_message: {error}")
    else:
        lines.append("- replay_status: not_run")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_csv_with_fields(path: Path) -> tuple[list[dict], list[str]]:
    if not path.exists():
        return [], []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _bool(value: bool) -> str:
    return "true" if value else "false"
