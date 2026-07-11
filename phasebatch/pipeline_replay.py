from __future__ import annotations

import csv
import json
import shutil
import tempfile
import time
from pathlib import Path

from .ir_equivalence import compare_ir_equivalence
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
        row = _comparison_row(run_dir, root_ir, optimized_pipeline, replay_output, final_ir, start, timeout)
        _write_csv(run_dir / "pipeline_replay.csv", PIPELINE_REPLAY_FIELDS, [row])
        return {**row, "pipeline_replay_csv": str(run_dir / "pipeline_replay.csv")}

    segments = _replay_segments(run_dir, optimized_pipeline)
    result = None
    try:
        opt = _opt_path(run_dir)
        with tempfile.TemporaryDirectory(prefix=".phasebatch-replay-", dir=run_dir) as tmp:
            current_input = root_ir
            for index, passes in enumerate(segments):
                output_ir = replay_output if index == len(segments) - 1 else Path(tmp) / f"step_{index:04d}.ll"
                result = run_opt(opt, current_input, passes, output_ir, timeout)
                if not result.success or not output_ir.exists():
                    break
                current_input = output_ir
    except Exception as exc:  # pragma: no cover - defensive for real tool failures.
        row = _row(run_dir, root_ir, optimized_pipeline, replay_output, final_ir, "", "", False, "failed", str(exc), start)
        _write_csv(run_dir / "pipeline_replay.csv", PIPELINE_REPLAY_FIELDS, [row])
        return {**row, "pipeline_replay_csv": str(run_dir / "pipeline_replay.csv")}

    if result is None or not result.success or not replay_output.exists():
        error_message = "opt failed" if result is None else (result.stderr or result.failure_kind or "opt failed").strip()
        final_hash = canonical_hash(final_ir)
        row = _row(run_dir, root_ir, optimized_pipeline, replay_output, final_ir, "", final_hash, False, "failed", error_message, start)
        _write_csv(run_dir / "pipeline_replay.csv", PIPELINE_REPLAY_FIELDS, [row])
        return {**row, "pipeline_replay_csv": str(run_dir / "pipeline_replay.csv")}

    row = _comparison_row(run_dir, root_ir, optimized_pipeline, replay_output, final_ir, start, timeout)
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
    *,
    text_hash_equal: bool | None = None,
    llvm_diff_equal: bool | None = None,
    module_fingerprint_equal: bool | None = None,
    equality_tier: str = "",
    equality_reason: str = "",
    can_hard_fold: bool | None = None,
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
        "text_hash_equal": _bool_or_empty(text_hash_equal),
        "llvm_diff_equal": _bool_or_empty(llvm_diff_equal),
        "module_fingerprint_equal": _bool_or_empty(module_fingerprint_equal),
        "equality_tier": equality_tier,
        "equality_reason": equality_reason,
        "can_hard_fold": _bool_or_empty(can_hard_fold),
        "replay_status": replay_status,
        "error_message": error_message,
        "time_ms": f"{(time.perf_counter() - start) * 1000:.3f}",
    }


def _comparison_row(
    run_dir: Path,
    root_ir: Path,
    optimized_pipeline: str,
    replay_output: Path,
    final_ir: Path,
    start: float,
    timeout: int,
) -> dict:
    equality = compare_ir_equivalence(replay_output, final_ir, tools=_replay_tools(run_dir), timeout=timeout)
    replay_hash = equality.left_hash or canonical_hash(replay_output)
    final_hash = equality.right_hash or canonical_hash(final_ir)
    if equality.can_hard_fold:
        status = "success"
        error_message = ""
    elif equality.tier == "failed":
        status = "failed"
        error_message = equality.error_message or equality.reason
    else:
        status = "mismatch"
        error_message = ""
    return _row(
        run_dir,
        root_ir,
        optimized_pipeline,
        replay_output,
        final_ir,
        replay_hash,
        final_hash,
        equality.can_hard_fold,
        status,
        error_message,
        start,
        text_hash_equal=equality.text_hash_equal,
        llvm_diff_equal=equality.llvm_diff_equal,
        module_fingerprint_equal=equality.module_fingerprint_equal,
        equality_tier=equality.tier,
        equality_reason=equality.reason,
        can_hard_fold=equality.can_hard_fold,
    )


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


def _replay_segments(run_dir: Path, optimized_pipeline: str) -> list[list[str]]:
    aggregate = _split_pipeline(optimized_pipeline)
    chosen_path = Path(run_dir) / "chosen_path.csv"
    if not chosen_path.exists():
        return [aggregate]

    with chosen_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    segments = [
        _split_batch_order(row.get("canonical_order") or row.get("batch_passes", ""))
        for row in rows
    ]
    segments = [segment for segment in segments if segment]
    flattened = [pass_name for segment in segments for pass_name in segment]
    return segments if flattened == aggregate else [aggregate]


def _split_batch_order(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def _opt_path(run_dir: Path) -> str:
    opt = _replay_tools(run_dir).get("opt")
    if opt:
        return str(opt)
    collected = collect_toolchain()
    opt = ((collected.get("tools") or {}).get("opt") or {}).get("path")
    if not opt:
        raise RuntimeError("could not find opt for pipeline replay")
    return str(opt)


def _replay_tools(run_dir: Path) -> dict[str, str]:
    metadata = _read_json(Path(run_dir) / "metadata.json")
    return {
        name: details["path"]
        for name, details in (metadata.get("tools") or {}).items()
        if isinstance(details, dict) and details.get("path")
    }


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
                f"- equality_tier: {replay_result.get('equality_tier', '')}",
                f"- equality_reason: {replay_result.get('equality_reason', '')}",
                f"- can_hard_fold: {replay_result.get('can_hard_fold', '')}",
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


def _bool_or_empty(value: bool | None) -> str:
    return "" if value is None else _bool(value)
