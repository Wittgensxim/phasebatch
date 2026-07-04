from __future__ import annotations

import csv
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .ir_parser import changed_regions, parse_ir_snapshot
from .normalizer import canonical_hash
from .runner import run_opt
from .schema import INVALID_PASS_FIELDS, PASS_PROFILE_FIELDS, VALID_PASS_FIELDS


def validate_passes(
    input_ll: Path,
    passes: list[str],
    tools: dict,
    out_dir: Path,
    timeout: int,
) -> tuple[list[str], list[dict]]:
    validate_dir = Path(out_dir) / "artifacts" / "validate"
    validate_dir.mkdir(parents=True, exist_ok=True)
    valid: list[str] = []
    valid_rows: list[dict] = []
    invalid_rows: list[dict] = []

    for pass_name in passes:
        output_ll = validate_dir / f"{_safe_name(pass_name)}.ll"
        result = run_opt(str(tools["opt"]), input_ll, [pass_name], output_ll, timeout)
        if result.success:
            valid.append(pass_name)
            valid_rows.append(
                {
                    "pass": pass_name,
                    "valid": "true",
                    "reason": "ok",
                    "test_time_ms": f"{result.time_ms:.3f}",
                }
            )
        else:
            invalid_rows.append(
                {
                    "pass": pass_name,
                    "valid": "false",
                    "reason": result.failure_kind or "failed",
                    "test_time_ms": f"{result.time_ms:.3f}",
                }
            )

    _write_csv(Path(out_dir) / "valid_passes.csv", VALID_PASS_FIELDS, valid_rows)
    _write_csv(Path(out_dir) / "invalid_passes.csv", INVALID_PASS_FIELDS, invalid_rows)
    return valid, invalid_rows


def profile_passes(
    input_ll: Path,
    valid_passes: list[str],
    tools: dict,
    out_dir: Path,
    jobs: int,
    timeout: int,
) -> list[dict]:
    out_dir = Path(out_dir)
    artifacts_dir = out_dir / "artifacts" / "single_pass"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    before = parse_ir_snapshot(input_ll)
    program = Path(input_ll).parent.name or Path(input_ll).stem
    input_hash = before.module_hash
    inst_before = before.features.get("instructions", 0)

    def run_one(pass_name: str) -> dict:
        output_ll = artifacts_dir / f"{_safe_name(pass_name)}.ll"
        result = run_opt(str(tools["opt"]), input_ll, [pass_name], output_ll, timeout)
        if result.success and output_ll.exists():
            after = parse_ir_snapshot(output_ll)
            output_hash = after.module_hash
            diff = changed_regions(before, after)
            inst_after = after.features.get("instructions", 0)
            active = output_hash != input_hash
            return {
                "program": program,
                "state_hash": input_hash,
                "pass": pass_name,
                "success": "true",
                "active": _bool(active),
                "input_hash": input_hash,
                "output_hash": output_hash,
                "inst_before": inst_before,
                "inst_after": inst_after,
                "inst_delta": inst_after - inst_before,
                "funcs_changed": diff["funcs_changed"],
                "blocks_changed": diff["blocks_changed"],
                "changed_functions": _join(diff["changed_functions"]),
                "changed_blocks": _join(diff["changed_blocks"]),
                "time_ms": f"{result.time_ms:.3f}",
                "stderr_path": _path_text(result.stderr_path),
                "failure_kind": "",
            }

        return {
            "program": program,
            "state_hash": input_hash,
            "pass": pass_name,
            "success": "false",
            "active": "false",
            "input_hash": input_hash,
            "output_hash": "",
            "inst_before": inst_before,
            "inst_after": "",
            "inst_delta": "",
            "funcs_changed": 0,
            "blocks_changed": 0,
            "changed_functions": "",
            "changed_blocks": "",
            "time_ms": f"{result.time_ms:.3f}",
            "stderr_path": _path_text(result.stderr_path),
            "failure_kind": result.failure_kind or "failed",
        }

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as executor:
        rows = list(executor.map(run_one, valid_passes))

    _write_csv(out_dir / "pass_profile.csv", PASS_PROFILE_FIELDS, rows)
    return rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "pass"


def _join(values: list[str]) -> str:
    return ";".join(values)


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _path_text(path: Path | None) -> str:
    return str(path) if path else ""
