from __future__ import annotations

import csv
from pathlib import Path

from .cli import analyze_state
from .config import load_passes
from .normalizer import canonical_hash
from .profiler import validate_passes
from .runner import prepare_input_ir
from .schema import STATE_FIELDS, STATE_TRANSITION_FIELDS
from .tools import collect_toolchain, write_metadata


def explore_states(
    input_path: Path,
    out_dir: Path,
    passes_path: Path,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
    max_depth: int,
    frontier_policy: str,
    top_k: int,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    states_dir = out_dir / "states"
    states_dir.mkdir(parents=True, exist_ok=True)
    program = out_dir.name

    configured_passes = load_passes(passes_path)
    metadata = collect_toolchain()
    metadata.update(
        {
            "input": str(input_path),
            "out_dir": str(out_dir),
            "pass_config": str(passes_path),
            "configured_pass_count": len(configured_passes),
            "jobs": jobs,
            "timeout": timeout,
            "max_pairs": max_pairs,
            "max_depth": max_depth,
            "frontier_policy": frontier_policy,
            "top_k": top_k,
        }
    )
    write_metadata(out_dir, metadata)
    tools = _tool_paths(metadata)

    root_ir = prepare_input_ir(Path(input_path), out_dir, tools, timeout)
    valid_passes, invalid_rows = validate_passes(root_ir, configured_passes, tools, out_dir, timeout)
    root_hash = canonical_hash(root_ir)

    state_rows: list[dict] = []
    transition_rows: list[dict] = []
    canonical_rows_by_id: dict[str, dict] = {}
    hash_to_state_id: dict[str, str] = {root_hash: "S0000"}
    frontier: list[dict] = []
    next_state_number = 1

    root_dir = states_dir / "S0000"
    analyze_state(
        root_ir,
        root_dir,
        tools,
        valid_passes=valid_passes,
        invalid_rows=invalid_rows,
        configured_pass_count=len(configured_passes),
        jobs=jobs,
        timeout=timeout,
        max_pairs=max_pairs,
        program=program,
        state_id="S0000",
        depth=0,
        parent_state_id="",
        transition_pass="",
    )
    root_row = _state_row_from_summary(
        root_dir,
        program=program,
        state_id="S0000",
        state_hash=root_hash,
        depth=0,
        parent_state_id="",
        transition_pass="",
        ir_path=root_ir,
        is_duplicate=False,
        duplicate_of="",
    )
    state_rows.append(root_row)
    canonical_rows_by_id["S0000"] = root_row
    frontier.append(root_row)

    while frontier:
        parent = frontier.pop(0)
        parent_depth = int(parent["depth"])
        if parent_depth >= max_depth:
            continue
        parent_dir = Path(parent["state_dir"])
        active_rows = _select_frontier(
            _active_profiles(parent_dir / "pass_profile.csv"),
            parent_dir / "pair_relation.csv",
            frontier_policy,
            top_k,
        )

        for active in active_rows:
            child_hash = active.get("output_hash", "")
            child_ir = active.get("output_path", "")
            if not child_hash or not child_ir or child_hash == parent["state_hash"]:
                continue

            child_id = f"S{next_state_number:04d}"
            next_state_number += 1
            duplicate_of = hash_to_state_id.get(child_hash, "")
            is_duplicate = bool(duplicate_of)

            if is_duplicate:
                canonical = canonical_rows_by_id[duplicate_of]
                child_row = _duplicate_state_row(
                    canonical,
                    state_id=child_id,
                    depth=parent_depth + 1,
                    parent_state_id=parent["state_id"],
                    transition_pass=active["pass"],
                    ir_path=child_ir,
                    duplicate_of=duplicate_of,
                )
            else:
                child_dir = states_dir / child_id
                analyze_state(
                    Path(child_ir),
                    child_dir,
                    tools,
                    valid_passes=valid_passes,
                    invalid_rows=invalid_rows,
                    configured_pass_count=len(configured_passes),
                    jobs=jobs,
                    timeout=timeout,
                    max_pairs=max_pairs,
                    program=program,
                    state_id=child_id,
                    depth=parent_depth + 1,
                    parent_state_id=parent["state_id"],
                    transition_pass=active["pass"],
                )
                child_row = _state_row_from_summary(
                    child_dir,
                    program=program,
                    state_id=child_id,
                    state_hash=child_hash,
                    depth=parent_depth + 1,
                    parent_state_id=parent["state_id"],
                    transition_pass=active["pass"],
                    ir_path=Path(child_ir),
                    is_duplicate=False,
                    duplicate_of="",
                )
                hash_to_state_id[child_hash] = child_id
                canonical_rows_by_id[child_id] = child_row
                frontier.append(child_row)

            state_rows.append(child_row)
            transition_rows.append(_transition_row(program, parent, child_row, active, is_duplicate, duplicate_of))

    _write_csv(out_dir / "states.csv", STATE_FIELDS, state_rows)
    _write_csv(out_dir / "state_transitions.csv", STATE_TRANSITION_FIELDS, transition_rows)
    return {
        "program": program,
        "out_dir": str(out_dir),
        "states": len(state_rows),
        "transitions": len(transition_rows),
        "states_csv": str(out_dir / "states.csv"),
        "state_transitions_csv": str(out_dir / "state_transitions.csv"),
    }


def _tool_paths(metadata: dict) -> dict[str, str]:
    return {
        name: details["path"]
        for name, details in metadata.get("tools", {}).items()
        if details.get("path")
    }


def _active_profiles(path: Path) -> list[dict]:
    return [
        row
        for row in _read_csv(path)
        if _is_true(row.get("success", "true")) and _is_true(row.get("active")) and row.get("output_path")
    ]


def _select_frontier(rows: list[dict], pair_relation_path: Path, policy: str, top_k: int) -> list[dict]:
    if policy == "all-active":
        return rows
    if policy == "top-k-change":
        return sorted(rows, key=lambda row: abs(_to_int(row.get("inst_delta"))), reverse=True)[:top_k]
    if policy == "sensitive-first":
        sensitive_passes = _sensitive_passes(pair_relation_path)
        ordered = sorted(
            rows,
            key=lambda row: (0 if row.get("pass") in sensitive_passes else 1, -abs(_to_int(row.get("inst_delta"))), row.get("pass", "")),
        )
        return ordered[:top_k]
    raise ValueError(f"unknown frontier policy: {policy}")


def _sensitive_passes(path: Path) -> set[str]:
    sensitive: set[str] = set()
    for row in _read_csv(path):
        if row.get("final_relation") == "final_order_sensitive" or row.get("dynamic_relation") == "dynamic_order_sensitive":
            sensitive.add(row.get("pass_a", ""))
            sensitive.add(row.get("pass_b", ""))
    sensitive.discard("")
    return sensitive


def _state_row_from_summary(
    state_dir: Path,
    *,
    program: str,
    state_id: str,
    state_hash: str,
    depth: int,
    parent_state_id: str,
    transition_pass: str,
    ir_path: Path,
    is_duplicate: bool,
    duplicate_of: str,
) -> dict:
    summary = _first_row(state_dir / "per_state_summary.csv")
    return {
        "program": program,
        "state_id": state_id,
        "state_hash": state_hash,
        "depth": depth,
        "parent_state_id": parent_state_id,
        "transition_pass": transition_pass,
        "ir_path": str(ir_path),
        "state_dir": str(state_dir),
        "is_duplicate": _bool(is_duplicate),
        "duplicate_of": duplicate_of,
        "active_passes": summary.get("active_passes", ""),
        "pairs_tested": summary.get("pairs_tested", ""),
        "dynamic_commute": summary.get("dynamic_commute", ""),
        "order_sensitive": summary.get("order_sensitive", ""),
        "unknown": summary.get("unknown", ""),
        "max_conflict_component": summary.get("max_conflict_component", ""),
        "total_time_ms": summary.get("total_time_ms", ""),
    }


def _duplicate_state_row(
    canonical: dict,
    *,
    state_id: str,
    depth: int,
    parent_state_id: str,
    transition_pass: str,
    ir_path: str,
    duplicate_of: str,
) -> dict:
    row = dict(canonical)
    row.update(
        {
            "state_id": state_id,
            "depth": depth,
            "parent_state_id": parent_state_id,
            "transition_pass": transition_pass,
            "ir_path": ir_path,
            "is_duplicate": "true",
            "duplicate_of": duplicate_of,
        }
    )
    return row


def _transition_row(
    program: str,
    parent: dict,
    child: dict,
    active: dict,
    is_duplicate: bool,
    duplicate_of: str,
) -> dict:
    return {
        "program": program,
        "parent_state_id": parent["state_id"],
        "child_state_id": child["state_id"],
        "parent_hash": parent["state_hash"],
        "child_hash": child["state_hash"],
        "transition_pass": active.get("pass", ""),
        "depth": child["depth"],
        "active": active.get("active", ""),
        "inst_before": active.get("inst_before", ""),
        "inst_after": active.get("inst_after", ""),
        "inst_delta": active.get("inst_delta", ""),
        "is_duplicate": _bool(is_duplicate),
        "duplicate_of": duplicate_of,
        "ir_path": active.get("output_path", ""),
    }


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _first_row(path: Path) -> dict:
    rows = _read_csv(path)
    return rows[0] if rows else {}


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _is_true(value: object) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _to_int(value: object) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0
