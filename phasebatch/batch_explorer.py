from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

from .batcher import build_batch_family, validate_batch_candidates
from .cli import analyze_state
from .config import load_passes
from .explorer import (
    _aggregate_by_depth,
    _bool,
    _duplicate_state_row,
    _enable_suppress_rows,
    _read_csv,
    _relation_flip_rows,
    _state_row_from_summary,
    _tool_paths,
    _write_csv,
    _write_multistate_summary,
)
from .normalizer import canonical_hash
from .profiler import validate_passes
from .runner import prepare_input_ir, run_opt
from .schema import (
    AGGREGATE_BY_DEPTH_FIELDS,
    BATCH_STATE_TRANSITION_FIELDS,
    ENABLE_SUPPRESS_FIELDS,
    RELATION_FLIP_FIELDS,
    STATE_FIELDS,
    STATE_TRANSITION_FIELDS,
)
from .tools import collect_toolchain, write_metadata


def explore_batches(
    input_path: Path,
    out_dir: Path,
    passes_path: Path,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
    max_depth: int,
    max_component_size: int,
    max_batch_candidates: int,
    validate_batches: bool,
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
            "max_component_size": max_component_size,
            "max_batch_candidates": max_batch_candidates,
            "validate_batches": validate_batches,
            "exploration_mode": "batches",
        }
    )
    write_metadata(out_dir, metadata)
    tools = _tool_paths(metadata)

    prepared_ir = prepare_input_ir(Path(input_path), out_dir, tools, timeout)
    valid_passes, invalid_rows = validate_passes(prepared_ir, configured_passes, tools, out_dir, timeout)

    root_dir = states_dir / "S0000"
    root_dir.mkdir(parents=True, exist_ok=True)
    root_ir = root_dir / "input.ll"
    shutil.copyfile(prepared_ir, root_ir)
    root_hash = canonical_hash(root_ir)

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
    root_batch_result = build_batch_family(
        root_dir,
        max_component_size=max_component_size,
        max_batch_candidates=max_batch_candidates,
    )
    if validate_batches:
        validate_batch_candidates(root_dir, tools, timeout=timeout, jobs=jobs)

    state_rows: list[dict] = []
    batch_transition_rows: list[dict] = []
    state_transition_rows: list[dict] = []
    enable_suppress_rows: list[dict] = []
    relation_flip_rows: list[dict] = []
    canonical_rows_by_id: dict[str, dict] = {}
    hash_to_state_id: dict[str, str] = {root_hash: "S0000"}
    next_state_number = 1

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

    if max_depth >= 1:
        validation_map = _validation_status_map(root_dir / "batch_validation.csv")
        for candidate in _read_csv(root_dir / "batch_candidates.csv"):
            order = _split_order(candidate.get("canonical_order") or candidate.get("batch_passes"))
            if not order:
                continue

            child_ir = root_dir / "artifacts" / "batch_successors" / f"{candidate.get('batch_id', 'batch')}.ll"
            child_ir.parent.mkdir(parents=True, exist_ok=True)
            result = run_opt(tools["opt"], root_ir, order, child_ir, timeout)
            if not result.success or not child_ir.exists():
                continue

            child_hash = canonical_hash(child_ir)
            child_id = f"S{next_state_number:04d}"
            next_state_number += 1
            duplicate_of = hash_to_state_id.get(child_hash, "")
            is_duplicate = bool(duplicate_of)
            batch_passes = ";".join(order)

            if is_duplicate:
                child_row = _duplicate_state_row(
                    canonical_rows_by_id[duplicate_of],
                    state_id=child_id,
                    depth=1,
                    parent_state_id="S0000",
                    transition_pass=batch_passes,
                    ir_path=str(child_ir),
                    duplicate_of=duplicate_of,
                )
            else:
                child_dir = states_dir / child_id
                analyze_state(
                    child_ir,
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
                    depth=1,
                    parent_state_id="S0000",
                    transition_pass=batch_passes,
                )
                build_batch_family(
                    child_dir,
                    max_component_size=max_component_size,
                    max_batch_candidates=max_batch_candidates,
                )
                child_row = _state_row_from_summary(
                    child_dir,
                    program=program,
                    state_id=child_id,
                    state_hash=child_hash,
                    depth=1,
                    parent_state_id="S0000",
                    transition_pass=batch_passes,
                    ir_path=child_ir,
                    is_duplicate=False,
                    duplicate_of="",
                )
                hash_to_state_id[child_hash] = child_id
                canonical_rows_by_id[child_id] = child_row

            state_rows.append(child_row)
            validation_status = validation_map.get(candidate.get("batch_id", ""), "not_validated")
            batch_transition_rows.append(
                _batch_transition_row(program, root_row, child_row, candidate, is_duplicate, duplicate_of, validation_status)
            )
            state_transition_rows.append(
                _state_transition_row(program, root_row, child_row, batch_passes, str(child_ir), is_duplicate, duplicate_of)
            )
            enable_suppress_rows.extend(_enable_suppress_rows(program, root_row, child_row, batch_passes, valid_passes))
            relation_flip_rows.extend(_relation_flip_rows(program, root_row, child_row, batch_passes))

    _write_csv(out_dir / "states.csv", STATE_FIELDS, state_rows)
    _write_csv(out_dir / "state_transitions.csv", STATE_TRANSITION_FIELDS, state_transition_rows)
    _write_csv(out_dir / "batch_state_transitions.csv", BATCH_STATE_TRANSITION_FIELDS, batch_transition_rows)
    _write_csv(out_dir / "enable_suppress.csv", ENABLE_SUPPRESS_FIELDS, enable_suppress_rows)
    _write_csv(out_dir / "relation_flip.csv", RELATION_FLIP_FIELDS, relation_flip_rows)
    aggregate_rows = _aggregate_by_depth(out_dir, program)
    _write_csv(out_dir / "aggregate_by_depth.csv", AGGREGATE_BY_DEPTH_FIELDS, aggregate_rows)
    _write_multistate_summary(out_dir / "multistate_summary.md", out_dir, aggregate_rows)
    _write_batch_explore_summary(
        out_dir / "batch_explore_summary.md",
        root_hash=root_hash,
        states=state_rows,
        transitions=batch_transition_rows,
        root_batch_result=root_batch_result,
        validate_batches=validate_batches,
    )
    return {
        "program": program,
        "out_dir": str(out_dir),
        "states": len(state_rows),
        "batch_transitions": len(batch_transition_rows),
        "states_csv": str(out_dir / "states.csv"),
        "batch_state_transitions_csv": str(out_dir / "batch_state_transitions.csv"),
        "enable_suppress_csv": str(out_dir / "enable_suppress.csv"),
        "relation_flip_csv": str(out_dir / "relation_flip.csv"),
        "aggregate_by_depth_csv": str(out_dir / "aggregate_by_depth.csv"),
        "multistate_summary": str(out_dir / "multistate_summary.md"),
        "batch_explore_summary": str(out_dir / "batch_explore_summary.md"),
    }


def _validation_status_map(path: Path) -> dict[str, str]:
    return {row.get("batch_id", ""): row.get("validation_status", "") for row in _read_csv(path) if row.get("batch_id")}


def _split_order(value: str | None) -> list[str]:
    return [part for part in str(value or "").split(";") if part]


def _batch_transition_row(
    program: str,
    parent: dict,
    child: dict,
    candidate: dict,
    is_duplicate: bool,
    duplicate_of: str,
    validation_status: str,
) -> dict:
    return {
        "program": program,
        "parent_state_id": parent["state_id"],
        "child_state_id": child["state_id"],
        "batch_id": candidate.get("batch_id", ""),
        "batch_passes": candidate.get("batch_passes", ""),
        "batch_size": candidate.get("batch_size", ""),
        "parent_hash": parent["state_hash"],
        "child_hash": child["state_hash"],
        "is_duplicate": _bool(is_duplicate),
        "duplicate_of": duplicate_of,
        "validation_status": validation_status or "not_validated",
    }


def _state_transition_row(
    program: str,
    parent: dict,
    child: dict,
    batch_passes: str,
    ir_path: str,
    is_duplicate: bool,
    duplicate_of: str,
) -> dict:
    return {
        "program": program,
        "parent_state_id": parent["state_id"],
        "child_state_id": child["state_id"],
        "parent_hash": parent["state_hash"],
        "child_hash": child["state_hash"],
        "transition_pass": batch_passes,
        "depth": child["depth"],
        "active": "",
        "inst_before": "",
        "inst_after": "",
        "inst_delta": "",
        "is_duplicate": _bool(is_duplicate),
        "duplicate_of": duplicate_of,
        "ir_path": ir_path,
    }


def _write_batch_explore_summary(
    path: Path,
    *,
    root_hash: str,
    states: list[dict],
    transitions: list[dict],
    root_batch_result: dict,
    validate_batches: bool,
) -> None:
    duplicate_states = sum(1 for row in states if row.get("is_duplicate") == "true")
    validation_counts = Counter(row.get("validation_status", "") for row in transitions if row.get("validation_status"))
    lines = [
        "# Batch Explore Summary",
        "",
        "## Overall",
        "",
        f"- root state hash: {root_hash}",
        f"- states explored: {len(states)}",
        f"- batch transitions: {len(transitions)}",
        f"- duplicate states: {duplicate_states}",
        f"- root batch candidates: {root_batch_result.get('batch_candidates', 0)}",
        f"- validate batches: {_bool(validate_batches)}",
        "",
        "## Validation",
        "",
    ]
    lines.extend(_markdown_table(["validation_status", "count"], [[key, str(count)] for key, count in sorted(validation_counts.items())]))
    lines.extend(["", "## Batch Transitions", ""])
    lines.extend(
        _markdown_table(
            ["batch", "size", "child", "duplicate", "validation"],
            [
                [
                    row.get("batch_id", ""),
                    row.get("batch_size", ""),
                    row.get("child_state_id", ""),
                    row.get("is_duplicate", ""),
                    row.get("validation_status", ""),
                ]
                for row in transitions[:20]
            ],
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["none"]
    lines = [f"| {' | '.join(headers)} |", f"| {' | '.join(['---'] * len(headers))} |"]
    lines.extend(f"| {' | '.join(row)} |" for row in rows)
    return lines
