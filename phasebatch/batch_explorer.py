from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

from .batch_correctness import classify_batch_correctness, skip_reason_for_correctness
from .batcher import build_batch_family, validate_batch_candidates
from .cli import analyze_state
from .coverage import build_coverage_report
from .explorer import (
    _aggregate_by_depth,
    _bool,
    _duplicate_state_row,
    _enable_suppress_rows,
    _first_row,
    _format_float,
    _read_csv,
    _relation_flip_rows,
    _state_row_from_summary,
    _tool_paths,
    _to_float,
    _write_csv,
    _write_multistate_summary,
)
from .footprint import build_footprint_overlap, write_aggregate_overlap_summary
from .normalizer import canonical_hash
from .pass_config import PassRegistry, load_pass_registry, resolve_pipeline_sequence
from .profiler import validate_passes
from .runner import prepare_input_ir, run_opt
from .schema import (
    AGGREGATE_BATCH_SUMMARY_FIELDS,
    AGGREGATE_COVERAGE_SUMMARY_FIELDS,
    AGGREGATE_BY_DEPTH_FIELDS,
    BATCH_STATE_TRANSITION_FIELDS,
    ENABLE_SUPPRESS_FIELDS,
    RELATION_FLIP_FIELDS,
    SKIPPED_BATCH_FIELDS,
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
    allow_sampled_batches: bool = False,
    max_batches_per_state: int = 20,
    max_frontier_states: int = 20,
    batch_frontier_policy: str = "all",
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    states_dir = out_dir / "states"
    states_dir.mkdir(parents=True, exist_ok=True)
    program = out_dir.name

    pass_registry = load_pass_registry(passes_path)
    configured_passes = pass_registry.names()
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
            "max_batches_per_state": max_batches_per_state,
            "max_frontier_states": max_frontier_states,
            "batch_frontier_policy": batch_frontier_policy,
            "validate_batches": validate_batches,
            "allow_sampled_batches": allow_sampled_batches,
            "exploration_mode": "batches",
        }
    )
    write_metadata(out_dir, metadata)
    tools = _tool_paths(metadata)
    tools["_pass_registry"] = pass_registry

    prepared_ir = prepare_input_ir(Path(input_path), out_dir, tools, timeout)
    valid_passes, invalid_rows = validate_passes(prepared_ir, configured_passes, tools, out_dir, timeout, pass_registry=pass_registry)

    root_dir = states_dir / "S0000"
    root_dir.mkdir(parents=True, exist_ok=True)
    root_ir = root_dir / "input.ll"
    shutil.copyfile(prepared_ir, root_ir)
    root_hash = canonical_hash(root_ir)

    _analyze_state(
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
    state_rows: list[dict] = []
    batch_transition_rows: list[dict] = []
    state_transition_rows: list[dict] = []
    enable_suppress_rows: list[dict] = []
    relation_flip_rows: list[dict] = []
    skipped_batch_rows: list[dict] = []
    canonical_rows_by_id: dict[str, dict] = {}
    hash_to_state_id: dict[str, str] = {root_hash: "S0000"}
    next_state_number = 1
    total_batch_candidates = 0
    selected_batch_candidates = 0
    root_batch_result: dict = {}

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

    frontier = [root_row]
    for depth in range(1, max_depth + 1):
        next_frontier: list[dict] = []
        for parent in frontier:
            parent_dir = Path(parent["state_dir"])
            parent_batch_result = build_batch_family(
                parent_dir,
                max_component_size=max_component_size,
                max_batch_candidates=max_batch_candidates,
            )
            if parent["state_id"] == "S0000":
                root_batch_result = parent_batch_result
            if validate_batches:
                validate_batch_candidates(parent_dir, tools, timeout=timeout, jobs=jobs)
            correctness_rows = classify_batch_correctness(
                parent_dir,
                allow_sampled_batches=allow_sampled_batches,
            )
            build_footprint_overlap(parent_dir)
            build_coverage_report(parent_dir)

            candidate_rows = _read_csv(parent_dir / "batch_candidates.csv")
            total_batch_candidates += len(candidate_rows)
            validation_map = _validation_status_map(parent_dir / "batch_validation.csv")
            correctness_map = _correctness_map(correctness_rows)
            selected_candidates = _select_candidate_batches(
                candidate_rows,
                validation_map=validation_map,
                policy=batch_frontier_policy,
                max_batches_per_state=max_batches_per_state,
            )
            selected_batch_candidates += len(selected_candidates)
            parent_input = _state_input_path(parent)
            if not parent_input.exists():
                continue

            for candidate in selected_candidates:
                result = _apply_batch_candidate(
                    candidate,
                    parent=parent,
                    parent_input=parent_input,
                    parent_dir=parent_dir,
                    states_dir=states_dir,
                    tools=tools,
                    valid_passes=valid_passes,
                    invalid_rows=invalid_rows,
                    configured_pass_count=len(configured_passes),
                    jobs=jobs,
                    timeout=timeout,
                    max_pairs=max_pairs,
                    program=program,
                    depth=depth,
                    validate_batches=validate_batches,
                    allow_sampled_batches=allow_sampled_batches,
                    correctness_map=correctness_map,
                    hash_to_state_id=hash_to_state_id,
                    canonical_rows_by_id=canonical_rows_by_id,
                    state_rows=state_rows,
                    batch_transition_rows=batch_transition_rows,
                    state_transition_rows=state_transition_rows,
                    enable_suppress_rows=enable_suppress_rows,
                    relation_flip_rows=relation_flip_rows,
                    skipped_batch_rows=skipped_batch_rows,
                    next_state_number=next_state_number,
                    max_component_size=max_component_size,
                    max_batch_candidates=max_batch_candidates,
                    max_depth=max_depth,
                )
                next_state_number = result["next_state_number"]
                if result["frontier_row"]:
                    next_frontier.append(result["frontier_row"])

        frontier = _select_next_frontier(
            next_frontier,
            max_frontier_states=max_frontier_states,
            batch_frontier_policy=batch_frontier_policy,
        )

    if not root_batch_result:
        root_batch_result = build_batch_family(
            root_dir,
            max_component_size=max_component_size,
            max_batch_candidates=max_batch_candidates,
        )
        if validate_batches:
            validate_batch_candidates(root_dir, tools, timeout=timeout, jobs=jobs)
        classify_batch_correctness(root_dir, allow_sampled_batches=allow_sampled_batches)
        build_footprint_overlap(root_dir)
        build_coverage_report(root_dir, terminal_not_validated=max_depth <= 0)
        total_batch_candidates = int(root_batch_result.get("batch_candidates", 0) or 0)
        selected_batch_candidates = 0

    _write_csv(out_dir / "states.csv", STATE_FIELDS, state_rows)
    _write_csv(out_dir / "state_transitions.csv", STATE_TRANSITION_FIELDS, state_transition_rows)
    _write_csv(out_dir / "batch_state_transitions.csv", BATCH_STATE_TRANSITION_FIELDS, batch_transition_rows)
    _write_csv(out_dir / "skipped_batches.csv", SKIPPED_BATCH_FIELDS, skipped_batch_rows)
    _write_csv(out_dir / "enable_suppress.csv", ENABLE_SUPPRESS_FIELDS, enable_suppress_rows)
    _write_csv(out_dir / "relation_flip.csv", RELATION_FLIP_FIELDS, relation_flip_rows)
    aggregate_rows = _aggregate_by_depth(out_dir, program)
    _write_csv(out_dir / "aggregate_by_depth.csv", AGGREGATE_BY_DEPTH_FIELDS, aggregate_rows)
    aggregate_batch_rows = _aggregate_batch_summary(out_dir, program)
    _write_csv(out_dir / "aggregate_batch_summary.csv", AGGREGATE_BATCH_SUMMARY_FIELDS, aggregate_batch_rows)
    aggregate_coverage_rows = _aggregate_coverage_summary(out_dir, program)
    _write_csv(out_dir / "aggregate_coverage_summary.csv", AGGREGATE_COVERAGE_SUMMARY_FIELDS, aggregate_coverage_rows)
    aggregate_overlap_rows = write_aggregate_overlap_summary(out_dir, program)
    correctness_rows = _batch_correctness_rows_from_states(state_rows)
    _write_multistate_summary(out_dir / "multistate_summary.md", out_dir, aggregate_rows)
    _write_batch_explore_summary(
        out_dir / "batch_explore_summary.md",
        root_hash=root_hash,
        states=state_rows,
        transitions=batch_transition_rows,
        skipped_batches=skipped_batch_rows,
        total_batch_candidates=total_batch_candidates,
        selected_batch_candidates=selected_batch_candidates,
        validate_batches=validate_batches,
        allow_sampled_batches=allow_sampled_batches,
        aggregate_batch_rows=aggregate_batch_rows,
        correctness_rows=correctness_rows,
        aggregate_coverage_rows=aggregate_coverage_rows,
        aggregate_overlap_rows=aggregate_overlap_rows,
    )
    return {
        "program": program,
        "out_dir": str(out_dir),
        "states": len(state_rows),
        "batch_transitions": len(batch_transition_rows),
        "states_csv": str(out_dir / "states.csv"),
        "batch_state_transitions_csv": str(out_dir / "batch_state_transitions.csv"),
        "skipped_batches_csv": str(out_dir / "skipped_batches.csv"),
        "enable_suppress_csv": str(out_dir / "enable_suppress.csv"),
        "relation_flip_csv": str(out_dir / "relation_flip.csv"),
        "aggregate_by_depth_csv": str(out_dir / "aggregate_by_depth.csv"),
        "aggregate_batch_summary_csv": str(out_dir / "aggregate_batch_summary.csv"),
        "aggregate_coverage_summary_csv": str(out_dir / "aggregate_coverage_summary.csv"),
        "aggregate_overlap_summary_csv": str(out_dir / "aggregate_overlap_summary.csv"),
        "multistate_summary": str(out_dir / "multistate_summary.md"),
        "batch_explore_summary": str(out_dir / "batch_explore_summary.md"),
    }


def _apply_batch_candidate(
    candidate: dict,
    *,
    parent: dict,
    parent_input: Path,
    parent_dir: Path,
    states_dir: Path,
    tools: dict,
    valid_passes: list[str],
    invalid_rows: list[dict],
    configured_pass_count: int,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
    program: str,
    depth: int,
    validate_batches: bool,
    allow_sampled_batches: bool,
    correctness_map: dict[str, dict],
    hash_to_state_id: dict[str, str],
    canonical_rows_by_id: dict[str, dict],
    state_rows: list[dict],
    batch_transition_rows: list[dict],
    state_transition_rows: list[dict],
    enable_suppress_rows: list[dict],
    relation_flip_rows: list[dict],
    skipped_batch_rows: list[dict],
    next_state_number: int,
    max_component_size: int,
    max_batch_candidates: int,
    max_depth: int,
) -> dict:
    correctness = correctness_map.get(
        candidate.get("batch_id", ""),
        {
            "validation_status": "not_validated",
            "correctness_class": "unvalidated_batch",
            "can_execute": "false",
        },
    )
    validation_status = correctness.get("validation_status", "not_validated")
    skip_reason = skip_reason_for_correctness(correctness)
    if skip_reason:
        skipped_batch_rows.append(
            _skipped_batch_row(program, parent, candidate, correctness, skip_reason)
        )
        return {"next_state_number": next_state_number, "frontier_row": None}

    order = _split_order(candidate.get("canonical_order") or candidate.get("batch_passes"))
    if not order:
        return {"next_state_number": next_state_number, "frontier_row": None}

    child_ir = parent_dir / "artifacts" / "batch_successors" / f"{candidate.get('batch_id', 'batch')}.ll"
    child_ir.parent.mkdir(parents=True, exist_ok=True)
    pass_registry = tools.get("_pass_registry")
    result = run_opt(
        tools["opt"],
        parent_input,
        resolve_pipeline_sequence(order, pass_registry if isinstance(pass_registry, PassRegistry) else None),
        child_ir,
        timeout,
    )
    if not result.success or not child_ir.exists():
        return {"next_state_number": next_state_number, "frontier_row": None}

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
            depth=depth,
            parent_state_id=parent["state_id"],
            transition_pass=batch_passes,
            ir_path=str(child_ir),
            duplicate_of=duplicate_of,
        )
        frontier_row = None
    else:
        child_dir = states_dir / child_id
        child_input = _materialize_state_input(child_dir, child_ir)
        _analyze_state(
            child_input,
            child_dir,
            tools,
            valid_passes=valid_passes,
            invalid_rows=invalid_rows,
            configured_pass_count=configured_pass_count,
            jobs=jobs,
            timeout=timeout,
            max_pairs=max_pairs,
            program=program,
            state_id=child_id,
            depth=depth,
            parent_state_id=parent["state_id"],
            transition_pass=batch_passes,
        )
        build_batch_family(
            child_dir,
            max_component_size=max_component_size,
            max_batch_candidates=max_batch_candidates,
        )
        classify_batch_correctness(child_dir, allow_sampled_batches=allow_sampled_batches)
        build_footprint_overlap(child_dir)
        build_coverage_report(child_dir, terminal_not_validated=depth >= max_depth)
        child_row = _state_row_from_summary(
            child_dir,
            program=program,
            state_id=child_id,
            state_hash=child_hash,
            depth=depth,
            parent_state_id=parent["state_id"],
            transition_pass=batch_passes,
            ir_path=child_input,
            is_duplicate=False,
            duplicate_of="",
        )
        hash_to_state_id[child_hash] = child_id
        canonical_rows_by_id[child_id] = child_row
        frontier_row = child_row

    state_rows.append(child_row)
    batch_transition_rows.append(
        _batch_transition_row(program, parent, child_row, candidate, is_duplicate, duplicate_of, validation_status)
    )
    state_transition_rows.append(
        _state_transition_row(program, parent, child_row, batch_passes, str(child_ir), is_duplicate, duplicate_of)
    )
    enable_suppress_rows.extend(_enable_suppress_rows(program, parent, child_row, batch_passes, valid_passes))
    relation_flip_rows.extend(_relation_flip_rows(program, parent, child_row, batch_passes))
    return {"next_state_number": next_state_number, "frontier_row": frontier_row}


def _state_input_path(state: dict) -> Path:
    direct = Path(state["state_dir"]) / "input.ll"
    if direct.exists():
        return direct
    return Path(state.get("ir_path", ""))


def _analyze_state(input_ll: Path, state_dir: Path, tools: dict, **kwargs) -> dict:
    try:
        return analyze_state(input_ll, state_dir, tools, **kwargs)
    except TypeError as exc:
        if "pass_registry" not in str(exc):
            raise
        fallback = dict(kwargs)
        fallback.pop("pass_registry", None)
        return analyze_state(input_ll, state_dir, tools, **fallback)


def _materialize_state_input(state_dir: Path, source_ir: Path) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    input_ll = state_dir / "input.ll"
    if source_ir.resolve() != input_ll.resolve():
        shutil.copyfile(source_ir, input_ll)
    return input_ll


def _select_candidate_batches(
    rows: list[dict],
    *,
    validation_map: dict[str, str],
    policy: str,
    max_batches_per_state: int,
) -> list[dict]:
    if max_batches_per_state <= 0:
        return []

    indexed = list(enumerate(rows))
    if policy in {"all", "diverse-hash"}:
        ordered = indexed
    elif policy == "largest-batch":
        ordered = sorted(indexed, key=lambda item: (-_to_int(item[1].get("batch_size")), item[0]))
    elif policy == "certified-first":
        ordered = sorted(
            indexed,
            key=lambda item: (
                _validation_selection_rank(validation_map.get(item[1].get("batch_id", ""), "not_validated")),
                item[0],
            ),
        )
    else:
        raise ValueError(f"unknown batch frontier policy: {policy}")

    return [row for _, row in ordered[:max_batches_per_state]]


def _select_next_frontier(
    rows: list[dict],
    *,
    max_frontier_states: int,
    batch_frontier_policy: str,
) -> list[dict]:
    if max_frontier_states <= 0:
        return []
    if batch_frontier_policy != "diverse-hash":
        return rows[:max_frontier_states]

    unique: list[dict] = []
    repeated: list[dict] = []
    seen_hashes: set[str] = set()
    for row in rows:
        state_hash = row.get("state_hash", "")
        if state_hash and state_hash not in seen_hashes:
            unique.append(row)
            seen_hashes.add(state_hash)
        else:
            repeated.append(row)
    return (unique + repeated)[:max_frontier_states]


def _validation_selection_rank(status: str) -> int:
    return {
        "all_permutations_same": 0,
        "sampled_same": 1,
        "not_validated": 2,
        "": 2,
    }.get(status, 3)


def _to_int(value: object) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def _validation_status_map(path: Path) -> dict[str, str]:
    return {row.get("batch_id", ""): row.get("validation_status", "") for row in _read_csv(path) if row.get("batch_id")}


def _correctness_map(rows: list[dict]) -> dict[str, dict]:
    return {row.get("batch_id", ""): row for row in rows if row.get("batch_id")}


_BATCH_VALIDATION_STATUSES = [
    "all_permutations_same",
    "sampled_same",
    "not_validated",
    "mismatch",
    "failed",
]


def _aggregate_batch_summary(out_dir: Path, program: str) -> list[dict]:
    state_rows = _read_csv(out_dir / "states.csv")
    transition_rows = _read_csv(out_dir / "batch_state_transitions.csv")
    skipped_rows = _read_csv(out_dir / "skipped_batches.csv")
    state_depths = {row.get("state_id", ""): _to_int(row.get("depth")) for row in state_rows}
    buckets: dict[int, dict] = {}

    for row in transition_rows:
        _batch_bucket(buckets, state_depths.get(row.get("parent_state_id", ""), 0))["executed"] += 1
    for row in skipped_rows:
        _batch_bucket(buckets, state_depths.get(row.get("parent_state_id", ""), 0))["skipped"] += 1

    seen_dirs: set[str] = set()
    for state in state_rows:
        if str(state.get("is_duplicate", "")).lower() == "true":
            continue
        state_dir_value = state.get("state_dir", "")
        if not state_dir_value:
            continue
        state_dir = Path(state_dir_value)
        state_dir_key = str(state_dir.resolve())
        if state_dir_key in seen_dirs:
            continue
        seen_dirs.add(state_dir_key)

        batch_summary = _first_row(state_dir / "batch_summary.csv")
        if not batch_summary:
            continue

        depth = _to_int(state.get("depth"))
        bucket = _batch_bucket(buckets, depth)
        bucket["states"] += 1
        bucket["candidate_counts"].append(_to_float(batch_summary.get("batch_candidates")))
        bucket["reductions"].append(_to_float(batch_summary.get("batch_reduction_estimate")))
        bucket["batch_sizes"].extend(_to_float(row.get("batch_size")) for row in _read_csv(state_dir / "batch_candidates.csv"))
        for validation in _read_csv(state_dir / "batch_validation.csv"):
            status = validation.get("validation_status") or "not_validated"
            bucket["validation_counts"][status] += 1

    aggregate_rows: list[dict] = []
    for depth in sorted(buckets):
        bucket = buckets[depth]
        validation_counts = bucket["validation_counts"]
        aggregate_rows.append(
            {
                "program": program,
                "depth": str(depth),
                "states": str(bucket["states"]),
                "avg_candidates": _format_float(_avg_batch_metric(bucket["candidate_counts"])),
                "avg_batch_size": _format_float(_avg_batch_metric(bucket["batch_sizes"])),
                "avg_reduction": _format_float(_avg_batch_metric(bucket["reductions"])),
                "executed": str(bucket["executed"]),
                "skipped": str(bucket["skipped"]),
                "all_permutations_same": str(validation_counts.get("all_permutations_same", 0)),
                "sampled_same": str(validation_counts.get("sampled_same", 0)),
                "not_validated": str(validation_counts.get("not_validated", 0)),
                "mismatch": str(validation_counts.get("mismatch", 0)),
                "failed": str(validation_counts.get("failed", 0)),
                "validation_counts": _format_validation_counts(validation_counts),
            }
        )
    return aggregate_rows


def _aggregate_coverage_summary(out_dir: Path, program: str) -> list[dict]:
    state_rows = _read_csv(out_dir / "states.csv")
    buckets: dict[int, dict] = {}
    seen_dirs: set[str] = set()
    for state in state_rows:
        if str(state.get("is_duplicate", "")).lower() == "true":
            continue
        state_dir_value = state.get("state_dir", "")
        if not state_dir_value:
            continue
        state_dir = Path(state_dir_value)
        state_dir_key = str(state_dir.resolve())
        if state_dir_key in seen_dirs:
            continue
        seen_dirs.add(state_dir_key)
        summary = _first_row(state_dir / "coverage_summary.csv")
        if not summary:
            continue
        bucket = _coverage_bucket(buckets, _to_int(state.get("depth")))
        bucket["states"] += 1
        for field in _COVERAGE_COUNT_FIELDS:
            bucket[field] += _to_int(summary.get(field))

    rows = []
    for depth in sorted(buckets):
        bucket = buckets[depth]
        row = {"program": program, "depth": str(depth), "states": str(bucket["states"])}
        for field in _COVERAGE_COUNT_FIELDS:
            row[field] = str(bucket[field])
        rows.append(row)
    return rows


_COVERAGE_COUNT_FIELDS = [
    "active_passes",
    "certified_covered",
    "heuristic_covered",
    "unresolved_conflict",
    "validation_rejected",
    "unvalidated_covered",
    "failed_or_unknown",
    "not_executed_due_to_max_depth",
    "dropped_active_passes",
]


def _coverage_bucket(buckets: dict[int, dict], depth: int) -> dict:
    if depth not in buckets:
        buckets[depth] = {"states": 0, **{field: 0 for field in _COVERAGE_COUNT_FIELDS}}
    return buckets[depth]


def _batch_bucket(buckets: dict[int, dict], depth: int) -> dict:
    if depth not in buckets:
        buckets[depth] = {
            "states": 0,
            "candidate_counts": [],
            "batch_sizes": [],
            "reductions": [],
            "executed": 0,
            "skipped": 0,
            "validation_counts": Counter(),
        }
    return buckets[depth]


def _avg_batch_metric(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _format_validation_counts(counts: Counter) -> str:
    ordered = [(status, counts.get(status, 0)) for status in _BATCH_VALIDATION_STATUSES]
    extras = sorted((status, count) for status, count in counts.items() if status not in _BATCH_VALIDATION_STATUSES)
    parts = [f"{status}={count}" for status, count in ordered + extras if count]
    return "; ".join(parts)


def _validation_skip_reason(
    validation_status: str,
    *,
    validate_batches: bool,
    allow_sampled_batches: bool,
) -> str:
    if not validate_batches:
        return ""

    status = validation_status or "not_validated"
    if status == "all_permutations_same":
        return ""
    if status == "sampled_same":
        if allow_sampled_batches:
            return ""
        return "sampled_batches_not_allowed"
    if status == "mismatch":
        return "validation_mismatch"
    if status == "failed":
        return "validation_failed"
    if status == "not_validated":
        return "validation_missing"
    return "validation_status_not_allowed"


def _split_order(value: str | None) -> list[str]:
    return [part for part in str(value or "").split(";") if part]


def _skipped_batch_row(
    program: str,
    parent: dict,
    candidate: dict,
    correctness: dict,
    skip_reason: str,
) -> dict:
    return {
        "program": program,
        "parent_state_id": parent["state_id"],
        "state_hash": parent.get("state_hash", ""),
        "batch_id": candidate.get("batch_id", ""),
        "batch_passes": candidate.get("batch_passes", ""),
        "batch_size": candidate.get("batch_size", ""),
        "validation_status": correctness.get("validation_status", "") or "not_validated",
        "correctness_class": correctness.get("correctness_class", ""),
        "skip_reason": skip_reason,
    }


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
    skipped_batches: list[dict],
    total_batch_candidates: int,
    selected_batch_candidates: int,
    validate_batches: bool,
    allow_sampled_batches: bool,
    aggregate_batch_rows: list[dict],
    correctness_rows: list[dict],
    aggregate_coverage_rows: list[dict],
    aggregate_overlap_rows: list[dict],
) -> None:
    duplicate_states = sum(1 for row in states if row.get("is_duplicate") == "true")
    validation_counts = Counter(row.get("validation_status", "") for row in transitions if row.get("validation_status"))
    skipped_counts = Counter(row.get("validation_status", "") for row in skipped_batches if row.get("validation_status"))
    lines = [
        "# Batch Explore Summary",
        "",
        "## Overall",
        "",
        f"- root state hash: {root_hash}",
        f"- states explored: {len(states)}",
        f"- batch transitions: {len(transitions)}",
        f"- duplicate states: {duplicate_states}",
        f"- total batch candidates: {total_batch_candidates}",
        f"- selected batch candidates: {selected_batch_candidates}",
        f"- executed batches: {len(transitions)}",
        f"- skipped batches: {len(skipped_batches)}",
        f"- validate batches: {_bool(validate_batches)}",
        f"- allow sampled batches: {_bool(allow_sampled_batches)}",
        "",
        "## Executed Validation Status",
        "",
    ]
    lines.extend(_markdown_table(["validation_status", "count"], [[key, str(count)] for key, count in sorted(validation_counts.items())]))
    lines.extend(["", "## Skipped By Validation Status", ""])
    lines.extend(_markdown_table(["validation_status", "count"], [[key, str(count)] for key, count in sorted(skipped_counts.items())]))
    lines.extend(["", "## Batch Correctness", ""])
    lines.extend(_batch_correctness_summary(correctness_rows, len(skipped_batches)))
    lines.extend(["", "## Coverage Invariant", ""])
    lines.extend(_coverage_invariant_summary(aggregate_coverage_rows))
    lines.extend(["", "## Coarse Footprint / Overlap Diagnostics", ""])
    lines.extend(_overlap_diagnostics_summary(aggregate_overlap_rows))
    lines.extend(["", "## By-depth Batch Summary", ""])
    lines.extend(_batch_depth_table(aggregate_batch_rows))
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


def _batch_depth_table(aggregate_batch_rows: list[dict]) -> list[str]:
    return _markdown_table(
        ["depth", "states", "avg candidates", "avg batch size", "avg reduction", "executed", "skipped", "validation counts"],
        [
            [
                row.get("depth", ""),
                row.get("states", ""),
                row.get("avg_candidates", ""),
                row.get("avg_batch_size", ""),
                row.get("avg_reduction", ""),
                row.get("executed", ""),
                row.get("skipped", ""),
                row.get("validation_counts", ""),
            ]
            for row in aggregate_batch_rows
        ],
    )


def _batch_correctness_rows_from_states(state_rows: list[dict]) -> list[dict]:
    rows = []
    seen_dirs: set[str] = set()
    for state in state_rows:
        if str(state.get("is_duplicate", "")).lower() == "true":
            continue
        state_dir_value = state.get("state_dir", "")
        if not state_dir_value:
            continue
        state_dir = Path(state_dir_value)
        state_dir_key = str(state_dir.resolve())
        if state_dir_key in seen_dirs:
            continue
        seen_dirs.add(state_dir_key)
        rows.extend(_read_csv(state_dir / "batch_correctness.csv"))
    return rows


def _batch_correctness_summary(correctness_rows: list[dict], skipped_count: int) -> list[str]:
    class_counts = Counter(row.get("correctness_class", "") for row in correctness_rows)
    executable_count = sum(1 for row in correctness_rows if row.get("can_execute") == "true")
    lines = [
        f"- total batch candidates: {len(correctness_rows)}",
        f"- certified_batch count: {class_counts.get('certified_batch', 0)}",
        f"- sampled_batch count: {class_counts.get('sampled_batch', 0)}",
        f"- rejected_batch count: {class_counts.get('rejected_batch', 0)}",
        f"- failed_batch count: {class_counts.get('failed_batch', 0)}",
        f"- unvalidated_batch count: {class_counts.get('unvalidated_batch', 0)}",
        f"- executable batch count: {executable_count}",
        f"- skipped batch count: {skipped_count}",
        "",
    ]
    table_rows = [[key or "missing", str(count)] for key, count in sorted(class_counts.items())]
    lines.extend(_markdown_table(["correctness_class", "count"], table_rows))
    return lines


def _coverage_invariant_summary(aggregate_coverage_rows: list[dict]) -> list[str]:
    totals = {field: sum(_to_int(row.get(field)) for row in aggregate_coverage_rows) for field in _COVERAGE_COUNT_FIELDS}
    lines = [
        f"- total active passes: {totals['active_passes']}",
        f"- certified covered: {totals['certified_covered']}",
        f"- heuristic covered: {totals['heuristic_covered']}",
        f"- unresolved conflict: {totals['unresolved_conflict']}",
        f"- rejected: {totals['validation_rejected']}",
        f"- unknown/unvalidated: {totals['unvalidated_covered'] + totals['failed_or_unknown']}",
        f"- not executed due max depth: {totals['not_executed_due_to_max_depth']}",
        f"- dropped: {totals['dropped_active_passes']}",
    ]
    if totals["dropped_active_passes"] > 0:
        lines.append("- WARNING: at least one active pass was dropped from batch coverage.")
    lines.extend(["", *_markdown_table(
        [
            "depth",
            "states",
            "active",
            "certified",
            "heuristic",
            "unresolved",
            "rejected",
            "unknown/unvalidated",
            "not executed due max depth",
            "dropped",
        ],
        [
            [
                row.get("depth", ""),
                row.get("states", ""),
                row.get("active_passes", ""),
                row.get("certified_covered", ""),
                row.get("heuristic_covered", ""),
                row.get("unresolved_conflict", ""),
                row.get("validation_rejected", ""),
                str(_to_int(row.get("unvalidated_covered")) + _to_int(row.get("failed_or_unknown"))),
                row.get("not_executed_due_to_max_depth", ""),
                row.get("dropped_active_passes", ""),
            ]
            for row in aggregate_coverage_rows
        ],
    )])
    return lines


def _overlap_diagnostics_summary(aggregate_overlap_rows: list[dict]) -> list[str]:
    lines = [
        "These coarse footprint labels are diagnostics only. They are not used as hard independence proof in this MVP.",
        "",
    ]
    lines.extend(
        _markdown_table(
            [
                "depth",
                "pairs",
                "disjoint_write",
                "same_function_overlap",
                "same_block_overlap",
                "unknown_overlap",
                "overlap_and_commute",
                "overlap_and_sensitive",
            ],
            [
                [
                    row.get("depth", ""),
                    row.get("total_pairs", ""),
                    row.get("disjoint_write", ""),
                    row.get("same_function_overlap", ""),
                    row.get("same_block_overlap", ""),
                    row.get("unknown_overlap", ""),
                    row.get("overlap_and_commute", ""),
                    row.get("overlap_and_order_sensitive", ""),
                ]
                for row in aggregate_overlap_rows
            ],
        )
    )
    return lines


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["none"]
    lines = [f"| {' | '.join(headers)} |", f"| {' | '.join(['---'] * len(headers))} |"]
    lines.extend(f"| {' | '.join(row)} |" for row in rows)
    return lines
