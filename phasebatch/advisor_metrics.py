from __future__ import annotations

import csv
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .footprint import build_footprint_overlap


PAIR_STATE_FIELDS = [
    "program", "state_id", "depth", "active_passes", "total_pairs", "commute_pairs",
    "order_sensitive_pairs", "unknown_pairs", "commute_ratio", "canonical_hash_commute",
    "structural_diff_commute", "comparator_failed", "timeout_pairs", "not_tested_pairs",
]
COMPONENT_FIELDS = [
    "program", "state_id", "depth", "component_id", "component_size", "component_passes",
    "same_function_edges", "same_block_edges", "possible_ww_edges", "order_sensitive_edges",
    "unknown_edges", "missing_relation_edges",
]
COMPONENT_PROGRAM_FIELDS = [
    "program", "component_count", "non_singleton_components", "mean_component_size",
    "median_component_size", "p90_component_size", "max_component_size", "singleton_ratio",
    "size_le_3_ratio", "size_ge_8_count",
]
COVERAGE_FIELDS = [
    "program", "total_active_passes", "certified_covered", "heuristic_covered",
    "unresolved_conflict", "validation_rejected", "unvalidated_covered", "failed_or_unknown",
    "terminal_due_max_depth", "dropped_active_passes",
]
REDUCTION_STATE_FIELDS = [
    "program", "state_id", "depth", "active_passes", "naive_orderings_log10",
    "batch_candidates", "certified_batches", "executable_batches", "sampled_batches",
    "bounded_batches", "rejected_batches", "failed_batches", "unvalidated_batches",
    "local_reduction_log10", "no_executable_batches", "selected_on_final_path",
]


def summarize_advisor_metrics(study_dir: Path) -> dict:
    study_dir = Path(study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)
    missing: list[dict] = []
    runs = _discover_runs(study_dir)
    study_run_map = {row.get("program", ""): row for row in _read_csv(study_dir / "study_runs.csv")}

    program_rows: list[dict] = []
    pair_state_rows: list[dict] = []
    tagged_pairs: list[dict] = []
    overlap_components: list[dict] = []
    conflict_components: list[dict] = []
    coverage_rows: list[dict] = []
    reduction_rows: list[dict] = []
    cost_program_rows: list[dict] = []
    cost_state_rows: list[dict] = []
    cache_rows: list[dict] = []
    effect_rows: list[dict] = []
    flip_rows: list[dict] = []
    state_records: dict[tuple[str, str], dict] = {}

    for program, run_dir in runs:
        collected = _collect_run(program, run_dir, study_run_map.get(program, {}), missing)
        program_rows.append(collected["program_summary"])
        pair_state_rows.extend(collected["pair_states"])
        tagged_pairs.extend(collected["pairs"])
        overlap_components.extend(collected["overlap_components"])
        conflict_components.extend(collected["conflict_components"])
        coverage_rows.append(collected["coverage"])
        reduction_rows.extend(collected["reductions"])
        cost_program_rows.append(collected["cost_program"])
        cost_state_rows.extend(collected["cost_states"])
        cache_rows.append(collected["cache"])
        effect_rows.extend(collected["effects"])
        flip_rows.extend(collected["flips"])
        state_records.update(collected["state_records"])

    pair_program_rows = _pair_program_rows(pair_state_rows)
    equality_rows = _equality_rows(tagged_pairs)
    failure_rows = _pair_failure_rows(tagged_pairs)
    overlap_program_rows = _component_program_rows(overlap_components)
    conflict_program_rows = _component_program_rows(conflict_components)
    overlap_bucket_rows = _component_bucket_rows(overlap_components)
    conflict_bucket_rows = _component_bucket_rows(conflict_components)
    pair_map = _pair_map_by_state(tagged_pairs)
    overlap_small = small_cluster_abba_summary("overlap", overlap_components, pair_map)
    conflict_small = small_cluster_abba_summary("conflict", conflict_components, pair_map)
    unknown_rows = _unknown_failure_rows(program_rows, tagged_pairs, state_records)
    reduction_program_rows = _reduction_program_rows(reduction_rows, coverage_rows)
    top_conflict_rows = _top_conflict_pass_rows(conflict_components, state_records)
    depth_rows = _state_aware_depth_rows(state_records, effect_rows, flip_rows, overlap_components, conflict_components)

    _write_csv(study_dir / "program_summary.csv", _program_summary_fields(), program_rows)
    _write_csv(study_dir / "pair_relation_summary.csv", _pair_program_fields(), pair_program_rows)
    _write_csv(study_dir / "pair_relation_by_state.csv", PAIR_STATE_FIELDS, pair_state_rows)
    _write_csv(study_dir / "equality_tier_summary_all.csv", ["program", "equality_tier", "count", "percentage"], equality_rows)
    _write_csv(study_dir / "pair_failure_summary.csv", ["program", "failure_kind", "count", "percentage"], failure_rows)
    _write_csv(study_dir / "overlap_components_by_state.csv", COMPONENT_FIELDS, overlap_components)
    _write_csv(study_dir / "overlap_component_program_summary.csv", COMPONENT_PROGRAM_FIELDS, overlap_program_rows)
    _write_csv(study_dir / "overlap_component_size_buckets.csv", ["program", "size_bucket", "components", "percentage"], overlap_bucket_rows)
    _write_csv(study_dir / "conflict_components_by_state.csv", COMPONENT_FIELDS, conflict_components)
    _write_csv(study_dir / "conflict_component_program_summary.csv", COMPONENT_PROGRAM_FIELDS, conflict_program_rows)
    _write_csv(study_dir / "conflict_component_size_buckets.csv", ["program", "size_bucket", "components", "percentage"], conflict_bucket_rows)
    _write_csv(study_dir / "small_overlap_cluster_abba.csv", _small_cluster_fields(), overlap_small)
    _write_csv(study_dir / "small_conflict_cluster_abba.csv", _small_cluster_fields(), conflict_small)
    _write_csv(study_dir / "unknown_failure_summary.csv", _unknown_fields(), unknown_rows)
    _write_csv(study_dir / "coverage_summary_all.csv", COVERAGE_FIELDS, coverage_rows)
    _write_csv(study_dir / "batch_reduction_by_state.csv", REDUCTION_STATE_FIELDS, reduction_rows)
    _write_csv(study_dir / "batch_reduction_program_summary.csv", _reduction_program_fields(), reduction_program_rows)
    _write_csv(study_dir / "cost_breakdown_by_program.csv", _cost_program_fields(), cost_program_rows)
    _write_csv(study_dir / "cost_breakdown_by_state.csv", _cost_state_fields(), cost_state_rows)
    _write_csv(study_dir / "cache_reuse_summary.csv", _cache_fields(), cache_rows)
    _write_csv(study_dir / "top_conflict_passes.csv", _top_conflict_fields(), top_conflict_rows, sort_rows=False)
    _write_csv(study_dir / "state_transition_effects.csv", _effect_fields(), effect_rows)
    _write_csv(study_dir / "state_aware_by_depth.csv", _depth_fields(), depth_rows)
    _write_csv(study_dir / "relation_flip_examples.csv", _flip_fields(), flip_rows[:50])
    _write_csv(study_dir / "missing_outputs.csv", ["program", "state_id", "output", "reason"], _dedupe_missing(missing))
    return {
        "programs": len(runs),
        "states": len(state_records),
        "pair_rows": len(tagged_pairs),
        "missing_outputs": len(_dedupe_missing(missing)),
    }


def summarize_pair_relations(rows: list[dict], *, expected_pairs: int | None = None) -> dict:
    commute = sum(1 for row in rows if row.get("final_relation") == "final_commute")
    sensitive = sum(1 for row in rows if row.get("final_relation") == "final_order_sensitive")
    explicit_unknown = sum(1 for row in rows if _relation_kind(row) == "unknown")
    missing = max(0, expected_pairs - len(rows)) if expected_pairs is not None else 0
    unknown = explicit_unknown + missing
    denominator = commute + sensitive + unknown
    return {
        "total_pairs": expected_pairs if expected_pairs is not None else len(rows),
        "commute_pairs": commute,
        "order_sensitive_pairs": sensitive,
        "unknown_pairs": unknown,
        "commute_ratio": commute / denominator if denominator else 0.0,
        "order_sensitive_ratio": sensitive / denominator if denominator else 0.0,
        "unknown_ratio": unknown / denominator if denominator else 0.0,
        "canonical_hash_commute": sum(1 for row in rows if row.get("final_relation") == "final_commute" and row.get("equality_tier") == "canonical_hash"),
        "structural_diff_commute": sum(1 for row in rows if row.get("final_relation") == "final_commute" and row.get("equality_tier") == "structural_diff"),
        "comparator_failed": sum(1 for row in rows if _is_comparator_failure(row)),
        "timeout_pairs": sum(1 for row in rows if "timeout" in str(row.get("failure_kind", "")).lower()),
        "not_tested_pairs": missing + sum(1 for row in rows if _is_not_tested(row)),
    }


def connected_components(nodes: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    adjacency = {node: set() for node in nodes}
    for left, right in edges:
        if left in adjacency and right in adjacency and left != right:
            adjacency[left].add(right)
            adjacency[right].add(left)
    remaining = set(nodes)
    components = []
    for start in nodes:
        if start not in remaining:
            continue
        remaining.remove(start)
        stack = [start]
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda item: (-len(item), item))


def component_statistics(components: list[list[str]]) -> dict:
    sizes = [len(component) for component in components]
    if not sizes:
        return {
            "component_count": 0, "non_singleton_components": 0, "mean_component_size": 0,
            "median_component_size": 0, "p90_component_size": 0, "max_component_size": 0,
            "singleton_ratio": 0, "size_le_3_ratio": 0, "size_ge_8_count": 0,
        }
    return {
        "component_count": len(sizes),
        "non_singleton_components": sum(1 for size in sizes if size > 1),
        "mean_component_size": statistics.mean(sizes),
        "median_component_size": statistics.median(sizes),
        "p90_component_size": percentile_nearest_rank(sizes, 0.9),
        "max_component_size": max(sizes),
        "singleton_ratio": sum(1 for size in sizes if size == 1) / len(sizes),
        "size_le_3_ratio": sum(1 for size in sizes if size <= 3) / len(sizes),
        "size_ge_8_count": sum(1 for size in sizes if size >= 8),
    }


def percentile_nearest_rank(values: list[int | float], quantile: float) -> int | float:
    if not values:
        return 0
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def local_reduction_metrics(active_passes: int, executable_batches: int) -> dict:
    if active_passes <= 1:
        naive = reduction = 0.0
    else:
        naive = math.lgamma(active_passes + 1) / math.log(10)
        reduction = naive - math.log10(max(1, executable_batches))
    return {
        "naive_orderings_log10": _fmt(naive, 6),
        "local_reduction_log10": _fmt(reduction, 6),
    }


def small_cluster_abba_summary(
    cluster_type: str,
    component_rows: list[dict],
    pair_rows_by_state: dict[tuple[str, str], list[dict]],
    *,
    component_state_keys: list[tuple[str, str]] | None = None,
) -> list[dict]:
    buckets: dict[str, dict] = {}
    for index, component in enumerate(component_rows):
        size = _int(component.get("component_size"))
        if size < 2:
            continue
        bucket_name = _size_bucket(size, small=True)
        bucket = buckets.setdefault(bucket_name, _empty_small_bucket(cluster_type, bucket_name))
        key = component_state_keys[index] if component_state_keys else (component.get("program", ""), component.get("state_id", ""))
        pair_map = {_pair_key(row.get("pass_a", ""), row.get("pass_b", "")): row for row in pair_rows_by_state.get(key, [])}
        passes = _split_passes(component.get("component_passes", ""))
        bucket["components"] += 1
        for left, right in itertools.combinations(passes, 2):
            bucket["internal_pairs"] += 1
            row = pair_map.get(_pair_key(left, right))
            if row is None or _relation_kind(row) == "unknown":
                bucket["unknown_pairs"] += 1
                if row:
                    bucket["pair_test_time_ms"] += _float(row.get("time_ms"))
                continue
            bucket["tested_pairs"] += 1
            bucket["pair_test_time_ms"] += _float(row.get("time_ms"))
            if row.get("final_relation") == "final_commute":
                bucket["commute_pairs"] += 1
            elif row.get("final_relation") == "final_order_sensitive":
                bucket["order_sensitive_pairs"] += 1
    rows = []
    for name in ["2", "3", "4-5", "6-7", "8-10", ">10"]:
        if name not in buckets:
            continue
        bucket = buckets[name]
        known = bucket["commute_pairs"] + bucket["order_sensitive_pairs"]
        bucket["ab_ba_equal_ratio"] = _fmt(bucket["commute_pairs"] / max(1, known), 6)
        bucket["unknown_ratio"] = _fmt(bucket["unknown_pairs"] / max(1, bucket["internal_pairs"]), 6)
        bucket["pair_test_time_ms"] = _fmt(bucket["pair_test_time_ms"], 3)
        rows.append({key: str(value) for key, value in bucket.items()})
    return rows


def _collect_run(program: str, run_dir: Path, study_run: dict, missing: list[dict]) -> dict:
    states_path = run_dir / "states.csv"
    states = [row for row in _read_required(states_path, program, "", missing) if not _is_true(row.get("is_duplicate"))]
    metadata = _read_json(run_dir / "metadata.json")
    state_records: dict[tuple[str, str], dict] = {}
    pair_states = []
    tagged_pairs = []
    overlap_components = []
    conflict_components = []
    reductions = []
    cost_states = []
    coverage_source = []
    coverage_complete = True
    selected_states = _selected_states(run_dir)

    for state in sorted(states, key=lambda row: (_int(row.get("depth")), row.get("state_id", ""))):
        state_id = state.get("state_id", "")
        depth = _int(state.get("depth"))
        state_dir = run_dir / "states" / state_id
        profiles = _read_required(state_dir / "pass_profile.csv", program, state_id, missing)
        active = sorted({row.get("pass", "") for row in profiles if row.get("pass") and _is_true(row.get("success")) and _is_true(row.get("active"))})
        pairs = _read_required(state_dir / "pair_relation.csv", program, state_id, missing)
        expected_pairs = len(active) * (len(active) - 1) // 2
        relation = summarize_pair_relations(pairs, expected_pairs=expected_pairs)
        pair_state = {
            "program": program, "state_id": state_id, "depth": str(depth), "active_passes": str(len(active)),
            **{key: str(relation[key]) for key in ("total_pairs", "commute_pairs", "order_sensitive_pairs", "unknown_pairs", "canonical_hash_commute", "structural_diff_commute", "comparator_failed", "timeout_pairs", "not_tested_pairs")},
            "commute_ratio": _fmt(relation["commute_ratio"], 6),
        }
        pair_states.append(pair_state)
        for pair in pairs:
            tagged_pairs.append({**pair, "program": program, "state_id": state_id, "depth": str(depth)})

        overlap_path = state_dir / "footprint_overlap.csv"
        if not overlap_path.exists() and profiles:
            try:
                build_footprint_overlap(state_dir)
            except Exception as exc:
                _missing(missing, program, state_id, overlap_path, f"derive_failed: {_one_line(exc)}")
        overlaps = _read_required(overlap_path, program, state_id, missing)
        overlap_components.extend(_build_component_rows("overlap", program, state_id, depth, active, pairs, overlaps))
        conflict_components.extend(_build_component_rows("conflict", program, state_id, depth, active, pairs, overlaps))

        ladder = _first_row(state_dir / "batch_validation_ladder_summary.csv")
        if not ladder:
            _missing(missing, program, state_id, state_dir / "batch_validation_ladder_summary.csv", "missing_or_empty")
        candidates = _read_required(state_dir / "batch_candidates.csv", program, state_id, missing)
        executable = _int_or_none(ladder.get("executable_batches"))
        reduction_metric = local_reduction_metrics(len(active), executable or 0)
        reductions.append(
            {
                "program": program, "state_id": state_id, "depth": str(depth), "active_passes": str(len(active)),
                "naive_orderings_log10": reduction_metric["naive_orderings_log10"],
                "batch_candidates": str(len(candidates)) if candidates or (state_dir / "batch_candidates.csv").exists() else "",
                "certified_batches": _value(ladder, "hard_certified_batches"),
                "executable_batches": _value(ladder, "executable_batches"),
                "sampled_batches": _value(ladder, "sampled_batches"),
                "bounded_batches": _value(ladder, "bounded_batches"),
                "rejected_batches": _value(ladder, "rejected_batches"),
                "failed_batches": _value(ladder, "failed_batches"),
                "unvalidated_batches": _value(ladder, "unvalidated_batches"),
                "local_reduction_log10": reduction_metric["local_reduction_log10"] if executable is not None else "",
                "no_executable_batches": _bool(executable == 0) if executable is not None else "",
                "selected_on_final_path": _bool(state_id in selected_states),
            }
        )

        coverage = _first_row(state_dir / "coverage_summary.csv")
        if coverage:
            coverage_source.append(coverage)
        else:
            coverage_complete = False
            _missing(missing, program, state_id, state_dir / "coverage_summary.csv", "missing_or_empty")
        per_state = _first_row(state_dir / "per_state_summary.csv")
        if not per_state:
            _missing(missing, program, state_id, state_dir / "per_state_summary.csv", "missing_or_empty")
        cost_states.append(
            {
                "program": program, "state_id": state_id, "depth": str(depth),
                "profiling_wall_ms": _value(per_state, "profile_time_ms"),
                "pair_testing_wall_ms": _value(per_state, "pair_time_ms"),
                "batch_validation_wall_ms": _value(ladder, "validation_time_ms"),
                "state_total_wall_ms": _value(per_state, "total_time_ms"),
                "ir_equality_cumulative_work_ms": _sum_field(pairs, "comparator_time_ms"),
            }
        )
        state_records[(program, state_id)] = {
            "program": program, "state_id": state_id, "depth": depth, "state": state,
            "run_dir": run_dir, "profiles": profiles, "active": active, "pairs": pairs,
        }

    effects, flips = _derive_state_changes(program, states, state_records)
    coverage_row = _coverage_program_row(program, coverage_source, coverage_complete)
    timing = _first_row(run_dir / "optimizer_timing.csv")
    if not timing:
        _missing(missing, program, "", run_dir / "optimizer_timing.csv", "missing_or_empty")
    pair_cost = _first_row(run_dir / "pair_cost_summary.csv")
    if not pair_cost:
        _missing(missing, program, "", run_dir / "pair_cost_summary.csv", "missing_or_empty")
    validation_rows = [_first_row(run_dir / "states" / state.get("state_id", "") / "batch_validation_ladder_summary.csv") for state in states]
    cost_program, cache = _cost_rows(program, timing, pair_cost, validation_rows, study_run)
    program_summary = _program_summary_row(program, run_dir, states, pair_states, coverage_row, timing, study_run)
    return {
        "program_summary": program_summary,
        "pair_states": pair_states,
        "pairs": tagged_pairs,
        "overlap_components": overlap_components,
        "conflict_components": conflict_components,
        "coverage": coverage_row,
        "reductions": reductions,
        "cost_program": cost_program,
        "cost_states": cost_states,
        "cache": cache,
        "effects": effects,
        "flips": flips,
        "state_records": state_records,
    }


def _build_component_rows(kind: str, program: str, state_id: str, depth: int, active: list[str], pairs: list[dict], overlaps: list[dict]) -> list[dict]:
    overlap_map = {_pair_key(row.get("pass_a", ""), row.get("pass_b", "")): row for row in overlaps}
    pair_map = {_pair_key(row.get("pass_a", ""), row.get("pass_b", "")): row for row in pairs}
    edges = []
    for left, right in itertools.combinations(active, 2):
        key = _pair_key(left, right)
        if kind == "overlap":
            overlap_kind = overlap_map.get(key, {}).get("overlap_kind", "")
            if overlap_kind in {"same_function_overlap", "same_block_overlap", "possible_ww_overlap"}:
                edges.append((left, right))
        else:
            pair = pair_map.get(key)
            if pair is None or pair.get("final_relation") != "final_commute":
                edges.append((left, right))
    components = connected_components(active, edges)
    rows = []
    for index, passes in enumerate(components):
        inside = {_pair_key(left, right) for left, right in itertools.combinations(passes, 2)}
        overlap_inside = [row for key, row in overlap_map.items() if key in inside]
        pair_inside = [(key, pair_map.get(key)) for key in inside]
        rows.append(
            {
                "program": program, "state_id": state_id, "depth": str(depth),
                "component_id": f"C{index:04d}", "component_size": str(len(passes)),
                "component_passes": ";".join(passes),
                "same_function_edges": str(sum(1 for row in overlap_inside if row.get("overlap_kind") == "same_function_overlap")),
                "same_block_edges": str(sum(1 for row in overlap_inside if row.get("overlap_kind") == "same_block_overlap")),
                "possible_ww_edges": str(sum(1 for row in overlap_inside if row.get("overlap_kind") == "possible_ww_overlap")),
                "order_sensitive_edges": str(sum(1 for _key, row in pair_inside if row and row.get("final_relation") == "final_order_sensitive")),
                "unknown_edges": str(sum(1 for _key, row in pair_inside if row and _relation_kind(row) == "unknown")),
                "missing_relation_edges": str(sum(1 for _key, row in pair_inside if row is None)),
            }
        )
    return rows


def _program_summary_row(program: str, run_dir: Path, states: list[dict], pair_states: list[dict], coverage: dict, timing: dict, study_run: dict) -> dict:
    active_values = [_int(row.get("active_passes")) for row in pair_states]
    commute = sum(_int(row.get("commute_pairs")) for row in pair_states)
    sensitive = sum(_int(row.get("order_sensitive_pairs")) for row in pair_states)
    unknown = sum(_int(row.get("unknown_pairs")) for row in pair_states)
    total = commute + sensitive + unknown
    chosen = _read_csv(run_dir / "chosen_path.csv")
    final_state = chosen[-1].get("child_state_id", "") if chosen else _read_text(run_dir / "final_state.txt")
    final_row = next((row for row in states if row.get("state_id") == final_state), states[-1] if states else {})
    chosen_summary = _first_row(run_dir / "chosen_path_summary.csv")
    pipeline = _read_text(run_dir / "optimized_pipeline.txt")
    pipeline_parts = [part.strip() for part in pipeline.replace("\n", ",").split(",") if part.strip()]
    total_wall = study_run.get("total_time_ms") or timing.get("optimizer_total_time_ms", "")
    return {
        "program": program,
        "input_path": str(_read_json(run_dir / "metadata.json").get("input", "")),
        "valid_passes": str(len(_read_csv(run_dir / "valid_passes.csv"))) if (run_dir / "valid_passes.csv").exists() else "",
        "invalid_passes": str(len(_read_csv(run_dir / "invalid_passes.csv"))) if (run_dir / "invalid_passes.csv").exists() else "",
        "states": str(len(states)),
        "transitions": str(len(_read_csv(run_dir / "state_dag.csv"))) if (run_dir / "state_dag.csv").exists() else "",
        "max_depth": str(max((_int(row.get("depth")) for row in states), default=0)),
        "avg_active_passes": _mean(active_values),
        "median_active_passes": _median(active_values),
        "max_active_passes": str(max(active_values, default=0)),
        "total_pair_rows": str(total), "commute_pairs": str(commute), "order_sensitive_pairs": str(sensitive), "unknown_pairs": str(unknown),
        "commute_ratio": _fmt(commute / total if total else 0, 6),
        "order_sensitive_ratio": _fmt(sensitive / total if total else 0, 6),
        "unknown_ratio": _fmt(unknown / total if total else 0, 6),
        "dropped_active_passes": coverage.get("dropped_active_passes", ""),
        "selected_path_steps": str(len(chosen)), "final_pipeline_length": str(len(pipeline_parts)),
        "final_ir_inst_count": chosen_summary.get("final_ir_inst_count") or final_row.get("ir_instructions", ""),
        "total_wall_time_ms": str(total_wall),
    }


def _coverage_program_row(program: str, rows: list[dict], complete: bool) -> dict:
    fields = {
        "total_active_passes": "active_passes", "certified_covered": "certified_covered",
        "heuristic_covered": "heuristic_covered", "unresolved_conflict": "unresolved_conflict",
        "validation_rejected": "validation_rejected", "unvalidated_covered": "unvalidated_covered",
        "failed_or_unknown": "failed_or_unknown", "terminal_due_max_depth": "not_executed_due_to_max_depth",
        "dropped_active_passes": "dropped_active_passes",
    }
    result = {"program": program}
    for output, source in fields.items():
        result[output] = str(sum(_int(row.get(source)) for row in rows)) if complete else ""
    return result


def _cost_rows(program: str, timing: dict, pair_cost: dict, validations: list[dict], study_run: dict) -> tuple[dict, dict]:
    total = _float_or_none(study_run.get("total_time_ms") or timing.get("optimizer_total_time_ms"))
    profiling = _float_or_none(timing.get("profiling_time_ms"))
    pair = _float_or_none(timing.get("pair_testing_time_ms"))
    validation = _float_or_none(timing.get("batch_validation_time_ms"))
    apply = _float_or_none(timing.get("batch_apply_time_ms"))
    known = sum(value for value in [profiling, pair, validation, apply] if value is not None)
    baseline = _int_or_none(pair_cost.get("pair_test_pass_invocations_baseline"))
    actual = _int_or_none(pair_cost.get("pair_test_pass_invocations_actual"))
    saved = _int_or_none(pair_cost.get("pair_test_pass_invocations_saved"))
    for row in validations:
        baseline = _add_optional(baseline, _int_or_none(row.get("validation_pass_invocations_baseline")))
        actual = _add_optional(actual, _int_or_none(row.get("validation_pass_invocations_actual")))
        saved = _add_optional(saved, _int_or_none(row.get("validation_pass_invocations_saved")))
    transition_hits = sum(_int(row.get("validation_transition_cache_hits") or row.get("validation_state_transition_cache_hits")) for row in validations)
    equivalence_hits = sum(_int(row.get("validation_equivalence_cache_hits") or row.get("validation_state_equivalence_cache_hits")) for row in validations)
    cost = {
        "program": program, "total_wall_time_ms": _optional_fmt(total),
        "input_prepare_wall_ms": "", "pass_validation_wall_ms": "",
        "profiling_wall_ms": _optional_fmt(profiling), "pair_testing_wall_ms": _optional_fmt(pair),
        "ir_equality_wall_ms": "", "batch_construction_wall_ms": "",
        "batch_validation_wall_ms": _optional_fmt(validation), "state_search_wall_ms": "", "replay_wall_ms": "",
        "other_wall_ms": _optional_fmt(max(0.0, total - known) if total is not None else None),
        "opt_process_invocations": str(timing.get("total_opt_invocations", "")),
        "pass_invocations_baseline": _optional_int(baseline), "pass_invocations_actual": _optional_int(actual),
        "pass_invocations_saved": _optional_int(saved),
        "pair_cache_hits": str(pair_cost.get("cache_hits", "")), "pair_cache_misses": str(pair_cost.get("cache_misses", "")),
        "validation_transition_cache_hits": str(transition_hits), "validation_equivalence_cache_hits": str(equivalence_hits),
        "ir_equality_cumulative_work_ms": str(pair_cost.get("comparator_time_ms", "")),
    }
    cache = {
        "program": program, "pair_cache_hits": cost["pair_cache_hits"], "pair_cache_misses": cost["pair_cache_misses"],
        "validation_transition_cache_hits": str(transition_hits), "validation_equivalence_cache_hits": str(equivalence_hits),
        "pass_invocations_baseline": cost["pass_invocations_baseline"], "pass_invocations_actual": cost["pass_invocations_actual"],
        "pass_invocations_saved": cost["pass_invocations_saved"],
    }
    return cost, cache


def _derive_state_changes(program: str, states: list[dict], records: dict[tuple[str, str], dict]) -> tuple[list[dict], list[dict]]:
    effects = []
    flips = []
    for state in states:
        child_id = state.get("state_id", "")
        parent_id = state.get("parent_state_id", "")
        if not parent_id or (program, parent_id) not in records or (program, child_id) not in records:
            continue
        parent = records[(program, parent_id)]
        child = records[(program, child_id)]
        parent_profiles = {row.get("pass", ""): row for row in parent["profiles"] if row.get("pass")}
        child_profiles = {row.get("pass", ""): row for row in child["profiles"] if row.get("pass")}
        for pass_name in sorted(set(parent_profiles) | set(child_profiles)):
            before = parent_profiles.get(pass_name)
            after = child_profiles.get(pass_name)
            before_active = bool(before and _is_true(before.get("active")) and _is_true(before.get("success")))
            after_active = bool(after and _is_true(after.get("active")) and _is_true(after.get("success")))
            kind = ""
            if not before_active and after_active:
                kind = "enable"
            elif before_active and not after_active:
                kind = "suppress"
            elif before_active and after_active and _effect_signature(before) != _effect_signature(after):
                kind = "effect_changed"
            if kind:
                effects.append(_effect_row(program, parent_id, child_id, child["depth"], "pass", pass_name, "", kind))

        parent_pairs = {_pair_key(row.get("pass_a", ""), row.get("pass_b", "")): row for row in parent["pairs"]}
        child_pairs = {_pair_key(row.get("pass_a", ""), row.get("pass_b", "")): row for row in child["pairs"]}
        for pair in sorted(set(parent_pairs) | set(child_pairs)):
            before = parent_pairs.get(pair)
            after = child_pairs.get(pair)
            flip = _classify_pair_change(before, after)
            if not flip:
                continue
            row = {
                "program": program, "parent_state_id": parent_id, "child_state_id": child_id,
                "depth": str(child["depth"]), "pass_a": pair[0], "pass_b": pair[1],
                "parent_relation": before.get("final_relation", "") if before else "",
                "child_relation": after.get("final_relation", "") if after else "", "flip_kind": flip,
            }
            flips.append(row)
            effects.append(_effect_row(program, parent_id, child_id, child["depth"], "pair", pair[0], pair[1], flip))
    return effects, flips


def _classify_pair_change(before: dict | None, after: dict | None) -> str:
    if before is None or after is None:
        return "pair_availability_change"
    left = before.get("final_relation", "")
    right = after.get("final_relation", "")
    if left == right:
        return ""
    if left == "final_commute" and right == "final_order_sensitive":
        return "commute_to_sensitive"
    if left == "final_order_sensitive" and right == "final_commute":
        return "sensitive_to_commute"
    if left in {"final_commute", "final_order_sensitive"} and _relation_kind(after) == "unknown":
        return "known_to_unknown"
    if _relation_kind(before) == "unknown" and right in {"final_commute", "final_order_sensitive"}:
        return "unknown_to_known"
    return ""


def _state_aware_depth_rows(records: dict, effects: list[dict], flips: list[dict], overlap: list[dict], conflict: list[dict]) -> list[dict]:
    rows = []
    programs = sorted({key[0] for key in records})
    for program in programs:
        depths = sorted({record["depth"] for key, record in records.items() if key[0] == program})
        for depth in depths:
            states = [record for key, record in records.items() if key[0] == program and record["depth"] == depth]
            pair_stats = [summarize_pair_relations(record["pairs"], expected_pairs=len(record["active"]) * (len(record["active"]) - 1) // 2) for record in states]
            depth_effects = [row for row in effects if row["program"] == program and _int(row["depth"]) == depth]
            depth_flips = [row for row in flips if row["program"] == program and _int(row["depth"]) == depth]
            overlap_sizes = [_int(row["component_size"]) for row in overlap if row["program"] == program and _int(row["depth"]) == depth]
            conflict_sizes = [_int(row["component_size"]) for row in conflict if row["program"] == program and _int(row["depth"]) == depth]
            rows.append(
                {
                    "program": program, "depth": str(depth), "states": str(len(states)),
                    "avg_active_passes": _mean([len(record["active"]) for record in states]),
                    "avg_commute_ratio": _fmt(statistics.mean([item["commute_ratio"] for item in pair_stats]) if pair_stats else 0, 6),
                    "enable_count": str(sum(1 for row in depth_effects if row["effect_kind"] == "enable")),
                    "suppress_count": str(sum(1 for row in depth_effects if row["effect_kind"] == "suppress")),
                    "effect_changed_count": str(sum(1 for row in depth_effects if row["effect_kind"] == "effect_changed")),
                    "true_relation_flip_count": str(sum(1 for row in depth_flips if row["flip_kind"] != "pair_availability_change")),
                    "pair_availability_change_count": str(sum(1 for row in depth_flips if row["flip_kind"] == "pair_availability_change")),
                    "avg_overlap_component_size": _mean(overlap_sizes), "max_overlap_component_size": str(max(overlap_sizes, default=0)),
                    "avg_conflict_component_size": _mean(conflict_sizes), "max_conflict_component_size": str(max(conflict_sizes, default=0)),
                }
            )
    return rows


def _pair_program_rows(rows: list[dict]) -> list[dict]:
    result = []
    for program in sorted({row["program"] for row in rows}):
        selected = [row for row in rows if row["program"] == program]
        totals = {field: sum(_int(row.get(field)) for row in selected) for field in ["total_pairs", "commute_pairs", "order_sensitive_pairs", "unknown_pairs", "canonical_hash_commute", "structural_diff_commute", "comparator_failed", "timeout_pairs", "not_tested_pairs"]}
        denominator = totals["commute_pairs"] + totals["order_sensitive_pairs"] + totals["unknown_pairs"]
        result.append({"program": program, **{key: str(value) for key, value in totals.items()}, "commute_ratio": _fmt(totals["commute_pairs"] / denominator if denominator else 0, 6), "order_sensitive_ratio": _fmt(totals["order_sensitive_pairs"] / denominator if denominator else 0, 6), "unknown_ratio": _fmt(totals["unknown_pairs"] / denominator if denominator else 0, 6)})
    return result


def _component_program_rows(rows: list[dict]) -> list[dict]:
    result = []
    for program in sorted({row["program"] for row in rows}):
        selected = [row for row in rows if row["program"] == program]
        stats = component_statistics([_split_passes(row["component_passes"]) for row in selected])
        result.append({"program": program, **{key: _fmt(value, 6) if isinstance(value, float) else str(value) for key, value in stats.items()}})
    return result


def _component_bucket_rows(rows: list[dict]) -> list[dict]:
    result = []
    for program in sorted({row["program"] for row in rows}):
        selected = [row for row in rows if row["program"] == program]
        counts = Counter(_size_bucket(_int(row["component_size"])) for row in selected)
        for bucket in ["1", "2", "3", "4-5", "6-7", "8-10", ">10"]:
            result.append({"program": program, "size_bucket": bucket, "components": str(counts[bucket]), "percentage": _fmt(counts[bucket] / max(1, len(selected)), 6)})
    return result


def _reduction_program_rows(rows: list[dict], coverage_rows: list[dict]) -> list[dict]:
    result = []
    coverage = {row.get("program", ""): row for row in coverage_rows}
    for program in sorted({row["program"] for row in rows}):
        selected = [row for row in rows if row["program"] == program]
        reductions = [_float(row["local_reduction_log10"]) for row in selected if row.get("local_reduction_log10") != ""]
        result.append(
            {
                "program": program, "avg_active_passes": _mean([_int(row["active_passes"]) for row in selected]),
                "avg_naive_orderings_log10": _mean([_float(row["naive_orderings_log10"]) for row in selected]),
                "avg_batch_candidates": _mean([_int(row["batch_candidates"]) for row in selected if row.get("batch_candidates") != ""]),
                "avg_executable_batches": _mean([_int(row["executable_batches"]) for row in selected if row.get("executable_batches") != ""]),
                "avg_local_reduction_log10": _mean(reductions), "median_local_reduction_log10": _median(reductions),
                "max_local_reduction_log10": _fmt(max(reductions), 6) if reductions else "",
                "total_certified_batches": _sum_present(selected, "certified_batches"),
                "total_dropped_active_passes": coverage.get(program, {}).get("dropped_active_passes", ""),
            }
        )
    return result


def _equality_rows(rows: list[dict]) -> list[dict]:
    result = []
    for program in sorted({row["program"] for row in rows}):
        selected = [row for row in rows if row["program"] == program]
        counts = Counter(_equality_bucket(row) for row in selected)
        for tier in ["canonical_hash", "structural_diff", "different", "failed"]:
            result.append({"program": program, "equality_tier": tier, "count": str(counts[tier]), "percentage": _fmt(counts[tier] / max(1, len(selected)), 6)})
    return result


def _pair_failure_rows(rows: list[dict]) -> list[dict]:
    result = []
    for program in sorted({row["program"] for row in rows}):
        selected = [row for row in rows if row["program"] == program]
        counts = Counter(row.get("failure_kind", "") for row in selected if row.get("failure_kind"))
        for kind in sorted(counts):
            result.append({"program": program, "failure_kind": kind, "count": str(counts[kind]), "percentage": _fmt(counts[kind] / max(1, len(selected)), 6)})
    return result


def _unknown_failure_rows(programs: list[dict], pairs: list[dict], records: dict) -> list[dict]:
    result = []
    for summary in programs:
        program = summary["program"]
        selected = [row for row in pairs if row["program"] == program]
        run_dir = next((record["run_dir"] for key, record in records.items() if key[0] == program), None)
        validation = []
        if run_dir:
            for path in sorted((run_dir / "states").glob("*/batch_validation_ladder_summary.csv")):
                row = _first_row(path)
                if row:
                    validation.append(row)
        result.append(
            {
                "program": program, "pair_unknown": summary.get("unknown_pairs", ""),
                "pair_timeout": str(sum(1 for row in selected if "timeout" in str(row.get("failure_kind", "")).lower())),
                "pair_opt_failed": str(sum(1 for row in selected if _is_pair_opt_failure(row))),
                "comparator_failed": str(sum(1 for row in selected if _is_comparator_failure(row))),
                "max_pairs_skipped": str(sum(1 for row in selected if row.get("failure_kind") == "max_pairs")),
                "lazy_budget_skipped": str(sum(1 for row in selected if _is_true(row.get("skipped_by_budget")))),
                "batch_mismatch": str(sum(_int(row.get("rejected_batches")) for row in validation)),
                "batch_validation_failed": str(sum(_int(row.get("failed_batches")) for row in validation)),
                "batch_unvalidated": str(sum(_int(row.get("unvalidated_batches")) for row in validation)),
                "invalid_passes": summary.get("invalid_passes", ""),
            }
        )
    return result


def _top_conflict_pass_rows(components: list[dict], records: dict) -> list[dict]:
    data: dict[str, dict] = defaultdict(lambda: {"programs": set(), "states": set(), "memberships": 0, "non_singleton": 0, "sizes": [], "sensitive": 0, "unknown": 0, "component_weight": 0})
    for component in components:
        size = _int(component["component_size"])
        for pass_name in _split_passes(component["component_passes"]):
            item = data[pass_name]
            item["programs"].add(component["program"])
            item["states"].add((component["program"], component["state_id"]))
            item["memberships"] += 1
            item["non_singleton"] += int(size > 1)
            item["sizes"].append(size)
            item["component_weight"] += max(0, size - 1)
    for (program, state_id), record in records.items():
        active = record["active"]
        pair_map = {_pair_key(row.get("pass_a", ""), row.get("pass_b", "")): row for row in record["pairs"]}
        for left, right in itertools.combinations(active, 2):
            row = pair_map.get(_pair_key(left, right))
            if row and row.get("final_relation") == "final_order_sensitive":
                data[left]["sensitive"] += 1
                data[right]["sensitive"] += 1
            elif row is None or _relation_kind(row) == "unknown":
                data[left]["unknown"] += 1
                data[right]["unknown"] += 1
    rows = []
    for pass_name, item in data.items():
        score = item["sensitive"] + 0.5 * item["unknown"] + item["component_weight"]
        rows.append(
            {
                "pass": pass_name, "programs_present": str(len(item["programs"])), "states_present": str(len(item["states"])),
                "component_memberships": str(item["memberships"]), "non_singleton_component_memberships": str(item["non_singleton"]),
                "max_component_size": str(max(item["sizes"], default=0)), "avg_component_size": _mean(item["sizes"]),
                "order_sensitive_degree": str(item["sensitive"]), "unknown_degree": str(item["unknown"]),
                "weighted_conflict_score": _fmt(score, 6),
            }
        )
    return sorted(rows, key=lambda row: (-_float(row["weighted_conflict_score"]), row["pass"]))[:20]


def _discover_runs(study_dir: Path) -> list[tuple[str, Path]]:
    runs = []
    programs_dir = study_dir / "programs"
    if programs_dir.is_dir():
        for program_dir in sorted(programs_dir.iterdir(), key=lambda path: path.name.lower()):
            run = program_dir / "optimize"
            if run.is_dir():
                runs.append((program_dir.name, run))
    return runs


def _selected_states(run_dir: Path) -> set[str]:
    selected = set()
    for row in _read_csv(run_dir / "chosen_path.csv"):
        selected.update(value for value in [row.get("parent_state_id"), row.get("child_state_id")] if value)
    return selected


def _pair_map_by_state(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    result: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        result[(row["program"], row["state_id"])].append(row)
    return result


def _empty_small_bucket(cluster_type: str, bucket: str) -> dict:
    return {"cluster_type": cluster_type, "size_bucket": bucket, "components": 0, "internal_pairs": 0, "tested_pairs": 0, "commute_pairs": 0, "order_sensitive_pairs": 0, "unknown_pairs": 0, "ab_ba_equal_ratio": 0, "unknown_ratio": 0, "pair_test_time_ms": 0.0}


def _effect_row(program: str, parent: str, child: str, depth: int, entity: str, left: str, right: str, kind: str) -> dict:
    return {"program": program, "parent_state_id": parent, "child_state_id": child, "depth": str(depth), "entity_type": entity, "pass_a": left, "pass_b": right, "effect_kind": kind}


def _effect_signature(row: dict) -> tuple[str, str, str]:
    return (str(row.get("inst_delta", "")), str(row.get("changed_functions", "")), str(row.get("changed_blocks", "")))


def _relation_kind(row: dict) -> str:
    relation = str(row.get("final_relation", ""))
    if relation == "final_commute":
        return "commute"
    if relation == "final_order_sensitive":
        return "sensitive"
    return "unknown"


def _equality_bucket(row: dict) -> str:
    tier = row.get("equality_tier", "")
    if tier in {"canonical_hash", "structural_diff"}:
        return tier
    if tier == "failed" or row.get("failure_kind"):
        return "failed"
    return "different"


def _is_comparator_failure(row: dict) -> bool:
    failure = str(row.get("failure_kind", "")).lower()
    reason = str(row.get("equality_reason", "")).lower()
    return (
        "comparator" in failure
        or "llvm_diff" in failure
        or "llvm-diff" in failure
        or reason in {"tool_failed", "comparator_failed", "llvm_diff_failed", "llvm-diff_failed"}
    )


def _is_pair_opt_failure(row: dict) -> bool:
    if "ab_success" in row or "ba_success" in row:
        return not _is_true(row.get("ab_success")) or not _is_true(row.get("ba_success"))
    failure = str(row.get("failure_kind", "")).lower()
    return failure in {"failed", "opt_failed", "llvm_fatal"} or failure.startswith("opt_")


def _is_not_tested(row: dict) -> bool:
    return row.get("dynamic_relation") == "not_tested" or row.get("final_relation") in {"", "not_tested"} or _is_true(row.get("skipped_by_budget")) or row.get("failure_kind") == "max_pairs"


def _read_required(path: Path, program: str, state_id: str, missing: list[dict]) -> list[dict]:
    if not path.is_file():
        _missing(missing, program, state_id, path, "missing")
        return []
    return _read_csv(path)


def _missing(rows: list[dict], program: str, state_id: str, path: Path, reason: str) -> None:
    rows.append({"program": program, "state_id": state_id, "output": str(path), "reason": reason})


def _dedupe_missing(rows: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for row in sorted(rows, key=lambda item: (item["program"], item["state_id"], item["output"], item["reason"])):
        key = tuple(row[field] for field in ["program", "state_id", "output", "reason"])
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _first_row(path: Path) -> dict:
    rows = _read_csv(path)
    return rows[0] if rows else {}


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _write_csv(path: Path, fields: list[str], rows: list[dict], *, sort_rows: bool = True) -> None:
    ordered = (
        sorted(rows, key=lambda row: tuple(str(row.get(field, "")) for field in ["program", "depth", "state_id", "component_id", "pass", "size_bucket", "equality_tier", "failure_kind"] if field in row))
        if sort_rows
        else list(rows)
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)


def _program_summary_fields() -> list[str]:
    return ["program", "input_path", "valid_passes", "invalid_passes", "states", "transitions", "max_depth", "avg_active_passes", "median_active_passes", "max_active_passes", "total_pair_rows", "commute_pairs", "order_sensitive_pairs", "unknown_pairs", "commute_ratio", "order_sensitive_ratio", "unknown_ratio", "dropped_active_passes", "selected_path_steps", "final_pipeline_length", "final_ir_inst_count", "total_wall_time_ms"]


def _pair_program_fields() -> list[str]:
    return ["program", "total_pairs", "commute_pairs", "order_sensitive_pairs", "unknown_pairs", "commute_ratio", "order_sensitive_ratio", "unknown_ratio", "canonical_hash_commute", "structural_diff_commute", "comparator_failed", "timeout_pairs", "not_tested_pairs"]


def _small_cluster_fields() -> list[str]:
    return ["cluster_type", "size_bucket", "components", "internal_pairs", "tested_pairs", "commute_pairs", "order_sensitive_pairs", "unknown_pairs", "ab_ba_equal_ratio", "unknown_ratio", "pair_test_time_ms"]


def _unknown_fields() -> list[str]:
    return ["program", "pair_unknown", "pair_timeout", "pair_opt_failed", "comparator_failed", "max_pairs_skipped", "lazy_budget_skipped", "batch_mismatch", "batch_validation_failed", "batch_unvalidated", "invalid_passes"]


def _reduction_program_fields() -> list[str]:
    return ["program", "avg_active_passes", "avg_naive_orderings_log10", "avg_batch_candidates", "avg_executable_batches", "avg_local_reduction_log10", "median_local_reduction_log10", "max_local_reduction_log10", "total_certified_batches", "total_dropped_active_passes"]


def _cost_program_fields() -> list[str]:
    return ["program", "total_wall_time_ms", "input_prepare_wall_ms", "pass_validation_wall_ms", "profiling_wall_ms", "pair_testing_wall_ms", "ir_equality_wall_ms", "batch_construction_wall_ms", "batch_validation_wall_ms", "state_search_wall_ms", "replay_wall_ms", "other_wall_ms", "opt_process_invocations", "pass_invocations_baseline", "pass_invocations_actual", "pass_invocations_saved", "pair_cache_hits", "pair_cache_misses", "validation_transition_cache_hits", "validation_equivalence_cache_hits", "ir_equality_cumulative_work_ms"]


def _cost_state_fields() -> list[str]:
    return ["program", "state_id", "depth", "profiling_wall_ms", "pair_testing_wall_ms", "batch_validation_wall_ms", "state_total_wall_ms", "ir_equality_cumulative_work_ms"]


def _cache_fields() -> list[str]:
    return ["program", "pair_cache_hits", "pair_cache_misses", "validation_transition_cache_hits", "validation_equivalence_cache_hits", "pass_invocations_baseline", "pass_invocations_actual", "pass_invocations_saved"]


def _top_conflict_fields() -> list[str]:
    return ["pass", "programs_present", "states_present", "component_memberships", "non_singleton_component_memberships", "max_component_size", "avg_component_size", "order_sensitive_degree", "unknown_degree", "weighted_conflict_score"]


def _effect_fields() -> list[str]:
    return ["program", "parent_state_id", "child_state_id", "depth", "entity_type", "pass_a", "pass_b", "effect_kind"]


def _flip_fields() -> list[str]:
    return ["program", "parent_state_id", "child_state_id", "depth", "pass_a", "pass_b", "parent_relation", "child_relation", "flip_kind"]


def _depth_fields() -> list[str]:
    return ["program", "depth", "states", "avg_active_passes", "avg_commute_ratio", "enable_count", "suppress_count", "effect_changed_count", "true_relation_flip_count", "pair_availability_change_count", "avg_overlap_component_size", "max_overlap_component_size", "avg_conflict_component_size", "max_conflict_component_size"]


def _size_bucket(size: int, small: bool = False) -> str:
    if size <= 3:
        return str(max(1, size))
    if size <= 5:
        return "4-5"
    if size <= 7:
        return "6-7"
    if size <= 10:
        return "8-10"
    return ">10"


def _split_passes(value: object) -> list[str]:
    return sorted({part.strip() for part in str(value or "").replace(",", ";").split(";") if part.strip()})


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def _value(row: dict, field: str) -> str:
    return str(row.get(field, "")) if field in row else ""


def _sum_field(rows: list[dict], field: str) -> str:
    values = [_float_or_none(row.get(field)) for row in rows]
    known = [value for value in values if value is not None]
    return _fmt(sum(known), 3) if known else ""


def _sum_present(rows: list[dict], field: str) -> str:
    values = [_int_or_none(row.get(field)) for row in rows]
    known = [value for value in values if value is not None]
    return str(sum(known)) if known else ""


def _mean(values: list[int | float]) -> str:
    return _fmt(statistics.mean(values), 6) if values else ""


def _median(values: list[int | float]) -> str:
    return _fmt(statistics.median(values), 6) if values else ""


def _int(value: object) -> int:
    try:
        return int(float(str(value or "0")))
    except ValueError:
        return 0


def _int_or_none(value: object) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(float(str(value)))
    except ValueError:
        return None


def _float(value: object) -> float:
    try:
        return float(str(value or "0"))
    except ValueError:
        return 0.0


def _float_or_none(value: object) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(str(value))
    except ValueError:
        return None


def _fmt(value: int | float, digits: int) -> str:
    return f"{float(value):.{digits}f}".rstrip("0").rstrip(".") or "0"


def _optional_fmt(value: float | None) -> str:
    return _fmt(value, 3) if value is not None else ""


def _optional_int(value: int | None) -> str:
    return str(value) if value is not None else ""


def _add_optional(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return left + right


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _one_line(value: object) -> str:
    return " ".join(str(value or "").split())[:1000]
