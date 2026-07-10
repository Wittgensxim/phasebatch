from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import itertools
import json
import math
import random
import shutil
import time
from pathlib import Path

from .ir_equivalence import EqualityResult, compare_ir_equivalence, safe_canonical_hash as hash_ir
from .batch_validation_dag import validate_batch_with_permutation_dag, write_batch_validation_dag_summary
from .pass_config import PassRegistry, resolve_pipeline_sequence
from .runner import materialize_run_result, release_run_result, run_opt, worker_handles_enabled
from .schema import BATCH_CANDIDATE_FIELDS, BATCH_COMPONENT_FIELDS, BATCH_SUMMARY_FIELDS, BATCH_VALIDATION_FIELDS
from .validation_runtime import ValidationRuntime, ValidationTransition, ValidationTransitionKey


def build_batch_family(
    state_dir: Path,
    max_component_size: int = 10,
    max_batch_candidates: int = 200,
) -> dict:
    state_dir = Path(state_dir)
    active_profiles = _active_profiles(state_dir / "pass_profile.csv")
    active_passes = [row["pass"] for row in active_profiles]
    pass_rank = {pass_name: index for index, pass_name in enumerate(active_passes)}
    relation_map = _relation_map(state_dir / "pair_relation.csv")
    program, state_id, state_hash = _state_identity(active_profiles, _read_csv(state_dir / "pair_relation.csv"))

    commute_edges: set[tuple[str, str]] = set()
    conflict_edges: set[tuple[str, str]] = set()
    adjacency = {pass_name: set() for pass_name in active_passes}

    for pass_a, pass_b in itertools.combinations(active_passes, 2):
        edge = _edge(pass_a, pass_b, pass_rank)
        if relation_map.get(frozenset([pass_a, pass_b])) == "final_commute":
            commute_edges.add(edge)
        else:
            conflict_edges.add(edge)
            adjacency[pass_a].add(pass_b)
            adjacency[pass_b].add(pass_a)

    components = _connected_components(active_passes, adjacency, pass_rank)
    component_infos = []
    component_rows = []
    for index, component in enumerate(components):
        component_id = f"C{index:04d}"
        component_conflicts = _component_edges(component, conflict_edges, pass_rank)
        component_commutes = _component_edges(component, commute_edges, pass_rank)
        if len(component) == 1:
            alternatives = [frozenset(component)]
            is_exact = True
            unresolved_reason = ""
        elif len(component) <= max_component_size:
            alternatives = _maximal_independent_sets(component, adjacency, pass_rank)
            is_exact = True
            unresolved_reason = ""
        else:
            alternatives = [frozenset([pass_name]) for pass_name in component]
            is_exact = False
            unresolved_reason = f"component_size>{max_component_size}"

        component_infos.append(
            {
                "component_id": component_id,
                "passes": component,
                "alternatives": alternatives,
                "is_exact": is_exact,
                "unresolved_reason": unresolved_reason,
            }
        )
        component_rows.append(
            {
                "program": program,
                "state_id": state_id,
                "state_hash": state_hash,
                "component_id": component_id,
                "component_size": str(len(component)),
                "component_passes": _join_passes(component, pass_rank),
                "conflict_edges": _join_edges(component_conflicts),
                "commute_edges": _join_edges(component_commutes),
                "is_exact": _bool(is_exact),
                "num_local_alternatives": str(len(alternatives)),
                "unresolved_reason": unresolved_reason,
            }
        )

    candidate_rows, truncated = _candidate_rows(
        program,
        state_id,
        state_hash,
        component_infos,
        pass_rank,
        max_batch_candidates,
    )
    summary_row = _summary_row(
        program,
        state_id,
        state_hash,
        active_passes,
        commute_edges,
        conflict_edges,
        components,
        component_infos,
        candidate_rows,
        truncated,
        max_batch_candidates,
    )

    _write_csv(state_dir / "batch_components.csv", BATCH_COMPONENT_FIELDS, component_rows)
    _write_csv(state_dir / "batch_candidates.csv", BATCH_CANDIDATE_FIELDS, candidate_rows)
    _write_csv(state_dir / "batch_summary.csv", BATCH_SUMMARY_FIELDS, [summary_row])
    _write_summary_md(state_dir / "batch_summary.md", summary_row, component_rows, candidate_rows, truncated)

    return {
        "program": program,
        "state_id": state_id,
        "state_hash": state_hash,
        "active_passes": len(active_passes),
        "conflict_components": len(components),
        "batch_candidates": len(candidate_rows),
        "truncated": truncated,
        "batch_components_csv": str(state_dir / "batch_components.csv"),
        "batch_candidates_csv": str(state_dir / "batch_candidates.csv"),
        "batch_summary_csv": str(state_dir / "batch_summary.csv"),
        "batch_summary_md": str(state_dir / "batch_summary.md"),
    }


def validate_batch_candidates(
    state_dir: Path,
    tools: dict,
    timeout: int,
    jobs: int,
    max_permutation_factorial: int = 120,
    max_validation_sequences: int = 200,
    max_validation_dag_nodes: int = 5000,
    max_validation_dag_edges: int = 20000,
    dump_validation_dag: bool = False,
    validation_dag_selected_only: bool = False,
    batch_validation_mode: str = "auto",
    samples: int = 20,
    pass_registry: PassRegistry | None = None,
    candidate_ids: list[str] | None = None,
    runtime: ValidationRuntime | None = None,
    keep_ir_artifacts: bool = False,
) -> dict:
    state_dir = Path(state_dir)
    keep_ir_artifacts = keep_ir_artifacts or _is_true(tools.get("_keep_ir_artifacts"))
    if pass_registry is None:
        maybe_registry = tools.get("_pass_registry") if isinstance(tools, dict) else None
        pass_registry = maybe_registry if isinstance(maybe_registry, PassRegistry) else None
    candidates = _read_csv(state_dir / "batch_candidates.csv")
    input_ll = _state_input_ll(state_dir)
    validation_root = state_dir / "artifacts" / "batch_validation"
    state_hash = candidates[0].get("state_hash", "") if candidates else ""
    profile_outputs = _load_profile_outputs(state_dir, state_hash)
    owns_runtime = runtime is None
    runtime = runtime or ValidationRuntime(state_dir, max_workers=jobs)
    if input_ll is not None and input_ll.exists():
        _seed_runtime_profile_transitions(runtime, input_ll, profile_outputs, pass_registry)
    if dump_validation_dag or keep_ir_artifacts:
        runtime.write_keep_marker()
    if keep_ir_artifacts:
        validation_root.mkdir(parents=True, exist_ok=True)
        (validation_root / ".keep_ir_artifacts").write_text(
            "batch validation IR retained\n",
            encoding="utf-8",
        )
    indexed_candidates = list(enumerate(candidates))
    existing_by_id = {
        row.get("batch_id", ""): row
        for row in _read_csv(state_dir / "batch_validation.csv")
        if row.get("batch_id")
    }
    ordered_rows: list[dict | None] = [None] * len(candidates)
    if candidate_ids is None:
        scheduled_candidates = indexed_candidates
    else:
        candidate_index = {
            candidate.get("batch_id", ""): (index, candidate)
            for index, candidate in indexed_candidates
            if candidate.get("batch_id")
        }
        scheduled_candidates = []
        seen_ids: set[str] = set()
        for batch_id in candidate_ids:
            if batch_id in seen_ids or batch_id not in candidate_index:
                continue
            seen_ids.add(batch_id)
            index, candidate = candidate_index[batch_id]
            existing = existing_by_id.get(batch_id)
            if existing and _validation_was_attempted(existing):
                ordered_rows[index] = existing
            else:
                scheduled_candidates.append((index, candidate))
        for index, candidate in indexed_candidates:
            if ordered_rows[index] is not None:
                continue
            existing = existing_by_id.get(candidate.get("batch_id", ""))
            if existing and _validation_was_attempted(existing):
                ordered_rows[index] = existing
                continue
            row = _validation_base_row(candidate, batch_validation_mode)
            row["validation_incomplete_reason"] = "budgeted_on_demand_not_selected"
            ordered_rows[index] = row

    effective_jobs = max(1, jobs)
    candidate_workers = min(effective_jobs, len(scheduled_candidates)) if scheduled_candidates else 1
    candidate_jobs = max(1, effective_jobs // candidate_workers)

    def validate_candidate(index: int, candidate: dict) -> tuple[int, dict]:
        row = _validate_one_batch(
            candidate,
            input_ll,
            validation_root,
            tools,
            timeout,
            candidate_jobs,
            max_permutation_factorial,
            max_validation_sequences,
            max_validation_dag_nodes,
            max_validation_dag_edges,
            dump_validation_dag,
            validation_dag_selected_only,
            batch_validation_mode,
            samples,
            pass_registry,
            profile_outputs,
            runtime,
            keep_ir_artifacts,
        )
        return index, row

    try:
        if candidate_workers == 1:
            for index, candidate in scheduled_candidates:
                _, ordered_rows[index] = validate_candidate(index, candidate)
        else:
            with ThreadPoolExecutor(max_workers=candidate_workers) as executor:
                futures = {
                    executor.submit(validate_candidate, index, candidate): index
                    for index, candidate in scheduled_candidates
                }
                for future in as_completed(futures):
                    index, ordered_rows[index] = future.result()
        rows = [row for row in ordered_rows if row is not None]

        _write_csv(state_dir / "batch_validation.csv", BATCH_VALIDATION_FIELDS, rows)
        dag_summary_path = write_batch_validation_dag_summary(state_dir, rows)
        _append_validation_summary(state_dir / "batch_summary.md", rows)
        status_counts = Counter(row["validation_status"] for row in rows)
        return {
            "validated_batches": len(rows),
            "attempted_batches": sum(1 for row in rows if _validation_was_attempted(row)),
            "validation_status_counts": dict(status_counts),
            "batch_validation_csv": str(state_dir / "batch_validation.csv"),
            "batch_validation_dag_summary_csv": str(dag_summary_path),
        }
    finally:
        if owns_runtime:
            runtime.close(timeout=timeout)


def _active_profiles(path: Path) -> list[dict]:
    return [
        row
        for row in _read_csv(path)
        if row.get("pass") and _is_true(row.get("success")) and _is_true(row.get("active"))
    ]


def _relation_map(path: Path) -> dict[frozenset[str], str]:
    relations = {}
    for row in _read_csv(path):
        pass_a = row.get("pass_a", "")
        pass_b = row.get("pass_b", "")
        if pass_a and pass_b:
            relations[frozenset([pass_a, pass_b])] = row.get("final_relation", "")
    return relations


def _state_identity(profile_rows: list[dict], pair_rows: list[dict]) -> tuple[str, str, str]:
    row = profile_rows[0] if profile_rows else pair_rows[0] if pair_rows else {}
    return row.get("program", ""), row.get("state_id", ""), row.get("state_hash", "")


def _connected_components(active_passes: list[str], adjacency: dict[str, set[str]], pass_rank: dict[str, int]) -> list[list[str]]:
    seen = set()
    components = []
    for pass_name in active_passes:
        if pass_name in seen:
            continue
        stack = [pass_name]
        component = []
        seen.add(pass_name)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current], key=pass_rank.get):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component, key=pass_rank.get))
    return sorted(components, key=lambda component: pass_rank[component[0]])


def _maximal_independent_sets(component: list[str], adjacency: dict[str, set[str]], pass_rank: dict[str, int]) -> list[frozenset[str]]:
    independent_sets = []
    for size in range(1, len(component) + 1):
        for subset in itertools.combinations(component, size):
            subset_set = frozenset(subset)
            if _is_independent(subset_set, adjacency):
                independent_sets.append(subset_set)

    maximal = []
    for candidate in independent_sets:
        if not any(candidate < other for other in independent_sets):
            maximal.append(candidate)
    return sorted(maximal, key=lambda choices: (-len(choices), [pass_rank[name] for name in sorted(choices, key=pass_rank.get)]))


def _is_independent(pass_set: frozenset[str], adjacency: dict[str, set[str]]) -> bool:
    for pass_a, pass_b in itertools.combinations(pass_set, 2):
        if pass_b in adjacency[pass_a]:
            return False
    return True


def _candidate_rows(
    program: str,
    state_id: str,
    state_hash: str,
    component_infos: list[dict],
    pass_rank: dict[str, int],
    max_batch_candidates: int,
) -> tuple[list[dict], bool]:
    candidate_rows = []
    truncated = False
    alternative_lists = [component["alternatives"] for component in component_infos]
    unresolved_count = sum(1 for component in component_infos if not component["is_exact"])
    all_exact = unresolved_count == 0
    if not alternative_lists:
        return candidate_rows, truncated

    for index, choices in enumerate(itertools.product(*alternative_lists)):
        if index >= max_batch_candidates:
            truncated = True
            break
        batch_passes = sorted(set().union(*choices), key=pass_rank.get) if choices else []
        component_choices = [
            f"{component['component_id']}:{_join_passes(sorted(choice, key=pass_rank.get), pass_rank)}"
            for component, choice in zip(component_infos, choices)
        ]
        canonical_order = _join_passes(batch_passes, pass_rank)
        candidate_rows.append(
            {
                "program": program,
                "state_id": state_id,
                "state_hash": state_hash,
                "batch_id": f"B{index:04d}",
                "batch_passes": canonical_order,
                "batch_size": str(len(batch_passes)),
                "component_choices": "|".join(component_choices),
                "is_exact": _bool(all_exact),
                "num_conflict_components": str(len(component_infos)),
                "unresolved_components": str(unresolved_count),
                "canonical_order": canonical_order,
            }
        )
    return candidate_rows, truncated


def _summary_row(
    program: str,
    state_id: str,
    state_hash: str,
    active_passes: list[str],
    commute_edges: set[tuple[str, str]],
    conflict_edges: set[tuple[str, str]],
    components: list[list[str]],
    component_infos: list[dict],
    candidate_rows: list[dict],
    truncated: bool,
    max_batch_candidates: int,
) -> dict:
    naive_orderings = math.factorial(len(active_passes))
    batch_count = len(candidate_rows)
    return {
        "program": program,
        "state_id": state_id,
        "state_hash": state_hash,
        "active_passes": str(len(active_passes)),
        "active_pairs": str(len(commute_edges) + len(conflict_edges)),
        "commute_pairs": str(len(commute_edges)),
        "conflict_pairs": str(len(conflict_edges)),
        "conflict_components": str(len(components)),
        "max_component_size": str(max((len(component) for component in components), default=0)),
        "batch_candidates": str(batch_count),
        "exact_components": str(sum(1 for component in component_infos if component["is_exact"])),
        "unresolved_components": str(sum(1 for component in component_infos if not component["is_exact"])),
        "naive_orderings_estimate": str(naive_orderings),
        "batch_reduction_estimate": f"{naive_orderings / max(1, batch_count):.2f}",
        "truncated": _bool(truncated),
        "max_batch_candidates": str(max_batch_candidates),
    }


def _validation_base_row(candidate: dict, batch_validation_mode: str) -> dict:
    passes = _split_passes(candidate.get("canonical_order") or candidate.get("batch_passes"))
    factorial_text = str(math.factorial(len(passes))) if passes else "0"
    return {
        "program": candidate.get("program", ""),
        "state_id": candidate.get("state_id", ""),
        "state_hash": candidate.get("state_hash", ""),
        "batch_id": candidate.get("batch_id", ""),
        "batch_size": candidate.get("batch_size", str(len(passes))),
        "canonical_order": _join_order(passes),
        "validation_mode": batch_validation_mode,
        "validation_tier": "unvalidated",
        "validation_sequences_tested": "0",
        "validation_sequences_total_estimate": factorial_text,
        "validation_complete": "false",
        "validation_hard_certificate": "false",
        "validation_incomplete_reason": "",
        "tested_orders": "0",
        "same_hash_count": "0",
        "different_hash_count": "0",
        "hash_equal_count": "0",
        "structural_equal_count": "0",
        "different_count": "0",
        "canonical_hash_equal_count": "0",
        "structural_diff_equal_count": "0",
        "equality_failed_count": "0",
        "validation_equality_tier": "",
        "validation_equality_reason": "",
        "validation_status": "not_validated",
        "canonical_hash": "",
        "first_mismatch_order": "",
        "first_mismatch_hash": "",
        "validation_dag_nodes": "0",
        "validation_dag_edges": "0",
        "validation_dag_final_classes": "0",
        "validation_dag_transition_cache_hits": "0",
        "validation_dag_transition_cache_misses": "0",
        "validation_dag_equivalence_cache_hits": "0",
        "validation_dag_equivalence_cache_misses": "0",
        "validation_dag_hash_merges": "0",
        "validation_dag_structural_merges": "0",
        "validation_dag_budget_exceeded": "false",
        "validation_dag_incomplete_reason": "",
        "factorial_permutations": factorial_text,
        "factorial_permutations_log10": f"{_factorial_log10(len(passes)):.6f}",
        "compression_vs_permutation": "",
        "validation_opt_invocations": "0",
        "validation_pass_invocations_baseline": "0",
        "validation_pass_invocations_actual": "0",
        "validation_pass_invocations_saved": "0",
        "validation_profile_reuse_hits": "0",
        "validation_state_transition_cache_hits": "0",
        "validation_state_equivalence_cache_hits": "0",
        "time_ms": "0.00",
    }


def _validation_was_attempted(row: dict) -> bool:
    return not (
        row.get("validation_status") == "not_validated"
        and row.get("validation_incomplete_reason") == "budgeted_on_demand_not_selected"
    )


def _validate_one_batch(
    candidate: dict,
    input_ll: Path | None,
    validation_root: Path,
    tools: dict,
    timeout: int,
    jobs: int,
    max_permutation_factorial: int,
    max_validation_sequences: int,
    max_validation_dag_nodes: int,
    max_validation_dag_edges: int,
    dump_validation_dag: bool,
    validation_dag_selected_only: bool,
    batch_validation_mode: str,
    samples: int,
    pass_registry: PassRegistry | None,
    profile_outputs: dict[str, Path],
    runtime: ValidationRuntime,
    keep_ir_artifacts: bool,
) -> dict:
    del validation_dag_selected_only
    start = time.perf_counter()
    passes = _split_passes(candidate.get("canonical_order") or candidate.get("batch_passes"))
    base_row = _validation_base_row(candidate, batch_validation_mode)
    opt = tools.get("opt")
    if not input_ll or not input_ll.exists() or not opt or not passes:
        base_row["time_ms"] = _elapsed_ms(start)
        base_row["validation_incomplete_reason"] = "missing_input_or_tool"
        return base_row

    batch_dir = validation_root / _safe_name(candidate.get("batch_id", "batch"))
    batch_dir.mkdir(parents=True, exist_ok=True)
    defer_materialization = (
        worker_handles_enabled()
        and not dump_validation_dag
        and not keep_ir_artifacts
    )
    dag_metric_updates: dict[str, str] = {}
    if _should_use_dag(batch_validation_mode, passes, max_permutation_factorial):
        dag_row = validate_batch_with_permutation_dag(
            input_ll,
            passes,
            passes,
            pass_registry,
            tools,
            batch_dir,
            max_nodes=max_validation_dag_nodes,
            max_edges=max_validation_dag_edges,
            dump_dag=dump_validation_dag,
            timeout=timeout,
            batch_id=candidate.get("batch_id", ""),
            program=candidate.get("program", ""),
            state_id=candidate.get("state_id", ""),
            state_hash=candidate.get("state_hash", ""),
            validation_mode=batch_validation_mode,
            runtime=runtime,
            jobs=jobs,
            keep_ir_artifacts=keep_ir_artifacts,
        )
        if batch_validation_mode != "auto" or dag_row.get("validation_status") != "incomplete":
            base_row.update(dag_row)
            return base_row
        dag_metric_updates = {
            key: value
            for key, value in dag_row.items()
            if key.startswith("validation_dag_")
            or key
            in {
                "factorial_permutations",
                "factorial_permutations_log10",
                "compression_vs_permutation",
                "validation_materializations",
                "validation_materializations_avoided",
            }
        }
        batch_validation_mode = "bounded"

    canonical_output = batch_dir / "canonical.ll"
    canonical_result = _run_validation_pipeline(
        opt,
        input_ll,
        passes,
        canonical_output,
        timeout,
        pass_registry,
        profile_outputs,
        allow_first_pass_reuse=False,
        runtime=runtime,
        defer_materialization=defer_materialization,
    )
    base_row["tested_orders"] = "1"
    _set_validation_costs(base_row, [canonical_result])
    if not canonical_result["success"] or (
        canonical_result.get("materialized") and not canonical_output.exists()
    ):
        base_row["validation_status"] = "failed"
        base_row["validation_tier"] = "failed"
        base_row["validation_sequences_tested"] = "1"
        base_row["validation_equality_tier"] = "failed"
        base_row["validation_equality_reason"] = "canonical_order_failed"
        base_row["validation_incomplete_reason"] = "canonical_order_failed"
        base_row["time_ms"] = _elapsed_ms(start)
        _release_validation_results([canonical_result], timeout)
        return base_row

    canonical_digest = canonical_result["hash"]
    base_row["canonical_hash"] = canonical_digest
    same_hash_count = 1
    different_hash_count = 0
    hash_equal_count = 1
    structural_equal_count = 0
    different_count = 0
    equality_failed_count = 0
    first_mismatch_order = ""
    first_mismatch_hash = ""
    first_non_fold_tier = ""
    first_non_fold_reason = ""
    structural_reason = ""

    plan = _validation_plan(
        passes,
        max_permutation_factorial=max_permutation_factorial,
        max_validation_sequences=max_validation_sequences,
        batch_validation_mode=batch_validation_mode,
        samples=samples,
    )
    base_row["validation_mode"] = plan["mode"]
    base_row["validation_tier"] = plan["tier"]
    base_row["validation_sequences_total_estimate"] = str(plan["total_estimate"])
    base_row["validation_complete"] = _bool(plan["complete"])
    base_row["validation_hard_certificate"] = "false"
    base_row["validation_incomplete_reason"] = plan["incomplete_reason"]
    if plan["tier"] == "unvalidated":
        base_row.update(
            {
                "tested_orders": "1",
                "validation_sequences_tested": "1",
                "validation_status": "not_validated",
                "time_ms": _elapsed_ms(start),
            }
        )
        _release_validation_results([canonical_result], timeout)
        return base_row

    validation_orders = plan["orders"]
    order_results = _run_validation_orders(
        opt,
        input_ll,
        validation_orders,
        batch_dir,
        timeout,
        max(1, jobs),
        pass_registry,
        profile_outputs,
        runtime,
        defer_materialization,
    )
    failed = False
    for result in order_results:
        if not result["success"]:
            failed = True
            if not first_non_fold_tier:
                first_non_fold_tier = "failed"
                first_non_fold_reason = "validation_order_failed"
            continue
        equality = _compare_validation_results(
            canonical_result,
            result,
            tools=tools,
            timeout=timeout,
        )
        result_hash = equality.right_hash or result["hash"]
        if equality.text_hash_equal:
            same_hash_count += 1
            hash_equal_count += 1
        else:
            different_hash_count += 1
        if equality.can_hard_fold:
            if equality.tier == "structural_diff":
                structural_equal_count += 1
                if not structural_reason:
                    structural_reason = equality.reason
            continue

        if equality.tier == "failed":
            failed = True
            equality_failed_count += 1
            if not first_non_fold_tier:
                first_non_fold_tier = "failed"
                first_non_fold_reason = equality.reason or "tool_failed"
            continue

        different_count += 1
        if not first_non_fold_tier:
            first_non_fold_tier = equality.tier
            first_non_fold_reason = equality.reason
        if not first_mismatch_order:
            first_mismatch_order = _join_order(result["order"])
            first_mismatch_hash = result_hash

    validation_tier, validation_reason = _validation_equality_summary(
        failed=failed,
        different_count=different_count,
        structural_equal_count=structural_equal_count,
        first_non_fold_tier=first_non_fold_tier,
        first_non_fold_reason=first_non_fold_reason,
        structural_reason=structural_reason,
    )

    if failed:
        status = "failed"
    elif different_count:
        status = "mismatch"
    elif plan["tier"] == "exhaustive_all_permutations":
        status = "all_permutations_same"
    elif plan["tier"] in {"bounded_insertion", "bounded_adjacent_swap"}:
        status = "bounded_same"
    else:
        status = "sampled_same"
    hard_certificate = status == "all_permutations_same" and plan["hard_certificate"]

    base_row.update(
        {
            "tested_orders": str(1 + len(validation_orders)),
            "validation_sequences_tested": str(1 + len(validation_orders)),
            "validation_complete": _bool(plan["complete"]),
            "validation_hard_certificate": _bool(hard_certificate),
            "same_hash_count": str(same_hash_count),
            "different_hash_count": str(different_hash_count),
            "hash_equal_count": str(hash_equal_count),
            "structural_equal_count": str(structural_equal_count),
            "different_count": str(different_count),
            "canonical_hash_equal_count": str(hash_equal_count),
            "structural_diff_equal_count": str(structural_equal_count),
            "equality_failed_count": str(equality_failed_count),
            "validation_equality_tier": validation_tier,
            "validation_equality_reason": validation_reason,
            "validation_status": status,
            "first_mismatch_order": first_mismatch_order,
            "first_mismatch_hash": first_mismatch_hash,
            "time_ms": _elapsed_ms(start),
        }
    )
    _set_validation_costs(base_row, [canonical_result, *order_results])
    _release_validation_results([canonical_result, *order_results], timeout)
    if dag_metric_updates:
        base_row.update(dag_metric_updates)
    return base_row


def _validation_equality_summary(
    *,
    failed: bool,
    different_count: int,
    structural_equal_count: int,
    first_non_fold_tier: str,
    first_non_fold_reason: str,
    structural_reason: str,
) -> tuple[str, str]:
    if failed:
        return "failed", first_non_fold_reason or "validation_order_failed"
    if different_count:
        return first_non_fold_tier or "different", first_non_fold_reason or "ir_difference"
    if structural_equal_count:
        return "structural_diff", structural_reason or "llvm_diff_equal_and_module_fingerprint_equal"
    return "canonical_hash", "hash_equal"


def _state_input_ll(state_dir: Path) -> Path | None:
    direct = state_dir / "input.ll"
    if direct.exists():
        return direct
    metadata_path = state_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    input_path = metadata.get("input")
    if not input_path:
        return None
    resolved = Path(input_path)
    return resolved if resolved.exists() else None


def _validation_plan(
    canonical_order: list[str],
    *,
    max_permutation_factorial: int,
    max_validation_sequences: int,
    batch_validation_mode: str,
    samples: int,
) -> dict:
    mode = batch_validation_mode if batch_validation_mode in {"auto", "exhaustive", "dag", "bounded", "sampled"} else "auto"
    canonical_tuple = tuple(canonical_order)
    permutation_count = math.factorial(len(canonical_order))
    if mode == "auto":
        selected = "exhaustive" if permutation_count <= max_permutation_factorial else "bounded"
    else:
        selected = mode

    if selected == "exhaustive":
        if permutation_count > max_permutation_factorial:
            return {
                "mode": mode,
                "tier": "unvalidated",
                "orders": [],
                "complete": False,
                "hard_certificate": False,
                "total_estimate": permutation_count,
                "incomplete_reason": "factorial_exceeds_max_permutation_factorial",
            }
        return {
            "mode": mode,
            "tier": "exhaustive_all_permutations",
            "orders": [list(order) for order in itertools.permutations(canonical_order) if order != canonical_tuple],
            "complete": True,
            "hard_certificate": True,
            "total_estimate": permutation_count,
            "incomplete_reason": "",
        }

    if selected == "bounded":
        return {
            "mode": mode,
            "tier": "bounded_insertion",
            "orders": _bounded_insertion_orders(canonical_order, max(0, max_validation_sequences - 1)),
            "complete": False,
            "hard_certificate": False,
            "total_estimate": permutation_count,
            "incomplete_reason": "bounded_insertion_does_not_cover_all_permutations",
        }

    if selected == "dag":
        return {
            "mode": mode,
            "tier": "unvalidated",
            "orders": [],
            "complete": False,
            "hard_certificate": False,
            "total_estimate": permutation_count,
            "incomplete_reason": "dag_validation_not_dispatched",
        }

    return {
        "mode": mode,
        "tier": "sampled_permutations",
        "orders": _sampled_orders(canonical_order, max(0, min(samples, max_validation_sequences - 1))),
        "complete": False,
        "hard_certificate": False,
        "total_estimate": permutation_count,
        "incomplete_reason": "sampled_permutations_do_not_cover_all_permutations",
    }


def _should_use_dag(batch_validation_mode: str, passes: list[str], max_permutation_factorial: int) -> bool:
    if batch_validation_mode == "dag":
        return True
    if batch_validation_mode != "auto":
        return False
    return math.factorial(len(passes)) > max_permutation_factorial


def _factorial_log10(size: int) -> float:
    return sum(math.log10(value) for value in range(1, size + 1))


def _sampled_orders(canonical_order: list[str], samples: int) -> list[list[str]]:
    canonical_tuple = tuple(canonical_order)
    if samples <= 0:
        return []

    rng = random.Random(0)
    seen = {canonical_tuple}
    orders = []
    attempts = 0
    max_attempts = max(samples * 20, 100)
    while len(orders) < samples and attempts < max_attempts:
        attempts += 1
        order = list(canonical_order)
        rng.shuffle(order)
        order_tuple = tuple(order)
        if order_tuple in seen:
            continue
        seen.add(order_tuple)
        orders.append(order)
    return orders


def _bounded_insertion_orders(canonical_order: list[str], limit: int) -> list[list[str]]:
    canonical_tuple = tuple(canonical_order)
    seen = {canonical_tuple}
    orders: list[list[str]] = []
    if limit <= 0:
        return orders

    for source_index, pass_name in enumerate(canonical_order):
        rest = canonical_order[:source_index] + canonical_order[source_index + 1 :]
        for insert_index in range(len(canonical_order)):
            order = list(rest)
            order.insert(insert_index, pass_name)
            order_tuple = tuple(order)
            if order_tuple in seen:
                continue
            seen.add(order_tuple)
            orders.append(order)
            if len(orders) >= limit:
                return orders
    return orders


def _run_validation_orders(
    opt: str,
    input_ll: Path,
    orders: list[list[str]],
    batch_dir: Path,
    timeout: int,
    jobs: int,
    pass_registry: PassRegistry | None,
    profile_outputs: dict[str, Path],
    runtime: ValidationRuntime,
    defer_materialization: bool,
) -> list[dict]:
    if not orders:
        return []
    if jobs <= 1:
        return [
            _run_one_validation_order(
                opt,
                input_ll,
                order,
                batch_dir,
                index,
                timeout,
                pass_registry,
                profile_outputs,
                runtime,
                defer_materialization,
            )
            for index, order in enumerate(orders)
        ]

    results: list[dict | None] = [None] * len(orders)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                _run_one_validation_order,
                opt,
                input_ll,
                order,
                batch_dir,
                index,
                timeout,
                pass_registry,
                profile_outputs,
                runtime,
                defer_materialization,
            ): index
            for index, order in enumerate(orders)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return [result for result in results if result is not None]


def _run_one_validation_order(
    opt: str,
    input_ll: Path,
    order: list[str],
    batch_dir: Path,
    index: int,
    timeout: int,
    pass_registry: PassRegistry | None,
    profile_outputs: dict[str, Path],
    runtime: ValidationRuntime,
    defer_materialization: bool,
) -> dict:
    output_ll = batch_dir / f"order_{index:04d}.ll"
    output_ll.parent.mkdir(parents=True, exist_ok=True)
    return _run_validation_pipeline(
        opt,
        input_ll,
        order,
        output_ll,
        timeout,
        pass_registry,
        profile_outputs,
        allow_first_pass_reuse=True,
        runtime=runtime,
        defer_materialization=defer_materialization,
    )


def _run_validation_pipeline(
    opt: str,
    input_ll: Path,
    order: list[str],
    output_ll: Path,
    timeout: int,
    pass_registry: PassRegistry | None,
    profile_outputs: dict[str, Path],
    *,
    allow_first_pass_reuse: bool,
    runtime: ValidationRuntime,
    defer_materialization: bool = False,
) -> dict:
    source = input_ll
    passes_to_run = list(order)
    profile_reuse_hits = 0
    if allow_first_pass_reuse and order:
        reusable = profile_outputs.get(order[0])
        if reusable is not None and reusable.exists():
            source = reusable
            passes_to_run = order[1:]
            profile_reuse_hits = 1

    baseline_invocations = len(order)
    actual_invocations = len(passes_to_run)
    if not passes_to_run:
        output_ll.write_bytes(Path(source).read_bytes())
        success = output_ll.exists()
        opt_invocations = 0
        run_result = None
        materialized = success
        materializations = 1 if success else 0
        materializations_avoided = 0
    else:
        if defer_materialization:
            result = runtime.run_with_opt_slot(
                lambda: run_opt(
                    opt,
                    source,
                    resolve_pipeline_sequence(passes_to_run, pass_registry),
                    output_ll,
                    timeout,
                    materialize=False,
                )
            )
        else:
            result = runtime.run_with_opt_slot(
                lambda: run_opt(
                    opt,
                    source,
                    resolve_pipeline_sequence(passes_to_run, pass_registry),
                    output_ll,
                    timeout,
                )
            )
        success = result.success and (not result.materialized or output_ll.exists())
        opt_invocations = 1
        run_result = result
        materialized = result.materialized and output_ll.exists()
        materializations = 1 if materialized else 0
        materializations_avoided = 1 if result.success and not result.materialized else 0

    return {
        "order": order,
        "success": success,
        "hash": (
            run_result.canonical_hash
            if success and run_result is not None and not run_result.materialized
            else hash_ir(output_ll) if success else ""
        ),
        "path": output_ll,
        "run_result": run_result,
        "materialized": materialized,
        "materializations": materializations,
        "materializations_avoided": materializations_avoided,
        "opt_invocations": opt_invocations,
        "pass_invocations_baseline": baseline_invocations,
        "pass_invocations_actual": actual_invocations,
        "pass_invocations_saved": baseline_invocations - actual_invocations,
        "profile_reuse_hits": profile_reuse_hits,
    }


def _set_validation_costs(row: dict, results: list[dict]) -> None:
    row["validation_opt_invocations"] = str(sum(int(result.get("opt_invocations", 0)) for result in results))
    row["validation_pass_invocations_baseline"] = str(
        sum(int(result.get("pass_invocations_baseline", 0)) for result in results)
    )
    row["validation_pass_invocations_actual"] = str(
        sum(int(result.get("pass_invocations_actual", 0)) for result in results)
    )
    row["validation_pass_invocations_saved"] = str(
        sum(int(result.get("pass_invocations_saved", 0)) for result in results)
    )
    row["validation_profile_reuse_hits"] = str(sum(int(result.get("profile_reuse_hits", 0)) for result in results))
    row["validation_materializations"] = str(sum(int(result.get("materializations", 0)) for result in results))
    row["validation_materializations_avoided"] = str(
        sum(int(result.get("materializations_avoided", 0)) for result in results)
    )


def _compare_validation_results(
    canonical: dict,
    candidate: dict,
    *,
    tools: dict,
    timeout: int,
) -> EqualityResult:
    canonical_run = canonical.get("run_result")
    candidate_run = candidate.get("run_result")
    if (
        canonical_run is not None
        and candidate_run is not None
        and canonical_run.backend == "worker"
        and candidate_run.backend == "worker"
        and canonical_run.canonical_hash
        and canonical_run.canonical_hash == candidate_run.canonical_hash
    ):
        return EqualityResult(
            equal=True,
            tier="canonical_hash",
            can_hard_fold=True,
            reason="worker_full_ir_hash_equal",
            text_hash_equal=True,
            left_hash=canonical_run.canonical_hash,
            right_hash=candidate_run.canonical_hash,
        )
    try:
        _ensure_validation_materialized(canonical, timeout)
        _ensure_validation_materialized(candidate, timeout)
    except (OSError, RuntimeError, ValueError) as exc:
        return EqualityResult(
            equal=False,
            tier="failed",
            can_hard_fold=False,
            reason="materialize_failed",
            error_message=str(exc),
        )
    return compare_ir_equivalence(canonical["path"], candidate["path"], tools=tools, timeout=timeout)


def _ensure_validation_materialized(result: dict, timeout: int) -> None:
    run_result = result.get("run_result")
    if run_result is not None and not run_result.materialized:
        materialize_run_result(run_result, result["path"], timeout=timeout)
        result["materialized"] = True
        result["materializations"] = 1
        result["materializations_avoided"] = 0
        result["hash"] = hash_ir(result["path"])
    if not Path(result["path"]).exists():
        raise OSError(f"validation output was not materialized: {result['path']}")


def _release_validation_results(results: list[dict], timeout: int) -> None:
    for result in results:
        run_result = result.get("run_result")
        if run_result is not None and run_result.backend == "worker":
            release_run_result(run_result, timeout=timeout)


def _load_profile_outputs(state_dir: Path, state_hash: str) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for row in _read_csv(Path(state_dir) / "pass_profile.csv"):
        if state_hash and row.get("state_hash", "") != state_hash:
            continue
        if not _is_true(row.get("success")) or not _is_true(row.get("active")):
            continue
        pass_name = row.get("pass", "")
        output_path = row.get("output_path", "")
        if not pass_name or not output_path:
            continue
        path = Path(output_path)
        if path.exists():
            outputs[pass_name] = path
    return outputs


def _seed_runtime_profile_transitions(
    runtime: ValidationRuntime,
    input_ll: Path,
    profile_outputs: dict[str, Path],
    pass_registry: PassRegistry | None,
) -> None:
    source_hash = hash_ir(input_ll)
    for pass_name, profile_path in profile_outputs.items():
        pipeline_key = ",".join(resolve_pipeline_sequence([pass_name], pass_registry))
        key = ValidationTransitionKey(source_hash, pass_name, pipeline_key)
        cache_path = runtime.transition_cache_path(key)
        try:
            if profile_path.resolve() != cache_path.resolve():
                shutil.copyfile(profile_path, cache_path)
            output_hash = hash_ir(cache_path)
        except OSError:
            continue
        runtime.seed_transition(
            key,
            ValidationTransition(cache_path, output_hash, "profile"),
        )


def _component_edges(component: list[str], edges: set[tuple[str, str]], pass_rank: dict[str, int]) -> list[tuple[str, str]]:
    component_set = set(component)
    return sorted(
        [edge for edge in edges if edge[0] in component_set and edge[1] in component_set],
        key=lambda edge: (pass_rank[edge[0]], pass_rank[edge[1]]),
    )


def _edge(pass_a: str, pass_b: str, pass_rank: dict[str, int]) -> tuple[str, str]:
    return tuple(sorted([pass_a, pass_b], key=pass_rank.get))


def _join_passes(passes: list[str], pass_rank: dict[str, int]) -> str:
    return ";".join(sorted(passes, key=pass_rank.get))


def _join_edges(edges: list[tuple[str, str]]) -> str:
    return ";".join(f"{pass_a}--{pass_b}" for pass_a, pass_b in edges)


def _split_passes(value: str | None) -> list[str]:
    return [part for part in str(value or "").split(";") if part]


def _join_order(passes: list[str]) -> str:
    return ";".join(passes)


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in value) or "batch"


def _elapsed_ms(start: float) -> str:
    return f"{(time.perf_counter() - start) * 1000:.2f}"


def _write_summary_md(
    path: Path,
    summary_row: dict,
    component_rows: list[dict],
    candidate_rows: list[dict],
    truncated: bool,
) -> None:
    lines = [
        "# Batch Summary",
        "",
        "## Overall",
        "",
        f"- active passes: {summary_row['active_passes']}",
        f"- active pairs: {summary_row['active_pairs']}",
        f"- commute pairs: {summary_row['commute_pairs']}",
        f"- conflict pairs: {summary_row['conflict_pairs']}",
        f"- conflict components: {summary_row['conflict_components']}",
        f"- batch candidates: {summary_row['batch_candidates']}",
        f"- truncated: {_bool(truncated)}",
        f"- naive orderings estimate: {summary_row['naive_orderings_estimate']}",
        f"- batch reduction estimate: {summary_row['batch_reduction_estimate']}",
        "",
        "## Components",
        "",
    ]
    lines.extend(_markdown_table(["component", "size", "passes", "exact", "local alternatives"], [
        [
            row["component_id"],
            row["component_size"],
            row["component_passes"],
            row["is_exact"],
            row["num_local_alternatives"],
        ]
        for row in component_rows
    ]))
    lines.extend(["", "## Candidates", ""])
    lines.extend(_markdown_table(["batch", "size", "passes", "exact"], [
        [row["batch_id"], row["batch_size"], row["batch_passes"], row["is_exact"]]
        for row in candidate_rows[:20]
    ]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_validation_summary(path: Path, rows: list[dict]) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Batch Summary\n"
    marker = "\n## Validation\n"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n"
    counts = Counter(row.get("validation_status", "") for row in rows)
    validation_tier_counts = Counter(row.get("validation_tier", "") for row in rows if row.get("validation_tier"))
    tier_counts = Counter(row.get("validation_equality_tier", "") for row in rows if row.get("validation_equality_tier"))
    tier_hard_folds = Counter(
        row.get("validation_equality_tier", "")
        for row in rows
        if row.get("validation_status") == "all_permutations_same"
        and row.get("validation_equality_tier") in {"canonical_hash", "structural_diff"}
    )
    lines = [
        existing.rstrip(),
        "",
        "## Validation",
        "",
        "- all_permutations_same is a strong batch certificate.",
        "- bounded_same is bounded validation evidence, not hard proof by default.",
        "- sampled_same is empirical evidence, not hard proof.",
        "",
    ]
    lines.extend(_markdown_table(["validation_status", "count"], [[status, str(count)] for status, count in sorted(counts.items())]))
    lines.extend(["", "### Validation Tier Summary", ""])
    lines.extend(_markdown_table(["validation_tier", "count"], [[tier, str(count)] for tier, count in sorted(validation_tier_counts.items())]))
    lines.extend(["", "### Equality Tier Summary", ""])
    lines.extend(_markdown_table(
        ["tier", "count", "hard_fold"],
        [[tier, str(count), str(tier_hard_folds.get(tier, 0))] for tier, count in sorted(tier_counts.items())],
    ))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["none"]
    lines = [f"| {' | '.join(headers)} |", f"| {' | '.join(['---'] * len(headers))} |"]
    lines.extend(f"| {' | '.join(row)} |" for row in rows)
    return lines


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _is_true(value: object) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def _bool(value: bool) -> str:
    return "true" if value else "false"
