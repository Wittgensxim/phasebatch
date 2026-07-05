from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import itertools
import json
import math
import random
import time
from pathlib import Path

from .normalizer import canonical_hash as hash_ir
from .pass_config import PassRegistry, resolve_pipeline_sequence
from .runner import run_opt
from .schema import BATCH_CANDIDATE_FIELDS, BATCH_COMPONENT_FIELDS, BATCH_SUMMARY_FIELDS, BATCH_VALIDATION_FIELDS


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
    samples: int = 20,
    pass_registry: PassRegistry | None = None,
) -> dict:
    state_dir = Path(state_dir)
    if pass_registry is None:
        maybe_registry = tools.get("_pass_registry") if isinstance(tools, dict) else None
        pass_registry = maybe_registry if isinstance(maybe_registry, PassRegistry) else None
    candidates = _read_csv(state_dir / "batch_candidates.csv")
    input_ll = _state_input_ll(state_dir)
    validation_root = state_dir / "artifacts" / "batch_validation"
    rows = []

    for candidate in candidates:
        rows.append(
            _validate_one_batch(
                candidate,
                input_ll,
                validation_root,
                tools,
                timeout,
                jobs,
                max_permutation_factorial,
                samples,
                pass_registry,
            )
        )

    _write_csv(state_dir / "batch_validation.csv", BATCH_VALIDATION_FIELDS, rows)
    _append_validation_summary(state_dir / "batch_summary.md", rows)
    status_counts = Counter(row["validation_status"] for row in rows)
    return {
        "validated_batches": len(rows),
        "validation_status_counts": dict(status_counts),
        "batch_validation_csv": str(state_dir / "batch_validation.csv"),
    }


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


def _validate_one_batch(
    candidate: dict,
    input_ll: Path | None,
    validation_root: Path,
    tools: dict,
    timeout: int,
    jobs: int,
    max_permutation_factorial: int,
    samples: int,
    pass_registry: PassRegistry | None,
) -> dict:
    start = time.perf_counter()
    passes = _split_passes(candidate.get("canonical_order") or candidate.get("batch_passes"))
    base_row = {
        "program": candidate.get("program", ""),
        "state_id": candidate.get("state_id", ""),
        "state_hash": candidate.get("state_hash", ""),
        "batch_id": candidate.get("batch_id", ""),
        "batch_size": candidate.get("batch_size", str(len(passes))),
        "canonical_order": _join_order(passes),
        "tested_orders": "0",
        "same_hash_count": "0",
        "different_hash_count": "0",
        "validation_status": "not_validated",
        "canonical_hash": "",
        "first_mismatch_order": "",
        "first_mismatch_hash": "",
        "time_ms": "0.00",
    }
    opt = tools.get("opt")
    if not input_ll or not input_ll.exists() or not opt or not passes:
        base_row["time_ms"] = _elapsed_ms(start)
        return base_row

    batch_dir = validation_root / _safe_name(candidate.get("batch_id", "batch"))
    batch_dir.mkdir(parents=True, exist_ok=True)
    canonical_output = batch_dir / "canonical.ll"
    canonical_result = run_opt(opt, input_ll, resolve_pipeline_sequence(passes, pass_registry), canonical_output, timeout)
    base_row["tested_orders"] = "1"
    if not canonical_result.success or not canonical_output.exists():
        base_row["validation_status"] = "failed"
        base_row["time_ms"] = _elapsed_ms(start)
        return base_row

    canonical_digest = hash_ir(canonical_output)
    base_row["canonical_hash"] = canonical_digest
    same_count = 1
    different_count = 0
    first_mismatch_order = ""
    first_mismatch_hash = ""

    validation_orders, exhaustive = _validation_orders(passes, max_permutation_factorial, samples)
    order_results = _run_validation_orders(
        opt,
        input_ll,
        validation_orders,
        batch_dir,
        timeout,
        max(1, jobs),
        pass_registry,
    )
    failed = False
    for result in order_results:
        if not result["success"]:
            failed = True
            continue
        if result["hash"] == canonical_digest:
            same_count += 1
        else:
            different_count += 1
            if not first_mismatch_order:
                first_mismatch_order = _join_order(result["order"])
                first_mismatch_hash = result["hash"]

    if failed:
        status = "failed"
    elif different_count:
        status = "mismatch"
    elif exhaustive:
        status = "all_permutations_same"
    else:
        status = "sampled_same"

    base_row.update(
        {
            "tested_orders": str(1 + len(validation_orders)),
            "same_hash_count": str(same_count),
            "different_hash_count": str(different_count),
            "validation_status": status,
            "first_mismatch_order": first_mismatch_order,
            "first_mismatch_hash": first_mismatch_hash,
            "time_ms": _elapsed_ms(start),
        }
    )
    return base_row


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


def _validation_orders(
    canonical_order: list[str],
    max_permutation_factorial: int,
    samples: int,
) -> tuple[list[list[str]], bool]:
    canonical_tuple = tuple(canonical_order)
    permutation_count = math.factorial(len(canonical_order))
    if permutation_count <= max_permutation_factorial:
        return [
            list(order)
            for order in itertools.permutations(canonical_order)
            if order != canonical_tuple
        ], True

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
    return orders, False


def _run_validation_orders(
    opt: str,
    input_ll: Path,
    orders: list[list[str]],
    batch_dir: Path,
    timeout: int,
    jobs: int,
    pass_registry: PassRegistry | None,
) -> list[dict]:
    if not orders:
        return []
    if jobs <= 1:
        return [
            _run_one_validation_order(opt, input_ll, order, batch_dir, index, timeout, pass_registry)
            for index, order in enumerate(orders)
        ]

    results: list[dict | None] = [None] * len(orders)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(_run_one_validation_order, opt, input_ll, order, batch_dir, index, timeout, pass_registry): index
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
) -> dict:
    output_ll = batch_dir / f"order_{index:04d}.ll"
    output_ll.parent.mkdir(parents=True, exist_ok=True)
    result = run_opt(opt, input_ll, resolve_pipeline_sequence(order, pass_registry), output_ll, timeout)
    if not result.success or not output_ll.exists():
        return {"order": order, "success": False, "hash": ""}
    return {"order": order, "success": True, "hash": hash_ir(output_ll)}


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
    counts = Counter(row["validation_status"] for row in rows)
    lines = [
        existing.rstrip(),
        "",
        "## Validation",
        "",
        "- all_permutations_same is a strong batch certificate.",
        "- sampled_same is empirical evidence, not hard proof.",
        "",
    ]
    lines.extend(_markdown_table(["validation_status", "count"], [[status, str(count)] for status, count in sorted(counts.items())]))
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
