from __future__ import annotations

import csv
import itertools
import math
from pathlib import Path

from .schema import BATCH_CANDIDATE_FIELDS, BATCH_COMPONENT_FIELDS, BATCH_SUMMARY_FIELDS


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

    for index, choices in enumerate(itertools.product(*alternative_lists) if alternative_lists else [()]):
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
    }


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
