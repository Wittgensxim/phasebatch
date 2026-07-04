from __future__ import annotations

import csv
import statistics
from pathlib import Path

from .schema import CLUSTER_DISTRIBUTION_FIELDS


def build_graph(pair_rows: list[dict], graph_type: str) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    for row in pair_rows:
        if not _include_edge(row, graph_type):
            continue
        a = row["pass_a"]
        b = row["pass_b"]
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    return graph


def connected_components(graph: dict[str, set[str]]) -> list[set[str]]:
    remaining = set(graph)
    components: list[set[str]] = []
    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbor in graph.get(node, set()):
                if neighbor in component:
                    continue
                component.add(neighbor)
                remaining.discard(neighbor)
                stack.append(neighbor)
        components.append(component)
    return sorted(components, key=lambda item: (-len(item), sorted(item)))


def component_stats(components: list[set[str]]) -> dict:
    sizes = [len(component) for component in components]
    if not sizes:
        return {
            "num_components": 0,
            "mean_size": 0,
            "median_size": 0,
            "max_size": 0,
            "size_1": 0,
            "size_2": 0,
            "size_3": 0,
            "size_4_7": 0,
            "size_gt_7": 0,
        }
    return {
        "num_components": len(sizes),
        "mean_size": f"{statistics.mean(sizes):.3f}",
        "median_size": f"{statistics.median(sizes):.3f}",
        "max_size": max(sizes),
        "size_1": sum(1 for size in sizes if size == 1),
        "size_2": sum(1 for size in sizes if size == 2),
        "size_3": sum(1 for size in sizes if size == 3),
        "size_4_7": sum(1 for size in sizes if 4 <= size <= 7),
        "size_gt_7": sum(1 for size in sizes if size > 7),
    }


def cluster_distribution_rows(pair_rows: list[dict], program: str, state_hash: str) -> list[dict]:
    rows: list[dict] = []
    for graph_type in ("order_sensitive_graph", "unknown_graph", "noncommute_graph", "static_overlap_graph"):
        graph = build_graph(pair_rows, graph_type)
        components = connected_components(graph)
        stats = component_stats(components)
        rows.append(
            {
                "program": program,
                "state_hash": state_hash,
                "graph_type": graph_type,
                "num_nodes": len(graph),
                "num_edges": sum(len(edges) for edges in graph.values()) // 2,
                **stats,
            }
        )
    return rows


def write_cluster_distribution(path: Path, rows: list[dict]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLUSTER_DISTRIBUTION_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _include_edge(row: dict, graph_type: str) -> bool:
    final = row.get("final_relation")
    static = row.get("static_relation")
    if graph_type == "order_sensitive_graph":
        return final == "final_order_sensitive"
    if graph_type == "unknown_graph":
        return final == "final_unknown"
    if graph_type == "noncommute_graph":
        return final in {"final_order_sensitive", "final_unknown"}
    if graph_type == "static_overlap_graph":
        return static in {"static_overlap_block", "static_overlap_function"}
    return False
