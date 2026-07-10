from __future__ import annotations

import csv
import itertools
import json
import re
from collections import Counter
from pathlib import Path

from .equality_summary import equality_tier_markdown, equality_tier_summary_from_rows
from .schema import COMPONENT_EDGE_FIELDS, COMPONENT_PROGRAM_SUMMARY_FIELDS, COMPONENT_SUMMARY_FIELDS


def summarize_components(
    *,
    run_dir: Path | None = None,
    run_dirs: list[Path] | None = None,
    out_dir: Path | None = None,
) -> dict:
    if run_dir is not None and run_dirs:
        raise RuntimeError("use either --run-dir or --run-dirs, not both")
    if run_dir is None and not run_dirs:
        raise RuntimeError("one of --run-dir or --run-dirs is required")

    multi_run = run_dir is None
    plans = [Path(path) for path in (run_dirs or [run_dir])]
    if multi_run:
        if out_dir is None:
            raise RuntimeError("--out is required with --run-dirs")
        output_dir = Path(out_dir)
        component_name = "component_summary_all.csv"
        edge_name = "component_edges_all.csv"
    else:
        output_dir = Path(run_dir)
        component_name = "component_summary.csv"
        edge_name = "component_edges.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    component_rows: list[dict] = []
    edge_rows: list[dict] = []
    state_rows: list[dict] = []
    dot_plans: list[dict] = []

    for plan in plans:
        result = _summarize_run(Path(plan))
        component_rows.extend(result["component_rows"])
        edge_rows.extend(result["edge_rows"])
        state_rows.extend(result["state_rows"])
        dot_plans.extend(result["dot_plans"])

    program_rows = _program_summary_rows(component_rows, edge_rows, state_rows)
    component_path = output_dir / component_name
    edge_path = output_dir / edge_name
    program_path = output_dir / "component_program_summary.csv"
    markdown_path = output_dir / "component_summary.md"

    _write_csv(component_path, COMPONENT_SUMMARY_FIELDS, component_rows)
    _write_csv(edge_path, COMPONENT_EDGE_FIELDS, edge_rows)
    _write_csv(program_path, COMPONENT_PROGRAM_SUMMARY_FIELDS, program_rows)
    _write_markdown(markdown_path, component_rows, edge_rows, program_rows, state_rows)
    dot_paths = _write_dot_files(output_dir / "components", dot_plans)

    result = {
        "states": len(state_rows),
        "components": len(component_rows),
        "edges": len(edge_rows),
        "programs": len(program_rows),
        "component_program_summary_csv": str(program_path),
        "component_summary_md": str(markdown_path),
        "dot_files": [str(path) for path in dot_paths],
    }
    if multi_run:
        result["component_summary_all_csv"] = str(component_path)
        result["component_edges_all_csv"] = str(edge_path)
    else:
        result["component_summary_csv"] = str(component_path)
        result["component_edges_csv"] = str(edge_path)
    return result


def _summarize_run(run_dir: Path) -> dict:
    states = _states(run_dir)
    if not states:
        raise RuntimeError(f"missing or empty states.csv in {run_dir}")
    program = _program_name(run_dir, states)
    state_dirs = _state_dirs(run_dir, states)
    selected_states = _selected_path_states(run_dir)

    component_rows: list[dict] = []
    edge_rows: list[dict] = []
    state_rows: list[dict] = []
    dot_candidates: dict[str, dict] = {}

    for state in states:
        state_id = state.get("state_id", "")
        state_dir = state_dirs.get(state_id, run_dir / "states" / state_id)
        state_result = _summarize_state(program, state, state_dir)
        component_rows.extend(state_result["component_rows"])
        edge_rows.extend(state_result["edge_rows"])
        state_rows.append(state_result["state_row"])
        if state_id == "S0000" or state_id in selected_states:
            dot_candidates[state_id] = state_result

    largest = max(
        component_rows,
        key=lambda row: (_int(row.get("component_size")), row.get("program", ""), row.get("state_id", "")),
        default={},
    )
    if largest.get("state_id"):
        largest_state_id = largest["state_id"]
        for state in states:
            if state.get("state_id") == largest_state_id:
                dot_candidates.setdefault(largest_state_id, _summarize_state(program, state, state_dirs.get(largest_state_id, run_dir / "states" / largest_state_id)))
                break

    dot_plans = [
        {
            "program": result["state_row"]["program"],
            "state_id": result["state_row"]["state_id"],
            "active_passes": result["active_passes"],
            "edges": result["edge_rows"],
        }
        for result in dot_candidates.values()
    ]
    return {
        "component_rows": component_rows,
        "edge_rows": edge_rows,
        "state_rows": state_rows,
        "dot_plans": dot_plans,
    }


def _summarize_state(program: str, state: dict, state_dir: Path) -> dict:
    per_state = _first_row(state_dir / "per_state_summary.csv")
    state_id = state.get("state_id") or per_state.get("state_id") or state_dir.name
    depth = state.get("depth") or per_state.get("depth", "")
    active_passes = _active_passes(state_dir)
    relation_rows = _relation_rows(state_dir)
    active_passes = _complete_active_passes(active_passes, relation_rows, state_dir)
    pass_rank = {name: index for index, name in enumerate(active_passes)}
    edges = _edge_rows(program, state_id, depth, active_passes, relation_rows)
    components = _component_defs(state_dir, active_passes, edges, pass_rank)
    component_by_pass = {
        pass_name: component["component_id"]
        for component in components
        for pass_name in component["passes"]
    }
    for edge in edges:
        pass_a = edge.get("pass_a", "")
        pass_b = edge.get("pass_b", "")
        component_id = component_by_pass.get(pass_a, "")
        edge["component_id"] = component_id if component_id and component_id == component_by_pass.get(pass_b, "") else ""

    candidate_rows = _read_csv(state_dir / "batch_candidates.csv")
    component_rows = [
        _component_summary_row(program, state_id, depth, component, edges, candidate_rows)
        for component in components
    ]
    state_row = {
        "program": program,
        "state_id": state_id,
        "depth": str(depth),
        "state_dir": str(state_dir),
        "batch_candidates": str(len(candidate_rows)),
        "max_component_size": str(max((_int(row.get("component_size")) for row in component_rows), default=0)),
    }
    return {
        "state_row": state_row,
        "component_rows": component_rows,
        "edge_rows": edges,
        "active_passes": active_passes,
    }


def _edge_rows(program: str, state_id: str, depth: str, active_passes: list[str], relation_rows: list[dict]) -> list[dict]:
    relation_map = {_pair_key(row.get("pass_a", ""), row.get("pass_b", "")): row for row in relation_rows if row.get("pass_a") and row.get("pass_b")}
    rows = []
    for pass_a, pass_b in itertools.combinations(active_passes, 2):
        relation = relation_map.get(_pair_key(pass_a, pass_b), {})
        final_relation = relation.get("final_relation", "") or "not_tested"
        rows.append(
            {
                "program": program,
                "state_id": relation.get("state_id") or state_id,
                "depth": relation.get("depth") or str(depth),
                "pass_a": pass_a,
                "pass_b": pass_b,
                "relation": final_relation,
                "edge_kind": _edge_kind(final_relation, relation),
                "same_hash": relation.get("same_hash", ""),
                "validation_status": relation.get("validation_status", ""),
                "component_id": "",
            }
        )
    return rows


def _component_defs(state_dir: Path, active_passes: list[str], edges: list[dict], pass_rank: dict[str, int]) -> list[dict]:
    existing = _read_csv(state_dir / "batch_components.csv")
    if existing:
        components = []
        for row in existing:
            passes = _split_passes(row.get("component_passes", ""))
            components.append(
                {
                    "component_id": row.get("component_id", f"C{len(components):04d}"),
                    "passes": sorted(passes, key=lambda name: pass_rank.get(name, len(pass_rank))),
                    "is_exact": row.get("is_exact", ""),
                    "num_local_alternatives": row.get("num_local_alternatives", ""),
                    "unresolved_reason": row.get("unresolved_reason", ""),
                }
            )
        return components

    adjacency = {pass_name: set() for pass_name in active_passes}
    for edge in edges:
        if edge.get("edge_kind") == "commute":
            continue
        pass_a = edge.get("pass_a", "")
        pass_b = edge.get("pass_b", "")
        if pass_a in adjacency and pass_b in adjacency:
            adjacency[pass_a].add(pass_b)
            adjacency[pass_b].add(pass_a)

    return [
        {
            "component_id": f"C{index:04d}",
            "passes": component,
            "is_exact": "true",
            "num_local_alternatives": "1" if len(component) == 1 else "",
            "unresolved_reason": "",
        }
        for index, component in enumerate(_connected_components(active_passes, adjacency, pass_rank))
    ]


def _component_summary_row(program: str, state_id: str, depth: str, component: dict, edges: list[dict], candidate_rows: list[dict]) -> dict:
    component_passes = set(component["passes"])
    inside_edges = [
        edge
        for edge in edges
        if edge.get("pass_a") in component_passes and edge.get("pass_b") in component_passes
    ]
    unknown_edges = sum(1 for edge in inside_edges if edge.get("edge_kind") in {"unknown", "failed", "not_tested"})
    conflict_edges = sum(1 for edge in inside_edges if edge.get("edge_kind") != "commute")
    component_id = component.get("component_id", "")
    contributed = _component_choice_count(candidate_rows, component_id)
    if contributed == 0:
        contributed = _int(component.get("num_local_alternatives"))
    return {
        "program": program,
        "state_id": state_id,
        "depth": str(depth),
        "component_id": component_id,
        "component_size": str(len(component["passes"])),
        "component_passes": ";".join(component["passes"]),
        "conflict_edges": str(conflict_edges),
        "commute_pairs_inside_component": str(sum(1 for edge in inside_edges if edge.get("edge_kind") == "commute")),
        "order_sensitive_edges": str(sum(1 for edge in inside_edges if edge.get("edge_kind") == "order_sensitive")),
        "unknown_edges": str(unknown_edges),
        "is_singleton": _bool(len(component["passes"]) == 1),
        "is_exact": component.get("is_exact") or "true",
        "num_local_alternatives": str(component.get("num_local_alternatives") or contributed),
        "unresolved_reason": component.get("unresolved_reason", ""),
        "batch_candidates_contributed": str(contributed),
    }


def _program_summary_rows(component_rows: list[dict], edge_rows: list[dict], state_rows: list[dict]) -> list[dict]:
    rows = []
    programs = sorted({row.get("program", "") for row in component_rows + state_rows if row.get("program")})
    for program in programs:
        components = [row for row in component_rows if row.get("program") == program]
        states = [row for row in state_rows if row.get("program") == program]
        edges = [row for row in edge_rows if row.get("program") == program]
        state_ids = {(row.get("program", ""), row.get("state_id", "")) for row in states}
        rows.append(
            {
                "program": program,
                "states": str(len(state_ids)),
                "total_components": str(len(components)),
                "singleton_components": str(sum(1 for row in components if row.get("is_singleton") == "true")),
                "non_singleton_components": str(sum(1 for row in components if row.get("is_singleton") != "true")),
                "avg_component_size": _avg([_int(row.get("component_size")) for row in components]),
                "max_component_size": str(max((_int(row.get("component_size")) for row in components), default=0)),
                "avg_conflict_edges": _avg([_int(row.get("conflict_edges")) for row in components]),
                "total_order_sensitive_edges": str(sum(1 for row in edges if row.get("edge_kind") == "order_sensitive")),
                "total_unknown_edges": str(sum(1 for row in edges if row.get("edge_kind") in {"unknown", "failed", "not_tested"})),
                "total_batch_candidates": str(sum(_int(row.get("batch_candidates")) for row in states)),
                "avg_batch_candidates_per_state": _avg([_int(row.get("batch_candidates")) for row in states]),
                "avg_max_component_size_per_state": _avg([_int(row.get("max_component_size")) for row in states]),
            }
        )
    return rows


def _write_markdown(path: Path, component_rows: list[dict], edge_rows: list[dict], program_rows: list[dict], state_rows: list[dict]) -> None:
    singleton = sum(1 for row in component_rows if row.get("is_singleton") == "true")
    total_unknown = sum(1 for row in edge_rows if row.get("edge_kind") in {"unknown", "failed", "not_tested"})
    lines = [
        "# Component / Interaction Graph Summary",
        "",
        "## Overall",
        "",
        f"- states summarized: {len({(row.get('program'), row.get('state_id')) for row in state_rows})}",
        f"- total components: {len(component_rows)}",
        f"- singleton components: {singleton}",
        f"- max component size: {max((_int(row.get('component_size')) for row in component_rows), default=0)}",
        f"- total order-sensitive edges: {sum(1 for row in edge_rows if row.get('edge_kind') == 'order_sensitive')}",
        f"- total unknown edges: {total_unknown}",
        "",
        "## Program Summary",
        "",
        *_markdown_table(
            ["program", "states", "components", "singleton %", "max component size", "avg component size", "batch candidates"],
            [
                [
                    row.get("program", ""),
                    row.get("states", ""),
                    row.get("total_components", ""),
                    _pct(_int(row.get("singleton_components")), _int(row.get("total_components"))),
                    row.get("max_component_size", ""),
                    row.get("avg_component_size", ""),
                    row.get("total_batch_candidates", ""),
                ]
                for row in program_rows
            ],
        ),
        "",
        "## Largest Components",
        "",
        *_markdown_table(
            ["program", "state", "depth", "component size", "passes", "order-sensitive edges", "unknown edges"],
            [
                [
                    row.get("program", ""),
                    row.get("state_id", ""),
                    row.get("depth", ""),
                    row.get("component_size", ""),
                    row.get("component_passes", ""),
                    row.get("order_sensitive_edges", ""),
                    row.get("unknown_edges", ""),
                ]
                for row in sorted(component_rows, key=lambda item: _int(item.get("component_size")), reverse=True)[:10]
            ],
        ),
        "",
        "## Relation Breakdown",
        "",
        *_markdown_table(
            ["program", "commute edges", "order-sensitive edges", "unknown edges"],
            _relation_breakdown_rows(edge_rows),
        ),
        "",
        *equality_tier_markdown(_equality_summary_for_component_states(state_rows)),
        "",
        "## Interpretation",
        "",
        "- Conflict components are built from non-commuting or unknown pass pairs. Batch candidates are generated by combining alternatives from these components with commuting passes.",
        "- This graph is state-local. A component in one state does not imply the same component in another state.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _equality_summary_for_component_states(state_rows: list[dict]) -> list[dict]:
    rows = []
    seen_dirs: set[str] = set()
    for state in state_rows:
        state_dir_value = state.get("state_dir", "")
        if not state_dir_value:
            continue
        state_dir = Path(state_dir_value)
        key = str(state_dir.resolve())
        if key in seen_dirs:
            continue
        seen_dirs.add(key)
        rows.extend(_read_csv(state_dir / "pair_relation.csv"))
    return equality_tier_summary_from_rows(rows)


def _write_dot_files(dot_dir: Path, dot_plans: list[dict]) -> list[Path]:
    dot_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    seen: set[tuple[str, str]] = set()
    for plan in dot_plans:
        key = (plan.get("program", ""), plan.get("state_id", ""))
        if key in seen:
            continue
        seen.add(key)
        path = dot_dir / f"{_safe_name(key[0])}_{_safe_name(key[1])}_interaction.dot"
        _write_dot(path, plan)
        paths.append(path)
    return paths


def _write_dot(path: Path, plan: dict) -> None:
    lines = [
        f"graph {_dot_id(plan.get('program', 'program') + '_' + plan.get('state_id', 'state'))} {{",
        "  rankdir=LR;",
    ]
    for pass_name in plan.get("active_passes", []):
        lines.append(f"  {_dot_id(pass_name)};")
    for edge in plan.get("edges", []):
        kind = edge.get("edge_kind", "")
        if kind == "commute":
            continue
        if kind == "order_sensitive":
            attr = 'color=red, style=solid, label="order_sensitive"'
        else:
            attr = f'color=gray, style=dashed, label="{kind or "unknown"}"'
        lines.append(f"  {_dot_id(edge.get('pass_a', ''))} -- {_dot_id(edge.get('pass_b', ''))} [{attr}];")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _relation_breakdown_rows(edge_rows: list[dict]) -> list[list[str]]:
    rows = []
    for program in sorted({row.get("program", "") for row in edge_rows if row.get("program")}):
        program_edges = [row for row in edge_rows if row.get("program") == program]
        rows.append(
            [
                program,
                str(sum(1 for row in program_edges if row.get("edge_kind") == "commute")),
                str(sum(1 for row in program_edges if row.get("edge_kind") == "order_sensitive")),
                str(sum(1 for row in program_edges if row.get("edge_kind") in {"unknown", "failed", "not_tested"})),
            ]
        )
    return rows


def _states(run_dir: Path) -> list[dict]:
    states = _read_csv(run_dir / "states.csv")
    if states:
        return states
    states_root = run_dir / "states"
    if not states_root.exists():
        return []
    return [
        {"program": _program_name(run_dir, []), "state_id": path.name, "depth": "", "state_hash": "", "state_dir": str(path)}
        for path in sorted(states_root.iterdir())
        if path.is_dir()
    ]


def _state_dirs(run_dir: Path, states: list[dict]) -> dict[str, Path]:
    mapping = {
        row.get("state_id", ""): Path(row.get("state_dir") or run_dir / "states" / row.get("state_id", ""))
        for row in states
        if row.get("state_id")
    }
    states_root = run_dir / "states"
    if states_root.exists():
        for path in states_root.iterdir():
            if path.is_dir():
                mapping.setdefault(path.name, path)
    return mapping


def _selected_path_states(run_dir: Path) -> set[str]:
    states: set[str] = set()
    for row in _read_csv(run_dir / "chosen_path.csv"):
        if row.get("parent_state_id"):
            states.add(row["parent_state_id"])
        if row.get("child_state_id"):
            states.add(row["child_state_id"])
    for row in _read_csv(run_dir / "leaf_states.csv"):
        if row.get("selected_as_final") == "true" and row.get("state_id"):
            states.add(row["state_id"])
    return states


def _active_passes(state_dir: Path) -> list[str]:
    rows = _read_csv(state_dir / "pass_profile.csv")
    passes = [row.get("pass", "") for row in rows if row.get("pass") and _is_true(row.get("success")) and _is_true(row.get("active"))]
    return _dedupe(passes)


def _complete_active_passes(active_passes: list[str], relation_rows: list[dict], state_dir: Path) -> list[str]:
    names = list(active_passes)
    for row in relation_rows:
        if row.get("pass_a"):
            names.append(row["pass_a"])
        if row.get("pass_b"):
            names.append(row["pass_b"])
    for row in _read_csv(state_dir / "batch_components.csv"):
        names.extend(_split_passes(row.get("component_passes", "")))
    return _dedupe(names)


def _relation_rows(state_dir: Path) -> list[dict]:
    return [row for row in _read_csv(state_dir / "pair_relation.csv") if row.get("pass_a") and row.get("pass_b")]


def _edge_kind(final_relation: str, row: dict) -> str:
    failure = row.get("failure_kind", "")
    if failure:
        return "failed"
    if final_relation == "final_commute":
        return "commute"
    if final_relation == "final_order_sensitive":
        return "order_sensitive"
    if final_relation in {"", "not_tested"}:
        return "not_tested"
    if "failed" in final_relation:
        return "failed"
    return "unknown"


def _connected_components(active_passes: list[str], adjacency: dict[str, set[str]], pass_rank: dict[str, int]) -> list[list[str]]:
    seen: set[str] = set()
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
            for neighbor in sorted(adjacency.get(current, set()), key=lambda name: pass_rank.get(name, len(pass_rank))):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)
        components.append(sorted(component, key=lambda name: pass_rank.get(name, len(pass_rank))))
    return components


def _component_choice_count(candidate_rows: list[dict], component_id: str) -> int:
    choices = set()
    prefix = f"{component_id}:"
    for row in candidate_rows:
        for item in str(row.get("component_choices", "")).split("|"):
            if item.startswith(prefix):
                choices.add(item[len(prefix) :])
    return len(choices)


def _program_name(run_dir: Path, states: list[dict]) -> str:
    metadata = _metadata(run_dir)
    if metadata.get("input"):
        return Path(str(metadata["input"])).stem
    if run_dir.name == "optimize" and run_dir.parent.name:
        return run_dir.parent.name
    for row in states:
        if row.get("program") and row.get("program") != "optimize":
            return row["program"]
    return run_dir.name


def _metadata(run_dir: Path) -> dict:
    path = run_dir / "metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _pair_key(pass_a: str, pass_b: str) -> tuple[str, str]:
    return tuple(sorted([pass_a, pass_b]))


def _split_passes(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"[;\n,]+", str(value or "")) if part.strip()]


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


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


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_escape_cell(value) for value in row) + " |")
    return lines


def _escape_cell(value: object) -> str:
    return " ".join(str(value).splitlines()).replace("|", "\\|")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "state")).strip("_") or "state"


def _dot_id(value: str) -> str:
    text = str(value or "node")
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", text):
        return text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _pct(part: int, total: int) -> str:
    if total <= 0:
        return "0%"
    return f"{(part / total) * 100:.1f}%"


def _avg(values: list[int]) -> str:
    if not values:
        return "0"
    value = sum(values) / len(values)
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _int(value: object) -> int:
    try:
        return int(float(str(value or "0")))
    except ValueError:
        return 0


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _bool(value: bool) -> str:
    return "true" if value else "false"
