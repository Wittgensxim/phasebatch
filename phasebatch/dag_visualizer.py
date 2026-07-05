from __future__ import annotations

import csv
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


DAG_METRIC_FIELDS = [
    "program",
    "run_dir",
    "unique_states",
    "transitions",
    "duplicate_transitions",
    "merge_rate",
    "max_depth",
    "leaf_states",
    "selected_path_steps",
    "selected_path_pass_invocations",
    "root_state",
    "selected_final_state",
    "final_objective",
    "root_objective",
    "objective_delta",
    "full_dot_path",
    "full_svg_path",
    "selected_svg_path",
    "depth_overview_svg_path",
]

DAG_DEPTH_METRIC_FIELDS = [
    "program",
    "depth",
    "states",
    "outgoing_transitions",
    "incoming_transitions",
    "duplicate_transitions",
    "avg_active_passes",
    "avg_tested_pairs",
    "avg_commute_pairs",
    "avg_order_sensitive_pairs",
    "avg_local_reduction_log10",
]

DAG_PATH_FIELDS = [
    "path_kind",
    "step",
    "parent_state_id",
    "batch_id",
    "child_state_id",
    "batch_passes",
    "canonical_order",
    "validation_status",
    "correctness_class",
    "ir_inst_before",
    "ir_inst_after",
    "ir_inst_delta",
]

MISSING_INPUT_FIELDS = ["program", "expected_file", "status"]


def visualize_dag(
    run_dir: Path,
    out_dir: Path,
    *,
    view: str = "all",
    formats: list[str] | None = None,
    max_full_nodes: int = 200,
    include_selected_path: bool = False,
    include_depth_overview: bool = False,
) -> dict:
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    formats = formats or ["dot"]
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_graph(run_dir)
    warnings = list(data["warnings"])
    missing_rows = data["missing_rows"]
    states = data["states"]
    edges = data["edges"]
    selected = data["selected"]
    program = data["program"]

    full_dot = out_dir / "state_dag_full.dot"
    selected_dot = out_dir / "state_dag_selected.dot"
    depth_dot = out_dir / "depth_overview.dot"

    _write_dot(full_dot, _dot_for_graph("Compressed Batch-State DAG", states, edges, selected))

    render_full = len(states) <= max_full_nodes
    if not render_full:
        warnings.append(f"graph too large for full render: states={len(states)} max_full_nodes={max_full_nodes}")

    selected_edges = _selected_view_edges(view, states, edges, selected)
    selected_states = _states_for_edges(states, selected_edges)
    if not selected_states:
        root_id = _root_state_id(states)
        if root_id:
            selected_states[root_id] = states[root_id]
        warnings.append("no selected path found")
    _write_dot(selected_dot, _dot_for_graph("Selected Batch-State DAG", selected_states, selected_edges, selected))

    depth_rows = _depth_metric_rows(program, states, edges, data["reduction"])
    _write_dot(depth_dot, _dot_for_depth_overview(depth_rows, edges, states))

    render_warnings, rendered = _render_outputs(
        formats=formats,
        full_dot=full_dot,
        selected_dot=selected_dot,
        depth_dot=depth_dot,
        render_full=render_full,
        render_selected=_should_generate_selected(view, include_selected_path),
        render_depth=_should_generate_depth(view, include_depth_overview),
    )
    warnings.extend(render_warnings)

    path_rows = _dag_path_rows(data["chosen_rows"])
    _write_csv(out_dir / "dag_paths.csv", DAG_PATH_FIELDS, path_rows)
    _write_csv(out_dir / "dag_depth_metrics.csv", DAG_DEPTH_METRIC_FIELDS, depth_rows)
    _write_csv(out_dir / "missing_inputs.csv", MISSING_INPUT_FIELDS, missing_rows)

    metric_row = _dag_metric_row(
        program,
        run_dir,
        states,
        edges,
        data,
        full_dot,
        rendered.get("full_svg", ""),
        rendered.get("selected_svg", ""),
        rendered.get("depth_svg", ""),
    )
    _write_csv(out_dir / "dag_metrics.csv", DAG_METRIC_FIELDS, [metric_row])
    _write_summary(
        out_dir / "dag_summary.md",
        run_dir,
        program,
        view,
        data,
        metric_row,
        depth_rows,
        path_rows,
        warnings,
        rendered,
        full_dot,
        selected_dot,
        depth_dot,
    )

    return {
        "dag_summary_md": str(out_dir / "dag_summary.md"),
        "dag_metrics_csv": str(out_dir / "dag_metrics.csv"),
        "dag_depth_metrics_csv": str(out_dir / "dag_depth_metrics.csv"),
        "dag_paths_csv": str(out_dir / "dag_paths.csv"),
        "full_dot_path": str(full_dot),
        "selected_dot_path": str(selected_dot),
        "depth_overview_dot_path": str(depth_dot),
        "unique_states": len(states),
        "transitions": len(edges),
        "duplicate_transitions": sum(1 for edge in edges if edge["is_duplicate"]),
        "warnings": warnings,
    }


def _load_graph(run_dir: Path) -> dict:
    warnings: list[str] = []
    missing_rows: list[dict] = []

    states_raw = _read_required(run_dir, "states.csv", missing_rows, warnings)
    state_dag_raw = _read_required(run_dir, "state_dag.csv", missing_rows, warnings)
    batch_transitions_raw = _read_optional(run_dir, "batch_state_transitions.csv", missing_rows)
    chosen_rows = _read_optional(run_dir, "chosen_path.csv", missing_rows)
    leaf_rows = _read_optional(run_dir, "leaf_states.csv", missing_rows)
    reduction_rows = _read_optional(run_dir, "reduction_by_state.csv", missing_rows)
    optimize_summary_exists = (run_dir / "optimize_summary.md").exists()
    if not optimize_summary_exists:
        missing_rows.append(_missing_row(_program_from_rows(states_raw, run_dir), "optimize_summary.md", "missing_optional"))
    _read_optional(run_dir, "frontier_scores.csv", missing_rows)
    replay_rows = _read_optional(run_dir, "pipeline_replay.csv", missing_rows)
    chosen_summary = _first_row(run_dir / "chosen_path_summary.csv")

    states = _state_map(run_dir, states_raw, leaf_rows, chosen_rows, chosen_summary)
    reduction = {row.get("state_id", ""): row for row in reduction_rows if row.get("state_id")}
    edges = _edge_rows(run_dir, state_dag_raw or batch_transitions_raw, states)
    selected = _selected_info(chosen_rows, chosen_summary)
    program = _program_name(run_dir, states_raw, chosen_summary)
    replay = replay_rows[0] if replay_rows else {}

    _ensure_endpoint_states(states, edges, program)

    return {
        "program": program,
        "states": states,
        "edges": edges,
        "chosen_rows": chosen_rows,
        "selected": selected,
        "leaf_rows": leaf_rows,
        "reduction": reduction,
        "chosen_summary": chosen_summary,
        "pipeline_replay": replay,
        "warnings": warnings,
        "missing_rows": missing_rows,
    }


def _read_required(run_dir: Path, name: str, missing_rows: list[dict], warnings: list[str]) -> list[dict]:
    path = run_dir / name
    if not path.exists():
        warnings.append(f"missing required input: {name}")
        missing_rows.append(_missing_row(run_dir.name, name, "missing_required"))
        return []
    rows = _read_csv(path)
    if not rows:
        warnings.append(f"empty required input: {name}")
        missing_rows.append(_missing_row(run_dir.name, name, "empty_required"))
    return rows


def _read_optional(run_dir: Path, name: str, missing_rows: list[dict]) -> list[dict]:
    path = run_dir / name
    if not path.exists():
        missing_rows.append(_missing_row(run_dir.name, name, "missing_optional"))
        return []
    return _read_csv(path)


def _missing_row(program: str, name: str, status: str) -> dict:
    return {"program": program, "expected_file": name, "status": status}


def _state_map(run_dir: Path, states_raw: list[dict], leaf_rows: list[dict], chosen_rows: list[dict], chosen_summary: dict) -> dict[str, dict]:
    leaf_by_id = {row.get("state_id", ""): row for row in leaf_rows if row.get("state_id")}
    objective_by_id = _objective_by_state(chosen_rows, chosen_summary, leaf_by_id)
    states: dict[str, dict] = {}
    for row in states_raw:
        state_id = _value(row, "state_id", "id")
        if not state_id:
            continue
        state_dir = _state_dir(run_dir, row, state_id)
        per_state = _first_row(state_dir / "per_state_summary.csv")
        leaf = leaf_by_id.get(state_id, {})
        states[state_id] = {
            "program": _value(row, "program") or _value(per_state, "program") or run_dir.name,
            "state_id": state_id,
            "state_hash": _value(row, "state_hash", "hash") or _value(per_state, "state_hash"),
            "depth": _int(_value(row, "depth") or _value(per_state, "depth")),
            "state_dir": str(state_dir),
            "active_passes": _metric(row, per_state, state_dir, "active_passes", default=""),
            "pairs_tested": _metric(row, per_state, state_dir, "pairs_tested", "pair_rows", "total_pairs", default=""),
            "dynamic_commute": _metric(row, per_state, state_dir, "dynamic_commute", "commute_pairs", "avg_commute", default=""),
            "order_sensitive": _metric(row, per_state, state_dir, "order_sensitive", "order_sensitive_pairs", default=""),
            "unknown": _metric(row, per_state, state_dir, "unknown", "unknown_pairs", default=""),
            "objective": objective_by_id.get(state_id) or _value(leaf, "objective_value"),
            "is_leaf": _is_true(_value(leaf, "is_leaf")) or _is_true(_value(leaf, "selected_as_final")),
            "leaf_reason": _value(leaf, "leaf_reason"),
        }
    return states


def _metric(row: dict, per_state: dict, state_dir: Path, *names: str, default: str = "") -> str:
    value = _value(row, *names) or _value(per_state, *names)
    if value not in {"", None}:
        return str(value)
    if names and names[0] == "active_passes":
        active = sum(1 for item in _read_csv(state_dir / "pass_profile.csv") if _is_true(item.get("success")) and _is_true(item.get("active")))
        return str(active) if active else default
    if names and names[0] in {"pairs_tested", "dynamic_commute", "order_sensitive", "unknown"}:
        counts = Counter(item.get("final_relation", "") for item in _read_csv(state_dir / "pair_relation.csv"))
        if names[0] == "pairs_tested":
            total = sum(counts.values())
        elif names[0] == "dynamic_commute":
            total = counts.get("final_commute", 0)
        elif names[0] == "order_sensitive":
            total = counts.get("final_order_sensitive", 0)
        else:
            total = counts.get("final_unknown", 0) + counts.get("unknown", 0)
        return str(total) if total else default
    return default


def _objective_by_state(chosen_rows: list[dict], chosen_summary: dict, leaf_by_id: dict[str, dict]) -> dict[str, str]:
    values = {state_id: _value(row, "objective_value") for state_id, row in leaf_by_id.items() if _value(row, "objective_value")}
    if chosen_summary:
        if _value(chosen_summary, "root_ir_inst_count"):
            values.setdefault("S0000", _value(chosen_summary, "root_ir_inst_count"))
        selected = _value(chosen_summary, "selected_final_state")
        if selected and _value(chosen_summary, "final_ir_inst_count"):
            values.setdefault(selected, _value(chosen_summary, "final_ir_inst_count"))
    for row in chosen_rows:
        parent = _value(row, "parent_state_id")
        child = _value(row, "child_state_id")
        if parent and _value(row, "ir_inst_before"):
            values.setdefault(parent, _value(row, "ir_inst_before"))
        if child and _value(row, "ir_inst_after"):
            values.setdefault(child, _value(row, "ir_inst_after"))
    return values


def _edge_rows(run_dir: Path, rows: list[dict], states: dict[str, dict]) -> list[dict]:
    edges = []
    for row in rows:
        source = _value(row, "source_state_id", "source_state", "source", "parent_state_id")
        target = _value(row, "target_state_id", "target_state", "target", "child_state_id")
        if not source or not target:
            continue
        duplicate = _is_true(_value(row, "is_duplicate", "is_duplicate_transition"))
        duplicate_of = _value(row, "duplicate_of")
        visual_target = duplicate_of if duplicate and duplicate_of in states else target
        batch_id = _value(row, "batch_id")
        evidence = _batch_evidence(run_dir, states.get(source, {}), batch_id)
        batch_passes = _value(row, "batch_passes") or _value(evidence["candidate"], "batch_passes")
        canonical_order = _value(row, "canonical_order") or _value(evidence["candidate"], "canonical_order") or batch_passes
        validation_status = _value(row, "validation_status") or _value(evidence["validation"], "validation_status") or _value(evidence["correctness"], "validation_status")
        correctness_class = _value(row, "correctness_class") or _value(evidence["correctness"], "correctness_class")
        edges.append(
            {
                "program": _value(row, "program") or _value(states.get(source, {}), "program") or run_dir.name,
                "source": source,
                "target": visual_target,
                "raw_target": target,
                "source_hash": _value(row, "source_hash", "parent_hash"),
                "target_hash": _value(row, "target_hash", "child_hash"),
                "transition_kind": _value(row, "transition_kind") or "batch",
                "batch_id": batch_id,
                "batch_passes": batch_passes,
                "batch_size": _value(row, "batch_size") or str(len(_split_passes(canonical_order or batch_passes))),
                "canonical_order": canonical_order,
                "validation_status": validation_status,
                "correctness_class": correctness_class,
                "is_duplicate": duplicate,
                "duplicate_of": duplicate_of,
                "ir_inst_delta": _value(row, "ir_inst_delta", "inst_delta"),
            }
        )
    return edges


def _batch_evidence(run_dir: Path, state: dict, batch_id: str) -> dict[str, dict]:
    state_dir = Path(state.get("state_dir") or run_dir / "states" / state.get("state_id", ""))
    return {
        "candidate": _row_by_batch(state_dir / "batch_candidates.csv", batch_id),
        "validation": _row_by_batch(state_dir / "batch_validation.csv", batch_id),
        "correctness": _row_by_batch(state_dir / "batch_correctness.csv", batch_id),
    }


def _row_by_batch(path: Path, batch_id: str) -> dict:
    if not batch_id:
        return {}
    for row in _read_csv(path):
        if row.get("batch_id") == batch_id:
            return row
    return {}


def _selected_info(chosen_rows: list[dict], chosen_summary: dict) -> dict:
    edges = set()
    states = set()
    for row in chosen_rows:
        parent = _value(row, "parent_state_id")
        child = _value(row, "child_state_id")
        batch_id = _value(row, "batch_id")
        if parent:
            states.add(parent)
        if child:
            states.add(child)
        if parent and child:
            edges.add((parent, child, batch_id))
    final = _value(chosen_summary, "selected_final_state")
    if final:
        states.add(final)
    return {"edge_keys": edges, "states": states, "selected_final_state": final}


def _ensure_endpoint_states(states: dict[str, dict], edges: list[dict], program: str) -> None:
    for edge in edges:
        for state_id in [edge["source"], edge["target"]]:
            if state_id not in states:
                states[state_id] = {
                    "program": program,
                    "state_id": state_id,
                    "state_hash": edge.get("target_hash") if state_id == edge["target"] else edge.get("source_hash", ""),
                    "depth": 0,
                    "state_dir": "",
                    "active_passes": "",
                    "pairs_tested": "",
                    "dynamic_commute": "",
                    "order_sensitive": "",
                    "unknown": "",
                    "objective": "",
                    "is_leaf": False,
                    "leaf_reason": "",
                }


def _dot_for_graph(title: str, states: dict[str, dict], edges: list[dict], selected: dict) -> str:
    lines = [
        "digraph G {",
        f'  graph [label="{_escape_dot(title)}", labelloc=t, fontsize=16];',
        "  rankdir=TB;",
        "  compound=true;",
        "  nodesep=0.4;",
        "  ranksep=0.6;",
        '  node [shape=box, style="rounded,filled", fillcolor="white", fontname="Consolas", fontsize=10];',
        '  edge [fontname="Consolas", fontsize=9];',
        "",
    ]
    for state_id in sorted(states, key=_state_sort_key):
        lines.append(f"  {state_id} [{_node_attrs(states[state_id], selected)}];")
    if states:
        lines.append("")
        for depth, ids in _states_by_depth(states).items():
            lines.append("  { rank=same; " + "; ".join(ids) + "; }")
    if edges:
        lines.append("")
        for edge in edges:
            lines.append(f"  {edge['source']} -> {edge['target']} [{_edge_attrs(edge, selected)}];")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _node_attrs(state: dict, selected: dict) -> str:
    state_id = state.get("state_id", "")
    label_parts = [state_id]
    label_parts.append(f"d={state.get('depth', '')}")
    if state.get("objective"):
        label_parts.append(f"inst={state['objective']}")
    if state.get("active_passes") != "":
        label_parts.append(f"act={state['active_passes']}")
    if state.get("pairs_tested") != "":
        label_parts.append(f"pairs={state['pairs_tested']}")
    if state.get("dynamic_commute") != "" or state.get("order_sensitive") != "":
        label_parts.append(f"C={state.get('dynamic_commute', '')} / S={state.get('order_sensitive', '')}")
    attrs = {
        "label": "\\n".join(str(part) for part in label_parts if str(part) != ""),
        "fillcolor": "#ffffff",
    }
    style = "rounded,filled"
    if state_id == "S0000":
        attrs["color"] = "black"
        attrs["penwidth"] = "2"
    if state_id in selected.get("states", set()):
        attrs["color"] = "green"
        attrs["penwidth"] = "3"
    if state.get("is_leaf"):
        attrs["shape"] = "doublecircle"
    attrs["style"] = style
    return _attr_text(attrs)


def _edge_attrs(edge: dict, selected: dict) -> str:
    selected_edge = _edge_is_selected(edge, selected)
    label = _edge_label(edge)
    attrs = {"label": label}
    if selected_edge:
        attrs["color"] = "green"
        attrs["penwidth"] = "3"
    elif edge["is_duplicate"]:
        attrs["color"] = "gray"
        attrs["style"] = "dashed"
    elif edge.get("validation_status") == "sampled_same" or edge.get("correctness_class") == "sampled_batch":
        attrs["color"] = "orange"
        attrs["style"] = "dashed"
    elif edge.get("validation_status") in {"mismatch", "failed"} or edge.get("correctness_class") in {"rejected_batch", "failed_batch"}:
        attrs["color"] = "red"
        attrs["style"] = "dotted"
    else:
        attrs["color"] = "blue"
    return _attr_text(attrs)


def _edge_label(edge: dict) -> str:
    parts = []
    if edge.get("batch_id"):
        parts.append(edge["batch_id"])
    if edge.get("batch_size"):
        parts.append(f"size={edge['batch_size']}")
    if edge.get("ir_inst_delta"):
        parts.append(f"delta={edge['ir_inst_delta']}")
    pass_label = _abbreviated_passes(edge.get("batch_passes") or edge.get("canonical_order", ""))
    if pass_label:
        parts.append(pass_label)
    if edge.get("is_duplicate"):
        parts.append(f"duplicate -> {edge.get('duplicate_of') or edge.get('target')}")
    else:
        parts.append(_strength_label(edge))
    return "\\n".join(parts)


def _strength_label(edge: dict) -> str:
    if edge.get("validation_status") == "all_permutations_same" or edge.get("correctness_class") == "certified_batch":
        return "strong"
    if edge.get("validation_status") == "sampled_same" or edge.get("correctness_class") == "sampled_batch":
        return "weak"
    if edge.get("validation_status") in {"mismatch", "failed"}:
        return edge.get("validation_status", "")
    return edge.get("correctness_class") or edge.get("validation_status") or ""


def _dot_for_depth_overview(depth_rows: list[dict], edges: list[dict], states: dict[str, dict]) -> str:
    lines = [
        "digraph G {",
        '  graph [label="Compressed Batch-State DAG Overview", labelloc=t, fontsize=16];',
        "  rankdir=TB;",
        '  node [shape=box, style="rounded,filled", fillcolor="white", fontname="Consolas", fontsize=10];',
        '  edge [fontname="Consolas", fontsize=9];',
        "",
    ]
    for row in depth_rows:
        depth = row["depth"]
        label = f"depth {depth}\\nstates: {row['states']}"
        if row.get("avg_active_passes"):
            label += f"\\navg act: {row['avg_active_passes']}"
        if row.get("avg_local_reduction_log10"):
            label += f"\\navg red log10: {row['avg_local_reduction_log10']}"
        lines.append(f'  D{depth} [label="{_escape_dot(label)}"];')
    for (source_depth, target_depth), grouped in sorted(_edges_by_depth(edges, states).items()):
        dup = sum(1 for edge in grouped if edge["is_duplicate"])
        lines.append(f'  D{source_depth} -> D{target_depth} [label="transitions={len(grouped)}, dup={dup}"];')
    lines.append("}")
    return "\n".join(lines) + "\n"


def _depth_metric_rows(program: str, states: dict[str, dict], edges: list[dict], reduction: dict[str, dict]) -> list[dict]:
    by_depth: dict[int, list[dict]] = defaultdict(list)
    for state in states.values():
        by_depth[_int(state.get("depth"))].append(state)
    outgoing = Counter(_int(states.get(edge["source"], {}).get("depth")) for edge in edges)
    incoming = Counter(_int(states.get(edge["target"], {}).get("depth")) for edge in edges)
    duplicate = Counter(_int(states.get(edge["source"], {}).get("depth")) for edge in edges if edge["is_duplicate"])
    rows = []
    for depth in sorted(by_depth):
        depth_states = by_depth[depth]
        rows.append(
            {
                "program": program,
                "depth": str(depth),
                "states": str(len(depth_states)),
                "outgoing_transitions": str(outgoing.get(depth, 0)),
                "incoming_transitions": str(incoming.get(depth, 0)),
                "duplicate_transitions": str(duplicate.get(depth, 0)),
                "avg_active_passes": _avg(depth_states, "active_passes"),
                "avg_tested_pairs": _avg(depth_states, "pairs_tested"),
                "avg_commute_pairs": _avg(depth_states, "dynamic_commute"),
                "avg_order_sensitive_pairs": _avg(depth_states, "order_sensitive"),
                "avg_local_reduction_log10": _avg([reduction.get(state["state_id"], {}) for state in depth_states], "local_reduction_log10"),
            }
        )
    return rows


def _selected_view_edges(view: str, states: dict[str, dict], edges: list[dict], selected: dict) -> list[dict]:
    if view == "selected-only":
        return [edge for edge in edges if _edge_is_selected(edge, selected)]
    if view == "selected-neighborhood":
        return _selected_neighborhood_edges(states, edges, selected)
    if view == "depth-overview":
        return []
    return _selected_neighborhood_edges(states, edges, selected)


def _selected_neighborhood_edges(states: dict[str, dict], edges: list[dict], selected: dict) -> list[dict]:
    selected_states = set(selected.get("states", set()))
    if not selected_states:
        return []
    included = []
    for edge in edges:
        if _edge_is_selected(edge, selected) or edge["source"] in selected_states or edge["target"] in selected_states:
            included.append(edge)
    return included


def _states_for_edges(states: dict[str, dict], edges: list[dict]) -> dict[str, dict]:
    ids = set()
    for edge in edges:
        ids.add(edge["source"])
        ids.add(edge["target"])
    return {state_id: states[state_id] for state_id in states if state_id in ids}


def _edge_is_selected(edge: dict, selected: dict) -> bool:
    keys = selected.get("edge_keys", set())
    return (edge["source"], edge.get("raw_target", edge["target"]), edge.get("batch_id", "")) in keys or (edge["source"], edge["target"], edge.get("batch_id", "")) in keys


def _should_generate_selected(view: str, include_selected_path: bool) -> bool:
    return include_selected_path or view in {"all", "selected-neighborhood", "selected-only"}


def _should_generate_depth(view: str, include_depth_overview: bool) -> bool:
    return include_depth_overview or view in {"all", "depth-overview"}


def _render_outputs(
    *,
    formats: list[str],
    full_dot: Path,
    selected_dot: Path,
    depth_dot: Path,
    render_full: bool,
    render_selected: bool,
    render_depth: bool,
) -> tuple[list[str], dict[str, str]]:
    warnings: list[str] = []
    rendered: dict[str, str] = {}
    non_dot_formats = [fmt for fmt in formats if fmt != "dot"]
    if not non_dot_formats:
        return warnings, rendered
    if shutil.which("dot") is None:
        warnings.append("graphviz unavailable: dot command not found")
        return warnings, rendered
    for fmt in non_dot_formats:
        if render_full:
            out = full_dot.with_suffix(f".{fmt}")
            warning = _run_dot(full_dot, fmt, out)
            if warning:
                warnings.append(warning)
            elif fmt == "svg":
                rendered["full_svg"] = str(out)
        if render_selected:
            out = selected_dot.with_suffix(f".{fmt}")
            warning = _run_dot(selected_dot, fmt, out)
            if warning:
                warnings.append(warning)
            elif fmt == "svg":
                rendered["selected_svg"] = str(out)
        if render_depth:
            out = depth_dot.with_suffix(f".{fmt}")
            warning = _run_dot(depth_dot, fmt, out)
            if warning:
                warnings.append(warning)
            elif fmt == "svg":
                rendered["depth_svg"] = str(out)
    return warnings, rendered


def _run_dot(dot_path: Path, fmt: str, output_path: Path) -> str:
    dot_command = shutil.which("dot")
    if dot_command is None:
        return "graphviz unavailable: dot command not found"
    command = [dot_command, f"-T{fmt}", str(dot_path), "-o", str(output_path)]
    if Path(dot_command).suffix.lower() in {".bat", ".cmd"}:
        command = ["cmd", "/c", *command]
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return f"graphviz render failed for {dot_path.name} as {fmt}: {result.stderr.strip()}"
    return ""


def _dag_path_rows(chosen_rows: list[dict]) -> list[dict]:
    rows = []
    for index, row in enumerate(chosen_rows):
        rows.append(
            {
                "path_kind": "selected",
                "step": _value(row, "step") or str(index),
                "parent_state_id": _value(row, "parent_state_id"),
                "batch_id": _value(row, "batch_id"),
                "child_state_id": _value(row, "child_state_id"),
                "batch_passes": _value(row, "batch_passes"),
                "canonical_order": _value(row, "canonical_order"),
                "validation_status": _value(row, "validation_status"),
                "correctness_class": _value(row, "correctness_class"),
                "ir_inst_before": _value(row, "ir_inst_before"),
                "ir_inst_after": _value(row, "ir_inst_after"),
                "ir_inst_delta": _value(row, "ir_inst_delta"),
            }
        )
    return rows


def _dag_metric_row(
    program: str,
    run_dir: Path,
    states: dict[str, dict],
    edges: list[dict],
    data: dict,
    full_dot: Path,
    full_svg: str,
    selected_svg: str,
    depth_svg: str,
) -> dict:
    chosen_summary = data["chosen_summary"]
    selected_final = _value(chosen_summary, "selected_final_state") or _selected_final_from_leaf(data["leaf_rows"]) or _selected_final_from_path(data["chosen_rows"])
    root_state = _root_state_id(states)
    root_objective = _value(chosen_summary, "root_ir_inst_count") or _value(states.get(root_state, {}), "objective")
    final_objective = _value(chosen_summary, "final_ir_inst_count") or _value(states.get(selected_final, {}), "objective")
    duplicate_count = sum(1 for edge in edges if edge["is_duplicate"])
    return {
        "program": program,
        "run_dir": str(run_dir),
        "unique_states": str(len(states)),
        "transitions": str(len(edges)),
        "duplicate_transitions": str(duplicate_count),
        "merge_rate": _fmt_float(duplicate_count / max(1, len(edges))),
        "max_depth": str(max((_int(state.get("depth")) for state in states.values()), default=0)),
        "leaf_states": str(_leaf_count(states, edges, data["leaf_rows"])),
        "selected_path_steps": str(len(data["chosen_rows"])),
        "selected_path_pass_invocations": _value(chosen_summary, "total_pass_invocations") or str(sum(len(_split_passes(row.get("canonical_order") or row.get("batch_passes", ""))) for row in data["chosen_rows"])),
        "root_state": root_state,
        "selected_final_state": selected_final,
        "final_objective": final_objective,
        "root_objective": root_objective,
        "objective_delta": _numeric_delta(root_objective, final_objective),
        "full_dot_path": str(full_dot),
        "full_svg_path": full_svg,
        "selected_svg_path": selected_svg,
        "depth_overview_svg_path": depth_svg,
    }


def _write_summary(
    path: Path,
    run_dir: Path,
    program: str,
    view: str,
    data: dict,
    metric: dict,
    depth_rows: list[dict],
    path_rows: list[dict],
    warnings: list[str],
    rendered: dict[str, str],
    full_dot: Path,
    selected_dot: Path,
    depth_dot: Path,
) -> None:
    lines = [
        "# State DAG Visualization Summary",
        "",
        "## Input",
        "",
        f"- run_dir: {run_dir}",
        f"- program: {program}",
        f"- selected mode: {view}",
        f"- objective: ir-inst-count",
        "",
        "## DAG Size",
        "",
        f"- unique states: {metric['unique_states']}",
        f"- transitions: {metric['transitions']}",
        f"- duplicate transitions: {metric['duplicate_transitions']}",
        f"- merge rate: {metric['merge_rate']}",
        f"- max depth: {metric['max_depth']}",
        f"- leaf states: {metric['leaf_states']}",
        "",
        "## Search-Space Compression Interpretation",
        "",
        "This graph visualizes the compressed batch-state search space. Nodes are canonical IR states. Edges are certified batch transitions. Multiple paths reaching the same canonical IR state are merged, turning a tree-like search into a DAG.",
        "",
        "## Generated Files",
        "",
        f"- full DOT: {full_dot}",
        f"- full SVG: {rendered.get('full_svg', '')}",
        f"- selected path DOT: {selected_dot}",
        f"- selected path SVG: {rendered.get('selected_svg', '')}",
        f"- depth overview DOT: {depth_dot}",
        f"- depth overview SVG: {rendered.get('depth_svg', '')}",
        f"- metrics CSV: {path.parent / 'dag_metrics.csv'}",
        f"- depth metrics CSV: {path.parent / 'dag_depth_metrics.csv'}",
        "",
        "## Selected Path",
        "",
    ]
    if path_rows:
        lines.extend(
            _markdown_table(
                ["step", "parent", "batch", "child", "validation", "objective delta"],
                [
                    [
                        row.get("step", ""),
                        row.get("parent_state_id", ""),
                        row.get("batch_id", ""),
                        row.get("child_state_id", ""),
                        row.get("validation_status", ""),
                        row.get("ir_inst_delta", ""),
                    ]
                    for row in path_rows
                ],
            )
        )
    else:
        lines.append("- No selected path was found.")
    lines.extend(
        [
            "",
            "## Depth Overview",
            "",
            *_markdown_table(
                ["depth", "states", "transitions", "duplicates", "avg active passes", "avg reduction log10"],
                [
                    [
                        row.get("depth", ""),
                        row.get("states", ""),
                        row.get("outgoing_transitions", ""),
                        row.get("duplicate_transitions", ""),
                        row.get("avg_active_passes", ""),
                        row.get("avg_local_reduction_log10", ""),
                    ]
                    for row in depth_rows
                ],
            ),
            "",
            "## Warnings",
            "",
        ]
    )
    if warnings:
        lines.extend([f"- {warning}" for warning in warnings])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Correctness Boundary",
            "",
            "Graph edges visualize executed or observed batch transitions. The visualization does not create new correctness evidence; it only displays evidence already recorded by batch validation and correctness classification.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _leaf_count(states: dict[str, dict], edges: list[dict], leaf_rows: list[dict]) -> int:
    explicit = [row for row in leaf_rows if _is_true(row.get("is_leaf"))]
    if explicit:
        return len(explicit)
    outgoing = {edge["source"] for edge in edges}
    return sum(1 for state_id in states if state_id not in outgoing)


def _selected_final_from_leaf(leaf_rows: list[dict]) -> str:
    for row in leaf_rows:
        if _is_true(row.get("selected_as_final")):
            return row.get("state_id", "")
    return ""


def _selected_final_from_path(chosen_rows: list[dict]) -> str:
    return chosen_rows[-1].get("child_state_id", "") if chosen_rows else ""


def _root_state_id(states: dict[str, dict]) -> str:
    if "S0000" in states:
        return "S0000"
    if not states:
        return ""
    return min(states.values(), key=lambda state: (_int(state.get("depth")), state.get("state_id", ""))).get("state_id", "")


def _states_by_depth(states: dict[str, dict]) -> dict[int, list[str]]:
    by_depth: dict[int, list[str]] = defaultdict(list)
    for state_id, state in states.items():
        by_depth[_int(state.get("depth"))].append(state_id)
    return {depth: sorted(ids, key=_state_sort_key) for depth, ids in sorted(by_depth.items())}


def _edges_by_depth(edges: list[dict], states: dict[str, dict]) -> dict[tuple[int, int], list[dict]]:
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for edge in edges:
        source_depth = _int(states.get(edge["source"], {}).get("depth"))
        target_depth = _int(states.get(edge["target"], {}).get("depth"))
        grouped[(source_depth, target_depth)].append(edge)
    return grouped


def _state_dir(run_dir: Path, row: dict, state_id: str) -> Path:
    raw = _value(row, "state_dir")
    if raw:
        return Path(raw)
    return run_dir / "states" / state_id


def _program_name(run_dir: Path, states_raw: list[dict], chosen_summary: dict) -> str:
    return _value(chosen_summary, "program") or _program_from_rows(states_raw, run_dir)


def _program_from_rows(rows: list[dict], run_dir: Path | str) -> str:
    for row in rows:
        if row.get("program"):
            return row["program"]
    return Path(run_dir).name


def _write_dot(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _first_row(path: Path) -> dict:
    rows = _read_csv(path)
    return rows[0] if rows else {}


def _value(row: dict, *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value not in {None, ""}:
            return str(value)
    return ""


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _int(value: object) -> int:
    try:
        return int(float(str(value or "0")))
    except (TypeError, ValueError):
        return 0


def _float(value: object) -> float:
    try:
        return float(str(value or "0"))
    except (TypeError, ValueError):
        return 0.0


def _avg(rows: list[dict], field: str) -> str:
    values = [_float(row.get(field)) for row in rows if row.get(field) not in {None, ""}]
    if not values:
        return ""
    return _fmt_float(sum(values) / len(values))


def _fmt_float(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _numeric_delta(left: str, right: str) -> str:
    if left == "" or right == "":
        return ""
    delta = _float(right) - _float(left)
    if delta.is_integer():
        return str(int(delta))
    return _fmt_float(delta)


def _split_passes(text: str) -> list[str]:
    if not text:
        return []
    return [part.strip() for part in re.split(r"[;,]", text) if part.strip()]


def _abbreviated_passes(text: str) -> str:
    parts = _split_passes(text)
    if not parts:
        return ""
    if len(parts) <= 3:
        return "+".join(parts)
    return "+".join(parts[:2]) + f"+...(+{len(parts) - 2})"


def _attr_text(attrs: dict[str, str]) -> str:
    parts = []
    for key, value in attrs.items():
        if key == "penwidth":
            parts.append(f"{key}={value}")
        else:
            parts.append(f'{key}="{_escape_dot(str(value))}"')
    return ", ".join(parts)


def _escape_dot(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _escape_md(value: object) -> str:
    return " ".join(str(value).splitlines()).replace("|", "\\|")


def _markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_escape_md(value) for value in row) + " |")
    return lines


def _state_sort_key(value: str | dict) -> tuple[int, str]:
    state_id = value if isinstance(value, str) else value.get("state_id", "")
    match = re.search(r"(\d+)$", state_id)
    return (int(match.group(1)) if match else 10**9, state_id)
