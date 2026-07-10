from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from .equality_summary import equality_tier_markdown, equality_tier_summary_for_run
from .normalizer import canonical_hash
from .pair_cost import write_pair_cost_summary
from .pass_config import load_pass_registry
from .profiler import validate_passes
from .runner import prepare_input_ir
from .schema import (
    AGGREGATE_BY_DEPTH_FIELDS,
    ENABLE_SUPPRESS_FIELDS,
    RELATION_FLIP_FIELDS,
    STATE_FIELDS,
    STATE_TRANSITION_FIELDS,
)
from .state_analysis import analyze_state
from .tools import collect_toolchain, write_metadata


PAIR_AVAILABILITY_CHANGE_KINDS = {"active_pair_to_missing", "missing_to_active_pair"}
TRUE_RELATION_FLIP_KINDS = {
    "commute_to_sensitive",
    "sensitive_to_commute",
    "known_to_unknown",
    "unknown_to_known",
    "other_flip",
}


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
            "frontier_policy": frontier_policy,
            "top_k": top_k,
        }
    )
    write_metadata(out_dir, metadata)
    tools = _tool_paths(metadata)
    tools["_pass_registry"] = pass_registry

    root_ir = prepare_input_ir(Path(input_path), out_dir, tools, timeout)
    valid_passes, invalid_rows = validate_passes(root_ir, configured_passes, tools, out_dir, timeout, pass_registry=pass_registry)
    root_hash = canonical_hash(root_ir)

    state_rows: list[dict] = []
    transition_rows: list[dict] = []
    relation_flip_rows: list[dict] = []
    enable_suppress_rows: list[dict] = []
    canonical_rows_by_id: dict[str, dict] = {}
    hash_to_state_id: dict[str, str] = {root_hash: "S0000"}
    frontier: list[dict] = []
    next_state_number = 1

    root_dir = states_dir / "S0000"
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
                _analyze_state(
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
            relation_flip_rows.extend(_relation_flip_rows(program, parent, child_row, active.get("pass", "")))
            enable_suppress_rows.extend(
                _enable_suppress_rows(program, parent, child_row, active.get("pass", ""), valid_passes)
            )

    _write_csv(out_dir / "states.csv", STATE_FIELDS, state_rows)
    _write_csv(out_dir / "state_transitions.csv", STATE_TRANSITION_FIELDS, transition_rows)
    _write_csv(out_dir / "relation_flip.csv", RELATION_FLIP_FIELDS, relation_flip_rows)
    _write_csv(out_dir / "enable_suppress.csv", ENABLE_SUPPRESS_FIELDS, enable_suppress_rows)
    aggregate_rows = _aggregate_by_depth(out_dir, program)
    _write_csv(out_dir / "aggregate_by_depth.csv", AGGREGATE_BY_DEPTH_FIELDS, aggregate_rows)
    _write_multistate_summary(out_dir / "multistate_summary.md", out_dir, aggregate_rows)
    pair_cost = write_pair_cost_summary(out_dir)
    return {
        "program": program,
        "out_dir": str(out_dir),
        "states": len(state_rows),
        "transitions": len(transition_rows),
        "states_csv": str(out_dir / "states.csv"),
        "state_transitions_csv": str(out_dir / "state_transitions.csv"),
        "relation_flip_csv": str(out_dir / "relation_flip.csv"),
        "enable_suppress_csv": str(out_dir / "enable_suppress.csv"),
        "aggregate_by_depth_csv": str(out_dir / "aggregate_by_depth.csv"),
        "multistate_summary": str(out_dir / "multistate_summary.md"),
        "pair_cost_summary_csv": pair_cost["pair_cost_summary_csv"],
        "pair_cost_summary_md": pair_cost["pair_cost_summary_md"],
    }


def _tool_paths(metadata: dict) -> dict[str, str]:
    tools = {
        name: details["path"]
        for name, details in metadata.get("tools", {}).items()
        if details.get("path")
    }
    tools["_toolchain_metadata"] = metadata
    return tools


def _analyze_state(input_ll: Path, state_dir: Path, tools: dict, **kwargs) -> dict:
    try:
        return analyze_state(input_ll, state_dir, tools, **kwargs)
    except TypeError as exc:
        if "pass_registry" not in str(exc):
            raise
        fallback = dict(kwargs)
        fallback.pop("pass_registry", None)
        return analyze_state(input_ll, state_dir, tools, **fallback)


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


def _relation_flip_rows(program: str, parent: dict, child: dict, transition_pass: str) -> list[dict]:
    parent_pairs = _pair_relation_map(Path(parent["state_dir"]) / "pair_relation.csv")
    child_pairs = _pair_relation_map(Path(child["state_dir"]) / "pair_relation.csv")
    rows = []
    for pass_a, pass_b in sorted(parent_pairs.keys() | child_pairs.keys()):
        parent_present = (pass_a, pass_b) in parent_pairs
        child_present = (pass_a, pass_b) in child_pairs
        parent_relation = parent_pairs.get((pass_a, pass_b), "")
        child_relation = child_pairs.get((pass_a, pass_b), "")
        rows.append(
            {
                "program": program,
                "parent_state_id": parent["state_id"],
                "child_state_id": child["state_id"],
                "transition_pass": transition_pass,
                "pass_a": pass_a,
                "pass_b": pass_b,
                "parent_relation": parent_relation,
                "child_relation": child_relation,
                "flip_kind": _classify_relation_flip(
                    parent_relation,
                    child_relation,
                    parent_present=parent_present,
                    child_present=child_present,
                ),
            }
        )
    return rows


def _pair_relation_map(path: Path) -> dict[tuple[str, str], str]:
    relations: dict[tuple[str, str], str] = {}
    for row in _read_csv(path):
        pass_a = row.get("pass_a", "")
        pass_b = row.get("pass_b", "")
        if not pass_a or not pass_b:
            continue
        key = tuple(sorted([pass_a, pass_b]))
        relations[key] = row.get("final_relation", "")
    return relations


def _classify_relation_flip(
    parent_relation: str,
    child_relation: str,
    *,
    parent_present: bool,
    child_present: bool,
) -> str:
    if not parent_present and child_present:
        return "missing_to_active_pair"
    if parent_present and not child_present:
        return "active_pair_to_missing"
    if parent_relation == child_relation:
        return "same"

    parent_unknown = _is_unknown_relation(parent_relation)
    child_unknown = _is_unknown_relation(child_relation)
    if not parent_unknown and child_unknown:
        return "known_to_unknown"
    if parent_unknown and not child_unknown:
        return "unknown_to_known"
    if parent_relation == "final_commute" and child_relation == "final_order_sensitive":
        return "commute_to_sensitive"
    if parent_relation == "final_order_sensitive" and child_relation == "final_commute":
        return "sensitive_to_commute"
    return "other_flip"


def _enable_suppress_rows(
    program: str,
    parent: dict,
    child: dict,
    transition_pass: str,
    valid_passes: list[str],
) -> list[dict]:
    parent_profiles = _profile_map(Path(parent["state_dir"]) / "pass_profile.csv")
    child_profiles = _profile_map(Path(child["state_dir"]) / "pass_profile.csv")
    rows = []
    for affected_pass in valid_passes:
        parent_row = parent_profiles.get(affected_pass)
        child_row = child_profiles.get(affected_pass)
        parent_status = _profile_status(parent_row)
        child_status = _profile_status(child_row)
        rows.append(
            {
                "program": program,
                "parent_state_id": parent["state_id"],
                "child_state_id": child["state_id"],
                "transition_pass": transition_pass,
                "affected_pass": affected_pass,
                "parent_status": parent_status,
                "child_status": child_status,
                "relation": _classify_enable_suppress(parent_row, child_row),
                "parent_inst_delta": _profile_value(parent_row, "inst_delta"),
                "child_inst_delta": _profile_value(child_row, "inst_delta"),
                "parent_blocks_changed": _profile_value(parent_row, "blocks_changed"),
                "child_blocks_changed": _profile_value(child_row, "blocks_changed"),
                "parent_changed_functions": _profile_value(parent_row, "changed_functions"),
                "child_changed_functions": _profile_value(child_row, "changed_functions"),
            }
        )
    return rows


def _profile_map(path: Path) -> dict[str, dict]:
    profiles = {}
    for row in _read_csv(path):
        pass_name = row.get("pass", "")
        if pass_name:
            profiles[pass_name] = row
    return profiles


def _profile_status(row: dict | None) -> str:
    if not row or not _is_true(row.get("success")):
        return "failed_or_unknown"
    if _is_true(row.get("active")):
        return "active"
    if str(row.get("active", "")).strip():
        return "dormant"
    return "failed_or_unknown"


def _classify_enable_suppress(parent_row: dict | None, child_row: dict | None) -> str:
    parent_status = _profile_status(parent_row)
    child_status = _profile_status(child_row)
    if parent_status == "failed_or_unknown" or child_status == "failed_or_unknown":
        return "failed_or_unknown"
    if parent_status == "dormant" and child_status == "active":
        return "enable"
    if parent_status == "active" and child_status == "dormant":
        return "suppress"
    if parent_status == "active" and child_status == "active":
        if _profile_effect(parent_row) != _profile_effect(child_row):
            return "effect_changed"
        return "still_active_similar"
    return "still_dormant"


def _profile_effect(row: dict | None) -> tuple[str, str, str, str]:
    return (
        _profile_value(row, "inst_delta"),
        _profile_value(row, "blocks_changed"),
        _profile_value(row, "changed_functions"),
        _profile_value(row, "changed_blocks"),
    )


def _profile_value(row: dict | None, field: str) -> str:
    if not row:
        return ""
    return row.get(field, "")


def _is_unknown_relation(relation: str) -> bool:
    return not relation or "unknown" in relation.lower()


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


def _to_float(value: object) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0.0


def _write_multistate_summary(
    path: Path,
    out_dir: Path,
    aggregate_rows: list[dict],
) -> None:
    state_rows = _read_csv(out_dir / "states.csv")
    transition_rows = _read_csv(out_dir / "state_transitions.csv")
    relation_flip_rows = _read_csv(out_dir / "relation_flip.csv")
    enable_suppress_rows = _read_csv(out_dir / "enable_suppress.csv")
    state_summaries = _state_summary_rows(state_rows)

    root = next((row for row in state_rows if row.get("state_id") == "S0000"), state_rows[0] if state_rows else {})
    max_depth = max((_to_int(row.get("depth")) for row in state_rows), default=0)
    duplicate_states = sum(1 for row in state_rows if _is_true(row.get("is_duplicate")))
    merge_rate = duplicate_states / len(state_rows) if state_rows else 0.0
    flip_counts = Counter(row.get("flip_kind", "") for row in relation_flip_rows if row.get("flip_kind"))
    relation_counts = Counter(row.get("relation", "") for row in enable_suppress_rows if row.get("relation"))
    pair_availability_changes = _count_pair_availability_changes(relation_flip_rows)
    true_relation_flips = _count_true_relation_flips(relation_flip_rows)
    max_component = max((_to_float(row.get("max_conflict_component")) for row in state_summaries), default=0.0)
    lines = [
        "# Multi-State Summary",
        "",
        "## Overall",
        "",
        f"- root state hash: {root.get('state_hash', '')}",
        f"- states explored: {len(state_rows)}",
        f"- transitions generated: {len(transition_rows)}",
        f"- duplicate states: {duplicate_states}",
        f"- merge rate: {_format_percent(merge_rate)}",
        f"- max depth: {max_depth}",
        "",
        "## By depth table",
        "",
    ]
    lines.extend(_by_depth_table(aggregate_rows))
    lines.extend([""])
    lines.extend(equality_tier_markdown(equality_tier_summary_for_run(out_dir)))
    lines.extend([""])
    lines.extend(["", "## Enable/Suppress", "", "Enable/suppress counts", ""])
    lines.extend(_counter_table(["relation", "count"], relation_counts))
    lines.extend(["", "Top transition_pass -> affected_pass", ""])
    lines.extend(
        _counter_table(
            ["relation", "transition_pass", "affected_pass", "count"],
            _top_enable_suppress_edges(enable_suppress_rows),
        )
    )
    lines.extend(["", "## Relation Flips", "", "Pair availability changes", ""])
    lines.append(f"total pair availability changes: {pair_availability_changes}")
    lines.extend(
        _counter_table(
            ["flip_kind", "count"],
            Counter({key: flip_counts[key] for key in PAIR_AVAILABILITY_CHANGE_KINDS if flip_counts[key]}),
        )
    )
    lines.extend(["", "True relation flips among pairs active in both states", ""])
    lines.append(f"total true relation flips: {true_relation_flips}")
    lines.extend(
        _counter_table(
            ["flip_kind", "count"],
            Counter({key: flip_counts[key] for key in TRUE_RELATION_FLIP_KINDS if flip_counts[key]}),
        )
    )
    lines.extend(["", "Top examples", ""])
    lines.extend(_relation_flip_examples(relation_flip_rows))
    lines.extend(["", "## Largest Components", ""])
    lines.extend(_largest_components_table(state_summaries))
    lines.extend(["", "## Interpretation", ""])
    if true_relation_flips:
        lines.append(
            "- relation is state-dependent: yes, observed "
            f"{true_relation_flips} true relation flips among pairs active in both states "
            f"and {pair_availability_changes} pair availability changes."
        )
    elif pair_availability_changes:
        lines.append(
            "- relation is state-dependent: no true relation flips among pairs active in both states were observed; "
            f"{pair_availability_changes} pair availability changes were observed."
        )
    else:
        lines.append("- relation is state-dependent: no pair availability changes or true relation flips were observed in this run.")
    if max_component:
        lines.append(f"- conflict components remain small: largest observed component has {max_component:g} passes.")
    else:
        lines.append("- conflict components remain small: no conflict component larger than zero was observed.")
    lines.append(
        "- refine next: increase max_depth or switch frontier policy to sensitive-first/top-k-change to test whether these state-dependent edges persist."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate_by_depth(out_dir: Path, program: str) -> list[dict]:
    state_rows = _read_csv(out_dir / "states.csv")
    transition_rows = _read_csv(out_dir / "state_transitions.csv")
    enable_rows = _read_csv(out_dir / "enable_suppress.csv")
    flip_rows = _read_csv(out_dir / "relation_flip.csv")
    state_summaries = _state_summary_rows(state_rows)
    state_depths = {row.get("state_id", ""): str(row.get("depth", "")) for row in state_rows}

    depths = sorted({_to_int(row.get("depth")) for row in state_rows})
    aggregate_rows = []
    for depth in depths:
        depth_key = str(depth)
        summaries = [row for row in state_summaries if str(row.get("depth", "")) == depth_key]
        transitions = [row for row in transition_rows if str(row.get("depth", "")) == depth_key]
        enable_at_depth = [row for row in enable_rows if state_depths.get(row.get("child_state_id", "")) == depth_key]
        flips_at_depth = [row for row in flip_rows if state_depths.get(row.get("child_state_id", "")) == depth_key]
        cache_hits_from_states = sum(
            1 for row in state_rows if str(row.get("depth", "")) == depth_key and _is_true(row.get("is_duplicate"))
        )
        cache_hits_from_transitions = sum(1 for row in transitions if _is_true(row.get("is_duplicate")))

        aggregate_rows.append(
            {
                "program": program,
                "depth": depth_key,
                "num_states": str(len(summaries)),
                "avg_active_passes": _format_float(_avg(_metric_values(summaries, "active_passes"))),
                "avg_dormant_passes": _format_float(_avg(_metric_values(summaries, "dormant_passes"))),
                "avg_pairs_tested": _format_float(_avg(_metric_values(summaries, "pairs_tested"))),
                "avg_dynamic_commute": _format_float(_avg(_metric_values(summaries, "dynamic_commute"))),
                "avg_order_sensitive": _format_float(_avg(_metric_values(summaries, "order_sensitive"))),
                "avg_unknown": _format_float(_avg(_metric_values(summaries, "unknown"))),
                "avg_max_conflict_component": _format_float(_avg(_metric_values(summaries, "max_conflict_component"))),
                "state_cache_hits": str(max(cache_hits_from_states, cache_hits_from_transitions)),
                "enable_count": str(_count_relation(enable_at_depth, "enable")),
                "suppress_count": str(_count_relation(enable_at_depth, "suppress")),
                "effect_changed_count": str(_count_relation(enable_at_depth, "effect_changed")),
                "relation_flip_count": str(_count_flips(flips_at_depth)),
                "pair_availability_change_count": str(_count_pair_availability_changes(flips_at_depth)),
                "true_relation_flip_count": str(_count_true_relation_flips(flips_at_depth)),
                "commute_to_sensitive": str(_count_flip_kind(flips_at_depth, "commute_to_sensitive")),
                "sensitive_to_commute": str(_count_flip_kind(flips_at_depth, "sensitive_to_commute")),
                "missing_to_active_pair": str(_count_flip_kind(flips_at_depth, "missing_to_active_pair")),
                "active_pair_to_missing": str(_count_flip_kind(flips_at_depth, "active_pair_to_missing")),
                "total_time_ms": _format_float(sum(_metric_values(summaries, "total_time_ms"))),
            }
        )
    return aggregate_rows


def _state_summary_rows(state_rows: list[dict]) -> list[dict]:
    summaries = []
    for state in state_rows:
        summary = dict(_first_row(Path(state.get("state_dir", "")) / "per_state_summary.csv"))
        summary.update(
            {
                "program": state.get("program", summary.get("program", "")),
                "state_id": state.get("state_id", summary.get("state_id", "")),
                "state_hash": state.get("state_hash", summary.get("state_hash", "")),
                "depth": state.get("depth", summary.get("depth", "")),
                "state_dir": state.get("state_dir", ""),
                "is_duplicate": state.get("is_duplicate", ""),
                "duplicate_of": state.get("duplicate_of", ""),
            }
        )
        for field in ["active_passes", "pairs_tested", "dynamic_commute", "order_sensitive", "unknown", "max_conflict_component", "total_time_ms"]:
            if not summary.get(field):
                summary[field] = state.get(field, "")
        summaries.append(summary)
    return summaries


def _metric_values(rows: list[dict], field: str) -> list[float]:
    return [_to_float(row.get(field)) for row in rows]


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _count_relation(rows: list[dict], relation: str) -> int:
    return sum(1 for row in rows if row.get("relation") == relation)


def _count_flip_kind(rows: list[dict], flip_kind: str) -> int:
    return sum(1 for row in rows if row.get("flip_kind") == flip_kind)


def _count_flips(rows: list[dict]) -> int:
    return sum(1 for row in rows if row.get("flip_kind") and row.get("flip_kind") != "same")


def _count_pair_availability_changes(rows: list[dict]) -> int:
    return sum(1 for row in rows if row.get("flip_kind") in PAIR_AVAILABILITY_CHANGE_KINDS)


def _count_true_relation_flips(rows: list[dict]) -> int:
    return sum(1 for row in rows if row.get("flip_kind") in TRUE_RELATION_FLIP_KINDS)


def _by_depth_table(aggregate_rows: list[dict]) -> list[str]:
    rows = [
        [
            row.get("depth", ""),
            row.get("num_states", ""),
            row.get("avg_active_passes", ""),
            row.get("avg_pairs_tested", ""),
            row.get("avg_dynamic_commute", ""),
            row.get("avg_order_sensitive", ""),
            row.get("avg_max_conflict_component", ""),
        ]
        for row in aggregate_rows
    ]
    return _markdown_table(
        ["depth", "states", "avg active", "avg pairs", "avg commute", "avg sensitive", "avg max component"],
        rows,
    )


def _relation_flip_examples(rows: list[dict]) -> list[str]:
    selected = [row for row in rows if row.get("flip_kind") != "same"]
    if not selected:
        selected = rows
    table_rows = [
        [
            row.get("child_state_id", ""),
            row.get("transition_pass", ""),
            f"{row.get('pass_a', '')} + {row.get('pass_b', '')}",
            row.get("parent_relation", ""),
            row.get("child_relation", ""),
            row.get("flip_kind", ""),
        ]
        for row in selected[:10]
    ]
    return _markdown_table(["child", "transition", "pair", "parent", "child relation", "flip_kind"], table_rows)


def _largest_components_table(rows: list[dict]) -> list[str]:
    ordered = sorted(rows, key=lambda row: _to_float(row.get("max_conflict_component")), reverse=True)[:10]
    table_rows = [
        [
            row.get("state_id", ""),
            row.get("depth", ""),
            _format_float(_to_float(row.get("max_conflict_component"))),
            _format_float(_to_float(row.get("active_passes"))),
        ]
        for row in ordered
    ]
    return _markdown_table(["state_id", "depth", "max component", "active passes"], table_rows)


def _top_enable_suppress_edges(rows: list[dict]) -> Counter:
    return Counter(
        (row.get("relation", ""), row.get("transition_pass", ""), row.get("affected_pass", ""))
        for row in rows
        if row.get("relation") in {"enable", "suppress", "effect_changed"}
    )


def _counter_table(headers: list[str], counter: Counter) -> list[str]:
    if not counter:
        return ["none"]

    lines = [f"| {' | '.join(headers)} |", f"| {' | '.join(['---'] * len(headers))} |"]
    for key, count in counter.most_common(10):
        cells = list(key) if isinstance(key, tuple) else [key]
        cells.append(str(count))
        lines.append(f"| {' | '.join(cells)} |")
    return lines


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["none"]
    lines = [f"| {' | '.join(headers)} |", f"| {' | '.join(['---'] * len(headers))} |"]
    lines.extend(f"| {' | '.join(row)} |" for row in rows)
    return lines


def _format_float(value: float) -> str:
    return f"{value:.2f}"


def _format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"
