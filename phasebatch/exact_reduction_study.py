from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path


EXACT_REDUCTION_RUN_FIELDS = [
    "program",
    "run_dir",
    "status",
    "exact_status",
    "max_rounds",
    "states_reached",
    "transitions",
    "selected_final_state",
    "final_ir_inst_count",
    "pipeline_length",
    "error_message",
]

REDUCTION_BY_STATE_ALL_FIELDS = [
    "program",
    "state_id",
    "depth",
    "state_hash",
    "selected_on_final_path",
    "active_passes",
    "tested_pairs",
    "commute_pairs",
    "order_sensitive_pairs",
    "unknown_pairs",
    "naive_orderings_log10",
    "batch_candidates",
    "certified_batches",
    "executable_batches",
    "sampled_batches",
    "rejected_batches",
    "failed_batches",
    "unvalidated_batches",
    "skipped_batches",
    "dropped_active_passes",
    "local_reduction_log10",
    "local_reduction_ratio_capped",
    "no_executable_batches",
    "terminal_due_max_rounds",
]

REDUCTION_BY_PROGRAM_FIELDS = [
    "program",
    "states",
    "max_depth",
    "total_active_passes",
    "total_tested_pairs",
    "total_commute_pairs",
    "total_order_sensitive_pairs",
    "total_unknown_pairs",
    "total_batch_candidates",
    "total_certified_batches",
    "total_executable_batches",
    "total_sampled_batches",
    "total_rejected_batches",
    "total_failed_batches",
    "total_unvalidated_batches",
    "total_skipped_batches",
    "total_dropped_active_passes",
    "avg_active_passes",
    "avg_executable_batches",
    "avg_local_reduction_log10",
    "max_local_reduction_log10",
    "selected_path_steps",
    "final_pipeline_length",
    "final_ir_inst_count",
]

EVIDENCE_BY_BATCH_ALL_FIELDS = [
    "program",
    "parent_state_id",
    "child_state_id",
    "batch_id",
    "batch_passes",
    "canonical_order",
    "validation_status",
    "correctness_class",
    "can_hard_fold",
    "can_execute",
    "evidence_strength",
    "tested_orders",
    "same_hash_count",
    "different_hash_count",
    "is_duplicate_transition",
    "duplicate_of",
]

SELECTED_PATH_EVIDENCE_FIELDS = [
    "program",
    "step",
    "parent_state_id",
    "batch_id",
    "batch_passes",
    "canonical_order",
    "child_state_id",
    "validation_status",
    "correctness_class",
    "evidence_strength",
    "ir_inst_before",
    "ir_inst_after",
    "ir_inst_delta",
]

COVERAGE_BY_PROGRAM_FIELDS = [
    "program",
    "total_active_passes",
    "certified_covered",
    "heuristic_covered",
    "unresolved",
    "rejected",
    "unknown",
    "terminal_due_max_rounds",
    "dropped_active_passes",
]


def summarize_exact_reduction_study(
    run_dirs: list[Path],
    out_dir: Path,
    *,
    label: str,
    root_dir: Path | None = None,
    summarize_components: bool = False,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    discovered = _discover_run_dirs(root_dir) if root_dir else []
    plans = _dedupe_paths([Path(path) for path in run_dirs] + discovered)
    if not plans:
        raise RuntimeError("no optimize-batches run directories were provided or discovered")

    run_rows: list[dict] = []
    state_rows: list[dict] = []
    program_rows: list[dict] = []
    evidence_rows: list[dict] = []
    selected_rows: list[dict] = []
    coverage_rows: list[dict] = []

    for run_dir in plans:
        try:
            result = _summarize_run(run_dir)
            run_rows.append(result["run_row"])
            state_rows.extend(result["state_rows"])
            program_rows.append(result["program_row"])
            evidence_rows.extend(result["evidence_rows"])
            selected_rows.extend(result["selected_rows"])
            coverage_rows.append(result["coverage_row"])
        except Exception as exc:
            run_rows.append(
                {
                    "program": _program_name(run_dir),
                    "run_dir": str(run_dir),
                    "status": "failed",
                    "exact_status": "",
                    "max_rounds": "",
                    "states_reached": "",
                    "transitions": "",
                    "selected_final_state": "",
                    "final_ir_inst_count": "",
                    "pipeline_length": "",
                    "error_message": str(exc),
                }
            )

    _write_csv(out_dir / "exact_reduction_runs.csv", EXACT_REDUCTION_RUN_FIELDS, run_rows)
    _write_csv(out_dir / "reduction_by_state_all.csv", REDUCTION_BY_STATE_ALL_FIELDS, state_rows)
    _write_csv(out_dir / "reduction_by_program.csv", REDUCTION_BY_PROGRAM_FIELDS, program_rows)
    _write_csv(out_dir / "evidence_by_batch_all.csv", EVIDENCE_BY_BATCH_ALL_FIELDS, evidence_rows)
    _write_csv(out_dir / "selected_path_evidence.csv", SELECTED_PATH_EVIDENCE_FIELDS, selected_rows)
    _write_csv(out_dir / "coverage_by_program.csv", COVERAGE_BY_PROGRAM_FIELDS, coverage_rows)
    summary_path = _write_summary(out_dir / "exact_reduction_summary.md", label, run_rows, program_rows, evidence_rows, selected_rows, coverage_rows)
    component_result = _try_component_summary([path for path in plans if (path / "states.csv").exists()], out_dir / "components") if summarize_components else {}

    successes = sum(1 for row in run_rows if row.get("status") == "success")
    result = {
        "out_dir": str(out_dir),
        "programs": len(run_rows),
        "successes": successes,
        "failures": len(run_rows) - successes,
        "exact_reduction_runs_csv": str(out_dir / "exact_reduction_runs.csv"),
        "reduction_by_state_all_csv": str(out_dir / "reduction_by_state_all.csv"),
        "reduction_by_program_csv": str(out_dir / "reduction_by_program.csv"),
        "evidence_by_batch_all_csv": str(out_dir / "evidence_by_batch_all.csv"),
        "selected_path_evidence_csv": str(out_dir / "selected_path_evidence.csv"),
        "coverage_by_program_csv": str(out_dir / "coverage_by_program.csv"),
        "exact_reduction_summary_md": str(summary_path),
    }
    if component_result:
        result["component_summary_md"] = component_result.get("component_summary_md", "")
    return result


def run_component_summary(run_dirs: list[Path], out_dir: Path) -> dict:
    from .component_summary import summarize_components

    return summarize_components(run_dirs=run_dirs, out_dir=out_dir)


def _try_component_summary(run_dirs: list[Path], out_dir: Path) -> dict:
    if not run_dirs:
        return {}
    try:
        return run_component_summary(run_dirs, out_dir)
    except Exception:
        return {}


def _summarize_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    states = _read_csv(run_dir / "states.csv")
    if not states:
        raise RuntimeError(f"missing or empty states.csv in {run_dir}")
    program = _program_name(run_dir, states)
    state_dirs = _state_dirs(run_dir, states)
    selected_state_ids = _selected_state_ids(run_dir)
    leaf_reasons = {row.get("state_id", ""): row.get("leaf_reason", "") for row in _read_csv(run_dir / "leaf_states.csv")}

    state_rows = [
        _state_row(program, run_dir, state, state_dirs.get(state.get("state_id", ""), run_dir / "states" / state.get("state_id", "")), selected_state_ids, leaf_reasons)
        for state in states
    ]
    selected_rows = [_selected_path_row(program, row, state_dirs) for row in _read_csv(run_dir / "chosen_path.csv")]
    evidence_rows = [_executed_batch_row(program, row, state_dirs) for row in _executed_transition_rows(run_dir)]
    coverage_row = _coverage_row(program, state_dirs.values(), leaf_reasons)
    program_row = _program_row(program, run_dir, state_rows, selected_rows)
    run_row = _run_row(program, run_dir, states, program_row)
    return {
        "run_row": run_row,
        "state_rows": state_rows,
        "program_row": program_row,
        "evidence_rows": evidence_rows,
        "selected_rows": selected_rows,
        "coverage_row": coverage_row,
    }


def _run_row(program: str, run_dir: Path, states: list[dict], program_row: dict) -> dict:
    chosen = _first_row(run_dir / "chosen_path_summary.csv")
    return {
        "program": program,
        "run_dir": str(run_dir),
        "status": "success",
        "exact_status": _read_text(run_dir / "exact_status.txt").strip(),
        "max_rounds": str(_metadata(run_dir).get("max_rounds", "")),
        "states_reached": str(len(states)),
        "transitions": str(len(_executed_transition_rows(run_dir))),
        "selected_final_state": chosen.get("selected_final_state") or _selected_final_state(run_dir),
        "final_ir_inst_count": _final_ir_inst_count(run_dir, chosen) or program_row.get("final_ir_inst_count", ""),
        "pipeline_length": program_row.get("final_pipeline_length", ""),
        "error_message": "",
    }


def _state_row(
    program: str,
    run_dir: Path,
    state: dict,
    state_dir: Path,
    selected_state_ids: set[str],
    leaf_reasons: dict[str, str],
) -> dict:
    state_id = state.get("state_id", "")
    per_state = _first_row(state_dir / "per_state_summary.csv")
    active = _active_passes(state_dir, per_state)
    relation = _relation_counts(state_dir, per_state)
    candidate_count = len(_read_csv(state_dir / "batch_candidates.csv"))
    correctness = _correctness_counts(state_dir, candidate_count)
    executable = correctness["executable_batches"]
    naive_log10 = _factorial_log10(active)
    local_log10 = 0.0 if active <= 1 else naive_log10 - math.log10(max(1, executable))
    no_executable = active > 1 and executable == 0
    return {
        "program": program,
        "state_id": state_id,
        "depth": state.get("depth") or per_state.get("depth", ""),
        "state_hash": state.get("state_hash") or per_state.get("state_hash", ""),
        "selected_on_final_path": _bool(state_id in selected_state_ids),
        "active_passes": str(active),
        "tested_pairs": str(relation["tested_pairs"]),
        "commute_pairs": str(relation["commute_pairs"]),
        "order_sensitive_pairs": str(relation["order_sensitive_pairs"]),
        "unknown_pairs": str(relation["unknown_pairs"]),
        "naive_orderings_log10": _fmt_float(naive_log10),
        "batch_candidates": str(candidate_count),
        "certified_batches": str(correctness["certified_batches"]),
        "executable_batches": str(executable),
        "sampled_batches": str(correctness["sampled_batches"]),
        "rejected_batches": str(correctness["rejected_batches"]),
        "failed_batches": str(correctness["failed_batches"]),
        "unvalidated_batches": str(correctness["unvalidated_batches"]),
        "skipped_batches": str(correctness["skipped_batches"]),
        "dropped_active_passes": str(_dropped_active_passes(state_dir)),
        "local_reduction_log10": _fmt_float(local_log10),
        "local_reduction_ratio_capped": _ratio_capped(active, executable),
        "no_executable_batches": _bool(no_executable),
        "terminal_due_max_rounds": _bool(leaf_reasons.get(state_id) == "max_rounds_reached"),
    }


def _program_row(program: str, run_dir: Path, rows: list[dict], selected_rows: list[dict]) -> dict:
    chosen = _first_row(run_dir / "chosen_path_summary.csv")
    return {
        "program": program,
        "states": str(len(rows)),
        "max_depth": str(max((_int(row.get("depth")) for row in rows), default=0)),
        "total_active_passes": str(_sum(rows, "active_passes")),
        "total_tested_pairs": str(_sum(rows, "tested_pairs")),
        "total_commute_pairs": str(_sum(rows, "commute_pairs")),
        "total_order_sensitive_pairs": str(_sum(rows, "order_sensitive_pairs")),
        "total_unknown_pairs": str(_sum(rows, "unknown_pairs")),
        "total_batch_candidates": str(_sum(rows, "batch_candidates")),
        "total_certified_batches": str(_sum(rows, "certified_batches")),
        "total_executable_batches": str(_sum(rows, "executable_batches")),
        "total_sampled_batches": str(_sum(rows, "sampled_batches")),
        "total_rejected_batches": str(_sum(rows, "rejected_batches")),
        "total_failed_batches": str(_sum(rows, "failed_batches")),
        "total_unvalidated_batches": str(_sum(rows, "unvalidated_batches")),
        "total_skipped_batches": str(_sum(rows, "skipped_batches")),
        "total_dropped_active_passes": str(_sum(rows, "dropped_active_passes")),
        "avg_active_passes": _avg(rows, "active_passes"),
        "avg_executable_batches": _avg(rows, "executable_batches"),
        "avg_local_reduction_log10": _avg(rows, "local_reduction_log10"),
        "max_local_reduction_log10": _fmt_float(max((_float(row.get("local_reduction_log10")) for row in rows), default=0.0)),
        "selected_path_steps": str(len(selected_rows)),
        "final_pipeline_length": str(len(_split_pipeline(_read_text(run_dir / "optimized_pipeline.txt")))),
        "final_ir_inst_count": _final_ir_inst_count(run_dir, chosen),
    }


def _executed_batch_row(program: str, row: dict, state_dirs: dict[str, Path]) -> dict:
    parent = row.get("source_state_id") or row.get("parent_state_id", "")
    child = row.get("target_state_id") or row.get("child_state_id", "")
    batch_id = row.get("batch_id", "")
    evidence = _batch_evidence(state_dirs.get(parent), batch_id)
    validation_status = evidence["validation"].get("validation_status") or row.get("validation_status", "")
    correctness_class = evidence["correctness"].get("correctness_class", "")
    strength = _evidence_strength(validation_status, correctness_class)
    return {
        "program": program,
        "parent_state_id": parent,
        "child_state_id": child,
        "batch_id": batch_id,
        "batch_passes": row.get("batch_passes") or evidence["candidate"].get("batch_passes", ""),
        "canonical_order": row.get("canonical_order") or evidence["validation"].get("canonical_order") or evidence["candidate"].get("canonical_order", ""),
        "validation_status": validation_status,
        "correctness_class": correctness_class,
        "can_hard_fold": evidence["correctness"].get("can_hard_fold", ""),
        "can_execute": evidence["correctness"].get("can_execute", ""),
        "evidence_strength": strength,
        "tested_orders": evidence["validation"].get("tested_orders", ""),
        "same_hash_count": evidence["validation"].get("same_hash_count", ""),
        "different_hash_count": evidence["validation"].get("different_hash_count", ""),
        "is_duplicate_transition": row.get("is_duplicate_transition") or row.get("is_duplicate", ""),
        "duplicate_of": row.get("duplicate_of", ""),
    }


def _selected_path_row(program: str, row: dict, state_dirs: dict[str, Path]) -> dict:
    parent = row.get("parent_state_id", "")
    batch_id = row.get("batch_id", "")
    evidence = _batch_evidence(state_dirs.get(parent), batch_id)
    validation_status = evidence["validation"].get("validation_status") or row.get("validation_status", "")
    correctness_class = evidence["correctness"].get("correctness_class", "")
    return {
        "program": program,
        "step": row.get("step", ""),
        "parent_state_id": parent,
        "batch_id": batch_id,
        "batch_passes": row.get("batch_passes") or evidence["candidate"].get("batch_passes", ""),
        "canonical_order": row.get("canonical_order") or evidence["validation"].get("canonical_order") or evidence["candidate"].get("canonical_order", ""),
        "child_state_id": row.get("child_state_id", ""),
        "validation_status": validation_status,
        "correctness_class": correctness_class,
        "evidence_strength": _evidence_strength(validation_status, correctness_class),
        "ir_inst_before": row.get("ir_inst_before", ""),
        "ir_inst_after": row.get("ir_inst_after", ""),
        "ir_inst_delta": row.get("ir_inst_delta", ""),
    }


def _coverage_row(program: str, state_dirs, leaf_reasons: dict[str, str]) -> dict:
    totals = Counter()
    for state_dir in state_dirs:
        summary = _first_row(state_dir / "coverage_summary.csv")
        if summary:
            totals["total_active_passes"] += _int(summary.get("active_passes"))
            totals["certified_covered"] += _int(summary.get("certified_covered"))
            totals["heuristic_covered"] += _int(summary.get("heuristic_covered"))
            totals["unresolved"] += _int(summary.get("unresolved_conflict"))
            totals["rejected"] += _int(summary.get("validation_rejected"))
            totals["unknown"] += _int(summary.get("unvalidated_covered")) + _int(summary.get("failed_or_unknown"))
            totals["terminal_due_max_rounds"] += _int(summary.get("not_executed_due_to_max_depth"))
            totals["dropped_active_passes"] += _int(summary.get("dropped_active_passes"))
            continue
        for row in _read_csv(state_dir / "coverage_report.csv"):
            totals["total_active_passes"] += 1
            status = row.get("coverage_status", "")
            if status == "certified_covered":
                totals["certified_covered"] += 1
            elif status == "heuristic_covered":
                totals["heuristic_covered"] += 1
            elif status == "unresolved_conflict":
                totals["unresolved"] += 1
            elif status == "validation_rejected":
                totals["rejected"] += 1
            elif status == "dropped":
                totals["dropped_active_passes"] += 1
            else:
                totals["unknown"] += 1
    totals["terminal_due_max_rounds"] += sum(1 for reason in leaf_reasons.values() if reason == "max_rounds_reached")
    return {
        "program": program,
        "total_active_passes": str(totals["total_active_passes"]),
        "certified_covered": str(totals["certified_covered"]),
        "heuristic_covered": str(totals["heuristic_covered"]),
        "unresolved": str(totals["unresolved"]),
        "rejected": str(totals["rejected"]),
        "unknown": str(totals["unknown"]),
        "terminal_due_max_rounds": str(totals["terminal_due_max_rounds"]),
        "dropped_active_passes": str(totals["dropped_active_passes"]),
    }


def _write_summary(
    path: Path,
    label: str,
    run_rows: list[dict],
    program_rows: list[dict],
    evidence_rows: list[dict],
    selected_rows: list[dict],
    coverage_rows: list[dict],
) -> Path:
    successful = [row for row in run_rows if row.get("status") == "success"]
    exact_complete = sum(1 for row in successful if row.get("exact_status") == "exact_complete")
    selected_counts = Counter(row.get("program", "") for row in selected_rows)
    selected_strengths = {
        program: Counter(row.get("evidence_strength", "") for row in selected_rows if row.get("program") == program)
        for program in selected_counts
    }
    evidence_by_program = {
        row["program"]: Counter(e.get("evidence_strength", "") for e in evidence_rows if e.get("program") == row["program"])
        for row in program_rows
    }
    total_transitions = sum(_int(row.get("transitions")) for row in successful)
    lines = [
        "# Exact r4 Reduction Evidence Summary",
        "",
        "## Overall",
        "",
        f"- study label: {label}",
        f"- programs: {len(run_rows)}",
        f"- successful programs: {len(successful)}",
        f"- exact complete programs: {exact_complete}",
        f"- total states: {sum(_int(row.get('states')) for row in program_rows)}",
        f"- total transitions: {total_transitions}",
        f"- total selected path steps: {len(selected_rows)}",
        "",
        "## Candidate Space Reduction",
        "",
        *_markdown_table(
            ["program", "states", "avg active passes", "total batch candidates", "total executable batches", "avg reduction log10", "max reduction log10"],
            [
                [
                    row.get("program", ""),
                    row.get("states", ""),
                    row.get("avg_active_passes", ""),
                    row.get("total_batch_candidates", ""),
                    row.get("total_executable_batches", ""),
                    row.get("avg_local_reduction_log10", ""),
                    row.get("max_local_reduction_log10", ""),
                ]
                for row in program_rows
            ],
        ),
        "",
        "## Pair Relation Evidence",
        "",
        *_markdown_table(
            ["program", "tested pairs", "commute", "order-sensitive", "unknown", "commute %"],
            [
                [
                    row.get("program", ""),
                    row.get("total_tested_pairs", ""),
                    row.get("total_commute_pairs", ""),
                    row.get("total_order_sensitive_pairs", ""),
                    row.get("total_unknown_pairs", ""),
                    _pct(_int(row.get("total_commute_pairs")), _int(row.get("total_tested_pairs"))),
                ]
                for row in program_rows
            ],
        ),
        "",
        "## Batch Correctness Evidence",
        "",
        *_markdown_table(
            ["program", "executed batches", "strong certs", "weak certs", "rejected", "failed", "unknown"],
            [
                [
                    row.get("program", ""),
                    str(sum(evidence_by_program.get(row.get("program", ""), Counter()).values())),
                    str(evidence_by_program.get(row.get("program", ""), Counter()).get("strong", 0)),
                    str(evidence_by_program.get(row.get("program", ""), Counter()).get("weak", 0)),
                    str(evidence_by_program.get(row.get("program", ""), Counter()).get("rejected", 0)),
                    str(evidence_by_program.get(row.get("program", ""), Counter()).get("failed", 0)),
                    str(evidence_by_program.get(row.get("program", ""), Counter()).get("unknown", 0)),
                ]
                for row in program_rows
            ],
        ),
        "",
        "## Coverage",
        "",
        *_markdown_table(
            ["program", "active passes", "certified covered", "heuristic covered", "unresolved", "unknown", "dropped"],
            [
                [
                    row.get("program", ""),
                    row.get("total_active_passes", ""),
                    row.get("certified_covered", ""),
                    row.get("heuristic_covered", ""),
                    row.get("unresolved", ""),
                    row.get("unknown", ""),
                    _warn_if_dropped(row.get("dropped_active_passes", "0")),
                ]
                for row in coverage_rows
            ],
        ),
        "",
        "## Selected Path Evidence",
        "",
        *_markdown_table(
            ["program", "selected path steps", "all strong?", "final pipeline length", "final IR inst count"],
            [
                [
                    row.get("program", ""),
                    str(selected_counts.get(row.get("program", ""), 0)),
                    _bool(selected_counts.get(row.get("program", ""), 0) > 0 and selected_strengths.get(row.get("program", ""), Counter()).get("strong", 0) == selected_counts.get(row.get("program", ""), 0)),
                    row.get("final_pipeline_length", ""),
                    row.get("final_ir_inst_count", ""),
                ]
                for row in program_rows
            ],
        ),
        "",
        "## Key Observations",
        "",
        *_observation_lines(program_rows, selected_rows, coverage_rows),
        "",
        "## Correctness Boundary",
        "",
        "Search-space reduction is state-local. It applies to reached states under the current pass set, compiler version, target, and IR normalization. Objective values are not used as commutation proof.",
        "",
    ]
    failed = [row for row in run_rows if row.get("status") == "failed"]
    if failed:
        lines.extend(
            [
                "## Failures",
                "",
                *_markdown_table(
                    ["program", "run dir", "error"],
                    [[row.get("program", ""), row.get("run_dir", ""), row.get("error_message", "")] for row in failed],
                ),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _observation_lines(program_rows: list[dict], selected_rows: list[dict], coverage_rows: list[dict]) -> list[str]:
    if not program_rows:
        return ["- No successful runs were available to summarize."]
    selected = Counter(row.get("evidence_strength", "") for row in selected_rows)
    dropped = sum(_int(row.get("dropped_active_passes")) for row in coverage_rows)
    max_reduction = max((_float(row.get("max_local_reduction_log10")) for row in program_rows), default=0.0)
    sensitive = sum(_int(row.get("total_order_sensitive_pairs")) for row in program_rows)
    tested = sum(_int(row.get("total_tested_pairs")) for row in program_rows)
    lines = []
    if selected and selected.get("strong", 0) == sum(selected.values()):
        lines.append("- All selected path batches have strong certificates in this study.")
    else:
        lines.append("- Some selected path batches lack strong certificates; inspect selected_path_evidence.csv before making hard-folding claims.")
    if dropped == 0:
        lines.append("- Dropped active passes are zero in the summarized runs.")
    else:
        lines.append(f"- **WARNING** dropped active passes are nonzero: {dropped}.")
    lines.append(f"- The largest observed local reduction is about 10^{_fmt_float(max_reduction)} in this run.")
    lines.append(f"- Order-sensitive pairs remain part of the evidence set: {sensitive} of {tested} tested pairs.")
    return lines


def _active_passes(state_dir: Path, per_state: dict) -> int:
    if per_state.get("active_passes") not in {None, ""}:
        return _int(per_state.get("active_passes"))
    return sum(1 for row in _read_csv(state_dir / "pass_profile.csv") if _is_true(row.get("success")) and _is_true(row.get("active")))


def _relation_counts(state_dir: Path, per_state: dict) -> dict[str, int]:
    if per_state:
        return {
            "tested_pairs": _int(per_state.get("pairs_tested") or per_state.get("total_pairs")),
            "commute_pairs": _int(per_state.get("dynamic_commute")),
            "order_sensitive_pairs": _int(per_state.get("order_sensitive")),
            "unknown_pairs": _int(per_state.get("unknown")),
        }
    counts = Counter(row.get("final_relation", "") for row in _read_csv(state_dir / "pair_relation.csv"))
    return {
        "tested_pairs": sum(counts.values()),
        "commute_pairs": counts.get("final_commute", 0),
        "order_sensitive_pairs": counts.get("final_order_sensitive", 0),
        "unknown_pairs": counts.get("final_unknown", 0) + counts.get("unknown", 0),
    }


def _correctness_counts(state_dir: Path, batch_candidates: int) -> dict[str, int]:
    path = state_dir / "batch_correctness.csv"
    rows = _read_csv(path)
    if batch_candidates and not path.exists():
        return {
            "certified_batches": 0,
            "executable_batches": 0,
            "sampled_batches": 0,
            "rejected_batches": 0,
            "failed_batches": 0,
            "unvalidated_batches": batch_candidates,
            "skipped_batches": batch_candidates,
        }
    classes = Counter(row.get("correctness_class", "") for row in rows)
    return {
        "certified_batches": classes.get("certified_batch", 0),
        "executable_batches": sum(1 for row in rows if _is_true(row.get("can_execute"))),
        "sampled_batches": classes.get("sampled_batch", 0),
        "rejected_batches": classes.get("rejected_batch", 0),
        "failed_batches": classes.get("failed_batch", 0),
        "unvalidated_batches": classes.get("unvalidated_batch", 0) + classes.get("unknown_batch", 0),
        "skipped_batches": sum(1 for row in rows if not _is_true(row.get("can_execute"))),
    }


def _batch_evidence(state_dir: Path | None, batch_id: str) -> dict[str, dict]:
    if state_dir is None:
        return {"validation": {}, "correctness": {}, "candidate": {}}
    return {
        "validation": _row_by_batch(state_dir / "batch_validation.csv", batch_id),
        "correctness": _row_by_batch(state_dir / "batch_correctness.csv", batch_id),
        "candidate": _row_by_batch(state_dir / "batch_candidates.csv", batch_id),
    }


def _row_by_batch(path: Path, batch_id: str) -> dict:
    for row in _read_csv(path):
        if row.get("batch_id", "") == batch_id:
            return row
    return {}


def _evidence_strength(validation_status: str, correctness_class: str) -> str:
    if validation_status == "all_permutations_same" and correctness_class == "certified_batch":
        return "strong"
    if validation_status == "sampled_same":
        return "weak"
    if validation_status == "mismatch":
        return "rejected"
    if validation_status == "failed":
        return "failed"
    return "unknown"


def _dropped_active_passes(state_dir: Path) -> int:
    summary = _first_row(state_dir / "coverage_summary.csv")
    if summary:
        return _int(summary.get("dropped_active_passes"))
    return sum(1 for row in _read_csv(state_dir / "coverage_report.csv") if row.get("coverage_status") == "dropped")


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


def _selected_state_ids(run_dir: Path) -> set[str]:
    states: set[str] = set()
    for row in _read_csv(run_dir / "chosen_path.csv"):
        if row.get("parent_state_id"):
            states.add(row["parent_state_id"])
        if row.get("child_state_id"):
            states.add(row["child_state_id"])
    selected = _selected_final_state(run_dir)
    if selected:
        states.add(selected)
    return states


def _selected_final_state(run_dir: Path) -> str:
    for row in _read_csv(run_dir / "leaf_states.csv"):
        if row.get("selected_as_final") == "true":
            return row.get("state_id", "")
    return _first_row(run_dir / "chosen_path_summary.csv").get("selected_final_state", "")


def _final_ir_inst_count(run_dir: Path, chosen_summary: dict | None = None) -> str:
    chosen_summary = chosen_summary or _first_row(run_dir / "chosen_path_summary.csv")
    if chosen_summary.get("final_ir_inst_count"):
        return chosen_summary["final_ir_inst_count"]
    if chosen_summary.get("final_objective"):
        return chosen_summary["final_objective"]
    chosen_path = _read_csv(run_dir / "chosen_path.csv")
    if chosen_path:
        return chosen_path[-1].get("ir_inst_after", "")
    return ""


def _executed_transition_rows(run_dir: Path) -> list[dict]:
    rows = _read_csv(run_dir / "state_dag.csv")
    if rows:
        return rows
    return _read_csv(run_dir / "batch_state_transitions.csv")


def _program_name(run_dir: Path, states: list[dict] | None = None) -> str:
    metadata = _metadata(run_dir)
    if metadata.get("input"):
        return Path(str(metadata["input"])).stem
    if run_dir.name == "optimize" and run_dir.parent.name:
        return run_dir.parent.name
    rows = states if states is not None else _read_csv(run_dir / "states.csv")
    for row in rows:
        if row.get("program") and row.get("program") != "optimize":
            return row["program"]
    return run_dir.name


def _discover_run_dirs(root_dir: Path | None) -> list[Path]:
    if root_dir is None or not Path(root_dir).exists():
        return []
    candidates = []
    for states_csv in Path(root_dir).rglob("states.csv"):
        run_dir = states_csv.parent
        if (run_dir / "optimized_pipeline.txt").exists() or (run_dir / "chosen_path.csv").exists():
            candidates.append(run_dir)
    return sorted(candidates)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique = []
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _metadata(run_dir: Path) -> dict:
    path = run_dir / "metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _factorial_log10(value: int) -> float:
    if value <= 1:
        return 0.0
    return math.lgamma(value + 1) / math.log(10)


def _ratio_capped(active_passes: int, executable_batches: int) -> str:
    if active_passes <= 1:
        return "1"
    if active_passes > 20:
        return "too_large"
    naive = math.factorial(active_passes)
    denominator = max(1, executable_batches)
    if naive % denominator == 0:
        return str(naive // denominator)
    return _fmt_float(naive / denominator)


def _split_pipeline(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").replace("\n", "").replace(";", ",").split(",") if part.strip()]


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _first_row(path: Path) -> dict:
    rows = _read_csv(path)
    return rows[0] if rows else {}


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _warn_if_dropped(value: object) -> str:
    count = _int(value)
    return f"**WARNING** {count}" if count > 0 else str(count)


def _pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.00%"
    return f"{(numerator / denominator) * 100:.2f}%"


def _avg(rows: list[dict], key: str) -> str:
    if not rows:
        return "0"
    return _fmt_float(sum(_float(row.get(key)) for row in rows) / len(rows))


def _sum(rows: list[dict], key: str) -> int:
    return sum(_int(row.get(key)) for row in rows)


def _fmt_float(value: float) -> str:
    if abs(value) < 0.0000005:
        return "0"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _int(value: object) -> int:
    try:
        return int(float(str(value or "0")))
    except ValueError:
        return 0


def _float(value: object) -> float:
    try:
        return float(str(value or "0"))
    except ValueError:
        return 0.0


def _is_true(value: object) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def _bool(value: bool) -> str:
    return "true" if value else "false"
