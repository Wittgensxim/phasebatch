from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


TRUE_RELATION_FLIPS = {
    "commute_to_sensitive",
    "sensitive_to_commute",
    "known_to_unknown",
    "unknown_to_known",
    "other_flip",
}

COVERAGE_STATUSES = [
    "certified_covered",
    "heuristic_covered",
    "unresolved_conflict",
    "validation_rejected",
    "unvalidated_covered",
    "failed_or_unknown",
    "not_executed_due_to_max_depth",
    "dropped",
]


def select_representative_state(program_dir: Path) -> str:
    program_dir = Path(program_dir)
    states = _read_csv(program_dir / "states.csv")
    if not states:
        return "S0000"

    flip_counts = _true_relation_flip_counts(program_dir / "relation_flip.csv")
    candidates = []
    for index, state in enumerate(states):
        state_id = state.get("state_id", "")
        if not state_id:
            continue
        state_dir = _state_dir(program_dir, state)
        batch_summary = _first_row(state_dir / "batch_summary.csv")
        if not batch_summary:
            continue
        candidates.append(
            (
                _to_float(batch_summary.get("batch_reduction_estimate")),
                _to_int(_first_value(batch_summary, ["active_passes"], default=state.get("active_passes", "0"))),
                flip_counts.get(state_id, 0),
                -index,
                state_id,
            )
        )

    if not candidates:
        return "S0000"
    return max(candidates)[4]


def export_case_studies(run_dir: Path, max_pairs: int = 20, max_batches: int = 10) -> dict:
    run_dir = Path(run_dir)
    runs = _read_csv(run_dir / "mainline_runs.csv")
    index_rows = []
    case_paths = []

    for run in runs:
        if run.get("status") != "success":
            continue
        program = run.get("program", "") or Path(run.get("output_dir", "")).name
        program_dir = _resolve_program_dir(run_dir, run.get("output_dir", ""), program)
        case_path, index_row = generate_case_study(program_dir, program=program, max_pairs=max_pairs, max_batches=max_batches)
        case_paths.append(case_path)
        index_rows.append(index_row)

    index_path = _write_case_studies_index(run_dir, index_rows)
    return {
        "run_dir": str(run_dir),
        "case_studies": len(case_paths),
        "case_studies_index": str(index_path),
        "case_study_paths": [str(path) for path in case_paths],
    }


def generate_case_study(program_dir: Path, *, program: str | None = None, max_pairs: int = 20, max_batches: int = 10) -> tuple[Path, dict]:
    program_dir = Path(program_dir)
    program = program or program_dir.name
    selected_state_id = select_representative_state(program_dir)
    state_row = _state_row(program_dir, selected_state_id)
    state_dir = _state_dir(program_dir, state_row)
    per_state = _first_row(state_dir / "per_state_summary.csv")
    batch_summary = _first_row(state_dir / "batch_summary.csv")
    selected = {**state_row, **per_state, **{k: v for k, v in batch_summary.items() if v not in (None, "")}}
    selected.setdefault("state_id", selected_state_id)

    coverage_counts = _coverage_counts(state_dir / "coverage_report.csv")
    dropped = coverage_counts.get("dropped", 0)
    batch_candidates = _read_csv(state_dir / "batch_candidates.csv")
    active_pass_count = _first_value(selected, ["active_passes"], default=str(_active_pass_count(state_dir / "pass_profile.csv")))
    batch_candidate_count = _first_value(batch_summary, ["batch_candidates"], default=str(len(batch_candidates)))
    reduction = _first_value(batch_summary, ["batch_reduction_estimate"], default="")

    lines = [
        f"# Case Study: {program}",
        "",
        "## Selected State",
        "",
        *[f"- {key}: {_display(selected.get(key, ''))}" for key in [
            "state_id",
            "depth",
            "state_hash",
            "parent_state_id",
            "transition_pass",
            "active_passes",
            "pairs_tested",
            "dynamic_commute",
            "order_sensitive",
            "unknown",
            "max_conflict_component",
        ]],
        "",
    ]
    if selected.get("transition_pass"):
        lines.append(f"- transition_batch: {_display(selected.get('transition_pass', ''))}")
        lines.append("")

    lines.extend(_active_passes_section(state_dir))
    lines.extend(_pair_relations_section(state_dir, max_pairs=max_pairs))
    lines.extend(_components_section(state_dir))
    lines.extend(_batch_candidates_section(state_dir, max_batches=max_batches))
    lines.extend(_batch_validation_section(state_dir))
    lines.extend(_coverage_section(state_dir, coverage_counts))
    lines.extend(_reduction_section(state_dir))
    lines.extend(_transition_evidence_section(program_dir, selected_state_id))
    lines.extend(_interpretation_section(selected, batch_summary, coverage_counts, state_dir))

    case_path = program_dir / f"case_study_{program}.md"
    case_path.parent.mkdir(parents=True, exist_ok=True)
    case_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    index_row = {
        "program": program,
        "case_study_path": _relative_path(case_path, program_dir.parent),
        "selected_state": selected_state_id,
        "active_passes": active_pass_count,
        "batch_candidates": batch_candidate_count,
        "reduction_estimate": reduction,
        "dropped_active_passes": str(dropped),
    }
    return case_path, index_row


def _active_passes_section(state_dir: Path) -> list[str]:
    path = state_dir / "pass_profile.csv"
    if not path.exists():
        return _missing_section("Active Passes", path.name)
    rows = [
        row for row in _read_csv(path)
        if _is_true(row.get("success")) and _is_true(row.get("active"))
    ]
    shown = rows[:30]
    lines = [
        "## Active Passes",
        "",
        *_markdown_table(
            ["pass", "inst_delta", "funcs_changed", "blocks_changed", "changed_functions", "changed_blocks"],
            [
                [
                    row.get("pass", ""),
                    row.get("inst_delta", ""),
                    row.get("funcs_changed", ""),
                    row.get("blocks_changed", ""),
                    row.get("changed_functions", ""),
                    row.get("changed_blocks", ""),
                ]
                for row in shown
            ],
        ),
    ]
    if len(rows) > len(shown):
        lines.append(f"Showing first {len(shown)} of {len(rows)} active passes.")
    lines.append("")
    return lines


def _pair_relations_section(state_dir: Path, *, max_pairs: int) -> list[str]:
    path = state_dir / "pair_relation.csv"
    if not path.exists():
        return _missing_section("Pair Relations", path.name)
    rows = _prioritize_pair_rows(_read_csv(path))
    shown = rows[:max_pairs]
    lines = [
        "## Pair Relations",
        "",
        *_markdown_table(
            ["pass_a", "pass_b", "final_relation", "dynamic_relation", "same_hash", "static_relation"],
            [
                [
                    row.get("pass_a", ""),
                    row.get("pass_b", ""),
                    row.get("final_relation", ""),
                    row.get("dynamic_relation", ""),
                    row.get("same_hash", ""),
                    row.get("static_relation", ""),
                ]
                for row in shown
            ],
        ),
    ]
    if len(rows) > len(shown):
        lines.append(f"Showing first {len(shown)} of {len(rows)} pair relations.")
    lines.append("")
    return lines


def _components_section(state_dir: Path) -> list[str]:
    path = state_dir / "batch_components.csv"
    if not path.exists():
        return _missing_section("Conflict / Component Structure", path.name)
    return [
        "## Conflict / Component Structure",
        "",
        *_markdown_table(
            ["component_id", "component_size", "component_passes", "is_exact", "num_local_alternatives", "unresolved_reason"],
            [
                [
                    row.get("component_id", ""),
                    row.get("component_size", ""),
                    row.get("component_passes", ""),
                    row.get("is_exact", ""),
                    row.get("num_local_alternatives", ""),
                    row.get("unresolved_reason", ""),
                ]
                for row in _read_csv(path)
            ],
        ),
        "",
    ]


def _batch_candidates_section(state_dir: Path, *, max_batches: int) -> list[str]:
    path = state_dir / "batch_candidates.csv"
    if not path.exists():
        return _missing_section("Batch Candidates", path.name)
    candidates = _read_csv(path)
    correctness = {row.get("batch_id", ""): row for row in _read_csv(state_dir / "batch_correctness.csv")}
    shown = candidates[:max_batches]
    lines = [
        "## Batch Candidates",
        "",
        *_markdown_table(
            ["batch_id", "batch_size", "batch_passes", "correctness_class", "can_hard_fold", "can_execute"],
            [
                [
                    row.get("batch_id", ""),
                    row.get("batch_size", ""),
                    row.get("batch_passes", ""),
                    correctness.get(row.get("batch_id", ""), {}).get("correctness_class", ""),
                    correctness.get(row.get("batch_id", ""), {}).get("can_hard_fold", ""),
                    correctness.get(row.get("batch_id", ""), {}).get("can_execute", ""),
                ]
                for row in shown
            ],
        ),
    ]
    if len(candidates) > len(shown):
        lines.append(f"Showing first {len(shown)} of {len(candidates)} batch candidates.")
    lines.append("")
    return lines


def _batch_validation_section(state_dir: Path) -> list[str]:
    path = state_dir / "batch_validation.csv"
    if not path.exists():
        return _missing_section("Batch Validation", path.name)
    return [
        "## Batch Validation",
        "",
        *_markdown_table(
            ["batch_id", "validation_status", "tested_orders", "same_hash_count", "different_hash_count"],
            [
                [
                    row.get("batch_id", ""),
                    row.get("validation_status", ""),
                    row.get("tested_orders", ""),
                    row.get("same_hash_count", ""),
                    row.get("different_hash_count", ""),
                ]
                for row in _read_csv(path)
            ],
        ),
        "",
    ]


def _coverage_section(state_dir: Path, counts: Counter) -> list[str]:
    path = state_dir / "coverage_report.csv"
    if not path.exists():
        return _missing_section("Coverage", path.name)
    lines = [
        "## Coverage",
        "",
        *[f"- {status}: {counts.get(status, 0)}" for status in COVERAGE_STATUSES],
    ]
    if counts.get("dropped", 0) > 0:
        lines.append("- WARNING: at least one active pass was dropped from batch coverage.")
    lines.append("")
    return lines


def _reduction_section(state_dir: Path) -> list[str]:
    path = state_dir / "batch_summary.csv"
    if not path.exists():
        return _missing_section("Reduction Estimate", path.name)
    row = _first_row(path)
    return [
        "## Reduction Estimate",
        "",
        f"- active_passes: {_display(row.get('active_passes', ''))}",
        f"- naive_orderings_estimate: {_display(row.get('naive_orderings_estimate', ''))}",
        f"- batch_candidates: {_display(row.get('batch_candidates', ''))}",
        f"- batch_reduction_estimate: {_display(row.get('batch_reduction_estimate', ''))}",
        "",
    ]


def _transition_evidence_section(program_dir: Path, state_id: str) -> list[str]:
    path = program_dir / "batch_state_transitions.csv"
    if not path.exists():
        return _missing_section("State Transition Evidence", path.name)
    rows = _read_csv(path)
    incoming = [row for row in rows if row.get("child_state_id") == state_id]
    outgoing = [row for row in rows if row.get("parent_state_id") == state_id]
    lines = ["## State Transition Evidence", ""]
    if incoming:
        first = incoming[0]
        lines.append(f"- parent transition: {first.get('parent_state_id', '')} -> {state_id}")
        lines.append(f"- parent batch: {first.get('batch_id', '')} {first.get('batch_passes', '')}".rstrip())
        lines.append("")
    lines.extend(
        _markdown_table(
            ["batch_id", "child_state_id", "batch_size", "validation_status", "is_duplicate"],
            [
                [
                    row.get("batch_id", ""),
                    row.get("child_state_id", ""),
                    row.get("batch_size", ""),
                    row.get("validation_status", ""),
                    row.get("is_duplicate", ""),
                ]
                for row in outgoing
            ],
        )
    )
    lines.append("")
    return lines


def _interpretation_section(selected: dict, batch_summary: dict, coverage_counts: Counter, state_dir: Path) -> list[str]:
    order_sensitive = _first_value(selected, ["order_sensitive"], default="0")
    active_passes = _first_value(selected, ["active_passes"], default=str(_active_pass_count(state_dir / "pass_profile.csv")))
    batch_candidates = _first_value(batch_summary, ["batch_candidates"], default="0")
    certified_batches = sum(1 for row in _read_csv(state_dir / "batch_correctness.csv") if row.get("correctness_class") == "certified_batch")
    dropped = coverage_counts.get("dropped", 0)
    bullets = [
        f"- This selected state has {active_passes} active pass(es) in this run.",
        f"- It has {order_sensitive} observed order-sensitive pair relation(s), so batching should remain state-local.",
        f"- The state emits {batch_candidates} batch candidate(s); this is an estimate of reduced ordering choices, not a global optimum.",
    ]
    if dropped:
        bullets.append(f"- Coverage reports {dropped} dropped active pass(es), so this state needs inspection before using its batch family.")
    else:
        bullets.append("- Coverage reports no dropped active passes for this selected state.")
    if certified_batches:
        bullets.append(f"- {certified_batches} candidate batch(es) have all_permutations_same-style correctness evidence for hard folding.")
    else:
        bullets.append("- No certified batch evidence is available for this selected state.")
    return ["## Interpretation", "", *bullets[:5], ""]


def _write_case_studies_index(run_dir: Path, rows: list[dict]) -> Path:
    path = run_dir / "case_studies_index.md"
    lines = [
        "# Case Studies Index",
        "",
        *_markdown_table(
            [
                "program",
                "case study path",
                "selected state",
                "active passes",
                "batch candidates",
                "reduction estimate",
                "dropped active passes",
            ],
            [
                [
                    row.get("program", ""),
                    row.get("case_study_path", ""),
                    row.get("selected_state", ""),
                    row.get("active_passes", ""),
                    row.get("batch_candidates", ""),
                    row.get("reduction_estimate", ""),
                    row.get("dropped_active_passes", ""),
                ]
                for row in rows
            ],
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _state_row(program_dir: Path, state_id: str) -> dict:
    for row in _read_csv(program_dir / "states.csv"):
        if row.get("state_id") == state_id:
            return row
    return {"state_id": state_id, "state_dir": str(program_dir / "states" / state_id)}


def _state_dir(program_dir: Path, state: dict) -> Path:
    state_id = state.get("state_id", "S0000") or "S0000"
    value = state.get("state_dir", "")
    candidates = []
    if value:
        path = Path(value)
        candidates.append(path)
        if not path.is_absolute():
            candidates.append(program_dir.parent / path)
            candidates.append(program_dir / path)
    candidates.append(program_dir / "states" / state_id)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return program_dir / "states" / state_id


def _resolve_program_dir(run_dir: Path, output_dir: str, program: str) -> Path:
    path = Path(output_dir) if output_dir else Path(program)
    candidates = [path, run_dir / path, run_dir / program]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return run_dir / program


def _true_relation_flip_counts(path: Path) -> Counter:
    counts = Counter()
    for row in _read_csv(path):
        if row.get("flip_kind") in TRUE_RELATION_FLIPS and row.get("child_state_id"):
            counts[row["child_state_id"]] += 1
    return counts


def _coverage_counts(path: Path) -> Counter:
    counts = Counter(row.get("coverage_status", "") for row in _read_csv(path))
    if "" in counts:
        del counts[""]
    return counts


def _active_pass_count(path: Path) -> int:
    return sum(1 for row in _read_csv(path) if _is_true(row.get("success")) and _is_true(row.get("active")))


def _prioritize_pair_rows(rows: list[dict]) -> list[dict]:
    def priority(item: tuple[int, dict]) -> tuple[int, int]:
        index, row = item
        final_relation = row.get("final_relation", "")
        dynamic_relation = row.get("dynamic_relation", "")
        if final_relation == "final_order_sensitive":
            rank = 0
        elif dynamic_relation == "dynamic_commute" or final_relation == "final_commute":
            rank = 1
        elif "unknown" in final_relation or "unknown" in dynamic_relation:
            rank = 2
        else:
            rank = 3
        return rank, index

    return [row for _, row in sorted(enumerate(rows), key=priority)]


def _missing_section(title: str, filename: str) -> list[str]:
    return [f"## {title}", "", f"Missing: {filename}", ""]


def _markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    if not rows:
        return ["none"]
    lines = [f"| {' | '.join(headers)} |", f"| {' | '.join(['---'] * len(headers))} |"]
    lines.extend(f"| {' | '.join(_display(cell) for cell in row)} |" for row in rows)
    return lines


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _first_row(path: Path) -> dict:
    rows = _read_csv(path)
    return rows[0] if rows else {}


def _first_value(row: dict, names: list[str], default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return default


def _display(value: object) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")


def _to_int(value: object) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def _to_float(value: object) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0.0


def _is_true(value: object) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
