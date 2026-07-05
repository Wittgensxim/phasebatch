from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path


REQUIRED_INPUTS = [
    "mainline_runs.csv",
    "mainline_aggregate_states.csv",
    "mainline_aggregate_batches.csv",
    "mainline_aggregate_coverage.csv",
    "mainline_aggregate_overlap.csv",
    "mainline_missing_outputs.csv",
]


def generate_mainline_summary(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    data, warnings = _load_inputs(run_dir)
    runs = data["mainline_runs.csv"]
    states = data["mainline_aggregate_states.csv"]
    batches = data["mainline_aggregate_batches.csv"]
    coverage = data["mainline_aggregate_coverage.csv"]
    overlap = data["mainline_aggregate_overlap.csv"]
    missing = data["mainline_missing_outputs.csv"]

    lines = [
        "# Mainline Summary",
        "",
    ]
    if warnings:
        lines.extend(_warning_section(warnings))

    lines.extend(_overall_section(runs, states, batches))
    lines.extend(_program_status_section(runs, states, batches))
    lines.extend(_state_relation_section(states))
    lines.extend(_batch_reduction_section(run_dir, runs, batches))
    lines.extend(_batch_validation_section(batches, coverage))
    lines.extend(_coverage_section(coverage, batches))
    lines.extend(_overlap_section(overlap))
    lines.extend(_observations_section(runs, states, batches, coverage, overlap))
    lines.extend(_missing_outputs_section(missing, runs))

    path = run_dir / "mainline_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _load_inputs(run_dir: Path) -> tuple[dict[str, list[dict]], list[str]]:
    data: dict[str, list[dict]] = {}
    warnings: list[str] = []
    for name in REQUIRED_INPUTS:
        path = run_dir / name
        if path.exists():
            data[name] = _read_csv(path)
        else:
            data[name] = []
            warnings.append(f"missing input CSV: {name}")
    return data, warnings


def _overall_section(runs: list[dict], states: list[dict], batches: list[dict]) -> list[str]:
    total_programs = len(runs)
    successful = sum(1 for row in runs if row.get("status") == "success")
    failed = sum(1 for row in runs if row.get("status") == "failed")
    total_states = sum(_to_int(_first_value(row, ["num_states", "states"])) for row in states)
    total_batch_transitions = sum(_to_int(row.get("executed")) for row in batches)
    max_depth = _max_depth(states, batches)
    total_time = sum(_to_float(row.get("total_time_ms")) for row in runs)
    return [
        f"- total programs: {total_programs}",
        f"- successful programs: {successful}",
        f"- failed programs: {failed}",
        f"- total states: {total_states}",
        f"- total batch transitions: {total_batch_transitions}",
        f"- max depth observed: {max_depth}",
        f"- total time ms: {_format_number(total_time)}",
        "",
    ]


def _program_status_section(runs: list[dict], states: list[dict], batches: list[dict]) -> list[str]:
    state_totals = defaultdict(int)
    batch_totals = defaultdict(int)
    max_depths = defaultdict(int)
    for row in states:
        program = row.get("program", "")
        state_totals[program] += _to_int(_first_value(row, ["num_states", "states"]))
        max_depths[program] = max(max_depths[program], _to_int(row.get("depth")))
    for row in batches:
        program = row.get("program", "")
        batch_totals[program] += _to_int(row.get("executed"))
        max_depths[program] = max(max_depths[program], _to_int(row.get("depth")))

    rows = []
    for run in runs:
        program = run.get("program", "")
        rows.append(
            [
                program,
                run.get("status", ""),
                str(state_totals.get(program, 0)),
                str(batch_totals.get(program, 0)),
                str(max_depths.get(program, 0)),
                run.get("total_time_ms", ""),
                run.get("error_message", ""),
            ]
        )
    return [
        "## Program Status",
        "",
        *_markdown_table(["program", "status", "states", "batch transitions", "max depth", "time ms", "error"], rows),
        "",
    ]


def _state_relation_section(states: list[dict]) -> list[str]:
    rows = [
        [
            row.get("program", ""),
            row.get("depth", ""),
            _first_value(row, ["num_states", "states"]),
            _first_value(row, ["avg_active_passes", "active_passes"]),
            _first_value(row, ["avg_pairs_tested", "pairs_tested", "avg_tested_pairs"]),
            _first_value(row, ["avg_dynamic_commute", "dynamic_commute"]),
            _first_value(row, ["avg_order_sensitive", "order_sensitive"]),
            _first_value(row, ["avg_unknown", "unknown"]),
        ]
        for row in states
    ]
    return [
        "## State Relation Summary",
        "",
        *_markdown_table(
            ["program", "depth", "states", "avg active passes", "avg tested pairs", "avg commute", "avg order-sensitive", "avg unknown"],
            rows,
        ),
        "",
    ]


def _batch_reduction_section(run_dir: Path, runs: list[dict], batches: list[dict]) -> list[str]:
    naive_log10 = _naive_log10_by_program_depth(run_dir, runs)
    rows = [
        [
            row.get("program", ""),
            row.get("depth", ""),
            row.get("states", ""),
            _first_value(row, ["avg_candidates", "avg_batch_candidates", "batch_candidates"]),
            _first_value(row, ["avg_batch_size", "batch_size"]),
            naive_log10.get(_program_depth_key(row), "N/A"),
            _first_value(row, ["avg_reduction", "batch_reduction_estimate", "avg_reduction_estimate"]),
            row.get("executed", ""),
            row.get("skipped", ""),
        ]
        for row in batches
    ]
    return [
        "## Batch Reduction Summary",
        "",
        *_markdown_table(
            [
                "program",
                "depth",
                "states",
                "avg batch candidates",
                "avg batch size",
                "avg log10 naive orderings",
                "avg reduction estimate",
                "executed batches",
                "skipped batches",
            ],
            rows,
        ),
        "",
    ]


def _batch_validation_section(batches: list[dict], coverage: list[dict]) -> list[str]:
    terminal_candidates = _terminal_candidate_counts_by_key(batches, coverage)
    rows = [
        [
            row.get("program", ""),
            row.get("depth", ""),
            _first_value(row, ["avg_candidates", "batch_candidates"], default="N/A"),
            _candidate_total(row),
            _first_value(row, ["all_permutations_same", "certified"], default="N/A"),
            _first_value(row, ["sampled_same", "sampled"], default="N/A"),
            row.get("mismatch", "N/A"),
            row.get("failed", "N/A"),
            _first_value(row, ["not_validated", "unvalidated"], default="N/A"),
            str(terminal_candidates.get(_program_depth_key(row), 0)),
        ]
        for row in batches
    ]
    return [
        "## Batch Validation Summary",
        "",
        *_markdown_table(
            [
                "program",
                "depth",
                "avg candidates",
                "total candidates",
                "certified",
                "sampled",
                "mismatch",
                "failed",
                "unvalidated",
                "terminal candidates due max depth",
            ],
            rows,
        ),
        "",
    ]


def _coverage_section(coverage: list[dict], batches: list[dict]) -> list[str]:
    terminal_batch_keys = _terminal_batch_keys(batches)
    rows = []
    for row in coverage:
        dropped = row.get("dropped_active_passes", "")
        dropped_text = f"**WARNING** {dropped}" if _to_int(dropped) > 0 else dropped
        unvalidated = _to_int(row.get("unvalidated_covered"))
        failed_or_unknown = _to_int(row.get("failed_or_unknown"))
        terminal_count = _to_int(row.get("not_executed_due_to_max_depth"))
        if terminal_count == 0 and _program_depth_key(row) in terminal_batch_keys and unvalidated > 0:
            terminal_count = unvalidated
            unvalidated = 0
        rows.append(
            [
                row.get("program", ""),
                row.get("depth", ""),
                row.get("active_passes", ""),
                row.get("certified_covered", ""),
                row.get("heuristic_covered", ""),
                row.get("unresolved_conflict", ""),
                row.get("validation_rejected", ""),
                str(unvalidated + failed_or_unknown),
                str(terminal_count),
                dropped_text,
            ]
        )
    return [
        "## Coverage Summary",
        "",
        *_markdown_table(
            [
                "program",
                "depth",
                "active passes",
                "certified covered",
                "heuristic covered",
                "unresolved",
                "rejected",
                "unknown",
                "terminal not covered due max depth",
                "dropped",
            ],
            rows,
        ),
        "",
    ]


def _overlap_section(overlap: list[dict]) -> list[str]:
    rows = [
        [
            row.get("program", ""),
            row.get("depth", ""),
            _first_value(row, ["total_pairs", "pairs"]),
            row.get("disjoint_write", ""),
            row.get("same_function_overlap", ""),
            row.get("same_block_overlap", ""),
            row.get("unknown_overlap", ""),
            row.get("overlap_and_commute", ""),
            _first_value(row, ["overlap_and_order_sensitive", "overlap_and_sensitive"]),
        ]
        for row in overlap
    ]
    return [
        "## Coarse Footprint / Overlap Diagnostics",
        "",
        "Coarse footprint labels are diagnostic only and are not used as hard independence proof.",
        "",
        *_markdown_table(
            [
                "program",
                "depth",
                "pairs",
                "disjoint_write",
                "same_function_overlap",
                "same_block_overlap",
                "unknown_overlap",
                "overlap_and_commute",
                "overlap_and_sensitive",
            ],
            rows,
        ),
        "",
    ]


def _observations_section(
    runs: list[dict],
    states: list[dict],
    batches: list[dict],
    coverage: list[dict],
    overlap: list[dict],
) -> list[str]:
    bullets: list[str] = []
    successful = sum(1 for row in runs if row.get("status") == "success")
    failed = sum(1 for row in runs if row.get("status") == "failed")
    if successful:
        bullets.append(f"- Observed {successful} successful program(s) in this run.")
    if failed:
        bullets.append(f"- Observed {failed} failed program(s); inspect Program Status before comparing aggregates.")

    depth0_active, later_active = _depth_active_averages(states)
    if depth0_active is not None and later_active is not None:
        if later_active < depth0_active:
            bullets.append("- Across successful programs, average active passes decrease from depth 0 to later depths in this run.")
        elif later_active > depth0_active:
            bullets.append("- Across successful programs, average active passes increase after depth 0 in this run.")
        else:
            bullets.append("- Average active passes are roughly unchanged between depth 0 and later depths in this run.")

    dropped = sum(_to_int(row.get("dropped_active_passes")) for row in coverage)
    if dropped == 0 and coverage:
        bullets.append("- Dropped active passes are zero in the observed coverage summaries.")
    elif dropped > 0:
        bullets.append("- Some dropped active passes are observed; coverage should be inspected before using batch results.")

    executed = sum(_to_int(row.get("executed")) for row in batches)
    certified = sum(_to_int(row.get("all_permutations_same")) for row in batches)
    if executed:
        bullets.append(f"- Executed batch transitions are observed ({executed}); {certified} candidate(s) are reported as all_permutations_same.")

    sensitive_overlap = sum(_to_int(row.get("overlap_and_order_sensitive")) for row in overlap)
    commute_overlap = sum(_to_int(row.get("overlap_and_commute")) for row in overlap)
    if sensitive_overlap or commute_overlap:
        bullets.append("- Coarse overlap appears alongside both commute and order-sensitive relations, suggesting it should remain diagnostic in this MVP.")

    if not bullets:
        bullets.append("- Data is partial; no strong aggregate observation is available from the existing CSVs.")

    return [
        "## Key Observations",
        "",
        *bullets[:6],
        "",
    ]


def _missing_outputs_section(missing: list[dict], runs: list[dict]) -> list[str]:
    missing_rows = [
        [row.get("program", ""), row.get("expected_file", ""), row.get("status", "")]
        for row in missing
        if row.get("status") != "present"
    ]
    failed_rows = [
        [row.get("program", ""), "program_run", row.get("error_message", "failed")]
        for row in runs
        if row.get("status") == "failed"
    ]
    rows = missing_rows + failed_rows
    return [
        "## Missing Outputs / Failures",
        "",
        *_markdown_table(["program", "expected file", "status"], rows),
        "",
    ]


def _terminal_candidate_counts_by_key(batches: list[dict], coverage: list[dict]) -> dict[tuple[str, str], int]:
    terminal_active = {
        _program_depth_key(row): _to_int(row.get("not_executed_due_to_max_depth"))
        for row in coverage
        if _to_int(row.get("not_executed_due_to_max_depth")) > 0
    }
    counts: dict[tuple[str, str], int] = {}
    for row in batches:
        key = _program_depth_key(row)
        if key not in terminal_active and not _looks_like_terminal_unexecuted_batch_row(row):
            continue
        candidates = _to_float(_first_value(row, ["avg_candidates", "batch_candidates"], default="0"))
        states = max(1, _to_int(row.get("states")))
        counts[key] = int(round(candidates * states))
    return counts


def _candidate_total(row: dict) -> str:
    explicit = _first_value(row, ["total_candidates", "batch_candidates_total", "candidates_total"])
    if explicit:
        return explicit
    candidates = _to_float(_first_value(row, ["avg_candidates", "batch_candidates"], default="0"))
    states = _to_int(row.get("states"))
    return str(int(round(candidates * states))) if states > 0 else "0"


def _naive_log10_by_program_depth(run_dir: Path, runs: list[dict]) -> dict[tuple[str, str], str]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for run in runs:
        program = run.get("program", "")
        if not program:
            continue
        program_dir = _resolve_program_output_dir(run_dir, run)
        if not program_dir.exists():
            continue
        state_depths = {
            row.get("state_id", ""): str(_to_int(row.get("depth")))
            for row in _read_csv(program_dir / "states.csv")
            if row.get("state_id")
        }
        states_dir = program_dir / "states"
        if not states_dir.exists():
            continue
        for summary_path in states_dir.glob("*/batch_summary.csv"):
            state_id = summary_path.parent.name
            depth = state_depths.get(state_id)
            if depth is None:
                continue
            summary = _first_row(summary_path)
            log_value = _log10_numeric_text(_first_value(summary, ["naive_orderings_estimate", "avg_naive_orderings"]))
            if log_value is not None:
                values[(program, depth)].append(log_value)
    return {key: _format_number(sum(logs) / len(logs)) for key, logs in values.items() if logs}


def _resolve_program_output_dir(run_dir: Path, run: dict) -> Path:
    program = run.get("program", "")
    raw_value = run.get("output_dir", "")
    candidates: list[Path] = []
    if raw_value:
        raw_path = Path(raw_value)
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.extend([run_dir / raw_path.name, run_dir / raw_path, raw_path])
    if program:
        candidates.append(run_dir / program)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else run_dir / program


def _log10_numeric_text(value: object) -> float | None:
    text = str(value or "").strip().replace(",", "")
    if not text:
        return None
    try:
        numeric = float(text)
        if numeric > 0 and math.isfinite(numeric):
            return math.log10(numeric)
    except (OverflowError, ValueError):
        pass
    whole = text.split(".", 1)[0].lstrip("+").lstrip("0")
    if not whole or not whole.isdigit():
        return None
    prefix = whole[:15]
    return math.log10(float(prefix)) + len(whole) - len(prefix)


def _terminal_batch_keys(batches: list[dict]) -> set[tuple[str, str]]:
    return {_program_depth_key(row) for row in batches if _looks_like_terminal_unexecuted_batch_row(row)}


def _looks_like_terminal_unexecuted_batch_row(row: dict) -> bool:
    if _to_int(row.get("executed")) > 0 or _to_int(row.get("skipped")) > 0:
        return False
    validation_fields = [
        "all_permutations_same",
        "sampled_same",
        "not_validated",
        "mismatch",
        "failed",
        "certified",
        "sampled",
        "unvalidated",
    ]
    if any(_to_int(row.get(field)) > 0 for field in validation_fields):
        return False
    if str(row.get("validation_counts", "")).strip():
        return False
    return _to_float(_first_value(row, ["avg_candidates", "batch_candidates"], default="0")) > 0


def _program_depth_key(row: dict) -> tuple[str, str]:
    return row.get("program", ""), str(_to_int(row.get("depth")))


def _warning_section(warnings: list[str]) -> list[str]:
    return [
        "## Warnings",
        "",
        *[f"- {warning}" for warning in warnings],
        "",
    ]


def _depth_active_averages(states: list[dict]) -> tuple[float | None, float | None]:
    depth0 = [_to_float(_first_value(row, ["avg_active_passes", "active_passes"])) for row in states if _to_int(row.get("depth")) == 0]
    later = [_to_float(_first_value(row, ["avg_active_passes", "active_passes"])) for row in states if _to_int(row.get("depth")) > 0]
    return _average(depth0), _average(later)


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _max_depth(*row_groups: list[dict]) -> int:
    depths = [_to_int(row.get("depth")) for rows in row_groups for row in rows]
    return max(depths, default=0)


def _first_value(row: dict, names: list[str], default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return default


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["none"]
    lines = [f"| {' | '.join(headers)} |", f"| {' | '.join(['---'] * len(headers))} |"]
    lines.extend(f"| {' | '.join(str(cell) for cell in row)} |" for row in rows)
    return lines


def _format_number(value: float) -> str:
    return f"{value:.2f}"


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


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _first_row(path: Path) -> dict:
    rows = _read_csv(path)
    return rows[0] if rows else {}
