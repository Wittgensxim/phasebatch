from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path

from .equality_summary import equality_tier_markdown, equality_tier_summary_for_run
from .schema import REDUCTION_BY_STATE_FIELDS, REDUCTION_SUMMARY_FIELDS


def summarize_reduction(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    states = _read_csv(run_dir / "states.csv")
    leaf_reasons = _leaf_reasons(run_dir)
    selected_states = _selected_path_states(run_dir)

    rows = [_state_reduction_row(run_dir, state, leaf_reasons, selected_states) for state in states]
    summary_rows = [_summary_row(run_dir, rows, states)]

    by_state_path = run_dir / "reduction_by_state.csv"
    summary_path = run_dir / "reduction_summary.csv"
    md_path = run_dir / "reduction_summary.md"
    _write_csv(by_state_path, REDUCTION_BY_STATE_FIELDS, rows)
    _write_csv(summary_path, REDUCTION_SUMMARY_FIELDS, summary_rows)
    _write_markdown(md_path, rows, summary_rows[0], run_dir)
    return {
        "reduction_by_state_csv": str(by_state_path),
        "reduction_summary_csv": str(summary_path),
        "reduction_summary_md": str(md_path),
        "states": len(rows),
    }


def _state_reduction_row(run_dir: Path, state: dict, leaf_reasons: dict[str, str], selected_states: set[str]) -> dict:
    state_id = state.get("state_id", "")
    state_dir = _state_dir(run_dir, state)
    per_state = _first_row(state_dir / "per_state_summary.csv")
    active_passes = _active_passes(state_dir, per_state)
    relation_counts = _relation_counts(state_dir, per_state)
    correctness_counts = _correctness_counts(state_dir)
    batch_candidates = len(_read_csv(state_dir / "batch_candidates.csv"))
    executable_batches = correctness_counts["executable_batches"]
    naive_log10 = _factorial_log10(active_passes)
    no_executable = active_passes > 1 and executable_batches == 0
    local_log10 = 0.0 if active_passes <= 1 else naive_log10 - math.log10(max(1, executable_batches))
    return {
        "program": state.get("program") or per_state.get("program") or run_dir.name,
        "state_id": state_id,
        "depth": state.get("depth") or per_state.get("depth", ""),
        "state_hash": state.get("state_hash") or per_state.get("state_hash", ""),
        "active_passes": str(active_passes),
        "tested_pairs": str(relation_counts["tested_pairs"]),
        "commute_pairs": str(relation_counts["commute_pairs"]),
        "order_sensitive_pairs": str(relation_counts["order_sensitive_pairs"]),
        "unknown_pairs": str(relation_counts["unknown_pairs"]),
        "naive_orderings_log10": _fmt_float(naive_log10),
        "batch_candidates": str(batch_candidates),
        "certified_batches": str(correctness_counts["certified_batches"]),
        "executable_batches": str(executable_batches),
        "sampled_batches": str(correctness_counts["sampled_batches"]),
        "rejected_batches": str(correctness_counts["rejected_batches"]),
        "failed_batches": str(correctness_counts["failed_batches"]),
        "unvalidated_batches": str(correctness_counts["unvalidated_batches"]),
        "skipped_batches": str(correctness_counts["skipped_batches"]),
        "dropped_active_passes": str(_dropped_active_passes(state_dir)),
        "local_reduction_log10": _fmt_float(local_log10),
        "local_reduction_ratio": _ratio_text(active_passes, executable_batches),
        "no_executable_batches": _bool(no_executable),
        "terminal_due_max_depth": _bool(leaf_reasons.get(state_id) == "max_rounds_reached"),
        "selected_on_final_path": _bool(state_id in selected_states),
    }


def _summary_row(run_dir: Path, rows: list[dict], states: list[dict]) -> dict:
    chosen_path = _read_csv(run_dir / "chosen_path.csv")
    pass_invocations = _selected_path_pass_invocations(run_dir, chosen_path)
    return {
        "program": rows[0]["program"] if rows else run_dir.name,
        "total_states": str(len(rows)),
        "max_depth": str(max((_int(row.get("depth")) for row in rows), default=0)),
        "total_active_passes": str(_sum(rows, "active_passes")),
        "total_tested_pairs": str(_sum(rows, "tested_pairs")),
        "total_commute_pairs": str(_sum(rows, "commute_pairs")),
        "total_order_sensitive_pairs": str(_sum(rows, "order_sensitive_pairs")),
        "total_batch_candidates": str(_sum(rows, "batch_candidates")),
        "total_certified_batches": str(_sum(rows, "certified_batches")),
        "total_executable_batches": str(_sum(rows, "executable_batches")),
        "total_executed_transitions": str(len(_read_csv(run_dir / "batch_state_transitions.csv"))),
        "total_skipped_batches": str(_sum(rows, "skipped_batches")),
        "total_dropped_active_passes": str(_sum(rows, "dropped_active_passes")),
        "avg_active_passes": _avg(rows, "active_passes"),
        "avg_batch_candidates": _avg(rows, "batch_candidates"),
        "avg_executable_batches": _avg(rows, "executable_batches"),
        "avg_local_reduction_log10": _avg(rows, "local_reduction_log10"),
        "max_local_reduction_log10": _fmt_float(max((_float(row.get("local_reduction_log10")) for row in rows), default=0.0)),
        "selected_path_steps": str(len(chosen_path)),
        "selected_path_pass_invocations": str(pass_invocations),
        "final_pipeline_length": str(len(_split_pipeline(_read_text(run_dir / "optimized_pipeline.txt")))),
    }


def _write_markdown(path: Path, rows: list[dict], summary: dict, run_dir: Path) -> None:
    selected = _selected_final_state(run_dir)
    pair_totals = {
        "tested": _sum(rows, "tested_pairs"),
        "commute": _sum(rows, "commute_pairs"),
        "sensitive": _sum(rows, "order_sensitive_pairs"),
        "unknown": _sum(rows, "unknown_pairs"),
    }
    correctness = {
        "certified": _sum(rows, "certified_batches"),
        "sampled": _sum(rows, "sampled_batches"),
        "rejected": _sum(rows, "rejected_batches"),
        "failed": _sum(rows, "failed_batches"),
        "unvalidated": _sum(rows, "unvalidated_batches"),
        "skipped": _sum(rows, "skipped_batches"),
        "executed": _int(summary.get("total_executed_transitions")),
    }
    lines = [
        "# Reduction Evidence Summary",
        "",
        "## Overall",
        "",
        f"- program: {summary.get('program', '')}",
        f"- states analyzed: {summary.get('total_states', '')}",
        f"- max depth: {summary.get('max_depth', '')}",
        f"- total batch transitions: {summary.get('total_executed_transitions', '')}",
        f"- selected final state: {selected}",
        "",
        "## Per-State Reduction",
        "",
        *_markdown_table(
            [
                "state",
                "depth",
                "active passes",
                "naive log10(n!)",
                "executable batches",
                "certified batches",
                "local reduction log10",
                "dropped",
            ],
            [
                [
                    row.get("state_id", ""),
                    row.get("depth", ""),
                    row.get("active_passes", ""),
                    row.get("naive_orderings_log10", ""),
                    row.get("executable_batches", ""),
                    row.get("certified_batches", ""),
                    row.get("local_reduction_log10", ""),
                    row.get("dropped_active_passes", ""),
                ]
                for row in rows
            ],
        ),
        "",
        "## Pair Relation Evidence",
        "",
        *_markdown_table(
            ["total tested pairs", "commute", "order-sensitive", "unknown"],
            [[str(pair_totals["tested"]), str(pair_totals["commute"]), str(pair_totals["sensitive"]), str(pair_totals["unknown"])]],
        ),
        "",
        *equality_tier_markdown(equality_tier_summary_for_run(run_dir)),
        "",
        "## Batch Correctness",
        "",
        *_markdown_table(
            ["certified", "sampled", "rejected", "failed", "unvalidated", "skipped", "executed"],
            [[str(correctness[key]) for key in ["certified", "sampled", "rejected", "failed", "unvalidated", "skipped", "executed"]]],
        ),
        "",
        "## Coverage",
        "",
        f"- dropped active passes: {summary.get('total_dropped_active_passes', '0')}",
        f"- terminal states due max depth: {sum(1 for row in rows if row.get('terminal_due_max_depth') == 'true')}",
        f"- unresolved states: {sum(1 for row in rows if row.get('no_executable_batches') == 'true')}",
        "",
        "## Interpretation",
        "",
        "- This run reduces local ordering choices from n! to certified/executable batch alternatives.",
        "- Reduction is state-local and applies only under the current compiler, pass set, target, and IR state.",
        "- Objective is not used as commutation proof.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _active_passes(state_dir: Path, per_state: dict) -> int:
    value = per_state.get("active_passes")
    if value not in {None, ""}:
        return _int(value)
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
    tested = sum(counts.values())
    return {
        "tested_pairs": tested,
        "commute_pairs": counts.get("final_commute", 0),
        "order_sensitive_pairs": counts.get("final_order_sensitive", 0),
        "unknown_pairs": counts.get("final_unknown", 0) + counts.get("unknown", 0),
    }


def _correctness_counts(state_dir: Path) -> dict[str, int]:
    rows = _read_csv(state_dir / "batch_correctness.csv")
    classes = Counter(row.get("correctness_class", "") for row in rows)
    executable = sum(1 for row in rows if _is_true(row.get("can_execute")))
    return {
        "certified_batches": classes.get("certified_batch", 0),
        "executable_batches": executable,
        "sampled_batches": classes.get("sampled_batch", 0),
        "rejected_batches": classes.get("rejected_batch", 0),
        "failed_batches": classes.get("failed_batch", 0),
        "unvalidated_batches": classes.get("unvalidated_batch", 0) + classes.get("unknown_batch", 0),
        "skipped_batches": sum(1 for row in rows if not _is_true(row.get("can_execute"))),
    }


def _dropped_active_passes(state_dir: Path) -> int:
    summary = _first_row(state_dir / "coverage_summary.csv")
    if summary:
        return _int(summary.get("dropped_active_passes"))
    return sum(1 for row in _read_csv(state_dir / "coverage_report.csv") if row.get("coverage_status") == "dropped")


def _state_dir(run_dir: Path, state: dict) -> Path:
    raw = state.get("state_dir", "")
    if raw:
        return Path(raw)
    return run_dir / "states" / state.get("state_id", "")


def _leaf_reasons(run_dir: Path) -> dict[str, str]:
    return {row.get("state_id", ""): row.get("leaf_reason", "") for row in _read_csv(run_dir / "leaf_states.csv") if row.get("state_id")}


def _selected_path_states(run_dir: Path) -> set[str]:
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
    chosen = _first_row(run_dir / "chosen_path_summary.csv")
    if chosen.get("selected_final_state"):
        return chosen["selected_final_state"]
    return ""


def _selected_path_pass_invocations(run_dir: Path, chosen_path: list[dict]) -> int:
    chosen_summary = _first_row(run_dir / "chosen_path_summary.csv")
    if chosen_summary.get("total_pass_invocations"):
        return _int(chosen_summary.get("total_pass_invocations"))
    return sum(len(_split_pipeline(row.get("canonical_order") or row.get("batch_passes", ""))) for row in chosen_path)


def _factorial_log10(active_passes: int) -> float:
    if active_passes <= 1:
        return 0.0
    return math.lgamma(active_passes + 1) / math.log(10)


def _ratio_text(active_passes: int, executable_batches: int) -> str:
    if active_passes <= 1:
        return "1"
    naive = math.factorial(active_passes)
    denominator = max(1, executable_batches)
    if naive % denominator == 0:
        return str(naive // denominator)
    return _fmt_float(naive / denominator)


def _avg(rows: list[dict], key: str) -> str:
    if not rows:
        return "0"
    return _fmt_float(sum(_float(row.get(key)) for row in rows) / len(rows))


def _sum(rows: list[dict], key: str) -> int:
    return sum(_int(row.get(key)) for row in rows)


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
