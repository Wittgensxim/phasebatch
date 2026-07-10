from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from .schema import VALIDATION_LADDER_SUMMARY_FIELDS


BOUNDARY_TEXT = (
    "Bounded and sampled validation reduce validation cost but do not become hard commutation proof by default. "
    "Only complete all-permutations validation or explicitly complete certificates are hard-foldable."
)

DAG_BOUNDARY_TEXT = (
    "Permutation DAG validation is a hard certificate only when exploration is complete and all full-subset paths "
    "merge into one final IR equivalence class. If the DAG budget is exceeded, the result is incomplete and cannot "
    "be used for hard folding."
)


def write_batch_validation_ladder_summary(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    state_dirs = _state_dirs(run_dir)
    rows = [_summary_row(state_dir) for state_dir in state_dirs if (state_dir / "batch_validation.csv").exists()]
    dag_rows = _dag_detail_rows(state_dirs)
    _write_csv(run_dir / "batch_validation_ladder_summary.csv", VALIDATION_LADDER_SUMMARY_FIELDS, rows)
    _write_markdown(run_dir / "batch_validation_ladder_summary.md", rows, dag_rows)
    return {
        "batch_validation_ladder_summary_csv": str(run_dir / "batch_validation_ladder_summary.csv"),
        "batch_validation_ladder_summary_md": str(run_dir / "batch_validation_ladder_summary.md"),
    }


def _state_dirs(run_dir: Path) -> list[Path]:
    if (run_dir / "states").exists():
        return sorted(path for path in (run_dir / "states").iterdir() if path.is_dir())
    return [run_dir]


def _summary_row(state_dir: Path) -> dict:
    validation_rows = _read_csv(state_dir / "batch_validation.csv")
    correctness_rows = _read_csv(state_dir / "batch_correctness.csv")
    status_counts = Counter(row.get("validation_status", "") for row in validation_rows)
    tier_counts = Counter(row.get("validation_tier", "") for row in validation_rows)
    class_counts = Counter(row.get("correctness_class", "") for row in correctness_rows)
    dag_rows = [row for row in validation_rows if _is_dag_row(row)]
    first = validation_rows[0] if validation_rows else correctness_rows[0] if correctness_rows else {}
    state_summary = _first_row(state_dir / "per_state_summary.csv")
    executable_batches = sum(1 for row in correctness_rows if row.get("can_execute") == "true")
    hard_certified_batches = sum(1 for row in validation_rows if row.get("validation_hard_certificate") == "true")
    return {
        "program": first.get("program", ""),
        "state_id": first.get("state_id", state_dir.name),
        "depth": state_summary.get("depth", first.get("depth", "")),
        "batch_candidates": str(max(len(validation_rows), len(correctness_rows))),
        "exhaustive_batches": str(tier_counts.get("exhaustive_all_permutations", 0)),
        "dag_batches": str(len(dag_rows)),
        "dag_certified_batches": str(tier_counts.get("permutation_dag_exact", 0)),
        "dag_mismatch_batches": str(tier_counts.get("permutation_dag_mismatch", 0)),
        "dag_incomplete_batches": str(tier_counts.get("permutation_dag_incomplete", 0)),
        "dag_nodes": str(sum(_to_int(row.get("validation_dag_nodes")) for row in dag_rows)),
        "dag_edges": str(sum(_to_int(row.get("validation_dag_edges")) for row in dag_rows)),
        "dag_hash_merges": str(sum(_to_int(row.get("validation_dag_hash_merges")) for row in dag_rows)),
        "dag_structural_merges": str(sum(_to_int(row.get("validation_dag_structural_merges")) for row in dag_rows)),
        "dag_transition_cache_hits": str(sum(_to_int(row.get("validation_dag_transition_cache_hits")) for row in dag_rows)),
        "dag_equivalence_cache_hits": str(sum(_to_int(row.get("validation_dag_equivalence_cache_hits")) for row in dag_rows)),
        "avg_dag_compression_vs_permutation": _format_float(_avg([_to_float(row.get("compression_vs_permutation")) for row in dag_rows])),
        "bounded_batches": str(sum(1 for row in validation_rows if row.get("validation_status") == "bounded_same" or row.get("validation_tier") in {"bounded_insertion", "bounded_adjacent_swap"})),
        "sampled_batches": str(sum(1 for row in validation_rows if row.get("validation_status") == "sampled_same" or row.get("validation_tier") == "sampled_permutations")),
        "rejected_batches": str(max(status_counts.get("mismatch", 0), class_counts.get("rejected_batch", 0))),
        "failed_batches": str(max(status_counts.get("failed", 0), class_counts.get("failed_batch", 0))),
        "unvalidated_batches": str(max(status_counts.get("not_validated", 0), class_counts.get("unvalidated_batch", 0))),
        "hard_certified_batches": str(hard_certified_batches),
        "executable_batches": str(executable_batches),
        "validation_sequences_tested": str(sum(_to_int(row.get("validation_sequences_tested") or row.get("tested_orders")) for row in validation_rows)),
        "validation_opt_invocations": str(sum(_to_int(row.get("validation_opt_invocations") or row.get("tested_orders")) for row in validation_rows)),
        "validation_pass_invocations_baseline": str(sum(_to_int(row.get("validation_pass_invocations_baseline")) for row in validation_rows)),
        "validation_pass_invocations_actual": str(sum(_to_int(row.get("validation_pass_invocations_actual")) for row in validation_rows)),
        "validation_pass_invocations_saved": str(sum(_to_int(row.get("validation_pass_invocations_saved")) for row in validation_rows)),
        "validation_profile_reuse_hits": str(sum(_to_int(row.get("validation_profile_reuse_hits")) for row in validation_rows)),
        "validation_state_transition_cache_hits": str(sum(_to_int(row.get("validation_state_transition_cache_hits")) for row in validation_rows)),
        "validation_state_equivalence_cache_hits": str(sum(_to_int(row.get("validation_state_equivalence_cache_hits")) for row in validation_rows)),
        "validation_time_ms": _format_ms(sum(_to_float(row.get("time_ms")) for row in validation_rows)),
    }


def _dag_detail_rows(state_dirs: list[Path]) -> list[dict]:
    rows = []
    for state_dir in state_dirs:
        for row in _read_csv(state_dir / "batch_validation.csv"):
            if not _is_dag_row(row):
                continue
            rows.append(
                {
                    "program": row.get("program", ""),
                    "state_id": row.get("state_id", state_dir.name),
                    "batch_id": row.get("batch_id", ""),
                    "batch_size": row.get("batch_size", ""),
                    "factorial_permutations_log10": row.get("factorial_permutations_log10", ""),
                    "validation_tier": row.get("validation_tier", ""),
                    "validation_status": row.get("validation_status", ""),
                    "dag_nodes": row.get("validation_dag_nodes", ""),
                    "dag_edges": row.get("validation_dag_edges", ""),
                    "final_equivalence_classes": row.get("validation_dag_final_classes", ""),
                    "hash_merges": row.get("validation_dag_hash_merges", ""),
                    "structural_merges": row.get("validation_dag_structural_merges", ""),
                    "transition_cache_hits": row.get("validation_dag_transition_cache_hits", ""),
                    "equivalence_cache_hits": row.get("validation_dag_equivalence_cache_hits", ""),
                    "compression_vs_permutation": row.get("compression_vs_permutation", ""),
                }
            )
    return rows


def _write_markdown(path: Path, rows: list[dict], dag_rows: list[dict]) -> None:
    validation_modes = sorted({row.get("validation_mode", "") for row in rows if row.get("validation_mode")})
    totals = {field: sum(_to_int(row.get(field)) for row in rows) for field in VALIDATION_LADDER_SUMMARY_FIELDS if field not in {"program", "state_id", "depth", "validation_time_ms"}}
    validation_time = sum(_to_float(row.get("validation_time_ms")) for row in rows)
    dag_compressions = [_to_float(row.get("compression_vs_permutation")) for row in dag_rows]
    lines = [
        "# Batch Validation Ladder Summary",
        "",
        "## Overall",
        "",
        f"- validation mode: {';'.join(validation_modes) if validation_modes else ''}",
        f"- exhaustive batches: {totals.get('exhaustive_batches', 0)}",
        f"- DAG validated batches: {totals.get('dag_batches', 0)}",
        f"- DAG certified batches: {totals.get('dag_certified_batches', 0)}",
        f"- DAG mismatches: {totals.get('dag_mismatch_batches', 0)}",
        f"- DAG incomplete: {totals.get('dag_incomplete_batches', 0)}",
        f"- bounded batches: {totals.get('bounded_batches', 0)}",
        f"- sampled batches: {totals.get('sampled_batches', 0)}",
        f"- rejected/failed: {totals.get('rejected_batches', 0) + totals.get('failed_batches', 0)}",
        f"- hard-certified batches: {totals.get('hard_certified_batches', 0)}",
        "",
        "## Cost",
        "",
        f"- total validation sequences tested: {totals.get('validation_sequences_tested', 0)}",
        f"- validation opt invocations: {totals.get('validation_opt_invocations', 0)}",
        f"- validation pass invocations baseline: {totals.get('validation_pass_invocations_baseline', 0)}",
        f"- validation pass invocations actual: {totals.get('validation_pass_invocations_actual', 0)}",
        f"- validation pass invocations saved: {totals.get('validation_pass_invocations_saved', 0)}",
        f"- profile reuse hits: {totals.get('validation_profile_reuse_hits', 0)}",
        f"- state transition cache hits: {totals.get('validation_state_transition_cache_hits', 0)}",
        f"- state equivalence cache hits: {totals.get('validation_state_equivalence_cache_hits', 0)}",
        f"- validation time ms: {_format_ms(validation_time)}",
        "",
        "## Permutation DAG Validation",
        "",
        f"- total DAG validated batches: {len(dag_rows)}",
        f"- DAG certified batches: {sum(1 for row in dag_rows if row.get('validation_tier') == 'permutation_dag_exact')}",
        f"- DAG mismatches: {sum(1 for row in dag_rows if row.get('validation_tier') == 'permutation_dag_mismatch')}",
        f"- DAG incomplete: {sum(1 for row in dag_rows if row.get('validation_tier') == 'permutation_dag_incomplete')}",
        f"- total DAG nodes: {sum(_to_int(row.get('dag_nodes')) for row in dag_rows)}",
        f"- total DAG edges: {sum(_to_int(row.get('dag_edges')) for row in dag_rows)}",
        f"- total hash merges: {sum(_to_int(row.get('hash_merges')) for row in dag_rows)}",
        f"- total structural merges: {sum(_to_int(row.get('structural_merges')) for row in dag_rows)}",
        f"- total transition cache hits: {sum(_to_int(row.get('transition_cache_hits')) for row in dag_rows)}",
        f"- total equivalence cache hits: {sum(_to_int(row.get('equivalence_cache_hits')) for row in dag_rows)}",
        f"- avg compression_vs_permutation: {_format_float(_avg(dag_compressions))}",
        "",
    ]
    lines.extend(
        _markdown_table(
            ["program", "state", "batch", "size", "K! log10", "DAG nodes", "DAG edges", "final classes", "tier", "compression vs K!"],
            [
                [
                    row.get("program", ""),
                    row.get("state_id", ""),
                    row.get("batch_id", ""),
                    row.get("batch_size", ""),
                    row.get("factorial_permutations_log10", ""),
                    row.get("dag_nodes", ""),
                    row.get("dag_edges", ""),
                    row.get("final_equivalence_classes", ""),
                    row.get("validation_tier", ""),
                    row.get("compression_vs_permutation", ""),
                ]
                for row in dag_rows[:50]
            ],
        )
    )
    lines.extend([
        "",
        "## Correctness Boundary",
        "",
        BOUNDARY_TEXT,
        "",
        DAG_BOUNDARY_TEXT,
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


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


def _to_int(value: object) -> int:
    try:
        return int(float(str(value).strip() or "0"))
    except ValueError:
        return 0


def _to_float(value: object) -> float:
    try:
        return float(str(value).strip() or "0")
    except ValueError:
        return 0.0


def _format_ms(value: float) -> str:
    return f"{value:.3f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["none"]
    lines = [f"| {' | '.join(headers)} |", f"| {' | '.join(['---'] * len(headers))} |"]
    lines.extend(f"| {' | '.join(row)} |" for row in rows)
    return lines


def _format_float(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".") if value else "0"


def _avg(values: list[float]) -> float:
    clean = [value for value in values if value]
    return sum(clean) / len(clean) if clean else 0.0


def _is_dag_row(row: dict) -> bool:
    return row.get("validation_tier", "").startswith("permutation_dag_") or _to_int(row.get("validation_dag_nodes")) > 0
