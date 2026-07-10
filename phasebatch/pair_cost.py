from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from .schema import PAIR_COST_SUMMARY_FIELDS


INTERPRETATION = "Pair-result memoization is keyed by canonical IR state and pass identity. It is not reused across different IR states."


def write_pair_cost_summary(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    files = _pair_relation_files(run_dir)
    rows = _pair_rows(files)
    summary_row = _summary_row(run_dir, rows, state_count=len(files))
    _write_csv(run_dir / "pair_cost_summary.csv", PAIR_COST_SUMMARY_FIELDS, [summary_row])
    _write_markdown(run_dir / "pair_cost_summary.md", summary_row, rows)
    return {
        "pair_cost_summary_csv": str(run_dir / "pair_cost_summary.csv"),
        "pair_cost_summary_md": str(run_dir / "pair_cost_summary.md"),
    }


def _pair_relation_files(run_dir: Path) -> list[Path]:
    state_files = sorted((run_dir / "states").rglob("pair_relation.csv")) if (run_dir / "states").exists() else []
    return state_files or ([run_dir / "pair_relation.csv"] if (run_dir / "pair_relation.csv").exists() else [])


def _pair_rows(files: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in files:
        rows.extend(_read_csv(path))
    return rows


def _summary_row(run_dir: Path, rows: list[dict], *, state_count: int) -> dict:
    hits = sum(1 for row in rows if _is_true(row.get("cache_hit")))
    pair_rows = len(rows)
    states = state_count or _state_count(rows)
    llvm_diff_time = _sum_optional(rows, "llvm_diff_time_ms")
    comparator_time = _sum_optional(rows, "comparator_time_ms")
    pass_invocations = [_pass_invocations(row) for row in rows]
    return {
        "program": _program(run_dir, rows),
        "states": str(states),
        "pair_rows": str(pair_rows),
        "cache_hits": str(hits),
        "cache_misses": str(pair_rows - hits),
        "cache_hit_rate": _format_rate(hits, pair_rows),
        "pair_test_opt_runs": str(sum(_to_int(row.get("pair_test_opt_runs")) for row in rows)),
        "avoided_opt_runs": str(sum(_to_int(row.get("avoided_opt_runs")) for row in rows)),
        "reused_single_pass_pairs": str(sum(1 for row in rows if _is_true(row.get("reused_single_pass_outputs")))),
        "pair_test_pass_invocations_baseline": str(sum(item[0] for item in pass_invocations)),
        "pair_test_pass_invocations_actual": str(sum(item[1] for item in pass_invocations)),
        "pair_test_pass_invocations_saved": str(sum(item[2] for item in pass_invocations)),
        "pair_test_time_ms": _format_ms(sum(_to_float(row.get("pair_test_time_ms") or row.get("time_ms")) for row in rows)),
        "llvm_diff_time_ms": "" if llvm_diff_time is None else _format_ms(llvm_diff_time),
        "comparator_time_ms": "" if comparator_time is None else _format_ms(comparator_time),
    }


def _write_markdown(path: Path, summary: dict, rows: list[dict]) -> None:
    lines = [
        "# Pair Detection Cost Summary",
        "",
        "## Overall",
        "",
        f"- states: {summary['states']}",
        f"- pair rows: {summary['pair_rows']}",
        f"- cache hits: {summary['cache_hits']}",
        f"- cache misses: {summary['cache_misses']}",
        f"- cache hit rate: {summary['cache_hit_rate']}",
        f"- opt runs avoided: {summary['avoided_opt_runs']}",
        f"- pass invocations saved: {summary['pair_test_pass_invocations_saved']}",
        f"- pair test time: {summary['pair_test_time_ms']} ms",
        "",
        "## By Depth",
        "",
        "| depth | pair rows | hits | misses | hit rate | time ms |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for depth, depth_rows in _rows_by_depth(rows).items():
        hits = sum(1 for row in depth_rows if _is_true(row.get("cache_hit")))
        count = len(depth_rows)
        time_ms = sum(_to_float(row.get("pair_test_time_ms") or row.get("time_ms")) for row in depth_rows)
        lines.append(f"| {depth} | {count} | {hits} | {count - hits} | {_format_rate(hits, count)} | {_format_ms(time_ms)} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            INTERPRETATION,
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _rows_by_depth(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("depth", ""))].append(row)
    return dict(sorted(grouped.items(), key=lambda item: (_to_int(item[0]) if str(item[0]).isdigit() else 10**9, str(item[0]))))


def _state_count(rows: list[dict]) -> int:
    identities = {
        (
            row.get("program", ""),
            row.get("state_id", ""),
            row.get("state_hash", ""),
            row.get("depth", ""),
        )
        for row in rows
    }
    identities.discard(("", "", "", ""))
    if identities:
        return len(identities)
    return 1 if rows else 0


def _program(run_dir: Path, rows: list[dict]) -> str:
    counts = Counter(row.get("program", "") for row in rows if row.get("program"))
    if counts:
        return counts.most_common(1)[0][0]
    return run_dir.name


def _sum_optional(rows: list[dict], field: str) -> float | None:
    values = [row.get(field, "") for row in rows]
    numeric = [_to_float(value) for value in values if str(value).strip() != ""]
    return sum(numeric) if numeric else None


def _pass_invocations(row: dict) -> tuple[int, int, int]:
    baseline = row.get("pair_test_pass_invocations_baseline")
    actual = row.get("pair_test_pass_invocations_actual")
    saved = row.get("pair_test_pass_invocations_saved")
    if str(baseline or actual or saved).strip():
        return _to_int(baseline), _to_int(actual), _to_int(saved)
    if row.get("dynamic_relation") == "not_tested" or row.get("failure_kind") == "max_pairs":
        return 0, 0, 0
    if _is_true(row.get("cache_hit")):
        return 4, 0, 4
    if _is_true(row.get("reused_single_pass_outputs")):
        return 4, 2, 2
    return 4, 4, 0


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


def _format_rate(numerator: int, denominator: int) -> str:
    return f"{(numerator / denominator if denominator else 0.0):.4f}"


def _format_ms(value: float) -> str:
    return f"{value:.3f}"


def _is_true(value: object) -> bool:
    return str(value).lower() in {"true", "1", "yes"}
