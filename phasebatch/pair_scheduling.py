from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from .schema import PAIR_SCHEDULING_SUMMARY_FIELDS


BOUNDARY_TEXT = "Lazy pair testing can reduce cost but cannot create commute evidence. Untested pairs are treated as unknown/conflict."


def write_pair_scheduling_summary(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    state_dirs = _state_dirs(run_dir)
    rows = [_summary_row(state_dir) for state_dir in state_dirs if (state_dir / "pair_relation.csv").exists()]
    _write_csv(run_dir / "pair_scheduling_summary.csv", PAIR_SCHEDULING_SUMMARY_FIELDS, rows)
    _write_markdown(run_dir / "pair_scheduling_summary.md", rows)
    return {
        "pair_scheduling_summary_csv": str(run_dir / "pair_scheduling_summary.csv"),
        "pair_scheduling_summary_md": str(run_dir / "pair_scheduling_summary.md"),
    }


def _state_dirs(run_dir: Path) -> list[Path]:
    if (run_dir / "states").exists():
        return sorted(path for path in (run_dir / "states").iterdir() if path.is_dir())
    return [run_dir]


def _summary_row(state_dir: Path) -> dict:
    pairs = _read_csv(state_dir / "pair_relation.csv")
    state_summary = _first_row(state_dir / "per_state_summary.csv")
    metadata = _read_json(state_dir / "metadata.json")
    first = pairs[0] if pairs else state_summary
    relation_counts = Counter(row.get("dynamic_relation", "") for row in pairs)
    modes = sorted({row.get("pair_testing_mode", "") for row in pairs if row.get("pair_testing_mode")})
    tested_pairs = [row for row in pairs if row.get("dynamic_relation") != "not_tested"]
    skipped_pairs = [row for row in pairs if _is_true(row.get("skipped_by_budget")) or row.get("failure_kind") == "lazy_budget"]
    cache_hits = [row for row in tested_pairs if _is_true(row.get("cache_hit"))]
    cache_misses = [row for row in tested_pairs if row.get("cache_hit") == "false"]
    pair_test_budget = str(metadata.get("pair_test_budget_per_state", ""))
    if not pair_test_budget and skipped_pairs:
        pair_test_budget = str(len(tested_pairs))
    return {
        "program": first.get("program", ""),
        "state_id": first.get("state_id", state_dir.name),
        "depth": state_summary.get("depth", first.get("depth", "")),
        "active_passes": state_summary.get("active_passes", ""),
        "total_pairs": str(len(pairs)),
        "tested_pairs": str(len(tested_pairs)),
        "skipped_pairs": str(len(skipped_pairs)),
        "pair_test_budget": pair_test_budget,
        "pair_testing_mode": ";".join(modes) if modes else str(metadata.get("pair_testing_mode", "")),
        "commute_pairs": str(relation_counts.get("dynamic_commute", 0)),
        "order_sensitive_pairs": str(relation_counts.get("dynamic_order_sensitive", 0)),
        "unknown_pairs": str(_unknown_count(pairs)),
        "cache_hits": str(len(cache_hits)),
        "cache_misses": str(len(cache_misses)),
    }


def _write_markdown(path: Path, rows: list[dict]) -> None:
    total_pairs = sum(_to_int(row.get("total_pairs")) for row in rows)
    tested_pairs = sum(_to_int(row.get("tested_pairs")) for row in rows)
    skipped_pairs = sum(_to_int(row.get("skipped_pairs")) for row in rows)
    unknown_pairs = sum(_to_int(row.get("unknown_pairs")) for row in rows)
    cache_hits = sum(_to_int(row.get("cache_hits")) for row in rows)
    cache_misses = sum(_to_int(row.get("cache_misses")) for row in rows)
    modes = sorted({row.get("pair_testing_mode", "") for row in rows if row.get("pair_testing_mode")})
    cache_total = cache_hits + cache_misses
    cache_hit_rate = (cache_hits / cache_total * 100.0) if cache_total else 0.0
    lines = [
        "# Pair Scheduling Summary",
        "",
        "## Overall",
        "",
        f"- pair testing mode: {';'.join(modes) if modes else ''}",
        f"- total pairs: {total_pairs}",
        f"- tested pairs: {tested_pairs}",
        f"- skipped pairs: {skipped_pairs}",
        f"- unknown pairs from budget: {skipped_pairs}",
        f"- cache hit rate: {cache_hit_rate:.2f}%",
        "",
        "## By State",
        "",
        "| state | depth | total pairs | tested | skipped | unknown | budget |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {state_id} | {depth} | {total_pairs} | {tested_pairs} | {skipped_pairs} | {unknown_pairs} | {pair_test_budget} |".format(
                **row
            )
        )
    lines.extend(["", "## Correctness Boundary", "", BOUNDARY_TEXT, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _unknown_count(rows: list[dict]) -> int:
    return sum(
        1
        for row in rows
        if row.get("dynamic_relation") in {"not_tested", "dynamic_timeout", "dynamic_failed"}
        or row.get("final_relation") == "final_unknown"
    )


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


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


def _to_int(value: object) -> int:
    try:
        return int(float(str(value).strip() or "0"))
    except ValueError:
        return 0
