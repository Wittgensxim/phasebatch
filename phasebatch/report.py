from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from .schema import CLUSTER_DISTRIBUTION_FIELDS, PAIR_RELATION_FIELDS, PASS_PROFILE_FIELDS, PER_STATE_SUMMARY_FIELDS


def write_summary(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    metadata = _read_json(out_dir / "metadata.json")
    profiles = _read_csv(out_dir / "pass_profile.csv")
    pairs = _read_csv(out_dir / "pair_relation.csv")
    clusters = _read_csv(out_dir / "cluster_distribution.csv")
    invalid = _read_csv(out_dir / "invalid_passes.csv")

    active = [row for row in profiles if _is_true(row.get("active"))]
    dormant = [row for row in profiles if row.get("success") == "true" and not _is_true(row.get("active"))]
    relation_counts = Counter(row.get("dynamic_relation", "") for row in pairs)
    final_counts = Counter(row.get("final_relation", "") for row in pairs)
    noncommute = next((row for row in clusters if row.get("graph_type") == "noncommute_graph"), {})

    lines = [
        "# Summary",
        "",
        f"- input: {metadata.get('input', '')}",
        f"- LLVM version: {_llvm_version(metadata)}",
        f"- valid pass count: {len(profiles)}",
        f"- active passes: {len(active)}",
        f"- dormant passes: {len(dormant)}",
        f"- pair rows: {len(pairs)}",
        f"- dynamic commute: {relation_counts.get('dynamic_commute', 0)}",
        f"- order-sensitive: {relation_counts.get('dynamic_order_sensitive', 0)}",
        f"- unknown/failed/not-tested: {_unknown_count(relation_counts)}",
        f"- max conflict component: {noncommute.get('max_size', 0)}",
        f"- median conflict component: {noncommute.get('median_size', 0)}",
        "",
        "# Relation Counts",
        "",
        "| relation | count |",
        "| --- | ---: |",
    ]
    for name, count in sorted(relation_counts.items()):
        if name:
            lines.append(f"| {name} | {count} |")
    for name, count in sorted(final_counts.items()):
        if name:
            lines.append(f"| {name} | {count} |")

    lines.extend(["", "# Top active passes", "", "| pass | inst_delta | blocks_changed | time_ms |", "| --- | ---: | ---: | ---: |"])
    for row in sorted(active, key=lambda item: int(item.get("blocks_changed") or 0), reverse=True)[:10]:
        lines.append(f"| {row.get('pass')} | {row.get('inst_delta')} | {row.get('blocks_changed')} | {row.get('time_ms')} |")

    lines.extend(["", "# Dynamic commute pairs", "", "| pass_a | pass_b | time_ms |", "| --- | --- | ---: |"])
    for row in [row for row in pairs if row.get("dynamic_relation") == "dynamic_commute"][:20]:
        lines.append(f"| {row.get('pass_a')} | {row.get('pass_b')} | {row.get('time_ms')} |")

    lines.extend(["", "# Order-sensitive pairs", "", "| pass_a | pass_b | time_ms |", "| --- | --- | ---: |"])
    for row in [row for row in pairs if row.get("dynamic_relation") == "dynamic_order_sensitive"][:20]:
        lines.append(f"| {row.get('pass_a')} | {row.get('pass_b')} | {row.get('time_ms')} |")

    lines.extend(["", "# Largest conflict components", "", "| graph_type | max_size | median_size |", "| --- | ---: | ---: |"])
    for row in clusters:
        lines.append(f"| {row.get('graph_type')} | {row.get('max_size')} | {row.get('median_size')} |")

    lines.extend(["", "# Invalid passes", "", "| pass | reason |", "| --- | --- |"])
    for row in invalid:
        lines.append(f"| {row.get('pass')} | {row.get('reason')} |")

    lines.extend(
        [
            "",
            "# Generated files",
            "",
            "- metadata.json",
            "- valid_passes.csv",
            "- invalid_passes.csv",
            "- pass_profile.csv",
            "- pair_relation.csv",
            "- cluster_distribution.csv",
            "- per_state_summary.csv",
            "",
            "# Caveats",
            "",
            "- This MVP uses coarse pass-level effects.",
            "- Dynamic AB/BA equality is state-local evidence, not global phase-ordering optimality.",
            "- Static disjointness is reported as candidate evidence only.",
        ]
    )

    path = out_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_per_state_summary(
    out_dir: Path,
    program: str,
    state_hash: str,
    state_id: str,
    depth: int,
    parent_state_id: str,
    transition_pass: str,
    pass_set_size: int,
    valid_passes: int,
    invalid_passes: int,
    profile_time_ms: float,
    pair_time_ms: float,
    total_time_ms: float,
) -> Path:
    out_dir = Path(out_dir)
    profiles = _read_csv(out_dir / "pass_profile.csv")
    pairs = _read_csv(out_dir / "pair_relation.csv")
    clusters = _read_csv(out_dir / "cluster_distribution.csv")
    relation_counts = Counter(row.get("dynamic_relation", "") for row in pairs)
    static_counts = Counter(row.get("static_relation", "") for row in pairs)
    noncommute = next((row for row in clusters if row.get("graph_type") == "noncommute_graph"), {})
    active_count = sum(1 for row in profiles if _is_true(row.get("active")))
    dormant_count = sum(1 for row in profiles if row.get("success") == "true" and not _is_true(row.get("active")))
    row = {
        "program": program,
        "state_id": state_id,
        "depth": depth,
        "parent_state_id": parent_state_id,
        "transition_pass": transition_pass,
        "state_hash": state_hash,
        "pass_set_size": pass_set_size,
        "valid_passes": valid_passes,
        "invalid_passes": invalid_passes,
        "active_passes": active_count,
        "dormant_passes": dormant_count,
        "total_pairs": len(pairs),
        "pairs_tested": sum(1 for pair in pairs if pair.get("dynamic_relation") != "not_tested"),
        "dynamic_commute": relation_counts.get("dynamic_commute", 0),
        "order_sensitive": relation_counts.get("dynamic_order_sensitive", 0),
        "unknown": relation_counts.get("dynamic_timeout", 0) + relation_counts.get("not_tested", 0),
        "failed": relation_counts.get("dynamic_failed", 0),
        "static_disjoint_function": static_counts.get("static_disjoint_function", 0),
        "static_disjoint_block": static_counts.get("static_disjoint_block", 0),
        "max_conflict_component": noncommute.get("max_size", 0),
        "median_conflict_component": noncommute.get("median_size", 0),
        "profile_time_ms": f"{profile_time_ms:.3f}",
        "pair_time_ms": f"{pair_time_ms:.3f}",
        "total_time_ms": f"{total_time_ms:.3f}",
    }
    path = out_dir / "per_state_summary.csv"
    _write_csv(path, PER_STATE_SUMMARY_FIELDS, [row])
    return path


def write_aggregate_report(out_dir: Path, program_dirs: list[Path]) -> Path:
    out_dir = Path(out_dir)
    aggregate_files = {
        "pass_profile.csv": PASS_PROFILE_FIELDS,
        "pair_relation.csv": PAIR_RELATION_FIELDS,
        "per_state_summary.csv": PER_STATE_SUMMARY_FIELDS,
        "cluster_distribution.csv": CLUSTER_DISTRIBUTION_FIELDS,
    }
    for filename, fields in aggregate_files.items():
        rows: list[dict] = []
        for program_dir in program_dirs:
            rows.extend(_read_csv(program_dir / filename))
        _write_csv(out_dir / filename, fields, rows)

    summaries = _read_csv(out_dir / "per_state_summary.csv")
    lines = [
        "# Aggregate Summary",
        "",
        "| program | valid passes | active passes | tested pairs | commute | order-sensitive | unknown | max component | time ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            "| {program} | {valid_passes} | {active_passes} | {pairs_tested} | {dynamic_commute} | "
            "{order_sensitive} | {unknown} | {max_conflict_component} | {total_time_ms} |".format(**row)
        )
    path = out_dir / "aggregate_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def _llvm_version(metadata: dict) -> str:
    opt = metadata.get("tools", {}).get("opt", {})
    lines = str(opt.get("version", "")).splitlines() if opt else []
    for line in lines:
        if "LLVM version" in line:
            return line.strip()
    return lines[0].strip() if lines else ""


def _unknown_count(counter: Counter) -> int:
    return counter.get("dynamic_failed", 0) + counter.get("dynamic_timeout", 0) + counter.get("not_tested", 0)


def _is_true(value: object) -> bool:
    return str(value).lower() in {"true", "1", "yes"}
