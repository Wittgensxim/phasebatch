from __future__ import annotations

import ast
import csv
import itertools
from dataclasses import dataclass
from pathlib import Path

from .schema import AGGREGATE_OVERLAP_SUMMARY_FIELDS, FOOTPRINT_OVERLAP_FIELDS


OVERLAP_KINDS = [
    "disjoint_write",
    "same_function_overlap",
    "same_block_overlap",
    "possible_ww_overlap",
    "unknown_overlap",
]


@dataclass(frozen=True)
class _ParsedSet:
    values: set[str]
    exact_known: bool


@dataclass(frozen=True)
class _PassFootprint:
    row: dict
    functions: _ParsedSet
    blocks: _ParsedSet
    changed_ir: bool


def parse_set_field(value: str) -> set[str]:
    return _parse_set_field(value, has_column=True).values


def build_footprint_overlap(state_dir: Path) -> list[dict]:
    state_dir = Path(state_dir)
    profiles = _active_profiles(state_dir / "pass_profile.csv")
    pair_map = _pair_relation_map(state_dir / "pair_relation.csv")
    program, state_id, state_hash = _state_identity(profiles, _read_csv(state_dir / "pair_relation.csv"))

    footprints = [_footprint(row) for row in profiles]
    rows = []
    for left, right in itertools.combinations(footprints, 2):
        pass_a = left.row.get("pass", "")
        pass_b = right.row.get("pass", "")
        relation = pair_map.get(frozenset([pass_a, pass_b]), {})
        func_overlap = left.functions.values & right.functions.values
        block_overlap = left.blocks.values & right.blocks.values
        overlap_kind = _classify_overlap(left, right, func_overlap, block_overlap)
        rows.append(
            {
                "program": left.row.get("program", program),
                "state_id": left.row.get("state_id", state_id),
                "state_hash": left.row.get("state_hash", state_hash),
                "pass_a": pass_a,
                "pass_b": pass_b,
                "pass_a_changed_functions": _join_set(left.functions.values),
                "pass_b_changed_functions": _join_set(right.functions.values),
                "pass_a_changed_blocks": _join_set(left.blocks.values),
                "pass_b_changed_blocks": _join_set(right.blocks.values),
                "write_func_overlap": str(len(func_overlap)),
                "write_block_overlap": str(len(block_overlap)),
                "same_function": _bool(bool(func_overlap)),
                "same_block": _bool(bool(block_overlap)),
                "overlap_kind": overlap_kind,
                "dynamic_relation": relation.get("dynamic_relation") or "not_tested",
                "final_relation": relation.get("final_relation") or "unknown",
            }
        )

    _write_csv(state_dir / "footprint_overlap.csv", FOOTPRINT_OVERLAP_FIELDS, rows)
    return rows


def aggregate_overlap_summary(out_dir: Path, program: str) -> list[dict]:
    out_dir = Path(out_dir)
    state_rows = _read_csv(out_dir / "states.csv")
    buckets: dict[int, dict] = {}
    seen_dirs: set[str] = set()

    for state in state_rows:
        if _is_true(state.get("is_duplicate")):
            continue
        state_dir_value = state.get("state_dir", "")
        if not state_dir_value:
            continue
        state_dir = Path(state_dir_value)
        state_dir_key = _safe_resolved_key(state_dir)
        if state_dir_key in seen_dirs:
            continue
        seen_dirs.add(state_dir_key)

        depth = _to_int(state.get("depth"))
        bucket = _overlap_bucket(buckets, depth)
        bucket["states"] += 1
        for row in _read_csv(state_dir / "footprint_overlap.csv"):
            _add_overlap_row(bucket, row)

    rows = []
    for depth in sorted(buckets):
        bucket = buckets[depth]
        rows.append(
            {
                "program": program,
                "depth": str(depth),
                "states": str(bucket["states"]),
                "total_pairs": str(bucket["total_pairs"]),
                "disjoint_write": str(bucket["disjoint_write"]),
                "same_function_overlap": str(bucket["same_function_overlap"]),
                "same_block_overlap": str(bucket["same_block_overlap"]),
                "possible_ww_overlap": str(bucket["possible_ww_overlap"]),
                "unknown_overlap": str(bucket["unknown_overlap"]),
                "disjoint_and_commute": str(bucket["disjoint_and_commute"]),
                "overlap_and_commute": str(bucket["overlap_and_commute"]),
                "overlap_and_order_sensitive": str(bucket["overlap_and_order_sensitive"]),
            }
        )
    return rows


def write_aggregate_overlap_summary(out_dir: Path, program: str) -> list[dict]:
    rows = aggregate_overlap_summary(out_dir, program)
    _write_csv(Path(out_dir) / "aggregate_overlap_summary.csv", AGGREGATE_OVERLAP_SUMMARY_FIELDS, rows)
    return rows


def _active_profiles(path: Path) -> list[dict]:
    rows = []
    seen: set[str] = set()
    for row in _read_csv(path):
        pass_name = row.get("pass", "")
        if not pass_name or pass_name in seen:
            continue
        if _is_true(row.get("success")) and _is_true(row.get("active")):
            rows.append(row)
            seen.add(pass_name)
    return rows


def _pair_relation_map(path: Path) -> dict[frozenset[str], dict]:
    relations = {}
    for row in _read_csv(path):
        pass_a = row.get("pass_a", "")
        pass_b = row.get("pass_b", "")
        if pass_a and pass_b:
            relations[frozenset([pass_a, pass_b])] = row
    return relations


def _footprint(row: dict) -> _PassFootprint:
    functions = _parse_set_field(row.get("changed_functions", ""), has_column="changed_functions" in row)
    blocks = _parse_set_field(row.get("changed_blocks", ""), has_column="changed_blocks" in row)
    changed_ir = (
        _is_true(row.get("active"))
        or _to_int(row.get("funcs_changed")) > 0
        or _to_int(row.get("blocks_changed")) > 0
        or bool(functions.values)
        or bool(blocks.values)
    )
    return _PassFootprint(row=row, functions=functions, blocks=blocks, changed_ir=changed_ir)


def _classify_overlap(
    left: _PassFootprint,
    right: _PassFootprint,
    func_overlap: set[str],
    block_overlap: set[str],
) -> str:
    blocks_exact = left.blocks.exact_known and right.blocks.exact_known
    all_exact = blocks_exact and left.functions.exact_known and right.functions.exact_known

    if block_overlap:
        return "same_block_overlap"
    if func_overlap and blocks_exact:
        return "same_function_overlap"
    if all_exact and not func_overlap and not block_overlap:
        return "disjoint_write"
    return "unknown_overlap"


def _parse_set_field(value: object, *, has_column: bool) -> _ParsedSet:
    if not has_column:
        return _ParsedSet(set(), False)
    text = str(value or "").strip()
    if not text:
        return _ParsedSet(set(), False)
    if text == "[]":
        return _ParsedSet(set(), True)
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return _ParsedSet(set(), False)
        if not isinstance(parsed, (list, tuple, set)):
            return _ParsedSet(set(), False)
        return _ParsedSet(_clean_items(parsed), True)

    raw_parts = []
    for semi_part in text.split(";"):
        raw_parts.extend(semi_part.split(","))
    return _ParsedSet(_clean_items(raw_parts), True)


def _clean_items(values) -> set[str]:
    cleaned = set()
    for value in values:
        text = str(value).strip().strip("'\"")
        if text:
            cleaned.add(text)
    return cleaned


def _state_identity(profile_rows: list[dict], pair_rows: list[dict]) -> tuple[str, str, str]:
    row = profile_rows[0] if profile_rows else pair_rows[0] if pair_rows else {}
    return row.get("program", ""), row.get("state_id", ""), row.get("state_hash", "")


def _overlap_bucket(buckets: dict[int, dict], depth: int) -> dict:
    if depth not in buckets:
        buckets[depth] = {
            "states": 0,
            "total_pairs": 0,
            "disjoint_and_commute": 0,
            "overlap_and_commute": 0,
            "overlap_and_order_sensitive": 0,
            **{kind: 0 for kind in OVERLAP_KINDS},
        }
    return buckets[depth]


def _add_overlap_row(bucket: dict, row: dict) -> None:
    kind = row.get("overlap_kind", "") or "unknown_overlap"
    final_relation = row.get("final_relation", "") or "unknown"
    bucket["total_pairs"] += 1
    if kind in OVERLAP_KINDS:
        bucket[kind] += 1
    else:
        bucket["unknown_overlap"] += 1

    if kind == "disjoint_write" and final_relation == "final_commute":
        bucket["disjoint_and_commute"] += 1
    if _is_overlap_kind(kind) and final_relation == "final_commute":
        bucket["overlap_and_commute"] += 1
    if _is_overlap_kind(kind) and final_relation == "final_order_sensitive":
        bucket["overlap_and_order_sensitive"] += 1


def _is_overlap_kind(kind: str) -> bool:
    return kind in {"same_function_overlap", "same_block_overlap", "possible_ww_overlap"}


def _safe_resolved_key(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _join_set(values: set[str]) -> str:
    return ";".join(sorted(values))


def _to_int(value: object) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def _is_true(value: object) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def _bool(value: bool) -> str:
    return "true" if value else "false"


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
