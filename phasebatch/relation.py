from __future__ import annotations

import csv
from pathlib import Path

from .schema import PAIR_RELATION_FIELDS


def static_relation(profile_a: dict, profile_b: dict) -> dict:
    funcs_a = _split(profile_a.get("changed_functions"))
    funcs_b = _split(profile_b.get("changed_functions"))
    blocks_a = _split(profile_a.get("changed_blocks"))
    blocks_b = _split(profile_b.get("changed_blocks"))
    overlap_functions = funcs_a & funcs_b
    overlap_blocks = blocks_a & blocks_b

    if funcs_a and funcs_b and not overlap_functions:
        relation = "static_disjoint_function"
    elif blocks_a and blocks_b and not overlap_blocks:
        relation = "static_disjoint_block"
    elif overlap_blocks:
        relation = "static_overlap_block"
    elif overlap_functions:
        relation = "static_overlap_function"
    else:
        relation = "static_unknown"

    return {
        "static_relation": relation,
        "overlap_functions": len(overlap_functions),
        "overlap_blocks": len(overlap_blocks),
    }


def final_relation(pair_row: dict) -> str:
    dynamic = pair_row.get("dynamic_relation")
    if dynamic == "dynamic_commute":
        return "final_commute"
    if dynamic == "dynamic_order_sensitive":
        return "final_order_sensitive"
    return "final_unknown"


def annotate_pair_relations(pair_rows: list[dict], profiles: dict[str, dict]) -> list[dict]:
    annotated: list[dict] = []
    for row in pair_rows:
        profile_a = profiles.get(row["pass_a"], {})
        profile_b = profiles.get(row["pass_b"], {})
        static = static_relation(profile_a, profile_b)
        new_row = dict(row)
        new_row.update(static)
        new_row["final_relation"] = final_relation(new_row)
        annotated.append(new_row)
    return annotated


def write_pair_relations(path: Path, pair_rows: list[dict]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_RELATION_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(pair_rows)


def _split(value: object) -> set[str]:
    if not value:
        return set()
    return {item for item in str(value).split(";") if item}
