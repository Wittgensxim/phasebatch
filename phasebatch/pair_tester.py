from __future__ import annotations

import csv
import itertools
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .normalizer import canonical_hash, count_ir_features
from .pass_config import PassRegistry
from .runner import run_opt
from .schema import PAIR_RELATION_FIELDS


def run_pair_tests(
    input_ll: Path,
    active_profiles: list[dict],
    tools: dict,
    out_dir: Path,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
    pass_registry: PassRegistry | None = None,
) -> list[dict]:
    out_dir = Path(out_dir)
    pair_dir = out_dir / "artifacts" / "pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    active_profiles = [row for row in active_profiles if _is_true(row.get("active"))]
    profiles = {row["pass"]: row for row in active_profiles}
    pairs = _ordered_pairs(active_profiles)
    tested_pairs = pairs if max_pairs is None else pairs[: max(0, max_pairs)]
    skipped_pairs = [] if max_pairs is None else pairs[max(0, max_pairs) :]

    def run_one(pair: tuple[str, str]) -> dict:
        pass_a, pass_b = pair
        profile_a = profiles[pass_a]
        profile_b = profiles[pass_b]
        safe = f"{_safe_name(pass_a)}__{_safe_name(pass_b)}"
        current_dir = pair_dir / safe
        current_dir.mkdir(parents=True, exist_ok=True)
        ab_path = current_dir / "ab.ll"
        ba_path = current_dir / "ba.ll"
        pipeline_a = _pipeline_for(pass_a, pass_registry)
        pipeline_b = _pipeline_for(pass_b, pass_registry)
        ab = run_opt(str(tools["opt"]), input_ll, [pipeline_a, pipeline_b], ab_path, timeout)
        ba = run_opt(str(tools["opt"]), input_ll, [pipeline_b, pipeline_a], ba_path, timeout)
        row = _base_row(input_ll, profile_a, profile_b, ab_path, ba_path)
        row["ab_success"] = _bool(ab.success)
        row["ba_success"] = _bool(ba.success)
        row["time_ms"] = f"{ab.time_ms + ba.time_ms:.3f}"

        if ab.success and ba.success and ab_path.exists() and ba_path.exists():
            ab_hash = canonical_hash(ab_path)
            ba_hash = canonical_hash(ba_path)
            same_hash = ab_hash == ba_hash
            ab_inst = count_ir_features(ab_path).get("instructions", 0)
            ba_inst = count_ir_features(ba_path).get("instructions", 0)
            row.update(
                {
                    "dynamic_relation": "dynamic_commute" if same_hash else "dynamic_order_sensitive",
                    "ab_hash": ab_hash,
                    "ba_hash": ba_hash,
                    "same_hash": _bool(same_hash),
                    "ab_inst": ab_inst,
                    "ba_inst": ba_inst,
                    "inst_delta_ab_ba": ab_inst - ba_inst,
                    "failure_kind": "",
                }
            )
        else:
            relation = "dynamic_timeout" if ab.timed_out or ba.timed_out else "dynamic_failed"
            failure = "timeout" if relation == "dynamic_timeout" else "failed"
            row.update({"dynamic_relation": relation, "same_hash": "false", "failure_kind": failure})
        return row

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as executor:
        rows = list(executor.map(run_one, tested_pairs))

    for pass_a, pass_b in skipped_pairs:
        row = _base_row(input_ll, profiles[pass_a], profiles[pass_b], Path(""), Path(""))
        row.update(
            {
                "dynamic_relation": "not_tested",
                "ab_success": "false",
                "ba_success": "false",
                "same_hash": "",
                "failure_kind": "max_pairs",
            }
        )
        rows.append(row)

    _write_csv(out_dir / "pair_relation.csv", PAIR_RELATION_FIELDS, rows)
    return rows


def _ordered_pairs(active_profiles: list[dict]) -> list[tuple[str, str]]:
    sorted_profiles = sorted(active_profiles, key=lambda row: row["pass"])

    def priority(pair: tuple[dict, dict]) -> tuple[int, str, str]:
        a, b = pair
        overlap = bool(_split(a.get("changed_functions")) & _split(b.get("changed_functions")))
        return (0 if overlap else 1, a["pass"], b["pass"])

    ordered = sorted(itertools.combinations(sorted_profiles, 2), key=priority)
    return [(a["pass"], b["pass"]) for a, b in ordered]


def _base_row(input_ll: Path, profile_a: dict, profile_b: dict, ab_path: Path, ba_path: Path) -> dict:
    return {
        "program": profile_a.get("program") or Path(input_ll).parent.name or Path(input_ll).stem,
        "state_id": profile_a.get("state_id", ""),
        "depth": profile_a.get("depth", ""),
        "parent_state_id": profile_a.get("parent_state_id", ""),
        "transition_pass": profile_a.get("transition_pass", ""),
        "state_hash": profile_a.get("state_hash", ""),
        "pass_a": profile_a["pass"],
        "pass_b": profile_b["pass"],
        "a_active": profile_a.get("active", "true"),
        "b_active": profile_b.get("active", "true"),
        "static_relation": "",
        "dynamic_relation": "",
        "final_relation": "",
        "ab_success": "",
        "ba_success": "",
        "ab_hash": "",
        "ba_hash": "",
        "same_hash": "",
        "ab_inst": "",
        "ba_inst": "",
        "inst_delta_ab_ba": "",
        "changed_funcs_a": profile_a.get("changed_functions", ""),
        "changed_funcs_b": profile_b.get("changed_functions", ""),
        "changed_blocks_a": profile_a.get("changed_blocks", ""),
        "changed_blocks_b": profile_b.get("changed_blocks", ""),
        "overlap_functions": "",
        "overlap_blocks": "",
        "time_ms": "",
        "failure_kind": "",
        "ab_path": str(ab_path) if str(ab_path) != "." else "",
        "ba_path": str(ba_path) if str(ba_path) != "." else "",
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "pass"


def _split(value: object) -> set[str]:
    if not value:
        return set()
    return {item for item in str(value).split(";") if item}


def _is_true(value: object) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _pipeline_for(pass_name: str, registry: PassRegistry | None) -> str:
    return registry.pipeline_for(pass_name) if registry else pass_name
