from __future__ import annotations

import json
import time
from pathlib import Path

from .graph import cluster_distribution_rows, write_cluster_distribution
from .normalizer import canonical_hash
from .pair_cost import write_pair_cost_summary
from .pair_scheduling import write_pair_scheduling_summary
from .pair_tester import run_pair_tests
from .pass_config import PassRegistry
from .profiler import profile_passes
from .relation import annotate_pair_relations, write_pair_relations
from .report import write_per_state_summary, write_summary
from .tools import write_metadata


def analyze_state(
    input_ll: Path,
    out_dir: Path,
    tools: dict,
    *,
    valid_passes: list[str],
    invalid_rows: list[dict],
    configured_pass_count: int,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
    program: str,
    state_id: str,
    depth: int,
    parent_state_id: str,
    transition_pass: str,
    pass_registry: PassRegistry | None = None,
    pair_testing_mode: str = "full",
    pair_test_budget_per_state: int = 0,
    pair_priority_policy: str = "mixed",
    batch_construction_mode: str = "pairwise",
    keep_ir_artifacts: bool = False,
) -> dict:
    start = time.perf_counter()
    input_ll = Path(input_ll)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_hash = canonical_hash(input_ll)
    pass_registry = pass_registry or tools.get("_pass_registry")
    if batch_construction_mode != "pairwise":
        raise ValueError("batch construction only supports pairwise")

    metadata = _read_metadata(out_dir)
    metadata.update(
        {
            "input": str(input_ll),
            "state_hash": state_hash,
            "state_id": state_id,
            "depth": depth,
            "parent_state_id": parent_state_id,
            "transition_pass": transition_pass,
            "pair_testing_mode": pair_testing_mode,
            "pair_test_budget_per_state": pair_test_budget_per_state,
            "pair_priority_policy": pair_priority_policy,
            "batch_construction_mode": batch_construction_mode,
            "pair_matrix_complete": False,
        }
    )
    write_metadata(out_dir, metadata)

    profile_start = time.perf_counter()
    profile_rows = profile_passes(
        input_ll,
        valid_passes,
        tools,
        out_dir,
        jobs,
        timeout,
        program=program,
        state_id=state_id,
        depth=depth,
        parent_state_id=parent_state_id,
        transition_pass=transition_pass,
        pass_registry=pass_registry if isinstance(pass_registry, PassRegistry) else None,
    )
    profile_time_ms = (time.perf_counter() - profile_start) * 1000
    active_profiles = [row for row in profile_rows if row.get("success") == "true" and row.get("active") == "true"]

    pair_start = time.perf_counter()
    pair_rows = run_pair_tests(
        input_ll,
        active_profiles,
        tools,
        out_dir,
        jobs,
        timeout,
        max_pairs,
        pass_registry=pass_registry if isinstance(pass_registry, PassRegistry) else None,
        pair_testing_mode=pair_testing_mode,
        pair_test_budget_per_state=pair_test_budget_per_state,
        pair_priority_policy=pair_priority_policy,
        keep_ir_artifacts=keep_ir_artifacts,
    )
    profile_map = {row["pass"]: row for row in profile_rows}
    pair_rows = annotate_pair_relations(pair_rows, profile_map)
    write_pair_relations(out_dir / "pair_relation.csv", pair_rows)
    pair_time_ms = (time.perf_counter() - pair_start) * 1000
    full_pair_count = len(active_profiles) * (len(active_profiles) - 1) // 2
    pair_matrix_complete = (
        pair_testing_mode == "full"
        and len(pair_rows) == full_pair_count
        and not any(
            row.get("dynamic_relation") == "not_tested"
            or row.get("failure_kind") in {"lazy_budget", "max_pairs"}
            or row.get("skipped_by_budget") == "true"
            for row in pair_rows
        )
    )

    cluster_rows = cluster_distribution_rows(pair_rows, program, state_hash)
    write_cluster_distribution(out_dir / "cluster_distribution.csv", cluster_rows)

    total_time_ms = (time.perf_counter() - start) * 1000
    write_per_state_summary(
        out_dir,
        program,
        state_hash,
        state_id=state_id,
        depth=depth,
        parent_state_id=parent_state_id,
        transition_pass=transition_pass,
        pass_set_size=configured_pass_count,
        valid_passes=len(valid_passes),
        invalid_passes=len(invalid_rows),
        profile_time_ms=profile_time_ms,
        pair_time_ms=pair_time_ms,
        total_time_ms=total_time_ms,
    )
    summary = write_summary(out_dir)
    pair_cost = write_pair_cost_summary(out_dir)
    pair_scheduling = write_pair_scheduling_summary(out_dir)

    metadata.update(
        {
            "valid_passes": len(valid_passes),
            "invalid_passes": len(invalid_rows),
            "active_passes": len(active_profiles),
            "pair_rows": len(pair_rows),
            "pair_matrix_complete": pair_matrix_complete,
            "summary": str(summary),
            "pair_cost_summary": pair_cost["pair_cost_summary_md"],
            "pair_scheduling_summary": pair_scheduling["pair_scheduling_summary_md"],
            "total_time_ms": total_time_ms,
        }
    )
    write_metadata(out_dir, metadata)
    return {
        "program": program,
        "out_dir": str(out_dir),
        "state_id": state_id,
        "depth": depth,
        "parent_state_id": parent_state_id,
        "transition_pass": transition_pass,
        "valid_passes": len(valid_passes),
        "active_passes": len(active_profiles),
        "pair_rows": len(pair_rows),
        "batch_construction_mode": batch_construction_mode,
        "pair_matrix_complete": "true" if pair_matrix_complete else "false",
        "summary_path": str(summary),
        "pair_cost_summary_csv": pair_cost["pair_cost_summary_csv"],
        "pair_cost_summary_md": pair_cost["pair_cost_summary_md"],
        "pair_scheduling_summary_csv": pair_scheduling["pair_scheduling_summary_csv"],
        "pair_scheduling_summary_md": pair_scheduling["pair_scheduling_summary_md"],
        "total_time_ms": total_time_ms,
    }


def _read_metadata(out_dir: Path) -> dict:
    path = Path(out_dir) / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
