from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PASS_PROFILE_FIELDS = [
    "program",
    "state_hash",
    "pass",
    "success",
    "active",
    "input_hash",
    "output_hash",
    "inst_before",
    "inst_after",
    "inst_delta",
    "funcs_changed",
    "blocks_changed",
    "changed_functions",
    "changed_blocks",
    "time_ms",
    "stderr_path",
    "failure_kind",
]

PAIR_RELATION_FIELDS = [
    "program",
    "state_hash",
    "pass_a",
    "pass_b",
    "a_active",
    "b_active",
    "static_relation",
    "dynamic_relation",
    "final_relation",
    "ab_success",
    "ba_success",
    "ab_hash",
    "ba_hash",
    "same_hash",
    "ab_inst",
    "ba_inst",
    "inst_delta_ab_ba",
    "changed_funcs_a",
    "changed_funcs_b",
    "changed_blocks_a",
    "changed_blocks_b",
    "overlap_functions",
    "overlap_blocks",
    "time_ms",
    "failure_kind",
    "ab_path",
    "ba_path",
]

CLUSTER_DISTRIBUTION_FIELDS = [
    "program",
    "state_hash",
    "graph_type",
    "num_nodes",
    "num_edges",
    "num_components",
    "mean_size",
    "median_size",
    "max_size",
    "size_1",
    "size_2",
    "size_3",
    "size_4_7",
    "size_gt_7",
]

PER_STATE_SUMMARY_FIELDS = [
    "program",
    "state_hash",
    "pass_set_size",
    "valid_passes",
    "invalid_passes",
    "active_passes",
    "dormant_passes",
    "total_pairs",
    "pairs_tested",
    "dynamic_commute",
    "order_sensitive",
    "unknown",
    "failed",
    "static_disjoint_function",
    "static_disjoint_block",
    "max_conflict_component",
    "median_conflict_component",
    "profile_time_ms",
    "pair_time_ms",
    "total_time_ms",
]

VALID_PASS_FIELDS = ["pass", "valid", "reason", "test_time_ms"]
INVALID_PASS_FIELDS = ["pass", "valid", "reason", "test_time_ms"]


@dataclass
class RunResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    time_ms: float
    timed_out: bool = False
    failure_kind: str = ""
    output_path: Path | None = None
    stderr_path: Path | None = None

    @property
    def success(self) -> bool:
        return self.returncode == 0 and not self.timed_out
