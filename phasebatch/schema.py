from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PASS_PROFILE_FIELDS = [
    "program",
    "state_id",
    "depth",
    "parent_state_id",
    "transition_pass",
    "state_hash",
    "pass",
    "success",
    "active",
    "input_hash",
    "output_hash",
    "output_path",
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
    "state_id",
    "depth",
    "parent_state_id",
    "transition_pass",
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
    "state_id",
    "depth",
    "parent_state_id",
    "transition_pass",
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

STATE_FIELDS = [
    "program",
    "state_id",
    "state_hash",
    "depth",
    "parent_state_id",
    "transition_pass",
    "ir_path",
    "state_dir",
    "is_duplicate",
    "duplicate_of",
    "active_passes",
    "pairs_tested",
    "dynamic_commute",
    "order_sensitive",
    "unknown",
    "max_conflict_component",
    "total_time_ms",
]

STATE_TRANSITION_FIELDS = [
    "program",
    "parent_state_id",
    "child_state_id",
    "parent_hash",
    "child_hash",
    "transition_pass",
    "depth",
    "active",
    "inst_before",
    "inst_after",
    "inst_delta",
    "is_duplicate",
    "duplicate_of",
    "ir_path",
]

RELATION_FLIP_FIELDS = [
    "program",
    "parent_state_id",
    "child_state_id",
    "transition_pass",
    "pass_a",
    "pass_b",
    "parent_relation",
    "child_relation",
    "flip_kind",
]

ENABLE_SUPPRESS_FIELDS = [
    "program",
    "parent_state_id",
    "child_state_id",
    "transition_pass",
    "affected_pass",
    "parent_status",
    "child_status",
    "relation",
    "parent_inst_delta",
    "child_inst_delta",
    "parent_blocks_changed",
    "child_blocks_changed",
    "parent_changed_functions",
    "child_changed_functions",
]

AGGREGATE_BY_DEPTH_FIELDS = [
    "program",
    "depth",
    "num_states",
    "avg_active_passes",
    "avg_dormant_passes",
    "avg_pairs_tested",
    "avg_dynamic_commute",
    "avg_order_sensitive",
    "avg_unknown",
    "avg_max_conflict_component",
    "state_cache_hits",
    "enable_count",
    "suppress_count",
    "effect_changed_count",
    "relation_flip_count",
    "commute_to_sensitive",
    "sensitive_to_commute",
    "missing_to_active_pair",
    "active_pair_to_missing",
    "total_time_ms",
]

BATCH_COMPONENT_FIELDS = [
    "program",
    "state_id",
    "state_hash",
    "component_id",
    "component_size",
    "component_passes",
    "conflict_edges",
    "commute_edges",
    "is_exact",
    "num_local_alternatives",
    "unresolved_reason",
]

BATCH_CANDIDATE_FIELDS = [
    "program",
    "state_id",
    "state_hash",
    "batch_id",
    "batch_passes",
    "batch_size",
    "component_choices",
    "is_exact",
    "num_conflict_components",
    "unresolved_components",
    "canonical_order",
]

BATCH_SUMMARY_FIELDS = [
    "program",
    "state_id",
    "state_hash",
    "active_passes",
    "active_pairs",
    "commute_pairs",
    "conflict_pairs",
    "conflict_components",
    "max_component_size",
    "batch_candidates",
    "exact_components",
    "unresolved_components",
    "naive_orderings_estimate",
    "batch_reduction_estimate",
]

BATCH_VALIDATION_FIELDS = [
    "program",
    "state_id",
    "state_hash",
    "batch_id",
    "batch_size",
    "canonical_order",
    "tested_orders",
    "same_hash_count",
    "different_hash_count",
    "validation_status",
    "canonical_hash",
    "first_mismatch_order",
    "first_mismatch_hash",
    "time_ms",
]

BATCH_STATE_TRANSITION_FIELDS = [
    "program",
    "parent_state_id",
    "child_state_id",
    "batch_id",
    "batch_passes",
    "batch_size",
    "parent_hash",
    "child_hash",
    "is_duplicate",
    "duplicate_of",
    "validation_status",
]


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
