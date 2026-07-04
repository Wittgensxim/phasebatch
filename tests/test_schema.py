import unittest

from phasebatch.schema import (
    AGGREGATE_BY_DEPTH_FIELDS,
    BATCH_CANDIDATE_FIELDS,
    BATCH_COMPONENT_FIELDS,
    BATCH_STATE_TRANSITION_FIELDS,
    BATCH_SUMMARY_FIELDS,
    BATCH_VALIDATION_FIELDS,
    ENABLE_SUPPRESS_FIELDS,
    PAIR_RELATION_FIELDS,
    PASS_PROFILE_FIELDS,
    PER_STATE_SUMMARY_FIELDS,
    RELATION_FLIP_FIELDS,
    SKIPPED_BATCH_FIELDS,
    STATE_FIELDS,
    STATE_TRANSITION_FIELDS,
)


class SchemaTests(unittest.TestCase):
    def test_state_metadata_fields_are_present_in_core_csv_schemas(self) -> None:
        required = ["state_id", "depth", "parent_state_id", "transition_pass"]

        for field in required:
            self.assertIn(field, PASS_PROFILE_FIELDS)
            self.assertIn(field, PAIR_RELATION_FIELDS)
            self.assertIn(field, PER_STATE_SUMMARY_FIELDS)

        self.assertIn("output_path", PASS_PROFILE_FIELDS)

    def test_state_graph_csv_schemas_are_present(self) -> None:
        self.assertEqual(
            STATE_FIELDS,
            [
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
            ],
        )
        self.assertEqual(
            STATE_TRANSITION_FIELDS,
            [
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
            ],
        )

    def test_cross_state_interaction_csv_schemas_are_present(self) -> None:
        self.assertEqual(
            RELATION_FLIP_FIELDS,
            [
                "program",
                "parent_state_id",
                "child_state_id",
                "transition_pass",
                "pass_a",
                "pass_b",
                "parent_relation",
                "child_relation",
                "flip_kind",
            ],
        )
        self.assertEqual(
            ENABLE_SUPPRESS_FIELDS,
            [
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
            ],
        )

    def test_aggregate_by_depth_schema_is_present(self) -> None:
        self.assertEqual(
            AGGREGATE_BY_DEPTH_FIELDS,
            [
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
                "pair_availability_change_count",
                "true_relation_flip_count",
                "commute_to_sensitive",
                "sensitive_to_commute",
                "missing_to_active_pair",
                "active_pair_to_missing",
                "total_time_ms",
            ],
        )

    def test_batch_construction_schemas_are_present(self) -> None:
        self.assertEqual(
            BATCH_COMPONENT_FIELDS,
            [
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
            ],
        )
        self.assertEqual(
            BATCH_CANDIDATE_FIELDS,
            [
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
            ],
        )
        self.assertEqual(
            BATCH_SUMMARY_FIELDS,
            [
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
            ],
        )
        self.assertEqual(
            BATCH_VALIDATION_FIELDS,
            [
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
            ],
        )
        self.assertEqual(
            BATCH_STATE_TRANSITION_FIELDS,
            [
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
            ],
        )
        self.assertEqual(
            SKIPPED_BATCH_FIELDS,
            [
                "program",
                "parent_state_id",
                "batch_id",
                "batch_passes",
                "batch_size",
                "validation_status",
                "skip_reason",
            ],
        )
