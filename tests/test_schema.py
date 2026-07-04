import unittest

from phasebatch.schema import (
    AGGREGATE_BY_DEPTH_FIELDS,
    ENABLE_SUPPRESS_FIELDS,
    PAIR_RELATION_FIELDS,
    PASS_PROFILE_FIELDS,
    PER_STATE_SUMMARY_FIELDS,
    RELATION_FLIP_FIELDS,
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
                "commute_to_sensitive",
                "sensitive_to_commute",
                "missing_to_active_pair",
                "active_pair_to_missing",
                "total_time_ms",
            ],
        )
