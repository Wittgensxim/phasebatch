import unittest

from phasebatch.schema import (
    PAIR_RELATION_FIELDS,
    PASS_PROFILE_FIELDS,
    PER_STATE_SUMMARY_FIELDS,
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
