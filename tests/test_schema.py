import unittest

from phasebatch.schema import PAIR_RELATION_FIELDS, PASS_PROFILE_FIELDS, PER_STATE_SUMMARY_FIELDS


class SchemaTests(unittest.TestCase):
    def test_state_metadata_fields_are_present_in_core_csv_schemas(self) -> None:
        required = ["state_id", "depth", "parent_state_id", "transition_pass"]

        for field in required:
            self.assertIn(field, PASS_PROFILE_FIELDS)
            self.assertIn(field, PAIR_RELATION_FIELDS)
            self.assertIn(field, PER_STATE_SUMMARY_FIELDS)

        self.assertIn("output_path", PASS_PROFILE_FIELDS)
