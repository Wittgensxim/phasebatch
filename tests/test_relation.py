import unittest

from phasebatch.relation import annotate_pair_relations, final_relation, static_relation


class RelationTests(unittest.TestCase):
    def test_static_relation_detects_disjoint_and_overlap(self) -> None:
        disjoint = static_relation(
            {"changed_functions": "f", "changed_blocks": "f::entry"},
            {"changed_functions": "g", "changed_blocks": "g::entry"},
        )
        overlap = static_relation(
            {"changed_functions": "f", "changed_blocks": "f::entry"},
            {"changed_functions": "f", "changed_blocks": "f::entry"},
        )

        self.assertEqual(disjoint["static_relation"], "static_disjoint_function")
        self.assertEqual(overlap["static_relation"], "static_overlap_block")
        self.assertEqual(overlap["overlap_blocks"], 1)

    def test_final_relation_follows_dynamic_result(self) -> None:
        self.assertEqual(final_relation({"dynamic_relation": "dynamic_commute"}), "final_commute")
        self.assertEqual(final_relation({"dynamic_relation": "dynamic_order_sensitive"}), "final_order_sensitive")
        self.assertEqual(final_relation({"dynamic_relation": "dynamic_failed"}), "final_unknown")

    def test_annotate_pair_relations_joins_profiles(self) -> None:
        rows = [{"pass_a": "a", "pass_b": "b", "dynamic_relation": "dynamic_commute"}]
        profiles = {
            "a": {"pass": "a", "changed_functions": "f", "changed_blocks": "f::entry"},
            "b": {"pass": "b", "changed_functions": "g", "changed_blocks": "g::entry"},
        }

        annotated = annotate_pair_relations(rows, profiles)

        self.assertEqual(annotated[0]["static_relation"], "static_disjoint_function")
        self.assertEqual(annotated[0]["final_relation"], "final_commute")
