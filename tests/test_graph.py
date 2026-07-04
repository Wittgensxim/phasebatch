import unittest

from phasebatch.graph import build_graph, component_stats, connected_components


class GraphTests(unittest.TestCase):
    def test_builds_order_sensitive_components(self) -> None:
        rows = [
            {"pass_a": "a", "pass_b": "b", "final_relation": "final_order_sensitive"},
            {"pass_a": "b", "pass_b": "c", "final_relation": "final_order_sensitive"},
            {"pass_a": "d", "pass_b": "e", "final_relation": "final_commute"},
        ]

        graph = build_graph(rows, "order_sensitive_graph")
        components = connected_components(graph)
        stats = component_stats(components)

        self.assertEqual(graph["a"], {"b"})
        self.assertEqual(stats["num_components"], 1)
        self.assertEqual(stats["max_size"], 3)
        self.assertEqual(stats["size_3"], 1)
