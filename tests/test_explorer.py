import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.explorer import _classify_enable_suppress, _classify_relation_flip, explore_states


class ExplorerTests(unittest.TestCase):
    def test_relation_flip_classifier_covers_requested_kinds(self) -> None:
        self.assertEqual(
            _classify_relation_flip("", "final_commute", parent_present=False, child_present=True),
            "missing_to_active_pair",
        )
        self.assertEqual(
            _classify_relation_flip("final_commute", "", parent_present=True, child_present=False),
            "active_pair_to_missing",
        )
        self.assertEqual(
            _classify_relation_flip("final_commute", "final_commute", parent_present=True, child_present=True),
            "same",
        )
        self.assertEqual(
            _classify_relation_flip(
                "final_commute",
                "final_order_sensitive",
                parent_present=True,
                child_present=True,
            ),
            "commute_to_sensitive",
        )
        self.assertEqual(
            _classify_relation_flip(
                "final_order_sensitive",
                "final_commute",
                parent_present=True,
                child_present=True,
            ),
            "sensitive_to_commute",
        )
        self.assertEqual(
            _classify_relation_flip("final_commute", "final_unknown", parent_present=True, child_present=True),
            "known_to_unknown",
        )
        self.assertEqual(
            _classify_relation_flip("final_unknown", "final_commute", parent_present=True, child_present=True),
            "unknown_to_known",
        )
        self.assertEqual(
            _classify_relation_flip("static_disjoint_function", "final_commute", parent_present=True, child_present=True),
            "other_flip",
        )

    def test_enable_suppress_classifier_covers_requested_kinds(self) -> None:
        dormant = {"success": "true", "active": "false", "inst_delta": "0", "blocks_changed": "0", "changed_functions": ""}
        active = {"success": "true", "active": "true", "inst_delta": "-1", "blocks_changed": "1", "changed_functions": "f"}
        similar_active = dict(active)
        changed_active = dict(active, inst_delta="-2")
        failed = {"success": "false", "active": "false"}

        self.assertEqual(_classify_enable_suppress(dormant, active), "enable")
        self.assertEqual(_classify_enable_suppress(active, dormant), "suppress")
        self.assertEqual(_classify_enable_suppress(active, changed_active), "effect_changed")
        self.assertEqual(_classify_enable_suppress(active, similar_active), "still_active_similar")
        self.assertEqual(_classify_enable_suppress(dormant, dormant), "still_dormant")
        self.assertEqual(_classify_enable_suppress(active, failed), "failed_or_unknown")

    def test_depth_one_explore_caches_duplicate_child_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.ll"
            input_path.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
            passes_path = root / "passes.yaml"
            passes_path.write_text("passes:\n  - pass-a\n  - pass-b\n", encoding="utf-8")
            out_dir = root / "explore"
            prepared_ir = out_dir / "input.ll"

            def fake_prepare(src, out, tools, timeout):
                out.mkdir(parents=True, exist_ok=True)
                prepared_ir.write_text(input_path.read_text(encoding="utf-8"), encoding="utf-8")
                return prepared_ir

            def fake_analyze_state(
                input_ll,
                state_dir,
                tools,
                *,
                valid_passes,
                invalid_rows,
                configured_pass_count,
                jobs,
                timeout,
                max_pairs,
                program,
                state_id,
                depth,
                parent_state_id,
                transition_pass,
            ):
                state_dir = Path(state_dir)
                state_dir.mkdir(parents=True, exist_ok=True)
                if state_id == "S0000":
                    child_a = state_dir / "a.ll"
                    child_b = state_dir / "b.ll"
                    child_a.write_text("define i32 @f() {\n  ret i32 1\n}\n", encoding="utf-8")
                    child_b.write_text("define i32 @f() {\n  ret i32 1\n}\n", encoding="utf-8")
                    _write_csv(
                        state_dir / "pass_profile.csv",
                        [
                            "program",
                            "state_id",
                            "depth",
                            "parent_state_id",
                            "transition_pass",
                            "state_hash",
                            "pass",
                            "success",
                            "active",
                            "output_hash",
                            "output_path",
                            "inst_before",
                            "inst_after",
                            "inst_delta",
                            "blocks_changed",
                            "changed_functions",
                        ],
                        [
                            {
                                "program": "explore",
                                "state_id": "S0000",
                                "depth": "0",
                                "parent_state_id": "",
                                "transition_pass": "",
                                "state_hash": "root-hash",
                                "pass": "pass-a",
                                "success": "true",
                                "active": "true",
                                "output_hash": "child-hash",
                                "output_path": str(child_a),
                                "inst_before": "10",
                                "inst_after": "8",
                                "inst_delta": "-2",
                                "blocks_changed": "1",
                                "changed_functions": "f",
                            },
                            {
                                "program": "explore",
                                "state_id": "S0000",
                                "depth": "0",
                                "parent_state_id": "",
                                "transition_pass": "",
                                "state_hash": "root-hash",
                                "pass": "pass-b",
                                "success": "true",
                                "active": "true",
                                "output_hash": "child-hash",
                                "output_path": str(child_b),
                                "inst_before": "10",
                                "inst_after": "8",
                                "inst_delta": "-2",
                                "blocks_changed": "1",
                                "changed_functions": "f",
                            },
                        ],
                    )
                    _write_csv(
                        state_dir / "pair_relation.csv",
                        ["program", "state_id", "pass_a", "pass_b", "final_relation"],
                        [
                            {
                                "program": "explore",
                                "state_id": "S0000",
                                "pass_a": "pass-a",
                                "pass_b": "pass-b",
                                "final_relation": "final_commute",
                            }
                        ],
                    )
                    _write_csv(
                        state_dir / "per_state_summary.csv",
                        ["program", "state_id", "state_hash", "active_passes", "dormant_passes", "pairs_tested", "dynamic_commute", "order_sensitive", "unknown", "max_conflict_component", "total_time_ms"],
                        [{"program": "explore", "state_id": "S0000", "state_hash": "root-hash", "active_passes": "2", "dormant_passes": "0", "pairs_tested": "1", "dynamic_commute": "1", "order_sensitive": "0", "unknown": "0", "max_conflict_component": "0", "total_time_ms": "11"}],
                    )
                    return {"program": "explore", "state_id": "S0000", "summary_path": str(state_dir / "summary.md")}

                _write_csv(
                    state_dir / "pass_profile.csv",
                    [
                        "program",
                        "state_id",
                        "depth",
                        "parent_state_id",
                        "transition_pass",
                        "state_hash",
                        "pass",
                        "success",
                        "active",
                        "output_hash",
                        "output_path",
                        "inst_delta",
                        "blocks_changed",
                        "changed_functions",
                    ],
                    [
                        {
                            "program": "explore",
                            "state_id": state_id,
                            "depth": str(depth),
                            "parent_state_id": parent_state_id,
                            "transition_pass": transition_pass,
                            "state_hash": "child-hash",
                            "pass": "pass-a",
                            "success": "true",
                            "active": "false",
                            "output_hash": "child-hash",
                            "output_path": "",
                            "inst_delta": "0",
                            "blocks_changed": "0",
                            "changed_functions": "",
                        },
                        {
                            "program": "explore",
                            "state_id": state_id,
                            "depth": str(depth),
                            "parent_state_id": parent_state_id,
                            "transition_pass": transition_pass,
                            "state_hash": "child-hash",
                            "pass": "pass-b",
                            "success": "true",
                            "active": "true",
                            "output_hash": "grandchild-hash",
                            "output_path": str(state_dir / "grandchild.ll"),
                            "inst_delta": "-4",
                            "blocks_changed": "2",
                            "changed_functions": "f,g",
                        },
                    ],
                )
                _write_csv(
                    state_dir / "pair_relation.csv",
                    ["program", "state_id", "pass_a", "pass_b", "final_relation"],
                    [
                        {
                            "program": "explore",
                            "state_id": state_id,
                            "pass_a": "pass-b",
                            "pass_b": "pass-a",
                            "final_relation": "final_order_sensitive",
                        }
                    ],
                )
                _write_csv(
                    state_dir / "per_state_summary.csv",
                    ["program", "state_id", "state_hash", "active_passes", "dormant_passes", "pairs_tested", "dynamic_commute", "order_sensitive", "unknown", "max_conflict_component", "total_time_ms"],
                    [{"program": "explore", "state_id": state_id, "state_hash": "child-hash", "active_passes": "0", "dormant_passes": "2", "pairs_tested": "0", "dynamic_commute": "0", "order_sensitive": "0", "unknown": "0", "max_conflict_component": "0", "total_time_ms": "7"}],
                )
                return {"program": "explore", "state_id": state_id, "summary_path": str(state_dir / "summary.md")}

            with mock.patch("phasebatch.explorer.collect_toolchain", return_value={"tools": {"opt": {"path": "opt", "version": "LLVM"}}}), \
                mock.patch("phasebatch.explorer.prepare_input_ir", side_effect=fake_prepare), \
                mock.patch("phasebatch.explorer.validate_passes", return_value=(["pass-a", "pass-b"], [])), \
                mock.patch("phasebatch.explorer.analyze_state", side_effect=fake_analyze_state) as fake_analyze:
                result = explore_states(
                    input_path,
                    out_dir,
                    passes_path,
                    jobs=1,
                    timeout=1,
                    max_pairs=5,
                    max_depth=1,
                    frontier_policy="all-active",
                    top_k=5,
                )

            states = _read_csv(out_dir / "states.csv")
            transitions = _read_csv(out_dir / "state_transitions.csv")
            relation_flips = _read_csv(out_dir / "relation_flip.csv")
            enable_suppress = _read_csv(out_dir / "enable_suppress.csv")
            aggregate_by_depth = _read_csv(out_dir / "aggregate_by_depth.csv")
            multistate_summary = (out_dir / "multistate_summary.md").read_text(encoding="utf-8")

        self.assertEqual(result["states"], 3)
        self.assertEqual(fake_analyze.call_count, 2)
        self.assertEqual([row["state_id"] for row in states], ["S0000", "S0001", "S0002"])
        self.assertEqual(states[2]["is_duplicate"], "true")
        self.assertEqual(states[2]["duplicate_of"], "S0001")
        self.assertEqual(transitions[1]["is_duplicate"], "true")
        self.assertEqual(transitions[1]["duplicate_of"], "S0001")
        self.assertEqual(len(relation_flips), 2)
        self.assertEqual(relation_flips[0]["pass_a"], "pass-a")
        self.assertEqual(relation_flips[0]["pass_b"], "pass-b")
        self.assertEqual(relation_flips[0]["flip_kind"], "commute_to_sensitive")
        self.assertEqual(len(enable_suppress), 4)
        self.assertIn("suppress", {row["relation"] for row in enable_suppress})
        self.assertIn("effect_changed", {row["relation"] for row in enable_suppress})
        self.assertIn("Top relation flips", multistate_summary)
        self.assertIn("Enable/suppress counts", multistate_summary)
        self.assertEqual([row["depth"] for row in aggregate_by_depth], ["0", "1"])
        self.assertEqual(aggregate_by_depth[0]["num_states"], "1")
        self.assertEqual(aggregate_by_depth[0]["avg_active_passes"], "2.00")
        self.assertEqual(aggregate_by_depth[1]["num_states"], "2")
        self.assertEqual(aggregate_by_depth[1]["avg_dormant_passes"], "2.00")
        self.assertEqual(aggregate_by_depth[1]["state_cache_hits"], "1")
        self.assertEqual(aggregate_by_depth[1]["suppress_count"], "2")
        self.assertEqual(aggregate_by_depth[1]["effect_changed_count"], "2")
        self.assertEqual(aggregate_by_depth[1]["relation_flip_count"], "2")
        self.assertEqual(aggregate_by_depth[1]["commute_to_sensitive"], "2")
        self.assertEqual(aggregate_by_depth[1]["total_time_ms"], "14.00")
        self.assertIn("## Overall", multistate_summary)
        self.assertIn(f"- root state hash: {states[0]['state_hash']}", multistate_summary)
        self.assertIn("- duplicate states: 1", multistate_summary)
        self.assertIn("## By depth table", multistate_summary)
        self.assertIn("## Relation Flips", multistate_summary)
        self.assertIn("## Largest Components", multistate_summary)
        self.assertIn("## Interpretation", multistate_summary)
        self.assertEqual(result["relation_flip_csv"], str(out_dir / "relation_flip.csv"))
        self.assertEqual(result["enable_suppress_csv"], str(out_dir / "enable_suppress.csv"))
        self.assertEqual(result["aggregate_by_depth_csv"], str(out_dir / "aggregate_by_depth.csv"))
        self.assertEqual(result["multistate_summary"], str(out_dir / "multistate_summary.md"))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
