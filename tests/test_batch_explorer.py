import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.batch_explorer import explore_batches
from phasebatch.schema import RunResult


class BatchExplorerTests(unittest.TestCase):
    def test_depth_one_batch_explore_caches_duplicate_batch_successors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.ll"
            input_path.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
            passes_path = root / "passes.yaml"
            passes_path.write_text("passes:\n  - pass-a\n  - pass-b\n", encoding="utf-8")
            out_dir = root / "batch_explore"
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
                                "program": "batch_explore",
                                "state_id": "S0000",
                                "depth": "0",
                                "parent_state_id": "",
                                "transition_pass": "",
                                "state_hash": "root-hash",
                                "pass": "pass-a",
                                "success": "true",
                                "active": "true",
                                "output_hash": "a-hash",
                                "output_path": str(state_dir / "a.ll"),
                                "inst_delta": "-1",
                                "blocks_changed": "1",
                                "changed_functions": "f",
                            },
                            {
                                "program": "batch_explore",
                                "state_id": "S0000",
                                "depth": "0",
                                "parent_state_id": "",
                                "transition_pass": "",
                                "state_hash": "root-hash",
                                "pass": "pass-b",
                                "success": "true",
                                "active": "true",
                                "output_hash": "b-hash",
                                "output_path": str(state_dir / "b.ll"),
                                "inst_delta": "-1",
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
                                "program": "batch_explore",
                                "state_id": "S0000",
                                "pass_a": "pass-a",
                                "pass_b": "pass-b",
                                "final_relation": "final_order_sensitive",
                            }
                        ],
                    )
                    _write_summary(state_dir, "S0000", "root-hash", "2", "0", "1", "1", "11")
                    return {"program": "batch_explore", "state_id": "S0000", "summary_path": str(state_dir / "summary.md")}

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
                            "program": "batch_explore",
                            "state_id": state_id,
                            "depth": str(depth),
                            "parent_state_id": parent_state_id,
                            "transition_pass": transition_pass,
                            "state_hash": "child-hash",
                            "pass": "pass-a",
                            "success": "true",
                            "active": "false",
                            "output_hash": "",
                            "output_path": "",
                            "inst_delta": "0",
                            "blocks_changed": "0",
                            "changed_functions": "",
                        },
                        {
                            "program": "batch_explore",
                            "state_id": state_id,
                            "depth": str(depth),
                            "parent_state_id": parent_state_id,
                            "transition_pass": transition_pass,
                            "state_hash": "child-hash",
                            "pass": "pass-b",
                            "success": "true",
                            "active": "true",
                            "output_hash": "next-hash",
                            "output_path": str(state_dir / "next.ll"),
                            "inst_delta": "-3",
                            "blocks_changed": "2",
                            "changed_functions": "f,g",
                        }
                    ],
                )
                _write_csv(
                    state_dir / "pair_relation.csv",
                    ["program", "state_id", "pass_a", "pass_b", "final_relation"],
                    [
                        {
                            "program": "batch_explore",
                            "state_id": state_id,
                            "pass_a": "pass-a",
                            "pass_b": "pass-b",
                            "final_relation": "final_commute",
                        }
                    ],
                )
                _write_summary(state_dir, state_id, "child-hash", "1", "1", "1", "1", "7")
                return {"program": "batch_explore", "state_id": state_id, "summary_path": str(state_dir / "summary.md")}

            def fake_run_opt(opt, src, passes, out, timeout):
                out.write_text("define i32 @f() {\n  ret i32 1\n}\n", encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            with mock.patch("phasebatch.batch_explorer.collect_toolchain", return_value={"tools": {"opt": {"path": "opt", "version": "LLVM"}}}), \
                mock.patch("phasebatch.batch_explorer.prepare_input_ir", side_effect=fake_prepare), \
                mock.patch("phasebatch.batch_explorer.validate_passes", return_value=(["pass-a", "pass-b"], [])), \
                mock.patch("phasebatch.batch_explorer.analyze_state", side_effect=fake_analyze_state) as fake_analyze, \
                mock.patch("phasebatch.batch_explorer.run_opt", side_effect=fake_run_opt), \
                mock.patch("phasebatch.batch_explorer.validate_batch_candidates", return_value={"validated_batches": 2}):
                result = explore_batches(
                    input_path,
                    out_dir,
                    passes_path,
                    jobs=1,
                    timeout=1,
                    max_pairs=5,
                    max_depth=1,
                    max_component_size=10,
                    max_batch_candidates=50,
                    validate_batches=True,
                )

            states = _read_csv(out_dir / "states.csv")
            transitions = _read_csv(out_dir / "batch_state_transitions.csv")
            enable_suppress = _read_csv(out_dir / "enable_suppress.csv")
            relation_flips = _read_csv(out_dir / "relation_flip.csv")
            aggregate = _read_csv(out_dir / "aggregate_by_depth.csv")
            multistate_summary = (out_dir / "multistate_summary.md").read_text(encoding="utf-8")
            summary = (out_dir / "batch_explore_summary.md").read_text(encoding="utf-8")

        self.assertEqual(result["states"], 3)
        self.assertEqual(result["batch_transitions"], 2)
        self.assertEqual(fake_analyze.call_count, 2)
        self.assertEqual([row["state_id"] for row in states], ["S0000", "S0001", "S0002"])
        self.assertEqual(states[2]["is_duplicate"], "true")
        self.assertEqual(states[2]["duplicate_of"], "S0001")
        self.assertEqual([row["batch_id"] for row in transitions], ["B0000", "B0001"])
        self.assertEqual(transitions[1]["is_duplicate"], "true")
        self.assertEqual(transitions[1]["duplicate_of"], "S0001")
        self.assertEqual(transitions[0]["validation_status"], "not_validated")
        self.assertEqual(len(enable_suppress), 4)
        self.assertIn("suppress", {row["relation"] for row in enable_suppress})
        self.assertIn("effect_changed", {row["relation"] for row in enable_suppress})
        self.assertEqual(len(relation_flips), 2)
        self.assertEqual({row["flip_kind"] for row in relation_flips}, {"sensitive_to_commute"})
        self.assertEqual(aggregate[1]["state_cache_hits"], "1")
        self.assertEqual(aggregate[1]["suppress_count"], "2")
        self.assertEqual(aggregate[1]["effect_changed_count"], "2")
        self.assertEqual(aggregate[1]["relation_flip_count"], "2")
        self.assertEqual(aggregate[1]["true_relation_flip_count"], "2")
        self.assertIn("Enable/suppress counts", multistate_summary)
        self.assertIn("True relation flips among pairs active in both states", multistate_summary)
        self.assertIn("Batch Explore Summary", summary)
        self.assertIn("batch transitions: 2", summary)
        self.assertEqual(result["enable_suppress_csv"], str(out_dir / "enable_suppress.csv"))
        self.assertEqual(result["relation_flip_csv"], str(out_dir / "relation_flip.csv"))
        self.assertEqual(result["batch_state_transitions_csv"], str(out_dir / "batch_state_transitions.csv"))
        self.assertEqual(result["batch_explore_summary"], str(out_dir / "batch_explore_summary.md"))


def _write_summary(
    state_dir: Path,
    state_id: str,
    state_hash: str,
    active_passes: str,
    dormant_passes: str,
    pairs_tested: str,
    max_conflict_component: str,
    total_time_ms: str,
) -> None:
    _write_csv(
        state_dir / "per_state_summary.csv",
        [
            "program",
            "state_id",
            "state_hash",
            "active_passes",
            "dormant_passes",
            "pairs_tested",
            "dynamic_commute",
            "order_sensitive",
            "unknown",
            "max_conflict_component",
            "total_time_ms",
        ],
        [
            {
                "program": "batch_explore",
                "state_id": state_id,
                "state_hash": state_hash,
                "active_passes": active_passes,
                "dormant_passes": dormant_passes,
                "pairs_tested": pairs_tested,
                "dynamic_commute": "0",
                "order_sensitive": pairs_tested,
                "unknown": "0",
                "max_conflict_component": max_conflict_component,
                "total_time_ms": total_time_ms,
            }
        ],
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
