import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.explorer import explore_states


class ExplorerTests(unittest.TestCase):
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
                            },
                        ],
                    )
                    _write_csv(
                        state_dir / "per_state_summary.csv",
                        ["program", "state_id", "state_hash", "active_passes", "pairs_tested", "dynamic_commute", "order_sensitive", "unknown", "max_conflict_component", "total_time_ms"],
                        [{"program": "explore", "state_id": "S0000", "state_hash": "root-hash", "active_passes": "2", "pairs_tested": "1", "dynamic_commute": "1", "order_sensitive": "0", "unknown": "0", "max_conflict_component": "0", "total_time_ms": "11"}],
                    )
                    return {"program": "explore", "state_id": "S0000", "summary_path": str(state_dir / "summary.md")}

                _write_csv(
                    state_dir / "pass_profile.csv",
                    ["program", "state_id", "depth", "parent_state_id", "transition_pass", "state_hash", "pass", "success", "active", "output_hash", "output_path"],
                    [],
                )
                _write_csv(
                    state_dir / "per_state_summary.csv",
                    ["program", "state_id", "state_hash", "active_passes", "pairs_tested", "dynamic_commute", "order_sensitive", "unknown", "max_conflict_component", "total_time_ms"],
                    [{"program": "explore", "state_id": state_id, "state_hash": "child-hash", "active_passes": "0", "pairs_tested": "0", "dynamic_commute": "0", "order_sensitive": "0", "unknown": "0", "max_conflict_component": "0", "total_time_ms": "7"}],
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

        self.assertEqual(result["states"], 3)
        self.assertEqual(fake_analyze.call_count, 2)
        self.assertEqual([row["state_id"] for row in states], ["S0000", "S0001", "S0002"])
        self.assertEqual(states[2]["is_duplicate"], "true")
        self.assertEqual(states[2]["duplicate_of"], "S0001")
        self.assertEqual(transitions[1]["is_duplicate"], "true")
        self.assertEqual(transitions[1]["duplicate_of"], "S0001")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
