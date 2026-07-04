import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.pair_tester import run_pair_tests
from phasebatch.schema import RunResult


class PairTesterTests(unittest.TestCase):
    def test_run_pair_tests_classifies_equal_hashes_as_commute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
            profiles = [
                {
                    "program": "x",
                    "state_id": "S0001",
                    "depth": 1,
                    "parent_state_id": "S0000",
                    "transition_pass": "mem2reg",
                    "state_hash": "s",
                    "pass": "a",
                    "active": "true",
                    "changed_functions": "f",
                    "changed_blocks": "f::entry",
                },
                {
                    "program": "x",
                    "state_id": "S0001",
                    "depth": 1,
                    "parent_state_id": "S0000",
                    "transition_pass": "mem2reg",
                    "state_hash": "s",
                    "pass": "b",
                    "active": "true",
                    "changed_functions": "f",
                    "changed_blocks": "f::entry",
                },
            ]

            def fake_run_opt(opt, src, passes, out, timeout):
                out.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
                return RunResult([opt], 0, "", "", 3.0)

            with mock.patch("phasebatch.pair_tester.run_opt", side_effect=fake_run_opt):
                rows = run_pair_tests(input_ll, profiles, {"opt": "opt"}, root, jobs=1, timeout=1, max_pairs=None)

        self.assertEqual(rows[0]["dynamic_relation"], "dynamic_commute")
        self.assertEqual(rows[0]["same_hash"], "true")
        self.assertEqual(rows[0]["state_id"], "S0001")
        self.assertEqual(rows[0]["depth"], 1)
        self.assertEqual(rows[0]["parent_state_id"], "S0000")
        self.assertEqual(rows[0]["transition_pass"], "mem2reg")

    def test_run_pair_tests_records_not_tested_when_max_pairs_caps_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
            profiles = [
                {"program": "x", "state_id": "S0000", "depth": 0, "parent_state_id": "", "transition_pass": "", "state_hash": "s", "pass": "a", "active": "true", "changed_functions": "f", "changed_blocks": "f::entry"},
                {"program": "x", "state_id": "S0000", "depth": 0, "parent_state_id": "", "transition_pass": "", "state_hash": "s", "pass": "b", "active": "true", "changed_functions": "g", "changed_blocks": "g::entry"},
                {"program": "x", "state_id": "S0000", "depth": 0, "parent_state_id": "", "transition_pass": "", "state_hash": "s", "pass": "c", "active": "true", "changed_functions": "h", "changed_blocks": "h::entry"},
            ]

            def fake_run_opt(opt, src, passes, out, timeout):
                out.write_text(input_ll.read_text(encoding="utf-8"), encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            with mock.patch("phasebatch.pair_tester.run_opt", side_effect=fake_run_opt):
                rows = run_pair_tests(input_ll, profiles, {"opt": "opt"}, root, jobs=1, timeout=1, max_pairs=1)

        self.assertEqual(len(rows), 3)
        self.assertEqual(sum(1 for row in rows if row["dynamic_relation"] == "not_tested"), 2)
        self.assertTrue(all(row["state_id"] == "S0000" for row in rows))
