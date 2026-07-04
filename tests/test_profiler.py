import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.ir_parser import parse_ir_snapshot
from phasebatch.profiler import profile_passes, validate_passes
from phasebatch.schema import RunResult


class ProfilerTests(unittest.TestCase):
    def test_validate_passes_writes_valid_and_invalid_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")

            def fake_run_opt(opt, src, passes, out, timeout):
                if passes == ["bad-pass"]:
                    return RunResult([opt], 1, "", "bad", 1.0, failure_kind="nonzero_exit")
                out.write_text(input_ll.read_text(encoding="utf-8"), encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            with mock.patch("phasebatch.profiler.run_opt", side_effect=fake_run_opt):
                valid, invalid = validate_passes(input_ll, ["instcombine", "bad-pass"], {"opt": "opt"}, root, timeout=1)

            self.assertEqual(valid, ["instcombine"])
            self.assertEqual(invalid[0]["pass"], "bad-pass")
            self.assertTrue((root / "valid_passes.csv").exists())
            self.assertTrue((root / "invalid_passes.csv").exists())

    def test_profile_passes_marks_changed_pass_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define i32 @f(i32 %x) {\nentry:\n  %a = add i32 %x, 0\n  ret i32 %a\n}\n", encoding="utf-8")

            def fake_run_opt(opt, src, passes, out, timeout):
                out.write_text("define i32 @f(i32 %x) {\nentry:\n  ret i32 %x\n}\n", encoding="utf-8")
                return RunResult([opt], 0, "", "", 2.0)

            with mock.patch("phasebatch.profiler.run_opt", side_effect=fake_run_opt):
                rows = profile_passes(input_ll, ["instcombine"], {"opt": "opt"}, root, jobs=1, timeout=1)

            self.assertEqual(rows[0]["active"], "true")
            self.assertEqual(rows[0]["funcs_changed"], 1)
            self.assertEqual(rows[0]["blocks_changed"], 1)
            with (root / "pass_profile.csv").open(encoding="utf-8", newline="") as handle:
                self.assertEqual(next(csv.DictReader(handle))["pass"], "instcombine")
