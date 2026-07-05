import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.runner import prepare_input_ir, run_opt


class RunnerTests(unittest.TestCase):
    def test_prepare_input_ir_copies_ll_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "x.ll"
            src.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")

            out = prepare_input_ir(src, root / "out", {"clang": "clang"}, timeout=1)

            self.assertEqual(out.name, "input.ll")
            self.assertEqual(out.read_text(encoding="utf-8"), src.read_text(encoding="utf-8"))

    def test_run_opt_records_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("", encoding="utf-8")
            output_ll = root / "out.ll"

            with mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired(["opt"], 1)):
                result = run_opt("opt", input_ll, ["instcombine"], output_ll, timeout=1)

        self.assertTrue(result.timed_out)
        self.assertEqual(result.failure_kind, "timeout")
        self.assertNotEqual(result.returncode, 0)

    def test_run_opt_uses_pipeline_segments_verbatim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("", encoding="utf-8")
            output_ll = root / "out.ll"

            completed = subprocess.CompletedProcess(["opt"], 0, "", "")
            with mock.patch("subprocess.run", return_value=completed) as fake_run:
                run_opt("opt", input_ll, ["mem2reg", "function(loop(licm))", "dce"], output_ll, timeout=1)

        command = fake_run.call_args.args[0]
        self.assertIn("-passes=mem2reg,function(loop(licm)),dce", command)
