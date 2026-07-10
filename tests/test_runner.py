import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.runner import compile_c_to_ll, prepare_input_ir, run_opt


class RunnerTests(unittest.TestCase):
    def test_compile_c_to_ll_supports_inlinable_unoptimized_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "x.c"
            src.write_text("int f(void) { return 0; }\n", encoding="utf-8")
            out = root / "x.ll"

            completed = subprocess.CompletedProcess(["clang"], 0, "", "")
            with mock.patch("subprocess.run", return_value=completed) as fake_run:
                compile_c_to_ll(
                    "clang",
                    src,
                    out,
                    timeout=1,
                    root_ir_mode="inlinable-unoptimized",
                )

        command = fake_run.call_args.args[0]
        self.assertIn("-O1", command)
        self.assertIn("-disable-llvm-passes", command)
        self.assertNotIn("-disable-O0-optnone", command)

    def test_compile_c_to_ll_keeps_legacy_o0_as_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "x.c"
            src.write_text("int f(void) { return 0; }\n", encoding="utf-8")
            out = root / "x.ll"

            completed = subprocess.CompletedProcess(["clang"], 0, "", "")
            with mock.patch("subprocess.run", return_value=completed) as fake_run:
                compile_c_to_ll("clang", src, out, timeout=1)

        command = fake_run.call_args.args[0]
        self.assertIn("-O0", command)
        self.assertIn("-disable-O0-optnone", command)

    def test_compile_c_to_ll_rejects_unknown_root_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "unknown root IR mode"):
                compile_c_to_ll(
                    "clang",
                    root / "x.c",
                    root / "x.ll",
                    timeout=1,
                    root_ir_mode="mystery",
                )

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
