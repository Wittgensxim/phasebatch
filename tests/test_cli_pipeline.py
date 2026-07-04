import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.cli import run_analysis, run_batch


class CliPipelineTests(unittest.TestCase):
    def test_run_analysis_connects_pipeline_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.ll"
            input_path.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
            passes_path = root / "passes.yaml"
            passes_path.write_text("passes:\n  - instcombine\n", encoding="utf-8")
            out_dir = root / "out"
            input_ll = out_dir / "input.ll"

            def fake_prepare(src, out, tools, timeout):
                out.mkdir(parents=True, exist_ok=True)
                input_ll.write_text(input_path.read_text(encoding="utf-8"), encoding="utf-8")
                return input_ll

            with mock.patch("phasebatch.cli.collect_toolchain", return_value={"tools": {"opt": {"path": "opt", "version": "LLVM"}}}), \
                mock.patch("phasebatch.cli.prepare_input_ir", side_effect=fake_prepare), \
                mock.patch("phasebatch.cli.validate_passes", return_value=(["instcombine"], [])), \
                mock.patch("phasebatch.cli.profile_passes", return_value=[{"program": "out", "state_hash": "s", "pass": "instcombine", "active": "true", "changed_functions": "f", "changed_blocks": "f::entry"}]), \
                mock.patch("phasebatch.cli.test_pairs", return_value=[]):
                result = run_analysis(input_path, out_dir, passes_path, jobs=1, timeout=1, max_pairs=1)

        self.assertEqual(result["program"], "out")
        self.assertTrue(result["summary_path"].endswith("summary.md"))

    def test_run_batch_expands_glob_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "a.c").write_text("int f(void){return 0;}\n", encoding="utf-8")
            (inputs / "b.c").write_text("int g(void){return 1;}\n", encoding="utf-8")
            passes_path = root / "passes.yaml"
            passes_path.write_text("passes:\n  - instcombine\n", encoding="utf-8")

            with mock.patch("phasebatch.cli.run_analysis") as fake_run:
                fake_run.side_effect = lambda input_path, out_dir, passes_path, jobs, timeout, max_pairs: {
                    "program": Path(input_path).stem,
                    "out_dir": str(out_dir),
                }
                result = run_batch([str(inputs / "*.c")], root / "out", passes_path, jobs=1, timeout=1, max_pairs=None)

        self.assertEqual(len(result["program_dirs"]), 2)
        self.assertEqual(fake_run.call_count, 2)
