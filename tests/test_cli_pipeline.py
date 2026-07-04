import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.cli import analyze_state, run_analysis, run_batch


class CliPipelineTests(unittest.TestCase):
    def test_analyze_state_connects_pipeline_outputs_without_preparing_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
            out_dir = root / "out"

            with mock.patch("phasebatch.cli.prepare_input_ir") as fake_prepare, \
                mock.patch("phasebatch.cli.profile_passes", return_value=[{"program": "out", "state_id": "S0001", "depth": 1, "parent_state_id": "S0000", "transition_pass": "mem2reg", "state_hash": "s", "pass": "instcombine", "active": "true", "changed_functions": "f", "changed_blocks": "f::entry"}]) as fake_profile, \
                mock.patch("phasebatch.cli.test_pairs", return_value=[]):
                result = analyze_state(
                    input_ll,
                    out_dir,
                    {"opt": "opt"},
                    valid_passes=["instcombine"],
                    invalid_rows=[],
                    configured_pass_count=1,
                    jobs=1,
                    timeout=1,
                    max_pairs=1,
                    program="out",
                    state_id="S0001",
                    depth=1,
                    parent_state_id="S0000",
                    transition_pass="mem2reg",
                )

            fake_prepare.assert_not_called()
            fake_profile.assert_called_once()
            self.assertEqual(fake_profile.call_args.kwargs["state_id"], "S0001")

        self.assertEqual(result["program"], "out")
        self.assertEqual(result["state_id"], "S0001")
        self.assertTrue(result["summary_path"].endswith("summary.md"))

    def test_run_analysis_prepares_input_validates_once_and_calls_root_state(self) -> None:
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
                mock.patch("phasebatch.cli.prepare_input_ir", side_effect=fake_prepare) as fake_prepare_call, \
                mock.patch("phasebatch.cli.validate_passes", return_value=(["instcombine"], [])) as fake_validate, \
                mock.patch("phasebatch.cli.analyze_state", return_value={"program": "out", "state_id": "S0000", "summary_path": "summary.md"}) as fake_analyze_state:
                result = run_analysis(input_path, out_dir, passes_path, jobs=1, timeout=1, max_pairs=1)

            fake_prepare_call.assert_called_once()
            fake_validate.assert_called_once()
            fake_analyze_state.assert_called_once()
            kwargs = fake_analyze_state.call_args.kwargs
            self.assertEqual(kwargs["state_id"], "S0000")
            self.assertEqual(kwargs["depth"], 0)
            self.assertEqual(kwargs["parent_state_id"], "")
            self.assertEqual(kwargs["transition_pass"], "")
            self.assertEqual(result["state_id"], "S0000")

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
