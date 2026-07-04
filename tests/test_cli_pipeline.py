import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.cli import analyze_state, run_analysis, run_batch, run_batchify, run_explore_batches


class CliPipelineTests(unittest.TestCase):
    def test_analyze_state_connects_pipeline_outputs_without_preparing_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
            out_dir = root / "out"

            with mock.patch("phasebatch.cli.prepare_input_ir") as fake_prepare, \
                mock.patch("phasebatch.cli.profile_passes", return_value=[{"program": "out", "state_id": "S0001", "depth": 1, "parent_state_id": "S0000", "transition_pass": "mem2reg", "state_hash": "s", "pass": "instcombine", "active": "true", "changed_functions": "f", "changed_blocks": "f::entry"}]) as fake_profile, \
                mock.patch("phasebatch.cli.run_pair_tests", return_value=[]):
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

    def test_run_batchify_calls_batcher_without_toolchain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / "state"
            state_dir.mkdir()

            with mock.patch("phasebatch.cli.build_batch_family", return_value={"batch_candidates": 3, "batch_summary_md": "summary.md"}) as fake_build, \
                mock.patch("phasebatch.cli.collect_toolchain") as fake_collect:
                result = run_batchify(state_dir, max_component_size=7, max_batch_candidates=11)

        fake_build.assert_called_once_with(state_dir, max_component_size=7, max_batch_candidates=11)
        fake_collect.assert_not_called()
        self.assertEqual(result["batch_candidates"], 3)

    def test_run_batchify_validates_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / "state"
            state_dir.mkdir()

            with mock.patch("phasebatch.cli.build_batch_family", return_value={"batch_candidates": 3, "batch_summary_md": "summary.md"}) as fake_build, \
                mock.patch("phasebatch.cli.collect_toolchain", return_value={"tools": {"opt": {"path": "opt", "version": "LLVM"}}}) as fake_collect, \
                mock.patch("phasebatch.cli.validate_batch_candidates", return_value={"validated_batches": 3, "batch_validation_csv": "validation.csv"}) as fake_validate:
                result = run_batchify(state_dir, max_component_size=7, max_batch_candidates=11, validate_batches=True)

        fake_build.assert_called_once_with(state_dir, max_component_size=7, max_batch_candidates=11)
        fake_collect.assert_called_once()
        fake_validate.assert_called_once_with(state_dir, {"opt": "opt"}, timeout=10, jobs=1)
        self.assertEqual(result["validated_batches"], 3)

    def test_run_explore_batches_calls_batch_explorer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.ll"
            passes_path = root / "passes.yaml"
            out_dir = root / "out"

            with mock.patch("phasebatch.cli.explore_batches", return_value={"states": 2, "batch_transitions": 1}) as fake_explore:
                result = run_explore_batches(
                    input_path,
                    out_dir,
                    passes_path,
                    jobs=8,
                    timeout=10,
                    max_pairs=300,
                    max_depth=1,
                    max_component_size=10,
                    max_batch_candidates=50,
                    max_batches_per_state=7,
                    max_frontier_states=3,
                    batch_frontier_policy="largest-batch",
                    validate_batches=True,
                    allow_sampled_batches=True,
                )

        fake_explore.assert_called_once_with(
            input_path,
            out_dir,
            passes_path,
            jobs=8,
            timeout=10,
            max_pairs=300,
            max_depth=1,
            max_component_size=10,
            max_batch_candidates=50,
            max_batches_per_state=7,
            max_frontier_states=3,
            batch_frontier_policy="largest-batch",
            validate_batches=True,
            allow_sampled_batches=True,
        )
        self.assertEqual(result["batch_transitions"], 1)
