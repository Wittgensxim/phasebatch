import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.cli import analyze_state, run_analysis, run_batch, run_batchify, run_budgeted_sensitivity, run_compare_baselines, run_component_summary, run_core_v1_budgeted_study, run_core_v1_case_study, run_diagnose_paths, run_eval_batches, run_evidence_pack, run_explore_batches, run_method_comparison, run_optimize_batches, run_passset_smoke, run_reduction_study, run_replay_final_pipeline, run_reduction_summary, run_round_sensitivity, run_select_and_run_exact_reference, run_summarize_exact_reduction_study, run_summarize_passsets, run_v2_extension_study, run_v3_loop_smoke, run_visualize_dag


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
                mock.patch("phasebatch.cli.collect_toolchain") as fake_collect, \
                mock.patch("phasebatch.cli.classify_batch_correctness", return_value=[{"batch_id": "B0000"}]) as fake_classify, \
                mock.patch("phasebatch.cli.build_footprint_overlap", return_value=[{"pass_a": "A", "pass_b": "B"}]) as fake_footprint, \
                mock.patch("phasebatch.cli.build_coverage_report", return_value=[{"active_pass": "A"}]) as fake_coverage:
                result = run_batchify(state_dir, max_component_size=7, max_batch_candidates=11)

        fake_build.assert_called_once_with(state_dir, max_component_size=7, max_batch_candidates=11)
        fake_collect.assert_not_called()
        fake_classify.assert_called_once_with(state_dir, allow_sampled_batches=False)
        fake_footprint.assert_called_once_with(state_dir)
        fake_coverage.assert_called_once_with(state_dir)
        self.assertEqual(result["batch_candidates"], 3)
        self.assertEqual(result["batch_correctness_rows"], 1)
        self.assertEqual(result["footprint_overlap_rows"], 1)
        self.assertEqual(result["coverage_rows"], 1)

    def test_run_batchify_validates_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / "state"
            state_dir.mkdir()

            with mock.patch("phasebatch.cli.build_batch_family", return_value={"batch_candidates": 3, "batch_summary_md": "summary.md"}) as fake_build, \
                mock.patch("phasebatch.cli.collect_toolchain", return_value={"tools": {"opt": {"path": "opt", "version": "LLVM"}}}) as fake_collect, \
                mock.patch("phasebatch.cli.validate_batch_candidates", return_value={"validated_batches": 3, "batch_validation_csv": "validation.csv"}) as fake_validate, \
                mock.patch("phasebatch.cli.classify_batch_correctness", return_value=[{"batch_id": "B0000"}]) as fake_classify, \
                mock.patch("phasebatch.cli.build_footprint_overlap", return_value=[{"pass_a": "A", "pass_b": "B"}]) as fake_footprint, \
                mock.patch("phasebatch.cli.build_coverage_report", return_value=[{"active_pass": "A"}]) as fake_coverage:
                result = run_batchify(
                    state_dir,
                    max_component_size=7,
                    max_batch_candidates=11,
                    validate_batches=True,
                    allow_sampled_batches=True,
                )

        fake_build.assert_called_once_with(state_dir, max_component_size=7, max_batch_candidates=11)
        fake_collect.assert_called_once()
        fake_validate.assert_called_once_with(state_dir, {"opt": "opt"}, timeout=10, jobs=1)
        fake_classify.assert_called_once_with(state_dir, allow_sampled_batches=True)
        fake_footprint.assert_called_once_with(state_dir)
        fake_coverage.assert_called_once_with(state_dir)
        self.assertEqual(result["validated_batches"], 3)
        self.assertEqual(result["batch_correctness_rows"], 1)
        self.assertEqual(result["footprint_overlap_rows"], 1)
        self.assertEqual(result["coverage_rows"], 1)

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

    def test_run_eval_batches_calls_objective_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"

            with mock.patch("phasebatch.cli.eval_batch_objectives", return_value={"rows": 2}) as fake_eval:
                result = run_eval_batches(run_dir, objective="ir-inst-count", recursive=True)

        fake_eval.assert_called_once_with(run_dir, objective="ir-inst-count", recursive=True)
        self.assertEqual(result["rows"], 2)

    def test_run_compare_baselines_calls_baseline_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            passes_path = Path(tmp) / "passes.yaml"

            with mock.patch("phasebatch.cli.compare_baselines", return_value={"rows": 5}) as fake_compare:
                result = run_compare_baselines(
                    run_dir,
                    passes_path,
                    objective="ir-inst-count",
                    methods=["greedy"],
                    max_rounds=2,
                    random_trials=20,
                    seed=7,
                    timeout=10,
                    jobs=8,
                    greedy_allow_nonimproving=True,
                    include_default_pipelines=True,
                    include_llvm_defaults=True,
                )

        fake_compare.assert_called_once_with(
            run_dir,
            passes_path,
            objective="ir-inst-count",
            methods=["greedy"],
            max_rounds=2,
            random_trials=20,
            seed=7,
            timeout=10,
            jobs=8,
            greedy_allow_nonimproving=True,
            include_default_pipelines=True,
            include_llvm_defaults=True,
        )
        self.assertEqual(result["rows"], 5)

    def test_run_method_comparison_calls_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            passes_path = root / "passes.yaml"

            with mock.patch("phasebatch.cli.run_method_comparison_impl", return_value={"programs": 2}) as fake_run:
                result = run_method_comparison(
                    ["a.c", "b.ll"],
                    out_dir,
                    passes_path,
                    optimizer_mode="budgeted",
                    objective="ir-inst-count",
                    max_rounds=3,
                    beam_width=4,
                    max_states=500,
                    max_batches_per_state=10,
                    batch_frontier_policy="score",
                    validate_batches=True,
                    baseline_max_rounds=5,
                    random_trials=20,
                    seed=7,
                    include_default_pipelines=True,
                    jobs=8,
                    timeout=10,
                    max_pairs=300,
                    overwrite=True,
                    continue_on_error=True,
                )

        fake_run.assert_called_once_with(
            ["a.c", "b.ll"],
            out_dir,
            passes_path,
            optimizer_mode="budgeted",
            objective="ir-inst-count",
            max_rounds=3,
            beam_width=4,
            max_states=500,
            max_batches_per_state=10,
            batch_frontier_policy="score",
            validate_batches=True,
            baseline_max_rounds=5,
            random_trials=20,
            seed=7,
            include_default_pipelines=True,
            jobs=8,
            timeout=10,
            max_pairs=300,
            overwrite=True,
            continue_on_error=True,
        )
        self.assertEqual(result["programs"], 2)

    def test_run_round_sensitivity_calls_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.c"
            out_dir = root / "out"
            passes_path = root / "passes.yaml"

            with mock.patch("phasebatch.cli.run_round_sensitivity_impl", return_value={"rows": 3}) as fake_run:
                result = run_round_sensitivity(
                    input_path,
                    out_dir,
                    passes_path,
                    rounds=[2, 3, 4],
                    optimizer_mode="exact",
                    objective="ir-inst-count",
                    beam_width=8,
                    max_states=5000,
                    max_batches_per_state=20,
                    batch_frontier_policy="score",
                    validate_batches=True,
                    jobs=8,
                    timeout=10,
                    max_pairs=300,
                    overwrite=True,
                )

        fake_run.assert_called_once_with(
            input_path,
            out_dir,
            passes_path,
            rounds=[2, 3, 4],
            optimizer_mode="exact",
            objective="ir-inst-count",
            beam_width=8,
            max_states=5000,
            max_batches_per_state=20,
            batch_frontier_policy="score",
            validate_batches=True,
            jobs=8,
            timeout=10,
            max_pairs=300,
            overwrite=True,
        )
        self.assertEqual(result["rows"], 3)

    def test_run_passset_smoke_calls_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            passsets = [root / "core_passes_v1.yaml", root / "scalar_passes_v2.yaml"]

            with mock.patch("phasebatch.cli.run_passset_smoke_impl", return_value={"runs": 2}) as fake_run:
                result = run_passset_smoke(
                    ["a.c", "b.c"],
                    passsets,
                    out_dir,
                    optimizer_mode="exact",
                    objective="ir-inst-count",
                    max_rounds=2,
                    beam_width=8,
                    max_states=5000,
                    max_batches_per_state=20,
                    batch_frontier_policy="score",
                    validate_batches=True,
                    jobs=8,
                    timeout=10,
                    max_pairs=600,
                    overwrite=True,
                    continue_on_error=True,
                )

        fake_run.assert_called_once_with(
            ["a.c", "b.c"],
            passsets,
            out_dir,
            optimizer_mode="exact",
            objective="ir-inst-count",
            max_rounds=2,
            beam_width=8,
            max_states=5000,
            max_batches_per_state=20,
            batch_frontier_policy="score",
            validate_batches=True,
            jobs=8,
            timeout=10,
            max_pairs=600,
            overwrite=True,
            continue_on_error=True,
        )
        self.assertEqual(result["runs"], 2)

    def test_run_v2_extension_study_calls_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            v1 = root / "core_passes.yaml"
            v2 = root / "scalar_passes_v2.yaml"

            with mock.patch("phasebatch.cli.run_v2_extension_study_impl", return_value={"programs": 2}) as fake_run:
                result = run_v2_extension_study(
                    ["a.c", "b.c"],
                    out_dir,
                    v1,
                    v2,
                    objective="ir-inst-count",
                    max_rounds=4,
                    beam_width=4,
                    max_states=500,
                    max_batches_per_state=20,
                    batch_frontier_policy="score",
                    validate_batches=True,
                    jobs=8,
                    timeout=10,
                    max_pairs=600,
                    random_trials=20,
                    seed=0,
                    overwrite=True,
                    continue_on_error=True,
                )

        fake_run.assert_called_once_with(
            ["a.c", "b.c"],
            out_dir,
            v1,
            v2,
            objective="ir-inst-count",
            max_rounds=4,
            beam_width=4,
            max_states=500,
            max_batches_per_state=20,
            batch_frontier_policy="score",
            validate_batches=True,
            jobs=8,
            timeout=10,
            max_pairs=600,
            random_trials=20,
            seed=0,
            overwrite=True,
            continue_on_error=True,
        )
        self.assertEqual(result["programs"], 2)

    def test_run_visualize_dag_calls_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            out_dir = root / "viz"

            with mock.patch("phasebatch.cli.visualize_dag_impl", return_value={"unique_states": 3}) as fake_run:
                result = run_visualize_dag(
                    run_dir,
                    out_dir,
                    view="selected-neighborhood",
                    formats=["dot", "svg"],
                    max_full_nodes=150,
                    include_selected_path=True,
                    include_depth_overview=True,
                )

        fake_run.assert_called_once_with(
            run_dir,
            out_dir,
            view="selected-neighborhood",
            formats=["dot", "svg"],
            max_full_nodes=150,
            include_selected_path=True,
            include_depth_overview=True,
        )
        self.assertEqual(result["unique_states"], 3)

    def test_run_v3_loop_smoke_calls_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            passes_path = root / "middleend_passes_v3.yaml"

            with mock.patch("phasebatch.cli.run_v3_loop_smoke_impl", return_value={"programs_attempted": 2}) as fake_run:
                result = run_v3_loop_smoke(
                    ["loop.c", "n-body.c"],
                    out_dir,
                    passes_path,
                    optimizer_mode="budgeted",
                    objective="ir-inst-count",
                    max_rounds=3,
                    beam_width=4,
                    max_states=800,
                    max_batches_per_state=12,
                    batch_frontier_policy="score",
                    validate_batches=True,
                    jobs=8,
                    timeout=10,
                    max_pairs=1000,
                    overwrite=True,
                    continue_on_error=True,
                )

        fake_run.assert_called_once_with(
            ["loop.c", "n-body.c"],
            out_dir,
            passes_path,
            optimizer_mode="budgeted",
            objective="ir-inst-count",
            max_rounds=3,
            beam_width=4,
            max_states=800,
            max_batches_per_state=12,
            batch_frontier_policy="score",
            validate_batches=True,
            jobs=8,
            timeout=10,
            max_pairs=1000,
            overwrite=True,
            continue_on_error=True,
        )
        self.assertEqual(result["programs_attempted"], 2)

    def test_run_summarize_passsets_calls_reporter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "report"
            inputs = [root / "passset", root / "v3"]

            with mock.patch("phasebatch.cli.summarize_passsets_impl", return_value={"matrix_rows": 3}) as fake_summary:
                result = run_summarize_passsets(inputs, out_dir)

        fake_summary.assert_called_once_with(inputs, out_dir)
        self.assertEqual(result["matrix_rows"], 3)

    def test_run_reduction_summary_calls_reporter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"

            with mock.patch("phasebatch.cli.summarize_reduction_impl", return_value={"states": 4}) as fake_summary:
                result = run_reduction_summary(run_dir)

        fake_summary.assert_called_once_with(run_dir)
        self.assertEqual(result["states"], 4)

    def test_run_evidence_pack_calls_exporter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"

            with mock.patch("phasebatch.cli.export_evidence_pack_impl", return_value={"selected_batches": 2}) as fake_export:
                result = run_evidence_pack(run_dir)

        fake_export.assert_called_once_with(run_dir)
        self.assertEqual(result["selected_batches"], 2)

    def test_run_diagnose_paths_calls_reporter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            baseline_dir = Path(tmp) / "baseline"

            with mock.patch("phasebatch.cli.diagnose_paths_impl", return_value={"methods": 4}) as fake_diagnose:
                result = run_diagnose_paths(run_dir, baseline_dir=baseline_dir, timeout=7)

        fake_diagnose.assert_called_once_with(run_dir, baseline_dir=baseline_dir, timeout=7)
        self.assertEqual(result["methods"], 4)

    def test_run_reduction_study_calls_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            passes_path = root / "passes.yaml"

            with mock.patch("phasebatch.cli.run_reduction_study_impl", return_value={"programs": 2}) as fake_run:
                result = run_reduction_study(
                    ["a.c", "b.c"],
                    out_dir,
                    passes_path,
                    optimizer_mode="exact",
                    objective="ir-inst-count",
                    max_rounds=2,
                    max_states=5000,
                    validate_batches=True,
                    jobs=8,
                    timeout=10,
                    max_pairs=300,
                    overwrite=True,
                    continue_on_error=True,
                )

        fake_run.assert_called_once_with(
            ["a.c", "b.c"],
            out_dir,
            passes_path,
            optimizer_mode="exact",
            objective="ir-inst-count",
            max_rounds=2,
            max_states=5000,
            validate_batches=True,
            jobs=8,
            timeout=10,
            max_pairs=300,
            summarize_components=False,
            overwrite=True,
            continue_on_error=True,
        )
        self.assertEqual(result["programs"], 2)

    def test_run_budgeted_sensitivity_calls_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            passes_path = root / "passes.yaml"
            exact_reference = root / "reference.csv"

            with mock.patch("phasebatch.cli.run_budgeted_sensitivity_impl", return_value={"attempted_runs": 4}) as fake_run:
                result = run_budgeted_sensitivity(
                    ["a.c", "b.c"],
                    out_dir,
                    passes_path,
                    objective="ir-inst-count",
                    max_rounds=4,
                    beam_widths=[4, 8],
                    max_states_list=[100, 200],
                    max_batches_per_state=20,
                    batch_frontier_policy="score",
                    validate_batches=True,
                    jobs=8,
                    timeout=10,
                    max_pairs=300,
                    exact_reference=exact_reference,
                    overwrite=True,
                    continue_on_error=True,
                )

        fake_run.assert_called_once_with(
            ["a.c", "b.c"],
            out_dir,
            passes_path,
            objective="ir-inst-count",
            max_rounds=4,
            beam_widths=[4, 8],
            max_states_list=[100, 200],
            max_batches_per_state=20,
            batch_frontier_policy="score",
            validate_batches=True,
            jobs=8,
            timeout=10,
            max_pairs=300,
            exact_reference=exact_reference,
            summarize_components=False,
            overwrite=True,
            continue_on_error=True,
        )
        self.assertEqual(result["attempted_runs"], 4)

    def test_run_core_v1_budgeted_study_calls_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            passes_path = root / "passes.yaml"

            with mock.patch("phasebatch.cli.run_core_v1_budgeted_study_impl", return_value={"programs": 3}) as fake_run:
                result = run_core_v1_budgeted_study(
                    ["a.c", "b.ll"],
                    out_dir,
                    passes_path,
                    objective="ir-inst-count",
                    max_rounds=4,
                    beam_width=4,
                    max_states=500,
                    max_batches_per_state=20,
                    batch_frontier_policy="score",
                    validate_batches=True,
                    jobs=8,
                    timeout=10,
                    max_pairs=300,
                    baseline_methods=["default", "greedy", "random", "batch"],
                    random_trials=20,
                    seed=0,
                    overwrite=True,
                    continue_on_error=True,
                )

        fake_run.assert_called_once_with(
            ["a.c", "b.ll"],
            out_dir,
            passes_path,
            objective="ir-inst-count",
            max_rounds=4,
            beam_width=4,
            max_states=500,
            max_batches_per_state=20,
            batch_frontier_policy="score",
            validate_batches=True,
            jobs=8,
            timeout=10,
            max_pairs=300,
            baseline_methods=["default", "greedy", "random", "batch"],
            random_trials=20,
            seed=0,
            overwrite=True,
            continue_on_error=True,
        )
        self.assertEqual(result["programs"], 3)

    def test_run_select_and_run_exact_reference_calls_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            budgeted_dir = root / "budgeted"
            out_dir = root / "out"
            passes_path = root / "passes.yaml"

            with mock.patch("phasebatch.cli.select_and_run_exact_reference_impl", return_value={"selected_programs": 3}) as fake_run:
                result = run_select_and_run_exact_reference(
                    budgeted_dir,
                    out_dir,
                    passes_path,
                    objective="ir-inst-count",
                    max_rounds=4,
                    max_states=5000,
                    validate_batches=True,
                    jobs=8,
                    timeout=10,
                    max_pairs=300,
                    num_easy=1,
                    num_medium=1,
                    num_hard=1,
                    overwrite=True,
                    continue_on_error=True,
                )

        fake_run.assert_called_once_with(
            budgeted_dir,
            out_dir,
            passes_path,
            objective="ir-inst-count",
            max_rounds=4,
            max_states=5000,
            validate_batches=True,
            jobs=8,
            timeout=10,
            max_pairs=300,
            num_easy=1,
            num_medium=1,
            num_hard=1,
            overwrite=True,
            continue_on_error=True,
        )
        self.assertEqual(result["selected_programs"], 3)

    def test_run_summarize_exact_reduction_study_calls_reporter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            run_dirs = [root / "a" / "optimize", root / "b" / "optimize"]
            root_dir = root / "runs"

            with mock.patch("phasebatch.cli.summarize_exact_reduction_study_impl", return_value={"programs": 2}) as fake_summary:
                result = run_summarize_exact_reduction_study(run_dirs, out_dir, label="exact_r4_core", root_dir=root_dir)

        fake_summary.assert_called_once_with(run_dirs, out_dir, label="exact_r4_core", root_dir=root_dir, summarize_components=False)
        self.assertEqual(result["programs"], 2)

    def test_run_core_v1_case_study_calls_reporter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            method = root / "method.csv"
            reduction = root / "reduction.md"
            budgeted = root / "budgeted.md"
            out_dir = root / "out"
            nbody = root / "nbody.md"
            puzzle = root / "puzzle.md"
            notes = root / "notes.md"

            with mock.patch("phasebatch.cli.summarize_core_v1_case_study_impl", return_value={"programs": 5}) as fake_summary:
                result = run_core_v1_case_study(
                    method,
                    reduction,
                    budgeted,
                    out_dir,
                    label="core_v1_exact_r4",
                    nbody_round_study=nbody,
                    puzzle_case_study=puzzle,
                    extra_notes=notes,
                )

        fake_summary.assert_called_once_with(
            method,
            reduction,
            budgeted,
            out_dir,
            label="core_v1_exact_r4",
            nbody_round_study=nbody,
            puzzle_case_study=puzzle,
            extra_notes=notes,
        )
        self.assertEqual(result["programs"], 5)

    def test_run_component_summary_calls_reporter_for_single_and_multi_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            out_dir = root / "summary"
            run_dirs = [root / "run_a", root / "run_b"]

            with mock.patch("phasebatch.cli.summarize_components_impl", return_value={"states": 2}) as fake_summary:
                single = run_component_summary(run_dir=run_dir)
                multi = run_component_summary(run_dirs=run_dirs, out_dir=out_dir)

        fake_summary.assert_has_calls(
            [
                mock.call(run_dir=run_dir, run_dirs=None, out_dir=None),
                mock.call(run_dir=None, run_dirs=run_dirs, out_dir=out_dir),
            ]
        )
        self.assertEqual(single["states"], 2)
        self.assertEqual(multi["states"], 2)

    def test_run_optimize_batches_passes_split_budgeted_policies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.ll"
            out_dir = root / "out"
            passes_path = root / "passes.yaml"

            with mock.patch("phasebatch.optimizer.optimize_batches", return_value={"states": 1}) as fake_optimize:
                result = run_optimize_batches(
                    input_path,
                    out_dir,
                    passes_path,
                    mode="budgeted",
                    objective="ir-inst-count",
                    max_rounds=2,
                    beam_width=4,
                    max_batches_per_state=5,
                    validate_batches=True,
                    allow_sampled_batches=False,
                    jobs=1,
                    timeout=10,
                    max_pairs=20,
                    batch_selection_policy="score",
                    frontier_selection_policy="objective",
                    verify_final_pipeline=False,
                )

        fake_optimize.assert_called_once()
        kwargs = fake_optimize.call_args.kwargs
        self.assertEqual(kwargs["batch_selection_policy"], "score")
        self.assertEqual(kwargs["frontier_selection_policy"], "objective")
        self.assertEqual(kwargs["verify_final_pipeline"], False)
        self.assertEqual(result["states"], 1)

    def test_run_replay_final_pipeline_calls_replay_layer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()

            with mock.patch(
                "phasebatch.cli.replay_optimized_pipeline",
                return_value={"replay_status": "success", "hashes_match": "true"},
            ) as fake_replay, mock.patch("phasebatch.cli.generate_final_summary") as fake_summary:
                result = run_replay_final_pipeline(run_dir, timeout=7)

        fake_replay.assert_called_once_with(run_dir, timeout=7)
        fake_summary.assert_called_once_with(run_dir)
        self.assertEqual(result["replay_status"], "success")
