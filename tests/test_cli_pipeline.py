import tempfile
import unittest
import os
from pathlib import Path
from unittest import mock

from phasebatch import cli as cli_module
from phasebatch.cli import (
    analyze_state,
    build_parser,
    run_analysis,
    run_batch,
    run_batchify,
    run_compare_baselines,
    run_component_summary,
    run_diagnose_paths,
    run_eval_batches,
    run_evidence_pack,
    run_explore_batches,
    run_optimize_batches,
    run_replay_final_pipeline,
    run_reduction_summary,
    run_visualize_dag,
)


class CliPipelineTests(unittest.TestCase):
    def test_advisor_report_parsers_use_stable_defaults(self) -> None:
        parser = build_parser()
        run_args = parser.parse_args(
            [
                "run-advisor-report-zh",
                "--test-suite-root",
                "suite",
                "--out",
                "out",
                "--passes",
                "passes.yaml",
            ]
        )
        summarize_args = parser.parse_args(
            ["summarize-advisor-report-zh", "--study-dir", "out"]
        )

        self.assertEqual(run_args.num_programs, 15)
        self.assertEqual(run_args.jobs, 8)
        self.assertEqual(run_args.timeout, 15)
        self.assertEqual(run_args.max_pairs, 300)
        self.assertEqual(run_args.pair_testing_mode, "full")
        self.assertEqual(run_args.batch_construction_mode, "pairwise")
        self.assertEqual(run_args.batch_validation_mode, "auto")
        self.assertTrue(run_args.validate_batches)
        self.assertEqual(run_args.opt_backend, "worker")
        self.assertEqual(summarize_args.study_dir, "out")

    def test_opt_backend_parser_defaults_worker_and_accepts_explicit_external(self) -> None:
        parser = build_parser()
        defaults = parser.parse_args(
            ["optimize-batches", "--input", "in.ll", "--out", "out", "--passes", "passes.yaml"]
        )
        external = parser.parse_args(
            [
                "optimize-batches",
                "--input",
                "in.ll",
                "--out",
                "out",
                "--passes",
                "passes.yaml",
                "--opt-backend",
                "external",
            ]
        )
        configured = parser.parse_args(
            [
                "optimize-batches",
                "--input",
                "in.ll",
                "--out",
                "out",
                "--passes",
                "passes.yaml",
                "--opt-backend",
                "worker",
                "--opt-worker",
                "phasebatch-worker.exe",
                "--opt-workers",
                "3",
            ]
        )

        self.assertEqual(defaults.opt_backend, "worker")
        self.assertIsNone(defaults.opt_worker)
        self.assertIsNone(defaults.opt_workers)
        self.assertEqual(external.opt_backend, "external")
        self.assertEqual(configured.opt_backend, "worker")
        self.assertEqual(configured.opt_worker, "phasebatch-worker.exe")
        self.assertEqual(configured.opt_workers, 3)

    def test_opt_backend_parser_reads_environment_defaults(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "PHASEBATCH_OPT_BACKEND": "worker",
                "PHASEBATCH_OPT_WORKER": "configured-worker.exe",
                "PHASEBATCH_OPT_WORKERS": "5",
            },
        ):
            args = build_parser().parse_args(
                ["analyze", "--input", "in.ll", "--out", "out", "--passes", "passes.yaml"]
            )

        self.assertEqual(args.opt_backend, "worker")
        self.assertEqual(args.opt_worker, "configured-worker.exe")
        self.assertEqual(args.opt_workers, 5)

    def test_optimize_staged_supports_worker_backend_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "optimize-staged",
                "--input",
                "in.ll",
                "--manifest",
                "stages.yaml",
                "--out",
                "out",
                "--jobs",
                "4",
                "--opt-backend",
                "auto",
            ]
        )

        self.assertEqual(args.opt_backend, "auto")
        self.assertIsNone(args.opt_workers)

    def test_direct_opt_commands_expose_worker_backend_options(self) -> None:
        parser = build_parser()
        commands = [
            ["audit-passes", "--input", "in.ll", "--passes", "passes.yaml", "--out", "out"],
            ["batchify", "--state-dir", "state"],
            ["compare-baselines", "--run-dir", "run", "--passes", "passes.yaml"],
            ["diagnose-paths", "--run-dir", "run"],
            ["replay-final-pipeline", "--run-dir", "run"],
        ]

        for command in commands:
            with self.subTest(command=command[0]):
                args = parser.parse_args([*command, "--opt-backend", "worker"])
                self.assertEqual(args.opt_backend, "worker")

    def test_main_scopes_selected_backend_around_command(self) -> None:
        with mock.patch("phasebatch.cli.opt_backend_session") as fake_session, \
            mock.patch("phasebatch.cli._run_analyze", return_value=17) as fake_run:
            result = cli_module.main(
                [
                    "analyze",
                    "--input",
                    "in.ll",
                    "--out",
                    "out",
                    "--passes",
                    "passes.yaml",
                    "--jobs",
                    "8",
                    "--opt-backend",
                    "worker",
                    "--opt-worker",
                    "phasebatch-worker.exe",
                    "--opt-workers",
                    "3",
                ]
            )

        self.assertEqual(result, 17)
        fake_session.assert_called_once_with(
            "worker",
            worker_path="phasebatch-worker.exe",
            workers=3,
        )
        fake_run.assert_called_once()

    def test_verify_opt_worker_parser_and_exit_status(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "verify-opt-worker",
                "--inputs",
                "one.ll",
                "two.c",
                "--passes",
                "passes.yaml",
                "--out",
                "out",
                "--opt-worker",
                "phasebatch-worker.exe",
                "--opt-workers",
                "2",
                "--max-passes",
                "7",
            ]
        )

        self.assertEqual(args.inputs, ["one.ll", "two.c"])
        self.assertEqual(args.opt_workers, 2)
        self.assertEqual(args.max_passes, 7)

        with mock.patch(
            "phasebatch.cli.verify_opt_worker_impl",
            return_value={
                "status": "failed",
                "rows": 3,
                "failed_cases": 1,
                "worker_differential_md": "out/worker_differential.md",
            },
        ) as fake_verify:
            exit_code = args.func(args)

        self.assertEqual(exit_code, 1)
        fake_verify.assert_called_once()

    def test_benchmark_opt_worker_parser_and_exit_status(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "benchmark-opt-worker",
                "--input",
                "input.c",
                "--out",
                "out",
                "--opt-worker",
                "phasebatch-worker.exe",
                "--iterations",
                "50",
            ]
        )
        self.assertEqual(args.iterations, 50)

        with mock.patch(
            "phasebatch.cli.benchmark_opt_worker_impl",
            return_value={
                "acceptance_status": "passed",
                "speedup": "5.000",
                "samples": 400,
                "worker_benchmark_md": "out/worker_benchmark.md",
            },
        ) as benchmark:
            exit_code = args.func(args)

        self.assertEqual(exit_code, 0)
        benchmark.assert_called_once()

    def test_optimize_parser_supports_explicit_root_ir_mode(self) -> None:
        parser = build_parser()
        defaults = parser.parse_args(
            ["optimize-batches", "--input", "in.c", "--out", "out", "--passes", "passes.yaml"]
        )
        configured = parser.parse_args(
            [
                "optimize-batches",
                "--input",
                "in.c",
                "--out",
                "out",
                "--passes",
                "passes.yaml",
                "--root-ir-mode",
                "inlinable-unoptimized",
            ]
        )

        self.assertEqual(defaults.root_ir_mode, "legacy-o0")
        self.assertEqual(configured.root_ir_mode, "inlinable-unoptimized")

    def test_optimize_parser_defaults_budgeted_validation_to_all(self) -> None:
        parser = build_parser()
        defaults = parser.parse_args(
            ["optimize-batches", "--input", "in.ll", "--out", "out", "--passes", "passes.yaml"]
        )
        configured = parser.parse_args(
            [
                "optimize-batches",
                "--input",
                "in.ll",
                "--out",
                "out",
                "--passes",
                "passes.yaml",
                "--budgeted-validation-strategy",
                "on-demand",
            ]
        )

        self.assertEqual(defaults.budgeted_validation_strategy, "all")
        self.assertEqual(configured.budgeted_validation_strategy, "on-demand")

    def test_optimize_parser_accepts_only_pairwise_batch_construction(self) -> None:
        parser = build_parser()
        defaults = parser.parse_args(
            ["optimize-batches", "--input", "in.ll", "--out", "out", "--passes", "passes.yaml"]
        )

        self.assertEqual(defaults.batch_construction_mode, "pairwise")
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "optimize-batches",
                    "--input",
                    "in.ll",
                    "--out",
                    "out",
                    "--passes",
                    "passes.yaml",
                    "--batch-construction-mode",
                    "cegar",
                ]
            )

    def test_analyze_state_connects_pipeline_outputs_without_preparing_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
            out_dir = root / "out"

            with mock.patch("phasebatch.state_analysis.profile_passes", return_value=[{"program": "out", "state_id": "S0001", "depth": 1, "parent_state_id": "S0000", "transition_pass": "mem2reg", "state_hash": "s", "pass": "instcombine", "active": "true", "changed_functions": "f", "changed_blocks": "f::entry"}]) as fake_profile, \
                mock.patch("phasebatch.state_analysis.run_pair_tests", return_value=[]):
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

            fake_profile.assert_called_once()
            self.assertEqual(fake_profile.call_args.kwargs["state_id"], "S0001")
            self.assertTrue((out_dir / "pair_cost_summary.csv").exists())
            self.assertTrue((out_dir / "pair_cost_summary.md").exists())

        self.assertEqual(result["program"], "out")
        self.assertEqual(result["state_id"], "S0001")
        self.assertTrue(result["summary_path"].endswith("summary.md"))
        self.assertTrue(result["pair_cost_summary_csv"].endswith("pair_cost_summary.csv"))

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
            batch_construction_mode="pairwise",
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
            llvm_diff = root / "llvm-diff"

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
                    budgeted_validation_strategy="on-demand",
                    max_component_size=17,
                    max_batch_candidates=23,
                    batchify_terminal_states=False,
                    validate_batches=True,
                    allow_sampled_batches=False,
                    pair_testing_mode="lazy",
                    pair_test_budget_per_state=7,
                    pair_priority_policy="effect-size",
                    batch_construction_mode="pairwise",
                    jobs=1,
                    timeout=10,
                    max_pairs=20,
                    batch_selection_policy="score",
                    frontier_selection_policy="objective",
                    verify_final_pipeline=False,
                    llvm_diff=llvm_diff,
                    root_ir_mode="inlinable-unoptimized",
                )

        fake_optimize.assert_called_once()
        kwargs = fake_optimize.call_args.kwargs
        self.assertEqual(kwargs["batch_selection_policy"], "score")
        self.assertEqual(kwargs["frontier_selection_policy"], "objective")
        self.assertEqual(kwargs["budgeted_validation_strategy"], "on-demand")
        self.assertEqual(kwargs["max_component_size"], 17)
        self.assertEqual(kwargs["max_batch_candidates"], 23)
        self.assertEqual(kwargs["batchify_terminal_states"], False)
        self.assertEqual(kwargs["verify_final_pipeline"], False)
        self.assertEqual(kwargs["llvm_diff"], llvm_diff)
        self.assertEqual(kwargs["root_ir_mode"], "inlinable-unoptimized")
        self.assertEqual(kwargs["pair_testing_mode"], "lazy")
        self.assertEqual(kwargs["pair_test_budget_per_state"], 7)
        self.assertEqual(kwargs["pair_priority_policy"], "effect-size")
        self.assertEqual(kwargs["batch_construction_mode"], "pairwise")
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
