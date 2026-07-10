from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

from .advisor_report import (
    run_advisor_report_zh as run_advisor_report_zh_impl,
    summarize_advisor_report_zh as summarize_advisor_report_zh_impl,
)
from .batch_correctness import classify_batch_correctness
from .batch_objective import eval_batch_objectives
from .batcher import build_batch_family, validate_batch_candidates
from .baselines import compare_baselines
from .config import load_passes
from .component_summary import summarize_components as summarize_components_impl
from .coverage import build_coverage_report
from .dag_visualizer import visualize_dag as visualize_dag_impl
from .evidence_pack import export_evidence_pack as export_evidence_pack_impl
from .footprint import build_footprint_overlap
from .final_summary import generate_final_summary
from .normalizer import canonical_hash
from .opt_backend import opt_backend_session
from .pass_audit import audit_passes as audit_passes_impl
from .pass_config import load_pass_registry
from .path_diagnostic import diagnose_paths as diagnose_paths_impl
from .pipeline_replay import replay_optimized_pipeline
from .profiler import validate_passes
from .reduction_summary import summarize_reduction as summarize_reduction_impl
from .report import write_aggregate_report
from .runner import prepare_input_ir
from .state_analysis import analyze_state
from .staged_optimizer import optimize_staged as optimize_staged_impl
from .tools import collect_toolchain, write_metadata
from .worker_benchmark import benchmark_opt_worker as benchmark_opt_worker_impl
from .worker_differential import verify_opt_worker as verify_opt_worker_impl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phasebatch",
        description="LLVM phase-ordering data MVP command line interface.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze one C or LLVM IR input.")
    _add_common_args(analyze)
    analyze.add_argument("--input", required=True, help="Input .c or .ll file.")
    analyze.set_defaults(func=_run_analyze)

    batch = subparsers.add_parser("batch", help="Analyze multiple C or LLVM IR inputs.")
    _add_common_args(batch)
    batch.add_argument("--inputs", required=True, nargs="+", help="Input .c or .ll files.")
    batch.set_defaults(func=_run_batch)

    explore = subparsers.add_parser("explore", help="Explore multiple IR states.")
    _add_common_args(explore)
    explore.add_argument("--input", required=True, help="Input .c or .ll file.")
    explore.add_argument("--max-depth", type=int, default=1, help="Maximum exploration depth.")
    explore.add_argument(
        "--frontier-policy",
        choices=["all-active", "top-k-change", "sensitive-first"],
        default="all-active",
        help="Policy for choosing successor states.",
    )
    explore.add_argument("--top-k", type=int, default=5, help="Frontier cap for top-k policies.")
    explore.set_defaults(func=_run_explore)

    explore_batches_parser = subparsers.add_parser("explore-batches", help="Explore states using batch candidates.")
    _add_common_args(explore_batches_parser)
    explore_batches_parser.add_argument("--input", required=True, help="Input .c or .ll file.")
    explore_batches_parser.add_argument("--max-depth", type=int, default=1, help="Maximum batch exploration depth.")
    explore_batches_parser.add_argument("--max-component-size", type=int, default=10, help="Maximum exact conflict component size.")
    explore_batches_parser.add_argument("--max-batch-candidates", type=int, default=50, help="Maximum batch candidates per state.")
    explore_batches_parser.add_argument("--max-batches-per-state", type=int, default=20, help="Maximum batch candidates to apply per state.")
    explore_batches_parser.add_argument("--max-frontier-states", type=int, default=20, help="Maximum non-duplicate frontier states to keep after each depth.")
    explore_batches_parser.add_argument(
        "--batch-frontier-policy",
        choices=["all", "largest-batch", "certified-first", "diverse-hash"],
        default="all",
        help="Policy for selecting batches and frontier states.",
    )
    explore_batches_parser.add_argument("--validate-batches", action="store_true", help="Validate batch candidates before applying them.")
    explore_batches_parser.add_argument(
        "--allow-sampled-batches",
        action="store_true",
        help="When validating, also apply sampled_same batch candidates.",
    )
    _add_batch_validation_ladder_args(explore_batches_parser)
    _add_batch_construction_args(explore_batches_parser)
    explore_batches_parser.set_defaults(func=_run_explore_batches)

    optimize_batches_parser = subparsers.add_parser("optimize-batches", help="Optimize by executing batch candidates.")
    _add_common_args(optimize_batches_parser)
    optimize_batches_parser.add_argument("--input", required=True, help="Input .c or .ll file.")
    optimize_batches_parser.add_argument(
        "--root-ir-mode",
        choices=["legacy-o0", "inlinable-unoptimized"],
        default="legacy-o0",
        help="Clang mode for materializing C inputs. The legacy mode remains the default.",
    )
    optimize_batches_parser.add_argument(
        "--mode",
        choices=["exact", "budgeted", "auto"],
        default="budgeted",
        help="Optimizer mode. exact expands certified batch DAGs; auto chooses exact or budgeted conservatively.",
    )
    optimize_batches_parser.add_argument(
        "--objective",
        choices=["ir-inst-count"],
        default="ir-inst-count",
        help="Objective used only for final path selection.",
    )
    optimize_batches_parser.add_argument("--max-rounds", type=int, default=5, help="Maximum optimizer rounds.")
    optimize_batches_parser.add_argument("--beam-width", type=int, default=8, help="Maximum frontier states to keep between budgeted rounds.")
    optimize_batches_parser.add_argument("--max-batches-per-state", type=int, default=20, help="Maximum executable batches to apply per state.")
    optimize_batches_parser.add_argument(
        "--budgeted-validation-strategy",
        choices=["all", "on-demand"],
        default="all",
        help="Budgeted validation scope. on-demand stops after enough executable batches are certified.",
    )
    optimize_batches_parser.add_argument("--max-component-size", type=int, default=10, help="Maximum conflict component size to enumerate exactly when building batch candidates.")
    optimize_batches_parser.add_argument("--max-batch-candidates", type=int, default=200, help="Maximum batch candidates to materialize per state.")
    optimize_batches_parser.add_argument(
        "--pair-testing-mode",
        choices=["full", "lazy"],
        default="full",
        help="Pair testing schedule. full preserves exhaustive pair testing; lazy tests prioritized pairs up to a per-state budget.",
    )
    optimize_batches_parser.add_argument(
        "--pair-test-budget-per-state",
        type=int,
        default=0,
        help="Lazy pair testing budget per state. 0 means unlimited.",
    )
    optimize_batches_parser.add_argument(
        "--pair-priority-policy",
        choices=["default", "history", "effect-size", "mixed"],
        default="mixed",
        help="Priority policy for lazy pair testing.",
    )
    optimize_batches_parser.add_argument(
        "--batchify-terminal-states",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build validation and coverage reports for max_rounds terminal frontier states without applying their batches.",
    )
    optimize_batches_parser.add_argument("--max-states", type=int, default=2000, help="Maximum unique states to reach.")
    optimize_batches_parser.add_argument(
        "--selection-seed",
        type=int,
        default=0,
        help="Deterministic seed for budgeted diversity tie-breaks.",
    )
    optimize_batches_parser.add_argument(
        "--batch-frontier-policy",
        choices=["score", "largest-batch", "certified-first", "objective", "diverse"],
        default=None,
        help="Deprecated compatibility alias that sets both batch and frontier selection policies.",
    )
    optimize_batches_parser.add_argument(
        "--batch-selection-policy",
        choices=["score", "largest-batch", "certified-first", "objective", "diverse"],
        default=None,
        help="Budgeted mode policy for selecting executable batch candidates within each state.",
    )
    optimize_batches_parser.add_argument(
        "--frontier-selection-policy",
        choices=["score", "largest-batch", "certified-first", "objective", "diverse"],
        default=None,
        help="Budgeted mode policy for selecting child states for the next beam frontier.",
    )
    optimize_batches_parser.add_argument(
        "--exact-fail-on-incomplete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop exact expansion when completeness assumptions are violated.",
    )
    optimize_batches_parser.add_argument("--validate-batches", action="store_true", help="Validate batch candidates before executing them.")
    optimize_batches_parser.add_argument(
        "--allow-sampled-batches",
        action="store_true",
        help="Allow sampled_same batches to execute; they are not hard-folding proof.",
    )
    _add_batch_validation_ladder_args(optimize_batches_parser)
    _add_batch_construction_args(optimize_batches_parser)
    optimize_batches_parser.add_argument(
        "--run-baselines",
        action="store_true",
        help="After optimization, run baseline comparisons from the root IR.",
    )
    optimize_batches_parser.add_argument(
        "--verify-final-pipeline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replay optimized_pipeline.txt from the root IR and verify it matches final.ll.",
    )
    optimize_batches_parser.add_argument(
        "--llvm-diff",
        default=None,
        help="Path to llvm-diff for structural IR equality fallback. Also configurable via PHASEBATCH_LLVM_DIFF.",
    )
    optimize_batches_parser.add_argument(
        "--keep-ir-artifacts",
        action="store_true",
        help="Keep generated .ll intermediate files after summaries are written.",
    )
    optimize_batches_parser.set_defaults(func=_run_optimize_batches)

    optimize_staged_parser = subparsers.add_parser(
        "optimize-staged",
        help="Optimize through an ordered stage manifest while reusing the batch optimizer per stage.",
    )
    optimize_staged_parser.add_argument("--input", required=True, help="Input .c or .ll file.")
    optimize_staged_parser.add_argument("--manifest", required=True, help="Staged optimization YAML manifest.")
    optimize_staged_parser.add_argument("--out", required=True, help="Output directory.")
    optimize_staged_parser.add_argument("--jobs", type=int, default=1, help="Maximum worker count inside each stage.")
    optimize_staged_parser.add_argument("--timeout", type=int, default=10, help="Per-command timeout in seconds.")
    optimize_staged_parser.add_argument("--keep-ir-artifacts", action="store_true", help="Keep staged IR artifacts.")
    _add_opt_backend_args(optimize_staged_parser)
    optimize_staged_parser.set_defaults(func=_run_optimize_staged)

    audit_passes_parser = subparsers.add_parser("audit-passes", help="Audit pass pipelines against local opt.")
    audit_passes_parser.add_argument("--input", required=True, help="Input .c or .ll file.")
    audit_passes_parser.add_argument("--passes", required=True, help="Path to pass config YAML.")
    audit_passes_parser.add_argument("--out", required=True, help="Output directory.")
    audit_passes_parser.add_argument("--timeout", type=int, default=10, help="Per-command timeout in seconds.")
    audit_passes_parser.add_argument("--jobs", type=int, default=1, help="Parallel worker count.")
    _add_opt_backend_args(audit_passes_parser)
    audit_passes_parser.set_defaults(func=_run_audit_passes)

    summarize_final = subparsers.add_parser("summarize-final", help="Regenerate a final optimize-batches markdown summary.")
    summarize_final.add_argument("--run-dir", required=True, help="Existing optimize-batches output directory.")
    summarize_final.set_defaults(func=_run_summarize_final)

    summarize_reduction = subparsers.add_parser("summarize-reduction", help="Generate search-space reduction evidence for an optimize-batches run.")
    summarize_reduction.add_argument("--run-dir", required=True, help="Existing optimize-batches output directory.")
    summarize_reduction.set_defaults(func=_run_summarize_reduction)

    summarize_components_parser = subparsers.add_parser("summarize-components", help="Summarize conflict graph components and interaction edges from optimize-batches outputs.")
    component_group = summarize_components_parser.add_mutually_exclusive_group(required=True)
    component_group.add_argument("--run-dir", help="Existing single optimize-batches output directory.")
    component_group.add_argument("--run-dirs", nargs="+", help="Existing optimize-batches output directories to aggregate.")
    summarize_components_parser.add_argument("--out", default=None, help="Output directory required with --run-dirs.")
    summarize_components_parser.set_defaults(func=_run_summarize_components)

    evidence_pack = subparsers.add_parser("export-evidence-pack", help="Export selected and executed batch certificates for an optimize-batches run.")
    evidence_pack.add_argument("--run-dir", required=True, help="Existing optimize-batches output directory.")
    evidence_pack.set_defaults(func=_run_export_evidence_pack)

    diagnose_paths = subparsers.add_parser("diagnose-paths", help="Compare a batch optimizer path against baseline paths.")
    diagnose_paths.add_argument("--run-dir", required=True, help="Existing optimize-batches output directory.")
    diagnose_paths.add_argument(
        "--baseline-dir",
        default=None,
        help="Optional directory containing baseline_results.csv and baseline path artifacts.",
    )
    diagnose_paths.add_argument("--timeout", type=int, default=10, help="Per-prefix opt replay timeout in seconds.")
    _add_opt_backend_args(diagnose_paths)
    diagnose_paths.set_defaults(func=_run_diagnose_paths)

    visualize_dag_parser = subparsers.add_parser("visualize-dag", help="Visualize an optimize-batches state DAG.")
    visualize_dag_parser.add_argument("--run-dir", required=True, help="Existing optimize-batches output directory.")
    visualize_dag_parser.add_argument("--out", required=True, help="Output directory for DOT/SVG/PNG and metrics files.")
    visualize_dag_parser.add_argument(
        "--view",
        choices=["all", "selected-neighborhood", "selected-only", "depth-overview"],
        default="all",
        help="DAG view to emphasize.",
    )
    visualize_dag_parser.add_argument(
        "--formats",
        nargs="+",
        choices=["dot", "svg", "png"],
        default=["dot"],
        help="Output graph formats. DOT is always written.",
    )
    visualize_dag_parser.add_argument("--max-full-nodes", type=int, default=200, help="Skip full SVG/PNG rendering above this many states.")
    visualize_dag_parser.add_argument("--include-selected-path", action="store_true", help="Generate selected path/neighborhood graph.")
    visualize_dag_parser.add_argument("--include-depth-overview", action="store_true", help="Generate depth overview graph.")
    visualize_dag_parser.set_defaults(func=_run_visualize_dag)

    batchify = subparsers.add_parser("batchify", help="Build batch candidates for one analyzed state.")
    batchify.add_argument("--state-dir", required=True, help="State directory containing pass_profile.csv and pair_relation.csv.")
    batchify.add_argument("--max-component-size", type=int, default=10, help="Maximum exact conflict component size.")
    batchify.add_argument("--max-batch-candidates", type=int, default=200, help="Maximum global batch candidates to emit.")
    batchify.add_argument("--validate-batches", action="store_true", help="Run opt to validate candidate order hashes.")
    batchify.add_argument(
        "--allow-sampled-batches",
        action="store_true",
        help="Allow sampled_same batches to be marked executable in batch_correctness.csv.",
    )
    _add_batch_validation_ladder_args(batchify)
    _add_opt_backend_args(batchify)
    batchify.set_defaults(func=_run_batchify)

    eval_batches = subparsers.add_parser("eval-batches", help="Evaluate batch transitions with an objective.")
    eval_batches.add_argument("--run-dir", required=True, help="Explore-batches output directory.")
    eval_batches.add_argument(
        "--objective",
        choices=["ir-inst-count"],
        default="ir-inst-count",
        help="Objective to compute for each batch transition.",
    )
    eval_batches.add_argument(
        "--recursive",
        action="store_true",
        help="Evaluate every program subdirectory with batch_state_transitions.csv.",
    )
    eval_batches.set_defaults(func=_run_eval_batches)

    compare_baselines_parser = subparsers.add_parser("compare-baselines", help="Compare final optimized pipeline against baselines.")
    compare_baselines_parser.add_argument("--run-dir", required=True, help="Existing optimize-batches output directory.")
    compare_baselines_parser.add_argument("--passes", required=True, help="Path to pass config YAML.")
    compare_baselines_parser.add_argument(
        "--objective",
        choices=["ir-inst-count"],
        default="ir-inst-count",
        help="Objective used for reporting only.",
    )
    compare_baselines_parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help="Baseline methods to run. Supports comma or space separated tokens: default, greedy, random, batch, all.",
    )
    compare_baselines_parser.add_argument("--max-rounds", type=int, default=2, help="Maximum greedy/random baseline rounds.")
    compare_baselines_parser.add_argument("--random-trials", type=int, default=20, help="Number of random single-pass trials.")
    compare_baselines_parser.add_argument("--seed", type=int, default=0, help="Deterministic random seed.")
    compare_baselines_parser.add_argument("--timeout", type=int, default=10, help="Per-command timeout in seconds.")
    compare_baselines_parser.add_argument("--jobs", type=int, default=1, help="Parallel worker count for profiling baselines.")
    compare_baselines_parser.add_argument(
        "--greedy-allow-nonimproving",
        action="store_true",
        help="Allow greedy single-pass baseline to take active passes that do not improve the objective.",
    )
    compare_baselines_parser.add_argument(
        "--include-default-pipelines",
        action="store_true",
        help="Also try opt default<O2> and default<Oz> pipelines when supported.",
    )
    compare_baselines_parser.add_argument(
        "--include-llvm-defaults",
        action="store_true",
        help="Deprecated alias for --include-default-pipelines.",
    )
    _add_opt_backend_args(compare_baselines_parser)
    compare_baselines_parser.set_defaults(func=_run_compare_baselines)

    replay_final = subparsers.add_parser("replay-final-pipeline", help="Replay optimized_pipeline.txt and compare against final.ll.")
    replay_final.add_argument("--run-dir", required=True, help="Existing optimize-batches output directory.")
    replay_final.add_argument("--timeout", type=int, default=10, help="Per-command timeout in seconds.")
    _add_opt_backend_args(replay_final)
    replay_final.set_defaults(func=_run_replay_final_pipeline)

    verify_worker = subparsers.add_parser(
        "verify-opt-worker",
        help="Compare external opt and the in-process LLVM worker over deterministic pipelines.",
    )
    verify_worker.add_argument("--inputs", required=True, nargs="+", help="Input .c or .ll files.")
    verify_worker.add_argument("--passes", required=True, help="Path to pass config YAML.")
    verify_worker.add_argument("--out", required=True, help="Output directory.")
    verify_worker.add_argument(
        "--opt-worker",
        default=os.environ.get("PHASEBATCH_OPT_WORKER"),
        help="Path to phasebatch-worker. Also configurable with PHASEBATCH_OPT_WORKER.",
    )
    verify_worker.add_argument(
        "--opt-workers",
        type=int,
        default=_environment_opt_workers(),
        help="Long-lived LLVM worker count. Defaults to 1.",
    )
    verify_worker.add_argument("--timeout", type=int, default=10, help="Per-pipeline timeout in seconds.")
    verify_worker.add_argument("--max-passes", type=int, default=None, help="Optional deterministic pass prefix limit.")
    verify_worker.add_argument("--keep-ir-artifacts", action="store_true", help="Keep differential .ll outputs.")
    verify_worker.set_defaults(func=_run_verify_opt_worker)

    benchmark_worker = subparsers.add_parser(
        "benchmark-opt-worker",
        help="Benchmark external opt against the long-lived LLVM worker.",
    )
    benchmark_worker.add_argument("--input", required=True, help="Input .c or .ll file.")
    benchmark_worker.add_argument("--out", required=True, help="Output directory.")
    benchmark_worker.add_argument(
        "--opt-worker",
        default=os.environ.get("PHASEBATCH_OPT_WORKER"),
        help="Path to phasebatch-worker. Also configurable with PHASEBATCH_OPT_WORKER.",
    )
    benchmark_worker.add_argument(
        "--opt-workers",
        type=int,
        default=_environment_opt_workers(),
        help="Long-lived LLVM worker count. Defaults to 1.",
    )
    benchmark_worker.add_argument("--iterations", type=int, default=100, help="Iterations per workload.")
    benchmark_worker.add_argument("--timeout", type=int, default=30, help="Per-operation timeout in seconds.")
    benchmark_worker.add_argument("--keep-ir-artifacts", action="store_true", help="Keep benchmark .ll outputs.")
    benchmark_worker.set_defaults(func=_run_benchmark_opt_worker)

    advisor_report = subparsers.add_parser(
        "run-advisor-report-zh",
        help="Run deterministic SingleSource experiments and generate a Chinese advisor report.",
    )
    advisor_report.add_argument("--test-suite-root", required=True, help="LLVM test-suite root containing SingleSource.")
    advisor_report.add_argument("--out", required=True, help="Advisor study output directory.")
    advisor_report.add_argument("--passes", required=True, help="Stable Phasebatch pass config.")
    advisor_report.add_argument("--benchmark-manifest", default=None, help="Optional explicit benchmark YAML manifest.")
    advisor_report.add_argument("--num-programs", type=int, default=15, help="Number of automatically selected programs.")
    advisor_report.add_argument("--max-source-bytes", type=int, default=200000, help="Maximum source size for automatic selection.")
    advisor_report.add_argument("--selection-seed", type=int, default=0, help="Deterministic benchmark selection seed.")
    advisor_report.add_argument("--resume", action="store_true", help="Reuse successful per-program optimize outputs.")
    advisor_report.add_argument("--overwrite", action="store_true", help="Replace the explicitly selected study directory.")
    advisor_report.add_argument("--continue-on-error", action="store_true", help="Record program failures and continue; worker infrastructure failures still abort.")
    advisor_report.add_argument("--mode", choices=["budgeted", "exact", "auto"], default="budgeted", help="Optimizer search mode.")
    advisor_report.add_argument("--max-rounds", type=int, default=2, help="Maximum optimizer rounds.")
    advisor_report.add_argument("--beam-width", type=int, default=4, help="Budgeted frontier width.")
    advisor_report.add_argument("--max-states", type=int, default=150, help="Maximum reached states per program.")
    advisor_report.add_argument("--max-batches-per-state", type=int, default=10, help="Maximum executable batches per state.")
    advisor_report.add_argument("--budgeted-validation-strategy", choices=["all"], default="all", help="Advisor v1 requires validation of all candidates.")
    advisor_report.add_argument("--pair-testing-mode", choices=["full"], default="full", help="Advisor v1 requires full pair testing.")
    advisor_report.add_argument("--batch-construction-mode", choices=["pairwise"], default="pairwise", help="Advisor v1 requires pairwise construction.")
    advisor_report.add_argument("--batch-validation-mode", choices=["auto"], default="auto", help="Advisor v1 uses automatic validation selection.")
    advisor_report.add_argument("--validate-batches", action="store_true", default=True, help="Validate every candidate before execution; always enabled in Advisor v1.")
    advisor_report.add_argument("--jobs", type=int, default=8, help="Parallel job and default worker count.")
    advisor_report.add_argument("--timeout", type=int, default=15, help="Per-operation timeout in seconds.")
    advisor_report.add_argument("--max-pairs", type=int, default=300, help="Maximum active pass pairs per state.")
    _add_opt_backend_args(advisor_report)
    advisor_report.set_defaults(func=_run_advisor_report_zh)

    summarize_advisor = subparsers.add_parser(
        "summarize-advisor-report-zh",
        help="Regenerate Chinese advisor tables, figures, DAGs, and Markdown from an existing study.",
    )
    summarize_advisor.add_argument("--study-dir", required=True, help="Existing Advisor Data Report study directory.")
    summarize_advisor.set_defaults(func=_run_summarize_advisor_report_zh)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    workers = getattr(args, "opt_workers", None) or getattr(args, "jobs", 1)
    with opt_backend_session(
        getattr(args, "opt_backend", "external"),
        worker_path=getattr(args, "opt_worker", None),
        workers=max(1, workers),
    ):
        return args.func(args)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--passes", required=True, help="Path to pass config YAML.")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel worker count.")
    parser.add_argument("--timeout", type=int, default=10, help="Per-command timeout in seconds.")
    parser.add_argument("--max-pairs", type=int, default=None, help="Maximum active pass pairs to test.")
    _add_opt_backend_args(parser)


def _add_opt_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--opt-backend",
        choices=["external", "worker", "auto"],
        default=os.environ.get("PHASEBATCH_OPT_BACKEND", "worker"),
        help=(
            "LLVM execution backend. Defaults to strict worker mode; use external explicitly "
            "for the legacy opt process path or auto to permit recorded fallback."
        ),
    )
    parser.add_argument(
        "--opt-worker",
        default=os.environ.get("PHASEBATCH_OPT_WORKER"),
        help="Path to phasebatch-worker. Also configurable with PHASEBATCH_OPT_WORKER.",
    )
    parser.add_argument(
        "--opt-workers",
        type=int,
        default=_environment_opt_workers(),
        help="Long-lived LLVM worker count. Defaults to --jobs.",
    )


def _environment_opt_workers() -> int | None:
    value = os.environ.get("PHASEBATCH_OPT_WORKERS")
    if value is None or not value.strip():
        return None
    try:
        workers = int(value)
    except ValueError as exc:
        raise ValueError("PHASEBATCH_OPT_WORKERS must be an integer") from exc
    if workers < 1:
        raise ValueError("PHASEBATCH_OPT_WORKERS must be positive")
    return workers


def _add_batch_validation_ladder_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--batch-validation-mode",
        choices=["auto", "exhaustive", "dag", "bounded", "sampled"],
        default="auto",
        help="Validation ladder mode for batch candidates.",
    )
    parser.add_argument(
        "--max-permutation-factorial",
        type=int,
        default=120,
        help="Largest factorial search space to validate exhaustively.",
    )
    parser.add_argument(
        "--max-validation-sequences",
        type=int,
        default=200,
        help="Maximum validation sequences per batch, including the canonical order.",
    )
    parser.add_argument(
        "--max-validation-dag-nodes",
        type=int,
        default=5000,
        help="Maximum permutation DAG nodes per batch validation.",
    )
    parser.add_argument(
        "--max-validation-dag-edges",
        type=int,
        default=20000,
        help="Maximum permutation DAG edges per batch validation.",
    )
    parser.add_argument(
        "--dump-validation-dag",
        action="store_true",
        help="Write per-batch validation DAG node/edge/DOT files and keep DAG IR artifacts for debugging.",
    )
    parser.add_argument(
        "--validation-dag-selected-only",
        action="store_true",
        help="Reserved compatibility flag for validating only selected DAG batches when a caller supplies such a selection.",
    )
    parser.add_argument(
        "--allow-bounded-validation",
        action="store_true",
        help="Allow bounded_same batches to execute in budgeted modes; they still are not hard-foldable proof.",
    )


def _add_batch_construction_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--batch-construction-mode",
        choices=["pairwise"],
        default="pairwise",
        help="Batch construction strategy. The maintained mainline uses the complete pair matrix.",
    )


def _batch_validation_ladder_kwargs(
    *,
    allow_bounded_validation: bool = False,
    batch_validation_mode: str = "auto",
    max_permutation_factorial: int = 120,
    max_validation_sequences: int = 200,
    max_validation_dag_nodes: int = 5000,
    max_validation_dag_edges: int = 20000,
    dump_validation_dag: bool = False,
    validation_dag_selected_only: bool = False,
) -> dict:
    kwargs = {}
    if allow_bounded_validation:
        kwargs["allow_bounded_validation"] = allow_bounded_validation
    if batch_validation_mode != "auto":
        kwargs["batch_validation_mode"] = batch_validation_mode
    if max_permutation_factorial != 120:
        kwargs["max_permutation_factorial"] = max_permutation_factorial
    if max_validation_sequences != 200:
        kwargs["max_validation_sequences"] = max_validation_sequences
    if max_validation_dag_nodes != 5000:
        kwargs["max_validation_dag_nodes"] = max_validation_dag_nodes
    if max_validation_dag_edges != 20000:
        kwargs["max_validation_dag_edges"] = max_validation_dag_edges
    if dump_validation_dag:
        kwargs["dump_validation_dag"] = dump_validation_dag
    if validation_dag_selected_only:
        kwargs["validation_dag_selected_only"] = validation_dag_selected_only
    return kwargs


def _pair_testing_kwargs(
    *,
    pair_testing_mode: str = "full",
    pair_test_budget_per_state: int = 0,
    pair_priority_policy: str = "mixed",
) -> dict:
    kwargs = {}
    if pair_testing_mode != "full":
        kwargs["pair_testing_mode"] = pair_testing_mode
    if pair_test_budget_per_state != 0:
        kwargs["pair_test_budget_per_state"] = pair_test_budget_per_state
    if pair_priority_policy != "mixed":
        kwargs["pair_priority_policy"] = pair_priority_policy
    return kwargs


def _batch_construction_kwargs(batch_construction_mode: str = "pairwise") -> dict:
    if batch_construction_mode != "pairwise":
        raise ValueError(f"unsupported batch construction mode: {batch_construction_mode}")
    return {"batch_construction_mode": "pairwise"}


def _run_analyze(args: argparse.Namespace) -> int:
    result = run_analysis(
        Path(args.input),
        Path(args.out),
        Path(args.passes),
        jobs=args.jobs,
        timeout=args.timeout,
        max_pairs=args.max_pairs,
    )
    print(
        "analyzed {program}: valid={valid_passes} active={active_passes} "
        "pairs={pair_rows} summary={summary_path}".format(**result)
    )
    return 0


def _run_batch(args: argparse.Namespace) -> int:
    result = run_batch(
        args.inputs,
        Path(args.out),
        Path(args.passes),
        jobs=args.jobs,
        timeout=args.timeout,
        max_pairs=args.max_pairs,
    )
    print(f"batch analyzed {len(result['program_dirs'])} programs: {result['aggregate_summary']}")
    return 0


def _run_explore(args: argparse.Namespace) -> int:
    from .explorer import explore_states

    result = explore_states(
        Path(args.input),
        Path(args.out),
        Path(args.passes),
        jobs=args.jobs,
        timeout=args.timeout,
        max_pairs=args.max_pairs,
        max_depth=args.max_depth,
        frontier_policy=args.frontier_policy,
        top_k=args.top_k,
    )
    print(
        "explored {program}: states={states} transitions={transitions} "
        "states_csv={states_csv}".format(**result)
    )
    return 0


def _run_explore_batches(args: argparse.Namespace) -> int:
    result = run_explore_batches(
        Path(args.input),
        Path(args.out),
        Path(args.passes),
        jobs=args.jobs,
        timeout=args.timeout,
        max_pairs=args.max_pairs,
        max_depth=args.max_depth,
        max_component_size=args.max_component_size,
        max_batch_candidates=args.max_batch_candidates,
        max_batches_per_state=args.max_batches_per_state,
        max_frontier_states=args.max_frontier_states,
        batch_frontier_policy=args.batch_frontier_policy,
        validate_batches=args.validate_batches,
        allow_sampled_batches=args.allow_sampled_batches,
        allow_bounded_validation=args.allow_bounded_validation,
        batch_validation_mode=args.batch_validation_mode,
        max_permutation_factorial=args.max_permutation_factorial,
        max_validation_sequences=args.max_validation_sequences,
        max_validation_dag_nodes=args.max_validation_dag_nodes,
        max_validation_dag_edges=args.max_validation_dag_edges,
        dump_validation_dag=args.dump_validation_dag,
        validation_dag_selected_only=args.validation_dag_selected_only,
        batch_construction_mode=args.batch_construction_mode,
    )
    print(
        "batch-explored {program}: states={states} batch_transitions={batch_transitions} "
        "states_csv={states_csv}".format(**result)
    )
    return 0


def _run_optimize_batches(args: argparse.Namespace) -> int:
    try:
        result = run_optimize_batches(
            Path(args.input),
            Path(args.out),
            Path(args.passes),
            mode=args.mode,
            objective=args.objective,
            max_rounds=args.max_rounds,
            beam_width=args.beam_width,
            max_batches_per_state=args.max_batches_per_state,
            budgeted_validation_strategy=args.budgeted_validation_strategy,
            max_component_size=args.max_component_size,
            max_batch_candidates=args.max_batch_candidates,
            batchify_terminal_states=args.batchify_terminal_states,
            max_states=args.max_states,
            pair_testing_mode=args.pair_testing_mode,
            pair_test_budget_per_state=args.pair_test_budget_per_state,
            pair_priority_policy=args.pair_priority_policy,
            batch_construction_mode=args.batch_construction_mode,
            batch_frontier_policy=args.batch_frontier_policy,
            batch_selection_policy=args.batch_selection_policy,
            frontier_selection_policy=args.frontier_selection_policy,
            selection_seed=args.selection_seed,
            exact_fail_on_incomplete=args.exact_fail_on_incomplete,
            validate_batches=args.validate_batches,
            allow_sampled_batches=args.allow_sampled_batches,
            allow_bounded_validation=args.allow_bounded_validation,
            batch_validation_mode=args.batch_validation_mode,
            max_permutation_factorial=args.max_permutation_factorial,
            max_validation_sequences=args.max_validation_sequences,
            max_validation_dag_nodes=args.max_validation_dag_nodes,
            max_validation_dag_edges=args.max_validation_dag_edges,
            dump_validation_dag=args.dump_validation_dag,
            validation_dag_selected_only=args.validation_dag_selected_only,
            run_baselines=args.run_baselines,
            verify_final_pipeline=args.verify_final_pipeline,
            llvm_diff=Path(args.llvm_diff) if args.llvm_diff else None,
            keep_ir_artifacts=args.keep_ir_artifacts,
            root_ir_mode=args.root_ir_mode,
            jobs=args.jobs,
            timeout=args.timeout,
            max_pairs=args.max_pairs,
        )
    except (NotImplementedError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        "optimized batches: states={states} transitions={batch_transitions} "
        "selected={selected_final_state} final_ll={final_ll}".format(**result)
    )
    return 0


def _run_optimize_staged(args: argparse.Namespace) -> int:
    try:
        result = optimize_staged_impl(
            Path(args.input),
            Path(args.out),
            Path(args.manifest),
            jobs=args.jobs,
            timeout=args.timeout,
            keep_ir_artifacts=args.keep_ir_artifacts,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        "optimized staged pipeline: stages={stages} selected={selected_final_state} "
        "replay_verified={replay_verified} summary={staged_summary_md}".format(**result)
    )
    return 0


def _run_batchify(args: argparse.Namespace) -> int:
    result = run_batchify(
        Path(args.state_dir),
        max_component_size=args.max_component_size,
        max_batch_candidates=args.max_batch_candidates,
        validate_batches=args.validate_batches,
        allow_sampled_batches=args.allow_sampled_batches,
        allow_bounded_validation=args.allow_bounded_validation,
        batch_validation_mode=args.batch_validation_mode,
        max_permutation_factorial=args.max_permutation_factorial,
        max_validation_sequences=args.max_validation_sequences,
        max_validation_dag_nodes=args.max_validation_dag_nodes,
        max_validation_dag_edges=args.max_validation_dag_edges,
        dump_validation_dag=args.dump_validation_dag,
        validation_dag_selected_only=args.validation_dag_selected_only,
    )
    print(
        "batchified {state_id}: candidates={batch_candidates} "
        "summary={batch_summary_md}".format(**result)
    )
    return 0


def _run_eval_batches(args: argparse.Namespace) -> int:
    result = run_eval_batches(Path(args.run_dir), objective=args.objective, recursive=args.recursive)
    if args.recursive:
        print(
            "evaluated batches recursively: programs={program_dirs} rows={rows} "
            "aggregate={aggregate_objective_signal_csv} summary={objective_summary_md}".format(**result)
        )
    else:
        print(
            "evaluated batches: rows={rows} objective_signal={objective_signal_csv} "
            "summary={objective_summary_md}".format(**result)
        )
    return 0


def _run_compare_baselines(args: argparse.Namespace) -> int:
    result = run_compare_baselines(
        Path(args.run_dir),
        Path(args.passes),
        objective=args.objective,
        methods=args.methods,
        max_rounds=args.max_rounds,
        random_trials=args.random_trials,
        seed=args.seed,
        timeout=args.timeout,
        jobs=args.jobs,
        greedy_allow_nonimproving=args.greedy_allow_nonimproving,
        include_default_pipelines=args.include_default_pipelines,
        include_llvm_defaults=args.include_llvm_defaults,
    )
    print(
        "compared baselines: rows={rows} baseline_results={baseline_results_csv} "
        "random_trials={random_baseline_trials_csv}".format(**result)
    )
    return 0


def _run_replay_final_pipeline(args: argparse.Namespace) -> int:
    result = run_replay_final_pipeline(Path(args.run_dir), timeout=args.timeout)
    print(
        "replayed final pipeline: status={replay_status} hashes_match={hashes_match} "
        "csv={pipeline_replay_csv}".format(**result)
    )
    return 0


def _run_verify_opt_worker(args: argparse.Namespace) -> int:
    try:
        result = verify_opt_worker_impl(
            [Path(value) for value in args.inputs],
            Path(args.out),
            Path(args.passes),
            worker_path=Path(args.opt_worker) if args.opt_worker else None,
            workers=args.opt_workers or 1,
            timeout=args.timeout,
            max_passes=args.max_passes,
            keep_ir_artifacts=args.keep_ir_artifacts,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        "verified opt worker: status={status} rows={rows} failed={failed_cases} "
        "summary={worker_differential_md}".format(**result)
    )
    return 0 if result["status"] == "passed" else 1


def _run_benchmark_opt_worker(args: argparse.Namespace) -> int:
    try:
        result = benchmark_opt_worker_impl(
            Path(args.input),
            Path(args.out),
            worker_path=Path(args.opt_worker) if args.opt_worker else None,
            workers=args.opt_workers or 1,
            iterations=args.iterations,
            timeout=args.timeout,
            keep_ir_artifacts=args.keep_ir_artifacts,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        "benchmarked opt worker: acceptance={acceptance_status} speedup={speedup}x "
        "samples={samples} summary={worker_benchmark_md}".format(**result)
    )
    return 0 if result["acceptance_status"] == "passed" else 1


def _run_advisor_report_zh(args: argparse.Namespace) -> int:
    result = run_advisor_report_zh_impl(
        test_suite_root=Path(args.test_suite_root),
        out_dir=Path(args.out),
        passes_path=Path(args.passes),
        benchmark_manifest=Path(args.benchmark_manifest) if args.benchmark_manifest else None,
        num_programs=args.num_programs,
        max_source_bytes=args.max_source_bytes,
        selection_seed=args.selection_seed,
        resume=args.resume,
        overwrite=args.overwrite,
        continue_on_error=args.continue_on_error,
        mode=args.mode,
        max_rounds=args.max_rounds,
        beam_width=args.beam_width,
        max_states=args.max_states,
        max_batches_per_state=args.max_batches_per_state,
        budgeted_validation_strategy=args.budgeted_validation_strategy,
        pair_testing_mode=args.pair_testing_mode,
        batch_construction_mode=args.batch_construction_mode,
        batch_validation_mode=args.batch_validation_mode,
        validate_batches=args.validate_batches,
        jobs=args.jobs,
        timeout=args.timeout,
        max_pairs=args.max_pairs,
        command=[sys.executable, "-m", "phasebatch", *sys.argv[1:]],
    )
    print(
        "导师报告运行完成：程序={programs} 成功={successes} 失败={failures} 输出={study_dir}".format(**result)
    )
    return 0 if result["failures"] == 0 else 1


def _run_summarize_advisor_report_zh(args: argparse.Namespace) -> int:
    result = summarize_advisor_report_zh_impl(Path(args.study_dir))
    print(f"导师报告已重新生成：{result['advisor_report_zh']}")
    return 0


def _run_summarize_final(args: argparse.Namespace) -> int:
    path = generate_final_summary(Path(args.run_dir))
    print(f"wrote {path}")
    return 0


def _run_summarize_reduction(args: argparse.Namespace) -> int:
    result = run_reduction_summary(Path(args.run_dir))
    print(
        "summarized reduction: states={states} by_state={reduction_by_state_csv} "
        "summary={reduction_summary_md}".format(**result)
    )
    return 0


def _run_summarize_components(args: argparse.Namespace) -> int:
    result = run_component_summary(
        run_dir=Path(args.run_dir) if args.run_dir else None,
        run_dirs=[Path(path) for path in args.run_dirs] if args.run_dirs else None,
        out_dir=Path(args.out) if args.out else None,
    )
    print(
        "summarized components: states={states} components={components} "
        "summary={component_summary_md}".format(**result)
    )
    return 0


def _run_export_evidence_pack(args: argparse.Namespace) -> int:
    result = run_evidence_pack(Path(args.run_dir))
    print(
        "exported evidence pack: selected={selected_batches} executed={executed_batches} "
        "report={evidence_pack_md}".format(**result)
    )
    return 0


def _run_diagnose_paths(args: argparse.Namespace) -> int:
    result = run_diagnose_paths(
        Path(args.run_dir),
        baseline_dir=Path(args.baseline_dir) if args.baseline_dir else None,
        timeout=args.timeout,
    )
    print(
        "diagnosed paths: methods={methods} report={path_diagnostic_md}".format(**result)
    )
    return 0


def _run_visualize_dag(args: argparse.Namespace) -> int:
    result = run_visualize_dag(
        Path(args.run_dir),
        Path(args.out),
        view=args.view,
        formats=args.formats,
        max_full_nodes=args.max_full_nodes,
        include_selected_path=args.include_selected_path,
        include_depth_overview=args.include_depth_overview,
    )
    print(
        "visualized DAG: states={unique_states} transitions={transitions} "
        "summary={dag_summary_md}".format(**result)
    )
    return 0


def _run_audit_passes(args: argparse.Namespace) -> int:
    result = run_audit_passes(
        Path(args.input),
        Path(args.passes),
        Path(args.out),
        timeout=args.timeout,
        jobs=args.jobs,
    )
    print(
        "audited passes: total={total_passes} valid={valid_passes} invalid={invalid_passes} "
        "summary={summary_md}".format(**result)
    )
    return 0


def run_reduction_summary(run_dir: Path) -> dict:
    return summarize_reduction_impl(run_dir)


def run_evidence_pack(run_dir: Path) -> dict:
    return export_evidence_pack_impl(run_dir)


def run_visualize_dag(
    run_dir: Path,
    out_dir: Path,
    *,
    view: str,
    formats: list[str],
    max_full_nodes: int,
    include_selected_path: bool,
    include_depth_overview: bool,
) -> dict:
    return visualize_dag_impl(
        run_dir,
        out_dir,
        view=view,
        formats=formats,
        max_full_nodes=max_full_nodes,
        include_selected_path=include_selected_path,
        include_depth_overview=include_depth_overview,
    )


def run_component_summary(
    *,
    run_dir: Path | None = None,
    run_dirs: list[Path] | None = None,
    out_dir: Path | None = None,
) -> dict:
    return summarize_components_impl(run_dir=run_dir, run_dirs=run_dirs, out_dir=out_dir)


def run_diagnose_paths(run_dir: Path, baseline_dir: Path | None = None, timeout: int = 10) -> dict:
    return diagnose_paths_impl(run_dir, baseline_dir=baseline_dir, timeout=timeout)


def run_audit_passes(
    input_path: Path,
    passes_path: Path,
    out_dir: Path,
    *,
    timeout: int = 10,
    jobs: int = 1,
) -> dict:
    return audit_passes_impl(
        input_path,
        passes_path,
        out_dir,
        timeout=timeout,
        jobs=jobs,
    )


def run_batchify(
    state_dir: Path,
    max_component_size: int = 10,
    max_batch_candidates: int = 200,
    validate_batches: bool = False,
    allow_sampled_batches: bool = False,
    allow_bounded_validation: bool = False,
    batch_validation_mode: str = "auto",
    max_permutation_factorial: int = 120,
    max_validation_sequences: int = 200,
    max_validation_dag_nodes: int = 5000,
    max_validation_dag_edges: int = 20000,
    dump_validation_dag: bool = False,
    validation_dag_selected_only: bool = False,
) -> dict:
    state_dir = Path(state_dir)
    result = build_batch_family(
        state_dir,
        max_component_size=max_component_size,
        max_batch_candidates=max_batch_candidates,
    )
    if validate_batches:
        tools = _tool_paths(collect_toolchain())
        validation_kwargs = {
            "timeout": 10,
            "jobs": 1,
        }
        validation_kwargs.update(
            _batch_validation_ladder_kwargs(
                batch_validation_mode=batch_validation_mode,
                max_permutation_factorial=max_permutation_factorial,
                max_validation_sequences=max_validation_sequences,
                max_validation_dag_nodes=max_validation_dag_nodes,
                max_validation_dag_edges=max_validation_dag_edges,
                dump_validation_dag=dump_validation_dag,
                validation_dag_selected_only=validation_dag_selected_only,
            )
        )
        validation = validate_batch_candidates(
            state_dir,
            tools,
            **validation_kwargs,
        )
        result.update(validation)
    correctness_kwargs = {"allow_sampled_batches": allow_sampled_batches}
    correctness_kwargs.update(_batch_validation_ladder_kwargs(allow_bounded_validation=allow_bounded_validation))
    correctness_rows = classify_batch_correctness(state_dir, **correctness_kwargs)
    footprint_rows = build_footprint_overlap(state_dir)
    coverage_rows = build_coverage_report(state_dir)
    result.update(
        {
            "batch_correctness_rows": len(correctness_rows),
            "batch_correctness_csv": str(state_dir / "batch_correctness.csv"),
            "footprint_overlap_rows": len(footprint_rows),
            "footprint_overlap_csv": str(state_dir / "footprint_overlap.csv"),
            "coverage_rows": len(coverage_rows),
            "coverage_report_csv": str(state_dir / "coverage_report.csv"),
            "coverage_summary_csv": str(state_dir / "coverage_summary.csv"),
        }
    )
    return result


def run_eval_batches(run_dir: Path, objective: str = "ir-inst-count", recursive: bool = False) -> dict:
    return eval_batch_objectives(run_dir, objective=objective, recursive=recursive)


def run_compare_baselines(
    run_dir: Path,
    passes_path: Path,
    *,
    objective: str = "ir-inst-count",
    methods: list[str] | None = None,
    max_rounds: int = 2,
    random_trials: int = 20,
    seed: int = 0,
    timeout: int = 10,
    jobs: int = 1,
    greedy_allow_nonimproving: bool = False,
    include_default_pipelines: bool = False,
    include_llvm_defaults: bool = False,
) -> dict:
    return compare_baselines(
        run_dir,
        passes_path,
        objective=objective,
        methods=methods,
        max_rounds=max_rounds,
        random_trials=random_trials,
        seed=seed,
        timeout=timeout,
        jobs=jobs,
        greedy_allow_nonimproving=greedy_allow_nonimproving,
        include_default_pipelines=include_default_pipelines,
        include_llvm_defaults=include_llvm_defaults,
    )


def run_replay_final_pipeline(run_dir: Path, timeout: int = 10) -> dict:
    result = replay_optimized_pipeline(run_dir, timeout=timeout)
    from .pipeline_replay import update_replay_status_artifacts
    from .equality_summary import write_equality_tier_summary

    replay_verified = "true" if result.get("hashes_match") == "true" else "false"
    update_replay_status_artifacts(run_dir, result, replay_verified)
    result.update(write_equality_tier_summary(run_dir))
    generate_final_summary(run_dir)
    return result


def run_optimize_batches(
    input_path: Path,
    out_dir: Path,
    passes_path: Path,
    *,
    mode: str,
    objective: str,
    max_rounds: int,
    beam_width: int = 8,
    max_batches_per_state: int,
    budgeted_validation_strategy: str = "all",
    max_component_size: int = 10,
    max_batch_candidates: int = 200,
    batchify_terminal_states: bool = True,
    validate_batches: bool,
    allow_sampled_batches: bool,
    allow_bounded_validation: bool = False,
    batch_validation_mode: str = "auto",
    max_permutation_factorial: int = 120,
    max_validation_sequences: int = 200,
    max_validation_dag_nodes: int = 5000,
    max_validation_dag_edges: int = 20000,
    dump_validation_dag: bool = False,
    validation_dag_selected_only: bool = False,
    pair_testing_mode: str = "full",
    pair_test_budget_per_state: int = 0,
    pair_priority_policy: str = "mixed",
    batch_construction_mode: str = "pairwise",
    jobs: int,
    timeout: int,
    max_pairs: int | None,
    run_baselines: bool = False,
    batch_selection_policy: str | None = None,
    frontier_selection_policy: str | None = None,
    max_states: int = 2000,
    batch_frontier_policy: str | None = None,
    selection_seed: int = 0,
    exact_fail_on_incomplete: bool = True,
    verify_final_pipeline: bool = True,
    llvm_diff: Path | None = None,
    keep_ir_artifacts: bool = False,
    root_ir_mode: str = "legacy-o0",
) -> dict:
    from .optimizer import optimize_batches as optimize_batches_impl

    kwargs = {
        "mode": mode,
        "objective": objective,
        "max_rounds": max_rounds,
        "beam_width": beam_width,
        "max_batches_per_state": max_batches_per_state,
        "budgeted_validation_strategy": budgeted_validation_strategy,
        "max_component_size": max_component_size,
        "max_batch_candidates": max_batch_candidates,
        "batchify_terminal_states": batchify_terminal_states,
        "max_states": max_states,
        "batch_frontier_policy": batch_frontier_policy,
        "batch_selection_policy": batch_selection_policy,
        "frontier_selection_policy": frontier_selection_policy,
        "selection_seed": selection_seed,
        "exact_fail_on_incomplete": exact_fail_on_incomplete,
        "validate_batches": validate_batches,
        "allow_sampled_batches": allow_sampled_batches,
        "run_baselines": run_baselines,
        "verify_final_pipeline": verify_final_pipeline,
        "llvm_diff": llvm_diff,
        "keep_ir_artifacts": keep_ir_artifacts,
        "root_ir_mode": root_ir_mode,
        "jobs": jobs,
        "timeout": timeout,
        "max_pairs": max_pairs,
    }
    kwargs.update(
        _batch_validation_ladder_kwargs(
            allow_bounded_validation=allow_bounded_validation,
            batch_validation_mode=batch_validation_mode,
            max_permutation_factorial=max_permutation_factorial,
            max_validation_sequences=max_validation_sequences,
            max_validation_dag_nodes=max_validation_dag_nodes,
            max_validation_dag_edges=max_validation_dag_edges,
            dump_validation_dag=dump_validation_dag,
            validation_dag_selected_only=validation_dag_selected_only,
        )
    )
    kwargs.update(
        _pair_testing_kwargs(
            pair_testing_mode=pair_testing_mode,
            pair_test_budget_per_state=pair_test_budget_per_state,
            pair_priority_policy=pair_priority_policy,
        )
    )
    kwargs.update(_batch_construction_kwargs(batch_construction_mode))
    return optimize_batches_impl(input_path, out_dir, passes_path, **kwargs)


def run_explore_batches(
    input_path: Path,
    out_dir: Path,
    passes_path: Path,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
    max_depth: int,
    max_component_size: int,
    max_batch_candidates: int,
    max_batches_per_state: int = 20,
    max_frontier_states: int = 20,
    batch_frontier_policy: str = "all",
    validate_batches: bool = False,
    allow_sampled_batches: bool = False,
    allow_bounded_validation: bool = False,
    batch_validation_mode: str = "auto",
    max_permutation_factorial: int = 120,
    max_validation_sequences: int = 200,
    max_validation_dag_nodes: int = 5000,
    max_validation_dag_edges: int = 20000,
    dump_validation_dag: bool = False,
    validation_dag_selected_only: bool = False,
    batch_construction_mode: str = "pairwise",
) -> dict:
    kwargs = {
        "jobs": jobs,
        "timeout": timeout,
        "max_pairs": max_pairs,
        "max_depth": max_depth,
        "max_component_size": max_component_size,
        "max_batch_candidates": max_batch_candidates,
        "max_batches_per_state": max_batches_per_state,
        "max_frontier_states": max_frontier_states,
        "batch_frontier_policy": batch_frontier_policy,
        "validate_batches": validate_batches,
        "allow_sampled_batches": allow_sampled_batches,
    }
    kwargs.update(
        _batch_validation_ladder_kwargs(
            allow_bounded_validation=allow_bounded_validation,
            batch_validation_mode=batch_validation_mode,
            max_permutation_factorial=max_permutation_factorial,
            max_validation_sequences=max_validation_sequences,
            max_validation_dag_nodes=max_validation_dag_nodes,
            max_validation_dag_edges=max_validation_dag_edges,
            dump_validation_dag=dump_validation_dag,
            validation_dag_selected_only=validation_dag_selected_only,
        )
    )
    kwargs.update(_batch_construction_kwargs(batch_construction_mode))
    return explore_batches(input_path, out_dir, passes_path, **kwargs)


def explore_batches(
    input_path: Path,
    out_dir: Path,
    passes_path: Path,
    *,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
    max_depth: int,
    max_component_size: int,
    max_batch_candidates: int,
    max_batches_per_state: int = 20,
    max_frontier_states: int = 20,
    batch_frontier_policy: str = "all",
    validate_batches: bool = False,
    allow_sampled_batches: bool = False,
    allow_bounded_validation: bool = False,
    batch_validation_mode: str = "auto",
    max_permutation_factorial: int = 120,
    max_validation_sequences: int = 200,
    max_validation_dag_nodes: int = 5000,
    max_validation_dag_edges: int = 20000,
    dump_validation_dag: bool = False,
    validation_dag_selected_only: bool = False,
    batch_construction_mode: str = "pairwise",
) -> dict:
    from .batch_explorer import explore_batches as explore_batches_impl

    return explore_batches_impl(
        input_path,
        out_dir,
        passes_path,
        jobs=jobs,
        timeout=timeout,
        max_pairs=max_pairs,
        max_depth=max_depth,
        max_component_size=max_component_size,
        max_batch_candidates=max_batch_candidates,
        max_batches_per_state=max_batches_per_state,
        max_frontier_states=max_frontier_states,
        batch_frontier_policy=batch_frontier_policy,
        validate_batches=validate_batches,
        allow_sampled_batches=allow_sampled_batches,
        allow_bounded_validation=allow_bounded_validation,
        batch_validation_mode=batch_validation_mode,
        max_permutation_factorial=max_permutation_factorial,
        max_validation_sequences=max_validation_sequences,
        max_validation_dag_nodes=max_validation_dag_nodes,
        max_validation_dag_edges=max_validation_dag_edges,
        dump_validation_dag=dump_validation_dag,
        validation_dag_selected_only=validation_dag_selected_only,
        batch_construction_mode=batch_construction_mode,
    )


def run_analysis(
    input_path: Path,
    out_dir: Path,
    passes_path: Path,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pass_registry = load_pass_registry(passes_path)
    configured_passes = pass_registry.names()

    metadata = collect_toolchain()
    metadata.update(
        {
            "input": str(input_path),
            "out_dir": str(out_dir),
            "pass_config": str(passes_path),
            "configured_pass_count": len(configured_passes),
            "jobs": jobs,
            "timeout": timeout,
            "max_pairs": max_pairs,
        }
    )
    write_metadata(out_dir, metadata)
    tools = _tool_paths(metadata)
    tools["_toolchain_metadata"] = metadata
    tools["_pass_registry"] = pass_registry

    input_ll = prepare_input_ir(Path(input_path), out_dir, tools, timeout)
    state_hash = canonical_hash(input_ll)
    program = out_dir.name
    metadata["state_hash"] = state_hash
    write_metadata(out_dir, metadata)

    valid_passes, invalid_rows = validate_passes(input_ll, configured_passes, tools, out_dir, timeout, pass_registry=pass_registry)

    result = analyze_state(
        input_ll,
        out_dir,
        tools,
        valid_passes=valid_passes,
        invalid_rows=invalid_rows,
        configured_pass_count=len(configured_passes),
        jobs=jobs,
        timeout=timeout,
        max_pairs=max_pairs,
        program=program,
        state_id="S0000",
        depth=0,
        parent_state_id="",
        transition_pass="",
    )

    metadata = _read_metadata(out_dir)
    metadata.update(
        {
            "valid_passes": result.get("valid_passes"),
            "invalid_passes": result.get("invalid_passes"),
            "active_passes": result.get("active_passes"),
            "pair_rows": result.get("pair_rows"),
            "summary": result.get("summary_path"),
            "total_time_ms": result.get("total_time_ms"),
        }
    )
    write_metadata(out_dir, metadata)
    return result


def run_batch(
    inputs: list[str],
    out_dir: Path,
    passes_path: Path,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    expanded = _expand_inputs(inputs)
    program_dirs: list[Path] = []

    for index, input_path in enumerate(expanded):
        program_name = _unique_program_name(input_path, program_dirs, index)
        program_out = out_dir / program_name
        run_analysis(input_path, program_out, passes_path, jobs, timeout, max_pairs)
        program_dirs.append(program_out)

    aggregate = write_aggregate_report(out_dir, program_dirs)
    return {
        "out_dir": str(out_dir),
        "program_dirs": [str(path) for path in program_dirs],
        "aggregate_summary": str(aggregate),
    }


def _tool_paths(metadata: dict) -> dict[str, str]:
    return {
        name: details["path"]
        for name, details in metadata.get("tools", {}).items()
        if details.get("path")
    }


def _read_metadata(out_dir: Path) -> dict:
    path = Path(out_dir) / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _expand_inputs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        matches = sorted(Path(match) for match in glob.glob(item))
        if matches:
            paths.extend(matches)
        elif any(char in item for char in "*?[]"):
            raise RuntimeError(f"input pattern matched no files: {item}")
        else:
            paths.append(Path(item))
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _unique_program_name(input_path: Path, existing_dirs: list[Path], index: int) -> str:
    stem = input_path.stem
    existing = {path.name for path in existing_dirs}
    if stem not in existing:
        return stem
    return f"{stem}_{index}"
