from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

from .batch_correctness import classify_batch_correctness
from .batch_objective import eval_batch_objectives
from .batcher import build_batch_family, validate_batch_candidates
from .baselines import compare_baselines
from .budgeted_sensitivity import run_budgeted_sensitivity as run_budgeted_sensitivity_impl
from .case_studies import export_case_studies
from .config import load_passes
from .component_summary import summarize_components as summarize_components_impl
from .core_v1_budgeted_study import run_core_v1_budgeted_study as run_core_v1_budgeted_study_impl
from .core_v1_case_study import summarize_core_v1_case_study as summarize_core_v1_case_study_impl
from .coverage import build_coverage_report
from .dag_visualizer import visualize_dag as visualize_dag_impl
from .evidence_pack import export_evidence_pack as export_evidence_pack_impl
from .exact_reference import select_and_run_exact_reference as select_and_run_exact_reference_impl
from .exact_reduction_study import summarize_exact_reduction_study as summarize_exact_reduction_study_impl
from .footprint import build_footprint_overlap
from .graph import cluster_distribution_rows, write_cluster_distribution
from .final_summary import generate_final_summary
from .mainline import run_mainline as run_mainline_impl
from .mainline_summary import generate_mainline_summary
from .method_comparison import run_method_comparison as run_method_comparison_impl
from .normalizer import canonical_hash
from .pair_tester import run_pair_tests
from .pass_audit import audit_passes as audit_passes_impl
from .pass_config import PassRegistry, load_pass_registry
from .passset_smoke import run_passset_smoke as run_passset_smoke_impl
from .passset_summary import summarize_passsets as summarize_passsets_impl
from .path_diagnostic import diagnose_paths as diagnose_paths_impl
from .pipeline_replay import replay_optimized_pipeline
from .profiler import profile_passes, validate_passes
from .reduction_study import run_reduction_study as run_reduction_study_impl
from .reduction_summary import summarize_reduction as summarize_reduction_impl
from .relation import annotate_pair_relations, write_pair_relations
from .report import write_aggregate_report, write_per_state_summary, write_summary
from .round_sensitivity import run_round_sensitivity as run_round_sensitivity_impl
from .runner import prepare_input_ir
from .tools import collect_toolchain, write_metadata
from .v2_extension import run_v2_extension_study as run_v2_extension_study_impl
from .v3_loop_smoke import run_v3_loop_smoke as run_v3_loop_smoke_impl


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
    explore_batches_parser.set_defaults(func=_run_explore_batches)

    optimize_batches_parser = subparsers.add_parser("optimize-batches", help="Optimize by executing batch candidates.")
    _add_common_args(optimize_batches_parser)
    optimize_batches_parser.add_argument("--input", required=True, help="Input .c or .ll file.")
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
    optimize_batches_parser.set_defaults(func=_run_optimize_batches)

    round_sensitivity = subparsers.add_parser("run-round-sensitivity", help="Run optimize-batches across max_rounds values and summarize convergence.")
    _add_common_args(round_sensitivity)
    round_sensitivity.add_argument("--input", required=True, help="Input .c or .ll file.")
    round_sensitivity.add_argument("--rounds", type=int, nargs="+", required=True, help="max_rounds values to evaluate, e.g. --rounds 2 3 4.")
    round_sensitivity.add_argument(
        "--mode",
        choices=["exact", "budgeted", "auto"],
        default="exact",
        help="Optimizer mode passed to optimize-batches.",
    )
    round_sensitivity.add_argument(
        "--objective",
        choices=["ir-inst-count"],
        default="ir-inst-count",
        help="Objective used for final path selection.",
    )
    round_sensitivity.add_argument("--beam-width", type=int, default=8, help="Budgeted optimizer beam width.")
    round_sensitivity.add_argument("--max-batches-per-state", type=int, default=20, help="Maximum executable batches per state.")
    round_sensitivity.add_argument("--max-states", type=int, default=2000, help="Maximum unique states to reach.")
    round_sensitivity.add_argument(
        "--batch-frontier-policy",
        choices=["score", "largest-batch", "certified-first", "objective", "diverse"],
        default=None,
        help="Compatibility policy passed to optimize-batches.",
    )
    round_sensitivity.add_argument("--validate-batches", action="store_true", help="Validate batch candidates before executing them.")
    round_sensitivity.add_argument("--overwrite", action="store_true", help="Delete existing round sensitivity output before rerun.")
    round_sensitivity.set_defaults(func=_run_round_sensitivity)

    reduction_study = subparsers.add_parser("run-reduction-study", help="Run optimize/reduction/evidence study over multiple inputs.")
    _add_common_args(reduction_study)
    reduction_study.add_argument("--inputs", required=True, nargs="+", help="Input .c/.ll files or glob patterns.")
    reduction_study.add_argument(
        "--optimizer-mode",
        choices=["exact", "budgeted", "auto"],
        default="exact",
        help="Mode passed to optimize-batches.",
    )
    reduction_study.add_argument(
        "--objective",
        choices=["ir-inst-count"],
        default="ir-inst-count",
        help="Objective used by optimize-batches for final path selection.",
    )
    reduction_study.add_argument("--max-rounds", type=int, default=2, help="Maximum optimize-batches rounds.")
    reduction_study.add_argument("--max-states", type=int, default=5000, help="Maximum optimizer states.")
    reduction_study.add_argument("--validate-batches", action="store_true", help="Validate batch candidates during optimize-batches.")
    reduction_study.add_argument("--summarize-components", action="store_true", help="Also summarize component/conflict-graph evidence for successful runs.")
    reduction_study.add_argument("--overwrite", action="store_true", help="Delete existing reduction study output before rerun.")
    reduction_study.add_argument("--continue-on-error", action="store_true", help="Record failed programs and continue with later inputs.")
    reduction_study.set_defaults(func=_run_reduction_study)

    budgeted_sensitivity = subparsers.add_parser("run-budgeted-sensitivity", help="Run budgeted beam/cap sensitivity experiments.")
    _add_common_args(budgeted_sensitivity)
    budgeted_sensitivity.add_argument("--inputs", required=True, nargs="+", help="Input .c/.ll files or glob patterns.")
    budgeted_sensitivity.add_argument(
        "--objective",
        choices=["ir-inst-count"],
        default="ir-inst-count",
        help="Objective used by optimize-batches for final path selection.",
    )
    budgeted_sensitivity.add_argument("--max-rounds", type=int, default=4, help="Maximum optimize-batches rounds.")
    budgeted_sensitivity.add_argument("--beam-widths", type=int, nargs="+", required=True, help="Budgeted beam widths to evaluate.")
    budgeted_sensitivity.add_argument("--max-states-list", type=int, nargs="+", required=True, help="State caps to evaluate.")
    budgeted_sensitivity.add_argument("--max-batches-per-state", type=int, default=20, help="Maximum executable batches per state.")
    budgeted_sensitivity.add_argument(
        "--batch-frontier-policy",
        choices=["score", "largest-batch", "certified-first", "objective", "diverse"],
        default="score",
        help="Budgeted optimizer policy passed to optimize-batches.",
    )
    budgeted_sensitivity.add_argument("--validate-batches", action="store_true", help="Validate batch candidates during optimize-batches.")
    budgeted_sensitivity.add_argument("--summarize-components", action="store_true", help="Also summarize component/conflict-graph evidence for successful runs.")
    budgeted_sensitivity.add_argument("--exact-reference", default=None, help="Optional exact r4 / baseline reference CSV.")
    budgeted_sensitivity.add_argument("--overwrite", action="store_true", help="Delete existing budgeted sensitivity output before rerun.")
    budgeted_sensitivity.add_argument("--continue-on-error", action="store_true", help="Record failed runs and continue.")
    budgeted_sensitivity.set_defaults(func=_run_budgeted_sensitivity)

    exact_reduction = subparsers.add_parser(
        "summarize-exact-reduction-study",
        help="Summarize reduction and certificate evidence from existing exact optimize-batches runs.",
    )
    exact_reduction.add_argument("--run-dirs", nargs="*", default=[], help="Existing optimize-batches run directories.")
    exact_reduction.add_argument("--root-dir", default=None, help="Optional root directory to recursively search for optimize runs.")
    exact_reduction.add_argument("--out", required=True, help="Output directory for the exact reduction study.")
    exact_reduction.add_argument("--label", default="exact_reduction_study", help="Study label written into the markdown report.")
    exact_reduction.add_argument("--summarize-components", action="store_true", help="Also summarize component/conflict-graph evidence from the same run dirs.")
    exact_reduction.set_defaults(func=_run_summarize_exact_reduction_study)

    core_v1 = subparsers.add_parser("summarize-core-v1-case-study", help="Create the final Core-v1 case-study report from existing results.")
    core_v1.add_argument("--exact-method-summary", required=True, help="Five-program exact r4 method summary CSV or markdown.")
    core_v1.add_argument("--exact-reduction-summary", required=True, help="Exact r4 reduction summary markdown.")
    core_v1.add_argument("--budgeted-sensitivity-summary", required=True, help="Budgeted sensitivity summary markdown.")
    core_v1.add_argument("--out", required=True, help="Output directory for Core-v1 case-study artifacts.")
    core_v1.add_argument("--label", default="core_v1_exact_r4", help="Study label written into the report.")
    core_v1.add_argument("--nbody-round-study", default=None, help="Optional n-body round-depth case-study note.")
    core_v1.add_argument("--puzzle-case-study", default=None, help="Optional puzzle hard-case note.")
    core_v1.add_argument("--extra-notes", default=None, help="Optional extra markdown notes to append.")
    core_v1.set_defaults(func=_run_core_v1_case_study)

    core_v1_budgeted = subparsers.add_parser(
        "run-core-v1-budgeted-study",
        help="Run a Core-v1 budgeted study over multiple inputs.",
    )
    _add_common_args(core_v1_budgeted)
    core_v1_budgeted.add_argument("--inputs", required=True, nargs="+", help="Input .c/.ll files or glob patterns.")
    core_v1_budgeted.add_argument(
        "--objective",
        choices=["ir-inst-count"],
        default="ir-inst-count",
        help="Objective used by optimize-batches and baselines.",
    )
    core_v1_budgeted.add_argument("--max-rounds", type=int, default=4, help="Maximum budgeted optimizer rounds.")
    core_v1_budgeted.add_argument("--beam-width", type=int, default=4, help="Budgeted optimizer beam width.")
    core_v1_budgeted.add_argument("--max-states", type=int, default=500, help="Maximum unique optimizer states.")
    core_v1_budgeted.add_argument("--max-batches-per-state", type=int, default=20, help="Maximum executable batches per state.")
    core_v1_budgeted.add_argument(
        "--batch-frontier-policy",
        choices=["score", "largest-batch", "certified-first", "objective", "diverse"],
        default="score",
        help="Budgeted optimizer policy passed to optimize-batches.",
    )
    core_v1_budgeted.add_argument("--validate-batches", action="store_true", help="Validate batch candidates before execution.")
    core_v1_budgeted.add_argument(
        "--baseline-methods",
        nargs="+",
        default=["default,greedy,random,batch"],
        help="Baseline methods to run, as names or comma-separated groups.",
    )
    core_v1_budgeted.add_argument("--random-trials", type=int, default=20, help="Random single-pass baseline trials.")
    core_v1_budgeted.add_argument("--seed", type=int, default=0, help="Deterministic random baseline seed.")
    core_v1_budgeted.add_argument("--overwrite", action="store_true", help="Delete existing output before rerun.")
    core_v1_budgeted.add_argument("--continue-on-error", action="store_true", help="Record failed programs and continue.")
    core_v1_budgeted.set_defaults(func=_run_core_v1_budgeted_study)

    exact_reference = subparsers.add_parser(
        "select-and-run-exact-reference",
        help="Select exact-reference programs from a budgeted study and run exact mode on them.",
    )
    exact_reference.add_argument("--budgeted-study-dir", required=True, help="Existing Core-v1 budgeted study output directory.")
    exact_reference.add_argument("--out", required=True, help="Output directory.")
    exact_reference.add_argument("--passes", required=True, help="Path to pass config YAML.")
    exact_reference.add_argument(
        "--objective",
        choices=["ir-inst-count"],
        default="ir-inst-count",
        help="Objective used by exact optimize-batches.",
    )
    exact_reference.add_argument("--max-rounds", type=int, default=4, help="Exact optimizer max rounds.")
    exact_reference.add_argument("--max-states", type=int, default=5000, help="Exact optimizer max states.")
    exact_reference.add_argument("--validate-batches", action="store_true", help="Validate batch candidates during exact runs.")
    exact_reference.add_argument("--jobs", type=int, default=1, help="Parallel worker count.")
    exact_reference.add_argument("--timeout", type=int, default=10, help="Per-command timeout in seconds.")
    exact_reference.add_argument("--max-pairs", type=int, default=None, help="Maximum active pass pairs to test.")
    exact_reference.add_argument("--num-easy", type=int, default=2, help="Number of easy reference programs to select.")
    exact_reference.add_argument("--num-medium", type=int, default=2, help="Number of medium reference programs to select.")
    exact_reference.add_argument("--num-hard", type=int, default=3, help="Number of hard reference programs to select.")
    exact_reference.add_argument("--overwrite", action="store_true", help="Delete existing exact-reference output before rerun.")
    exact_reference.add_argument("--continue-on-error", action="store_true", help="Record failed exact runs and continue.")
    exact_reference.set_defaults(func=_run_select_and_run_exact_reference)

    mainline = subparsers.add_parser("run-mainline", help="Run explore-batches over multiple inputs.")
    _add_common_args(mainline)
    mainline.add_argument("--inputs", required=True, nargs="+", help="Input .c/.ll files or glob patterns.")
    mainline.add_argument("--max-depth", type=int, default=1, help="Maximum batch exploration depth.")
    mainline.add_argument("--max-component-size", type=int, default=10, help="Maximum exact conflict component size.")
    mainline.add_argument("--max-batch-candidates", type=int, default=50, help="Maximum batch candidates per state.")
    mainline.add_argument("--max-batches-per-state", type=int, default=20, help="Maximum batch candidates to apply per state.")
    mainline.add_argument("--max-frontier-states", type=int, default=20, help="Maximum non-duplicate frontier states to keep after each depth.")
    mainline.add_argument(
        "--batch-frontier-policy",
        choices=["all", "largest-batch", "certified-first", "diverse-hash"],
        default="all",
        help="Policy for selecting batches and frontier states.",
    )
    mainline.add_argument("--validate-batches", action="store_true", help="Validate batch candidates before applying them.")
    mainline.add_argument("--allow-sampled-batches", action="store_true", help="Also apply sampled_same batches.")
    mainline.add_argument(
        "--eval-objective",
        choices=["ir-inst-count"],
        default=None,
        help="After the mainline run, recursively evaluate batch transitions with this objective.",
    )
    mainline.add_argument("--overwrite", action="store_true", help="Delete existing per-program output directories before rerun.")
    mainline.add_argument("--continue-on-error", action="store_true", help="Record failed programs and continue with later inputs.")
    mainline.set_defaults(func=_run_mainline)

    method_comparison = subparsers.add_parser("run-method-comparison", help="Run optimize-batches and method baselines over multiple inputs.")
    _add_common_args(method_comparison)
    method_comparison.add_argument("--inputs", required=True, nargs="+", help="Input .c/.ll files or glob patterns.")
    method_comparison.add_argument(
        "--optimizer-mode",
        choices=["exact", "budgeted", "auto"],
        default="budgeted",
        help="Mode passed to optimize-batches.",
    )
    method_comparison.add_argument(
        "--objective",
        choices=["ir-inst-count"],
        default="ir-inst-count",
        help="Objective used for optimization and method comparison.",
    )
    method_comparison.add_argument("--max-rounds", type=int, default=3, help="Maximum optimize-batches rounds.")
    method_comparison.add_argument("--beam-width", type=int, default=4, help="Budgeted optimizer beam width.")
    method_comparison.add_argument("--max-states", type=int, default=500, help="Maximum optimizer states.")
    method_comparison.add_argument("--max-batches-per-state", type=int, default=10, help="Maximum executable batches per state.")
    method_comparison.add_argument(
        "--batch-frontier-policy",
        choices=["score", "largest-batch", "certified-first", "objective", "diverse"],
        default="score",
        help="Compatibility policy passed to optimize-batches.",
    )
    method_comparison.add_argument("--validate-batches", action="store_true", help="Validate batch candidates during optimize-batches.")
    method_comparison.add_argument("--baseline-max-rounds", type=int, default=5, help="Maximum greedy/random baseline rounds.")
    method_comparison.add_argument("--random-trials", type=int, default=20, help="Number of random baseline trials.")
    method_comparison.add_argument("--seed", type=int, default=0, help="Deterministic random baseline seed.")
    method_comparison.add_argument("--include-default-pipelines", action="store_true", help="Also run default<O2> and default<Oz> when supported.")
    method_comparison.add_argument("--overwrite", action="store_true", help="Delete existing per-program output directories before rerun.")
    method_comparison.add_argument("--continue-on-error", action="store_true", help="Record failed programs and continue with later inputs.")
    method_comparison.set_defaults(func=_run_method_comparison)

    passset_smoke = subparsers.add_parser("run-passset-smoke", help="Audit and optimize multiple pass-set configs over smoke inputs.")
    passset_smoke.add_argument("--inputs", required=True, nargs="+", help="Input .c/.ll files or glob patterns.")
    passset_smoke.add_argument("--passsets", required=True, nargs="+", help="Pass config YAML files to compare.")
    passset_smoke.add_argument("--out", required=True, help="Output directory.")
    passset_smoke.add_argument(
        "--optimizer-mode",
        choices=["exact", "budgeted", "auto"],
        default="exact",
        help="Mode passed to optimize-batches.",
    )
    passset_smoke.add_argument(
        "--objective",
        choices=["ir-inst-count"],
        default="ir-inst-count",
        help="Objective used for optimization reporting.",
    )
    passset_smoke.add_argument("--max-rounds", type=int, default=2, help="Maximum optimize-batches rounds.")
    passset_smoke.add_argument("--beam-width", type=int, default=8, help="Budgeted optimizer beam width.")
    passset_smoke.add_argument("--max-states", type=int, default=5000, help="Maximum optimizer states.")
    passset_smoke.add_argument("--max-batches-per-state", type=int, default=20, help="Maximum executable batches per state.")
    passset_smoke.add_argument(
        "--batch-frontier-policy",
        choices=["score", "largest-batch", "certified-first", "objective", "diverse"],
        default="score",
        help="Compatibility policy passed to optimize-batches.",
    )
    passset_smoke.add_argument("--validate-batches", action="store_true", help="Validate batch candidates during optimize-batches.")
    passset_smoke.add_argument("--jobs", type=int, default=1, help="Parallel worker count.")
    passset_smoke.add_argument("--timeout", type=int, default=10, help="Per-command timeout in seconds.")
    passset_smoke.add_argument("--max-pairs", type=int, default=None, help="Maximum active pass pairs to test.")
    passset_smoke.add_argument("--overwrite", action="store_true", help="Delete existing smoke output before rerun.")
    passset_smoke.add_argument("--continue-on-error", action="store_true", help="Record failed passset runs and continue.")
    passset_smoke.set_defaults(func=_run_passset_smoke)

    v2_extension = subparsers.add_parser("run-v2-extension-study", help="Run the Core-v1 vs scalar-v2 extension study.")
    v2_extension.add_argument("--inputs", required=True, nargs="+", help="Input .c/.ll files or glob patterns.")
    v2_extension.add_argument("--out", required=True, help="Output directory.")
    v2_extension.add_argument("--v1-passes", required=True, help="Core-v1 pass config YAML.")
    v2_extension.add_argument("--v2-passes", required=True, help="Scalar-v2 pass config YAML.")
    v2_extension.add_argument(
        "--objective",
        choices=["ir-inst-count"],
        default="ir-inst-count",
        help="Objective used by optimize-batches.",
    )
    v2_extension.add_argument("--max-rounds", type=int, default=4, help="Maximum budgeted optimizer rounds.")
    v2_extension.add_argument("--beam-width", type=int, default=4, help="Budgeted optimizer beam width.")
    v2_extension.add_argument("--max-states", type=int, default=500, help="Maximum optimizer states.")
    v2_extension.add_argument("--max-batches-per-state", type=int, default=20, help="Maximum executable batches per state.")
    v2_extension.add_argument(
        "--batch-frontier-policy",
        choices=["score", "largest-batch", "certified-first", "objective", "diverse"],
        default="score",
        help="Budgeted optimizer policy passed to optimize-batches.",
    )
    v2_extension.add_argument("--validate-batches", action="store_true", help="Validate batch candidates during optimize-batches.")
    v2_extension.add_argument("--jobs", type=int, default=1, help="Parallel worker count.")
    v2_extension.add_argument("--timeout", type=int, default=10, help="Per-command timeout in seconds.")
    v2_extension.add_argument("--max-pairs", type=int, default=None, help="Maximum active pass pairs to test.")
    v2_extension.add_argument("--random-trials", type=int, default=20, help="Recorded random-trials setting for comparability.")
    v2_extension.add_argument("--seed", type=int, default=0, help="Recorded random seed setting for comparability.")
    v2_extension.add_argument("--overwrite", action="store_true", help="Delete existing v2 extension output before rerun.")
    v2_extension.add_argument("--continue-on-error", action="store_true", help="Record failures and continue.")
    v2_extension.set_defaults(func=_run_v2_extension_study)

    v3_loop_smoke = subparsers.add_parser("run-v3-loop-smoke", help="Audit and optimize v3 middle-end loop pass sets over loop-heavy inputs.")
    v3_loop_smoke.add_argument("--inputs", required=True, nargs="+", help="Loop-heavy input .c/.ll files or glob patterns.")
    v3_loop_smoke.add_argument("--out", required=True, help="Output directory.")
    v3_loop_smoke.add_argument("--passes", required=True, help="Path to v3 pass config YAML.")
    v3_loop_smoke.add_argument(
        "--optimizer-mode",
        choices=["exact", "budgeted", "auto"],
        default="budgeted",
        help="Mode passed to optimize-batches.",
    )
    v3_loop_smoke.add_argument(
        "--objective",
        choices=["ir-inst-count"],
        default="ir-inst-count",
        help="Objective used for optimization reporting.",
    )
    v3_loop_smoke.add_argument("--max-rounds", type=int, default=3, help="Maximum optimize-batches rounds.")
    v3_loop_smoke.add_argument("--beam-width", type=int, default=4, help="Budgeted optimizer beam width.")
    v3_loop_smoke.add_argument("--max-states", type=int, default=800, help="Maximum optimizer states.")
    v3_loop_smoke.add_argument("--max-batches-per-state", type=int, default=12, help="Maximum executable batches per state.")
    v3_loop_smoke.add_argument(
        "--batch-frontier-policy",
        choices=["score", "largest-batch", "certified-first", "objective", "diverse"],
        default="score",
        help="Compatibility policy passed to optimize-batches.",
    )
    v3_loop_smoke.add_argument("--validate-batches", action="store_true", help="Validate batch candidates during optimize-batches.")
    v3_loop_smoke.add_argument("--jobs", type=int, default=1, help="Parallel worker count.")
    v3_loop_smoke.add_argument("--timeout", type=int, default=10, help="Per-command timeout in seconds.")
    v3_loop_smoke.add_argument("--max-pairs", type=int, default=None, help="Maximum active pass pairs to test.")
    v3_loop_smoke.add_argument("--overwrite", action="store_true", help="Delete existing v3 loop smoke output before rerun.")
    v3_loop_smoke.add_argument("--continue-on-error", action="store_true", help="Record failed programs and continue.")
    v3_loop_smoke.set_defaults(func=_run_v3_loop_smoke)

    summarize_passsets = subparsers.add_parser("summarize-passsets", help="Generate a unified v1/v2/v3 pass-set comparison report.")
    summarize_passsets.add_argument("--inputs", required=True, nargs="+", help="Passset smoke, v3 loop smoke, or direct run directories.")
    summarize_passsets.add_argument("--out", required=True, help="Output directory for report artifacts.")
    summarize_passsets.set_defaults(func=_run_summarize_passsets)

    audit_passes_parser = subparsers.add_parser("audit-passes", help="Audit pass pipelines against local opt.")
    audit_passes_parser.add_argument("--input", required=True, help="Input .c or .ll file.")
    audit_passes_parser.add_argument("--passes", required=True, help="Path to pass config YAML.")
    audit_passes_parser.add_argument("--out", required=True, help="Output directory.")
    audit_passes_parser.add_argument("--timeout", type=int, default=10, help="Per-command timeout in seconds.")
    audit_passes_parser.add_argument("--jobs", type=int, default=1, help="Parallel worker count.")
    audit_passes_parser.set_defaults(func=_run_audit_passes)

    summarize_mainline = subparsers.add_parser("summarize-mainline", help="Regenerate a mainline markdown summary.")
    summarize_mainline.add_argument("--run-dir", required=True, help="Existing run-mainline output directory.")
    summarize_mainline.set_defaults(func=_run_summarize_mainline)

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

    case_studies = subparsers.add_parser("export-case-studies", help="Export representative per-program case studies.")
    case_studies.add_argument("--run-dir", required=True, help="Existing run-mainline output directory.")
    case_studies.add_argument("--max-pairs", type=int, default=20, help="Maximum pair relation rows per case study.")
    case_studies.add_argument("--max-batches", type=int, default=10, help="Maximum batch candidate rows per case study.")
    case_studies.set_defaults(func=_run_export_case_studies)

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
    compare_baselines_parser.set_defaults(func=_run_compare_baselines)

    replay_final = subparsers.add_parser("replay-final-pipeline", help="Replay optimized_pipeline.txt and compare against final.ll.")
    replay_final.add_argument("--run-dir", required=True, help="Existing optimize-batches output directory.")
    replay_final.add_argument("--timeout", type=int, default=10, help="Per-command timeout in seconds.")
    replay_final.set_defaults(func=_run_replay_final_pipeline)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--passes", required=True, help="Path to pass config YAML.")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel worker count.")
    parser.add_argument("--timeout", type=int, default=10, help="Per-command timeout in seconds.")
    parser.add_argument("--max-pairs", type=int, default=None, help="Maximum active pass pairs to test.")


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
            max_states=args.max_states,
            batch_frontier_policy=args.batch_frontier_policy,
            batch_selection_policy=args.batch_selection_policy,
            frontier_selection_policy=args.frontier_selection_policy,
            selection_seed=args.selection_seed,
            exact_fail_on_incomplete=args.exact_fail_on_incomplete,
            validate_batches=args.validate_batches,
            allow_sampled_batches=args.allow_sampled_batches,
            run_baselines=args.run_baselines,
            verify_final_pipeline=args.verify_final_pipeline,
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


def _run_batchify(args: argparse.Namespace) -> int:
    result = run_batchify(
        Path(args.state_dir),
        max_component_size=args.max_component_size,
        max_batch_candidates=args.max_batch_candidates,
        validate_batches=args.validate_batches,
        allow_sampled_batches=args.allow_sampled_batches,
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


def _run_summarize_mainline(args: argparse.Namespace) -> int:
    path = generate_mainline_summary(Path(args.run_dir))
    print(f"wrote {path}")
    return 0


def _run_summarize_final(args: argparse.Namespace) -> int:
    path = generate_final_summary(Path(args.run_dir))
    print(f"wrote {path}")
    return 0


def _run_export_case_studies(args: argparse.Namespace) -> int:
    result = export_case_studies(Path(args.run_dir), max_pairs=args.max_pairs, max_batches=args.max_batches)
    print(
        "exported case studies: count={case_studies} index={case_studies_index}".format(**result)
    )
    return 0


def _run_mainline(args: argparse.Namespace) -> int:
    result = run_mainline(
        args.inputs,
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
        eval_objective=args.eval_objective,
        overwrite=args.overwrite,
        continue_on_error=args.continue_on_error,
    )
    print(
        "mainline run: programs={programs} successes={successes} failures={failures} "
        "runs_csv={mainline_runs_csv}".format(**result)
    )
    return 0


def _run_method_comparison(args: argparse.Namespace) -> int:
    result = run_method_comparison(
        args.inputs,
        Path(args.out),
        Path(args.passes),
        optimizer_mode=args.optimizer_mode,
        objective=args.objective,
        max_rounds=args.max_rounds,
        beam_width=args.beam_width,
        max_states=args.max_states,
        max_batches_per_state=args.max_batches_per_state,
        batch_frontier_policy=args.batch_frontier_policy,
        validate_batches=args.validate_batches,
        baseline_max_rounds=args.baseline_max_rounds,
        random_trials=args.random_trials,
        seed=args.seed,
        include_default_pipelines=args.include_default_pipelines,
        jobs=args.jobs,
        timeout=args.timeout,
        max_pairs=args.max_pairs,
        overwrite=args.overwrite,
        continue_on_error=args.continue_on_error,
    )
    print(
        "method comparison run: programs={programs} successes={successes} failures={failures} "
        "results_csv={method_comparison_results_csv}".format(**result)
    )
    return 0


def _run_round_sensitivity(args: argparse.Namespace) -> int:
    result = run_round_sensitivity(
        Path(args.input),
        Path(args.out),
        Path(args.passes),
        rounds=args.rounds,
        optimizer_mode=args.mode,
        objective=args.objective,
        beam_width=args.beam_width,
        max_states=args.max_states,
        max_batches_per_state=args.max_batches_per_state,
        batch_frontier_policy=args.batch_frontier_policy,
        validate_batches=args.validate_batches,
        jobs=args.jobs,
        timeout=args.timeout,
        max_pairs=args.max_pairs,
        overwrite=args.overwrite,
    )
    print(
        "round sensitivity run: rows={rows} csv={round_sensitivity_csv} "
        "summary={round_sensitivity_md}".format(**result)
    )
    return 0


def _run_reduction_study(args: argparse.Namespace) -> int:
    result = run_reduction_study(
        args.inputs,
        Path(args.out),
        Path(args.passes),
        optimizer_mode=args.optimizer_mode,
        objective=args.objective,
        max_rounds=args.max_rounds,
        max_states=args.max_states,
        validate_batches=args.validate_batches,
        jobs=args.jobs,
        timeout=args.timeout,
        max_pairs=args.max_pairs,
        summarize_components=args.summarize_components,
        overwrite=args.overwrite,
        continue_on_error=args.continue_on_error,
    )
    print(
        "reduction study run: programs={programs} successes={successes} failures={failures} "
        "summary={reduction_study_summary_md}".format(**result)
    )
    return 0


def _run_budgeted_sensitivity(args: argparse.Namespace) -> int:
    result = run_budgeted_sensitivity(
        args.inputs,
        Path(args.out),
        Path(args.passes),
        objective=args.objective,
        max_rounds=args.max_rounds,
        beam_widths=args.beam_widths,
        max_states_list=args.max_states_list,
        max_batches_per_state=args.max_batches_per_state,
        batch_frontier_policy=args.batch_frontier_policy,
        validate_batches=args.validate_batches,
        jobs=args.jobs,
        timeout=args.timeout,
        max_pairs=args.max_pairs,
        exact_reference=Path(args.exact_reference) if args.exact_reference else None,
        summarize_components=args.summarize_components,
        overwrite=args.overwrite,
        continue_on_error=args.continue_on_error,
    )
    print(
        "budgeted sensitivity run: attempted={attempted_runs} successes={successes} failures={failures} "
        "summary={budgeted_sensitivity_summary_md}".format(**result)
    )
    return 0


def _run_summarize_exact_reduction_study(args: argparse.Namespace) -> int:
    result = run_summarize_exact_reduction_study(
        [Path(path) for path in args.run_dirs],
        Path(args.out),
        label=args.label,
        root_dir=Path(args.root_dir) if args.root_dir else None,
        summarize_components=args.summarize_components,
    )
    print(
        "summarized exact reduction study: programs={programs} successes={successes} "
        "failures={failures} summary={exact_reduction_summary_md}".format(**result)
    )
    return 0


def _run_core_v1_case_study(args: argparse.Namespace) -> int:
    result = run_core_v1_case_study(
        Path(args.exact_method_summary),
        Path(args.exact_reduction_summary),
        Path(args.budgeted_sensitivity_summary),
        Path(args.out),
        label=args.label,
        nbody_round_study=Path(args.nbody_round_study) if args.nbody_round_study else None,
        puzzle_case_study=Path(args.puzzle_case_study) if args.puzzle_case_study else None,
        extra_notes=Path(args.extra_notes) if args.extra_notes else None,
    )
    print(
        "summarized Core-v1 case study: programs={programs} missing_inputs={missing_inputs} "
        "summary={core_v1_case_study_summary_md}".format(**result)
    )
    return 0


def _run_core_v1_budgeted_study(args: argparse.Namespace) -> int:
    result = run_core_v1_budgeted_study(
        args.inputs,
        Path(args.out),
        Path(args.passes),
        objective=args.objective,
        max_rounds=args.max_rounds,
        beam_width=args.beam_width,
        max_states=args.max_states,
        max_batches_per_state=args.max_batches_per_state,
        batch_frontier_policy=args.batch_frontier_policy,
        validate_batches=args.validate_batches,
        jobs=args.jobs,
        timeout=args.timeout,
        max_pairs=args.max_pairs,
        baseline_methods=args.baseline_methods,
        random_trials=args.random_trials,
        seed=args.seed,
        overwrite=args.overwrite,
        continue_on_error=args.continue_on_error,
    )
    print(
        "Core-v1 budgeted study: programs={programs} successes={successes} failures={failures} "
        "summary={budgeted_study_summary_md}".format(**result)
    )
    return 0


def _run_select_and_run_exact_reference(args: argparse.Namespace) -> int:
    result = run_select_and_run_exact_reference(
        Path(args.budgeted_study_dir),
        Path(args.out),
        Path(args.passes),
        objective=args.objective,
        max_rounds=args.max_rounds,
        max_states=args.max_states,
        validate_batches=args.validate_batches,
        jobs=args.jobs,
        timeout=args.timeout,
        max_pairs=args.max_pairs,
        num_easy=args.num_easy,
        num_medium=args.num_medium,
        num_hard=args.num_hard,
        overwrite=args.overwrite,
        continue_on_error=args.continue_on_error,
    )
    print(
        "exact reference study: selected={selected_programs} successes={successes} failures={failures} "
        "summary={exact_reference_summary_md}".format(**result)
    )
    return 0


def _run_passset_smoke(args: argparse.Namespace) -> int:
    result = run_passset_smoke(
        args.inputs,
        [Path(path) for path in args.passsets],
        Path(args.out),
        optimizer_mode=args.optimizer_mode,
        objective=args.objective,
        max_rounds=args.max_rounds,
        beam_width=args.beam_width,
        max_states=args.max_states,
        max_batches_per_state=args.max_batches_per_state,
        batch_frontier_policy=args.batch_frontier_policy,
        validate_batches=args.validate_batches,
        jobs=args.jobs,
        timeout=args.timeout,
        max_pairs=args.max_pairs,
        overwrite=args.overwrite,
        continue_on_error=args.continue_on_error,
    )
    print(
        "passset smoke run: runs={runs} successes={successes} failures={failures} "
        "comparison_csv={passset_comparison_csv}".format(**result)
    )
    return 0


def _run_v2_extension_study(args: argparse.Namespace) -> int:
    result = run_v2_extension_study(
        args.inputs,
        Path(args.out),
        Path(args.v1_passes),
        Path(args.v2_passes),
        objective=args.objective,
        max_rounds=args.max_rounds,
        beam_width=args.beam_width,
        max_states=args.max_states,
        max_batches_per_state=args.max_batches_per_state,
        batch_frontier_policy=args.batch_frontier_policy,
        validate_batches=args.validate_batches,
        jobs=args.jobs,
        timeout=args.timeout,
        max_pairs=args.max_pairs,
        random_trials=args.random_trials,
        seed=args.seed,
        overwrite=args.overwrite,
        continue_on_error=args.continue_on_error,
    )
    print(
        "v2 extension study: programs={programs} successes={successes} failures={failures} "
        "comparison_csv={v2_extension_comparison_csv}".format(**result)
    )
    return 0


def _run_v3_loop_smoke(args: argparse.Namespace) -> int:
    result = run_v3_loop_smoke(
        args.inputs,
        Path(args.out),
        Path(args.passes),
        optimizer_mode=args.optimizer_mode,
        objective=args.objective,
        max_rounds=args.max_rounds,
        beam_width=args.beam_width,
        max_states=args.max_states,
        max_batches_per_state=args.max_batches_per_state,
        batch_frontier_policy=args.batch_frontier_policy,
        validate_batches=args.validate_batches,
        jobs=args.jobs,
        timeout=args.timeout,
        max_pairs=args.max_pairs,
        overwrite=args.overwrite,
        continue_on_error=args.continue_on_error,
    )
    print(
        "v3 loop smoke run: programs={programs_attempted} successes={successes} failures={failures} "
        "summary_csv={v3_loop_summary_csv}".format(**result)
    )
    return 0


def _run_summarize_passsets(args: argparse.Namespace) -> int:
    result = run_summarize_passsets([Path(path) for path in args.inputs], Path(args.out))
    print(
        "summarized passsets: rows={matrix_rows} failures={failures} "
        "report={passset_comparison_report_md}".format(**result)
    )
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


def run_mainline(
    inputs: list[str],
    out_dir: Path,
    passes_path: Path,
    *,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
    max_depth: int,
    max_component_size: int,
    max_batch_candidates: int,
    max_batches_per_state: int,
    max_frontier_states: int = 20,
    batch_frontier_policy: str = "all",
    validate_batches: bool = False,
    allow_sampled_batches: bool = False,
    eval_objective: str | None = None,
    overwrite: bool = False,
    continue_on_error: bool = False,
) -> dict:
    return run_mainline_impl(
        inputs,
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
        eval_objective=eval_objective,
        overwrite=overwrite,
        continue_on_error=continue_on_error,
    )


def run_method_comparison(
    inputs: list[str],
    out_dir: Path,
    passes_path: Path,
    *,
    optimizer_mode: str,
    objective: str,
    max_rounds: int,
    beam_width: int,
    max_states: int,
    max_batches_per_state: int,
    batch_frontier_policy: str | None,
    validate_batches: bool,
    baseline_max_rounds: int,
    random_trials: int,
    seed: int,
    include_default_pipelines: bool,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
    overwrite: bool = False,
    continue_on_error: bool = False,
) -> dict:
    return run_method_comparison_impl(
        inputs,
        out_dir,
        passes_path,
        optimizer_mode=optimizer_mode,
        objective=objective,
        max_rounds=max_rounds,
        beam_width=beam_width,
        max_states=max_states,
        max_batches_per_state=max_batches_per_state,
        batch_frontier_policy=batch_frontier_policy,
        validate_batches=validate_batches,
        baseline_max_rounds=baseline_max_rounds,
        random_trials=random_trials,
        seed=seed,
        include_default_pipelines=include_default_pipelines,
        jobs=jobs,
        timeout=timeout,
        max_pairs=max_pairs,
        overwrite=overwrite,
        continue_on_error=continue_on_error,
    )


def run_round_sensitivity(
    input_path: Path,
    out_dir: Path,
    passes_path: Path,
    *,
    rounds: list[int],
    optimizer_mode: str,
    objective: str,
    beam_width: int,
    max_states: int,
    max_batches_per_state: int,
    batch_frontier_policy: str | None,
    validate_batches: bool,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
    overwrite: bool = False,
) -> dict:
    return run_round_sensitivity_impl(
        input_path,
        out_dir,
        passes_path,
        rounds=rounds,
        optimizer_mode=optimizer_mode,
        objective=objective,
        beam_width=beam_width,
        max_states=max_states,
        max_batches_per_state=max_batches_per_state,
        batch_frontier_policy=batch_frontier_policy,
        validate_batches=validate_batches,
        jobs=jobs,
        timeout=timeout,
        max_pairs=max_pairs,
        overwrite=overwrite,
    )


def run_reduction_study(
    inputs: list[str],
    out_dir: Path,
    passes_path: Path,
    *,
    optimizer_mode: str,
    objective: str,
    max_rounds: int,
    max_states: int,
    validate_batches: bool,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
    summarize_components: bool = False,
    overwrite: bool = False,
    continue_on_error: bool = False,
) -> dict:
    return run_reduction_study_impl(
        inputs,
        out_dir,
        passes_path,
        optimizer_mode=optimizer_mode,
        objective=objective,
        max_rounds=max_rounds,
        max_states=max_states,
        validate_batches=validate_batches,
        jobs=jobs,
        timeout=timeout,
        max_pairs=max_pairs,
        summarize_components=summarize_components,
        overwrite=overwrite,
        continue_on_error=continue_on_error,
    )


def run_budgeted_sensitivity(
    inputs: list[str],
    out_dir: Path,
    passes_path: Path,
    *,
    objective: str,
    max_rounds: int,
    beam_widths: list[int],
    max_states_list: list[int],
    max_batches_per_state: int,
    batch_frontier_policy: str | None,
    validate_batches: bool,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
    exact_reference: Path | None = None,
    summarize_components: bool = False,
    overwrite: bool = False,
    continue_on_error: bool = False,
) -> dict:
    return run_budgeted_sensitivity_impl(
        inputs,
        out_dir,
        passes_path,
        objective=objective,
        max_rounds=max_rounds,
        beam_widths=beam_widths,
        max_states_list=max_states_list,
        max_batches_per_state=max_batches_per_state,
        batch_frontier_policy=batch_frontier_policy,
        validate_batches=validate_batches,
        jobs=jobs,
        timeout=timeout,
        max_pairs=max_pairs,
        exact_reference=exact_reference,
        summarize_components=summarize_components,
        overwrite=overwrite,
        continue_on_error=continue_on_error,
    )


def run_passset_smoke(
    inputs: list[str],
    passsets: list[Path],
    out_dir: Path,
    *,
    optimizer_mode: str,
    objective: str,
    max_rounds: int,
    beam_width: int,
    max_states: int,
    max_batches_per_state: int,
    batch_frontier_policy: str | None,
    validate_batches: bool,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
    overwrite: bool = False,
    continue_on_error: bool = False,
) -> dict:
    return run_passset_smoke_impl(
        inputs,
        passsets,
        out_dir,
        optimizer_mode=optimizer_mode,
        objective=objective,
        max_rounds=max_rounds,
        beam_width=beam_width,
        max_states=max_states,
        max_batches_per_state=max_batches_per_state,
        batch_frontier_policy=batch_frontier_policy,
        validate_batches=validate_batches,
        jobs=jobs,
        timeout=timeout,
        max_pairs=max_pairs,
        overwrite=overwrite,
        continue_on_error=continue_on_error,
    )


def run_v2_extension_study(
    inputs: list[str],
    out_dir: Path,
    v1_passes: Path,
    v2_passes: Path,
    *,
    objective: str,
    max_rounds: int,
    beam_width: int,
    max_states: int,
    max_batches_per_state: int,
    batch_frontier_policy: str | None,
    validate_batches: bool,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
    random_trials: int,
    seed: int,
    overwrite: bool = False,
    continue_on_error: bool = False,
) -> dict:
    return run_v2_extension_study_impl(
        inputs,
        out_dir,
        v1_passes,
        v2_passes,
        objective=objective,
        max_rounds=max_rounds,
        beam_width=beam_width,
        max_states=max_states,
        max_batches_per_state=max_batches_per_state,
        batch_frontier_policy=batch_frontier_policy,
        validate_batches=validate_batches,
        jobs=jobs,
        timeout=timeout,
        max_pairs=max_pairs,
        random_trials=random_trials,
        seed=seed,
        overwrite=overwrite,
        continue_on_error=continue_on_error,
    )


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


def run_summarize_exact_reduction_study(
    run_dirs: list[Path],
    out_dir: Path,
    *,
    label: str,
    root_dir: Path | None = None,
    summarize_components: bool = False,
) -> dict:
    return summarize_exact_reduction_study_impl(
        run_dirs,
        out_dir,
        label=label,
        root_dir=root_dir,
        summarize_components=summarize_components,
    )


def run_core_v1_case_study(
    exact_method_summary: Path,
    exact_reduction_summary: Path,
    budgeted_sensitivity_summary: Path,
    out_dir: Path,
    *,
    label: str,
    nbody_round_study: Path | None = None,
    puzzle_case_study: Path | None = None,
    extra_notes: Path | None = None,
) -> dict:
    return summarize_core_v1_case_study_impl(
        exact_method_summary,
        exact_reduction_summary,
        budgeted_sensitivity_summary,
        out_dir,
        label=label,
        nbody_round_study=nbody_round_study,
        puzzle_case_study=puzzle_case_study,
        extra_notes=extra_notes,
    )


def run_core_v1_budgeted_study(
    inputs: list[str],
    out_dir: Path,
    passes_path: Path,
    *,
    objective: str,
    max_rounds: int,
    beam_width: int,
    max_states: int,
    max_batches_per_state: int,
    batch_frontier_policy: str | None,
    validate_batches: bool,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
    baseline_methods: list[str] | None,
    random_trials: int,
    seed: int,
    overwrite: bool = False,
    continue_on_error: bool = False,
) -> dict:
    return run_core_v1_budgeted_study_impl(
        inputs,
        out_dir,
        passes_path,
        objective=objective,
        max_rounds=max_rounds,
        beam_width=beam_width,
        max_states=max_states,
        max_batches_per_state=max_batches_per_state,
        batch_frontier_policy=batch_frontier_policy,
        validate_batches=validate_batches,
        jobs=jobs,
        timeout=timeout,
        max_pairs=max_pairs,
        baseline_methods=baseline_methods,
        random_trials=random_trials,
        seed=seed,
        overwrite=overwrite,
        continue_on_error=continue_on_error,
    )


def run_select_and_run_exact_reference(
    budgeted_study_dir: Path,
    out_dir: Path,
    passes_path: Path,
    *,
    objective: str,
    max_rounds: int,
    max_states: int,
    validate_batches: bool,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
    num_easy: int,
    num_medium: int,
    num_hard: int,
    overwrite: bool = False,
    continue_on_error: bool = False,
) -> dict:
    return select_and_run_exact_reference_impl(
        budgeted_study_dir,
        out_dir,
        passes_path,
        objective=objective,
        max_rounds=max_rounds,
        max_states=max_states,
        validate_batches=validate_batches,
        jobs=jobs,
        timeout=timeout,
        max_pairs=max_pairs,
        num_easy=num_easy,
        num_medium=num_medium,
        num_hard=num_hard,
        overwrite=overwrite,
        continue_on_error=continue_on_error,
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


def run_v3_loop_smoke(
    inputs: list[str],
    out_dir: Path,
    passes_path: Path,
    *,
    optimizer_mode: str,
    objective: str,
    max_rounds: int,
    beam_width: int,
    max_states: int,
    max_batches_per_state: int,
    batch_frontier_policy: str | None,
    validate_batches: bool,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
    overwrite: bool = False,
    continue_on_error: bool = False,
) -> dict:
    return run_v3_loop_smoke_impl(
        inputs,
        out_dir,
        passes_path,
        optimizer_mode=optimizer_mode,
        objective=objective,
        max_rounds=max_rounds,
        beam_width=beam_width,
        max_states=max_states,
        max_batches_per_state=max_batches_per_state,
        batch_frontier_policy=batch_frontier_policy,
        validate_batches=validate_batches,
        jobs=jobs,
        timeout=timeout,
        max_pairs=max_pairs,
        overwrite=overwrite,
        continue_on_error=continue_on_error,
    )


def run_summarize_passsets(inputs: list[Path], out_dir: Path) -> dict:
    return summarize_passsets_impl(inputs, out_dir)


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
) -> dict:
    state_dir = Path(state_dir)
    result = build_batch_family(
        state_dir,
        max_component_size=max_component_size,
        max_batch_candidates=max_batch_candidates,
    )
    if validate_batches:
        tools = _tool_paths(collect_toolchain())
        validation = validate_batch_candidates(state_dir, tools, timeout=10, jobs=1)
        result.update(validation)
    correctness_rows = classify_batch_correctness(state_dir, allow_sampled_batches=allow_sampled_batches)
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

    replay_verified = "true" if result.get("hashes_match") == "true" else "false"
    update_replay_status_artifacts(run_dir, result, replay_verified)
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
    validate_batches: bool,
    allow_sampled_batches: bool,
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
) -> dict:
    from .optimizer import optimize_batches as optimize_batches_impl

    return optimize_batches_impl(
        input_path,
        out_dir,
        passes_path,
        mode=mode,
        objective=objective,
        max_rounds=max_rounds,
        beam_width=beam_width,
        max_batches_per_state=max_batches_per_state,
        max_states=max_states,
        batch_frontier_policy=batch_frontier_policy,
        batch_selection_policy=batch_selection_policy,
        frontier_selection_policy=frontier_selection_policy,
        selection_seed=selection_seed,
        exact_fail_on_incomplete=exact_fail_on_incomplete,
        validate_batches=validate_batches,
        allow_sampled_batches=allow_sampled_batches,
        run_baselines=run_baselines,
        verify_final_pipeline=verify_final_pipeline,
        jobs=jobs,
        timeout=timeout,
        max_pairs=max_pairs,
    )


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
) -> dict:
    return explore_batches(
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
    )


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


def analyze_state(
    input_ll: Path,
    out_dir: Path,
    tools: dict,
    *,
    valid_passes: list[str],
    invalid_rows: list[dict],
    configured_pass_count: int,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
    program: str,
    state_id: str,
    depth: int,
    parent_state_id: str,
    transition_pass: str,
    pass_registry: PassRegistry | None = None,
) -> dict:
    start = time.perf_counter()
    input_ll = Path(input_ll)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_hash = canonical_hash(input_ll)
    pass_registry = pass_registry or tools.get("_pass_registry")

    metadata = _read_metadata(out_dir)
    metadata.update(
        {
            "input": str(input_ll),
            "state_hash": state_hash,
            "state_id": state_id,
            "depth": depth,
            "parent_state_id": parent_state_id,
            "transition_pass": transition_pass,
        }
    )
    write_metadata(out_dir, metadata)

    profile_start = time.perf_counter()
    profile_rows = profile_passes(
        input_ll,
        valid_passes,
        tools,
        out_dir,
        jobs,
        timeout,
        program=program,
        state_id=state_id,
        depth=depth,
        parent_state_id=parent_state_id,
        transition_pass=transition_pass,
        pass_registry=pass_registry if isinstance(pass_registry, PassRegistry) else None,
    )
    profile_time_ms = (time.perf_counter() - profile_start) * 1000
    active_profiles = [row for row in profile_rows if row.get("success") == "true" and row.get("active") == "true"]

    pair_start = time.perf_counter()
    pair_rows = run_pair_tests(
        input_ll,
        active_profiles,
        tools,
        out_dir,
        jobs,
        timeout,
        max_pairs,
        pass_registry=pass_registry if isinstance(pass_registry, PassRegistry) else None,
    )
    profile_map = {row["pass"]: row for row in profile_rows}
    pair_rows = annotate_pair_relations(pair_rows, profile_map)
    write_pair_relations(out_dir / "pair_relation.csv", pair_rows)
    pair_time_ms = (time.perf_counter() - pair_start) * 1000

    cluster_rows = cluster_distribution_rows(pair_rows, program, state_hash)
    write_cluster_distribution(out_dir / "cluster_distribution.csv", cluster_rows)

    total_time_ms = (time.perf_counter() - start) * 1000
    write_per_state_summary(
        out_dir,
        program,
        state_hash,
        state_id=state_id,
        depth=depth,
        parent_state_id=parent_state_id,
        transition_pass=transition_pass,
        pass_set_size=configured_pass_count,
        valid_passes=len(valid_passes),
        invalid_passes=len(invalid_rows),
        profile_time_ms=profile_time_ms,
        pair_time_ms=pair_time_ms,
        total_time_ms=total_time_ms,
    )
    summary = write_summary(out_dir)

    metadata.update(
        {
            "valid_passes": len(valid_passes),
            "invalid_passes": len(invalid_rows),
            "active_passes": len(active_profiles),
            "pair_rows": len(pair_rows),
            "summary": str(summary),
            "total_time_ms": total_time_ms,
        }
    )
    write_metadata(out_dir, metadata)
    return {
        "program": program,
        "out_dir": str(out_dir),
        "state_id": state_id,
        "depth": depth,
        "parent_state_id": parent_state_id,
        "transition_pass": transition_pass,
        "valid_passes": len(valid_passes),
        "active_passes": len(active_profiles),
        "pair_rows": len(pair_rows),
        "summary_path": str(summary),
        "total_time_ms": total_time_ms,
    }


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
