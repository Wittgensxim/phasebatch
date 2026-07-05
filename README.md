# Phase Ordering MVP Data System

This repository contains a data-producing MVP for LLVM phase-ordering research.
It provides a small Python CLI that produces:

- LLVM toolchain metadata capture;
- active/dormant pass profiling;
- dynamic AB/BA pass-pair testing;
- conflict graph statistics;
- CSV and Markdown reports.

The MVP uses coarse pass-level effects. It is designed to support progress
meetings with concrete structural data, not to claim global phase-ordering
optimality.

## Quick Start

```bash
python -m phasebatch --help
python -m phasebatch analyze \
  --input benchmarks/tiny/branch.c \
  --out outputs/branch \
  --passes configs/core_passes.yaml \
  --jobs 8 \
  --timeout 10 \
  --max-pairs 300

python -m phasebatch batch \
  --inputs benchmarks/tiny/*.c \
  --out outputs/mvp_run \
  --passes configs/core_passes.yaml \
  --jobs 8 \
  --timeout 10 \
  --max-pairs 300

python -m phasebatch explore \
  --input benchmarks/tiny/branch.c \
  --out outputs/explore_branch \
  --passes configs/core_passes.yaml \
  --max-depth 1 \
  --frontier-policy all-active \
  --jobs 8 \
  --timeout 10 \
  --max-pairs 300

python -m phasebatch explore-batches \
  --input benchmarks/tiny/branch.c \
  --out outputs/batch_explore_branch \
  --passes configs/core_passes.yaml \
  --max-depth 1 \
  --max-component-size 10 \
  --max-batch-candidates 50 \
  --max-batches-per-state 20 \
  --max-frontier-states 20 \
  --batch-frontier-policy all \
  --validate-batches \
  --allow-sampled-batches \
  --jobs 8 \
  --timeout 10 \
  --max-pairs 300

python -m phasebatch run-mainline \
  --inputs benchmarks/tiny/*.c \
  --out outputs/mainline_run \
  --passes configs/core_passes.yaml \
  --max-depth 1 \
  --max-component-size 10 \
  --max-batch-candidates 50 \
  --max-batches-per-state 20 \
  --validate-batches \
  --jobs 8 \
  --timeout 10 \
  --max-pairs 300

python -m phasebatch summarize-mainline \
  --run-dir outputs/mainline_run

python -m phasebatch batchify \
  --state-dir outputs/explore_branch/states/S0000 \
  --max-component-size 10 \
  --max-batch-candidates 200

python -m phasebatch batchify \
  --state-dir outputs/explore_branch/states/S0000 \
  --max-component-size 10 \
  --max-batch-candidates 200 \
  --validate-batches \
  --allow-sampled-batches

python -m phasebatch eval-batches \
  --run-dir outputs/batch_explore_branch \
  --objective ir-inst-count

python -m phasebatch optimize-batches \
  --input benchmarks/tiny/branch.c \
  --out outputs/optimized_branch \
  --passes configs/core_passes.yaml \
  --mode budgeted \
  --objective ir-inst-count \
  --max-rounds 1 \
  --max-batches-per-state 20 \
  --batch-selection-policy score \
  --frontier-selection-policy score \
  --validate-batches \
  --jobs 8 \
  --timeout 10 \
  --max-pairs 300

python -m phasebatch optimize-batches \
  --input benchmarks/tiny/branch.c \
  --out outputs/optimized_branch_exact \
  --passes configs/core_passes.yaml \
  --mode exact \
  --objective ir-inst-count \
  --max-rounds 3 \
  --max-states 5000 \
  --validate-batches \
  --jobs 8 \
  --timeout 10 \
  --max-pairs 300
```

On this machine, use the DLM Conda environment:

```bash
D:/Miniconda/envs/dlm/python.exe -m phasebatch --help
```

`scripts/run_smoke.sh` prefers `D:/Miniconda/envs/dlm/python.exe` when present,
then a `dlm` command if one is on PATH, then `python`.

## Research Workflows

Core-v1 remains the stable baseline pass set. Use it for the main exact and
budgeted case studies:

```bash
python -m phasebatch run-core-v1-budgeted-study \
  --inputs benchmarks/tiny/*.c \
  --out outputs/core_v1_budgeted \
  --passes configs/core_passes.yaml \
  --max-rounds 4 \
  --beam-width 4 \
  --max-states 500 \
  --max-batches-per-state 20 \
  --batch-frontier-policy score \
  --validate-batches \
  --jobs 8 \
  --timeout 10 \
  --max-pairs 300

python -m phasebatch select-and-run-exact-reference \
  --budgeted-study-dir outputs/core_v1_budgeted \
  --out outputs/core_v1_exact_reference \
  --passes configs/core_passes.yaml \
  --max-rounds 4 \
  --max-states 5000 \
  --validate-batches \
  --jobs 8 \
  --timeout 10 \
  --max-pairs 300
```

Use the reduction/evidence/reporting commands on existing optimize-batches runs;
they do not rerun optimization:

```bash
python -m phasebatch summarize-reduction \
  --run-dir outputs/optimized_branch_exact

python -m phasebatch export-evidence-pack \
  --run-dir outputs/optimized_branch_exact

python -m phasebatch diagnose-paths \
  --run-dir outputs/optimized_branch_exact

python -m phasebatch visualize-dag \
  --run-dir outputs/optimized_branch_exact \
  --out outputs/optimized_branch_exact/dag_viz \
  --view all \
  --formats dot svg png \
  --max-full-nodes 200 \
  --include-selected-path \
  --include-depth-overview
```

`visualize-dag` follows the CGO 2006 phase-ordering DAG idea: nodes are
canonical LLVM IR states, edges are executed or observed batch transitions, and
duplicate transitions show multiple paths reaching the same canonical state.
Graphviz is optional. If `dot` is unavailable, DOT/CSV/Markdown files are still
written and the summary records a warning.

Use pass-set extension workflows to test scalability without replacing Core-v1:

```bash
python -m phasebatch run-v2-extension-study \
  --inputs benchmarks/tiny/*.c \
  --out outputs/v2_extension_tiny \
  --v1-passes configs/core_passes.yaml \
  --v2-passes configs/scalar_passes_v2.yaml \
  --objective ir-inst-count \
  --max-rounds 4 \
  --beam-width 4 \
  --max-states 500 \
  --max-batches-per-state 20 \
  --batch-frontier-policy score \
  --validate-batches \
  --jobs 8 \
  --timeout 10 \
  --max-pairs 600 \
  --overwrite \
  --continue-on-error

python -m phasebatch run-v3-loop-smoke \
  --inputs benchmarks/tiny/loop.c E:/llvm-test-suite/SingleSource/Benchmarks/BenchmarkGame/n-body.c \
  --out outputs/v3_loop_smoke \
  --passes configs/middleend_passes_v3.yaml \
  --optimizer-mode budgeted \
  --objective ir-inst-count \
  --max-rounds 3 \
  --beam-width 4 \
  --max-states 800 \
  --max-batches-per-state 12 \
  --batch-frontier-policy score \
  --validate-batches \
  --jobs 8 \
  --timeout 10 \
  --max-pairs 1000 \
  --overwrite \
  --continue-on-error
```

## Outputs

Each `analyze` output directory contains:

- `metadata.json`
- `valid_passes.csv`
- `invalid_passes.csv`
- `pass_profile.csv`
- `pair_relation.csv`
- `cluster_distribution.csv`
- `per_state_summary.csv`
- `summary.md`
- `artifacts/`

The `batch` command also writes aggregate CSVs and `aggregate_summary.md`.
The `explore` command writes `states.csv`, `state_transitions.csv`, and one
analysis directory per state under `states/`.
The `explore-batches` command analyzes the root state, repeatedly batchifies
frontier states up to `--max-depth`, applies eligible batch candidates, caches
duplicate child hashes, and writes `batch_state_transitions.csv`, `skipped_batches.csv`,
`enable_suppress.csv`, `relation_flip.csv`, `states.csv`,
`aggregate_by_depth.csv`, `aggregate_batch_summary.csv`,
`aggregate_coverage_summary.csv`, `aggregate_overlap_summary.csv`,
`multistate_summary.md`, and `batch_explore_summary.md`. When
`--validate-batches` is enabled, only
`all_permutations_same` candidates are applied by default. Add
`--allow-sampled-batches` to also apply `sampled_same` candidates. `mismatch`
and `failed` candidates are never applied. Use `--max-batches-per-state` to cap
how many selected batch candidates are applied from each state, use
`--max-frontier-states` to cap non-duplicate states kept after each depth, and
choose `--batch-frontier-policy all`, `largest-batch`, `certified-first`, or
`diverse-hash` to order batch/frontier selection.
The `run-mainline` command runs `explore-batches` once per input file or glob
match, using one output subdirectory per program. It writes `mainline_runs.csv`,
`mainline_missing_outputs.csv`, `mainline_aggregate_states.csv`,
`mainline_aggregate_batches.csv`, `mainline_aggregate_coverage.csv`, and
`mainline_aggregate_overlap.csv` at the run root, then writes
`mainline_summary.md` for advisor/reporting use. Existing per-program output
directories are protected by default; pass `--overwrite` to rerun them and
`--continue-on-error` to keep running later inputs after a failure.
The `summarize-mainline` command regenerates `mainline_summary.md` from those
existing CSVs without rerunning compilation or exploration.
The `eval-batches` command consumes an existing `explore-batches` run, computes
the requested objective for every row in `batch_state_transitions.csv`, and
writes `objective_signal.csv` and `objective_summary.md` (`objective_eval.csv`
is also kept as a legacy compatibility output). The initial supported objective
is `ir-inst-count`, which compares approximate parent and child state `input.ll`
instruction counts. Add `--recursive` when pointing it at a `run-mainline` root;
this evaluates every program subdirectory and writes
`aggregate_objective_signal.csv` at the run root. Objective signals are for
ranking/evaluation only, not commutation or independence proof. `run-mainline`
can run this post-pass automatically with `--eval-objective ir-inst-count`.
The `optimize-batches` command analyzes the root state, builds and optionally
validates batch candidates, executes eligible batches, analyzes resulting child
states with the full valid pass list, and picks the best reached state by
`ir-inst-count`. `--mode budgeted` separates state-local batch candidate
selection from beam frontier state selection. By default both policies use
`score`: batch selection mixes score, component diversity, and deterministic
hash diversity; frontier selection uses Pareto-aware score, novelty, and
objective buckets. `--batch-frontier-policy` is kept only as a compatibility
alias for setting both policies at once. `--mode exact` expands all certified
executable batch alternatives across a
frontier loop up to `--max-rounds` and `--max-states`; it never executes sampled,
failed, rejected, unknown, or unvalidated batches. Exact mode is complete only
inside the explored certified batch-state graph and current bounds, not globally
optimal for LLVM. If truncation, unresolved components, dropped active passes,
missing validation, or state cap overflow occur, `exact_status.txt` and
`optimize_summary.md` report `exact_incomplete` or
`exact_incomplete_continued`. The command writes `states.csv`, `state_dag.csv`,
`batch_state_transitions.csv`, `leaf_states.csv`, `chosen_path.csv`,
`optimized_batches.txt`, `optimized_pipeline.txt`, `final.ll`,
`exact_status.txt`, `optimize_summary.md`, `final_summary.md`, and
`final_summary_index.csv`. With `--verify-final-pipeline` enabled, it also
writes `pipeline_replay.csv` and `replayed_final.ll`, proving that
`optimized_pipeline.txt` replayed from `states/S0000/input.ll` reaches the same
canonical IR hash as `final.ll`. Regenerate the replay check with
`python -m phasebatch replay-final-pipeline --run-dir outputs/your_optimize_run`.
Regenerate the final human-readable report with
`python -m phasebatch summarize-final --run-dir outputs/your_optimize_run`.
The objective is used only for path selection, not as commutation or independence proof.
The `batchify` command consumes an existing state directory, reads only
`pass_profile.csv` and `pair_relation.csv`, and writes `batch_components.csv`,
`batch_candidates.csv`, `batch_summary.csv`, `batch_correctness.csv`,
`footprint_overlap.csv`, and `coverage_report.csv`, `coverage_summary.csv`,
and `batch_summary.md`. It does not run opt unless `--validate-batches` is set.
With validation enabled it runs opt over candidate permutations, writes
`batch_validation.csv`, and updates `batch_summary.md` with validation and
correctness counts. `all_permutations_same` becomes a `certified_batch` that
can be hard-folded and executed. `sampled_same` becomes a `sampled_batch`;
it is not a hard certificate and is executable only when
`--allow-sampled-batches` is set.
The coverage report is an accounting invariant: every active pass in
`pass_profile.csv` must be classified as certified, heuristic, unresolved,
rejected, unvalidated/unknown, or dropped. Normal successful runs should have
`dropped_active_passes = 0`.
`footprint_overlap.csv` and `aggregate_overlap_summary.csv` are diagnostics only.
They summarize coarse changed-function/block overlap and relation joins, but are
not used as hard independence proof in this MVP.

`summarize-reduction` writes `reduction_by_state.csv`,
`reduction_summary.csv`, and `reduction_summary.md`, showing how local ordering
choices shrink from `n!` to certified/executable batch alternatives for reached
states. `export-evidence-pack` writes selected and executed batch certificate
CSVs plus `evidence_pack.md`. `diagnose-paths` compares the selected batch path
against greedy/random/config-order baselines and writes prefix replay curves.
These reports are explanatory artifacts only; they do not create new
correctness evidence.

`visualize-dag` writes `state_dag_full.dot`, `state_dag_selected.dot`,
`depth_overview.dot`, `dag_metrics.csv`, `dag_depth_metrics.csv`,
`dag_paths.csv`, `missing_inputs.csv`, and `dag_summary.md`. SVG/PNG files are
also written when Graphviz is installed and the full graph is below
`--max-full-nodes` or the selected/depth views are requested.

## Versioned Pass Sets

- `configs/core_passes.yaml` / `configs/core_passes_v1.yaml`: stable Core-v1
  scalar, CFG, memory, and cleanup pass set for main case studies.
- `configs/scalar_passes_v2.yaml`: Core-v1 plus `sccp`, `dse`, `memcpyopt`,
  `sink`, and `tailcallelim`; use this as a scalability extension study.
- `configs/middleend_passes_v3.yaml`: v2 plus loop-middle-end candidates. Run
  `audit-passes` first because loop passes may require nested New Pass Manager
  syntax such as `function(loop(...))`.

See `docs/pass_sets.md` for the pass-set format and current boundaries.

## Repository Hygiene

Generated experiment outputs live under `outputs/` and are ignored by Git except
for `outputs/.gitkeep`. Keep large or machine-local artifacts out of commits.
For public GitHub publishing, this repository intentionally excludes local
agent/editor metadata and the reference PDF copy used during development.
