# Phasebatch Project Status

This document summarizes the current public project shape. It is meant to help
new readers understand what is implemented, which outputs matter, and what is
safe to claim from the generated evidence.

## Current Scope

Phasebatch is a data-producing LLVM phase-ordering research prototype. Its main
unit of reasoning is a state-local LLVM IR graph:

- node: a canonical LLVM IR state
- edge: an executed or observed batch transition
- merge: multiple paths reaching the same canonical IR hash

The project focuses on pass interaction analysis, certified batch construction,
state-aware exploration, and evidence reporting. It does not claim global LLVM
phase-ordering optimality.

## Main Commands

Use these commands for the stable Core-v1 workflow:

- `analyze`: profile one input state and test active pass pairs.
- `batchify`: build batch candidates from an existing state directory.
- `optimize-batches`: execute certified batch candidates and select a final
  reached state by objective.
- `run-core-v1-budgeted-study`: run budgeted Core-v1 optimization over multiple
  inputs.
- `select-and-run-exact-reference`: choose and run exact reference cases from a
  budgeted study.
- `run-round-sensitivity`: compare max-round settings for one input.

Use these commands for reporting and explanation over existing outputs:

- `summarize-reduction`: compute local search-space reduction metrics.
- `export-evidence-pack`: collect selected and executed batch certificate rows.
- `diagnose-paths`: compare batch optimizer paths with greedy/random/config
  order baselines.
- `visualize-dag`: generate CGO-2006-style DOT/CSV/Markdown views of the
  compressed state DAG.
- `summarize-components`: summarize conflict graph components and interaction
  edges.
- `summarize-core-v1-case-study`: assemble the final Core-v1 narrative report.

Use these commands for pass-set extension studies:

- `audit-passes`: resolve pass pipelines accepted by the local LLVM build.
- `run-v2-extension-study`: compare Core-v1 with scalar/memory/CFG v2.
- `run-v3-loop-smoke`: test loop-middle-end v3 on loop-heavy programs.
- `summarize-passsets`: combine pass-set smoke outputs into one report.

## Evidence Boundary

Only `certified_batch` candidates with `all_permutations_same` validation are
hard-folding evidence. Sampled, objective-only, rejected, failed, unknown, or
unvalidated data are diagnostic or evaluation signals.

Important boundaries:

- Objective values are used for ranking or path selection only.
- Coarse footprint/overlap labels are diagnostic only.
- Reduction metrics are state-local and apply only to reached states.
- DAG visualization does not create new correctness evidence; it displays
  evidence already recorded by validation and correctness files.
- Exact mode is exact only within the certified batch-state graph and the
  configured bounds.

## Important Output Families

Optimize-batches runs write:

- `states.csv`
- `state_dag.csv`
- `batch_state_transitions.csv`
- `leaf_states.csv`
- `chosen_path.csv`
- `chosen_path_summary.csv`
- `optimized_pipeline.txt`
- `final.ll`
- `pipeline_replay.csv`
- `optimize_summary.md`
- `final_summary.md`

Reduction and evidence reports add:

- `reduction_by_state.csv`
- `reduction_summary.csv`
- `reduction_summary.md`
- `evidence_pack.csv`
- `selected_batch_certificates.csv`
- `executed_batch_certificates.csv`
- `evidence_pack.md`

DAG visualization adds:

- `state_dag_full.dot`
- `state_dag_selected.dot`
- `depth_overview.dot`
- `dag_metrics.csv`
- `dag_depth_metrics.csv`
- `dag_paths.csv`
- `dag_summary.md`

SVG and PNG graph files are produced only when Graphviz `dot` is installed.

## Repository Hygiene

Generated experiment outputs should stay under `outputs/`. The repository keeps
`outputs/.gitkeep` tracked but ignores the generated output tree. Public commits
should include source code, configs, tests, docs, tiny benchmarks, and scripts.

Do not publish:

- generated `outputs/` result trees
- local editor or agent metadata
- third-party PDFs copied into the workspace
- machine-local cache directories

## Suggested Verification

Before publishing or reporting a result, run:

```bash
D:/Miniconda/envs/dlm/python.exe -m pytest -q
```

For a fresh DAG visualization smoke:

```bash
D:/Miniconda/envs/dlm/python.exe -m phasebatch visualize-dag \
  --run-dir outputs/verify_dag_branch_exact \
  --out outputs/verify_dag_branch_exact/dag_viz \
  --view all \
  --formats dot svg png \
  --max-full-nodes 200 \
  --include-selected-path \
  --include-depth-overview
```

If Graphviz is not installed, the command should still write DOT, CSV, and
Markdown files and record the missing `dot` command in `dag_summary.md`.
