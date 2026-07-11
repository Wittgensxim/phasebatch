# Phasebatch Project Status

## Current Scope

Phasebatch is a data-producing LLVM phase-ordering research prototype. The
maintained system reasons over a state-local LLVM IR DAG:

- node: a canonical LLVM IR state;
- edge: a correctness-allowed batch transition;
- merge: multiple paths reaching the same canonical IR hash.

The project compresses pass-order choices. It does not claim global LLVM
phase-ordering optimality or use runtime/objective values as correctness proof.

## Maintained Commands

Execution and search:

- `analyze`
- `batch`
- `explore`
- `explore-batches`
- `batchify`
- `optimize-batches`
- `optimize-staged`
- `audit-passes`

Evaluation and evidence over current runs:

- `eval-batches`
- `compare-baselines`
- `summarize-final`
- `summarize-reduction`
- `summarize-components`
- `export-evidence-pack`
- `diagnose-paths`
- `visualize-dag`
- `replay-final-pipeline`

Worker acceptance:

- `verify-opt-worker`
- `benchmark-opt-worker`

Advisor reporting:

- `run-advisor-report-zh`
- `summarize-advisor-report-zh`

The maintained end-to-end path is full pairwise plus reuse/cache, certified
batch validation, rolling exact search, and the strict in-process worker.
Legacy fixed-depth exact, budgeted search, bounded/sampled validation, lazy pair
testing, and staged runtime reranking remain explicit comparison modes.

## Maintained Search Mainline

`rolling-exact` is the default optimizer mode. It fully expands every exact
executable batch for two local layers, then retains up to five deterministic,
diverse open terminals and repeats from all retained states until closure.

- No beam or per-state executable-batch cap is used inside a rolling window.
- Boundary selection uses objective, direct-call, memory, branch, and novelty
  buckets; closed states do not consume continuation slots.
- A temporarily non-improving path may survive long enough to expose a useful
  later step.
- Normal closure is `no_active_passes`, `no_executable_batches`, or
  `state_graph_closed`.
- Pair omissions, candidate truncation, unresolved components, validation DAG
  limits, apply failures, state caps, or a positive window cap produce
  `rolling_exact_incomplete`.
- `exact_status` reports evidence/window completeness. `global_search_complete`
  is false whenever more than five open boundary states force pruning.
- The scope is `rolling_global_exact_to_closure` when no boundary was pruned,
  otherwise `rolling_window_exact_frontier_limited`; neither claims optimality
  over arbitrary LLVM pass sequences.

Fixed-depth `exact-rN` remains available with `--mode exact`. `budgeted` retains
its candidate and beam pruning for performance/scalability comparisons.

The matched five-program H=2/K=5 versus H=3/K=5 pilot completed on 2026-07-11.
H=3 fully preserved one additional local layer, but total wall-clock increased
from 91.265 s to 180.526 s (1.978x). All five final IR objectives and static
feature vectors matched; four of five pipelines were identical. H=2 is therefore
the maintained default and H=3 is retained as the explicit depth ablation. Evidence:
`outputs/rolling_depth_comparison_5programs_fixed_20260711/depth_comparison_summary.md`.

## Correctness Boundary

Only a `certified_batch` with an `all_permutations_same` hard certificate may
be hard-folded. Sampled, bounded, rejected, failed, unknown, or unvalidated
results are not hard-folding evidence.

- Pair relations are state-local.
- Unknown/failed pairs remain separate evidence categories but are operational
  conflict edges with `can_hard_fold=false`.
- Coarse footprint/overlap is diagnostic only.
- DAG visualization displays evidence; it does not create evidence.
- Runtime and instruction-count objectives rank already safe states only.
- Exactness is bounded by the configured pass set, reached state graph,
  component/candidate limits, validation limits, and state cap.
- A staged run is exact only when every stage is exact and complete.

## Output Families

Optimizer root outputs:

- `states.csv`
- `state_dag.csv`
- `batch_state_transitions.csv`
- `leaf_states.csv`
- `chosen_path.csv`
- `chosen_path_summary.csv`
- `optimized_pipeline.txt`
- `pipeline_replay.csv`
- `optimize_summary.md`
- `final_summary.md`
- `exact_status.txt`
- `rolling_windows.csv`

Per-state evidence:

- `pass_profile.csv`
- `pair_relation.csv`
- `pair_cost_summary.csv`
- `batch_components.csv`
- `batch_candidates.csv`
- `batch_validation.csv`
- `batch_correctness.csv`
- `coverage_report.csv`
- `coverage_summary.csv`
- `footprint_overlap.csv`

Staged runs add:

- `staged_summary.csv`
- `staged_pipeline.csv`
- `staged_replay.csv`
- `staged_summary.md`
- optional `runtime_candidates.csv`, `runtime_trials.csv`,
  `runtime_summary.csv`, and `runtime_selection.md`.

## Strict LLVM Worker

The long-lived C++ LLVM worker is the strict default for commands that execute
LLVM. It provides isolated LLVM contexts, in-memory pass pipelines,
reference-counted module handles, bounded path caches, deferred materialization,
and in-process `LLVMDiff` comparison.

If a borrowed pair-result handle is invalidated by a concurrent worker restart,
the pair tester reruns the same AB/BA pipelines once with direct materialization.
This keeps pair classification deterministic across worker counts without an
external-opt fallback; retry counts are explicit in `pair_relation.csv`.

`--opt-backend external` remains available only for intentional comparison.
Pass-side `LLVM ERROR:` exits become conservative `llvm_fatal` pipeline
failures and restart the affected worker. Timeout, protocol, and infrastructure
errors raise in strict mode. No strict-worker failure silently falls back.

The matched Salsa20 Core-v1 exact run measured:

| Metric | External opt | Worker | Change |
|---|---:|---:|---:|
| wall-clock | 87.798 s | 25.053 s | 3.504x faster |
| optimizer time | 84.593 s | 22.713 s | 3.724x faster |
| pair testing | 7.252 s | 3.084 s | 2.352x faster |
| batch validation | 74.300 s | 17.614 s | 4.218x faster |

Both runs selected the same state and pipeline, matched all pair and batch
classifications, and passed replay.

On the local 8-core/16-thread Ryzen 7 9800X3D, matched `crc8.be H=3,K=5`
runs took 46.795 s with 8 workers, 48.758 s with 12, and 48.762 s with 16.
All reached the same 66 states and 78 transitions. The maintained local default
therefore remains 8 workers. Evidence:
`outputs/worker_scaling_crc8_h3k5_fixed_20260711/worker_scaling_summary.md`.

## Staged Runtime Result

The retained Salsa20 E5 study uses required IPO, scalar v2, loop/cleanup v4,
and an isolated vector/cleanup v5 stage. Runtime reranking correctly rejected
the slower vectorized candidates. The selected E5 result was 28.23% slower than
LLVM `default<O2>` on the matched cyclic benchmark, so the vector sequence is
not part of the default pipeline.

Evidence:
`outputs/salsa20_staged_v5_fixed_20260710/e5_experiment_report.md`.

## Advisor Report

The retained Advisor Data Report v1 study contains 20 deterministic
SingleSource C programs under strict worker mode. It is pilot evidence produced
with budgeted depth 2 and beam width 4. It remains useful for structure, cost,
and visualization, but is not exact-to-closure evidence.

The formal report command now defaults to 50 programs, `rolling-exact`, a
complete two-layer window, a five-state boundary frontier, unlimited windows
until closure, full pair testing without a default pair cap, and a 2000-state
safety guard. Reaching a safety guard is incomplete; boundary pruning is
reported separately as `global_search_complete=false`. Formal data must be
written to a new study directory rather than relabeling the 20-program pilot.

Evidence:
`outputs/advisor_report_zh_20programs/advisor_report_zh.md`.

## Repository Hygiene

The source tree keeps only the maintained code, configs, tests, docs, tiny
benchmarks, worker source/build entrypoint, and selected local evidence runs.
Generated runs remain ignored by Git.

Normal execution removes `.ll` intermediates and empty directories.
`--keep-ir-artifacts` or `--dump-validation-dag` intentionally preserves debug
IR. Local caches, copied papers, archives, bytecode and old experiment trees do
not belong in the maintained workspace.

## Verification

```powershell
D:/Miniconda/envs/dlm/python.exe -m pytest -q
```

Graphviz is optional. Without `dot`, DAG commands still emit deterministic DOT,
CSV and Markdown and record the missing renderer.
