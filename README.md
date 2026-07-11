# Phasebatch

Phasebatch is an LLVM phase-ordering research prototype that compresses
state-local ordering choices into correctness-gated pass batches. The maintained
mainline uses a complete pair matrix, validation DAGs, a strict in-process LLVM
worker, rolling local-exact state search, staged optimization, and reproducible
Chinese advisor reports. Fixed-depth exact and budgeted search remain comparison
modes.

## Maintained Mainline

For each reached LLVM IR state, Phasebatch:

1. profiles configured passes and keeps the active ones;
2. runs full AB/BA pair tests with single-pass reuse and state-local caches;
3. classifies pairs as commute, order-sensitive, or unknown;
4. builds a conflict graph where only proven commute pairs lack conflict edges;
5. enumerates maximal independent sets and combines them into batch candidates;
6. validates complete batches by exhaustive permutations or a permutation DAG;
7. executes only candidates allowed by the correctness classifier;
8. merges equal canonical IR states in the optimizer DAG;
9. completely expands a two-layer window, retains up to five diverse open
   terminals, and repeats from all retained states until the state graph closes;
10. replays the selected pipeline and removes normal-run `.ll` intermediates.

Unknown or failed pairs are never marked commute. They remain distinct in the
evidence tables, but operationally they are non-foldable conflict edges.

## Requirements

- Python 3.10 or newer
- LLVM tools including `clang`, `opt`, `llvm-diff`, `llc`, and `llvm-size`
- the Phasebatch worker built against the same LLVM tree
- Graphviz `dot` only when rendered DAG SVG/PNG files are required

The local reference toolchain is under `E:/llvm/build/bin`.

## Build The Worker

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_worker.ps1
```

Commands that execute LLVM default to strict worker mode:

```text
--opt-backend worker
--opt-worker <PHASEBATCH_OPT_WORKER or worker/build/phasebatch-worker.exe>
--opt-workers <same value as --jobs>
```

Missing workers, protocol errors, and timeouts fail closed. A pass-side
`LLVM ERROR:` is recorded as a conservative `llvm_fatal` pipeline failure and
the affected worker is restarted. Strict mode never silently falls back to an
external `opt.exe`. Use `--opt-backend external` only for an intentional legacy
baseline.

## Core Commands

Analyze one state:

```powershell
D:/Miniconda/envs/dlm/python.exe -m phasebatch analyze `
  --input benchmarks/tiny/branch.c `
  --out outputs/branch `
  --passes configs/core_passes_v1.yaml `
  --jobs 8
```

Build candidates from an analyzed state:

```powershell
D:/Miniconda/envs/dlm/python.exe -m phasebatch batchify `
  --state-dir outputs/branch `
  --max-component-size 10 `
  --max-batch-candidates 200 `
  --validate-batches
```

Explore batch transitions without final-state optimization:

```powershell
D:/Miniconda/envs/dlm/python.exe -m phasebatch explore-batches `
  --input benchmarks/tiny/branch.c `
  --out outputs/explore_branch `
  --passes configs/core_passes_v1.yaml `
  --max-depth 2 `
  --validate-batches
```

Run the maintained optimizer:

```powershell
D:/Miniconda/envs/dlm/python.exe -m phasebatch optimize-batches `
  --input benchmarks/tiny/branch.c `
  --out outputs/optimized_branch `
  --passes configs/core_passes_v1.yaml `
  --mode rolling-exact `
  --rolling-window-depth 2 `
  --rolling-frontier-width 5 `
  --max-rolling-windows 0 `
  --max-states 2000 `
  --batch-construction-mode pairwise `
  --pair-testing-mode full `
  --batch-validation-mode auto `
  --validate-batches `
  --jobs 8
```

`rolling-exact` uses no pruning or per-state executable-batch cap inside a
window. At the second-layer checkpoint it keeps up to five states using the
objective/call/memory/branch/novelty buckets. Correctness evidence completeness
and global search completeness are reported separately: frontier pruning sets
`global_search_complete=false`, while any safety or evidence failure produces
an incomplete exact status. `exact` retains fixed-depth exact-rN semantics, and
`budgeted` retains beam/batch caps.
No mode may execute a failed, rejected, unknown, or unvalidated batch. The
objective is used only for path selection, never as correctness evidence.

Useful result-side commands are:

```text
summarize-final
summarize-reduction
summarize-components
export-evidence-pack
diagnose-paths
visualize-dag
eval-batches
compare-baselines
replay-final-pipeline
```

These commands read or evaluate existing runs. `summarize-*`, evidence export,
and DAG visualization do not run `opt` and do not create new correctness proof.

## Staged Optimization

`optimize-staged` runs ordered small pass pools through the same pairwise and
batch-validation engine. It supports safe terminal-state runtime reranking and
always replays the aggregate pipeline.

```powershell
D:/Miniconda/envs/dlm/python.exe -m phasebatch optimize-staged `
  --input E:/llvm-test-suite/SingleSource/Benchmarks/Misc/salsa20.c `
  --manifest configs/staged_salsa20_v5.yaml `
  --out outputs/salsa20_staged_v5 `
  --jobs 8
```

Runtime measurements rank only already reached, replayable terminal states.
They do not certify pairs, batches, or paths.

## Worker Verification

Run semantic differential checks before accepting a rebuilt worker:

```powershell
D:/Miniconda/envs/dlm/python.exe -m phasebatch verify-opt-worker `
  --inputs benchmarks/tiny/branch.c benchmarks/tiny/loop.c `
  --passes configs/core_passes_v1.yaml `
  --out outputs/verify_opt_worker `
  --opt-worker worker/build/phasebatch-worker.exe
```

Measure worker throughput separately:

```powershell
D:/Miniconda/envs/dlm/python.exe -m phasebatch benchmark-opt-worker `
  --input E:/llvm-test-suite/SingleSource/Benchmarks/Misc/salsa20.c `
  --out outputs/worker_benchmark `
  --opt-worker worker/build/phasebatch-worker.exe `
  --iterations 100
```

The matched Salsa20 Core-v1 exact run measured 87.798 s with external `opt` and
25.053 s with eight workers, a 3.504x wall-clock speedup. Pair relations, batch
classifications, selected state, pipeline, and replay result matched.

## 中文导师数据报告

Run the formal rolling-exact report workflow (50 programs by default):

```powershell
D:/Miniconda/envs/dlm/python.exe -m phasebatch run-advisor-report-zh `
  --test-suite-root E:/llvm-test-suite `
  --out outputs/advisor_report_zh_50programs_rolling_exact `
  --passes configs/core_passes_v1.yaml `
  --mode rolling-exact `
  --rolling-window-depth 2 `
  --rolling-frontier-width 5 `
  --max-rolling-windows 0 `
  --jobs 8 `
  --timeout 15 `
  --resume `
  --continue-on-error
```

The retained `outputs/advisor_report_zh_20programs` study is a budgeted
depth-2/beam-4 pilot. It is not relabelled as rolling-exact evidence.

Rebuild all aggregate CSV, figures, DAGs, and Markdown without running LLVM:

```powershell
D:/Miniconda/envs/dlm/python.exe -m phasebatch summarize-advisor-report-zh `
  --study-dir outputs/advisor_report_zh_20programs
```

The report keeps overlap diagnostics separate from pair/batch correctness,
preserves missing values, distinguishes wall-clock from cumulative work, and
emits nine PNG/SVG figure families. Graphviz absence leaves deterministic DOT
files and a recorded warning.

## Important Outputs

An `optimize-batches` run writes the state graph and selected path:

```text
states.csv
state_dag.csv
batch_state_transitions.csv
chosen_path.csv
chosen_path_summary.csv
optimized_pipeline.txt
pipeline_replay.csv
optimize_summary.md
final_summary.md
exact_status.txt
rolling_windows.csv
```

Each analyzed state records:

```text
pass_profile.csv
pair_relation.csv
pair_cost_summary.csv
batch_components.csv
batch_candidates.csv
batch_validation.csv
batch_correctness.csv
coverage_report.csv
coverage_summary.csv
footprint_overlap.csv
```

`footprint_overlap.csv` and aggregate overlap summaries are diagnostics only.
`batch_validation.csv`, `batch_correctness.csv`, and replay artifacts carry the
correctness boundary. `objective_signal.csv`, `objective_eval.csv`, and
`objective_summary.md` are ranking/evaluation outputs.

## Pass Configurations

- `configs/core_passes_v1.yaml`: maintained 14-pass Core-v1 set.
- `configs/scalar_passes_v2.yaml`: scalar/memory/CFG pool used by staged runs.
- `configs/ipo_inline_v4.yaml`: staged IPO pool.
- `configs/loop_passes_v4.yaml`: staged loop pool.
- `configs/cleanup_passes_v4.yaml`: staged cleanup pool.
- `configs/vector_cleanup_passes_v5.yaml`: isolated vector/cleanup pool.
- `configs/staged_salsa20_v5.yaml`: reproducible Salsa20 staged manifest.
- `configs/staged_salsa20_v5_smoke.yaml`: bounded staged smoke manifest.

## Repository Hygiene

Generated experiments belong under `outputs/`; only `outputs/.gitkeep` is
tracked. Normal runs remove intermediate `.ll` files and empty directories.
`--keep-ir-artifacts` and `--dump-validation-dag` retain debugging IR
intentionally.

Do not publish local caches, agent/editor metadata, copied papers, worker build
intermediates, or generated result trees.

Run the regression suite with:

```powershell
D:/Miniconda/envs/dlm/python.exe -m pytest -q
```
