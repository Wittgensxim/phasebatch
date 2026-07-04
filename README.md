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

python -m phasebatch batchify \
  --state-dir outputs/explore_branch/states/S0000 \
  --max-component-size 10 \
  --max-batch-candidates 200

python -m phasebatch batchify \
  --state-dir outputs/explore_branch/states/S0000 \
  --max-component-size 10 \
  --max-batch-candidates 200 \
  --validate-batches
```

On this machine, use the DLM Conda environment:

```bash
D:/Miniconda/envs/dlm/python.exe -m phasebatch --help
```

`scripts/run_smoke.sh` prefers `D:/Miniconda/envs/dlm/python.exe` when present,
then a `dlm` command if one is on PATH, then `python`.

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
`aggregate_by_depth.csv`, `multistate_summary.md`, and
`batch_explore_summary.md`. When `--validate-batches` is enabled, only
`all_permutations_same` candidates are applied by default. Add
`--allow-sampled-batches` to also apply `sampled_same` candidates. `mismatch`
and `failed` candidates are never applied. Use `--max-batches-per-state` to cap
how many selected batch candidates are applied from each state, use
`--max-frontier-states` to cap non-duplicate states kept after each depth, and
choose `--batch-frontier-policy all`, `largest-batch`, `certified-first`, or
`diverse-hash` to order batch/frontier selection.
The `batchify` command consumes an existing state directory, reads only
`pass_profile.csv` and `pair_relation.csv`, and writes `batch_components.csv`,
`batch_candidates.csv`, `batch_summary.csv`, and `batch_summary.md`. It does not run opt
unless `--validate-batches` is set. With validation enabled it runs opt over
candidate permutations, writes `batch_validation.csv`, and updates
`batch_summary.md` with validation counts. `all_permutations_same` is strong
evidence for that candidate; `sampled_same` is empirical evidence only.
