# Current Pass Sets

Phasebatch pass configuration files may use either a simple list or structured
entries with `name`, `pipeline`, `category`, and `stage`. The maintained configs
use structured entries whenever a pass needs a nested New Pass Manager pipeline.

## Core-v1

`configs/core_passes_v1.yaml` is the stable 14-pass set used by pairwise
regression tests and the Advisor Report. It is the default choice for a single
`optimize-batches` experiment.

## Staged Pools

The staged optimizer intentionally uses several smaller pools instead of one
large global pass set:

- `configs/ipo_inline_v4.yaml`: IPO/inlining preparation.
- `configs/scalar_passes_v2.yaml`: scalar, memory and CFG transformations.
- `configs/loop_passes_v4.yaml`: loop canonicalization and optimization.
- `configs/cleanup_passes_v4.yaml`: post-loop scalar cleanup.
- `configs/vector_cleanup_passes_v5.yaml`: isolated vector and cleanup choices.

`configs/staged_salsa20_v5.yaml` defines the reproducible full Salsa20 staged
study. `configs/staged_salsa20_v5_smoke.yaml` provides a smaller verification
manifest.

## Boundaries

- Pass-set membership does not imply activity on a particular IR state.
- A valid pipeline is not automatically safe to combine with another pass.
- Pair and batch correctness remain state-local.
- Expanding a pool increases profiling and pair-matrix cost quadratically and
  may increase permutation validation work.
- Runtime reranking applies only to terminal states already reached through the
  correctness-gated optimizer.
