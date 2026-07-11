# Phasebatch Rolling Exact Mainline Design

## Status

- Date: 2026-07-11
- Scope: optimizer search mainline and advisor-study defaults
- Decision: use complete two-layer rolling windows with a five-state boundary frontier as the maintained mainline; retain H=3 as a depth ablation
- Implementation: complete and verified in the 2026-07-11 workspace state
- Compatibility: retain legacy `exact`, `budgeted`, and `auto` modes as explicit comparison modes

## Motivation

The current advisor study uses `budgeted`, `max_rounds=2`, `beam_width=4`, and
`max_batches_per_state=10`. That configuration is useful as a pilot, but it does
not implement the search procedure discussed with the advisor:

1. cover all safe batch choices at the current state;
2. look ahead two layers without sacrificing breadth inside the window;
3. retain five complementary open terminal states at the window boundary;
4. continue from all retained states;
5. repeat until the reachable state graph closes.

The existing `exact` mode is exact only inside a fixed global depth `r`. The new
mode treats two layers as a receding lookahead window rather than as the end
of the optimization. "Local" names the finite window scope; expansion inside
those two layers is complete.

## User-Visible Mode

Add optimizer mode:

```text
rolling-exact
```

Its main controls are:

```text
--rolling-window-depth 2
--rolling-frontier-width 5
--max-rolling-windows 0
```

`0` means no window-count cutoff: stop only at closure or at another explicit
safety limit. A positive window cutoff is allowed for controlled experiments,
but reaching it produces `rolling_exact_incomplete`, never a complete claim.

`--max-rounds` keeps its existing meaning for legacy `exact` and `budgeted`.
`beam_width` and `max_batches_per_state` do not control `rolling-exact`.
`rolling_frontier_width` applies only after both layers are complete. H=3 remains
available through the same option for the recorded depth ablation.

## Window Semantics

For root set `R` (up to five canonical states) and horizon `H`:

1. Start a local breadth-first expansion at `(R, local_depth=0)`.
2. At every expanded state, run full state analysis, full pairwise construction,
   validation, and the existing correctness classifier.
3. Reject the entire exact claim if pair coverage, component enumeration,
   candidate materialization, validation, or state expansion is incomplete.
4. Apply every batch satisfying the existing exact executable predicate:
   `certified_batch`, `can_hard_fold=true`, and
   `validation_status=all_permutations_same`.
5. Canonical-hash merge equal child states, while preserving every incoming DAG
   edge and one deterministic local route to each canonical state.
6. Continue every distinct reachable canonical state until local depth `H`;
   there is no intermediate beam or batch cap.
7. Treat depth-`H` states and earlier closed states as window terminals. Closed
   states remain final candidates but do not consume continuation slots.
8. If at most five open terminals remain, retain all of them. Otherwise retain
   one state from each applicable objective, direct-call, memory, branch, and
   novelty bucket, then deterministically fill any remaining slots by score.
9. Make all retained terminals the next window roots.

The root is not an eligible terminal when it has executable outgoing edges. This
allows a non-improving first step to expose a better second step.

## Closure Semantics

The run closes only when every retained frontier path reaches one of these conditions:

- `no_active_passes`: profiling reports zero active passes;
- `no_executable_batches`: complete construction and validation find no exact
  executable batch;
- `state_graph_closed`: every terminal returns to a canonical state whose
  outgoing transitions were already completely expanded.

A newly discovered state may be retained even when its objective is temporarily
worse. This is necessary for pass-enabling sequences. A fully expanded canonical
state is not expanded again.

## Completeness Boundary

`rolling-exact` uses the same conservative evidence boundary as legacy exact and
adds window-level checks. The run is incomplete if any expanded state has:

- lazy or `max_pairs` pair omissions;
- unresolved conflict components;
- truncated batch candidates;
- dropped active passes;
- missing, failed, incomplete, sampled, bounded, or unknown batch validation;
- validation DAG node/edge budget exhaustion;
- failed batch application;
- unique-state cap exhaustion;
- a positive rolling-window cap reached before closure.

A completed `mismatch` is negative evidence rather than missing evidence. It
certifies that the candidate must be rejected, so the candidate is skipped
without making the rolling scope incomplete.

The status values are:

```text
rolling_exact_complete
rolling_exact_incomplete
rolling_exact_incomplete_continued
```

The continued form is available only when the existing
`exact_fail_on_incomplete=false` escape hatch is explicitly used. It remains an
incomplete result.

The scope is recorded separately from correctness evidence status:

```text
exact_scope = rolling_global_exact_to_closure
# only when no open boundary state was pruned

exact_scope = rolling_window_exact_frontier_limited
# when a checkpoint had more than K open states

global_search_complete = true|false
```

This does not claim global optimality over arbitrary LLVM pass sequences. It is
exact only for the configured pass set, batch construction rules, validation
rules, objective, and completed rolling windows.

## State and Path Accounting

State IDs remain deterministic and monotonically assigned in BFS/candidate CSV
order. Canonical state records and best deterministic root paths are global
across windows. The five retained states therefore carry valid independent
paths from `S0000`; canonical convergence never invents a path.

Add `rolling_windows.csv` with one row per attempted window:

- window index and all roots;
- open/closed terminal counts and new-state count;
- all retained states, objectives, and selection buckets;
- boundary-pruned state count;
- status and closure reason.

Optimizer events record window start, exact expansion, commit, closure, and
incomplete evidence. `frontier_scores.csv` also records rolling checkpoints with
`rolling_checkpoint_selected` or `rolling_checkpoint_pruned`; it never labels
them as budgeted `beam_pruned` decisions.

## Mainline Defaults

`optimize-batches` and `run-advisor-report-zh` default to `rolling-exact`.

Advisor-study defaults become:

- formal sample target: 50 programs;
- rolling window depth: 2;
- rolling frontier width: 5;
- unlimited windows until closure;
- full pair testing with no default pair cap;
- pairwise batch construction;
- automatic exhaustive/DAG validation ladder;
- no sampled or bounded execution;
- strict in-process worker backend;
- state cap retained only as an explicit safety guard and reported as incomplete
  if reached.

The existing 20-program budgeted report remains a pilot artifact. It must not be
relabelled as rolling exact or exact-complete evidence.

## Compatibility

- `exact`: unchanged fixed-depth exhaustive state-DAG expansion (`exact-rN`).
- `budgeted`: unchanged candidate cap and beam pruning.
- `auto`: unchanged conservative choice between legacy exact and budgeted.
- correctness classification and batch validation modes are unchanged.
- no CEGAR path is reintroduced.

## Acceptance Tests

1. A two-layer window expands every certified batch even when beam and batch caps
   are set to one.
2. Five selected depth-two terminals become the roots of a second window.
3. A temporarily worse first layer can be selected because a better second layer
   terminal exists.
4. Zero active passes and zero executable batches both close successfully.
5. A canonical cycle closes deterministically without an infinite loop.
6. Candidate truncation, incomplete pairs, failed apply, state cap, and window cap
   all produce incomplete status.
7. Final chosen path consists only of real certified transitions from `S0000`.
8. Existing exact and budgeted tests remain unchanged and green.
9. CLI and Advisor defaults expose rolling exact, depth two, frontier width
   five, and no within-window beam claim.
