# Phasebatch Validation Performance Design

## Status

- Date: 2026-07-10
- Scope: validation execution and budgeted validation scheduling
- Approved direction: staged, exact-preserving optimization
- Current baseline: Salsa20 full pairwise, exact, one round, 81.426 seconds wall-clock

## Goal

Reduce Phasebatch wall-clock time by eliminating repeated validation pass work and using the configured worker budget more effectively, while preserving:

- full pairwise correctness semantics;
- exact completeness rules;
- batch certificate classification;
- deterministic candidate, node, edge, CSV, and DOT output;
- the rule that no unvalidated batch may execute.

## Baseline Evidence

The fresh Salsa20 baseline at `outputs/runtime_baseline_salsa20_pairwise_20260710` recorded:

| Metric | Value |
|---|---:|
| Observed wall-clock | 81.426 s |
| Internal optimizer time | 79.327 s |
| Batch validation time | 71.604 s |
| Total opt invocations | 2496 |
| Validation work units | 2037 |
| States / transitions | 6 / 5 |
| Certified batches | 22 |
| Final objective | 354 |

The root size-8 validation DAG alone took 32.457 seconds. Validation is therefore the first optimization target.

## Feasibility Assessment

### 1. Reuse Profiling Outputs During Validation

**Feasible: yes. Risk: low. Phase: 1.**

Each state already materializes and verifies `A(S)` for every active pass. Validation can reuse that output when a permutation starts with A.

For exhaustive, bounded, and sampled validation:

- keep the canonical order as one unsplit full-pipeline anchor;
- for every non-canonical order, use the existing first-pass output when available;
- run only the remaining pipeline;
- for an empty remaining pipeline, copy the verified output without invoking opt.

For validation DAG:

- seed root transitions from the same single-pass outputs;
- compute the DAG-specific safe canonical hash before inserting the seed;
- count the seeded transition as profile reuse, not as an opt invocation.

If the profile output is absent, failed, stale, or not from the current state directory, validation falls back to the existing full execution.

### 2. Global Opt Worker Pool

**Feasible: yes, but a repository-wide pool is not required in Phase 1. Risk: medium.**

Profiling, pair testing, and validation currently run as separate phases, so they do not compete concurrently. Phase 1 introduces one state-local validation execution context with a shared opt-slot limit equal to `jobs`.

This context prevents nested candidate/order/DAG executors from running more than `jobs` external opt processes at once.

A repository-wide runner service would touch profiling, pair testing, optimizer application, baselines, and replay. That larger refactor is deferred until measurements show that phase-local control is insufficient.

### 3. Parallel Batch Candidate Validation

**Feasible: yes. Risk: medium. Phase: 1.**

Candidates have unique IDs and output directories, so their expensive validation work is independent.

The implementation will:

- preassign candidate indexes from the existing deterministic CSV order;
- validate candidates concurrently;
- store each result in its original index;
- write final CSV rows in original candidate order;
- divide the worker budget between candidate-level and order/DAG-level work;
- enforce the shared opt-slot limit as the final oversubscription guard.

Candidate completion order must never affect report order.

### 4. Parallel Validation DAG Edges at the Same Depth

**Feasible: yes. Risk: medium-high. Phase: 1.**

DAG depth `d+1` depends on the completed nodes at depth `d`, but transitions leaving the same completed depth can run concurrently.

Each depth is processed in two phases:

1. **Parallel transition phase**
   - snapshot and sort source nodes by node ID;
   - enumerate unused passes in canonical order;
   - preassign deterministic edge indexes and output paths;
   - execute/cache transitions and calculate hashes concurrently.

2. **Deterministic merge phase**
   - consume transition results in preassigned edge order;
   - compare against existing equivalence classes;
   - assign deterministic node IDs;
   - update node/edge metrics;
   - stop conservatively on the first deterministic failure or budget violation.

Equivalence-class mutation remains serialized. Only opt execution and hashing are parallel.

### 5. Promote DAG Cache to State Scope

**Feasible: yes. Risk: medium. Phase: 1.**

Current transition and equivalence caches are local to one candidate, although candidates from the same state overlap heavily.

Phase 1 adds a state-local cache shared by all candidate validations:

- transition key: source safe canonical hash, pass name, resolved pipeline;
- equivalence key: ordered pair of IR hashes plus comparator version;
- transition files: state-local content-addressed cache directory;
- synchronization: per-key single-flight so only one thread computes a missing entry;
- lifetime: current state analysis only;
- persistence: none across Python processes.

Failed transitions and failed comparator calls are not treated as reusable success values.

When DAG dumping is enabled, state-cache IR is covered by a keep marker so the debug graph does not reference prematurely deleted IR.

### 6. Budgeted On-Demand Validation

**Feasible: yes. Risk: high because it changes explored candidate scope. Phase: 2.**

This optimization applies only to budgeted mode and is opt-in initially.

Proposed option:

```text
--budgeted-validation-strategy {all,on-demand}
```

Default remains `all`.

`on-demand` behavior:

1. build all deterministic candidates;
2. rank them with a cheap score that does not claim validation evidence;
3. validate in rank order;
4. continue past rejected/failed candidates;
5. stop after finding `max_batches_per_state` executable candidates or exhausting the list;
6. emit explicit `not_validated` rows for candidates not attempted;
7. run the existing correctness classifier;
8. execute only rows with `can_execute=true`.

Exact mode always forces `all` and rejects an explicit on-demand request.

The objective reached by on-demand budgeted search may differ from validate-all budgeted search because fewer candidates are explored. Reports must state this scope explicitly.

## Architecture

### New Module: `phasebatch/validation_runtime.py`

This module owns validation-only execution coordination.

It provides:

- `ValidationRuntime`: state-local lifetime and worker budget;
- `ValidationTransitionKey`: immutable transition identity;
- thread-safe transition cache;
- thread-safe equivalence cache;
- per-key single-flight coordination;
- profile-output seed registration;
- cost counters for opt calls, pass invocations, reuse, and state-cache hits.

It does not decide whether a batch is correct. Existing validator and classifier modules retain that responsibility.

### Updated `phasebatch/batcher.py`

Responsibilities:

- create one `ValidationRuntime` per state;
- load current-state profile outputs;
- schedule candidates deterministically;
- use first-pass reuse for non-DAG validation orders;
- preserve result row ordering;
- write new validation cost fields.

### Updated `phasebatch/batch_validation_dag.py`

Responsibilities:

- accept `jobs` and `ValidationRuntime`;
- seed root transitions;
- execute same-depth transitions concurrently;
- perform deterministic serial equivalence-class merging;
- report local and state-level cache work.

### Updated `phasebatch/optimizer.py`

Phase 1:

- pass the configured validation worker budget unchanged;
- use explicit validation opt-invocation fields in optimizer timing.

Phase 2:

- split candidate construction from on-demand validation;
- apply on-demand only in budgeted mode;
- preserve validate-all behavior in exact and by default.

## Cost Accounting

Every validation row will record:

- `validation_opt_invocations`;
- `validation_pass_invocations_baseline`;
- `validation_pass_invocations_actual`;
- `validation_pass_invocations_saved`;
- `validation_profile_reuse_hits`;
- `validation_state_transition_cache_hits`;
- `validation_state_equivalence_cache_hits`;
- existing DAG local cache hit/miss metrics.

Definitions:

```text
baseline pass invocations
  = pass work without profile reuse or transition reuse

actual pass invocations
  = pass work actually submitted to opt

saved
  = baseline - actual
```

`optimizer_timing.csv` will use `validation_opt_invocations` instead of estimating validation opt calls from `tested_orders`.

## Determinism Requirements

The following must be identical between `jobs=1` and `jobs>1`:

- batch candidate row order;
- validation row order;
- validation status and hard-certificate result;
- DAG node IDs;
- DAG edge IDs;
- representative paths chosen during deterministic merge;
- DOT/CSV ordering;
- selected final state, final hash, and objective.

Wall-clock timings and thread completion order are not expected to match.

## Failure Handling

- A failed reused profile path triggers normal execution fallback.
- A transition failure is attached to its preassigned deterministic edge.
- If multiple parallel transitions fail, the earliest deterministic edge determines the reported failure.
- Cache waiters receive the same success or failure result as the single-flight owner.
- Comparator failure remains non-foldable.
- Node/edge budget excess remains incomplete and non-executable.
- Parallel work already in flight may finish after a deterministic failure is known, but its result is not merged or used.

## Testing Strategy

Tests are written before production changes.

Phase 1 tests cover:

1. exhaustive validation omits the first pass when a valid profile output exists;
2. missing profile output preserves old behavior;
3. DAG root transitions use profile seeds without opt calls;
4. overlapping candidates hit one shared state transition cache;
5. candidate results remain in canonical input order under out-of-order completion;
6. observed concurrent opt count is greater than one and never exceeds `jobs`;
7. DAG `jobs=1` and `jobs=4` produce identical nodes, edges, status, and certificate;
8. a parallel transition failure is reported using deterministic edge order;
9. validation cost fields and optimizer total invocation accounting are exact;
10. existing exhaustive, bounded, sampled, DAG, exact, and budgeted tests remain green.

Phase 2 tests cover:

1. default budgeted strategy still validates all candidates;
2. on-demand stops after enough executable batches;
3. rejected candidates cause the next candidate to be validated;
4. untouched candidates become explicit unvalidated rows;
5. unvalidated candidates never execute;
6. exact rejects on-demand;
7. output is deterministic.

## Benchmark Acceptance

After Phase 1, rerun the exact Salsa20 command matching the 81.426-second baseline.

Correctness gates:

- process exit code 0;
- `exact_complete`;
- pair matrix complete;
- 22 certified batches;
- selected state `S0002`;
- final objective 354;
- replay success with matching hash;
- zero remaining `.ll` files and zero empty directories.

Performance evidence:

- report observed wall-clock;
- report validation time;
- report total opt invocations;
- report baseline/actual/saved validation pass invocations;
- report profile and state-cache hits;
- compare against the existing baseline without claiming speedup from a single noisy metric alone.

## Non-Goals

- No persistent cross-run pair or validation cache in this change.
- No change to pairwise construction semantics.
- No weakening of `can_hard_fold` or `can_execute`.
- No change to pairwise full-matrix semantics.
- No automatic use of sampled or bounded validation.
- No repository-wide multiprocessing rewrite.
- No attempt to make exact globally optimal outside its existing scope.

## Delivery Stages

### Phase 1: Exact-Preserving Validation Runtime

- profiling-output reuse;
- state-local cache;
- bounded candidate concurrency;
- deterministic DAG same-depth concurrency;
- precise cost accounting;
- unit/integration tests;
- fresh exact Salsa20 benchmark.

### Phase 2: Budgeted On-Demand Validation

- opt-in CLI strategy;
- deterministic cheap ranking;
- incremental validation;
- explicit unvalidated evidence;
- budgeted correctness tests;
- fresh budgeted Salsa20 comparison.
