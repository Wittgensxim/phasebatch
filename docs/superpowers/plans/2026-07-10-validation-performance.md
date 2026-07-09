# Validation Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce Phasebatch validation wall-clock and opt/pass work while preserving exact certificates, deterministic artifacts, and the rule that only validated executable batches run.

**Architecture:** Add one state-local `ValidationRuntime` that owns the validation opt-slot budget, profile-output seeds, thread-safe transition/equivalence caches, single-flight coordination, and cost counters. `batcher.py` uses it for first-pass reuse and deterministic candidate scheduling; `batch_validation_dag.py` uses it for state-cache reuse and same-depth parallel transition execution. Budgeted on-demand validation is added later as an opt-in strategy and never applies to exact mode.

**Tech Stack:** Python 3, `concurrent.futures`, `threading`, LLVM `opt`, CSV schemas, `unittest`/pytest, existing Phasebatch IR equivalence and optimizer modules.

---

## File Map

- Create `phasebatch/validation_runtime.py`: validation worker slots, state-local caches, single-flight, profile seeds, cost counters.
- Create `tests/test_validation_runtime.py`: direct runtime concurrency/cache tests.
- Modify `phasebatch/batcher.py`: runtime construction, first-pass reuse, candidate scheduling, selected candidate validation.
- Modify `phasebatch/batch_validation_dag.py`: runtime-backed transition execution and deterministic same-depth parallelism.
- Modify `phasebatch/schema.py`: validation cost fields and budgeted strategy metadata fields.
- Modify `phasebatch/optimizer.py`: precise validation invocation accounting and on-demand budgeted orchestration.
- Modify `phasebatch/cli.py`: opt-in budgeted validation strategy.
- Modify `phasebatch/batch_validation_ladder.py`: aggregate new validation cost metrics.
- Modify `phasebatch/final_summary.py`: expose validation strategy and cost evidence where configuration is summarized.
- Modify `docs/phasebatch_project_logic_zh.md`: document new runtime, metrics, and on-demand boundary.
- Modify focused tests in `tests/test_batcher.py`, `tests/test_batch_validation_dag.py`, `tests/test_optimizer.py`, `tests/test_cli_pipeline.py`, and `tests/test_schema.py`.

### Task 1: State-Local Validation Runtime

**Files:**
- Create: `phasebatch/validation_runtime.py`
- Create: `tests/test_validation_runtime.py`

- [ ] **Step 1: Write failing opt-slot concurrency test**

Add a real threaded test that starts four operations against a runtime configured with two slots:

```python
def test_opt_slots_never_exceed_worker_budget():
    runtime = ValidationRuntime(state_dir, max_workers=2)
    active = 0
    peak = 0
    lock = threading.Lock()

    def operation():
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return "ok"

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: runtime.run_with_opt_slot(operation), range(4)))

    assert results == ["ok"] * 4
    assert peak == 2
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
python -m pytest tests/test_validation_runtime.py::ValidationRuntimeTests::test_opt_slots_never_exceed_worker_budget -q
```

Expected: import failure because `phasebatch.validation_runtime` does not exist.

- [ ] **Step 3: Write failing transition single-flight test**

Use two threads requesting the same key. The compute callback increments a count, writes one IR file, and returns path/hash. Assert callback count is one and both callers receive the same entry.

- [ ] **Step 4: Write failing equivalence-cache and failure tests**

Assert:

- equal hash-pair requests share one comparator call;
- transition compute exceptions wake all waiters;
- failed transition results are not stored as successful cache entries;
- cache paths are content-addressed under `state_dir/artifacts/validation_cache`.

- [ ] **Step 5: Implement minimal runtime**

Implement immutable keys and a state-local runtime:

```python
@dataclass(frozen=True)
class ValidationTransitionKey:
    source_hash: str
    pass_name: str
    pipeline_key: str


@dataclass(frozen=True)
class ValidationTransition:
    ir_path: Path
    canonical_hash: str
    source: str


class ValidationRuntime:
    def __init__(self, state_dir: Path, max_workers: int):
        self.state_dir = Path(state_dir)
        self.max_workers = max(1, max_workers)
        self._opt_slots = BoundedSemaphore(self.max_workers)
        self._lock = RLock()
        self._transitions = {}
        self._equivalences = {}
        self._inflight = {}

    def run_with_opt_slot(self, operation):
        with self._opt_slots:
            return operation()
```

Add `get_or_compute_transition`, `get_or_compute_equivalence`, `seed_transition`, and thread-safe counters.

- [ ] **Step 6: Run focused tests GREEN**

Run:

```powershell
python -m pytest tests/test_validation_runtime.py -q
```

Expected: all runtime tests pass.

- [ ] **Step 7: Commit Task 1**

```powershell
git add phasebatch/validation_runtime.py tests/test_validation_runtime.py
git commit -m "feat: add state-local validation runtime"
```

### Task 2: Validation Cost Fields and First-Pass Reuse

**Files:**
- Modify: `phasebatch/schema.py`
- Modify: `phasebatch/batcher.py`
- Modify: `tests/test_batcher.py`
- Modify: `tests/test_schema.py`

- [ ] **Step 1: Write failing exhaustive reuse test**

Create `pass_profile.csv` with verified outputs for A and B. Validate candidate `A;B` exhaustively. Keep canonical `A;B` as the full anchor and assert the non-canonical order invokes only B's remaining counterpart:

```python
self.assertEqual(seen_calls[0]["passes"], ["A", "B"])
self.assertEqual(seen_calls[1]["src"], profile_b_path)
self.assertEqual(seen_calls[1]["passes"], ["A"])
```

Assert row metrics:

```python
self.assertEqual(row["validation_opt_invocations"], "2")
self.assertEqual(row["validation_pass_invocations_baseline"], "4")
self.assertEqual(row["validation_pass_invocations_actual"], "3")
self.assertEqual(row["validation_pass_invocations_saved"], "1")
self.assertEqual(row["validation_profile_reuse_hits"], "1")
```

- [ ] **Step 2: Run reuse test RED**

Run:

```powershell
python -m pytest tests/test_batcher.py::BatcherTests::test_validation_reuses_profile_output_for_noncanonical_orders -q
```

Expected: missing validation cost fields and full `B,A` pipeline still executed.

- [ ] **Step 3: Write fallback and singleton tests**

Assert:

- missing profile output executes the original full order;
- a singleton candidate keeps its canonical full-pipeline anchor and reports zero reuse;
- a profile row from another state hash is ignored;
- pipeline registry names resolve correctly after the first pass is removed.

- [ ] **Step 4: Extend schema**

Add these fields immediately before `time_ms` in `BATCH_VALIDATION_FIELDS`:

```python
"validation_opt_invocations",
"validation_pass_invocations_baseline",
"validation_pass_invocations_actual",
"validation_pass_invocations_saved",
"validation_profile_reuse_hits",
"validation_state_transition_cache_hits",
"validation_state_equivalence_cache_hits",
```

Update schema tests with the exact order.

- [ ] **Step 5: Implement profile-output loading and order execution**

Add:

```python
def _load_profile_outputs(state_dir: Path, state_hash: str) -> dict[str, Path]:
    ...

def _run_validation_pipeline(
    opt: str,
    input_ll: Path,
    order: list[str],
    output: Path,
    timeout: int,
    pass_registry: PassRegistry | None,
    profile_outputs: dict[str, Path],
    allow_first_pass_reuse: bool,
    runtime: ValidationRuntime,
) -> dict:
    ...
```

The canonical order passes `allow_first_pass_reuse=False`; non-canonical orders pass true. Each result reports opt and pass invocation costs.

- [ ] **Step 6: Aggregate order costs into validation rows**

For exhaustive, bounded, and sampled rows, set all seven new fields. Existing correctness fields must remain unchanged.

- [ ] **Step 7: Run focused tests GREEN**

Run:

```powershell
python -m pytest tests/test_batcher.py tests/test_schema.py -q
```

- [ ] **Step 8: Commit Task 2**

```powershell
git add phasebatch/schema.py phasebatch/batcher.py tests/test_batcher.py tests/test_schema.py
git commit -m "feat: reuse profiling outputs in batch validation"
```

### Task 3: State-Level DAG Cache and Profile Seeds

**Files:**
- Modify: `phasebatch/validation_runtime.py`
- Modify: `phasebatch/batcher.py`
- Modify: `phasebatch/batch_validation_dag.py`
- Modify: `tests/test_batch_validation_dag.py`
- Modify: `tests/test_validation_runtime.py`

- [ ] **Step 1: Write failing DAG profile-seed test**

Create A/B/C profile outputs and validate one DAG candidate. Assert no opt call is made for root-to-A/B/C transitions and:

```python
self.assertEqual(row["validation_profile_reuse_hits"], "3")
self.assertGreaterEqual(int(row["validation_state_transition_cache_hits"]), 3)
```

- [ ] **Step 2: Run seed test RED**

Run the exact new test and confirm root transitions still call `run_opt`.

- [ ] **Step 3: Write failing cross-candidate cache test**

Create two overlapping candidates, `A;B;C` and `A;B;D`, validate sequentially with one shared runtime, and assert the second candidate reuses at least one state transition and performs fewer fake opt calls than isolated caches.

- [ ] **Step 4: Implement runtime seeding**

When creating the state runtime:

- compute the root safe canonical hash;
- validate each profile row belongs to the current state;
- resolve the single-pass pipeline key;
- compute the profile output safe hash;
- seed a `ValidationTransitionKey`;
- copy the seed into the content-addressed cache only when ownership is needed for dump retention.

- [ ] **Step 5: Replace local DAG caches with runtime-backed lookup**

Keep candidate-local hit counters for existing metrics, but consult shared runtime before running opt or llvm-diff. Increment the state-level fields separately.

- [ ] **Step 6: Preserve dump artifacts**

When `dump_validation_dag` is true, write a keep marker covering `artifacts/validation_cache` and ensure dumped nodes reference existing paths.

- [ ] **Step 7: Run DAG/cache tests GREEN**

Run:

```powershell
python -m pytest tests/test_validation_runtime.py tests/test_batch_validation_dag.py tests/test_batcher.py -q
```

- [ ] **Step 8: Commit Task 3**

```powershell
git add phasebatch/validation_runtime.py phasebatch/batcher.py phasebatch/batch_validation_dag.py tests/test_validation_runtime.py tests/test_batch_validation_dag.py
git commit -m "feat: share validation DAG cache within states"
```

### Task 4: Bounded Candidate and DAG Concurrency

**Files:**
- Modify: `phasebatch/batcher.py`
- Modify: `phasebatch/batch_validation_dag.py`
- Modify: `tests/test_batcher.py`
- Modify: `tests/test_batch_validation_dag.py`

- [ ] **Step 1: Write failing candidate concurrency/order test**

Use candidates with controlled sleeps so B0002 completes first and B0000 last. Assert:

- peak active fake opt operations is greater than one;
- peak never exceeds `jobs`;
- written `batch_validation.csv` order remains B0000, B0001, B0002.

- [ ] **Step 2: Run candidate concurrency test RED**

Expected: peak equals one because candidates are currently sequential.

- [ ] **Step 3: Write failing DAG jobs determinism test**

Run the same commuting DAG with `jobs=1` and `jobs=4` in separate state directories. Normalize timing/path-only fields and assert identical:

- validation status/tier;
- node IDs and subset masks;
- edge IDs/source/target/pass;
- final classes;
- cache metrics defined by deterministic edge order.

- [ ] **Step 4: Write failing deterministic parallel failure test**

Make a later edge fail faster than an earlier edge. Assert the reported failure corresponds to the earliest preassigned failing edge, not completion order.

- [ ] **Step 5: Implement bounded candidate scheduling**

Use deterministic indexed futures:

```python
candidate_workers = min(max(1, jobs), len(candidates))
inner_workers = max(1, jobs // candidate_workers)
rows: list[dict | None] = [None] * len(candidates)
```

All leaf opt calls still pass through `ValidationRuntime.run_with_opt_slot`.

- [ ] **Step 6: Implement two-phase DAG depth execution**

Add a transition-spec dataclass containing deterministic edge number, source node, pass, subset, cache key, and output path. Execute specs concurrently; merge sorted results serially.

- [ ] **Step 7: Run focused concurrency tests GREEN**

Run:

```powershell
python -m pytest tests/test_batcher.py tests/test_batch_validation_dag.py tests/test_validation_runtime.py -q
```

- [ ] **Step 8: Repeat determinism test**

Run the jobs determinism test at least five times:

```powershell
1..5 | ForEach-Object { python -m pytest tests/test_batch_validation_dag.py -q }
```

- [ ] **Step 9: Commit Task 4**

```powershell
git add phasebatch/batcher.py phasebatch/batch_validation_dag.py tests/test_batcher.py tests/test_batch_validation_dag.py
git commit -m "feat: parallelize deterministic batch validation"
```

### Task 5: Precise Timing and Validation Summaries

**Files:**
- Modify: `phasebatch/optimizer.py`
- Modify: `phasebatch/batch_validation_ladder.py`
- Modify: `phasebatch/final_summary.py`
- Modify: `tests/test_optimizer.py`
- Modify: `tests/test_batch_validation_ladder.py`
- Modify: `tests/test_final_summary.py`

- [ ] **Step 1: Write failing optimizer invocation-accounting test**

Create validation rows where `tested_orders=10` but `validation_opt_invocations=6`. Assert optimizer timing adds six, not ten.

- [ ] **Step 2: Run accounting test RED**

Expected: current timing counts ten.

- [ ] **Step 3: Extend ladder aggregation tests**

Assert run-level summaries include total:

- validation opt invocations;
- baseline/actual/saved pass invocations;
- profile reuse hits;
- state transition/equivalence hits.

- [ ] **Step 4: Implement precise accounting**

In `_write_optimizer_timing`, prefer:

```python
validation_opt_invocations += _int(
    row.get("validation_opt_invocations")
    or row.get("tested_orders")
)
```

Aggregate the new fields in validation ladder CSV/Markdown and expose a concise validation-cost section in final summary.

- [ ] **Step 5: Run focused report tests GREEN**

Run:

```powershell
python -m pytest tests/test_optimizer.py tests/test_batch_validation_ladder.py tests/test_final_summary.py -q
```

- [ ] **Step 6: Commit Task 5**

```powershell
git add phasebatch/optimizer.py phasebatch/batch_validation_ladder.py phasebatch/final_summary.py tests/test_optimizer.py tests/test_batch_validation_ladder.py tests/test_final_summary.py
git commit -m "feat: report precise validation work savings"
```

### Task 6: Opt-In Budgeted On-Demand Validation

**Files:**
- Modify: `phasebatch/cli.py`
- Modify: `phasebatch/optimizer.py`
- Modify: `phasebatch/batcher.py`
- Modify: `phasebatch/schema.py`
- Modify: `tests/test_cli_pipeline.py`
- Modify: `tests/test_optimizer.py`
- Modify: `tests/test_batcher.py`

- [ ] **Step 1: Write failing CLI default/forwarding tests**

Assert:

```text
--budgeted-validation-strategy {all,on-demand}
```

defaults to `all` and forwards `on-demand` to `optimize_batches`.

- [ ] **Step 2: Write failing exact rejection test**

Assert exact mode with on-demand raises:

```text
Exact mode requires budgeted_validation_strategy=all.
```

- [ ] **Step 3: Write failing on-demand stopping test**

Create four candidates where:

- first rejects;
- second and third certify;
- fourth would certify;
- `max_batches_per_state=2`.

Assert only first three receive real validation, fourth gets an explicit `not_validated` row, and only second/third execute.

- [ ] **Step 4: Write failing default compatibility test**

With strategy `all`, assert all four candidates are validated exactly as before.

- [ ] **Step 5: Implement selected candidate validation API**

Refactor `validate_batch_candidates` to accept an optional ordered candidate subset while always emitting one row per original candidate. Unattempted rows use the existing unvalidated base shape:

```python
validation_status = "not_validated"
validation_tier = "unvalidated"
validation_complete = "false"
validation_hard_certificate = "false"
validation_incomplete_reason = "budgeted_on_demand_not_selected"
```

- [ ] **Step 6: Implement deterministic cheap pre-validation ranking**

Rank without claiming evidence, using:

1. larger active-pass coverage;
2. larger batch size;
3. fewer unresolved components;
4. canonical candidate order as final tie-break.

Validate in that order until enough executable rows are found. Re-run the existing correctness classifier after each attempted candidate or deterministic chunk.

- [ ] **Step 7: Record scope**

Add strategy to metadata and summaries. Exact always forces `all`; CEGAR remains unchanged unless explicitly supported by a future design.

- [ ] **Step 8: Run focused on-demand tests GREEN**

Run:

```powershell
python -m pytest tests/test_cli_pipeline.py tests/test_optimizer.py tests/test_batcher.py -q
```

- [ ] **Step 9: Commit Task 6**

```powershell
git add phasebatch/cli.py phasebatch/optimizer.py phasebatch/batcher.py phasebatch/schema.py tests/test_cli_pipeline.py tests/test_optimizer.py tests/test_batcher.py
git commit -m "feat: add budgeted on-demand batch validation"
```

### Task 7: Documentation, Full Regression, and Benchmarks

**Files:**
- Modify: `docs/phasebatch_project_logic_zh.md`
- Create: `outputs/runtime_optimized_salsa20_pairwise_20260710/`
- Create: `outputs/runtime_optimized_budgeted_salsa20_20260710/`

- [ ] **Step 1: Update project logic documentation**

Document:

- validation profile reuse;
- state-level cache lifetime;
- worker-slot semantics;
- deterministic parallel DAG phases;
- new validation cost fields;
- budgeted `all` versus `on-demand`;
- exact rejection of on-demand.

- [ ] **Step 2: Run focused validation suite**

```powershell
python -m pytest tests/test_validation_runtime.py tests/test_batcher.py tests/test_batch_validation_dag.py tests/test_batch_validation_ladder.py tests/test_optimizer.py tests/test_cli_pipeline.py tests/test_schema.py tests/test_final_summary.py -q
```

Expected: zero failures.

- [ ] **Step 3: Run full suite**

```powershell
python -m pytest -q
```

Expected: zero failures.

- [ ] **Step 4: Run fresh optimized exact Salsa20 baseline**

Use the exact command from `runtime_baseline_report.md`, changing only the output directory to:

```text
outputs/runtime_optimized_salsa20_pairwise_20260710
```

Capture observed wall-clock separately.

- [ ] **Step 5: Audit exact correctness**

Assert:

- `exact_complete`;
- `pair_matrix_complete=true`;
- 22 certified batches;
- selected state S0002;
- final objective 354;
- replay success;
- zero `.ll` and empty directories after cleanup.

- [ ] **Step 6: Compare performance**

Write CSV/Markdown comparing:

- 81.426-second old exact wall-clock;
- fresh optimized exact wall-clock;
- validation time;
- opt calls;
- pass invocations saved;
- reuse/cache hits;
- DAG nodes/edges;
- final result equality.

- [ ] **Step 7: Run budgeted on-demand Salsa20**

Use the matched budgeted command with:

```text
--budgeted-validation-strategy on-demand
--max-batches-per-state 2
--beam-width 2
--no-batchify-terminal-states
```

Record scope, objective, validation work, certification, replay, and wall-clock. Do not compare its objective as if it were exact.

- [ ] **Step 8: Commit documentation**

```powershell
git add docs/phasebatch_project_logic_zh.md
git commit -m "docs: explain validation performance runtime"
```

- [ ] **Step 9: Final review**

Review the complete diff for:

- accidental changes outside task files;
- non-deterministic iteration;
- stale path reuse;
- incorrect hard-certificate promotion;
- nested oversubscription;
- missing cost fields;
- exact/on-demand boundary violations.
