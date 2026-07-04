# Codex Implementation Plan: Phase Ordering MVP Data System

## 0. Goal for the next meeting

Build a **data-producing MVP**, not a full optimizer.

The next meeting should show concrete evidence for 5–10 small programs:

1. how many passes are active/dormant on the current IR;
2. how many active pass pairs commute by dynamic `A;B == B;A` testing;
3. how many pairs are order-sensitive / failed / unknown;
4. how large the overlap/conflict components are;
5. which passes cause the largest conflict components;
6. how much the profiling and pair testing cost;
7. a generated `summary.md` report plus CSV files.

This MVP uses **coarse pass-level effects**. It does not claim effect-level correctness yet. It is enough for the upcoming progress meeting because it produces the structural data the advisor asked for.

---

## 1. MVP scope

### In scope

- LLVM IR `.ll` input, optionally compile `.c` to `.ll`.
- LLVM middle-end `opt` passes only.
- Python wrapper around `clang` and `opt`.
- Active/dormant profiling for a configurable pass set.
- Canonical-ish IR hash.
- Coarse IR diff by function and basic block.
- Pairwise `A;B` vs `B;A` testing for active passes.
- Relation classification.
- Conflict component statistics.
- CSV + Markdown report.
- Parallel execution.

### Out of scope for this week

- LLVM pass plugin.
- Precise effect-level read/write instrumentation.
- Alive2.
- ML.
- Runtime benchmarking.
- Full LLVM pass set.
- Complete state-space search.
- Any claim of global optimality.

---

## 2. Repository layout

Create this structure:

```text
phase-order-mvp/
  README.md
  pyproject.toml
  configs/
    core_passes.yaml
  benchmarks/
    tiny/
      branch.c
      loop.c
      memory.c
      arithmetic.c
  outputs/
    .gitkeep
  phasebatch/
    __init__.py
    cli.py
    config.py
    tools.py
    runner.py
    normalizer.py
    ir_parser.py
    profiler.py
    pair_tester.py
    relation.py
    graph.py
    report.py
    schema.py
  scripts/
    run_smoke.sh
    run_mvp.sh
  tests/
    test_normalizer.py
    test_ir_parser.py
    test_graph.py
```

Use only Python standard library for the MVP unless absolutely necessary:

```text
argparse
subprocess
pathlib
hashlib
json
csv
re
time
statistics
itertools
concurrent.futures
shutil
```

---

## 3. Default pass set

Create `configs/core_passes.yaml`:

```yaml
passes:
  - mem2reg
  - sroa
  - instcombine
  - aggressive-instcombine
  - instsimplify
  - simplifycfg
  - early-cse
  - dce
  - adce
  - bdce
  - reassociate
  - gvn
  - jump-threading
  - correlated-propagation
```

The system must validate passes before use. If a pass name is invalid for the local LLVM version, record it in `invalid_passes.csv` and skip it.

---

## 4. CLI targets

The main command should be:

```bash
python -m phasebatch analyze \
  --input benchmarks/tiny/branch.c \
  --out outputs/branch \
  --passes configs/core_passes.yaml \
  --jobs 8 \
  --timeout 10 \
  --max-pairs 300
```

It should produce:

```text
outputs/branch/
  input.ll
  metadata.json
  valid_passes.csv
  invalid_passes.csv
  pass_profile.csv
  pair_relation.csv
  cluster_distribution.csv
  per_state_summary.csv
  summary.md
  artifacts/
    single_pass/
    pairs/
```

A multi-program driver should be:

```bash
python -m phasebatch batch \
  --inputs benchmarks/tiny/*.c \
  --out outputs/mvp_run \
  --passes configs/core_passes.yaml \
  --jobs 8 \
  --timeout 10 \
  --max-pairs 300
```

It should produce one directory per program and an aggregate `outputs/mvp_run/aggregate_summary.md`.

---

## 5. Data schemas

### 5.1 `valid_passes.csv`

```text
pass,valid,reason,test_time_ms
```

### 5.2 `pass_profile.csv`

```text
program,state_hash,pass,success,active,input_hash,output_hash,
inst_before,inst_after,inst_delta,funcs_changed,blocks_changed,
changed_functions,changed_blocks,time_ms,stderr_path,failure_kind
```

### 5.3 `pair_relation.csv`

```text
program,state_hash,pass_a,pass_b,
a_active,b_active,
static_relation,dynamic_relation,final_relation,
ab_success,ba_success,ab_hash,ba_hash,same_hash,
ab_inst,ba_inst,inst_delta_ab_ba,
changed_funcs_a,changed_funcs_b,changed_blocks_a,changed_blocks_b,
overlap_functions,overlap_blocks,
time_ms,failure_kind,ab_path,ba_path
```

Recommended relation values:

```text
static_disjoint_function
static_disjoint_block
static_overlap_function
static_overlap_block
static_unknown

dynamic_commute
dynamic_order_sensitive
dynamic_failed
dynamic_timeout
not_tested

final_commute
final_order_sensitive
final_unknown
```

### 5.4 `cluster_distribution.csv`

```text
program,state_hash,graph_type,num_nodes,num_edges,num_components,
mean_size,median_size,max_size,size_1,size_2,size_3,size_4_7,size_gt_7
```

Graph types:

```text
order_sensitive_graph
unknown_graph
overlap_graph
noncommute_graph
```

### 5.5 `per_state_summary.csv`

```text
program,state_hash,pass_set_size,valid_passes,invalid_passes,
active_passes,dormant_passes,total_pairs,pairs_tested,
dynamic_commute,order_sensitive,unknown,failed,
static_disjoint_function,static_disjoint_block,
max_conflict_component,median_conflict_component,
profile_time_ms,pair_time_ms,total_time_ms
```

---

## 6. Task list for Codex

Use one task per Codex session. Do not ask Codex to build the whole system at once.

---

### Task 0 — Project bootstrap

**Goal:** create repository skeleton, package, CLI entry point, config loader.

**Files:**

```text
pyproject.toml
phasebatch/__init__.py
phasebatch/cli.py
phasebatch/config.py
configs/core_passes.yaml
scripts/run_smoke.sh
```

**Required behavior:**

```bash
python -m phasebatch --help
python -m phasebatch analyze --help
python -m phasebatch batch --help
```

**Codex prompt:**

```text
Create a Python project for an LLVM phase-ordering data MVP. Use only stdlib. Implement package phasebatch with CLI commands analyze and batch. Add a config loader that reads configs/core_passes.yaml. Do not implement LLVM execution yet; create stubs that print parsed arguments. Include pyproject.toml, README skeleton, and scripts/run_smoke.sh. The CLI must run via python -m phasebatch.
```

**Acceptance:**

- `python -m phasebatch --help` works.
- `python -m phasebatch analyze --input x.c --out outputs/x --passes configs/core_passes.yaml` parses arguments.

---

### Task 1 — Toolchain detection

**Goal:** detect `clang`, `opt`, `llc`, `llvm-size`; capture versions.

**Files:**

```text
phasebatch/tools.py
phasebatch/schema.py
```

**Functions:**

```python
find_tool(name: str) -> str
run_version(tool: str) -> str
collect_toolchain() -> dict
write_metadata(out_dir: Path, metadata: dict) -> None
```

**Codex prompt:**

```text
Implement phasebatch/tools.py. It must find clang, opt, llc, llvm-size using shutil.which, collect --version output, and write metadata.json. Add clear RuntimeError messages if clang or opt is missing. Integrate this into analyze command so it creates output directory and metadata.json.
```

**Acceptance:**

```bash
python -m phasebatch analyze --input benchmarks/tiny/branch.c --out outputs/branch --passes configs/core_passes.yaml
cat outputs/branch/metadata.json
```

---

### Task 2 — C to LLVM IR compilation

**Goal:** convert `.c` to `.ll`, or accept `.ll` unchanged.

**Files:**

```text
phasebatch/runner.py
```

**Functions:**

```python
compile_c_to_ll(clang: str, src: Path, out_ll: Path, timeout: int) -> RunResult
prepare_input_ir(input_path: Path, out_dir: Path, tools: dict, timeout: int) -> Path
```

**Command:**

```bash
clang -O0 -Xclang -disable-O0-optnone -S -emit-llvm input.c -o input.ll
```

**Codex prompt:**

```text
Implement LLVM IR input preparation. If input is .ll, copy it to out/input.ll. If input is .c, run clang -O0 -Xclang -disable-O0-optnone -S -emit-llvm to produce out/input.ll. Use subprocess.run with timeout, capture stdout/stderr, and return a RunResult dataclass. Integrate into analyze command. Add two tiny benchmark C files under benchmarks/tiny.
```

**Acceptance:**

- `outputs/branch/input.ll` exists.
- Failure messages are written to stderr and metadata.

---

### Task 3 — Generic opt runner

**Goal:** run arbitrary pass sequence through `opt`.

**Files:**

```text
phasebatch/runner.py
```

**Functions:**

```python
run_opt(opt: str, input_ll: Path, passes: list[str], output_ll: Path, timeout: int) -> RunResult
```

**Command:**

```bash
opt -S -verify-each -passes="pass1,pass2" input.ll -o output.ll
```

**Codex prompt:**

```text
Implement run_opt. It must run opt -S -verify-each -passes=<comma-joined passes> input.ll -o output.ll. It must capture stdout/stderr, wall time in ms, exit code, timeout, and write stderr to a sidecar .stderr.txt file when requested. Add a small manual test in scripts/run_smoke.sh that runs instcombine on input.ll.
```

**Acceptance:**

- Running `opt` with `instcombine` creates an output `.ll`.
- Timeout produces `failure_kind=timeout`.

---

### Task 4 — IR normalizer and feature counter

**Goal:** produce stable enough hashes and basic feature counts.

**Files:**

```text
phasebatch/normalizer.py
```

**Functions:**

```python
normalize_ir_text(text: str) -> str
hash_text(text: str) -> str
canonical_hash(path: Path) -> str
count_ir_features(path: Path) -> dict
```

**Normalization v0:**

- remove `; ModuleID = ...`;
- remove `source_filename = ...`;
- strip comments starting with `;` unless line begins with `; <label>`;
- remove `, !dbg !N`;
- remove metadata definition lines beginning with `!`;
- collapse blank lines;
- trim trailing whitespace.

**Feature counts:**

```text
functions
basic_blocks
instructions
branches
loads
stores
calls
phis
selects
allocas
```

**Codex prompt:**

```text
Implement normalizer.py for textual LLVM IR. Add normalize_ir_text, canonical_hash, and count_ir_features. Use conservative regexes. Do not rename SSA values in v0. Add unit tests for stripping debug metadata and stable hashing.
```

**Acceptance:**

- Same file hash stable across repeated runs.
- Feature counts appear in `summary.md` later.

---

### Task 5 — LLVM IR parser for function/block hashes

**Goal:** find changed functions and basic blocks.

**Files:**

```text
phasebatch/ir_parser.py
```

**Data structures:**

```python
@dataclass
class IRSnapshot:
    module_hash: str
    functions: dict[str, str]
    blocks: dict[str, str]  # key: function::block_label
    features: dict[str, int]
```

**Functions:**

```python
parse_ir_snapshot(path: Path) -> IRSnapshot
changed_regions(before: IRSnapshot, after: IRSnapshot) -> dict
```

**Parser v0 rules:**

- function starts with `define ... @name(`;
- function ends at a line exactly `}`;
- basic block label line matches `^([A-Za-z$._-][\w$._-]*|\d+):`;
- if instructions appear before first label, use block label `entry`;
- hash normalized text within each function/block.

**Codex prompt:**

```text
Implement ir_parser.py. Parse textual LLVM IR into function hashes and basic block hashes. Implement changed_regions to return changed_functions, changed_blocks, added/deleted functions/blocks, and counts. Add tests with a small hand-written .ll string.
```

**Acceptance:**

- Can detect that `instcombine` changed at least one block on a tiny arithmetic program.

---

### Task 6 — Pass validation

**Goal:** skip invalid local LLVM pass names.

**Files:**

```text
phasebatch/profiler.py
```

**Functions:**

```python
validate_passes(input_ll: Path, passes: list[str], tools: dict, out_dir: Path, timeout: int) -> tuple[list[str], list[dict]]
```

**Behavior:**

- Run each pass on the input.
- Valid if `opt` exit code is 0.
- Invalid if parse error / unsupported / timeout / verifier failure.
- Write `valid_passes.csv` and `invalid_passes.csv`.

**Codex prompt:**

```text
Implement pass validation. For each pass in config, run opt with that single pass on input.ll into artifacts/validate/<pass>.ll. Write valid_passes.csv and invalid_passes.csv. Invalid passes must not crash the whole analysis. Integrate into analyze.
```

**Acceptance:**

- Bad pass name appears in `invalid_passes.csv`.
- Valid pass list is used by later stages.

---

### Task 7 — Single-pass profiler

**Goal:** collect active/dormant and footprint data.

**Files:**

```text
phasebatch/profiler.py
```

**Functions:**

```python
profile_passes(input_ll: Path, valid_passes: list[str], tools: dict, out_dir: Path, jobs: int, timeout: int) -> list[dict]
```

**Behavior:**

For each pass `p`:

1. run `p` on input;
2. hash input and output;
3. active iff hash differs;
4. compute feature deltas;
5. compute changed functions and blocks;
6. write `pass_profile.csv`.

**Codex prompt:**

```text
Implement parallel single-pass profiling using ThreadPoolExecutor. For each valid pass, run it on input.ll, compute canonical hashes, feature deltas, changed functions and blocks. Write pass_profile.csv with all schema fields. Integrate into analyze.
```

**Acceptance:**

- `pass_profile.csv` has one row per valid pass.
- Active and dormant counts appear in console output.

---

### Task 8 — Pairwise AB/BA tester

**Goal:** test commutation for active pass pairs.

**Files:**

```text
phasebatch/pair_tester.py
```

**Functions:**

```python
test_pairs(input_ll: Path, active_profiles: list[dict], tools: dict, out_dir: Path, jobs: int, timeout: int, max_pairs: int | None) -> list[dict]
```

**Behavior:**

For each unordered active pair `(A, B)`:

```text
AB = opt -passes="A,B"
BA = opt -passes="B,A"
if both success and hash(AB) == hash(BA): dynamic_commute
if both success and hash differs: dynamic_order_sensitive
else: dynamic_failed / timeout
```

If too many pairs, respect `--max-pairs`. Selection policy:

1. test pairs with overlapping changed functions first;
2. then pairs with active passes sorted by pass name;
3. record untested pairs as `not_tested` only if needed.

**Codex prompt:**

```text
Implement pair_tester.py. It must take active pass profiles and test unordered pairs by running A,B and B,A through opt. Compute AB and BA hashes and instruction counts. Use parallel execution. Respect max_pairs. Write artifacts/pairs/<A>__<B>/ab.ll and ba.ll. Return pair rows but do not classify static relation yet.
```

**Acceptance:**

- `pair_relation.csv` contains tested active pairs.
- At least one pair is classified `dynamic_commute` or `dynamic_order_sensitive` on tiny benchmarks.

---

### Task 9 — Static relation and final classification

**Goal:** combine single-pass footprints and dynamic tests.

**Files:**

```text
phasebatch/relation.py
```

**Functions:**

```python
static_relation(profile_a: dict, profile_b: dict) -> dict
final_relation(pair_row: dict) -> str
annotate_pair_relations(pair_rows: list[dict], profiles: dict[str, dict]) -> list[dict]
```

**Rules v0:**

```text
if changed_functions disjoint:
    static_relation = static_disjoint_function
elif changed_blocks disjoint:
    static_relation = static_disjoint_block
elif changed_blocks overlap:
    static_relation = static_overlap_block
else:
    static_relation = static_overlap_function or static_unknown

if dynamic_relation == dynamic_commute:
    final_relation = final_commute
elif dynamic_relation == dynamic_order_sensitive:
    final_relation = final_order_sensitive
else:
    final_relation = final_unknown
```

Important: static disjoint is reported as candidate evidence, but in MVP only dynamic commute is hard evidence.

**Codex prompt:**

```text
Implement relation.py that joins pair test rows with single-pass footprint rows. Add static_relation based on changed function/block set overlap and final_relation based on dynamic AB/BA result. Write the full pair_relation.csv schema.
```

**Acceptance:**

- CSV contains overlap counts and final relation.

---

### Task 10 — Conflict graph and component statistics

**Goal:** compute the numbers the advisor wants.

**Files:**

```text
phasebatch/graph.py
```

**Functions:**

```python
build_graph(pair_rows: list[dict], graph_type: str) -> dict[str, set[str]]
connected_components(graph: dict[str, set[str]]) -> list[set[str]]
component_stats(components: list[set[str]]) -> dict
write_cluster_distribution(...)
```

Graph definitions:

```text
order_sensitive_graph: edges where final_relation == final_order_sensitive
unknown_graph: edges where final_relation == final_unknown
noncommute_graph: order_sensitive + unknown
static_overlap_graph: edges with static_overlap_block/function
```

**Codex prompt:**

```text
Implement graph.py with simple undirected graph utilities using stdlib. Build order_sensitive, unknown, noncommute, and static_overlap graphs from pair_relation rows. Compute num_nodes, num_edges, num_components, mean/median/max component size, and bucket counts size_1,size_2,size_3,size_4_7,size_gt_7. Write cluster_distribution.csv.
```

**Acceptance:**

- `cluster_distribution.csv` exists and reports max component size.

---

### Task 11 — Report generator

**Goal:** generate meeting-ready `summary.md`.

**Files:**

```text
phasebatch/report.py
```

**Report sections:**

```text
# Summary
- input
- LLVM version
- valid pass count
- active/dormant count
- pair counts
- dynamic commute / order-sensitive / unknown / failed
- max/median conflict component size
- profiling time / pair testing time

# Top active passes
# Dynamic commute pairs
# Order-sensitive pairs
# Largest conflict components
# Invalid passes
# Generated files
# Caveats
```

**Codex prompt:**

```text
Implement report.py. It must read metadata.json, pass_profile.csv, pair_relation.csv, cluster_distribution.csv and write summary.md. The summary must be readable in a meeting, with markdown tables for relation counts, top active passes, dynamic commute pairs, order-sensitive pairs, and largest components. Integrate into analyze and batch.
```

**Acceptance:**

- `summary.md` can be opened and shown directly.

---

### Task 12 — Batch runner and aggregate report

**Goal:** run 5–10 programs and aggregate.

**Files:**

```text
phasebatch/cli.py
phasebatch/report.py
```

**Codex prompt:**

```text
Implement the batch command. It accepts multiple input paths, creates one output directory per input, runs the existing analyze pipeline, and writes aggregate_summary.md plus aggregate CSVs concatenating per-program pass_profile, pair_relation, per_state_summary, and cluster_distribution.
```

**Acceptance:**

```bash
python -m phasebatch batch --inputs benchmarks/tiny/*.c --out outputs/mvp_run --passes configs/core_passes.yaml --jobs 8 --timeout 10 --max-pairs 300
```

produces `outputs/mvp_run/aggregate_summary.md`.

---

### Task 13 — Optional: one-round state loop

**Goal:** show that relations change after IR changes.

**MVP behavior:**

1. choose a simple canonical batch: all active passes sorted by name that are not in the largest conflict component, capped at 3 passes;
2. run that batch;
3. repeat profiling and pair testing on the new state with `round=1`;
4. report relation flips.

**Codex prompt:**

```text
Add optional --rounds N. For N=1, after round 0 choose up to 3 passes that are active and not in the largest noncommute component, run them in sorted order to produce state_1.ll, then rerun analysis on state_1. Add round column to all CSVs. Generate a relation_flip_summary comparing pair final_relation across rounds.
```

**Acceptance:**

- `--rounds 1` produces round 0 and round 1 CSV rows.
- Summary says how many pair relations changed.

---

### Task 14 — Optional: objective metrics

**Goal:** include code size numbers if time remains.

**Behavior:**

- compile final IR to object with `clang -c` or `llc`;
- run `llvm-size`;
- report `.text` size.

**Codex prompt:**

```text
Add optional objective measurement. Given an .ll file, compile it to object with clang -c and run llvm-size. Parse .text size where possible. Add objective_eval.csv with baseline input, O2, Oz, and one generated batch sequence. This must not affect relation correctness; it is only an objective metric.
```

---

## 7. Smoke benchmarks

Add these tiny C files.

### `benchmarks/tiny/arithmetic.c`

```c
int f(int x, int y) {
  int a = x + 0;
  int b = y * 1;
  int c = (a + b) - 0;
  return c;
}
```

### `benchmarks/tiny/branch.c`

```c
int f(int x) {
  int y = x + 1;
  if (y > 0) return y;
  else return y + 0;
}
```

### `benchmarks/tiny/memory.c`

```c
int f(int *p) {
  int a = *p;
  int b = *p;
  return a + b;
}
```

### `benchmarks/tiny/loop.c`

```c
int f(int *a, int n) {
  int s = 0;
  for (int i = 0; i < n; i++) {
    s += a[i] * 1;
  }
  return s;
}
```

---

## 8. Meeting slide outline

Use the generated data to make 5 slides.

### Slide 1 — Problem framing

- Not trying to find global best pipeline.
- Current goal: measure state-local pass commutation and conflict structure.
- Pass type is not consumed; current MVP uses round 0, optionally round 1.

### Slide 2 — System pipeline

```text
C/LL input
  -> clang/opt runner
  -> single-pass profiler
  -> AB/BA tester
  -> relation classifier
  -> conflict graph
  -> CSV + summary report
```

### Slide 3 — Per-program data table

Columns:

```text
program | valid passes | active passes | tested pairs | commute | order-sensitive | unknown | max component | time
```

### Slide 4 — Conflict component distribution

Show:

```text
mean / median / max component size
size_1 / size_2 / size_3 / size_4_7 / size_gt_7
```

### Slide 5 — What this means

- If many dynamic commute pairs exist: ordering decisions can be folded.
- If conflict components are small: local exact search is plausible.
- If conflict components are large: need finer effect-level instrumentation.
- Next task: replace coarse pass-level footprint with effect-level hooks for top conflict passes.

---

## 9. What not to claim in the meeting

Do not claim:

```text
1. We already solve phase ordering.
2. Static disjoint at pass-level is a proof.
3. RW overlap is commute.
4. Code size equality proves independence.
5. We can enumerate all batch combinations.
6. The current pass-level footprint is final.
```

Do claim:

```text
1. We built a data pipeline that measures active/dormant passes and AB/BA commutation.
2. We can show concrete relation distributions on small programs.
3. We can identify largest conflict components and the passes responsible.
4. These numbers tell us whether effect-level instrumentation is necessary and where to add it first.
```

---

## 10. Minimum success criteria for the next meeting

A successful next meeting needs only this:

```text
- 5 tiny or LLVM-test-suite programs processed.
- 10–15 valid LLVM passes after validation.
- pass_profile.csv exists for all programs.
- pair_relation.csv exists for all programs.
- cluster_distribution.csv exists for all programs.
- aggregate_summary.md shows active/dormant, commute/order-sensitive/unknown, component sizes, and cost.
```

Even if code size results are absent, this is a valid research progress report because it directly addresses the structural question: are there many false ordering decisions, and how large are the remaining conflict components?
