# In-Process LLVM Worker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a crash-isolated, long-lived LLVM New PM worker pool and replace external `opt.exe` calls without changing Phasebatch correctness.

**Architecture:** A C++ NDJSON worker owns LLVM contexts and module handles. A Python backend pool exposes the existing `run_opt` contract, first with file-compatible materialization and then with handle-aware lazy materialization for pair and validation transitions.

**Tech Stack:** C++17, LLVM 23 C++ APIs, CMake/Ninja, Python 3.10, NDJSON, pytest.

---

### Task 1: Python Protocol Client

**Files:**
- Create: `phasebatch/opt_worker.py`
- Create: `tests/test_opt_worker.py`

- [x] Write failing tests for request IDs, successful responses, stderr capture, timeout restart, malformed JSON, EOF, and shutdown using a fake line-oriented worker process.
- [x] Implement `WorkerProcess` with reader/stderr threads, a single-request lock, timeout enforcement, restart, and structured `WorkerError` subclasses.
- [x] Implement `WorkerPool` with deterministic checkout, round-robin fairness, clean shutdown, and counters; backend policy owns external fallback.
- [x] Run `pytest -q tests/test_opt_worker.py`.

### Task 2: C++ Worker and Build

**Files:**
- Create: `worker/CMakeLists.txt`
- Create: `worker/phasebatch_worker.cpp`
- Create: `scripts/build_worker.ps1`
- Create: `tests/test_worker_binary.py`

- [x] Write binary smoke tests for `ping`, invalid operations, load, nested pipeline apply, invalid pipeline, materialization, release, and shutdown; skip only when the worker has not been built.
- [x] Add a CMake target using `find_package(LLVM CONFIG)`, LLVM definitions/includes, and the Core, Support, IRReader, Passes, Analysis, TransformUtils, ScalarOpts, InstCombine, IPO, Vectorize, BitWriter, and LLVMDiff components.
- [x] Implement NDJSON parsing with `llvm::json`, module loading with `parseIRFile`, cloning with `CloneModule`, New PM construction with fresh analysis managers and `StandardInstrumentations`, pipeline parsing, verification, SHA-256-backed handles, feature counts, atomic IR materialization, and in-process structural comparison.
- [x] Build with `scripts/build_worker.ps1` against `E:/llvm/build/lib/cmake/llvm`.
- [x] Run `pytest -q tests/test_worker_binary.py`.

### Task 3: File-Compatible `run_opt` Backend

**Files:**
- Create: `phasebatch/opt_backend.py`
- Modify: `phasebatch/runner.py`
- Modify: `phasebatch/schema.py`
- Modify: `phasebatch/cli.py`
- Modify: `phasebatch/tools.py`
- Test: `tests/test_opt_backend.py`
- Test: `tests/test_runner.py`
- Test: `tests/test_cli_pipeline.py`

- [x] Write failing tests proving external behavior is unchanged, worker responses map to `RunResult`, output files are required in file-compatible mode, and CLI/env configuration resolves deterministically.
- [x] Add `WorkerOptBackend`, a process-lifetime backend registry, and explicit shutdown while preserving external execution in the existing runner path.
- [x] Make `runner.run_opt` dispatch through the configured backend while retaining the exact command and stderr artifacts for external mode.
- [x] Add `--opt-backend`, `--opt-worker`, and `--opt-workers` to common and staged CLI paths and record backend metadata.
- [x] Run focused runner, CLI, and backend tests.

### Task 4: Differential Correctness Gate

**Files:**
- Create: `phasebatch/worker_differential.py`
- Create: `tests/test_worker_differential.py`
- Modify: `phasebatch/cli.py`

- [x] Write failing tests for comparison rows, canonical-hash mismatch, fingerprint mismatch, invalid-pipeline parity, and summary failure status.
- [x] Implement `verify-opt-worker` to run external and worker pipelines over configurable inputs/pass configs and emit `worker_differential.csv`, `worker_differential_summary.csv`, and `worker_differential.md`.
- [x] Include no-op, dormant, active, nested loop, invalid, pair AB/BA, and replay-shaped pipelines.
- [x] Refuse worker-as-default recommendations when any semantic differential fails.

### Task 5: Handle Cache and Lazy Materialization

**Files:**
- Modify: `phasebatch/opt_worker.py`
- Modify: `phasebatch/opt_backend.py`
- Modify: `phasebatch/schema.py`
- Modify: `phasebatch/pair_tester.py`
- Modify: `phasebatch/batch_validation_dag.py`
- Modify: `phasebatch/validation_runtime.py`
- Test: `tests/test_opt_worker.py`
- Test: `tests/test_pair_tester.py`
- Test: `tests/test_batch_validation_dag.py`

- [x] Write failing tests for path-to-handle reuse, worker affinity, non-materialized equal-hash pair results, hash-different fallback materialization, dump/keep-forced materialization, worker restart invalidation, and LRU release.
- [x] Extend `RunResult` with worker ID, handle, canonical hash, feature counts, and materialized flag.
- [x] Add `materialize=False` support and a `materialize_result` operation.
- [x] Use worker hash metadata for canonical-hash equality; materialize only hash-different pairs before the unchanged structural-diff ladder.
- [x] Preserve existing CSV fields and add explicit materialization/cache metrics.

### Task 6: Performance Accounting and Full Benchmark

**Files:**
- Create: `phasebatch/worker_benchmark.py`
- Create: `tests/test_worker_benchmark.py`
- Modify: `phasebatch/cli.py`
- Modify: `phasebatch/schema.py`

- [x] Write failing tests for external/worker benchmark accounting, percentile summaries, speedup, cache hit rates, restart counts, and acceptance status.
- [x] Implement `benchmark-opt-worker` with startup, no-op, single-pass, pair, and validation-shaped workloads.
- [x] Run the 100-call Salsa20 microbenchmark and require at least 3x file-compatible throughput.
- [x] Run matched Salsa20 external and worker full flows; compare objective, states, transitions, pair relations, validation, selected pipeline, replay, wall-clock, opt invocations, and cleanup.

### Task 7: Documentation and Final Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/project_status.md`
- Modify: `docs/phasebatch_project_logic_zh.md`
- Modify: `docs/code_file_roles.md`

- [x] Document build commands, backend selection, protocol scope, crash/fallback behavior, differential evidence, metrics, and limitations.
- [x] Run `D:/Miniconda/envs/dlm/python.exe -m pytest -q`.
- [x] Verify external remains compatible, worker outputs pass the differential gate, runtime artifacts are deterministic, and normal cleanup leaves zero `.ll` files and zero empty directories.

Completion evidence: 494 tests and 31 subtests passed; the full Salsa20 matched run retained the same selected pipeline, objective, state/transition counts, pair decisions, validation classifications, and replay result. Wall-clock fell from 87.798 s to 25.053 s (3.504x), and normal cleanup left zero `.ll` files and zero empty directories in both compared output trees. The final 100-iteration file-compatible microbenchmark reached 6.847x aggregate median speedup. Core-v1 and middle-end differential gates passed 44/44 and 88/88 cases after fresh-context and crash-recovery hardening.
