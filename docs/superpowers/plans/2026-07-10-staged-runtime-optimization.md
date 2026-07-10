# Staged Runtime Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the manually validated Salsa20 staged workflow into a deterministic Phasebatch command with inlinable root IR, multi-signal frontier preservation, optional safe top-K runtime reranking, and an isolated vector experiment.

**Architecture:** Keep `optimize_batches` as the per-stage correctness engine. Add a thin staged orchestrator that materializes one root, executes stage manifests sequentially with full pairwise/DAG validation, optionally reranks reached terminal states, replays all selected stage segments, writes aggregate evidence, and performs one final IR cleanup.

**Tech Stack:** Python 3.10, argparse, LLVM New Pass Manager, CSV/JSON/YAML artifacts, unittest/pytest.

---

### Task 1: Inlinable Root IR

**Files:** `phasebatch/runner.py`, `phasebatch/optimizer.py`, `phasebatch/cli.py`, `tests/test_runner.py`, `tests/test_cli_pipeline.py`

- [ ] Add failing tests for `legacy-o0` and `inlinable-unoptimized` Clang command construction and unknown-mode rejection.
- [ ] Add `root_ir_mode` with backward-compatible `legacy-o0` default.
- [ ] Expose `--root-ir-mode` on `optimize-batches` and record it in metadata.
- [ ] Run focused runner/CLI tests.

### Task 2: Staged Orchestrator

**Files:** `phasebatch/staged_optimizer.py`, `phasebatch/staged_config.py`, `phasebatch/cli.py`, `tests/test_staged_optimizer.py`, `tests/test_staged_config.py`

- [ ] Add failing manifest parsing tests for deterministic stage IDs, relative pass paths, required fields, and mixed exact/budgeted scope.
- [ ] Add failing orchestration tests proving stage handoff, selected pipeline aggregation, sequential replay, and final cleanup.
- [ ] Implement `optimize-staged --input --manifest --out` while reusing `optimize_batches` unchanged as the stage engine.
- [ ] Emit `staged_summary.csv`, `staged_pipeline.csv`, `staged_replay.csv`, `staged_summary.md`, and aggregate metadata.

### Task 3: Multi-Signal Frontier

**Files:** `phasebatch/normalizer.py`, `phasebatch/optimizer.py`, `phasebatch/schema.py`, `tests/test_normalizer.py`, `tests/test_optimizer.py`, `tests/test_schema.py`

- [ ] Add failing tests for direct, intrinsic, indirect, and tail call classification.
- [ ] Add failing beam tests for deterministic `direct_call_bucket`, `memory_bucket`, and `branch_bucket` preservation.
- [ ] Record per-state features and parent/root deltas without changing correctness classification.
- [ ] Extend Pareto comparison and frontier CSV fields; retain IR objective for legacy final selection.

### Task 4: Safe Runtime Top-K

**Files:** `phasebatch/runtime_rerank.py`, `phasebatch/staged_optimizer.py`, `tests/test_runtime_rerank.py`, `tests/test_staged_optimizer.py`

- [ ] Add failing tests that select only reached, replayable terminal states and deduplicate hashes.
- [ ] Add failing tests for cyclic interleaving, median ranking, nonzero-exit rejection, and deterministic tie breaks.
- [ ] Compile candidate IR with `llc`, link with `clang`, run only manifest-approved commands, and retain raw trials.
- [ ] Let the staged orchestrator use a runtime winner only after successful compile/run verification; runtime remains an objective signal, never correctness proof.

### Task 5: Isolated E5 Vector Stage

**Files:** `configs/vector_cleanup_passes_v5.yaml`, `configs/staged_salsa20_v5.yaml`

- [ ] Audit explicit manager-compatible candidates for `loop-vectorize`, `slp-vectorizer`, `vector-combine`, and `loop-unroll<O2>`.
- [ ] Keep vector and cleanup candidates in a small final stage so top-K can retain transformations that increase static IR.
- [ ] Run Salsa20 with the staged command and compare runtime, search work, replay, and O2 gap.

### Task 6: Verification and Documentation

**Files:** `README.md`, `docs/phasebatch_project_logic_zh.md`, `docs/project_status.md`

- [ ] Run focused red/green tests after every task and the full test suite at the end.
- [ ] Verify all selected stages replay, all timed executions return the expected code, and no unapproved candidate executes.
- [ ] Verify zero remaining `.ll` files and zero empty directories when keep/dump flags are disabled.
- [ ] Document the new command, exact-scope boundary, runtime nondeterminism, and E5 evidence.
