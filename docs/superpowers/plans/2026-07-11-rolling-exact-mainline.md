# Rolling Exact Mainline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make complete two-layer rolling windows with a deterministic five-state boundary frontier the maintained Phasebatch mainline while preserving H=3, fixed-depth exact, and budgeted modes for reproducible comparisons.

**Architecture:** Use a deterministic rolling scheduler beside the existing exact and budgeted schedulers. It reuses current pairwise construction, validation DAG, correctness classifier, canonical state merge, and final artifact writers; each default window expands every safe transition for two layers, then retains up to five open states using objective/call/memory/branch/novelty buckets. Correctness completeness and global frontier completeness are reported separately.

**Tech Stack:** Python 3, LLVM worker backend, CSV/Markdown artifacts, `unittest`/pytest, existing Phasebatch optimizer and report modules.

---

## File Map

- Modify `phasebatch/optimizer.py`: rolling scheduler, status/scope metadata, selected-state override, window CSV.
- Modify `phasebatch/cli.py`: rolling mode/options and maintained defaults.
- Modify `phasebatch/advisor_report.py`: formal study defaults and rolling option forwarding.
- Modify `phasebatch/advisor_markdown.py`: unambiguous rolling configuration labels.
- Modify `phasebatch/final_summary.py`: expose rolling scope and closure evidence.
- Modify `tests/test_optimizer.py`: rolling expansion, continuation, closure, and incomplete tests.
- Modify `tests/test_cli_pipeline.py`: parser defaults and forwarding tests.
- Modify `tests/test_advisor_report.py`: advisor mainline forwarding tests.
- Modify `tests/test_advisor_markdown.py`: rolling configuration rendering test.
- Modify `docs/phasebatch_project_logic_zh.md`, `docs/project_status.md`, and `docs/code_file_roles.md`: current mainline documentation.

### Task 1: Freeze the Rolling Search Contract in Tests

- [x] Add a failing optimizer test that requests `mode="rolling-exact"`, uses horizon two, and proves all root candidates execute despite `beam_width=1` and `max_batches_per_state=1`.
- [x] Run the focused test and confirm it fails because the mode is unsupported.
- [x] Add a failing two-window chain test whose best depth-two terminal has further work; assert the final path spans the next window.
- [x] Add failing closure tests for no active passes, no executable batches, and a canonical cycle.
- [x] Add failing incomplete tests for a state cap and positive window cap.

Run:

```powershell
python -m pytest tests/test_optimizer.py -k rolling_exact -q
```

Expected before implementation: failures caused by unsupported mode or missing rolling fields.

### Task 2: Implement Deterministic Rolling Exact Search

- [x] Extend optimizer arguments with `rolling_window_depth`, `rolling_frontier_width=5`, and `max_rolling_windows=0`; validate positive values and non-negative cap.
- [x] Apply exact safety restrictions to both `exact` and `rolling-exact`.
- [x] Implement `_run_rolling_exact` with complete BFS expansion inside each window and no batch/beam selection calls.
- [x] Keep window-local routes separate from global exploratory paths; commit only the selected terminal route.
- [x] Detect active/executable closure, committed-state cycles, state caps, apply failures, and window caps.
- [x] Add deterministic `rolling_windows.csv` and optimizer events.
- [x] Allow `_finish_run` to receive the last committed state explicitly.
- [x] Record rolling status, scope, window count, committed depth, and closure reason in metadata, summary, and return values.
- [x] Run rolling tests until green, then run all optimizer tests.

Run:

```powershell
python -m pytest tests/test_optimizer.py -q
```

Expected: all optimizer tests pass; legacy exact/budgeted behavior is unchanged.

### Task 3: Switch CLI and Advisor Mainline Defaults

- [x] Add failing parser tests for default mode `rolling-exact`, configured window depth, frontier width five, and unlimited window cap.
- [x] Add failing advisor test asserting 50-program formal default, no default pair cap, rolling mode, and rolling option forwarding.
- [x] Extend CLI mode choices and forwarding without changing explicit legacy mode behavior.
- [x] Change Advisor defaults to rolling exact, 50 programs, state cap 2000, and `max_pairs=None`.
- [x] Render rolling depth/window cap separately from legacy `max_rounds`; never show beam width as a rolling control.
- [x] Run CLI, advisor, and Markdown tests until green.

Run:

```powershell
python -m pytest tests/test_cli_pipeline.py tests/test_advisor_report.py tests/test_advisor_markdown.py -q
```

### Task 4: Update Project Documentation

- [x] Update the Chinese project logic document with rolling window pseudocode, closure rules, exact boundary, and legacy mode comparison.
- [x] Update project status so the maintained path is full pairwise + certified validation + rolling exact + strict worker.
- [x] Update code-file roles with the new scheduler and artifacts.
- [x] State clearly that the existing 20-program budgeted report is pilot evidence and that formal reruns require new output directories.

### Task 5: Regression and End-to-End Verification

- [x] Run focused optimizer/report suites.
- [x] Run the complete test suite.
- [x] Run one small worker-backed rolling exact smoke test with the original horizon-two implementation.

## Revision: Three-Layer K=5 Frontier

- [x] Add RED tests proving five checkpoint states enter the next window.
- [x] Prove K is not applied inside the configured complete window.
- [x] Exclude already closed states from continuation slots.
- [x] Reuse the objective/call/memory/branch/novelty selection buckets.
- [x] Add `rolling_frontier_pruned`, `rolling_frontier_states_pruned`, and `global_search_complete` metadata.
- [x] Extend `rolling_windows.csv` and `frontier_scores.csv` with deterministic checkpoint evidence.
- [x] Initially compare H=3,K=5 against H=2,K=5 without removing either mode.
- [x] Pass the full Python regression suite.
- [x] Compare H=2,K=5 and H=3,K=5 sequentially on five representative programs.
- [x] Fix borrowed-handle materialization races so worker count cannot change pair classification.
- [x] Verify 8/12/16 worker scaling and keep 8 as the local default.
- [x] Adopt H=2,K=5 as the default after H=3 doubled wall time without improving five-program objectives.
- [ ] Run the formal 50-program H=2,K=5 study.
- [x] Verify status/scope/window CSV, deterministic chosen path, pipeline replay, `.ll` cleanup, and absence of empty directories.
- [x] Review `git diff` for accidental output or unrelated source changes.

Run:

```powershell
python -m pytest -q
```

Expected: zero failures.
