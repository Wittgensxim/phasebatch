# Advisor Data Report v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build deterministic LLVM test-suite orchestration and Chinese advisor-ready CSV, chart, DAG, and Markdown reporting without changing Phasebatch correctness or search algorithms.

**Architecture:** Add a thin runner around the existing optimizer and a separately callable offline summarizer. Keep benchmark discovery, metric extraction, plotting, and Markdown in focused modules; all human conclusions are derived from persisted CSV and carry the state-local correctness boundary.

**Tech Stack:** Python standard library, existing Phasebatch APIs, PyYAML, matplotlib, Graphviz through the existing DAG visualizer, pytest/unittest.

---

### Task 1: Benchmark discovery and deterministic selection

**Files:**
- Create: `phasebatch/advisor_benchmarks.py`
- Create: `tests/test_advisor_benchmarks.py`

- [ ] Write tests covering `.c` filtering, source-size skipping, compile failures, directory caps, manifest validation, `num_programs`, and repeated deterministic output.
- [ ] Run `python -m pytest -q tests/test_advisor_benchmarks.py` and confirm missing imports/functions fail.
- [ ] Implement candidate records, YAML loading, clang smoke compilation, deterministic category-aware selection, and three output files.
- [ ] Re-run the benchmark tests and confirm they pass.

### Task 2: Core metric helpers and graph construction

**Files:**
- Create: `phasebatch/advisor_metrics.py`
- Create: `tests/test_advisor_metrics.py`

- [ ] Write tests for commute ratios, nearest-rank p90, overlap/conflict components including singleton nodes, small-cluster ABBA denominators, and `log10(n!)` with zero executable batches.
- [ ] Run the focused tests and confirm failure.
- [ ] Implement typed conversion helpers, deterministic components, component summaries, relation summaries, ABBA aggregation, and reduction rows.
- [ ] Re-run the focused tests.

### Task 3: Coverage, cost, conflict ranking, and state-aware metrics

**Files:**
- Modify: `phasebatch/advisor_metrics.py`
- Modify: `tests/test_advisor_metrics.py`

- [ ] Add failing tests for missing coverage versus zero, dropped-pass warning facts, wall-clock/cumulative-work separation, weighted conflict score, enable/suppress/effect changes, true flips, and availability changes.
- [ ] Implement per-run parsing and all requested CSV writers, recording unavailable sources in `missing_outputs.csv`.
- [ ] Assert deterministic field order and row sorting for every table.
- [ ] Run metric tests.

### Task 4: Chinese figures

**Files:**
- Create: `phasebatch/advisor_figures.py`
- Create: `tests/test_advisor_figures.py`

- [ ] Write tests that build minimal source CSV, patch font discovery to empty, and require PNG/SVG plus manifest rows without exceptions.
- [ ] Implement Chinese font configuration, nine requested plot families, readable no-data placeholders, and manifest warning/status fields.
- [ ] Run figure tests and inspect generated dimensions.

### Task 5: DAG export and Chinese report generation

**Files:**
- Create: `phasebatch/advisor_markdown.py`
- Create: `tests/test_advisor_markdown.py`

- [ ] Write failing tests for the 14 report sections, dropped-pass warning, conservative conditional conclusions, fixed correctness paragraph, five-minute talking-point structure, and data dictionary terms.
- [ ] Implement Markdown/table helpers, key-number CSV, report metadata JSON, talking points, and data dictionary.
- [ ] Add representative-program selection and a wrapper around existing `visualize_dag()` with DOT-only Graphviz degradation.
- [ ] Run Markdown and DAG tests.

### Task 6: Run and summarize orchestration

**Files:**
- Create: `phasebatch/advisor_report.py`
- Create: `tests/test_advisor_report.py`

- [ ] Write tests proving summarize never calls optimizer, resume reuses successful runs, fixed pairwise/full/auto/validated settings reach `optimize_batches()`, continue-on-error records failures, and metadata includes tool/config/git/command/time fields.
- [ ] Implement the study runner and idempotent summarizer using existing summary/equality/coverage/DAG functions where available.
- [ ] Run orchestration tests.

### Task 7: CLI integration and documentation

**Files:**
- Modify: `phasebatch/cli.py`
- Modify: `tests/test_cli_bootstrap.py`
- Modify: `tests/test_cli_pipeline.py`
- Modify: `README.md`
- Modify: `docs/project_status.md`
- Modify: `docs/code_file_roles.md`

- [ ] Add parser tests for both commands, defaults, strict-worker inheritance, and dispatch.
- [ ] Add both parsers and thin wrappers.
- [ ] Document commands, outputs, correctness boundary, and resume behavior.
- [ ] Run CLI and documentation tests.

### Task 8: Full verification and 20-program study

**Files:**
- Output: `outputs/advisor_report_zh_20programs/`

- [ ] Run `D:\Miniconda\envs\dlm\python.exe -m pytest -q` and require zero failures.
- [ ] Run the requested 20-program command with strict worker defaults, `--resume`, and `--continue-on-error`.
- [ ] Run `summarize-advisor-report-zh` again and verify no optimizer timestamps or run rows changed.
- [ ] Check every required CSV/Markdown/PNG/SVG, inspect representative images, and verify normal cleanup leaves no `.ll` files or empty directories.
