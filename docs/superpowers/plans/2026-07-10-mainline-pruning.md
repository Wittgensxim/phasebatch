# Phasebatch Mainline Pruning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove abandoned algorithms, legacy experiment wrappers, and historical outputs while preserving the current pairwise worker/staged/advisor mainline.

**Architecture:** Prune public CLI surfaces first, then remove their isolated modules and tests. Keep the pairwise optimizer contract stable, update current documentation to the reduced surface, and delete generated artifacts only through an absolute-path allowlist check.

**Tech Stack:** Python 3.10+, argparse, pytest, PowerShell, LLVM worker.

---

### Task 1: Remove CEGAR

**Files:**
- Modify: `phasebatch/cli.py`
- Modify: `phasebatch/optimizer.py`
- Modify: `phasebatch/batch_explorer.py`
- Modify: `phasebatch/state_analysis.py`
- Modify: `phasebatch/schema.py`
- Delete: `phasebatch/cegar_batcher.py`
- Delete: `phasebatch/cegar_comparison.py`
- Delete: `tests/test_cegar_batcher.py`
- Delete: `tests/test_cegar_comparison.py`
- Delete: `tests/test_cegar_optimizer.py`

- [ ] Remove imports, dispatch branches, exact-scope handling and CEGAR fields.
- [ ] Keep `batch_construction_mode=pairwise` metadata compatibility.
- [ ] Update optimizer, batch explorer, architecture and CLI tests.
- [ ] Run the focused optimizer/batcher/CLI tests.

### Task 2: Remove Legacy Study Wrappers

**Files:**
- Modify: `phasebatch/cli.py`
- Delete: isolated study modules listed in the design.
- Delete: corresponding `tests/test_*.py` files.
- Modify: `tests/test_cli_bootstrap.py`
- Modify: `tests/test_cli_pipeline.py`

- [ ] Remove parser registrations, handlers and public wrapper functions.
- [ ] Delete modules and dedicated tests after imports are gone.
- [ ] Confirm retained report and optimizer modules do not import deleted code.
- [ ] Run CLI and architecture tests.

### Task 3: Prune Configs, Scripts and Docs

**Files:**
- Modify: `README.md`
- Modify: `docs/project_status.md`
- Modify: `docs/phasebatch_project_logic_zh.md`
- Modify: `docs/code_file_roles.md`
- Modify: `docs/pass_sets.md`
- Delete: obsolete configs, scripts and CEGAR design records.

- [ ] Change examples to `configs/core_passes_v1.yaml`.
- [ ] Remove deleted command and module descriptions.
- [ ] Document the reduced command/config surface.
- [ ] Scan non-history docs for removed symbols.

### Task 4: Delete Historical Artifacts

**Files:**
- Preserve: the seven output directories in the design.
- Delete: all other generated output directories and root output logs.
- Delete: caches, bytecode, zip/task/PDF artifacts.

- [ ] Resolve every target under `E:\PO2` before deletion.
- [ ] Delete only non-allowlisted output entries.
- [ ] Verify preserved evidence files still exist.
- [ ] Report reclaimed disk space.

### Task 5: Verification

**Files:**
- Test: retained `tests/` suite.

- [ ] Run `D:\Miniconda\envs\dlm\python.exe -m pytest -q`.
- [ ] Run CLI help for optimize, staged, worker and advisor commands.
- [ ] Scan source/tests/current docs for removed modules and commands.
- [ ] Verify the output allowlist and complete Advisor artifact gate.
