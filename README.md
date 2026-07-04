# Phase Ordering MVP Data System

This repository contains a data-producing MVP for LLVM phase-ordering research.
The first milestone is a small Python CLI that will grow into a pipeline for:

- LLVM toolchain metadata capture;
- active/dormant pass profiling;
- dynamic AB/BA pass-pair testing;
- conflict graph statistics;
- CSV and Markdown reports.

The current bootstrap provides the `phasebatch` package, CLI command skeletons,
and a small stdlib-only config loader.

## Quick Start

```bash
python -m phasebatch --help
python -m phasebatch analyze --help
python -m phasebatch batch --help
python -m phasebatch analyze --input x.c --out outputs/x --passes configs/core_passes.yaml
```

On this machine, use the DLM Conda environment:

```bash
D:/Miniconda/envs/dlm/python.exe -m phasebatch --help
```

`scripts/run_smoke.sh` prefers `D:/Miniconda/envs/dlm/python.exe` when present,
then a `dlm` command if one is on PATH, then `python`.
