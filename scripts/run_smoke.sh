#!/usr/bin/env sh
set -eu

if [ -n "${PYTHON_CMD:-}" ]; then
  :
elif [ -x "D:/Miniconda/envs/dlm/python.exe" ]; then
  PYTHON_CMD="D:/Miniconda/envs/dlm/python.exe"
elif [ -x "/d/Miniconda/envs/dlm/python.exe" ]; then
  PYTHON_CMD="/d/Miniconda/envs/dlm/python.exe"
elif command -v dlm >/dev/null 2>&1; then
  PYTHON_CMD="dlm run python"
else
  PYTHON_CMD="python"
fi

$PYTHON_CMD -m phasebatch --help
$PYTHON_CMD -m phasebatch analyze --help
$PYTHON_CMD -m phasebatch batch --help
$PYTHON_CMD -m phasebatch analyze \
  --input benchmarks/tiny/branch.c \
  --out outputs/smoke_branch \
  --passes configs/core_passes.yaml \
  --jobs 2 \
  --timeout 10 \
  --max-pairs 20

OPT_CMD="${OPT_CMD:-E:/llvm/build/bin/opt.exe}"
if [ -x "$OPT_CMD" ]; then
  "$OPT_CMD" -S -verify-each -passes=function\(instcombine\) \
    outputs/smoke_branch/input.ll \
    -o outputs/smoke_branch/artifacts/instcombine_smoke.ll
fi
