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
  --input x.c \
  --out outputs/x \
  --passes configs/core_passes.yaml
