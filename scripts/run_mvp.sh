#!/usr/bin/env sh
set -eu

if [ -n "${PYTHON_CMD:-}" ]; then
  :
elif [ -x "D:/Miniconda/envs/dlm/python.exe" ]; then
  PYTHON_CMD="D:/Miniconda/envs/dlm/python.exe"
elif [ -x "/d/Miniconda/envs/dlm/python.exe" ]; then
  PYTHON_CMD="/d/Miniconda/envs/dlm/python.exe"
else
  PYTHON_CMD="python"
fi

$PYTHON_CMD -m phasebatch batch \
  --inputs benchmarks/tiny/*.c \
  --out outputs/mvp_run \
  --passes configs/core_passes.yaml \
  --jobs 8 \
  --timeout 10 \
  --max-pairs 300
