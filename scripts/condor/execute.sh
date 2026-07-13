#!/usr/bin/env bash
# HTCondor wrapper (modeled after gpu_instructions/execute.sh)
set -euo pipefail

# Default locations; override by exporting env vars in your shell or in the .sub file.
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/super_resolution}"
DATA_ROOT="${DATA_ROOT:-/scratch/teaching/hlcv/hlcv_team019}"
COCO_ROOT="${COCO_ROOT:-$DATA_ROOT/coco}"
CONDA_PYTHON_BINARY_PATH="${CONDA_PYTHON_BINARY_PATH:-$HOME/miniconda3/envs/super_resolution/bin/python}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export DATA_ROOT
export COCO_ROOT

"$CONDA_PYTHON_BINARY_PATH" "$@"
