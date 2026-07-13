#!/usr/bin/env bash
# HTCondor wrapper (modeled after gpu_instructions/execute.sh)
set -euo pipefail

# Default locations; override by exporting env vars in your shell or in the .sub file.
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/super_resolution}"
DATA_ROOT="${DATA_ROOT:-/scratch/teaching/hlcv/hlcv_team019}"
COCO_ROOT="${COCO_ROOT:-$DATA_ROOT/coco}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-super_resolution}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_PYTHON_BINARY_PATH="${CONDA_PYTHON_BINARY_PATH:-}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export DATA_ROOT
export COCO_ROOT

if [[ -n "$CONDA_PYTHON_BINARY_PATH" ]]; then
  exec "$CONDA_PYTHON_BINARY_PATH" "$@"
fi

if [[ -f "$CONDA_SH" ]]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
  conda activate "$CONDA_ENV_NAME"
  exec python "$@"
fi

echo "ERROR: No Python found."
echo "Either set CONDA_PYTHON_BINARY_PATH (recommended for HTCondor) or ensure conda.sh exists at: $CONDA_SH"
exit 1
