#!/usr/bin/env bash
# HTCondor wrapper (modeled after gpu_instructions/execute.sh)
set -euo pipefail

# Default locations; override by exporting env vars in your shell or in the .sub file.
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/super_resolution}"
DATA_ROOT="${DATA_ROOT:-/scratch/teaching/hlcv/hlcv_team019}"
COCO_ROOT="${COCO_ROOT:-$DATA_ROOT/coco}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-super_resolution}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export DATA_ROOT
export COCO_ROOT

if [[ ! -f "$CONDA_SH" ]]; then
  echo "ERROR: conda.sh not found at: $CONDA_SH"
  echo "Set CONDA_SH to your conda.sh path or install Miniconda under \$HOME/miniconda3."
  exit 1
fi

# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"

python "$@"
