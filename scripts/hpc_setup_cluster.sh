#!/usr/bin/env bash
# =============================================================================
# HPC cluster setup for conduit (hlcv_team019@conduit.hpc.uni-saarland.de)
#
# Installs conda environment + pip dependencies, sets paths.
# Run ON THE CLUSTER only — not on your local Windows machine.
#
# Usage (after cloning/copying project to ~/super_resolution):
#   cd ~/super_resolution
#   bash scripts/hpc_setup_cluster.sh
#
# Options:
#   --skip-env          Skip conda environment creation
#   --skip-data         (deprecated) no-op; kept for compatibility
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SKIP_ENV=0
SKIP_DATA=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-env) SKIP_ENV=1; shift ;;
    --skip-data) SKIP_DATA=1; shift ;; # kept for compatibility
    -h|--help)
      sed -n '1,18p' "$0"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

DATA_ROOT="${DATA_ROOT:-$HOME/data}"
export PROJECT_ROOT
export DATA_ROOT
export COCO_ROOT="$DATA_ROOT/coco"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo "=============================================="
echo " HPC cluster setup"
echo "=============================================="
echo "Project:   $PROJECT_ROOT"
echo "Data root: $DATA_ROOT"
echo "COCO_ROOT: $COCO_ROOT"
echo ""

if [[ "$SKIP_ENV" -eq 0 ]]; then
  echo ">>> Step 1: Conda environment"
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  else
    echo "ERROR: conda not found. Run: source ~/miniconda3/etc/profile.d/conda.sh"
    exit 1
  fi

  if conda env list | awk '{print $1}' | grep -qx "super_resolution"; then
    echo "Updating existing env: super_resolution"
    conda env update -f "$PROJECT_ROOT/environment.yml" --prune -y
  else
    echo "Creating env: super_resolution (this may take 10–20 min)"
    conda env create -f "$PROJECT_ROOT/environment.yml" -y || {
      echo ""
      echo "If pytorch-cuda=12.1 failed, edit environment.yml to use pytorch-cuda=11.8 and retry."
      exit 1
    }
  fi

  conda activate super_resolution
  pip install -q -r "$PROJECT_ROOT/requirements.txt"

  python -c "
import torch, cv2, ultralytics, transformers
print('  torch', torch.__version__, '| cuda:', torch.cuda.is_available())
print('  opencv', cv2.__version__)
print('  ultralytics OK | transformers OK')
"
  echo ""
else
  echo ">>> Step 1: Skipped (--skip-env)"
  source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate super_resolution 2>/dev/null || true
fi

echo ">>> Step 2: Environment variables"
MARKER="# super_resolution project paths"
if ! grep -q "$MARKER" "$HOME/.bashrc" 2>/dev/null; then
  cat >> "$HOME/.bashrc" << EOF

$MARKER
export PROJECT_ROOT="$PROJECT_ROOT"
export DATA_ROOT="$DATA_ROOT"
export COCO_ROOT="$COCO_ROOT"
export PYTHONPATH="\$PROJECT_ROOT:\$PYTHONPATH"
EOF
  echo "Added paths to ~/.bashrc"
else
  echo "Paths already in ~/.bashrc"
fi
echo ""

echo ">>> Step 3: Dataset"
echo "COCO is expected to already exist on the cluster."
echo "Set COCO_ROOT to the existing path (e.g. /scratch/teaching/hlcv/hlcv_team019/coco)."
echo ""

echo "=============================================="
echo " Setup complete"
echo "=============================================="
echo ""
echo "Every new SSH session:"
echo "  source ~/miniconda3/etc/profile.d/conda.sh"
echo "  conda activate super_resolution"
echo "  cd $PROJECT_ROOT"
echo ""
echo "Dataset layout:"
echo "  $COCO_ROOT/val2017/       (images)"
echo "  $COCO_ROOT/annotations/  (ground truth JSONs)"
echo ""
echo "Quick tests:"
echo "  python scripts/load_yolo.py"
echo "  python -c \"import torch; print(torch.cuda.is_available())\""
echo ""
