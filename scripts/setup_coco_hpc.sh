#!/usr/bin/env bash
# Download COCO on the HPC GPU cluster (conduit). Do not run on your local machine.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COCO_ROOT="${COCO_ROOT:-$HOME/data/coco}"
SPLIT="${1:-val2017}"

echo "Project:  $PROJECT_ROOT"
echo "COCO root: $COCO_ROOT"
echo "Split:    $SPLIT"
echo ""

python "$PROJECT_ROOT/scripts/download_coco.py" --split "$SPLIT" --root "$COCO_ROOT"

echo ""
echo "COCO is ready. Add this to your ~/.bashrc on the cluster:"
echo "  export COCO_ROOT=$COCO_ROOT"
