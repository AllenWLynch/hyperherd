#!/bin/bash
# Launcher for MNIST training example.
# Adapt this for your cluster: container, conda, modules, etc.
set -euo pipefail

OVERRIDES="$1"

echo "=== HyperWhip MNIST Launcher ==="
echo "  HYPERWHIP_TRIAL_ID:          ${HYPERWHIP_TRIAL_ID:-unset}"
echo "  HYPERWHIP_EXPERIMENT_NAME:   ${HYPERWHIP_EXPERIMENT_NAME:-unset}"
echo ""

# --- Adapt one of these for your environment ---

# Option A: Direct execution (if pytorch-lightning is in your PATH)
cd "$(dirname "$0")"
python train.py $OVERRIDES

# Option B: Conda environment
# source /opt/conda/etc/profile.d/conda.sh
# conda activate pytorch
# cd "$(dirname "$0")"
# python train.py $OVERRIDES

# Option C: Apptainer container
# CONTAINER="/shared/containers/pytorch.sif"
# apptainer exec --nv --bind "/scratch:/scratch" "$CONTAINER" \
#     python "$(dirname "$0")/train.py" $OVERRIDES
