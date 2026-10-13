#!/bin/bash
# ============================================================================
# Run the unit tests INSIDE the container on a COMPUTE node (apptainer should not run
# on a login node). A GPU is requested because the image bundles NVIDIA DALI and the
# multifiles=False dataloader variants need CUDA; the CPU-only tests run either way.
#
# The image installs makani with --no-deps and no [test] extra, so pytest/parameterized
# are absent. We make a --system-site-packages venv (sees the image's torch/makani/DALI,
# avoids PEP-668 "externally-managed" refusals) under $MAKANI_DATA and add the test deps;
# compute nodes need outbound internet for that first pip install.
#
# Usage (submit from the repo root):
#   sbatch analysis/run_tests.sbatch                                   # tests/test_dataloader.py -k HDF5
#   sbatch analysis/run_tests.sbatch tests/test_dataloader.py ""       # the WHOLE file
#   sbatch analysis/run_tests.sbatch tests ""                          # everything (slow)
#   sbatch analysis/run_tests.sbatch tests/test_dataloader.py reverse_time
#
# Args (optional):
#   $1 TARGET    pytest target            (default tests/test_dataloader.py)
#   $2 KFILTER   pytest -k expression     (default "HDF5"; pass "" to run the whole TARGET)
# Partition names are our cluster's; override with sbatch -p / --gres.
# ============================================================================
#SBATCH --job-name=makani-pytest
#SBATCH --partition=b200-devel
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=makani-pytest-%j.log
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,INVALID_DEPEND,TIME_LIMIT

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from your repo root}"
# site-specific paths (container image, data root): edit site.env once
source "$SLURM_SUBMIT_DIR/site.env"

IMG="${MAKANI_IMG:?set MAKANI_IMG in site.env}"
TARGET="${1:-tests/test_dataloader.py}"
KFILTER="${2-HDF5}"                       # note ${2-...}: an explicit "" disables the -k filter
VENV="${MAKANI_VENV:-${MAKANI_DATA:?set MAKANI_DATA in site.env}/.pytest_venv}"

test -e "$TARGET" || { echo "ERROR: test target '$TARGET' not found in $(pwd)"; exit 1; }

apptainer exec --nv \
  --env PIP_NO_CACHE_DIR=1 \
  --env VENV="$VENV" \
  --env TARGET="$TARGET" \
  --env KFILTER="$KFILTER" \
  --env WANDB_MODE=disabled \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --bind "$SLURM_SUBMIT_DIR":/opt/makani \
  --bind "$MAKANI_DATA":"$MAKANI_DATA" \
  "$IMG" bash -c '
    set -euo pipefail
    [ -x "$VENV/bin/python" ] || python -m venv --system-site-packages "$VENV"
    "$VENV/bin/pip" install --quiet --no-cache-dir "pytest>=6.0.0" parameterized
    cd /opt/makani
    echo "=== $("$VENV/bin/python" -m pytest --version 2>&1) | makani: $("$VENV/bin/python" -c "import makani, os; print(os.path.dirname(makani.__file__))") ==="
    if [ -n "$KFILTER" ]; then
      "$VENV/bin/python" -m pytest "$TARGET" -v -k "$KFILTER"
    else
      "$VENV/bin/python" -m pytest "$TARGET" -v
    fi
  '
