#!/bin/bash
# ============================================================================
# Render backward-rollout VIDEOS from the raw-field HDF5 dumped by
# backward_rollout.py --save_fields_anchors (on by default in backward_rollout_fcn3.sbatch).
# One three-panel animation per channel — ERA5 truth | backward prediction | error —
# stepping 6 h further into the past per frame. Any channel present in the dump can be
# rendered without another GPU pass.
#
# Output -> alongside the fields file (or --out_dir via EXTRA args):
#   backward_rollout_<chan>_anchor<A>.mp4   (.gif when the container lacks ffmpeg)
#
# CPU-only (numpy/h5py/matplotlib; no GPU, no makani/torch import). Runs inside the
# container on a compute node.
#
# Usage (submit from the repo root):
#   sbatch analysis/make_backward_video.sbatch <fields_file> [<channel>] [extra args...]
#   sbatch analysis/make_backward_video.sbatch $MAKANI_DATA/runs/<target>/<run>/scores/backward_fields.h5 z500
#   # every dumped channel: pass 'all'; extra args go to make_backward_video.py
#   # (e.g. --anchor 1 --fps 6 --reverse)
#
# Args:
#   $1 FIELDS_FILE   required — backward_fields.h5 from the rollout job
#   $2 CHANNEL       default z500 ('all' = every channel in the dump)
#   $3+              passed through to make_backward_video.py
# ============================================================================
#SBATCH --job-name=makani-bwd-video
#SBATCH --partition=cpu                # rendering is CPU-only; switch to b200-devel if cpu is busy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00               # 'all' on the 72-ch dump renders 72 animations; single channel ~ minutes
#SBATCH --output=makani-bwd-video-%j.log
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,INVALID_DEPEND,TIME_LIMIT

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from your repo root}"
# site-specific paths (container image, data root): edit site.env once
source "$SLURM_SUBMIT_DIR/site.env"

IMG="${MAKANI_IMG:?set MAKANI_IMG in site.env}"
FIELDS_FILE="${1:?usage: sbatch analysis/make_backward_video.sbatch <fields_file> [<channel>] [extra args...]}"
CHANNEL="${2:-z500}"
shift $(( $# >= 2 ? 2 : 1 ))

test -f analysis/make_backward_video.py || { echo "ERROR: analysis/make_backward_video.py not in $(pwd)"; exit 1; }
test -f analysis/backward_fields_io.py || { echo "ERROR: analysis/backward_fields_io.py not in $(pwd) (imported by the renderer)"; exit 1; }
test -f "$FIELDS_FILE" || { echo "ERROR: fields file missing: $FIELDS_FILE"; echo "       (re-run backward_rollout_fcn3.sbatch — it dumps backward_fields.h5 by default now)"; exit 1; }

# matplotlib needs a writable config/cache dir; use node-local tmpfs (/dev/shm).
export MPLCONFIGDIR="/dev/shm/mplconfig_${SLURM_JOB_ID:-$$}"
mkdir -p "$MPLCONFIGDIR"
trap 'rm -rf "$MPLCONFIGDIR" 2>/dev/null || true' EXIT

# No makani/torch/CUDA import, so no --nv and no single-proc distributed env needed.
apptainer exec \
  --env MPLCONFIGDIR="$MPLCONFIGDIR" \
  --bind "$SLURM_SUBMIT_DIR":/opt/makani \
  --bind "$MAKANI_DATA":"$MAKANI_DATA" \
  "$IMG" \
  python /opt/makani/analysis/make_backward_video.py \
    --fields_file "$FIELDS_FILE" --channel "$CHANNEL" "$@"

# NB: with a pass-through --out_dir the videos land there instead and this listing is empty
echo "video outputs (in $(dirname "$FIELDS_FILE") unless --out_dir was passed):"
ls -lah "$(dirname "$FIELDS_FILE")"/backward_rollout_*.{mp4,gif} 2>/dev/null || true
