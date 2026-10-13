#!/bin/bash
# ============================================================================
# Render the backward-rollout diagnosis figures from the JSON + spectrum HDF5 produced by
# backward_rollout_fcn3.sbatch (plotter: analysis/plot_backward_rollout.py).
#
# Figures -> <scores_dir>/
#   backward_channel_overview.png  ACC + RMSE/climatology heatmaps, channel x lead
#   backward_rmse_vs_lead.png      RMSE vs lead per dynamical channel + persistence/climatology ceilings
#   backward_acc_vs_lead.png       ACC vs lead
#   backward_tail_power_ratio.png  small-scale P_pred/P_truth vs lead (>1 rising = instability; <1 = blur)
#   backward_spectrum_<chan>.png   pred-vs-truth power spectra at several leads, per dynamical channel
#   backward_ensemble_skill.png / backward_rank_histogram.png   (ensemble runs)
#
# CPU-only (numpy/h5py/matplotlib; no GPU, no makani/torch import). Runs inside the
# container on a compute node.
#
# Usage (submit from the repo root):
#   sbatch analysis/plot_backward_rollout.sbatch <scores_dir> [<forward_spectrum.h5>]
#
# Args (optional):
#   $1 SCORES_DIR        default $MAKANI_DATA/runs/rewind_fcn3_ens_10yr_backward/fcn3_ens_10yr/scores
#   $2 FORWARD_SPECTRUM  optional forward spectrum.h5 to overlay (skipped if absent; none by default)
# ============================================================================
#SBATCH --job-name=makani-bwd-plot
#SBATCH --partition=cpu                # plotting is CPU-only; switch to b200-devel if cpu is busy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=makani-bwd-plot-%j.log
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,INVALID_DEPEND,TIME_LIMIT

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from your repo root}"
# site-specific paths (container image, data root): edit site.env once
source "$SLURM_SUBMIT_DIR/site.env"

IMG="${MAKANI_IMG:?set MAKANI_IMG in site.env}"
RUNS="${MAKANI_DATA:?set MAKANI_DATA in site.env}/runs"
SCORES_DIR="${1:-$RUNS/rewind_fcn3_ens_10yr_backward/fcn3_ens_10yr/scores}"
FORWARD_SPECTRUM="${2:-}"

test -f analysis/plot_backward_rollout.py || { echo "ERROR: analysis/plot_backward_rollout.py not in $(pwd)"; exit 1; }
test -f "$SCORES_DIR/backward_rollout.json" || { echo "ERROR: $SCORES_DIR/backward_rollout.json missing (run backward_rollout_fcn3.sbatch first)"; exit 1; }

# matplotlib needs a writable config/cache dir; use node-local tmpfs (/dev/shm).
export MPLCONFIGDIR="/dev/shm/mplconfig_${SLURM_JOB_ID:-$$}"
mkdir -p "$MPLCONFIGDIR"
trap 'rm -rf "$MPLCONFIGDIR" 2>/dev/null || true' EXIT

# The plotter imports no makani/torch/CUDA, so no --nv and no single-proc distributed env needed.
# Wrap the call so the optional --forward_spectrum flag can be omitted cleanly under `set -u`.
run_plot() {
  apptainer exec \
    --env MPLCONFIGDIR="$MPLCONFIGDIR" \
    --bind "$SLURM_SUBMIT_DIR":/opt/makani \
    --bind "$MAKANI_DATA":"$MAKANI_DATA" \
    "$IMG" \
    python /opt/makani/analysis/plot_backward_rollout.py --scores_dir "$SCORES_DIR" "$@"
}

if [ -n "$FORWARD_SPECTRUM" ] && [ -f "$FORWARD_SPECTRUM" ]; then
  echo "overlaying forward spectrum: $FORWARD_SPECTRUM"
  run_plot --forward_spectrum "$FORWARD_SPECTRUM"
else
  [ -n "$FORWARD_SPECTRUM" ] && echo "NOTE: forward spectrum not found ($FORWARD_SPECTRUM); plotting backward only."
  run_plot
fi

echo "PNG outputs:"; ls -lah "$SCORES_DIR"/backward_*.png