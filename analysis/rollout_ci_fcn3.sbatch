#!/bin/bash
# ============================================================================
# Rollout-CI: a CHEAP backward-rollout snapshot of a training checkpoint, archived per
# epoch, so multistep skill is tracked WHILE training runs: single-step convergence does
# not imply rollout stability, so rollout skill must be tracked during training, not
# discovered after it.
#
# Triggering: train_fcn3.sbatch submits this at LINK START with
# --dependency=afterany:<its own job id> (ROLLOUT_CI=1 default), so it fires when
# the link ends — INCLUDING timeout-kills, which is how chain links normally end.
# It lands on a 1-GPU development partition, a separate pool from the training chain.
#
# Which checkpoint: the file in slot ckpt_mp0_v0.tar of the run's checkpoint ring. The
# trainer rotates slots (checkpoint_num_versions), so with a ring deeper than 1 this is
# not necessarily the newest epoch; the snapshot is keyed by the epoch stored in the file.
#
# Race safety: the checkpoint tar is COPIED to node-local /dev/shm before eval.
# Skip-if-done: snapshots are keyed by the checkpoint's stored epoch counter
# (scores/ci/epoch_NNNN/); re-runs against an unchanged checkpoint exit early.
#
# Cost: R=20 (-120 h) x 12 anchors x E=4 folded members ~ 15-25 min on one B200
# (E=4 keeps the decoder DISCO buffer ~80 GiB; the CI's job is the TREND, not the
# decision-grade eval).
#
# Usage (normally submitted by train_fcn3.sbatch; manual run is fine too):
#   sbatch analysis/rollout_ci_fcn3.sbatch [config_target] [run_num] [steps] [anchors] [E]
# ============================================================================
#SBATCH --job-name=makani-rollout-ci
#SBATCH --partition=b200-devel
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --output=makani-rollout-ci-%j.log
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,INVALID_DEPEND,TIME_LIMIT

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from your repo root}"
# site-specific paths (container image, data root): edit site.env once
source "$SLURM_SUBMIT_DIR/site.env"

IMG="${MAKANI_IMG:?set MAKANI_IMG in site.env}"
CONFIG_TARGET="${1:-rewind_fcn3_ens_10yr_backward}"
RUN_NUM="${2:-fcn3_ens_10yr}"
NUM_STEPS="${3:-20}"
NUM_ANCHORS="${4:-12}"
ENSEMBLE_SIZE="${5:-4}"

RUNS="${MAKANI_DATA:?set MAKANI_DATA in site.env}/runs"
RUN_DIR="$RUNS/$CONFIG_TARGET/$RUN_NUM"
CKPT="$RUN_DIR/training_checkpoints/ckpt_mp0_v0.tar"
SCORES="$RUN_DIR/scores"

test -f config/rewind_fcn3.yaml || { echo "ERROR: config/rewind_fcn3.yaml not in $(pwd)"; exit 1; }
test -f "$CKPT" || { echo "ERROR: no checkpoint at $CKPT (no completed epoch yet?)"; exit 1; }

# ---- GPU/CUDA preflight (KEEP IN SYNC with backward_rollout_fcn3.sbatch) ----
apptainer exec --nv \
  --bind "$MAKANI_DATA":"$MAKANI_DATA" \
  "$IMG" \
  python -c "
import socket, torch
host = socket.gethostname()
if not torch.cuda.is_available():
    raise SystemExit(f'PREFLIGHT FAIL: CUDA unavailable on {host}; resubmit or sbatch --exclude={host}')
GiB = 1024**3
free, total = torch.cuda.mem_get_info(0)
used = (total - free) / GiB
if used > 2.0:
    raise SystemExit(f'PREFLIGHT FAIL: cuda:0 on {host} already has {used:.1f} GiB used; resubmit or exclude')
print(f'  PREFLIGHT OK on {host}: free={free/GiB:.1f}/{total/GiB:.1f} GiB')
" || { echo 'PREFLIGHT FAIL: aborting'; exit 1; }

# ---- copy the checkpoint to tmpfs + read its epoch counter -------------------
CI_TMP="/dev/shm/$USER/rollout_ci_$SLURM_JOB_ID"
mkdir -p "$CI_TMP/training_checkpoints"
trap 'rm -rf "$CI_TMP"' EXIT
cp "$CKPT" "$CI_TMP/training_checkpoints/"

EPOCH=$(apptainer exec \
  --bind /dev/shm:/dev/shm \
  "$IMG" \
  python -c "
import torch
ck = torch.load('$CI_TMP/training_checkpoints/ckpt_mp0_v0.tar', map_location='cpu', weights_only=False)
print(int(ck['epoch']))
")
CI_DIR="$SCORES/ci/epoch_$(printf '%04d' "$EPOCH")"
if [ -f "$CI_DIR/backward_rollout.json" ]; then
  echo "=== CI snapshot for epoch $EPOCH already exists ($CI_DIR) — nothing to do"
  exit 0
fi
mkdir -p "$CI_DIR"
echo "=== rollout-CI: $CONFIG_TARGET/$RUN_NUM epoch $EPOCH -> $CI_DIR (R=$NUM_STEPS A=$NUM_ANCHORS E=$ENSEMBLE_SIZE)"

# ---- the snapshot rollout (latest checkpoint, evaluated from the tmpfs copy) --
apptainer exec --nv \
  --env RANK=0 --env WORLD_SIZE=1 --env LOCAL_RANK=0 \
  --env MASTER_ADDR=127.0.0.1 --env MASTER_PORT=29500 \
  --env WANDB_MODE=disabled \
  --bind "$SLURM_SUBMIT_DIR":/opt/makani \
  --bind "$MAKANI_DATA":"$MAKANI_DATA" \
  --bind /dev/shm:/dev/shm \
  "$IMG" \
  python /opt/makani/analysis/backward_rollout.py \
    --yaml_config=/opt/makani/config/rewind_fcn3.yaml \
    --config="$CONFIG_TARGET" \
    --run_num="$RUN_NUM" \
    --checkpoint_path="$CI_TMP/training_checkpoints" \
    --checkpoint_select=latest \
    --num_steps="$NUM_STEPS" \
    --num_anchors="$NUM_ANCHORS" \
    --ensemble_size="$ENSEMBLE_SIZE" \
    --save_fields_anchors=0 \
    --output_json="$CI_DIR/backward_rollout.json" \
    --spectrum_file="ci/epoch_$(printf '%04d' "$EPOCH")/backward_spectrum.h5" \
    --amp_mode=bf16

# ---- refresh the trend CSV + verdict + figure ---------------------------------
apptainer exec \
  --bind "$SLURM_SUBMIT_DIR":/opt/makani \
  --bind "$MAKANI_DATA":"$MAKANI_DATA" \
  "$IMG" \
  python /opt/makani/analysis/rollout_ci_trend.py \
    --ci_dir "$SCORES/ci" \
    --out "$SCORES/ci/rollout_trend.png"

echo "=== rollout-CI done: epoch $EPOCH archived; trend at $SCORES/ci/rollout_trend.{csv,png}"
