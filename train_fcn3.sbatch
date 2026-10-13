#!/bin/bash
# ============================================================================
# Single-node DDP training for the backward FCN3 targets in config/rewind_fcn3.yaml.
#
# Launch is via torchrun (one task; torchrun spawns the per-GPU workers). The ENSEMBLE
# targets must run
#   python -m makani.ensemble        (EnsembleTrainer: wires the ["ensemble","batch"]
#                                     data-parallel split + local_ensemble_size)
# while a DETERMINISTIC target runs
#   python -m makani.train
# The module is picked from the target name: targets containing "_ens" run
# makani.ensemble, everything else makani.train. Override explicitly with
#   TRAIN_MODULE=makani.ensemble sbatch ...
#
# Sequence we used (2-GPU development partition for the smokes, 8 GPUs for the run):
#   sbatch --partition=<devel> --gres=gpu:2 --time=02:00:00 train_fcn3.sbatch rewind_fcn3_det_smoke det_smoke1
#   sbatch --partition=<devel> --gres=gpu:2 --time=02:00:00 train_fcn3.sbatch rewind_fcn3_ens_smoke ens_smoke1
#   sbatch train_fcn3.sbatch rewind_fcn3_ens_10yr_backward fcn3_ens_10yr --ensemble_parallel_size=2 --checkpointing_level=3
#
# BATCH DIVISIBILITY: makani requires global batch_size % (#batch-parallel GPUs) == 0.
# The smoke targets carry global batch 4 and 2 (run them on 2 GPUs); the 10yr target
# carries global batch 4 with --ensemble_parallel_size=2 on 8 GPUs. With ensemble_size E
# folded into the batch, per-GPU forward cost/memory ~ (batch/NGPU) * E samples.
#
# RESUME across the scheduler's time limit: re-submit the SAME (target, run_num); makani
# auto-detects the checkpoint. Chain with --dependency=afterany:<JOBID>. The trainer
# persists best_valid_loss and the early-stopping / divergence counters in the checkpoint.
#
# Run: cd <repo root>; sbatch train_fcn3.sbatch [config_target] [run_num] [extra makani args...]
# Extra args are forwarded verbatim (e.g. --ensemble_parallel_size=2, --checkpointing_level=3).
# Partition names below are our cluster's; override with sbatch -p / --gres.
# ============================================================================
#SBATCH --job-name=makani-train-fcn3
#SBATCH --partition=b200-batch       # override to b200-devel (+ --gres=gpu:2) for the smokes
#SBATCH --nodes=1
#SBATCH --ntasks=1                    # ONE task; torchrun spawns the per-GPU workers itself
#SBATCH --gres=gpu:8                  # 8-way DDP on one node (override --gres=gpu:2 on devel)
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=1-00:00:00            # 1-day limit -> resume chain
#SBATCH --output=makani-train-fcn3-%j.log
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,INVALID_DEPEND,TIME_LIMIT

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from your repo root}"
# site-specific paths (container image, data root): edit site.env once
source "$SLURM_SUBMIT_DIR/site.env"

IMG="${MAKANI_IMG:?set MAKANI_IMG in site.env}"
DATA="${MAKANI_DATA:?set MAKANI_DATA in site.env}"
META=/opt/makani/data_process/examples/metadata_fcn3_72.json   # in-container path
INV="$DATA/invariants"

# positional args: 1=config target, 2=run_num; anything after is forwarded to the trainer
CONFIG_TARGET="${1:-rewind_fcn3_ens_smoke}"
RUN_NUM="${2:-run_${CONFIG_TARGET}}"
EXTRA_ARGS=("${@:3}")

# module selection: ensemble targets (name contains "_ens") -> makani.ensemble, else makani.train.
# Explicit override: TRAIN_MODULE=makani.ensemble sbatch ...
if [ -z "${TRAIN_MODULE:-}" ]; then
  case "$CONFIG_TARGET" in
    *_ens*) TRAIN_MODULE="makani.ensemble" ;;
    *)      TRAIN_MODULE="makani.train" ;;
  esac
fi
echo "=== trainer module: $TRAIN_MODULE (target=$CONFIG_TARGET) ==="

# NGPU from the actual allocation (2 on devel, 8 on batch)
NGPU="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
[ "${NGPU:-0}" -ge 1 ] || { echo "ERROR: no GPUs visible (NGPU=$NGPU) — request --gres=gpu:N"; exit 1; }

# ---- prerequisite checks (fail fast, before any GPU time) -------------------
test -f config/rewind_fcn3.yaml || { echo "ERROR: config/rewind_fcn3.yaml not in $(pwd)"; exit 1; }
test -f data_process/examples/metadata_fcn3_72.json || { echo "ERROR: metadata_fcn3_72.json missing"; exit 1; }
test -f "$IMG" || { echo "ERROR: image $IMG missing"; exit 1; }

# stats (all 7 .npy the config points at; tendency files carry the _dt1 suffix)
for s in mins maxs time_means global_means global_stds time_diff_means_dt1 time_diff_stds_dt1; do
  test -f "$DATA/stats/$s.npy" || { echo "ERROR: stats $DATA/stats/$s.npy missing — run stats_era5.sbatch"; exit 1; }
done

# data present. Every target validates on valid/ (2019); the smokes also TRAIN on it,
# and the 10yr target trains on the train_10yr symlink dir — check what this target will read.
NVALID=$(ls -1 "$DATA"/valid/*.h5 2>/dev/null | wc -l) || NVALID=0
[ "$NVALID" -ge 1 ] || { echo "ERROR: no valid .h5 in $DATA/valid — run convert_era5.sbatch"; exit 1; }
case "$CONFIG_TARGET" in
  *_10yr*)
    N10=$(ls -1 "$DATA"/train_10yr/*.h5 2>/dev/null | wc -l) || N10=0
    [ "$N10" -ge 1 ] || { echo "ERROR: no train .h5 in $DATA/train_10yr (a directory of year files or symlinks) — create it"; exit 1; }
    echo "=== data: $N10 train_10yr year(s), $NVALID valid year(s) ===" ;;
  *)
    echo "=== data: (smoke trains on valid/) $NVALID valid year(s) ===" ;;
esac
echo "=== target=$CONFIG_TARGET run=$RUN_NUM NGPU=$NGPU ==="

# ---- rollout-CI hook (ROLLOUT_CI=0 to disable) -------------------------------
# Submitted at LINK START with a dependency on THIS job, because chain links normally
# end by TIMEOUT-kill — a hook at the end of this script would never run. When the link
# dies (any way: afterany), the CI job snapshots the latest checkpoint's backward rollout
# on b200-devel (1 GPU, separate pool — never competes with the chain) and updates the
# skill-vs-epoch trend in <run>/scores/ci/. Real-training targets only (not smokes/probes).
if [ "${ROLLOUT_CI:-1}" != "0" ]; then
  case "$CONFIG_TARGET" in
    *smoke*) : ;;  # smokes: no CI
    *_ens*)
      CI_JID=$(sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" \
               analysis/rollout_ci_fcn3.sbatch "$CONFIG_TARGET" "$RUN_NUM" 2>/dev/null) \
        && echo "=== rollout-CI armed: job $CI_JID fires when this link ends (ROLLOUT_CI=0 to disable)" \
        || echo "WARNING: rollout-CI submission failed (sbatch from compute node?) — run it manually"
      ;;
  esac
fi

# invariant geo-sanity confirmation (offline — a flipped/NaN static cue
# is catastrophic and silent, so fail in seconds rather than after hours of GPU)
echo "=== invariant geo-sanity (offline) ==="
apptainer exec \
  --bind "$SLURM_SUBMIT_DIR":/opt/makani \
  --bind "$MAKANI_DATA":"$MAKANI_DATA" \
  "$IMG" \
  python /opt/makani/data_process/make_invariants.py --check_existing --metadata_file "$META" --output_dir "$INV"

# ---- GPU binding/occupancy preflight ---------------------------------------
# Fail in seconds, not hours, if torch sees the wrong GPU count or a GPU is already
# occupied by a stray process.
gpu_state_dump() {
  echo "=== GPU state on $(hostname) at preflight failure (capture for a node-hygiene bug report) ==="
  nvidia-smi || true
  echo "--- owners of processes currently on this node's GPUs ---"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | while IFS=, read -r pid _; do
    pid=$(echo "$pid" | tr -dc '0-9')
    [ -n "$pid" ] && { ps -o user=,pid=,etime=,cmd= -p "$pid" 2>/dev/null || echo "  pid $pid: owner not resolvable here"; }
  done
}

echo "=== GPU binding/occupancy preflight (expect $NGPU empty GPUs) ==="
apptainer exec --nv \
  --env EXPECT_NGPU="$NGPU" \
  --bind "$MAKANI_DATA":"$MAKANI_DATA" \
  "$IMG" \
  python -c "
import os, socket, torch
expect = int(os.environ['EXPECT_NGPU'])
dc = torch.cuda.device_count()
host = socket.gethostname()
GiB = 1024**3
bad = False
if dc != expect:
    print(f'  ERROR: torch sees {dc} GPU(s) on {host} but the allocation is {expect} (CUDA_VISIBLE_DEVICES={os.environ.get(\"CUDA_VISIBLE_DEVICES\", \"<unset>\")}) — DDP would mis-size'); bad = True
for i in range(dc):
    free, total = torch.cuda.mem_get_info(i)
    used = (total - free) / GiB
    print(f'  {host} cuda:{i}  free={free/GiB:.1f}/{total/GiB:.1f} GiB (used {used:.1f})')
    if used > 2.0:
        print(f'    ERROR: cuda:{i} on {host} already has {used:.1f} GiB used before we allocate — orphan/contention (nvidia-smi, then kill the stray PID)'); bad = True
if bad:
    raise SystemExit('PREFLIGHT FAIL: fix GPU visibility/occupancy before training (see errors above)')
print(f'  PREFLIGHT OK: {dc} visible GPU(s), all empty')
" || { gpu_state_dump; echo "PREFLIGHT FAIL: aborting before training (GPU state captured above)"; exit 1; }

# ---- LAUNCH: single-node DDP via torchrun -----------------------------------
# torchrun sets RANK/LOCAL_RANK/WORLD_SIZE/MASTER_* per worker, which physicsnemo's
# DistributedManager reads first (ahead of SLURM_PROCID, which a plain sbatch leaks as 0
# for every process). Do NOT pass --env RANK/WORLD_SIZE here — torchrun owns those.
# PYTORCH_CUDA_ALLOC_CONF is set only via
# --env below — the in-container env is the single source of truth.

# torchrun temp/log dir on tmpfs, not NFS: torch-elastic rmtree's it at teardown while worker
# file descriptors are still open, which fails on NFS (nodes without local disk)
SHM_LOGDIR="/dev/shm/makani_fcn3_${SLURM_JOB_ID:-$$}"
trap 'rm -rf "$SHM_LOGDIR" 2>/dev/null || true' EXIT

# wandb: offline by default (config sets log_to_wandb: True); override e.g. WANDB_MODE=disabled.
# A run that BEGAN wandb-off cannot be resumed wandb-on (wandb.init resume="must") — fresh run_num.
WANDB_MODE="${WANDB_MODE:-offline}"

echo "=== launching torchrun: $NGPU-way DDP, module=$TRAIN_MODULE target=$CONFIG_TARGET run=$RUN_NUM (extra: ${EXTRA_ARGS[*]:-none}) ==="
apptainer exec --nv \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env WANDB_MODE="$WANDB_MODE" \
  --env OMP_NUM_THREADS=4 \
  --env TMPDIR=/dev/shm --env TEMP=/dev/shm --env TMP=/dev/shm \
  --bind "$SLURM_SUBMIT_DIR":/opt/makani \
  --bind "$MAKANI_DATA":"$MAKANI_DATA" \
  "$IMG" \
  torchrun --standalone --nnodes=1 --nproc_per_node="$NGPU" --log-dir "$SHM_LOGDIR" \
    -m "$TRAIN_MODULE" \
    --yaml_config=/opt/makani/config/rewind_fcn3.yaml \
    --config="$CONFIG_TARGET" \
    --run_num="$RUN_NUM" \
    --amp_mode=bf16 \
    "${EXTRA_ARGS[@]}"

echo "=== done. checkpoints + logs under $DATA/runs/$CONFIG_TARGET/$RUN_NUM ==="
echo "    resume/continue (same args) after a timeout: sbatch --dependency=afterany:\$SLURM_JOB_ID train_fcn3.sbatch $CONFIG_TARGET $RUN_NUM ${EXTRA_ARGS[*]:-}"
