#!/bin/bash
# ============================================================================
# Multistep BACKWARD rollout for the FCN3 targets (config/rewind_fcn3.yaml), with the
# ensemble metrics (CRPS / spread / SSR / rank histograms). The Python
# (analysis/backward_rollout.py) is architecture-agnostic; --ensemble_size 1 is the
# deterministic harness.
#
# SINGLE-GPU inference, E members folded into the batch dim. E=4 is the MEASURED ceiling
# on a 178-GiB GPU: FCN3's DISCO convolutions allocate ~13.37 GiB per folded sample in ONE
# fp32 buffer, so E=16 asks for ~214 GiB and OOMs at the first forward. For a large
# ensemble run independent seeded tasks and pool them (backward_rollout_ensemble.sbatch).
#
# Outputs -> $MAKANI_DATA/runs/<CONFIG_TARGET>/<RUN_NUM>/scores/
#            { backward_spectrum.h5, backward_rollout.json, backward_fields.h5 }
#   backward_rollout.json: rmse/acc/tail on the ensemble MEAN (schema-compatible with a
#   deterministic run -> compare_backward_rollouts.py overlays both), plus per-channel
#   crps/spread/ssr/rank_histogram/tail_power_ratio_members vs lead.
#   backward_spectrum.h5 columns: [truth, ensemble-mean field, member 1..E].
#
# Usage (submit from the repo root):
#   sbatch analysis/backward_rollout_fcn3.sbatch rewind_fcn3_ens_10yr_backward fcn3_ens_10yr
#   # smoke rollouts (tiny: short, few anchors — plumbing, not science):
#   sbatch analysis/backward_rollout_fcn3.sbatch rewind_fcn3_ens_smoke ens_smoke1 8 4 2 0
#   sbatch analysis/backward_rollout_fcn3.sbatch rewind_fcn3_det_smoke det_smoke1 8 4 1 0
#
# Args (all optional):
#   $1 CONFIG_TARGET        default rewind_fcn3_ens_10yr_backward
#   $2 RUN_NUM              default fcn3_ens_10yr
#   $3 NUM_STEPS (R)        default 40   (40 * 6 h = 10 days into the past)
#   $4 NUM_ANCHORS          default 64   (initial conditions spread over the validation year)
#   $5 ENSEMBLE_SIZE (E)    default 4    (the DISCO ceiling, NOT the training ensemble_size; 1 = deterministic)
#   $6 SAVE_FIELDS_ANCHORS  default 1    (raw-field dump for videos; 0 = off)
#   $7 SAVE_FIELDS_CHANNELS default dynamical  ('all' is ~25 GB/anchor at 72 ch)
#   $8+                     forwarded verbatim to backward_rollout.py — e.g.
#                           --fields_member 0 (dump a member SAMPLE instead of the mean),
#                           --crps_type cdf, --anchor_date 2019-09-01 (case study)
# ============================================================================
#SBATCH --job-name=makani-bwd-rollout-fcn3
#SBATCH --partition=b200-devel        # 1-GPU inference; separate pool from the b200-batch training
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=04:00:00               # E x the deterministic cost; devel cap is 4 h — trim anchors/steps if it times out
#SBATCH --output=makani-bwd-rollout-fcn3-%j.log
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,INVALID_DEPEND,TIME_LIMIT

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from your repo root}"
# site-specific paths (container image, data root): edit site.env once
source "$SLURM_SUBMIT_DIR/site.env"

IMG="${MAKANI_IMG:?set MAKANI_IMG in site.env}"
CONFIG_TARGET="${1:-rewind_fcn3_ens_10yr_backward}"
RUN_NUM="${2:-fcn3_ens_10yr}"
NUM_STEPS="${3:-40}"
NUM_ANCHORS="${4:-64}"
ENSEMBLE_SIZE="${5:-4}"
SAVE_FIELDS_ANCHORS="${6:-1}"
SAVE_FIELDS_CHANNELS="${7:-dynamical}"

# All seven positionals are required before any forwarded flag. Omitting one silently shifts a
# flag into a positional slot -- the flag is swallowed and its VALUE reaches argparse bare, which
# fails with an opaque "unrecognized arguments: <value>" only after the GPU is allocated and the
# checkpoint listed. Catch it here instead.
for _slot in "$NUM_STEPS" "$NUM_ANCHORS" "$ENSEMBLE_SIZE" "$SAVE_FIELDS_ANCHORS" "$SAVE_FIELDS_CHANNELS"; do
  case "$_slot" in
    -*) echo "ERROR: '$_slot' appeared in a POSITIONAL slot. All 7 positionals are required before"
        echo "       any forwarded flag:  CONFIG RUN_NUM NUM_STEPS NUM_ANCHORS E SAVE_ANCHORS SAVE_CHANNELS [flags...]"
        echo "       e.g. sbatch $0 rewind_fcn3_ens_10yr_backward fcn3_ens_10yr 40 8 4 0 dynamical --amp_mode none --tf32 off"
        exit 1 ;;
  esac
done
# anything after the 7 positionals is forwarded verbatim to backward_rollout.py
_ALL_ARGS="$*"
shift $(( $# < 7 ? $# : 7 ))

# ---------------------------------------------------------------------------
# Run provenance. Two jobs of this script differ only in FORWARDED flags (e.g. the
# --tf32/--amp_mode precision arms), and the log name carries only %j -- so a day later
# there is no way to tell which log belongs to which invocation. The --mail-type directives
# above cover job lifecycle notification, but mail
# never carries the forwarded arguments, which are the only thing distinguishing two otherwise
# identical runs. So also stamp the invocation into the log AND append one line to a central
# manifest: job id -> arguments -> output files stays recoverable indefinitely and is greppable.
# ---------------------------------------------------------------------------
MANIFEST="$MAKANI_DATA/runs/job_manifest.log"
echo "=== RUN PROVENANCE ==="
echo "  job id     : ${SLURM_JOB_ID:-<none>}   node: $(hostname)"
echo "  submitted  : $(date -u '+%Y-%m-%dT%H:%M:%SZ') UTC"
echo "  script     : $0"
echo "  arguments  : $_ALL_ARGS"
echo "  git hash   : $(git -C "$SLURM_SUBMIT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  log file   : $SLURM_SUBMIT_DIR/makani-bwd-rollout-fcn3-${SLURM_JOB_ID:-unknown}.log"
mkdir -p "$(dirname "$MANIFEST")" 2>/dev/null || true
printf '%s\tjob=%s\tnode=%s\tgit=%s\tscript=%s\targs=%s\n' \
  "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${SLURM_JOB_ID:-unknown}" "$(hostname)" \
  "$(git -C "$SLURM_SUBMIT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)" \
  "$(basename "$0")" "$_ALL_ARGS" >> "$MANIFEST" 2>/dev/null || true

RUNS="${MAKANI_DATA:?set MAKANI_DATA in site.env}/runs"
CKPT_DIR="$RUNS/$CONFIG_TARGET/$RUN_NUM/training_checkpoints"

test -f config/rewind_fcn3.yaml || { echo "ERROR: config/rewind_fcn3.yaml not in $(pwd)"; exit 1; }
test -f analysis/backward_rollout.py || { echo "ERROR: analysis/backward_rollout.py not in $(pwd)"; exit 1; }
test -f analysis/verify_direction.py || { echo "ERROR: analysis/verify_direction.py not in $(pwd) (imported by backward_rollout.py)"; exit 1; }

if [ ! -d "$CKPT_DIR" ]; then
  echo "ERROR: checkpoint dir missing: $CKPT_DIR"
  echo "       The run $CONFIG_TARGET/$RUN_NUM has not checkpointed yet (>=1 completed epoch needed),"
  echo "       or the run_num is wrong. Check: ls $RUNS/$CONFIG_TARGET/"
  exit 1
fi
echo "Checkpoint dir contents:"; ls -lah "$CKPT_DIR"
echo "Outputs -> $RUNS/$CONFIG_TARGET/$RUN_NUM/scores/{backward_spectrum.h5,backward_rollout.json,backward_fields.h5}"
echo "Ensemble: E=$ENSEMBLE_SIZE (1 = deterministic path, ensemble metrics off)"

# ---- GPU/CUDA preflight (fail fast + legibly on a bad node) -----------------
# Fail fast on a node where CUDA is unavailable inside the container or a GPU is already occupied.
echo "=== GPU/CUDA preflight on $(hostname) ==="
apptainer exec --nv \
  --bind "$MAKANI_DATA":"$MAKANI_DATA" \
  "$IMG" \
  python -c "
import socket, torch
host = socket.gethostname()
if not torch.cuda.is_available():
    raise SystemExit(f'PREFLIGHT FAIL: CUDA unavailable on {host} — likely a node driver/runtime '
                     f'mismatch (cudaGetDeviceCount errored; e.g. CUDA 803 during cluster maintenance). '
                     f'Resubmit to land on a healthy node, or exclude this one: sbatch --exclude={host} ...')
dc = torch.cuda.device_count()
GiB = 1024**3
free, total = torch.cuda.mem_get_info(0)
used = (total - free) / GiB
print(f'  {host}: {dc} GPU(s) visible; cuda:0 free={free/GiB:.1f}/{total/GiB:.1f} GiB (used {used:.1f})')
if used > 2.0:
    raise SystemExit(f'PREFLIGHT FAIL: cuda:0 on {host} already has {used:.1f} GiB used before we start '
                     f'— orphan/contention; resubmit or sbatch --exclude={host} ...')
print('  PREFLIGHT OK')
" || { echo 'PREFLIGHT FAIL: aborting before the rollout (see message above)'; exit 1; }

# Single-process (1-GPU) distributed env: physicsnemo's DistributedManager reads RANK first, so
# setting it keeps it off the SLURM branch (which needs srun-launched variables).
apptainer exec --nv \
  --env RANK=0 --env WORLD_SIZE=1 --env LOCAL_RANK=0 \
  --env MASTER_ADDR=127.0.0.1 --env MASTER_PORT=29500 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env WANDB_MODE=disabled \
  --bind "$SLURM_SUBMIT_DIR":/opt/makani \
  --bind "$MAKANI_DATA":"$MAKANI_DATA" \
  "$IMG" \
  python /opt/makani/analysis/backward_rollout.py \
    --yaml_config=/opt/makani/config/rewind_fcn3.yaml \
    --config="$CONFIG_TARGET" \
    --run_num="$RUN_NUM" \
    --checkpoint_select=best \
    --num_steps="$NUM_STEPS" \
    --num_anchors="$NUM_ANCHORS" \
    --ensemble_size="$ENSEMBLE_SIZE" \
    --save_fields_anchors="$SAVE_FIELDS_ANCHORS" \
    --save_fields_channels="$SAVE_FIELDS_CHANNELS" \
    --amp_mode=bf16 "$@"

echo "=== done. plot with:  sbatch analysis/plot_backward_rollout.sbatch $RUNS/$CONFIG_TARGET/$RUN_NUM/scores ==="
echo "===       video with: sbatch analysis/make_backward_video.sbatch $RUNS/$CONFIG_TARGET/$RUN_NUM/scores/backward_fields.h5 <channel|all> ==="
