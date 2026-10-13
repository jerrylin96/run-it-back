#!/bin/bash
# ============================================================================
# LARGE-ENSEMBLE backward rollout for a single case-study date, as a SLURM JOB ARRAY.
#
# WHY AN ARRAY AND NOT ONE BIG JOB. backward_rollout.py folds the ensemble into the
# batch dim (`state.repeat(E,1,1,1)`), so one process holds E member states at once AND
# runs them through the model together. Two separate ceilings bite, the second much
# earlier than the first:
#   (1) resident state — at 72 channels / 0.25 deg / fp32 one member state is ~285 MB, so
#       E=1000 would need ~278 GiB.
#   (2) THE DISCO INTERMEDIATE, which is the binding one. FCN3's spherical convolutions
#       allocate a single fp32 buffer of ~13.37 GiB PER FOLDED SAMPLE, in ONE allocation,
#       so the ceiling on a 178-GiB GPU is 4 samples (~53 GiB). E=16 asks for ~214 GiB in
#       one go and OOMs immediately. Do not raise E without re-measuring.
# Running N INDEPENDENT tasks at the known-good E and pooling them needs no chunking or
# offload, and each task is short, so a wall-clock overrun costs one seed instead of the
# whole ensemble (the JSON and the HDF5 are finalized at the end of a run).
#
# WHAT MAKES THE POOL VALID: members differ ONLY through the input noise, and
# --noise_seed reseeds the preprocessor RNG, so tasks with distinct seeds draw
# independent members of the same distribution. Concatenating their dumps on the member
# axis is therefore exact. Each dump records its own noise_seed in the file attrs so this
# is checkable after the fact, not assumed.
#
# SEED 0 IS RESERVED / REJECTED: --noise_seed 0 means "do not reseed", so two tasks at
# seed 0 would emit IDENTICAL members. Submit arrays starting at 1.
#
# WHAT IS *NOT* POOLABLE: the per-task ensemble METRICS (CRPS, spread, SSR, rank
# histograms) describe E=4, not E=1000 — CRPS has a pairwise term and spread an (E-1)
# correction, so neither averages across tasks into its 1000-member value. The pooled
# product here is the FIELD DUMP; distributional metrics on the full pool are computed
# offline from the dumps (analysis/backward_fields_io.load_channel_members and the
# analysis/*_horizon.py tools). Per-member behavior pools exactly.
#
# Run (250 tasks x E=4 = 1000 members exactly):
#   cd <repo root>
#   sbatch --array=1-250%32 analysis/backward_rollout_ensemble.sbatch
#   sbatch --array=1-250%32 analysis/backward_rollout_ensemble.sbatch rewind_fcn3_ens_casestudy \
#          fcn3_ens_10yr 30 4 2021-06-28T12:00 t2m,z500
#
# Positionals (all optional, in order):
#   $1 CONFIG_TARGET  default rewind_fcn3_ens_casestudy   (its inf_data_path must hold the year)
#   $2 RUN_NUM        default fcn3_ens_10yr               (which trained run's checkpoint)
#   $3 NUM_STEPS (R)  default 30                          (-180 h)
#   $4 ENSEMBLE (E)   default 4                           (the DISCO ceiling -- see above)
#   $5 ANCHOR_DATE    default 2021-06-28T12:00            (PNW heat dome, 05:00 PDT; UTC ISO)
#   $6 CHANNELS       default the 32-channel union         (comma list, or all/dynamical; see below)
#   $7+               forwarded verbatim to the Python
# ============================================================================
#SBATCH --job-name=makani-bwd-ens
#SBATCH --partition=b200-batch        # array needs many GPUs; b200-devel caps at 2 total
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=01:00:00               # ~3 min of rollout at the defaults; the rest is model+data load
#SBATCH --output=makani-bwd-ens-%A_%a.log
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,INVALID_DEPEND,TIME_LIMIT,TIME_LIMIT_90
# Array mail is ARRAY-LEVEL, not per-task: slurm applies BEGIN/END/FAIL to the array as a
# whole unless ARRAY_TASKS is also given, so a 250-task array sends ~2 mails, not 500.
# Do NOT add ARRAY_TASKS -- that is what would generate one mail per task.
# TIME_LIMIT_90 warns at 90% of the wall clock, before the job is killed.
# Slurm's mail body carries no resource accounting; get that after the fact with
#   seff <jobid>            (per-array-task efficiency: CPU, memory, wall clock)
#   sacct -j <jobid> -X --format=JobID,State,Elapsed,MaxRSS,ReqTRES
# The script also prints its own elapsed time at the end of each task's log.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from your repo root}"
# site-specific paths (container image, data root): edit site.env once
source "$SLURM_SUBMIT_DIR/site.env"
_T0=$SECONDS

IMG="${MAKANI_IMG:?set MAKANI_IMG in site.env}"
RUNS="${MAKANI_DATA:?set MAKANI_DATA in site.env}/runs"

CONFIG_TARGET="${1:-rewind_fcn3_ens_casestudy}"
RUN_NUM="${2:-fcn3_ens_10yr}"
NUM_STEPS="${3:-30}"
ENSEMBLE_SIZE="${4:-4}"
# 2021-06-28T12:00Z = 05:00 PDT on the 28th, the requested rewind origin for the PNW heat
# dome. The TIME MATTERS: --anchor_date takes the frame AT OR JUST BEFORE the request, and a
# bare "2021-06-28" means 00:00 UTC -- which is 17:00 PDT on the 27th, a different frame and
# half a day earlier than intended. Data is 6-hourly (00/06/12/18 UTC), so 12:00 lands exactly
# on a frame with no rounding; an off-grid request (say 13:00) would silently snap back to 12:00.
ANCHOR_DATE="${5:-2021-06-28T12:00}"
# 32 channels = the UNION of two requirements, and it is the DEFAULT deliberately: passing a
# long channel string on the command line is a silent-failure trap, because an unset or empty
# shell variable expands to "" and `${6:-...}` treats empty exactly like unset -- so a lost
# variable falls back to this list with no error, after 250 tasks have run. Making the list we
# actually want the default means the fallback IS the right answer.
#   (a) cBottle-video's STORED_VARIABLES in the companion study (Lin et al. 2026, arXiv:2608.19008; github.com/jerrylin96/rewinding-the-extremes),
#       the head-to-head comparison set: u10m, v10m, t2m,
#       msl, tcwv, u200, v200, z300, u500, v500, t500, z500, u850, v850. Their `sst` is NOT
#       here and cannot be: our 72 channels are prognostic atmospheric state, and SST is a
#       prescribed boundary field (an input, never a target).
#   (b) Ertel PV: u/v/t/q on 150-400 hPa -- winds and temperature at the level of interest AND
#       its vertical neighbours for the dtheta/dp term. Five levels, not three, because a
#       strong ridge LIFTS the tropopause, so a 200-300 hPa window can sit entirely below the
#       feature; q carries the moist/diabatic term.
# u500/v500 with z500 also make the 500 hPa ageostrophic diagnostic computable WITHIN one
# dump -- it needs V and Vg from the SAME trajectory, so splitting them across two passes
# would difference fields from different realizations.
# Dropping channels is the cheapest way to shrink the dump; recovering one costs a re-rollout.
CHANNELS_DEFAULT="t2m,z500,u10m,v10m,msl,tcwv,z300,u500,v500,t500,u850,v850,u150,u200,u250,u300,u400,v150,v200,v250,v300,v400,t150,t200,t250,t300,t400,q150,q200,q250,q300,q400"
CHANNELS="${6:-$CHANNELS_DEFAULT}"
if [ "$#" -gt 6 ]; then shift 6; else shift "$#"; fi

SEED="${SLURM_ARRAY_TASK_ID:-1}"
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "NOTE: not an array job (no SLURM_ARRAY_TASK_ID) — running a single task at seed $SEED."
    echo "      For a pooled ensemble submit with --array=1-250%32."
fi
if [ "$SEED" -eq 0 ] && [ "$ENSEMBLE_SIZE" -gt 1 ]; then
    echo "ERROR: --noise_seed 0 means 'do not reseed', so this task would emit the SAME members"
    echo "       as any other seed-0 task — silent duplicates, not independent members."
    echo "       Submit arrays starting at 1 (e.g. --array=1-250%32)."
    exit 1
fi

if [ "$ENSEMBLE_SIZE" -gt 4 ]; then
    awk -v e="$ENSEMBLE_SIZE" 'BEGIN{
        printf "WARNING: E=%d is above the MEASURED DISCO ceiling of 4 folded samples/GPU.\n", e
        printf "         FCN3 allocates ~13.37 GiB per sample in ONE fp32 buffer -> %.1f GiB here,\n", 13.37*e
        printf "         against 178 GiB total. E=16 (213.9 GiB) OOMed on a real attempt.\n"
        printf "         Pure fp32 skips the bf16 cast copy seen during training, so E=8 MIGHT\n"
        printf "         fit -- but that has never been measured. Expect an OOM after the model loads.\n" }'
fi

SCORES="$RUNS/$CONFIG_TARGET/$RUN_NUM/scores"
# The case-study target is EVALUATION-ONLY -- it is never trained, so its own run dir holds
# no training_checkpoints. The weights come from the TRAINED target (identical architecture;
# the case-study config differs only in inf_data_path), and we pass them explicitly with
# --checkpoint_path rather than relying on a symlink someone has to remember to create.
# Override with CKPT_CONFIG/CKPT_RUN if evaluating a different trained run. If CKPT_CONFIG equals CONFIG_TARGET this reduces to the ordinary layout.
CKPT_CONFIG="${CKPT_CONFIG:-rewind_fcn3_ens_10yr_backward}"
CKPT_RUN="${CKPT_RUN:-$RUN_NUM}"
CKPT_DIR="$RUNS/$CKPT_CONFIG/$CKPT_RUN/training_checkpoints"
# Per-CASE, per-seed output names. Two distinct collisions to avoid:
#   (1) ACROSS TASKS: the harness defaults (backward_rollout.json / backward_fields.h5 /
#       backward_spectrum.h5) are FIXED names, so concurrent array tasks sharing a scores
#       dir would overwrite each other and the whole array would leave one file.
#   (2) ACROSS CASE STUDIES: the scores dir is keyed on <config>/<run_num>, and every case
#       study uses the SAME evaluation-only target and the same trained run. Without a case
#       tag, rolling out a second event (Ian) would silently overwrite the first (the PNW
#       heat dome) -- terabytes of finished ensemble, gone, with no error. So the anchor date
#       goes in the filename. Override CASE_TAG for a custom label.
CASE_TAG="${CASE_TAG:-$(printf '%s' "$ANCHOR_DATE" | tr -cd '[:alnum:]')}"
TAG=$(printf "%s_seed%03d" "$CASE_TAG" "$SEED")
OUT_JSON="$SCORES/backward_rollout_${TAG}.json"
FIELDS_FILE="backward_fields_${TAG}.h5"
SPECTRUM_FILE="backward_spectrum_${TAG}.h5"

N_CH=$(awk -F, '{print NF}' <<< "$CHANNELS")
echo "=== large-ensemble backward rollout ==="
echo "  array task : ${SLURM_ARRAY_TASK_ID:-<none>}   seed: $SEED"
echo "  config     : $CONFIG_TARGET   run: $RUN_NUM"
echo "  checkpoint : $CKPT_DIR"
echo "  rollout    : R=$NUM_STEPS steps (-$((NUM_STEPS * 6)) h), E=$ENSEMBLE_SIZE members"
echo "  anchor     : $ANCHOR_DATE"
echo "  channels   : $N_CH -> $CHANNELS"
echo "  port       : $((29500 + SEED % 1000))   (per-task; array tasks share nodes)"
echo "  outputs    : $OUT_JSON"
echo "               $SCORES/$FIELDS_FILE"
# Storage is the thing most likely to bite overnight, so state it before writing any of it.
# 721*1440*4 B = 4.153 MB per channel-frame; (R+1) leads x E members x N_CH channels, plus
# one member-independent truth copy per file.
awk -v r="$NUM_STEPS" -v e="$ENSEMBLE_SIZE" -v c="$N_CH" \
  'BEGIN{ mb=4.153; p=(r+1)*e*c*mb/1024; t=(r+1)*c*mb/1024;
          printf "  dump size  : %.1f GB pred + %.1f GB truth = %.1f GB this task\n", p, t, p+t }'
test -d "$CKPT_DIR" || { echo "ERROR: no checkpoint dir $CKPT_DIR"; exit 1; }
mkdir -p "$SCORES"

# Preflight: CUDA visible INSIDE the container. Catches the stale-NGC-compat error 803
# (see docker/makani.def) as a ~2 s self-explanatory failure instead of a 10 s traceback -- and on an
# array, catches it once per task rather than leaving a whole array of identical tracebacks.
apptainer exec --nv \
  --bind "$SLURM_SUBMIT_DIR":/opt/makani \
  --bind "$MAKANI_DATA":"$MAKANI_DATA" \
  "$IMG" \
  python -c "
import sys, torch
if not torch.cuda.is_available():
    print('PREFLIGHT FAIL: torch.cuda.is_available() is False inside the container.')
    print('  If the log above mentions CUDA error 803 / unsupported display driver, this is the')
    print('  stale NGC compat-lib fault (docker/makani.def): bind an empty dir over /usr/local/cuda/compat')
    print('  or rebuild the image. It is NOT a bad node -- resubmitting will hit it again.')
    sys.exit(1)
print(f'preflight OK: {torch.cuda.get_device_name(0)}')
" || exit 1

# --amp_mode none --tf32 off: fp32 throughout, so a high-l artifact cannot be blamed on
# the rounding floor. sigma stays at the TRAINED value (1.0); changing it here would make
# these members incomparable to every other result.
# Single-process (1-GPU) distributed env, same as the other rollout sbatches: physicsnemo's
# DistributedManager checks RANK -> SLURM_PROCID -> OMPI_*, and a plain sbatch leaks
# SLURM_PROCID=0, so without RANK set it takes the SLURM branch, calls initialize_slurm(),
# and dies with `TypeError: str expected, not NoneType` when MASTER_ADDR comes back None
# (those vars only exist under srun). Setting RANK forces the env branch.
#
# MASTER_PORT IS PER-TASK, unlike the single-job siblings that hardcode 29500. Up to 32 array
# tasks run at once on 8-GPU nodes, so several share a node; even a world_size=1 group binds
# a TCPStore on this port, and identical ports on one node collide.
MASTER_PORT=$((29500 + SEED % 1000))
apptainer exec --nv \
  --bind "$SLURM_SUBMIT_DIR":/opt/makani \
  --bind "$MAKANI_DATA":"$MAKANI_DATA" \
  --env RANK=0 --env WORLD_SIZE=1 --env LOCAL_RANK=0 \
  --env MASTER_ADDR=127.0.0.1 --env MASTER_PORT="$MASTER_PORT" \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env WANDB_MODE=disabled \
  "$IMG" \
  python /opt/makani/analysis/backward_rollout.py \
    --yaml_config=/opt/makani/config/rewind_fcn3.yaml \
    --config="$CONFIG_TARGET" \
    --run_num="$RUN_NUM" \
    --num_steps="$NUM_STEPS" \
    --ensemble_size="$ENSEMBLE_SIZE" \
    --noise_seed="$SEED" \
    --anchor_date="$ANCHOR_DATE" \
    --anchor_date_exact \
    --checkpoint_path="$CKPT_DIR" \
    --save_fields_anchors=1 \
    --save_fields_channels="$CHANNELS" \
    --fields_member=all \
    --fields_file="$FIELDS_FILE" \
    --spectrum_file="$SPECTRUM_FILE" \
    --output_json="$OUT_JSON" \
    --amp_mode none \
    --tf32 off \
    "$@"

echo "=== task $SEED complete in $((SECONDS - _T0)) s  (seff $SLURM_JOB_ID for resource usage) ==="
ls -lh "$SCORES/$FIELDS_FILE"
echo
echo "POOL (after the array finishes): the dumps concatenate on the member axis --"
echo "  from backward_fields_io import load_channel_members"
echo "  import glob, numpy as np"
echo "  files = sorted(glob.glob('$SCORES/backward_fields_${CASE_TAG}_seed*.h5'))"
echo "  pred = np.concatenate([load_channel_members(f, 't2m')['pred'] for f in files], axis=0)"
echo "  # pred: (n_files*E, R+1, H, W)   -- check each file's noise_seed attr is distinct"
