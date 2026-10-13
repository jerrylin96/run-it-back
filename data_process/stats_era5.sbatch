#!/bin/bash
# ============================================================================
# Normalization stats: get_stats.py over the 40 TRAIN years (1979-2018) of the 72-ch /
# 0.25 deg / 6-hourly set. Writes the seven .npy files config/rewind_fcn3.yaml points at
# ($DATA/stats/{mins,maxs,time_means,global_means,global_stds,time_diff_means_dt1,
# time_diff_stds_dt1}.npy). Valid/test reuse the train stats. Settings follow the
# FourCastNet 3 paper (Table 4):
#   --quadrature_rule clenshaw-curtiss  equiangular grid -> correct area weights
#   --wind_angle_aware                  zero-mean / total-wind-magnitude-std z-score
#                                       variant for u*/v* pairs; u and v of a pair get
#                                       the SAME std, so normalization cannot rotate winds
#   --dt 1                              6-hourly single step (dt in units of dhours=6)
#   NO --fail_on_nan: imputed timestamps are legitimately NaN and valid_data-masked;
#   get_stats masks them out of every reduction (mean/std AND min/max), so the q*/tcwv
#   min/max stats the config's normalization map consumes are impute-safe.
#
# WHERE THIS RUNS: a GPU partition is REQUIRED - get_stats.py computes
# `comm_rank % torch.cuda.device_count()`, which is a modulo-by-ZERO on a CPU-only
# node. The compute is trivial and the job is NFS-read-bound, so use the cheapest GPU
# partition available. 8 MPI ranks / 8 GPUs on one node parallelize the reads;
# comm_size (8) must be divisible by --reduction_group_size (8).
#
# MEMORY: get_stats converts each batch to float64 on GPU. At 72ch a batch-16 peak
# (data + dt-shifted copy + diff + masks, ~10 GB each) crowds a 96 GB GPU, so
# --batch_size 8 (~25-30 GB peak); free, since the job is I/O-bound.
#
# NO RESUME: get_stats writes only at the end, so a timeout loses everything - hence
# the full 1-day limit for a job expected to take hours (16 TiB / NFS bandwidth).
# The footer prints the effective read bandwidth, which sizes training I/O.
# Run: cd <repo root>; sbatch data_process/stats_era5.sbatch
# ============================================================================
#SBATCH --job-name=makani-stats-era5
#SBATCH --partition=rtx-batch        # GPU required (see header); rtx, not b200, on purpose
#SBATCH --nodes=1
#SBATCH --ntasks=8                   # MPI ranks; --ntasks=N + `mpirun -np N` resolves the
#SBATCH --cpus-per-task=4            #   container OpenMPI "not enough slots" gotcha
#SBATCH --gres=gpu:8                 # 1 GPU per rank (local_rank = comm_rank % device_count)
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00            # no partial output/resume -> generous limit (see header)
#SBATCH --output=makani-stats-era5-%j.log
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,INVALID_DEPEND,TIME_LIMIT

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from your repo root}"
# site-specific paths (container image, data root): edit site.env once
source "$SLURM_SUBMIT_DIR/site.env"

IMG="${MAKANI_IMG:?set MAKANI_IMG in site.env}"
META=/opt/makani/data_process/examples/metadata_fcn3_72.json
DATA="${MAKANI_DATA:?set MAKANI_DATA in site.env}"

test -f data_process/get_stats.py || { echo "ERROR: get_stats.py not in $(pwd)"; exit 1; }
test -f data_process/examples/metadata_fcn3_72.json || { echo "ERROR: metadata_fcn3_72.json missing"; exit 1; }
NTRAIN=$(ls -1 "$DATA"/train/*.h5 2>/dev/null | wc -l) || NTRAIN=0  # empty glob must not set-e-abort before the message below
[ "$NTRAIN" -eq 40 ] || { echo "ERROR: expected 40 train files in $DATA/train, found $NTRAIN — finish convert_era5.sbatch first"; exit 1; }
if compgen -G "$DATA/train/*.h5.tmp" > /dev/null; then echo "ERROR: stray .h5.tmp in $DATA/train — a conversion is running or died; resolve first"; exit 1; fi
mkdir -p "$DATA/stats"

# PRE-SCAN: count imputed timesteps (valid_data==0) across the train years. Purely
# informational for mean/std/min/max (get_stats masks them), but time_means.npy is the
# one output where upstream deliberately lets NaN propagate ("nan stay nan since they
# are localized in time") - so a nonzero count here tells the post-check below to
# expect (and report, not fail on) NaN pixels in time_means.
echo "=== pre-scan: imputed timesteps in train years ==="
apptainer exec --bind "$MAKANI_DATA":"$MAKANI_DATA" "$IMG" \
  python -c "
import glob, h5py
total = 0
for fn in sorted(glob.glob('$DATA/train/*.h5')):
    with h5py.File(fn, 'r') as f:
        n = int((f['valid_data'][...] == 0).sum())
    if n:
        print(f'  {fn}: {n} (timestep,channel) entries imputed')
    total += n
print(f'  total imputed entries: {total}')
open('$DATA/stats/.imputed_count', 'w').write(str(total))
"

# mpirun runs INSIDE the container (its from-source OpenMPI); --ntasks=N gives it N slots.
# get_stats does its own torch.distributed wireup from the MPI env (MASTER_ADDR etc.), so
# no manual RANK/WORLD_SIZE env is needed.
: "${SLURM_NTASKS:?--ntasks must be set so mpirun -np matches the allocation}"

# PREFLIGHT: fail in SECONDS, not after hours, on the two ways a multi-GPU stats run wastes a
# day. get_stats binds each rank to cuda:(rank % torch.cuda.device_count()); this reproduces
# that binding under the exact same mpirun launch and aborts if either (a) two ranks land on
# the same physical GPU (binding collision -> device_count < ntasks) or (b) a GPU already holds
# memory before we allocate (an orphan rank from a prior cancelled run, or another job's process
# squatting on the node). Both surface as "another process on GPU N" -> the OOM that killed the
# first attempt. This preflight was ADDED after that attempt (rank 5 OOM'd because GPU 5 already
# had ~90 GiB in use by another process while the rest progressed to 50%).
# On preflight failure, dump the node's GPU state + the OWNER/PID/elapsed-time of every process
# on its GPUs. This runs on the compute node (host nvidia-smi/ps, outside the container), where
# ps CAN see other tenants' processes and usernames -- so if a foreign or orphaned process is
# squatting on an allocated GPU, its owner lands in THIS log, ready for a node-hygiene report.
gpu_state_dump() {
  echo "=== GPU state on $(hostname) at preflight failure (capture for a node-hygiene bug report) ==="
  nvidia-smi || true
  echo "--- owners of processes currently on this node's GPUs ---"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | while IFS=, read -r pid _; do
    pid=$(echo "$pid" | tr -dc '0-9')
    [ -n "$pid" ] && { ps -o user=,pid=,etime=,cmd= -p "$pid" 2>/dev/null || echo "  pid $pid: owner not resolvable here"; }
  done
}

echo "=== GPU binding/occupancy preflight ($SLURM_NTASKS ranks) ==="
apptainer exec --nv \
  --bind "$MAKANI_DATA":"$MAKANI_DATA" \
  "$IMG" \
  mpirun -np "$SLURM_NTASKS" python -c "
import os, socket, torch
from collections import defaultdict
from mpi4py import MPI
c = MPI.COMM_WORLD; r = c.Get_rank(); n = c.Get_size()
dc = torch.cuda.device_count()
li = r % dc if dc else -1
if dc:
    torch.cuda.set_device(li)
    free, total = torch.cuda.mem_get_info(li)
else:
    free = total = 0
info = c.gather((r, socket.gethostname(), os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>'), dc, li, free, total), root=0)
if r == 0:
    bad = False
    seen = defaultdict(dict)
    GiB = 1024**3
    for rr, host, cvd, d, idx, fr, tot in sorted(info):
        used = (tot - fr) / GiB
        print(f'  rank {rr} {host} CVD={cvd} device_count={d} -> cuda:{idx}  free={fr/GiB:.1f}/{tot/GiB:.1f} GiB (used {used:.1f})')
        if idx in seen[host]:
            print(f'    ERROR: rank {rr} and rank {seen[host][idx]} both bound cuda:{idx} on {host} (device_count={d}) — binding collision'); bad = True
        seen[host][idx] = rr
        if used > 2.0:
            print(f'    ERROR: cuda:{idx} on {host} already has {used:.1f} GiB used before we allocate — orphan/contention (free it: nvidia-smi, then kill the stray PID)'); bad = True
    if bad:
        raise SystemExit('PREFLIGHT FAIL: fix GPU binding/occupancy before running get_stats (see errors above)')
    print(f'  PREFLIGHT OK: {n} ranks each on a distinct, empty GPU')
" || { gpu_state_dump; echo "PREFLIGHT FAIL: aborting before the multi-hour run (GPU state captured above)"; exit 1; }

echo "=== get_stats over $NTRAIN train years -> $DATA/stats ==="
T0=$(date +%s)
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True: anti-fragmentation (the OOM message's own
# suggestion). Harmless if memory is ample; buys headroom on the float64 alloc/free churn.
apptainer exec --nv \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --bind "$SLURM_SUBMIT_DIR":/opt/makani \
  --bind "$MAKANI_DATA":"$MAKANI_DATA" \
  "$IMG" \
  mpirun -np "$SLURM_NTASKS" \
    python /opt/makani/data_process/get_stats.py \
      --input_path "$DATA/train" \
      --metadata_file "$META" \
      --output_path "$DATA/stats" \
      --dt 1 \
      --quadrature_rule clenshaw-curtiss \
      --wind_angle_aware \
      --reduction_group_size "$SLURM_NTASKS" \
      --batch_size 8
T1=$(date +%s)

# Effective sequential-read bandwidth over the train set.
# get_stats reads each batch region TWICE (the batch + the dt-shifted re-read for tendency
# stats), so the achieved NFS rate is ~2x stored-bytes/time - report that; it is the number
# that sizes training I/O. Guarded with || so a du/arithmetic hiccup on this vanity metric
# can never abort the script between a successful stats run and the acceptance gate below.
ELAPSED=$((T1-T0)); [ "$ELAPSED" -gt 0 ] || ELAPSED=1
TOTAL_BYTES=$(du -cb "$DATA"/train/*.h5 2>/dev/null | tail -1 | cut -f1) || TOTAL_BYTES=0
echo "=== data read throughput: ~$(( 2 * TOTAL_BYTES / ELAPSED / 1000000 )) MB/s aggregate (2 x $TOTAL_BYTES stored bytes / ${ELAPSED} s) ===" || true

# ACCEPTANCE: the seven files exist with the right shapes; stats are finite; the FCN3
# normalization-map channels (q*/tcwv -> minmax) have usable min<max; wind_angle_aware
# actually took effect (u/v of a pair share one std). Fails loud -> do NOT train on
# stats that fail here.
echo "=== stats acceptance checks ==="
apptainer exec \
  --bind "$SLURM_SUBMIT_DIR":/opt/makani \
  --bind "$MAKANI_DATA":"$MAKANI_DATA" \
  "$IMG" \
  python /opt/makani/data_process/check_stats.py \
    --stats_dir "$DATA/stats" \
    --metadata_file "$META" \
    --dt 1 \
    --imputed_count_file "$DATA/stats/.imputed_count"

echo "=== stats written to $DATA/stats ==="
ls -lh "$DATA/stats"
