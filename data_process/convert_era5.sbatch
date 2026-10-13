#!/bin/bash
# ============================================================================
# Convert the training dataset from ARCO-ERA5 -> makani HDF5: the full FCN3 72-channel
# set, 0.25 deg / 6-hourly, 1979-2018 train / 2019 valid / 2020 test, RECENT-YEARS-FIRST
# so shorter training records are usable while the rest converts. MPI-parallel.
#
# SOURCE = the canonical ARCO store gs://gcp-public-data-arco-era5/ar/...  (277 vars,
# zarr_format 2 despite the "-v3" name). We use it INSTEAD of the WeatherBench2 subset
# store because WB2 omits u100m/v100m. ARCO has them, on the identical grid/chunking/dtype
# as WB2, so there is no download-speed cost. Fallback if the ARCO bucket is unreachable or
# requester-pays: the WB2 store + metadata_fcn3_70.json (drop u100m/v100m).
#
# STORAGE = contiguous (the converter's default), NOT chunked. The fields dataset is
# (T, C, H, W) row-major, so each (timestep, channel) plane is already a contiguous block:
# the multifiles loader's channel-subset read (`[t, channels, :, :]`) touches only the
# requested channels' planes. Explicit (1,1,H,W) chunking adds no read benefit AND is
# unsafe under this MPIO parallel write (HDF5 requires chunk allocation to be collective).
#
# WHERE THIS RUNS: a CPU partition. The job is CPU/network only (xarray/gcsfs/h5py) and
# uses no GPU. GATE: run data_process/net_probe.sbatch FIRST to confirm the partition
# reaches GCS; if it does not, use a GPU partition that has outbound internet.
#
# RESUME: the converter writes one file per year (atomically, via a .tmp rename) and
# SKIPS existing files, so if the job times out before all 40 train years finish, just
# resubmit — it picks up where it left off.
# Run: cd <repo root>; sbatch data_process/convert_era5.sbatch
# ============================================================================
#SBATCH --job-name=makani-convert-era5
#SBATCH --partition=cpu              # CPU/network-only job; NOT a GPU partition (see header)
#SBATCH --nodes=1
#SBATCH --ntasks=8                   # MPI ranks; --ntasks=N + `mpirun -np N` resolves the
#SBATCH --cpus-per-task=4            #   container OpenMPI "not enough slots" gotcha
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00            # cpu partition allows 1 day; resubmit to resume (skips done years)
#SBATCH --output=makani-convert-era5-%j.log
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,INVALID_DEPEND,TIME_LIMIT

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from your repo root}"
# site-specific paths (container image, data root): edit site.env once
source "$SLURM_SUBMIT_DIR/site.env"

IMG="${MAKANI_IMG:?set MAKANI_IMG in site.env}"
META=/opt/makani/data_process/examples/metadata_fcn3_72.json
DATA="${MAKANI_DATA:?set MAKANI_DATA in site.env}"
ZARR="gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

test -f data_process/convert_wb2_to_makani_input.py || { echo "ERROR: converter not in $(pwd)"; exit 1; }
test -f data_process/examples/metadata_fcn3_72.json || { echo "ERROR: metadata_fcn3_72.json missing"; exit 1; }
mkdir -p "$DATA/train" "$DATA/valid" "$DATA/test"

# mpirun runs INSIDE the container (its from-source OpenMPI); --ntasks=N gives it N slots
# (resolves the "not enough slots" gotcha). Single-node only: container-internal mpirun does not
# span nodes, so keep --nodes=1. No --nv (CPU/network only). NO --skip_missing_channels: we
# verified all 72 channels exist in the canonical ARCO store, so a missing channel should FAIL
# LOUD with IndexError (silently skipping would leave an all-zero channel whose zero-variance
# z-score stats become NaN). --impute_missing_timestamps handles the (rare) missing 6-hourly step
# via NaN + valid_data=0, which get_stats masks out of every reduction (mean/std AND min/max).
: "${SLURM_NTASKS:?--ntasks must be set so mpirun -np matches the allocation}"
run_convert () {                      # $1 = output dir, $2.. = years
  local out="$1"; shift
  apptainer exec \
    --bind "$SLURM_SUBMIT_DIR":/opt/makani \
    --bind "$MAKANI_DATA":"$MAKANI_DATA" \
    "$IMG" \
    mpirun -np "$SLURM_NTASKS" \
      python /opt/makani/data_process/convert_wb2_to_makani_input.py \
        --input_file "$ZARR" \
        --output_dir "$out" \
        --metadata_file "$META" \
        --years "$@" \
        --batch_size 8 \
        --impute_missing_timestamps
}

# Run the three passes independently: a transient GCS/network failure on one must NOT abort the
# others (the converter skips already-written years, so resubmitting resumes). valid (2019) +
# test (2020) first (small, needed for eval + the first lean rung), then train RECENT-YEARS-FIRST
# (descending) so shorter training records are usable before the full record finishes.
rc=0
echo "=== VALID 2019 -> $DATA/valid ==="; run_convert "$DATA/valid" 2019              || { echo "valid pass FAILED (resubmit to retry)"; rc=1; }
echo "=== TEST  2020 -> $DATA/test  ==="; run_convert "$DATA/test"  2020              || { echo "test pass FAILED (resubmit to retry)";  rc=1; }
echo "=== TRAIN 2018..1979 (recent-first) -> $DATA/train ==="
run_convert "$DATA/train" $(seq 2018 -1 1979)                                          || { echo "train pass FAILED (resubmit to resume)"; rc=1; }

# SPOT-CHECK: confirm u100m/v100m converted to REAL values (they're the reason we chose the ARCO
# store over WB2). The store's real data spans 1940-2025, so our 1979-2020 years must be finite; a
# mostly-NaN result flags a bad time range or a present-but-NaN value the impute path won't catch.
echo "=== 100m-wind non-NaN spot-check ==="
apptainer exec \
  --bind "$SLURM_SUBMIT_DIR":/opt/makani \
  --bind "$MAKANI_DATA":"$MAKANI_DATA" \
  "$IMG" \
  python -c "
import glob, json, sys, numpy as np, h5py
ch = json.load(open('$META'))['coords']['channel']
files = sorted(glob.glob('$DATA/valid/*.h5')) or sorted(glob.glob('$DATA/train/*.h5'))
if not files:
    print('  no converted file yet — skip'); sys.exit(0)
with h5py.File(files[0], 'r') as f:
    for name in ('u100m', 'v100m'):
        x = f['fields'][0, ch.index(name)]
        frac = float(np.isfinite(x).mean())
        print(f'  {name}: {frac*100:.1f}% finite, mean={np.nanmean(x):.3f} ({files[0]})')
        assert frac > 0.9, f'{name} is mostly NaN — investigate before running stats'
print('  100m winds OK (real, not NaN-fill)')
" || { echo "100m spot-check FAILED — do NOT proceed to stats"; rc=1; }

echo "=== conversion pass complete (rc=$rc) ==="
echo "ACCEPTANCE: (1) job COMPLETES WITHOUT IndexError — since --skip_missing_channels is OFF, a"
echo "missing channel aborts with IndexError (intended fail-loud), so clean completion IS the"
echo "channel-presence check; (2) the 100m spot-check above prints OK. Per-year files present:"
ls -1 "$DATA"/train/*.h5 "$DATA"/valid/*.h5 "$DATA"/test/*.h5 2>/dev/null | wc -l
echo "(expect up to 40 train + 1 valid + 1 test = 42 files once fully converted; resubmit to finish.)"
exit $rc
