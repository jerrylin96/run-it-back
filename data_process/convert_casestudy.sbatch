#!/bin/bash
# ============================================================================
# Convert ARBITRARY years from ARCO-ERA5 -> makani HDF5, for CASE-STUDY evaluation on
# dates outside the training record (1979-2020). Same 72 channels / 0.25 deg / 6-hourly /
# canonical ARCO store as convert_era5.sbatch — a parameterized twin of that script, NOT
# a new pipeline, so the two produce byte-compatible files.
#
# WHY THIS EXISTS: the case studies we retrodict post-date the converted record — the
# 2021 Pacific Northwest heat dome (Jun 2021) and Hurricane Ian (Sep 2022). The ARCO store
# spans 1940-2025, so these years are simply ones never downloaded.
#
# OUTPUT DIR defaults to $DATA/casestudy, which is what config/rewind_fcn3.yaml's
# `rewind_fcn3_ens_casestudy` target points inf_data_path/valid_data_path at.
# The multifiles loader globs the directory and reads each file's own timestamp scale,
# so several NON-OVERLAPPING years may share one dir (2021 + 2022 is fine). Do NOT
# put overlapping slices of the same period in one dir.
#
# WHERE THIS RUNS: a CPU partition — CPU/network only, zero GPU compute. Gate with
# data_process/net_probe.sbatch if GCS reachability is in doubt.
#
# RESUME: the converter writes one file per year and SKIPS existing files, so
# resubmitting is safe and picks up where it left off.
#
# Run: cd <repo root>
#      sbatch data_process/convert_casestudy.sbatch 2021
#      sbatch data_process/convert_casestudy.sbatch 2021 2022      # both at once
#      OUT_DIR=/path/to/dir sbatch data_process/convert_casestudy.sbatch 2022
#
# Expect ~18 min and ~0.44 TB per year.
# ============================================================================
#SBATCH --job-name=makani-convert-case
#SBATCH --partition=cpu              # CPU/network-only job; NOT a GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=8                   # MPI ranks; --ntasks=N + `mpirun -np N` resolves the
#SBATCH --cpus-per-task=4            #   container OpenMPI "not enough slots" gotcha
#SBATCH --mem=128G
#SBATCH --time=04:00:00              # ~18 min/yr; short request schedules sooner than the 1-day cap
#SBATCH --output=makani-convert-case-%j.log
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,INVALID_DEPEND,TIME_LIMIT

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from your repo root}"
# site-specific paths (container image, data root): edit site.env once
source "$SLURM_SUBMIT_DIR/site.env"

IMG="${MAKANI_IMG:?set MAKANI_IMG in site.env}"
META=/opt/makani/data_process/examples/metadata_fcn3_72.json
DATA="${MAKANI_DATA:?set MAKANI_DATA in site.env}"
ZARR="gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
OUT_DIR="${OUT_DIR:-$DATA/casestudy}"

# Years come from the command line. Fail loud rather than silently converting a default
# year nobody asked for: a wrong year here costs ~18 min and ~0.44 TB before anyone notices.
if [ "$#" -lt 1 ]; then
    echo "ERROR: no years given."
    echo "usage: sbatch $(basename "$0") <year> [year ...]      (e.g. 2021, or 2021 2022)"
    exit 1
fi
YEARS=("$@")
for y in "${YEARS[@]}"; do
    case "$y" in
        [0-9][0-9][0-9][0-9]) ;;
        *) echo "ERROR: '$y' is not a 4-digit year"; exit 1 ;;
    esac
    # The ARCO store covers 1940-2025; anything outside it fails deep inside xarray with a
    # confusing slice error, so reject it here where the message is readable.
    if [ "$y" -lt 1940 ] || [ "$y" -gt 2025 ]; then
        echo "ERROR: year $y is outside the ARCO store's 1940-2025 range"; exit 1
    fi
done

test -f data_process/convert_wb2_to_makani_input.py || { echo "ERROR: converter not in $(pwd)"; exit 1; }
test -f data_process/examples/metadata_fcn3_72.json || { echo "ERROR: metadata_fcn3_72.json missing"; exit 1; }
mkdir -p "$OUT_DIR"

echo "=== case-study conversion ==="
echo "  years      : ${YEARS[*]}"
echo "  output dir : $OUT_DIR"
echo "  store      : $ZARR"
echo "  metadata   : $META (72 channels)"
df -h "$OUT_DIR" 2>/dev/null | tail -1 || true

: "${SLURM_NTASKS:?--ntasks must be set so mpirun -np matches the allocation}"

# Flags identical to convert_era5.sbatch so the files are interchangeable with the
# training record. NO --skip_missing_channels: a missing channel must FAIL LOUD with IndexError
# rather than leave an all-zero channel. --impute_missing_timestamps handles a rare missing
# 6-hourly step via NaN + valid_data=0.
apptainer exec \
  --bind "$SLURM_SUBMIT_DIR":/opt/makani \
  --bind "$MAKANI_DATA":"$MAKANI_DATA" \
  "$IMG" \
  mpirun -np "$SLURM_NTASKS" \
    python /opt/makani/data_process/convert_wb2_to_makani_input.py \
      --input_file "$ZARR" \
      --output_dir "$OUT_DIR" \
      --metadata_file "$META" \
      --years "${YEARS[@]}" \
      --batch_size 8 \
      --impute_missing_timestamps

# ACCEPTANCE: shape/finiteness per converted year. A widened version of convert_era5.sbatch's
# 100m-wind spot-check, because a case study anchors on ONE date: a year that converted
# with a silently truncated time axis would fail the anchor lookup at rollout time (hours later,
# in a different job) rather than here. Checks the channels this run exists to serve.
echo "=== acceptance: per-year shape + finiteness ==="
apptainer exec \
  --bind "$SLURM_SUBMIT_DIR":/opt/makani \
  --bind "$MAKANI_DATA":"$MAKANI_DATA" \
  "$IMG" \
  python -c "
import json, sys, os, numpy as np, h5py
ch = json.load(open('$META'))['coords']['channel']
years = '${YEARS[*]}'.split()
bad = 0
for y in years:
    p = os.path.join('$OUT_DIR', y + '.h5')
    if not os.path.isfile(p):
        print(f'  {y}: MISSING {p}'); bad += 1; continue
    with h5py.File(p, 'r') as f:
        T, C, H, W = f['fields'].shape
        # 6-hourly: 1460 steps, or 1464 in a leap year.
        leap = (int(y) % 4 == 0 and int(y) % 100 != 0) or (int(y) % 400 == 0)
        want = 1464 if leap else 1460
        ok = (T == want and C == 72 and (H, W) == (721, 1440))
        print(f'  {y}: shape ({T},{C},{H},{W}) expect ({want},72,721,1440) -> {\"OK\" if ok else \"MISMATCH\"}')
        if not ok:
            bad += 1
        # Spot-check the channels the case studies actually use, at both ends of the year.
        for name in ('t2m', 'z500', 'u250', 't250', 'q250'):
            i = ch.index(name)
            fr = min(float(np.isfinite(f['fields'][t, i]).mean()) for t in (0, T - 1))
            flag = 'OK' if fr > 0.9 else 'MOSTLY-NaN'
            print(f'      {name}: {fr*100:.1f}% finite (worst of first/last step) -> {flag}')
            if fr <= 0.9:
                bad += 1
sys.exit(1 if bad else 0)
" || { echo "ACCEPTANCE FAILED — do NOT run the rollout against this data"; exit 1; }

echo "=== conversion complete ==="
ls -lh "$OUT_DIR"/*.h5 2>/dev/null || true
echo
echo "NEXT: point the rollout at this directory. config/rewind_fcn3.yaml's"
echo "      rewind_fcn3_ens_casestudy target already sets inf_data_path/valid_data_path to"
echo "      $DATA/casestudy — if OUT_DIR differs from that, update the target before rolling out."
