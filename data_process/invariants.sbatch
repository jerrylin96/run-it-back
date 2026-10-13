#!/bin/bash
# ============================================================================
# Build the orography + land-sea-mask invariants from the WeatherBench2 ERA5 zarr
# (geopotential_at_surface -> key Z, land_sea_mask -> key LSM), grid-aligned to
# metadata_fcn3_72.json. Input-only static auxiliaries (never targets).
# Tiny + fast (two 721x1440 reads); needs OUTBOUND INTERNET (GCS), not a GPU.
# Run: cd <repo root>; sbatch data_process/invariants.sbatch
# ============================================================================
#SBATCH --job-name=makani-invariants
#SBATCH --partition=b200-devel       # confirmed outbound internet (GPU is UNUSED here)
#SBATCH --gres=gpu:1                  #   -> if your `cpu` partition has internet, prefer it:
#SBATCH --nodes=1                     #      set --partition=cpu and delete the --gres line
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=makani-invariants-%j.log
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,INVALID_DEPEND,TIME_LIMIT

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from your repo root}"
# site-specific paths (container image, data root): edit site.env once
source "$SLURM_SUBMIT_DIR/site.env"

IMG="${MAKANI_IMG:?set MAKANI_IMG in site.env}"
META=/opt/makani/data_process/examples/metadata_fcn3_72.json
OUT="${MAKANI_DATA:?set MAKANI_DATA in site.env}/invariants"
ZARR="gs://weatherbench2/datasets/era5/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2"

test -f data_process/make_invariants.py || { echo "ERROR: make_invariants.py not in $(pwd)"; exit 1; }
mkdir -p "$OUT"

# no --nv — this is CPU/network only (xarray/gcsfs/h5py). Single mount set, used by both calls.
run () {
  apptainer exec \
    --bind "$SLURM_SUBMIT_DIR":/opt/makani \
    --bind "$MAKANI_DATA":"$MAKANI_DATA" \
    "$IMG" \
    python /opt/makani/data_process/make_invariants.py "$@"
}

# Self-test the HDF5 file-format contract first (fast, offline) — fail early if the read/write
# contract regressed before spending the network round-trip. Write the synthetic files to a
# throwaway dir and remove it, so the real $OUT holds only the two invariants.
SELFTEST="$OUT/.selftest_${SLURM_JOB_ID:-tmp}"
run --self_test --output_dir "$SELFTEST"
rm -rf "$SELFTEST"

# Build the real invariants. Aligns to the SAME grid/orientation as the converted fields (match),
# and HARD-FAILS on NaN or a flipped grid (the geographic gate), so a bad invariant can't be written.
run --input_file "$ZARR" --metadata_file "$META" --output_dir "$OUT"

echo "=== invariants written to $OUT ==="
ls -lh "$OUT"
echo "ACCEPTANCE: the geographic sanity is a HARD gate inside make_invariants.py — if the grid"
echo "were flipped/mis-aligned, NaN-bearing, or the wrong variable, the step above would have"
echo "raised and (set -e) failed this job before writing. Reaching here with both .h5 files"
echo "present means the gate passed (Himalaya >> Pacific; Sahara land, Pacific sea)."
