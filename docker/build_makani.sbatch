#!/bin/bash
# ============================================================================
# Build the container image (docker/makani.def) as a batch job. Submit from the repo root
# so the relative %files paths in the definition resolve:
#     cd <repo root>
#     sbatch docker/build_makani.sbatch
# The image is written to $MAKANI_IMG (site.env). Needs outbound internet (NGC pull, pip,
# git clones); the GPU itself is unused by the build.
# Partition names are our cluster's; override with sbatch -p / --gres.
# ============================================================================
#SBATCH --job-name=makani-build
#SBATCH --partition=b200-devel      # idle + confirmed internet (the GPU itself is unused by the build)
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32          # parallelism for make -j / pip builds
#SBATCH --mem=128G                  # headroom; the RAM-backed tmp below counts against this
#SBATCH --time=04:00:00
#SBATCH --output=makani-build-%j.log
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,INVALID_DEPEND,TIME_LIMIT

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?must be run via sbatch from your repo root}"
# site-specific paths (container image, data root): edit site.env once
source "$SLURM_SUBMIT_DIR/site.env"
test -f docker/makani.def || { echo "ERROR: docker/makani.def not found in $(pwd) — submit from your repo root"; exit 1; }

IMG="${MAKANI_IMG:?set MAKANI_IMG in site.env}"

# Assemble the build on node-local RAM (fast + xattr-capable) instead of NFS: this is what
# makes the extraction finish quickly and silences NFS xattr warnings. If /dev/shm is too
# small (the build errors "no space"), point APPTAINER_TMPDIR at fast local or scratch disk.
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/dev/shm/$USER/apptainer_tmp}"
export APPTAINER_CACHEDIR="${MAKANI_APPTAINER_CACHE:-$HOME/.apptainer/cache}"   # downloaded layers cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$(dirname "$IMG")"
trap 'rm -rf "$APPTAINER_TMPDIR"' EXIT                                          # free the RAM afterward

echo "Submit dir : $SLURM_SUBMIT_DIR"
echo "Image out  : $IMG"
echo "Def file   : $(pwd)/docker/makani.def"
echo "Tmp (build): $APPTAINER_TMPDIR"

time apptainer build --fakeroot "$IMG" docker/makani.def
echo "Done: $IMG"
