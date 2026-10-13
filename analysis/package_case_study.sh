#!/bin/bash
# ============================================================================
# Assemble a SELF-CONTAINED case-study bundle for a collaborator: the rollout
# OUTPUTS plus the analysis tools that read them, and nothing else.
#
# Why this exists: handing over the trained model means handing over the container
# (CUDA DISCO kernels), the checkpoint + stats + invariants, a GPU large enough for
# the decoder buffer, and a converted ERA5 year (~0.44 TB). Handing over the
# ROLLOUTS instead removes all of it — every analysis tool in this repo that reads
# rollout output is torch-free and makani-free (numpy + h5py + matplotlib only), so
# the recipient needs a laptop and three pip packages.
#
# The bundle is also the reproducibility unit: one canonical run of the model, not
# two environments that might silently differ.
#
# Usage:
#   analysis/package_case_study.sh <scores_dir> <out_dir> [extra files...]
#
#   # after running the case-study rollouts:
#   analysis/package_case_study.sh \
#     /work/.../runs/rewind_fcn3_ens_casestudy/fcn3_ens_10yr/scores \
#     /work/.../ship/pnw2021
#
# Then ship <out_dir> (rsync/Globus/tar). Re-running after a better checkpoint
# refreshes the same bundle in one command.
# ============================================================================
set -euo pipefail

SCORES="${1:?usage: package_case_study.sh <scores_dir> <out_dir> [extra files...]}"
OUT="${2:?usage: package_case_study.sh <scores_dir> <out_dir> [extra files...]}"
shift 2

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -d "$SCORES" ] || { echo "ERROR: scores dir not found: $SCORES" >&2; exit 1; }

mkdir -p "$OUT/data" "$OUT/tools"

# ---- 1. rollout outputs (json + spectra + field dumps; ci/ deliberately excluded) ----
found=0
for f in "$SCORES"/*.json "$SCORES"/*.h5; do
  [ -e "$f" ] || continue
  cp -v "$f" "$OUT/data/"
  found=$((found + 1))
done
[ "$found" -gt 0 ] || { echo "ERROR: no .json/.h5 outputs in $SCORES — run the rollout first" >&2; exit 1; }

# ---- 2. the tools that read them (verified torch-free / makani-free) ----------------
# backward_fields_io.py is REQUIRED by make_backward_video.py (shared schema module) —
# shipping the video tool without it is the classic broken-bundle failure.
for t in plot_backward_rollout.py make_backward_video.py backward_fields_io.py \
         compare_backward_rollouts.py rollout_ci_trend.py; do
  cp -v "$REPO/analysis/$t" "$OUT/tools/"
done

# ---- 3. anything else the caller wants alongside (docs, notes, figures) --------------
for extra in "$@"; do
  [ -e "$extra" ] && cp -rv "$extra" "$OUT/"
done

# ---- 4. README, then the manifest LAST so it lists a complete bundle ------------------
cat > "$OUT/README.md" <<'EOF'
# Backward (retrodiction) rollout bundle

Output of a **backward** weather emulator: an ensemble model that predicts the atmospheric
state *going into the past* (each rollout step = −6 h). You do not need the model, a GPU, or
any ERA5 download to work with this — everything here is already computed.

## Setup (the whole thing)

```bash
pip install numpy h5py matplotlib
```

`ffmpeg` is optional; without it the video tool writes an animated GIF instead of MP4.

## Files

* `data/*.json` — per channel × lead: `acc`, `rmse`, `rmse_over_clim`, `tail_power_ratio`
  (scored on the ensemble **mean**), plus `crps`, `spread`, `ssr`, `rank_histogram`,
  `tail_power_ratio_members`. Plain JSON — load it in anything.
* `data/*spectrum*.h5` — power spectra, columns `[truth, ensemble mean, member 1…E]`.
* `data/*fields*.h5` — raw predicted and true fields, frame by frame.
* `tools/` — the analysis scripts below.
* `MANIFEST.txt` — what is in here and how big.

## Figures

```bash
python tools/plot_backward_rollout.py --scores_dir data
```
Writes skill/spread/CRPS/rank-histogram/spectrum figures next to the data.

## Animation

```bash
python tools/make_backward_video.py --fields_file data/<fields file>.h5 --channel z500
```
Three panels — ERA5 truth | backward prediction | error — one frame per −6 h step. Add
`--reverse` to play chronologically (oldest → newest) instead of in rollout order.

## Your own analysis

`tools/backward_fields_io.py` loads any dumped channel as a plain numpy array:

```python
import sys; sys.path.insert(0, "tools")
from backward_fields_io import load_channel_trajectory
traj = load_channel_trajectory("data/<fields file>.h5", "z500", anchor=0)
# traj.pred, traj.truth: (steps, lat, lon); traj.lead_hours: hours INTO THE PAST
```

## Reading the numbers

* `acc` — anomaly correlation of the retrodicted past against ERA5 truth; higher is better.
* `ssr` — spread-skill ratio. **≈ 1 is calibrated**; < 1 means the ensemble is over-confident.
* `tail_power_ratio` — small-scale power, prediction ÷ truth. **1 is truth**; < 1 is blur
  (expected when running backward); **> 1 and rising with lead is a numerical artifact**, not
  physical structure — see the limitations note shipped with this bundle.

The ensemble **mean** is a conditional average over plausible pasts, not a physical state. For
anything about fronts, gradients, or extremes, look at individual **members**.
EOF

# generated last, and skipping itself, so it reflects the finished bundle
( cd "$OUT" && find . -type f ! -name MANIFEST.txt | sort | while read -r f; do
    printf '%10s  %s\n' "$(du -h "$f" | cut -f1)" "${f#./}"
  done ) > "$OUT/MANIFEST.txt"

echo
echo "=== bundle ready: $OUT"
du -sh "$OUT"
echo "    ship it:  rsync -avP $OUT/ <host>:<path>/"
