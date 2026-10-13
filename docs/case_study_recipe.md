# Case-study recipe — retrodicting one event with the backward ensemble

Two ways to work with a case study. **Option A ships the rollout *outputs*** — the recipient
needs a laptop and three pip packages. **Option B ships the model** — they need a GPU cluster,
the container, and a converted ERA5 year. Unless new experiments are planned, Option A is the
better reproducibility unit: one canonical run of the model, rather than two environments that
might silently differ.

---

## What the model does

It runs weather **backward**. Given the atmospheric state at time *t*, it predicts
*t − 6 h*; rolled autoregressively, it walks further into the past. The question is not "can we
forecast" but **how far back the past stays recoverable from the present**.

Forward integration of a dissipative atmosphere destroys small-scale information, so the forward
map is many-to-one and the backward map is **one-to-many**: many plausible pasts fit one present.
A deterministic backward model can only regress to the average of those pasts, which is a blur.
This model is an **ensemble** — it samples plausible pasts instead of averaging them, which is why
the **spread** across members, not just the mean trajectory, is the interesting output.

| | |
|---|---|
| Architecture | FourCastNet 3 (hybrid local+global spherical blocks), `embed_dim` 384, 10 layers |
| Objective | CRPS + spectral CRPS, 8-scale correlated input noise (`config/rewind_fcn3.yaml`) |
| Ensemble | E = 8 members in training; 4 per GPU at inference (see below) |
| State | 72 ERA5 channels (u,v,z,t,q × 13 levels + 7 surface), 0.25°, 6-hourly |
| Training data | 2009–2018 (`rewind_fcn3_ens_10yr_backward`) |
| Direction | backward; one rollout step = **−6 h** |

---

# Option A — ship the rollouts

Every analysis tool in this repo that reads rollout output is torch-free and makani-free
(numpy + h5py + matplotlib only), so a bundle of outputs is self-sufficient.

## A1. Convert the case-study year

The training archive covers **1979–2018 train / 2019 valid / 2020 test** — a 2021 or 2022 case
study needs its year converted first (`$MAKANI_DATA` is set in `site.env`):

```bash
sbatch data_process/convert_casestudy.sbatch 2021        # or: 2021 2022
```
Submit-and-forget on a CPU partition (~20 min and ~0.44 TB per year), writing to
`$MAKANI_DATA/casestudy`, the directory `rewind_fcn3_ens_casestudy` already points at.
`data_process/subset_time_range.py` can cut a year down to a few tens of GB around one event.

## A2. Run the case-study rollouts

`config/rewind_fcn3.yaml` ships `rewind_fcn3_ens_casestudy`, architecturally identical to the
trained model so the checkpoint loads unchanged, differing only in `inf_data_path`.

A small run, dumping the ensemble mean and then one member:

```bash
S=$MAKANI_DATA/runs/rewind_fcn3_ens_casestudy/fcn3_ens_10yr/scores

sbatch analysis/backward_rollout_fcn3.sbatch rewind_fcn3_ens_casestudy fcn3_ens_10yr \
  10 1 4 1 z500,t500,t850,t2m,msl,u500,v500 \
  --anchor_date 2021-06-28T12:00 \
  --output_json $S/rollout_peak.json --fields_file fields_peak.h5 --spectrum_file spectrum_peak.h5

sbatch analysis/backward_rollout_fcn3.sbatch rewind_fcn3_ens_casestudy fcn3_ens_10yr \
  10 1 4 1 z500,t500,t850,t2m,msl,u500,v500 \
  --anchor_date 2021-06-28T12:00 --fields_member 0 \
  --output_json $S/rollout_peak_member0.json --fields_file fields_peak_member0.h5 --spectrum_file spectrum_peak_member0.h5
```

The member run matters: members are **sharp**, the mean is **smooth**, and the contrast between
them is itself a result. The metrics JSON scores **all 72 channels** regardless — the channel
list only limits the raw field dump used for videos.

The large pooled ensemble (250 array tasks × 4 members = 1000 independent members) is one
command; see the header of `analysis/backward_rollout_ensemble.sbatch` for why it is an array
and what does and does not pool across tasks:

```bash
sbatch --array=1-250%32 analysis/backward_rollout_ensemble.sbatch rewind_fcn3_ens_casestudy \
       fcn3_ens_10yr 30 4 2021-06-28T12:00
```

### Choosing the anchor date

The rollout **starts at the anchor and walks into the past**, so the anchor is the *latest*
moment in the output. For the 2021 Pacific Northwest heat dome, the peak (June 28–29) covers
ridge amplification and onset; an onset anchor (June 25) covers the antecedent week but not
the event itself. `--anchor_date` takes the frame at or just before the request, and a bare
date means 00:00 UTC, so give the hour (`2021-06-28T12:00`) when it matters.

## A3. Package and ship

```bash
analysis/package_case_study.sh $S /path/to/ship/pnw2021
rsync -avP /path/to/ship/pnw2021/ <collaborator-host>:<path>/
```

The script copies the rollout outputs, the analysis tools that read them, and generates a
README plus a manifest from the finished bundle.

## A4. What the recipient does

```bash
pip install numpy h5py matplotlib          # the entire setup

python tools/plot_backward_rollout.py --scores_dir data
python tools/make_backward_video.py --fields_file data/fields_peak.h5 --channel z500
```

Plus direct access to the numbers — `data/*.json` is plain JSON, and
`tools/backward_fields_io.py` loads any dumped channel as numpy:

```python
import sys; sys.path.insert(0, "tools")
from backward_fields_io import load_channel_trajectory
traj = load_channel_trajectory("data/fields_peak.h5", "z500", anchor=0)
# traj.pred, traj.truth: (steps, lat, lon); traj.lead_hours: hours INTO THE PAST
```

The pooled 1000-member dumps are read by `analysis/plot_ensemble_horizon.py` and
`analysis/plot_spectral_horizon.py` (both torch-free; their docstrings give the per-case
reduction boxes).

## Reading the numbers

| Metric | How to read it |
|---|---|
| `acc` | anomaly correlation of the retrodicted past against ERA5 truth; higher is better. The "climatology" is `time_means.npy`, one all-years mean map, not a day-of-year climatology |
| `ssr` | spread-skill ratio. **≈ 1 is calibrated**; < 1 is underdispersive (over-confident) |
| `tail_power_ratio` | small-scale power, prediction ÷ truth. **1 is truth**; < 1 is blur (expected backward); **> 1 rising with lead is a numerical artifact**, not physical structure |
| `crps`, `spread`, `rank_histogram` | probabilistic skill, ensemble dispersion, and calibration of the member distribution |

The ensemble **mean is not a physical state** — it is a conditional average over plausible pasts.
For anything about fronts, gradients, or extremes, look at **members**. The per-task ensemble
metrics describe the per-task E (CRPS has a pairwise term, spread an (E−1) correction), so the
distributional numbers for a pooled ensemble are recomputed from the field dumps.

---

# Option B — ship the model

Only if the collaborator needs to run *new* inference themselves (new dates, new channels, new
ensemble sizes) rather than analyze what was produced.

## What to copy

| Item | Location | Size |
|---|---|---|
| Checkpoint | `runs/rewind_fcn3_ens_10yr_backward/<run>/training_checkpoints/best_ckpt_mp0.tar` | ~3 GB |
| Stats — **all 7** `.npy` | `$MAKANI_DATA/stats/` | ~350 MB |
| Invariants | `$MAKANI_DATA/invariants/` | ~10 MB |
| Channel metadata | `data_process/examples/metadata_fcn3_72.json` | KB |
| Code | this repository | — |
| Container | the image built from `docker/makani.def` | ~6.4 GB |
| Converted ERA5 year | — | ~0.44 TB, or a slice via `subset_time_range.py` |

Keep the directory layout: the harness looks for
`<runs>/<config_target>/<run_num>/training_checkpoints/`. Wrong or missing `stats/` produces
plausible-looking fields with meaningless values — it fails silently.

## The environment trap

The model uses **DISCO** spherical convolutions from `torch-harmonics`. A stock
`pip install torch-harmonics` gives a **CPU-only** build of those kernels: it imports fine, builds
fine, starts fine, then dies on the first convolution with

```
NotImplementedError: Could not run 'disco_kernels::forward' with arguments from the 'CUDA' backend
```

Build the image from `docker/makani.def` — it pins `torch-harmonics` to `v0.9.1`, builds the CUDA
kernels, and **fails the build** if `cuda_kernels_is_available()` is false. Verify first:

```bash
apptainer exec --nv makani.sif python -c \
  "import torch, torch_harmonics as th; print(torch.cuda.is_available(), th.disco.optimized.cuda_kernels_is_available())"
# want: True True
```

## GPU memory sets the ensemble size

FCN3's decoder DISCO convolution materializes one large intermediate at full output resolution,
then casts it. Per **folded sample** (folded = batch × members on one GPU):

```
384 embed × 9 basis × 721 × 1440 × 4 B ≈ 13.4 GiB   fp32 buffer
                                       +  6.7 GiB   bf16 cast (coexists with it)
                                       ≈   20 GiB   per folded sample, transient, at inference
```

At inference batch = 1, so folded = E:

| GPU | Safe ensemble size per process |
|---|---|
| B200 / H200 (140–180 GB) | E = 4 (measured; E = 16 OOMs at the first forward) |
| H100 / A100 80 GB | E = 2–3 |
| A100 40 GB, L40S 48 GB | E = 1 (deterministic; loses the spread) |

Members are independent, so N runs with different `--noise_seed` values pool into one larger
ensemble; that is exactly what `analysis/backward_rollout_ensemble.sbatch` does.

## Data preparation

Convert the year as in **A1**, then optionally cut the event window to ship less:

```bash
python data_process/subset_time_range.py \
  --input  .../2021.h5 --output .../casestudy/2021_pnw.h5 \
  --start 2021-06-01 --end 2021-07-05 \
  --check_anchor 2021-06-28 --check_steps 40
```

Slicing is safe because the dataloader reads timestamps from **each file's own HDF5 dimension
scale**, not the filename — a correctly cut file self-describes its dates, so the solar-zenith
forcing stays aligned. The tool enforces the loader's two contracts (contiguous in time; one
non-overlapping slice per directory) and asserts the anchor leaves a full rollout window: R steps
from date D needs **R × 6 h of data before D** in the file.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `NotImplementedError: disco_kernels::forward … CUDA` | CPU-only torch-harmonics |
| `torch.OutOfMemoryError`, one huge allocation | ensemble too large for the GPU |
| `CUDA error 803` / `torch.cuda.is_available()` False inside the container only | stale NGC forward-compat libcuda shadowing a newer host driver; see `docker/makani.def` |
| `time difference between steps is not constant` | non-contiguous data file |
| `files might have overlapping date ranges` | more than one overlapping slice in the data dir |
| Anchor rejected / window error | not enough frames before the anchor |
| Sensible fields, nonsense metrics | wrong or missing `stats/` |
