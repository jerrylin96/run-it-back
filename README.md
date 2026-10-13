# Run It Back: a backward-in-time ensemble weather emulator

Code for **[PAPER TITLE]** ([AUTHORS], NeurIPS 2026 Workshop on Tackling Climate Change with
Machine Learning). <!-- TODO: title, authors, paper link -->

[![tests](https://github.com/jerrylin96/run-it-back/actions/workflows/tests.yml/badge.svg)](https://github.com/jerrylin96/run-it-back/actions/workflows/tests.yml)

This repository is a fork of [NVIDIA/makani](https://github.com/NVIDIA/makani) (fork point
`ac26308`) that turns FourCastNet 3 into a **retrodiction** model: given the atmospheric state at
time *t*, it predicts the state at *t − 6 h*, and rolled autoregressively it walks into the past.
Because forward integration of a dissipative atmosphere destroys small-scale information, the
backward map is one-to-many — many plausible pasts fit one present — so the model is trained as a
**CRPS ensemble** that samples plausible pasts instead of regressing to their average.

The code here is the state of our development branch at the time of the paper (snapshot of
2026-08-25), with two later bug fixes carried in (see *Corrections after the snapshot*).

## What changed relative to upstream makani

Everything upstream still works as documented in [docs/MAKANI_README.md](docs/MAKANI_README.md).
The fork adds:

- **The backward task as a dataset flag.** `reverse_time: True` in the multifiles dataloader
  (`makani/utils/dataloaders/data_loader_multifiles.py`) reverses the window's time offsets
  before loading, so the input is *X(t)* and the target *X(t − Δ)*, each read with its own
  normalization and its own timestamp's solar zenith angle. Toggleable forward/backward from one
  key; architecture-agnostic; no changes to any network. Locked by
  `tests/test_dataloader.py::test_reverse_time`. Requires `multifiles: True` (the DALI loader does
  not implement it and the code refuses to train silently forward).
- **Training hardening** in the deterministic and ensemble trainers: `best_valid_loss` and the
  guard counters are persisted in the checkpoint (a resumed run can no longer overwrite the best
  checkpoint with a worse epoch), early stopping with a minimum delta, a relative divergence
  guard, and a skip-step guard that rejects an optimizer step whose pre-clip gradient norm is
  non-finite or above a threshold (DDP-safe via an explicit collective). All opt-in via config.
- **The evaluation harness** in `analysis/`: a backward autoregressive rollout with spectra,
  area-weighted RMSE/ACC against persistence and climatology ceilings, a small-scale
  tail-power ratio, and (ensemble mode) CRPS, spread, spread–skill ratio and rank histograms;
  per-member field dumps; a pooled large-ensemble launcher; the case-study horizon diagnostics;
  video rendering; a cross-run comparison tool; training-log diagnostics.
- **The data pipeline** for the full 72-channel FCN3 state from ARCO-ERA5 at 0.25° / 6-hourly
  (conversion, orography and land-sea-mask invariants, normalization statistics with
  acceptance checks, case-study years, time-range subsetting).
- **An Apptainer recipe** (`docker/makani.def`) that builds torch-harmonics with CUDA DISCO
  kernels and fails the build if they are missing.
- Fixes to defects inherited from upstream, documented in [UPSTREAM_ISSUES.md](UPSTREAM_ISSUES.md).

## The model

| | |
|---|---|
| Architecture | FourCastNet 3 (`nettype: FCN3`), `embed_dim` 384, 10 layers, latent 360×720 |
| Objective | CRPS + 0.1 × spectral CRPS, 8-scale correlated input noise (upstream's FCN3 recipe) |
| Ensemble | 8 members in training |
| State | 72 ERA5 channels: u, v, z, t, q × 13 pressure levels + u10m, v10m, u100m, v100m, t2m, msl, tcwv |
| Grid / step | 0.25° (721×1440), 6-hourly; one step = −6 h |
| Data | ERA5 via ARCO; train 2009–2018, validation 2019 |
| Config | `config/rewind_fcn3.yaml`, target `rewind_fcn3_ens_10yr_backward` |

## Companion study

The case studies (the 2021 Pacific Northwest heat dome and Hurricane Ian) and their analysis
regions follow the companion paper *Extremes on Rewind: Generating 1,000-Member Ensembles
Initialized at a Final Condition* (Lin, Chien, Sakarvadia and Barnes, 2026,
[arXiv:2608.19008](https://arxiv.org/abs/2608.19008)), which generates ensembles conditioned on
a final state with cBottle-video. Its code and case definitions are at
[jerrylin96/rewinding-the-extremes](https://github.com/jerrylin96/rewinding-the-extremes)
(archived at [10.5281/zenodo.22000716](https://doi.org/10.5281/zenodo.22000716)). The horizon
diagnostics here take their per-(case, variable) reduction boxes verbatim from that repository's
`scripts/_shared/cases/*.yaml`, so the two studies reduce over the same pixels, and report
geopotential as height in metres by the same convention.

## Reproducing the pipeline

All launchers are SLURM batch scripts that run inside the container. Set the two paths in
[`site.env`](site.env) first (the image, and the data root, which is bind-mounted into the
container at the same path). Partition names in the scripts are our cluster's; override with
`sbatch -p ... --gres=...`.

1. **Container.** `sbatch docker/build_makani.sbatch` builds `docker/makani.def`
   (NGC PyTorch base, parallel HDF5, torch-harmonics v0.9.1 with CUDA kernels, physicsnemo
   v1.3.0, this package editable at `/opt/makani`). Day-to-day, your checkout is bind-mounted
   over `/opt/makani`, so code edits need no rebuild; `./crun <command>` runs any command in
   the container on a compute node.
2. **Data.** `sbatch data_process/net_probe.sbatch` (can the partition reach GCS?), then
   `sbatch data_process/convert_era5.sbatch` (ARCO-ERA5 → one HDF5 file per year; ~0.44 TB per
   year, resumable), `sbatch data_process/invariants.sbatch` (orography + land-sea mask) and
   `sbatch data_process/stats_era5.sbatch` (normalization statistics over the 40 training years,
   with acceptance checks). Make `$MAKANI_DATA/train_10yr` a directory of the 2009–2018 files
   (symlinks are fine).
3. **Train.** Two smokes on 2 GPUs, then the run on 8:
   ```bash
   sbatch --gres=gpu:2 train_fcn3.sbatch rewind_fcn3_det_smoke det_smoke1
   sbatch --gres=gpu:2 train_fcn3.sbatch rewind_fcn3_ens_smoke ens_smoke1
   sbatch train_fcn3.sbatch rewind_fcn3_ens_10yr_backward fcn3_ens_10yr --ensemble_parallel_size=2 --checkpointing_level=3
   ```
   Resume across the time limit by re-submitting the same arguments
   (`--dependency=afterany:<jobid>`). Each link arms a rollout snapshot
   (`analysis/rollout_ci_fcn3.sbatch`) so multistep skill is tracked while training runs.
4. **Verify the direction** (optional): `analysis/verify_direction.py` feeds the present through
   the checkpoint and correlates the predicted increment with the true past and future
   increments, which point opposite ways; a backward model scores `r_past ≫ r_future`.
5. **Roll out.** `sbatch analysis/backward_rollout_fcn3.sbatch rewind_fcn3_ens_10yr_backward fcn3_ens_10yr`
   writes `backward_rollout.json`, `backward_spectrum.h5` and a field dump under the run's
   `scores/`; `sbatch analysis/plot_backward_rollout.sbatch <scores_dir>` draws the figures and
   `sbatch analysis/make_backward_video.sbatch <fields.h5> <channel>` renders an animation.
6. **Case studies and the 1000-member ensembles.** [docs/case_study_recipe.md](docs/case_study_recipe.md):
   convert the event's year, roll out from the event date with the evaluation-only
   `rewind_fcn3_ens_casestudy` target, pool 250 seeded array tasks with
   `analysis/backward_rollout_ensemble.sbatch`, then run the horizon diagnostics
   (`analysis/plot_ensemble_horizon.py`, `analysis/plot_spectral_horizon.py`).

A note on memory: FCN3's decoder DISCO convolution allocates ~13.4 GiB of fp32 per folded sample
(batch × members on one GPU), so a 178 GiB GPU holds 4 folded samples. That sets the training
layout above and the 4-member-per-task inference; larger ensembles are pooled across seeds.

## Repository layout

```
config/rewind_fcn3.yaml          the backward FCN3 targets (smokes, the trained model, the case-study twin)
site.env                         the two site-specific paths every launcher reads
train_fcn3.sbatch                torchrun DDP training (makani.ensemble / makani.train)
crun                             run one command in the container on a compute node
makani/                          the library (upstream + the changes listed above)
analysis/                        rollout harness, ensemble launcher, diagnostics, plots, videos
data_process/                    ARCO-ERA5 conversion, invariants, statistics, case-study years
docker/                          Apptainer definition + build job
docs/                            case-study recipe; the upstream makani README
tests/                           unit tests (CI runs them on every push)
```

## Tests

`pytest tests` (see `.github/workflows/tests.yml` for the pinned stack: torch 2.7.1,
torch-harmonics v0.9.1, physicsnemo v1.3.0). On the cluster,
`sbatch analysis/run_tests.sbatch tests ""` runs the suite inside the container. The tests this
fork adds are `test_dataloader.py::test_reverse_time*`, `test_training_helpers.py`,
`test_resume_best_checkpoint.py`, `test_configs.py`, `test_ensemble_horizon.py`,
`test_backward_fields_io.py`, `test_metric_fixes.py`, `test_backward_rollout_metrics.py` and
`test_fcn3_history.py`.

## Corrections after the snapshot

Two fixes made after the snapshot date are included because they do not change anything the
paper reports:

- **Humidity ACC.** `analysis/backward_rollout.py` and `analysis/verify_direction.py` added an
  absolute 1e-12 under the square root of the correlation denominator. ACC is unit-free and the
  guard is not: in kg/kg the variance product of specific humidity is ~1e-12 at 500 hPa and far
  smaller aloft, so q500's ACC read ~0.7× its value and q200–q50 read 0.000. Removed; other
  channels and all RMSEs are unaffected (`tests/test_backward_rollout_metrics.py`).
- **FCN3 with `n_history >= 2`** raised an IndexError at the first forward
  (`UPSTREAM_ISSUES.md` #12). Inert for every configuration here.

One known limitation is left as is: `analysis/rollout_ci_fcn3.sbatch` scores the checkpoint in
slot `ckpt_mp0_v0.tar`, which with a checkpoint ring deeper than one is not necessarily the newest
epoch (the snapshot is keyed by the epoch stored in the file, so nothing is mislabeled).

## Climatology convention

"Climatology" in the rollout harness and in trainer validation is `time_means.npy`, a single
all-years mean map (upstream makani's default), not a day-of-year climatology. ACC therefore reads
higher, and RMSE-over-climatology lower, than under WeatherBench2 conventions, by an amount that
grows with lead. Comparisons between runs sharing the same statistics are unaffected.

## License and attribution

Apache-2.0, as upstream makani (see [LICENSE](LICENSE)); the original NVIDIA copyright and
SPDX headers are retained. Makani was developed by NVIDIA and NERSC; please also cite the
FourCastNet 3 paper when using this code:

```bibtex
@misc{bonev2025fourcastnet3,
      title={FourCastNet 3: A geometric approach to probabilistic machine-learning weather forecasting at scale},
      author={Boris Bonev and Thorsten Kurth and Ankur Mahesh and Mauro Bisson and Jean Kossaifi and Karthik Kashinath and Anima Anandkumar and William D. Collins and Michael S. Pritchard and Alexander Keller},
      year={2025},
      eprint={2507.12144},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.12144},
}
```

## Citation

The companion study:

```bibtex
@misc{lin2026extremesonrewind,
      title={Extremes on Rewind: Generating 1,000-Member Ensembles Initialized at a Final Condition},
      author={Jerry Lin and Mu-Ting Chien and Mansi Sakarvadia and Elizabeth A. Barnes},
      year={2026},
      eprint={2608.19008},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2608.19008},
}
```

This paper:

<!-- TODO: replace with the workshop paper's bibtex -->
```bibtex
@inproceedings{runitback2026,
  title     = {[PAPER TITLE]},
  author    = {[AUTHORS]},
  booktitle = {NeurIPS 2026 Workshop on Tackling Climate Change with Machine Learning},
  year      = {2026},
}
```
