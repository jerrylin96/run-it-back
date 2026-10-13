# Upstream issues — defects inherited from NVIDIA makani, fixed on this fork

Every entry here is a defect in code we **inherited** from
[NVIDIA/makani](https://github.com/NVIDIA/makani) (fork point `ac26308`), not code this project
wrote. Each was confirmed by reading the code (and, where the dependency was available,
reproduced) before it was touched. The file exists so these can be offered upstream as PRs, or
simply used as a reference for why this fork diverges from upstream in places that have nothing
to do with backward-in-time retrodiction. None of them affected the results in the paper: each
was either in a code path this project never executed, or guarded off by a configuration choice.

### 1. Leaked loop variable in `concatenate_dataset.py`

**File:** `data_process/concatenate_dataset.py` (timestamp-extraction loop)
**Severity:** high when it applies; latent otherwise.

The validation loop `for input_dir in input_dirs:` leaves `input_dir` bound to `input_dirs[-1]`.
The timestamp-extraction loop below then opens `os.path.join(input_dir, fname)` while the adjacent
layout/metadata path correctly uses `input_dirs[0]`. Timestamps are therefore read from the **last**
input directory and attached to a layout derived from the **first**. Silent whenever the
directories have equal file counts — which they must, since they are concatenated along the
channel dimension. Invisible with a single input directory.

**Fix:** use `input_dirs[0]`, matching the neighbouring metadata read.

### 2. Unsorted glob in multi-rank MPI conversion

**File:** `data_process/convert_makani_output_to_wb2.py` (`main`)
**Severity:** high when it applies.

`glob()` returns entries in filesystem order, which is not guaranteed stable or identical across
processes. Each MPI rank globs independently and then writes Zarr regions indexed by position in
that list, so ranks disagreeing about the ordering write each other's regions — silent cross-file
corruption — or disagree about the count and hang.

**Fix:** `sorted(glob(...))`.

### 3. Cartesian-product level lookup crashes on non-rectangular channel sets

**Files:** `data_process/convert_wb2_to_makani_input.py`, `data_process/convert_makani_output_to_wb2.py`
**Severity:** medium — a bad error message, not a wrong answer.

The atmospheric-level list is the **union** of levels across variables, and the converters then take
the full cartesian product `variable × level`. A channel set like `z500 + t850` queries `z850`, and
`channel_names.index()` raises a bare `ValueError: 'z850' is not in list`. makani genuinely requires
a rectangular grid here (`get_channel_groups`), so refusing is correct — the defect is that the
refusal does not say what rule was broken.

**Fix:** guard the lookup and raise naming the rectangular-grid requirement and the offending pair.

### 4. Micro-mode ACC omits the epsilon the macro path has

**File:** `makani/utils/metrics/functions.py` (`GeometricACC.finalize`)
**Severity:** medium, latent (`method` defaults to `"macro"`).

`finalize` divides by `torch.sqrt(vals[1] * vals[2])` with no `+ self.eps`, while the macro branch
inherits a guarded division. A constant or fully-masked region gives a zero denominator and the
metric returns NaN rather than a score.

**Fix:** `vals[..., 0] / (torch.sqrt(vals[..., 1] * vals[..., 2]) + self.eps)`.

### 5. `best_valid_loss` resets on every resume in the secondary trainers

**Files:** `makani/utils/training/stochastic_trainer.py`, `makani/utils/training/autoencoder_trainer.py`
(and, before this fork fixed them the same way, `deterministic_trainer.py` and `ensemble_trainer.py`)
**Severity:** high for anyone resume-chaining these trainers.

All four trainers initialised a **local** `best_valid_loss = 1.0e6` at the top of `train()`. On a
resume that value is reconstructed from nothing, so the first post-resume epoch compares against
`1e6`, wins, and overwrites `best_checkpoint_path` — replacing the true best model with whatever the
first epoch after the resume happened to produce. On a multi-day run split across the scheduler's
time limit, the saved "best" checkpoint is then simply the last resume's first epoch.

**Fix:** persist `best_valid_loss` in the checkpoint counters (`Driver._restore_checkpoint_*` and
`Driver.save_checkpoint` already handle it generically), read it into `self.best_valid_loss` beside
the other counters, and update it **before** the counters are assembled for the checkpoint (an
update inside the best-checkpoint save block records a one-epoch-stale best). The deterministic and
ensemble trainers additionally persist the early-stopping and divergence-guard counters.

**Locked by a test.** `tests/test_resume_best_checkpoint.py` runs the second leg that
`tests/test_trainers.py` never runs — for all four trainers — and asserts `best_ckpt` moves only on
a genuine improvement, with a control that reinstates the `1e6` reset and shows the clobber return.

### 6. Skipped channels leave unwritten zeros marked valid

**File:** `data_process/convert_wb2_to_makani_input.py`
**Severity:** high when `--skip_missing_channels` is used.

`valid_data` is initialised to all-ones. When `skip_missing_channels` fires, the channel's slab is
never written and keeps the file's initialised **zeros**, but its `valid_data` column stays `1`.
Downstream loaders and `get_stats.py` then treat a column of exactly `0.0` as real observations —
worse than the NaN path it sits next to, since a channel of exact zeros looks like plausible
normalized data.

**Fix:** on skip, set `valid_data = 0` **and write NaN into the channel's slab**, in both the surface
and atmospheric branches. The flag alone is not enough: `get_stats.py` masks through
`data_process_helpers.mask_data()`, which keys on NaN and never reads `valid_data`.

### 7. Empty dynamical-channel set reported as INSTABILITY — **ours, not upstream**

**File:** `analysis/backward_rollout.py`
**Severity:** medium.

The rollout signature is computed over `["z500", "t500", "u500", "v500", "msl"]` filtered to what the
dataset has. A channel subset containing none of them leaves the selection empty, the guarded
`float("nan")` is passed to `_signature`, and `_signature` treats non-finite input as divergence —
so a perfectly healthy rollout is labelled `INSTABILITY`. Recorded here because it was found in the
same audit pass.

**Fix:** fall back to scoring all channels, and print that the fallback happened.

### 8. `GeometricSpread` divides by zero at E=1 — **withdrawn, not a bug**

**File:** `makani/utils/metrics/functions.py` (`GeometricSpread.forward`, `GeometricSSR.forward`)
**Status:** investigated, refuted, no change made. Do not file upstream.

At `ens_size == 1` the **numerator is identically zero**: `predictions = sum(forecasts, dim=1) / 1.0`
is bitwise equal to `forecasts[:, 0]`, so `predictions.unsqueeze(1) - forecasts` is exactly `0.0` and
the sum of squares is `0`. The expression is therefore `0 / 0`, which is **NaN** — already the
correct representation of an undefined spread. "Divides by (E-1)" is not sufficient grounds to call
something a defect; the numerator has to be checked too.

### 9. `one_hot` land-sea mask infers its class count from the data

**File:** `makani/models/preprocessor_helpers.py` (both the `floor` and `round` branches)
**Severity:** low, latent.

`torch.nn.functional.one_hot(lsm)` with no `num_classes` infers the count from the values present, so
an all-sea (or all-land) mask yields **1** channel where the model expects 2. Harmless on a global
grid, and a wrong count crashes at the first forward rather than corrupting quietly.

**Fix:** pass `num_classes=2` at both sites.

### 10. The same `one_hot` defect at a third site — soil type, where the count is contractual

**File:** `makani/models/preprocessor_helpers.py` (`add_soiltype` branch)
**Severity:** medium when it applies.

The `add_soiltype` block does `torch.nn.functional.one_hot(st)` with no `num_classes` — identical to
#9, twenty lines below the two sites #9 fixed. Here the count is a **contract**: `driver.py` does
`params.N_static_channels += 8` unconditionally, so a crop whose soil-type field happens to contain
only 6 of the 8 categories produces 6 static channels against an `N_in_channels` computed for 8.

**Fix:** `num_classes=8`, matching `driver.py`.

### 11. `landmask_preprocessing` has no `else`, so a bad value raises `UnboundLocalError`

**File:** `makani/models/preprocessor_helpers.py` (`add_landmask` branch)
**Severity:** low; a diagnosability defect rather than a wrong answer.

The `floor` / `round` / `raw` chain has no `else`. Any other value leaves `lsm` unbound, and execution
falls through to the shard slice below: `UnboundLocalError: local variable 'lsm' referenced before
assignment`, which names neither the config key responsible nor the legal values.

**Fix:** an `else` raising `ValueError` naming the key and the three legal values.

### 12. FCN3 `n_history >= 2` re-offsets the dynamic auxiliary channels

**File:** `makani/models/networks/fourcastnet3.py` (`_precompute_channel_groups`)
**Severity:** low; loud (IndexError at the first forward), not silent. Inert for `n_history <= 1`.

The history loop builds per-frame index lists. Three of its four lines iterate the frame-0 lists;
the fourth, `dyn_aux_chans += [(c + ih*n_dyn_chans) for c in dyn_aux_chans]`, iterates the list it is
extending. At `ih=2` it also sees the already-offset frame-1 entries and offsets them again: with 9
dynamic aux channels (zenith + 8 noise) and `n_history=2` that is 39 aux indices where
`n_aux_chans` counts 30, the largest past the input width.

**Fix:** snapshot the per-frame list before the loop and iterate that, as the sibling lines do.
Locked by `tests/test_fcn3_history.py`.
