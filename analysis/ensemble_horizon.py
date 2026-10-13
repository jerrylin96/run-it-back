#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Shared core for the 1000-member case-study instability/horizon diagnostics.

Why this module exists: the three diagnostics in plot_ensemble_horizon.py all pool the SAME
250 per-seed field dumps, censor members the SAME way, and reduce over the SAME domain.
Writing that three times is three chances for the heat-dome and hurricane figures to
disagree about which pixels were averaged or which members survived -- and the result is
a CONTRAST between those two cases, so a drift between them is not a cosmetic bug, it is
the result. One core, two thin CLIs (plot_ensemble_horizon.py, plot_spectral_horizon.py).

Torch-free: numpy + stdlib only (h5py lazily, matplotlib never). Runs on the cpu
partition via crun.

WHAT IS AND IS NOT POOLABLE (the trap this module exists to route around)
------------------------------------------------------------------------
The 1000 members are 250 independent array tasks at E=4 (backward_rollout_ensemble.sbatch).
Their per-task ``backward_rollout_*.json`` metrics describe **E=4, not E=1000** -- CRPS
carries a pairwise term and spread an (E-1) correction, so neither averages across tasks
into its 1000-member value. Nothing here reads those JSONs. Every distributional number
is recomputed from the FIELD dumps, where per-member behaviour pools exactly.

CENSORING (the settled reporting methodology; do not invent a threshold here)
----------------------------------------------------------------------------
The ensemble MEAN is not robust -- one member at 1e8 K destroys a 1000-member mean -- so
every statistic past divergence onset is over SURVIVORS. A member is censored from its
own first failing lead onward (failure is made MONOTONE: a trajectory that has blown up
does not un-blow-up, and a transient dip back under a ceiling is not a recovery). Two
detectors, both thresholded, following analysis/compare_backward_rollouts.py's precedent
of masking blow-up rather than clipping an axis:

  * ``tail``  -- the member's high-l spectral power ratio P_pred/P_truth exceeds a ceiling.
                 Fires on the anti-diffusive MECHANISM, so it fires much earlier than a
                 range check. Needs the per-seed backward_spectrum_*.h5 dumps.
  * ``range`` -- the member's field leaves a physical interval, or goes non-finite. The
                 crude backstop; always available from the field dump alone.

``either`` censors at whichever fires first. The default is ``auto``: ``tail`` when
spectra were supplied, else ``range``.

Survivorship is bias, so the survival fraction is returned alongside every survivor
statistic and the callers plot it as its own panel rather than a footnote.

PATHS ARE ALWAYS ARGUMENTS
--------------------------
Two filename schemes are supported -- an early 22-channel PNW dump uses pre-tag
``backward_fields_seed%03d.h5`` and the current 32-channel dumps use case-tagged
``backward_fields_<YYYYmmddTHHMM>_seed%03d.h5``. Nothing here hardcodes either. The
early-dump folder may be removed at any time; this module opens
files "r" and never writes into an input directory.
"""

import glob as _glob
import math
import os
import re

import numpy as np

# --- divergence ceilings ----------------------------------------------------
# Deliberately the SAME numbers analysis/compare_backward_rollouts.py uses for its
# stability scan, so "diverged" means one thing across the project. Bounded blur
# saturates at tail ratio ~O(1-2) (the prototype's stable z500 sat at 1.08), so 10 trips
# only on real anti-diffusive blow-up.
TAIL_CEIL_DEFAULT = 10.0
# t2m in K. ERA5's global extremes over the satellite era sit inside ~180-335 K, so this
# interval cannot censor a physically plausible retrodiction; it catches numerical
# divergence only. Callers override per channel.
RANGE_DEFAULT = {"t2m": (150.0, 350.0)}

# ---------------------------------------------------------------------------
# display units
# ---------------------------------------------------------------------------
# Standard gravity (WMO), m/s^2. The one magic conversion factor in this
# codebase, and deliberately the SAME one the companion study uses (Lin et al. 2026, arXiv:2608.19008; github.com/jerrylin96/rewinding-the-extremes,
# scripts/_shared/var_metadata.py: DISPLAY_GRAVITY), so a geopotential number of
# ours can be set beside one of theirs.
DISPLAY_GRAVITY = 9.80665

_GEOPOTENTIAL_RE = re.compile(r"^z(\d+)?$")

# Storage units, i.e. what the dumps actually contain. display_unit() applies
# the geopotential conversion on top of this.
_STORAGE_UNITS = {
    "t2m": "K", "t500": "K", "t850": "K",
    "z500": "m^2/s^2", "z850": "m^2/s^2", "z": "m^2/s^2",
    "msl": "Pa", "sp": "Pa",
    "u10m": "m/s", "v10m": "m/s", "u100m": "m/s", "v100m": "m/s",
    "u500": "m/s", "v500": "m/s",
    "tcwv": "kg/m^2",
}


def is_geopotential(channel):
    """True for geopotential variables (``z``, ``z500``, ``z850``, ...).

    Mirrors the companion study's is_geopotential (rewinding-the-extremes) so the two codebases agree on
    which fields get converted.
    """
    return bool(_GEOPOTENTIAL_RE.match(str(channel)))


def to_display_units(channel, field, power=1):
    """Scale a raw field into the units ``display_unit`` names.

    ERA5 stores geopotential in m^2/s^2; domain scientists read geopotential
    HEIGHT in m, which is what the companion study (arXiv:2608.19008) publishes, so a plot of ours
    beside a plot of theirs should carry the same numbers. Division by standard
    gravity is the only conversion here.

    ``power`` is the exponent of the field the quantity represents: a spread or
    an RMSE is power 1, a variance or a power spectrum is power 2 (g**2).

    Ratios are invariant and must NOT be passed through this -- ACC, spread/skill,
    the high-l tail power ratio, the banded spectral ratios, and the e-folding
    RATE all divide the factor out exactly. Neither may physics that genuinely
    needs geopotential (geostrophic wind); that must use the raw m^2/s^2 field.
    """
    if is_geopotential(channel):
        return field / DISPLAY_GRAVITY ** power
    return field


def storage_unit(channel):
    """Unit string for the RAW field as the dumps store it (geopotential unconverted)."""
    return _STORAGE_UNITS.get(str(channel), "")


def display_unit(channel):
    """Unit string matching what ``to_display_units`` produces. Empty when the
    channel is not in the table, which is honest -- a guessed unit on a paper
    figure is not."""
    if is_geopotential(channel):
        return "m"
    return _STORAGE_UNITS.get(str(channel), "")

_SEED_RE = re.compile(r"seed(\d+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# discovery + provenance
# ---------------------------------------------------------------------------
def parse_seed(path):
    """Extract the noise seed from a dump filename. Both schemes end in ``seed<NNN>.h5``,
    so the LAST match wins -- a case tag like ``20210628T1200`` carries no 'seed' token,
    but a directory in the path might."""
    hits = _SEED_RE.findall(os.path.basename(path))
    if not hits:
        raise ValueError(f"no 'seed<N>' token in filename: {os.path.basename(path)}")
    return int(hits[-1])


def discover(pattern, expect=None, max_files=None):
    """Resolve a glob to a seed-ordered file list. Fails loud on an empty match and on a
    duplicated seed (two files claiming the same seed are the same draw counted twice --
    silently inflating the apparent ensemble size, which is exactly the failure mode the
    launcher's seed-0 rejection exists to prevent)."""
    paths = sorted(_glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no files matched glob: {pattern}")
    seen = {}
    for p in paths:
        s = parse_seed(p)
        if s in seen:
            raise SystemExit(f"duplicate seed {s} in glob {pattern}:\n  {seen[s]}\n  {p}")
        seen[s] = p
    out = [(s, seen[s]) for s in sorted(seen)]
    if expect is not None and len(out) != int(expect):
        raise SystemExit(f"glob {pattern} matched {len(out)} files, expected {expect}")
    if max_files is not None:
        out = out[: int(max_files)]
    return out


def read_attrs(path):
    """File-level HDF5 attrs only -- no field slab is touched.

    Exists so a pool can be provenance-checked BEFORE the multi-hour read: everything
    check_pool_provenance needs lives in the attrs, and a seed-0 or duplicate-seed file
    should abort in seconds, not after 5 TB of I/O."""
    import h5py

    with h5py.File(path, "r") as f:
        return dict(f.attrs)


def check_pool_provenance(attrs_by_seed, strict=True):
    """Verify the pool is a real pool: distinct recorded noise seeds, every dump keeping
    ALL its members, and one consistent ensemble size.

    ``attrs_by_seed``: {filename_seed: dict(file attrs)}. Returns a report dict; raises on
    a violation when ``strict``. This is what makes "1000 independent members" checkable
    rather than asserted -- members differ ONLY through the input noise, so identical
    seeds would mean identical trajectories.
    """
    problems, recorded, sizes, kept = [], {}, set(), set()
    for fseed, a in sorted(attrs_by_seed.items()):
        def _s(k):
            v = a.get(k, "")
            return v.decode() if isinstance(v, bytes) else str(v)

        ns = _s("noise_seed")
        try:
            ns_i = int(ns)
        except ValueError:
            problems.append(f"seed {fseed}: unparseable noise_seed attr {ns!r}")
            continue
        if ns_i == 0:
            # --noise_seed 0 means "do not reseed": two such tasks emit IDENTICAL members.
            problems.append(f"seed {fseed}: noise_seed == 0 (not reseeded -- members duplicate another task's)")
        if ns_i in recorded:
            problems.append(f"noise_seed {ns_i} used by BOTH file-seeds {recorded[ns_i]} and {fseed}")
        recorded[ns_i] = fseed
        if ns_i != fseed:
            problems.append(f"seed {fseed}: filename says {fseed} but attrs say noise_seed={ns_i}")
        src = _s("pred_source")
        if src and src != "all_members":
            problems.append(f"seed {fseed}: pred_source={src!r}, not 'all_members' -- this dump is not a member pool")
        if _s("ensemble_size"):
            sizes.add(_s("ensemble_size"))
        if _s("n_members_dumped"):
            kept.add(_s("n_members_dumped"))
    if len(sizes) > 1:
        problems.append(f"mixed ensemble_size across the pool: {sorted(sizes)}")
    if len(kept) > 1:
        problems.append(f"mixed n_members_dumped across the pool: {sorted(kept)}")
    report = {
        "n_files": len(attrs_by_seed),
        "n_distinct_noise_seeds": len(recorded),
        "ensemble_size": sorted(sizes),
        "n_members_dumped": sorted(kept),
        "problems": problems,
    }
    if problems and strict:
        raise SystemExit("pool provenance check FAILED:\n  " + "\n  ".join(problems))
    return report


# ---------------------------------------------------------------------------
# domain reduction
# ---------------------------------------------------------------------------
def _wrap360(x):
    return np.mod(np.asarray(x, dtype=float), 360.0)


def box_mask(lat, lon, box):
    """(H, W) bool mask for a lon/lat box given as (west, east, south, north) in degrees.

    Longitudes are compared modulo 360, so the SAME box argument works whether the dump's
    lon axis runs 0..360 (ARCO/ERA5, ours) or -180..180 (what a reader will type, and what
    the companion study's ``--region -126 -110 38 55`` uses). A box whose wrapped west
    exceeds its east crosses the prime meridian and is handled as a union. Latitude is
    compared directly, so a descending ERA5 lat axis needs no special case.
    """
    if box is None:
        return np.ones((len(lat), len(lon)), dtype=bool)
    west, east, south, north = (float(v) for v in box)
    if south > north:
        raise ValueError(f"box south ({south}) > north ({north})")
    lo = _wrap360(lon)
    w, e = float(_wrap360(west)), float(_wrap360(east))
    if abs(east - west) >= 360.0 or (w == e and east > west):
        # A full zonal span -- '-180 180', '0 360' -- wraps to w == e, and the general
        # comparison below would then select the SINGLE meridian lon == w. That is not an
        # empty domain, so nothing downstream raises: the run silently reduces over a 1-D
        # slice. Full-zonal latitude bands are a real invocation here (the ageostrophic
        # diagnostic is quoted over a 30-60 deg band), so this is a wrong answer, not a
        # corner case. Degenerate 'west == east' with no span still means one meridian.
        in_lon = np.ones(len(lo), dtype=bool)
    else:
        in_lon = (lo >= w) & (lo <= e) if w <= e else ((lo >= w) | (lo <= e))
    in_lat = (np.asarray(lat, dtype=float) >= south) & (np.asarray(lat, dtype=float) <= north)
    return in_lat[:, None] & in_lon[None, :]


def dump_channels(path):
    """Channel names in a field dump, without reading a single field slab. Used to fail on a
    mistyped --channel in a second, with the available names listed, instead of a hundred
    files later or -- worse -- behind a different guard's error message."""
    import h5py

    with h5py.File(path, "r") as f:
        return [c.decode() if isinstance(c, bytes) else str(c) for c in f["channel"][...]]


def load_land_fraction(path, key="LSM"):
    """Read the land-sea mask invariant exactly as makani/utils/auxiliary_fields.py does
    (``f[key][0, :, :]``), so the pixels this analysis calls land are the pixels the model
    was told were land. Raw fractional 0..1; the caller thresholds."""
    import h5py

    with h5py.File(path, "r") as f:
        if key not in f:
            raise SystemExit(f"key {key!r} not in {path}; available: {list(f.keys())}")
        return np.asarray(f[key][0, :, :], dtype=np.float64)


def domain_weights(lat, lon, box=None, land_fraction=None, land_threshold=0.5, want="land"):
    """(H, W) non-negative area weights: cos(lat) x in-box x land/sea selection.

    cos(lat) is the correct analog on our 0.25 deg equiangular grid of the comparison
    paper's uniform HEALPix pixel weights -- their pixels are equal-area, ours are not, so
    an unweighted mean would over-count high latitudes. (It is also the weighting
    backward_rollout.py already uses for every area-weighted metric, so a domain mean here
    and an RMSE there weight the same sphere.)

    Fails loud on an empty domain rather than returning a NaN mean later.
    """
    lat = np.asarray(lat, dtype=float)
    w = np.cos(np.deg2rad(lat)).clip(min=0.0)[:, None] * np.ones((1, len(lon)))
    w = w * box_mask(lat, lon, box)
    if land_fraction is not None:
        if land_fraction.shape != w.shape:
            raise SystemExit(f"land mask shape {land_fraction.shape} != field grid {w.shape}")
        sel = land_fraction >= float(land_threshold)
        if want == "sea":
            sel = ~sel
        elif want != "land":
            raise ValueError(f"want must be 'land' or 'sea', got {want!r}")
        w = w * sel
    total = float(w.sum())
    if not (total > 0):
        raise SystemExit("domain is empty (box x mask selected zero grid cells) -- widen the box or drop the mask")
    return w


def _support(weights):
    """Bounding row/column slice of the weights' non-zero support.

    A case box covers ~0.1% of a 721x1440 grid, so reducing the full sphere upcasts a
    (M, L, 721, 1440) block to float64 -- ~1 GB of temporary per dump, 250 dumps per case --
    to multiply ~10^5 of those cells by a non-zero weight. Slicing to the support first
    reduces both the copy and the arithmetic to the box: everything dropped was multiplied
    by exactly zero, so the result changes only in float SUMMATION ORDER -- measured at
    <1e-15 relative on a 721x1440 grid, against ~230x fewer cells reduced for a case box.
    A global domain has full-grid support, so that path is untouched and bitwise identical.
    """
    rows = np.flatnonzero(np.any(weights > 0, axis=1))
    cols = np.flatnonzero(np.any(weights > 0, axis=0))
    if rows.size == 0 or cols.size == 0:
        return slice(None), slice(None)
    return slice(int(rows[0]), int(rows[-1]) + 1), slice(int(cols[0]), int(cols[-1]) + 1)


def weighted_mean(fields, weights):
    """Area-weighted mean over the trailing (H, W) axes. ``fields`` may carry any leading
    axes; returns those axes. Kept separate from the weight construction so a caller can
    reduce a (M, L, H, W) member block and an (L, H, W) truth block with ONE weight array
    -- the guarantee that ERA5 and the members are averaged over identical pixels."""
    # asarray BEFORE slicing: the contract is "any leading axes", which a caller may satisfy
    # with a nested list, and a list does not accept fields[..., r, c].
    fields, weights = np.asarray(fields), np.asarray(weights)
    r, c = _support(weights)
    f = np.asarray(fields[..., r, c], dtype=np.float64)
    w = np.asarray(weights[r, c], dtype=np.float64)
    return np.tensordot(f, w, axes=([-2, -1], [0, 1])) / w.sum()


def weighted_rms(fields, weights):
    """Area-weighted root-mean-square over the trailing (H, W) axes."""
    fields, weights = np.asarray(fields), np.asarray(weights)
    r, c = _support(weights)
    f = np.asarray(fields[..., r, c], dtype=np.float64)
    w = np.asarray(weights[r, c], dtype=np.float64)
    return np.sqrt(np.tensordot(f * f, w, axes=([-2, -1], [0, 1])) / w.sum())


# ---------------------------------------------------------------------------
# censoring / survivors
# ---------------------------------------------------------------------------
def first_failure(series, ceil):
    """First index where ``series`` is non-finite or exceeds ``ceil``; None if it never
    does. The scalar twin of compare_backward_rollouts._first_divergence."""
    for i, v in enumerate(series):
        v = float(v)
        if not math.isfinite(v) or v > float(ceil):
            return i
    return None


def first_true(flags):
    """First index where a boolean series is True; None if never. The boolean twin of
    first_failure, for detectors that produce a flag per lead rather than a scalar."""
    idx = np.flatnonzero(np.asarray(flags, dtype=bool))
    return int(idx[0]) if idx.size else None


def combine_failures(*fail_leads):
    """Earliest of several per-member failure leads; None only if every detector says the
    member never failed. This is the ``either`` criterion -- a trajectory is censored from
    the first moment ANY detector calls it diverged, never the last."""
    vals = [int(f) for f in fail_leads if f is not None]
    return min(vals) if vals else None


def make_monotone_alive(fail_lead, n_leads):
    """(n_leads,) bool: True while the member is alive. ``fail_lead`` None = never failed.

    MONOTONE by construction. A member whose tail ratio spikes past the ceiling at lead 12
    and dips back under at lead 13 has not recovered -- it is a diverging trajectory
    sampled at a lucky moment -- so once dead it stays dead. Making that explicit here
    (rather than per-lead thresholding at each call site) is what keeps the survival curve
    monotonically non-increasing, which is the property that makes it readable as a
    survival curve at all.
    """
    alive = np.ones(int(n_leads), dtype=bool)
    if fail_lead is not None:
        alive[int(fail_lead):] = False
    return alive


def survivor_stats(values, alive, quantiles=(0.25, 0.5, 0.75)):
    """Per-lead quantiles over surviving members only.

    ``values``: (M, L) float. ``alive``: (M, L) bool. Returns dict with ``q`` (len(quantiles), L),
    ``n_alive`` (L,), ``survival`` (L,) in [0, 1]. Leads with NO survivor come back NaN
    rather than raising -- a fully-censored deep lead is the expected end state of these
    rollouts, not an error, and the caller's line simply ends there.
    """
    v = np.asarray(values, dtype=np.float64)
    a = np.asarray(alive, dtype=bool)
    if v.shape != a.shape:
        raise ValueError(f"values {v.shape} and alive {a.shape} must match")
    M, L = v.shape
    q = np.full((len(quantiles), L), np.nan)
    n_alive = a.sum(axis=0).astype(int)
    for k in range(L):
        col = v[a[:, k], k]
        col = col[np.isfinite(col)]
        if col.size:
            q[:, k] = np.quantile(col, quantiles)
    return {"q": q, "quantiles": tuple(quantiles), "n_alive": n_alive,
            "survival": n_alive / float(M) if M else np.zeros(L)}


#: ``horizon_with_status`` statuses. THREE outcomes, not two -- see the docstring there.
CROSSED = "crossed"
BEFORE_FIRST = "before_first"
NEVER = "never"


def horizon_with_status(x, y, level, decreasing=True):
    """First crossing of ``level``, as ``(value, status)``. THREE outcomes, because two is
    the bug this function exists to prevent.

    A bare "None means no crossing" answer is OVERLOADED: it conflates

      * ``NEVER``        -- y stayed on the near side for the whole sampled range, i.e. the
                            horizon is DEEPER than the rollout, and
      * ``BEFORE_FIRST`` -- y was ALREADY past ``level`` at the first sample, i.e. the
                            horizon is SHALLOWER than the first sample,

    which are opposite conclusions. That mattered concretely: the spectra start at the
    first rolled step (there is no lead-0 row, see ``read_spectrum``), so a small-scale
    band already blurred below the ratio by -6 h returned None and was drawn as an open
    marker pinned to the ROLLOUT DEPTH -- the fastest-dying scales plotted as the
    longest-surviving, inverting the one ordering diagnostic 4 exists to test.

    ``BEFORE_FIRST`` reports the first FINITE sample's ``x`` as a BOUND, not a measurement:
    the true crossing is somewhere in (anchor, that x] and this grid cannot resolve where.
    It is ``x[finite[0]]``, not ``x[0]`` -- they differ when the leading rows are non-finite,
    and a caller that pins its marker to ``x[0]`` will disagree with the number returned
    here. Callers render it
    "<= -6 h". We deliberately do NOT prepend a synthetic (0 h, ratio 1.0) initial point to
    interpolate against -- it is physically true that pred == truth at the anchor, but
    interpolating a fabricated sample yields a specific-looking number ("-3.2 h") that is
    an artifact of the injection rather than of the data, and this analysis quotes its
    horizons in a paper.

    Interpolating between real samples (rather than snapping to the nearest lead) is kept:
    the lead grid is 6 h coarse and the crossings are what the two cases are compared on.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = [i for i in range(len(y)) if math.isfinite(y[i])]
    if not finite:
        return None, NEVER
    y0f = y[finite[0]]
    if (y0f < level) if decreasing else (y0f > level):
        return float(x[finite[0]]), BEFORE_FIRST
    for i in range(1, len(y)):
        y0, y1 = y[i - 1], y[i]
        if not (math.isfinite(y0) and math.isfinite(y1)):
            continue
        crossed = (y0 >= level > y1) if decreasing else (y0 <= level < y1)
        if crossed:
            if y1 == y0:
                return float(x[i]), CROSSED
            return float(x[i - 1] + (level - y0) * (x[i] - x[i - 1]) / (y1 - y0)), CROSSED
    return None, NEVER


def horizon_at(x, y, level, decreasing=True):
    """``horizon_with_status`` reduced to a single number: the crossing, or None if it never
    crosses within the sampled range.

    Retained for the SURVIVAL horizons, where the series starts at 1.0 by construction so
    ``BEFORE_FIRST`` cannot arise. A caller whose series may start past ``level`` -- any
    ratio-like quantity, notably the banded spectra -- must use ``horizon_with_status``
    instead and render the three cases distinctly. This wrapper returns that bound (the
    first finite sample's ``x``) rather than None in that case, so a naive caller gets a
    bound instead of the exact opposite conclusion, but it still cannot tell the caller
    that it IS a bound.
    """
    value, _status = horizon_with_status(x, y, level, decreasing=decreasing)
    return value


def exponential_fit(x, y):
    """Least-squares fit of log(y) vs x over the finite, positive, non-censored samples.

    Returns (growth_per_unit_x, intercept, n_used) or (nan, nan, 0). Diagnostic 3 fits
    spread vs lead HOURS, so ``growth`` is per hour; the callers convert to a per-step
    amplification exp(growth * step_hours) and a doubling time ln(2)/growth. Fitting in
    log space is what makes exponential growth a straight line whose SLOPE is the rate --
    the point of the figure.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
    if ok.sum() < 2:
        return float("nan"), float("nan"), int(ok.sum())
    slope, intercept = np.polyfit(x[ok], np.log(y[ok]), 1)
    return float(slope), float(intercept), int(ok.sum())


# ---------------------------------------------------------------------------
# spectra (per-member tail ratios; diagnostic 4 and the 'tail' censoring criterion)
# ---------------------------------------------------------------------------
def load_member_spectra(path):
    """Read one backward_spectrum_<tag>_seed<N>.h5 written by makani's SpectrumAverageBuffer.

    Layout (makani/utils/inference/rollout_buffer.py:_write_output_data, as constructed by
    backward_rollout.py): ``mean`` is (R, C, cols, lmax) with col 0 = truth, col 1 = the
    ensemble-MEAN field, cols 2..E+1 = members 0..E-1 -- the same member ORDER as the field
    dump's axis 1, because backward_rollout.py builds both from the same ``pred`` tensor
    (``cat([pred.mean(0, keepdim=True), pred])``).

    *** LEAD OFF-BY-ONE, and it is load-bearing. *** This file has R rows starting at the
    FIRST rolled step, because there is nothing to score at the initial condition. The
    field dump has L = R+1 frames starting at lead 0 (the IC, where pred == truth). So
    spectrum row k is field lead k+1. Returned as ``lead_index`` = k+1 so callers index
    field-lead space throughout and never have to remember this.

    ``lead_time`` in the file is in units of the model's dt (steps), NOT hours -- the
    buffer is constructed with ``rollout_dt=params.dt`` and dt is in dhours units (1 for
    our 6-hourly configs). Hours are therefore derived from an explicit step_hours by the
    caller rather than read from here, where the units are ambiguous.
    """
    import h5py

    with h5py.File(path, "r") as f:
        mean = np.asarray(f["mean"][...], dtype=np.float64)   # (R, C, cols, lmax)
        chans = [c.decode() if isinstance(c, bytes) else str(c) for c in f["channel"][...]]
    if mean.ndim != 4:
        raise SystemExit(f"{path}: expected a 4-D 'mean' (R,C,cols,lmax), got {mean.shape}")
    R, C, cols, lmax = mean.shape
    if cols < 3:
        raise SystemExit(f"{path}: only {cols} spectrum columns -- this is a DETERMINISTIC dump "
                         "(truth, pred), not an ensemble one; per-member spectra need E>1")
    return {"mean": mean, "channels": chans, "n_members": cols - 2,
            "lead_index": np.arange(1, R + 1), "lmax": lmax}


def band_edges(lmax, n_bands, l_min=1):
    """Log-spaced spherical-harmonic bands over [l_min, lmax). Log rather than linear
    because the recoverability claim is that the horizon SHRINKS WITH SCALE, and a linear
    split puts ~everything in the small-scale bins."""
    if n_bands < 1:
        raise ValueError("n_bands must be >= 1")
    if not (0 < lmax) or not (0 <= l_min < lmax):
        raise SystemExit(f"l_min={l_min} is outside this dump's spectrum [0, lmax={lmax}) -- "
                         "every band would be empty")
    edges = np.unique(np.round(np.geomspace(max(1, l_min), lmax, n_bands + 1)).astype(int))
    if len(edges) < 2:
        raise SystemExit(f"lmax={lmax} too small for {n_bands} bands")
    return edges


def tail_ratio_members(spec, channel, l0=None):
    """Per-member high-l power ratio P_pred/P_truth vs lead for one channel.

    Returns (M, R) with R in spectrum-row order; pair with ``spec['lead_index']`` for field
    leads. ``l0`` defaults to lmax//2 -- the SAME 'upper half of the spectrum = small
    scales' split backward_rollout.py records as ``tail_l0``, so this censoring criterion
    and the JSON's tail_power_ratio mean the same thing.
    """
    if channel not in spec["channels"]:
        raise SystemExit(f"channel {channel!r} not in spectrum dump; available: {spec['channels']}")
    ci = spec["channels"].index(channel)
    lmax = spec["lmax"]
    l0 = lmax // 2 if l0 is None else int(l0)
    # An l0 at or past lmax makes the tail slice EMPTY, every ratio 0, and nothing is ever
    # censored -- a run then reports survival 1.000 to the deepest lead while the fields
    # are at 1e20. Silent and catastrophic, so it fails loud instead.
    if not (0 <= l0 < lmax):
        raise SystemExit(f"tail_l0={l0} is outside this dump's spectrum [0, lmax={lmax}) -- the "
                         "tail slice would be empty and NOTHING would ever be censored")
    m = spec["mean"][:, ci, :, l0:]                       # (R, cols, lmax-l0)
    truth = m[:, 0, :].sum(axis=-1)                       # (R,)
    members = m[:, 2:, :].sum(axis=-1)                    # (R, M)
    return (members / np.clip(truth, 1e-30, None)[:, None]).T   # (M, R)


def banded_ratio_members(spec, channel, edges):
    """Per-member per-BAND power ratio P_pred/P_truth vs lead -> (M, R, n_bands).

    Diagnostic 4's payload: the recoverability horizon as a function of wavenumber. Bands
    are half-open [edges[b], edges[b+1]) so no degree is double-counted; l=0 (the domain
    mean) is excluded by band_edges' l_min=1 since its 'ratio' is a mean-offset artifact,
    not a scale.
    """
    if channel not in spec["channels"]:
        raise SystemExit(f"channel {channel!r} not in spectrum dump; available: {spec['channels']}")
    ci = spec["channels"].index(channel)
    m = spec["mean"][:, ci]                                # (R, cols, lmax)
    R, cols, _ = m.shape
    nb = len(edges) - 1
    out = np.full((cols - 2, R, nb), np.nan)
    for b in range(nb):
        sl = slice(int(edges[b]), int(edges[b + 1]))
        truth = m[:, 0, sl].sum(axis=-1)                   # (R,)
        members = m[:, 2:, sl].sum(axis=-1)                # (R, M)
        out[:, :, b] = (members / np.clip(truth, 1e-30, None)[:, None]).T
    return out
