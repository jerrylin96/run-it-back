#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Case-study instability / horizon diagnostics 1-3, from the pooled 1000-member field dumps.

  1. DOMAIN-MEAN SPAGHETTI vs lead -- Figure 1 q/r of the companion study (Lin et al. 2026,
     arXiv:2608.19008; github.com/jerrylin96/rewinding-the-extremes), extended. Every
     member thin, ERA5 dashed, survivor median + IQR, survival fraction underneath, both
     cases on shared axes. Their entire 66 h span is our first 11 steps on the same grid,
     so the panels overlay directly; the only change is an x-axis running to -180 h.
  2. SURVIVAL FRACTION vs lead, with the threshold sensitivity check. Turns "it goes
     unstable" into a curve with a number on it, and near-parallel curves at 2-3
     thresholds show the horizon is a property of the model, not of the threshold.
  3. log(ENSEMBLE SPREAD) vs lead -> e-folding time. Exponential growth is a straight line
     on a log axis and its slope is the growth rate.

THE CLAIM BEING TESTED. If the retrodiction horizon is DYNAMICAL rather than
ARCHITECTURAL it must be REGIME-DEPENDENT -- shorter for a fast, small-scale hurricane
than for a persistent, large-scale ridge. That is why every figure here takes several
cases and draws them on shared axes: the contrast is the result, not a decoration. A
66-hour context window is a property of a model; a horizon that moves with the weather
regime is a property of the atmosphere.

WHAT IT DOES NOT READ. Not the per-task ``backward_rollout_*.json``. Those describe E=4,
not E=1000 (CRPS has a pairwise term, spread an (E-1) correction), so they do not average
across the 250 tasks into a 1000-member value. Every distributional number here is
recomputed from the field dumps, where per-member behaviour pools exactly.

Usage (every path and glob is an argument; both the pre-tag and the case-tagged dump
filename schemes work):

  # one case, PNW t2m over its impact box, land-masked
  S=$MAKANI_DATA/runs/rewind_fcn3_ens_casestudy/fcn3_ens_10yr/scores
  crun -t 02:00:00 python /opt/makani/analysis/plot_ensemble_horizon.py \
      "PNW heat dome=$S/backward_fields_20210628T1200_seed*.h5" \
      --spectrum "PNW heat dome=$S/backward_spectrum_20210628T1200_seed*.h5" \
      --channel t2m --box -126 -110 38 55 \
      --land_mask $MAKANI_DATA/invariants/land_sea_mask.h5 \
      --out_dir $S/../diagnostics

  # both 32-channel cases on shared axes. --case_box is REQUIRED here: the two
  # regimes are in different PLACES, so one shared box would reduce the Carolina hurricane
  # over the Pacific Northwest and still label the curve "Hurricane Ian".
  crun -t 04:00:00 python /opt/makani/analysis/plot_ensemble_horizon.py \
      "PNW heat dome=$S/backward_fields_20210628T1200_seed*.h5" \
      "Hurricane Ian=$S/backward_fields_20220930T1800_seed*.h5" \
      --spectrum "PNW heat dome=$S/backward_spectrum_20210628T1200_seed*.h5" \
      --spectrum "Hurricane Ian=$S/backward_spectrum_20220930T1800_seed*.h5" \
      --channel msl --box -170 -100 25 70 \
      --case_box "Hurricane Ian=-90,-75,22,36" \
      --out_dir $S/../diagnostics

THE DOMAIN IS A PROPERTY OF THE (CASE, VARIABLE) PAIR, NOT OF THE CASE. Taken verbatim
from the companion study's own case definitions (Lin et al. 2026, arXiv:2608.19008; github.com/jerrylin96/rewinding-the-extremes,
`scripts/_shared/cases/{pnw_heatwave,ian}.yaml`, key `diagnostics.spread_rmse_crps.domains`)
so our numbers land on the same pixels as theirs. One --channel per run, so pick the row:

  case            channel   region     --box / --case_box W,E,S,N     mask
  PNW heat dome   t2m       impact     -126 -110 38 55                land
  PNW heat dome   z500      synoptic   -170 -100 25 70                none
  PNW heat dome   msl       synoptic   -170 -100 25 70                none
  Hurricane Ian   z500      synoptic   -110  -50 15 50                none
  Hurricane Ian   msl       impact      -90  -75 22 36                none

Their rationale, worth keeping because it is not arbitrary: z500 is the synoptic-scale
CAUSE (the steering flow / the omega block), so it reduces over the same pixels their EOF
analysis uses; the impact variable is the EFFECT, so it reduces over the tighter box. For
Ian that effect variable is `msl` -- the storm's own mass signature -- and averaging it
over the full synoptic box "would dilute the cyclone's mass signature with quiet
background". NOTE Ian has NO t2m row: their Ian analysis does not use t2m at all, so a
t2m run on that case is comparable to nothing they published.

THE LAND MASK IS GLOBAL (there is no --case_mask), and exactly one row above wants it:
PNW t2m. Their reason is ours -- ocean t2m is essentially the prescribed, member-identical
SST, so it contributes full weight to the pixel mean and almost no spread. z500 and msl
are meaningful over land and ocean alike and are unmasked in BOTH cases. So pass
--land_mask only on the PNW-t2m run, and never on a run that includes Ian.

TWO THINGS THAT ARE NOT APPLES TO APPLES, by construction rather than by oversight.
(1) GRID. They reduce on equal-area HEALPix with uniform weights; we are on 0.25 deg
equiangular, where the correct analog is cos(lat) -- an unweighted mean over a 38-55 N box
is biased by several percent. That is why domain_weights() exists.
(2) z500 UNITS -- resolved by ADOPTING their convention. ERA5 stores geopotential in
m^2/s^2; their var_metadata.to_display_units divides by g to report geopotential HEIGHT in
m, which is also what a domain scientist expects to read off a 500 hPa plot. So this tool
now converts too, at the point the dimensional quantities are produced, meaning stdout, the
JSON and the figures are all in m and cannot disagree. The JSON records "units" and
"geopotential_height" so the convention is never inferred. --raw_geopotential opts out.
Ratios -- ACC, spread/skill, the tail power ratio, the banded spectral ratios, the
e-folding RATE -- are invariant and are NOT converted. Physics that genuinely needs
geopotential (geostrophic wind) must read the raw dumps, not this tool's output.

Prints every number it plots to stdout and writes them to ensemble_horizon.json, so the
figures are never the only record. Reads inputs "r" only -- it never writes into a dump
directory.
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backward_fields_io import load_channel_members  # noqa: E402
import ensemble_horizon as eh  # noqa: E402

# The companion study's PNW impact domain, verbatim from its own case definition
# (rewinding-the-extremes, scripts/_shared/cases/pnw_heatwave.yaml: regions.impact
# lon [-126, -110] lat [38, 55], used with mask land).
# It is the right default for PNW t2m ONLY; see the (case, variable) table in the module
# docstring for every other combination.
# Only a DEFAULT -- --box overrides, and --no_box reduces globally.
DEFAULT_BOX = (-126.0, -110.0, 38.0, 55.0)
# Their context-window limit. Marked on every lead axis because the whole point of the
# x-axis running to -180 h is that theirs stops here.
DEFAULT_MARKS = ["cbottle-video 66 h context=66"]

_UNSET = object()   # "no box passed", distinct from None = "reduce globally"

def _display(args, x, power=1):
    """Raw storage units -> the units the figures and the JSON are labelled in.

    Geopotential only, and only when --raw_geopotential is not set. Applied to the
    DIMENSIONAL quantities (the domain-mean scalar, ERA5, the spread) at the point they
    are produced, so stdout, the JSON and the figures can never disagree about units --
    this tool's contract is that the record matches the plot.
    """
    return x if args.raw_geopotential else eh.to_display_units(args.channel, x, power)


def _unit(args):
    return eh.storage_unit(args.channel) if args.raw_geopotential else eh.display_unit(args.channel)



def _parse_labeled(args, what):
    """'label=value' pairs -> ordered dict. Bare values self-label from the basename."""
    out = {}
    for a in args or []:
        if "=" in a:
            label, val = a.split("=", 1)
        else:
            label, val = os.path.basename(a).replace("*", "").strip("_"), a
        if label in out:
            raise SystemExit(f"duplicate {what} label {label!r}")
        out[label] = val
    return out


def _range_failures(pred, lo, hi):
    """(M, L) bool: this member's field at this lead is non-finite or outside [lo, hi].

    Checked over the WHOLE globe, not the analysis box: a member that has blown up in the
    Pacific is a diverged trajectory even while its PNW box-mean still looks like weather,
    and the rollout feeds itself, so the blow-up is already in its next input.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # all-NaN slices are caught by the finiteness test
        mn = np.nanmin(pred, axis=(-2, -1))
        mx = np.nanmax(pred, axis=(-2, -1))
    nonfinite = ~np.isfinite(pred).all(axis=(-2, -1))
    return nonfinite | (mn < lo) | (mx > hi)


def _load_case(label, fields_glob, spectrum_glob, args, log, box=_UNSET):
    """One case: pool every seed's members, reduce to the domain-mean scalar, censor each
    member at its own failure lead, and (unless --no_spread) accumulate the field-space
    ensemble variance over survivors."""
    # _UNSET, not None: None is a legitimate box meaning "reduce globally", so it cannot
    # double as "caller did not say". Defaulting to args.box keeps every existing caller.
    box = args.box if box is _UNSET else box
    files = eh.discover(fields_glob, max_files=args.max_files)
    log(f"[{label}] {len(files)} field dumps matched {fields_glob}")
    have = eh.dump_channels(files[0][1])
    if args.channel not in have:
        raise SystemExit(f"[{label}] channel {args.channel!r} is not in these dumps; available: {have}")
    spec_by_seed = {}
    if spectrum_glob:
        for s, p in eh.discover(spectrum_glob, max_files=args.max_files):
            spec_by_seed[s] = p
        log(f"[{label}] {len(spec_by_seed)} spectrum dumps matched {spectrum_glob}")

    # Provenance BEFORE the heavy read: every field this checks lives in the file attrs, so a
    # seed-0 or duplicate-seed dump aborts in seconds instead of after the whole pool (5 TB per
    # case) has been streamed. Re-checked after the read too, against the attrs actually loaded.
    prov = eh.check_pool_provenance({s: eh.read_attrs(p) for s, p in files},
                                    strict=not args.allow_bad_provenance)
    log(f"[{label}] provenance: {prov['n_files']} files, {prov['n_distinct_noise_seeds']} distinct "
        f"noise seeds, ensemble_size={prov['ensemble_size']}"
        + (f"  PROBLEMS: {prov['problems']}" if prov["problems"] else ""))

    rng_leads = []
    criterion = args.criterion
    if criterion == "auto":
        criterion = "tail" if spec_by_seed else "range"
    if criterion in ("tail", "either") and not spec_by_seed:
        raise SystemExit(f"[{label}] --criterion {criterion} needs --spectrum '<label>=<glob>'")
    if criterion in ("range", "either") and (args.range_lo is None or args.range_hi is None):
        raise SystemExit(f"[{label}] the {criterion} criterion needs a physical range for channel "
                         f"{args.channel!r}; pass --range_lo/--range_hi "
                         f"(defaults exist for {sorted(eh.RANGE_DEFAULT)})")
    log(f"[{label}] censoring criterion = {criterion}"
        + (f" (tail ceiling {args.tail_ceil})" if criterion != "range" else "")
        + (f" (range {args.range_lo}..{args.range_hi})" if criterion != "tail" else ""))

    lo, hi = args.range_lo, args.range_hi
    scal, alive, attrs_by_seed, seeds = [], [], {}, []
    weights = lat = lon = leads = truth_scalar = None
    sum_d = sumsq_d = None            # (L, H', W') ensemble-variance accumulators over survivors
    n_alive_grid = None               # (L,) survivor count feeding those accumulators
    st = args.spread_stride
    t0 = time.time()

    for n, (seed, path) in enumerate(files, 1):
        d = load_channel_members(path, args.channel, anchor=args.anchor)
        pred, truth = d["pred"], d["truth"]        # (M,L,H,W), (L,H,W)
        if weights is None:
            lat, lon = d["lat"], d["lon"]
            leads = np.asarray(d["lead_hours_into_past"], dtype=float)
            land = eh.load_land_fraction(args.land_mask) if args.land_mask else None
            weights = eh.domain_weights(lat, lon, box, land, args.land_threshold, args.mask_want)
            ncell = int((weights > 0).sum())
            log(f"[{label}] domain: box={box} mask={args.mask_want if args.land_mask else 'none'} "
                f"-> {ncell} of {weights.size} grid cells ({100.0 * ncell / weights.size:.2f}%)")
            truth_scalar = eh.weighted_mean(truth, weights)
            if not args.no_spread:
                Hs, Ws = weights[::st, ::st].shape
                sum_d = np.zeros((len(leads), Hs, Ws))
                sumsq_d = np.zeros((len(leads), Hs, Ws))
                n_alive_grid = np.zeros(len(leads), dtype=np.int64)
        elif pred.shape[-2:] != weights.shape:
            raise SystemExit(f"[{label}] {path}: grid {pred.shape[-2:]} != {weights.shape} (mismatched dumps)")
        elif len(d["lead_hours_into_past"]) != len(leads):
            raise SystemExit(f"[{label}] {path}: {len(d['lead_hours_into_past'])} leads != {len(leads)} "
                             "-- pooling dumps of different depth would misalign every curve")

        M, L = pred.shape[0], pred.shape[1]
        rng_fail = _range_failures(pred, lo, hi) if criterion in ("range", "either") else None
        tail = None
        if criterion in ("tail", "either"):
            if seed not in spec_by_seed:
                raise SystemExit(f"[{label}] seed {seed} has a field dump but no spectrum dump "
                                 "-- the tail criterion needs both; check the --spectrum glob")
            spec = eh.load_member_spectra(spec_by_seed[seed])
            tail = eh.tail_ratio_members(spec, args.channel, args.tail_l0)   # (M, R)
            if tail.shape[0] != M:
                raise SystemExit(f"[{label}] seed {seed}: spectrum has {tail.shape[0]} members, "
                                 f"field dump has {M}")
            if tail.shape[1] != L - 1:
                raise SystemExit(f"[{label}] seed {seed}: spectrum has {tail.shape[1]} rows, expected "
                                 f"{L - 1} (= L-1; spectrum row k is field lead k+1)")

        for m in range(M):
            f_rng = eh.first_true(rng_fail[m]) if rng_fail is not None else None
            f_tail = None
            if tail is not None:
                k = eh.first_failure(tail[m], args.tail_ceil)
                f_tail = None if k is None else k + 1   # spectrum row k -> field lead k+1
            # kept in flat member order, matching scal/alive, so _survival_sensitivity can
            # re-censor at other tail ceilings WITHOUT dropping the range half of 'either'
            rng_leads.append(f_rng)
            alive.append(eh.make_monotone_alive(eh.combine_failures(f_rng, f_tail), L))
        scal.append(eh.weighted_mean(pred, weights))    # (M, L)
        seeds.extend([seed] * M)
        attrs_by_seed[seed] = d["attrs"]

        if not args.no_spread:
            a = np.asarray(alive[-M:])                                    # (M, L)
            # ONE MEMBER AT A TIME. Stacking all M built three (M,L,H,W) float64 temporaries at
            # once -- the subtraction, the np.where copy and the square -- for a ~4.2 GB peak at
            # M=4 on the 0.25-deg grid, which the cluster's default memory allocation OOM-killed.
            # Per member the arithmetic is identical (the accumulators stay float64; only the
            # order of the M-way summation into them changes) and the peak drops to ~1.5 GB.
            tr = truth[:, ::st, ::st]
            for m in range(M):
                dev = (pred[m, :, ::st, ::st] - tr).astype(np.float64)
                dev[~a[m]] = 0.0                                          # dead members contribute nothing
                sum_d += dev
                np.multiply(dev, dev, out=dev)          # square in place: no second (L,H',W') temp
                sumsq_d += dev
            n_alive_grid += a.sum(axis=0)

        if args.progress and (n % args.progress == 0 or n == len(files)):
            log(f"[{label}]   {n}/{len(files)} files, {time.time() - t0:.0f}s")

    scal = np.concatenate(scal, axis=0)          # (Mtot, L)
    scal = _display(args, scal)
    truth_scalar = _display(args, truth_scalar)
    alive = np.asarray(alive)                    # (Mtot, L)
    prov = eh.check_pool_provenance(attrs_by_seed, strict=not args.allow_bad_provenance)
    log(f"[{label}] pool: {scal.shape[0]} members from {prov['n_files']} files, "
        f"{prov['n_distinct_noise_seeds']} distinct noise seeds"
        + (f"  PROBLEMS: {prov['problems']}" if prov["problems"] else ""))

    stats = eh.survivor_stats(scal, alive)
    spread = None
    spread_lost_max = 0.0
    if not args.no_spread:
        n = n_alive_grid.astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            # unbiased ensemble variance per grid point, over SURVIVORS at each lead.
            # deviations were taken about truth: variance is shift-invariant, and the shift
            # keeps the sum of squares of ~300 K fields well-conditioned in float64.
            var = (sumsq_d - sum_d ** 2 / n[:, None, None]) / np.maximum(n[:, None, None] - 1.0, 1.0)
        var = np.where((n > 1)[:, None, None], var, np.nan)
        w = weights[::st, ::st]
        # Striding can step straight over a small box -- a PNW-sized domain at
        # --spread_stride 12 keeps zero weighted cells -- and w.sum() == 0 then yields a
        # bare RuntimeWarning, a NaN spread column and a NaN e-folding fit in the JSON.
        # domain_weights fails loud on an empty domain; so does this.
        if not (w.sum() > 0):
            raise SystemExit(f"[{label}] --spread_stride {st} strides over the whole domain "
                             f"({int((weights > 0).sum())} weighted cells before striding, 0 after) -- "
                             "lower the stride or widen the box")
        # Reduce over the FINITE cells only, renormalising by their weight. The previous
        # nan_to_num(var, 0.0) treated a non-finite variance as a variance of exactly ZERO,
        # which biases the spread -- and the e-folding fit derived from it -- silently low.
        # A member can carry non-finite grid points and still be 'alive': under --criterion
        # tail nothing inspects the field values, and NaN comparisons are False, so the range
        # detector does not fire on NaN either. Those points then poison sum_d/sumsq_d for
        # every deeper lead. Everywhere else in this module a non-finite value is censored or
        # fails loud; it must not be quietly zeroed here.
        finite = np.isfinite(var)
        w_finite = np.tensordot(finite.astype(np.float64), w, axes=([-2, -1], [0, 1]))   # (L,)
        with np.errstate(invalid="ignore", divide="ignore"):
            spread = np.sqrt(np.tensordot(np.where(finite, var, 0.0), w, axes=([-2, -1], [0, 1])) / w_finite)
        spread = np.where(n > 1, spread, np.nan)
        spread = _display(args, spread)          # an RMS, so power 1 like the scalar
        # Report the loss rather than absorb it. Leads with fewer than 2 survivors are NaN by
        # construction and excluded, so anything left is a genuine non-finite field value.
        # The trigger is the EXACT boolean `bad.any()`, not `w_finite < w.sum()`: those two
        # sums run in different orders and disagree at ~1e-16 even when every cell is finite,
        # which would fire this warning on a clean pool.
        real = n > 1
        lost = np.zeros(len(leads))
        bad = real[:, None, None] & ~finite
        if bad.any():
            lost[real] = np.clip(1.0 - w_finite[real] / w.sum(), 0.0, 1.0)
        spread_lost_max = float(lost.max())
        if spread_lost_max > 0:
            # .3g, not .2f: a single bad grid cell in a global domain is ~1e-6 of the weight
            # and would print as "0.00%", reading as "nothing happened" on the one line that
            # exists to say something did.
            log(f"[{label}] WARNING: non-finite ensemble variance over up to "
                f"{100 * spread_lost_max:.3g}% of the domain weight (worst lead "
                f"{leads[int(np.argmax(lost))]:g} h); the spread there is over the finite cells only")

    return {"label": label, "leads": leads, "scalar": scal, "alive": alive, "seeds": np.asarray(seeds),
            "truth_scalar": truth_scalar, "stats": stats, "spread": spread, "criterion": criterion,
            "provenance": prov, "n_files": len(files), "spec_by_seed": spec_by_seed,
            "file_seeds": [s for s, _ in files], "rng_leads": rng_leads,
            "spread_nonfinite_weight_fraction": spread_lost_max, "box": box,
            "n_domain_cells": int((weights > 0).sum()), "grid": weights.shape}


def _survival_sensitivity(case, args, log):
    """Survival curves at several tail ceilings -- methodology item 5. Near-parallel curves
    demonstrate the horizon is a property of the model rather than of the threshold, and
    end that reviewer objection in a sentence. Only the spectra are re-read, so this costs
    a cheap second pass over kilobytes rather than one over terabytes of fields.

    The censoring rule is the CASE'S OWN, not tail-only. Under --criterion either the
    headline curve is censored by min(tail, range), so a tail-only sensitivity curve sat
    ABOVE it in the same figure -- including at the shared default ceiling of 10, where the
    two are supposed to coincide -- and the "near-parallel curves" argument would have been
    read off a different rule from the number it is defending. Under --criterion range the
    tail ceiling does not enter the headline curve at all, so there is nothing to vary and
    the panel says so instead of drawing a curve nothing depends on.
    """
    if not case["spec_by_seed"] or case["criterion"] == "range":
        return {}
    L = len(case["leads"])
    ceilings = [float(c) for c in args.sensitivity_ceilings]
    # SAME order as the field-dump pass, so rng_leads[i] belongs to tails[i]. Iterating the
    # field-dump seeds (not sorted(spec_by_seed), which may hold extra seeds) is what keeps
    # the two aligned; _load_case has already failed loud on a field dump with no spectrum.
    tails = []
    for seed in case["file_seeds"]:
        spec = eh.load_member_spectra(case["spec_by_seed"][seed])
        tails.append(eh.tail_ratio_members(spec, args.channel, args.tail_l0))
    tails = np.concatenate(tails, axis=0)        # (Mtot, R)
    rng_leads = case["rng_leads"] if case["criterion"] == "either" else [None] * tails.shape[0]
    if len(rng_leads) != tails.shape[0]:
        raise SystemExit(f"[{case['label']}] sensitivity: {tails.shape[0]} spectrum members vs "
                         f"{len(rng_leads)} field members -- the two passes disagree")
    out = {}
    for c in ceilings:
        alive = np.asarray([
            eh.make_monotone_alive(
                eh.combine_failures(f_rng,
                                    None if (k := eh.first_failure(t, c)) is None else k + 1), L)
            for t, f_rng in zip(tails, rng_leads)])
        out[c] = alive.sum(axis=0) / float(alive.shape[0])
    return out


def _resolve_case_box(label, args):
    """The analysis box for ONE case: --case_box LABEL=W,E,S,N, else the shared --box.

    Per-case boxes are not a convenience. These figures exist to contrast weather REGIMES,
    which live in different places -- a Pacific Northwest ridge and a Carolina landfall -- and
    one shared box reduces the second case over the first case's geography while labelling
    the curve with the second case's name. That is a wrong number that looks like a result,
    so the override is spelled the same way as --spectrum and validated the same way.
    """
    raw = args.case_box.get(label)
    if raw is None:
        return args.box
    if raw.strip().lower() in ("global", "none"):
        return None
    parts = raw.replace(",", " ").split()
    if len(parts) != 4:
        raise SystemExit(f"--case_box {label!r}: expected W,E,S,N (4 numbers), got {raw!r}")
    try:
        w, e, so, no = (float(x) for x in parts)
    except ValueError:
        raise SystemExit(f"--case_box {label!r}: {raw!r} is not four numbers")
    if not (-90.0 <= so < no <= 90.0):
        raise SystemExit(f"--case_box {label!r}: latitudes must satisfy -90 <= south < north <= 90, got {so}, {no}")
    return (w, e, so, no)


def main():
    p = argparse.ArgumentParser(
        description="Diagnostics 1-3: domain-mean spaghetti, survival fraction, spread growth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("cases", nargs="+", metavar="LABEL=FIELDS_GLOB",
                   help="One per case, e.g. 'PNW heat dome=/work/.../backward_fields_2021*_seed*.h5'. "
                        "QUOTE it so the shell does not expand the glob.")
    p.add_argument("--spectrum", action="append", metavar="LABEL=SPECTRUM_GLOB",
                   help="Per-case per-seed spectrum glob, enabling the tail censoring criterion "
                        "and the survival sensitivity check. Repeatable; labels must match the cases.")
    p.add_argument("--channel", default="t2m", help="Channel to reduce (must be in the dumps).")
    p.add_argument("--anchor", default=0, type=int, help="Anchor slot within each dump (these are 1-anchor dumps).")
    p.add_argument("--box", nargs=4, type=float, default=list(DEFAULT_BOX),
                   metavar=("WEST", "EAST", "SOUTH", "NORTH"),
                   help="Analysis box in degrees; longitudes are matched modulo 360 so -126 and 234 "
                        "both work. Pass --no_box to reduce globally.")
    p.add_argument("--no_box", action="store_true", help="Reduce over the whole globe (ignore --box).")
    p.add_argument("--case_box", action="append", metavar="LABEL=W,E,S,N",
                   help="Per-case box overriding --box, e.g. 'Hurricane Ian=-90,-75,22,36'. Repeatable; "
                        "labels must match the cases. REQUIRED whenever the cases are in different "
                        "places: one --box applied to both would reduce a Florida hurricane over a "
                        "Pacific Northwest land box and label the result 'Hurricane Ian'. Pass "
                        "'LABEL=global' to reduce that one case over the whole globe.")
    p.add_argument("--land_mask", default=None,
                   help="Path to the land_sea_mask.h5 invariant (key LSM). Omit for no mask. The "
                        "companion study masks t2m to land because ocean pixels are driven by a "
                        "member-identical prescribed SST and dilute the ensemble spread.")
    p.add_argument("--land_threshold", default=0.5, type=float, help="Land fraction counted as land.")
    p.add_argument("--mask_want", default="land", choices=["land", "sea"], help="Which side of the mask to keep.")
    p.add_argument("--criterion", default="auto", choices=["auto", "tail", "range", "either"],
                   help="Per-member failure detector. auto = tail when --spectrum is given, else range.")
    p.add_argument("--tail_ceil", default=eh.TAIL_CEIL_DEFAULT, type=float,
                   help="High-l P_pred/P_truth above this = diverged (compare_backward_rollouts.py's ceiling).")
    p.add_argument("--tail_l0", default=None, type=int,
                   help="First spherical-harmonic degree counted as 'small scale'. Default lmax//2, "
                        "matching backward_rollout.py's tail_l0.")
    p.add_argument("--range_lo", default=None, type=float,
                   help="Physical lower bound for the range criterion. Default: the channel's entry in "
                        f"ensemble_horizon.RANGE_DEFAULT ({eh.RANGE_DEFAULT}); required for a channel with none.")
    p.add_argument("--range_hi", default=None, type=float, help="Physical upper bound for the range criterion.")
    p.add_argument("--sensitivity_ceilings", nargs="+", type=float, default=[3.0, 10.0, 30.0],
                   help="Tail ceilings for the survival sensitivity panel (methodology item 5).")
    p.add_argument("--spread_stride", default=1, type=int,
                   help="Stride the grid when accumulating the field-space ensemble variance. 1 = every "
                        "cell (~0.5 GB of float64 accumulators at 721x1440xR=31); raise it if memory bites.")
    p.add_argument("--no_spread", action="store_true", help="Skip diagnostic 3's accumulator (faster).")
    p.add_argument("--fit_min_survival", default=0.5, type=float,
                   help="Fit the spread growth only over leads where at least this fraction survives. Past "
                        "that the pool has collapsed and the 'spread' describes a handful of trajectories.")
    p.add_argument("--mark", action="append", metavar="LABEL=HOURS", default=None,
                   help=f"Vertical reference line on every lead axis. Default: {DEFAULT_MARKS}. "
                        "Pass --mark none to draw none.")
    p.add_argument("--max_files", default=None, type=int, help="Use only the first N seeds (smoke test).")
    p.add_argument("--progress", default=25, type=int, help="Log every N files (0 = silent).")
    p.add_argument("--allow_bad_provenance", action="store_true",
                   help="Downgrade the pool-provenance check (distinct seeds, all members kept) to a warning. "
                        "Only for deliberately partial pools -- a failure normally means the members are "
                        "duplicates and the ensemble size is a fiction.")
    p.add_argument("--out_dir", default=None, help="Write PNGs + ensemble_horizon.json here (default: no figures).")
    p.add_argument("--member_alpha", default=None, type=float,
                   help="Opacity of the individual member lines. Default: scaled to the pool size "
                        "(clip(6/M, 0.02, 0.6)), so 48 members and 1000 read alike.")
    p.add_argument("--raw_geopotential", action="store_true",
                   help="Plot and record geopotential in its STORAGE units (m^2/s^2) instead of "
                        "converting to geopotential height in m. Off by default: the comparison "
                        "paper publishes height, and height is what domain scientists read off a "
                        "500 hPa plot. Ratios (ACC, spread/skill, tail power, the e-folding RATE) "
                        "are unaffected either way; physics needing true geopotential must use the "
                        "raw dumps, not this tool's output.")
    p.add_argument("--dpi", default=140, type=int)
    args = p.parse_args()
    args.box = None if args.no_box else tuple(args.box)
    args.case_box = _parse_labeled(args.case_box, "case_box")
    dflt = eh.RANGE_DEFAULT.get(args.channel)
    if args.range_lo is None:
        args.range_lo = dflt[0] if dflt else None
    if args.range_hi is None:
        args.range_hi = dflt[1] if dflt else None
    if args.criterion in ("range", "either") and (args.range_lo is None or args.range_hi is None):
        raise SystemExit(f"--criterion {args.criterion} on channel {args.channel!r} has no default physical "
                         "range; pass --range_lo and --range_hi explicitly "
                         f"(defaults exist for {sorted(eh.RANGE_DEFAULT)})")
    if args.range_lo is not None and args.range_hi is not None and args.range_lo >= args.range_hi:
        raise SystemExit(f"--range_lo ({args.range_lo}) must be below --range_hi ({args.range_hi})")

    def log(msg):
        print(msg, flush=True)

    cases_g = _parse_labeled(args.cases, "case")
    spec_g = _parse_labeled(args.spectrum, "spectrum")
    unknown = set(spec_g) - set(cases_g)
    if unknown:
        raise SystemExit(f"--spectrum labels {sorted(unknown)} do not match any case label {sorted(cases_g)}")
    unknown = set(args.case_box) - set(cases_g)
    if unknown:
        raise SystemExit(f"--case_box labels {sorted(unknown)} do not match any case label {sorted(cases_g)}")
    boxes = {lab: _resolve_case_box(lab, args) for lab in cases_g}
    if len(set(boxes.values())) > 1:
        log("per-case boxes: " + ", ".join(f"{k} -> {v}" for k, v in boxes.items()))

    marks = []
    raw_marks = DEFAULT_MARKS if args.mark is None else args.mark
    if raw_marks != ["none"]:
        for m in raw_marks:
            lab, _, hrs = m.rpartition("=")
            marks.append((lab or m, float(hrs)))

    cases = [_load_case(lab, g, spec_g.get(lab), args, log, box=boxes[lab]) for lab, g in cases_g.items()]

    # Shared axes are only meaningful on a shared lead GRID. A different rollout depth is
    # fine (the shorter line just ends sooner), but a different STEP would silently align
    # -42 h against -84 h -- the same footgun compare_backward_rollouts.py fails loud on.
    ref = cases[0]
    for c in cases[1:]:
        n = min(len(ref["leads"]), len(c["leads"]))
        if not np.allclose(ref["leads"][:n], c["leads"][:n]):
            raise SystemExit(f"lead grids differ between {ref['label']!r} and {c['label']!r}: "
                             f"{ref['leads'][:n]} vs {c['leads'][:n]} -- a shared-axis comparison "
                             "would align different lead times against each other")
        if len(ref["leads"]) != len(c["leads"]):
            log(f"NOTE: {ref['label']!r} has {len(ref['leads'])} leads, {c['label']!r} has "
                f"{len(c['leads'])} -- same step, different depth; the shorter curve simply ends earlier.")
    # ... and only on a shared CENSORING RULE. --criterion auto resolves per case, so
    # forgetting one --spectrum label silently puts one case on the tail detector and the
    # other on the range backstop -- and the tail detector fires much earlier by design.
    # Their horizon difference would then be an artifact of the detector, reported as the
    # regime contrast the paper's claim rests on. Fails loud, like the lead-grid check.
    crits = {c["criterion"] for c in cases}
    if len(crits) > 1:
        detail = ", ".join(f"{c['label']!r}: {c['criterion']}" for c in cases)
        raise SystemExit(f"cases were censored by DIFFERENT criteria ({detail}) -- their horizons "
                         "are not comparable. Pass --spectrum for every case, or force one rule "
                         "with --criterion range.")

    # ---- numbers to stdout (the figures are never the only record) ----
    summary = {"channel": args.channel, "units": _unit(args),
               "geopotential_height": eh.is_geopotential(args.channel) and not args.raw_geopotential,
               "box": args.box,
               "case_box": {c["label"]: c["box"] for c in cases}, "land_mask": args.land_mask,
               "mask_want": args.mask_want if args.land_mask else None,
               "criterion": {c["label"]: c["criterion"] for c in cases},
               "tail_ceil": args.tail_ceil, "range": [args.range_lo, args.range_hi],
               "marks": marks, "cases": {}}
    sep = "-" * 100
    for c in cases:
        leads, st_ = c["leads"], c["stats"]
        surv = st_["survival"]
        q25, q50, q75 = st_["q"]
        print(sep)
        print(f"CASE {c['label']}  |  {c['scalar'].shape[0]} members from {c['n_files']} dumps  "
              f"|  criterion={c['criterion']}  |  domain={c['n_domain_cells']} cells of {c['grid']}")
        print(f"{'lead_h':>7} {'survival':>9} {'n_alive':>8} {'ERA5':>10} {'median':>10} "
              f"{'IQR_lo':>10} {'IQR_hi':>10} {'med-ERA5':>10}" + ("  " + f"{'spread':>10}" if c["spread"] is not None else ""))
        for k in range(len(leads)):
            row = (f"{leads[k]:>7.0f} {surv[k]:>9.3f} {st_['n_alive'][k]:>8d} {c['truth_scalar'][k]:>10.4f} "
                   f"{q50[k]:>10.4f} {q25[k]:>10.4f} {q75[k]:>10.4f} {q50[k] - c['truth_scalar'][k]:>10.4f}")
            if c["spread"] is not None:
                row += f"  {c['spread'][k]:>10.4g}"
            print(row)
        h50 = eh.horizon_at(leads, surv, 0.5)
        h90 = eh.horizon_at(leads, surv, 0.9)
        print(f"  survival horizon: 90% at {'-%.1f h' % h90 if h90 is not None else 'never crossed'}; "
              f"50% at {'-%.1f h' % h50 if h50 is not None else 'never crossed'}")
        ent = {"n_members": int(c["scalar"].shape[0]), "n_files": c["n_files"],
               "criterion": c["criterion"], "n_domain_cells": c["n_domain_cells"],
               "lead_hours_into_past": leads.tolist(), "survival": surv.tolist(),
               "n_alive": st_["n_alive"].tolist(), "era5_domain_mean": c["truth_scalar"].tolist(),
               "median": q50.tolist(), "iqr_lo": q25.tolist(), "iqr_hi": q75.tolist(),
               "survival_horizon_h": {"0.9": h90, "0.5": h50},
               "provenance": c["provenance"]}
        if c["spread"] is not None:
            # fit on the SURVIVOR-supported leads only: past the point where the pool has
            # collapsed the spread is a statement about a handful of trajectories.
            ok = surv >= args.fit_min_survival
            slope, icept, nfit = eh.exponential_fit(leads[ok], c["spread"][ok])
            amp = float(np.exp(slope * 6.0)) if np.isfinite(slope) else float("nan")
            dbl = float(np.log(2.0) / slope) if np.isfinite(slope) and slope > 0 else float("nan")
            print(f"  spread growth (fit over {nfit} leads with survival >= {args.fit_min_survival:g}): "
                  f"{slope:.5f} /h  ->  x{amp:.3f} per 6 h step,  doubling time {dbl:.2f} h")
            ent["spread"] = c["spread"].tolist()
            # so a spread biased by non-finite fields is visible in the record, not just in the log
            ent["spread_nonfinite_weight_fraction"] = c["spread_nonfinite_weight_fraction"]
            ent["spread_fit"] = {"growth_per_hour": slope, "intercept_log": icept, "n_leads_fit": nfit,
                                 "amplification_per_6h_step": amp, "doubling_time_h": dbl}
        sens = _survival_sensitivity(c, args, log)
        if sens:
            ent["survival_sensitivity"] = {str(k): v.tolist() for k, v in sens.items()}
            ent["survival_sensitivity_horizon_h"] = {
                str(k): eh.horizon_at(leads, v, 0.5) for k, v in sens.items()}
            print("  sensitivity (50% survival horizon vs tail ceiling): "
                  + ", ".join(f"ceil {k:g} -> "
                              + (f"-{h:.1f} h" if (h := eh.horizon_at(leads, v, 0.5)) is not None else "never")
                              for k, v in sens.items()))
        c["sens"] = sens
        summary["cases"][c["label"]] = ent
    print(sep)

    if len(cases) > 1:
        print("REGIME CONTRAST (the falsifiable prediction: a dynamical horizon must be regime-dependent)")
        for lev in (0.9, 0.5):
            hs = {c["label"]: eh.horizon_at(c["leads"], c["stats"]["survival"], lev) for c in cases}
            txt = ", ".join(f"{k}: " + (f"-{v:.1f} h" if v is not None else "never") for k, v in hs.items())
            print(f"  {int(lev * 100)}% survival horizon -> {txt}")
            vals = [v for v in hs.values() if v is not None]
            if len(vals) == len(hs) and len(vals) > 1:
                # A 0.0 h horizon (every member censored at the anchor) is a real reading
                # now that 0.0 is no longer swallowed as falsy -- the ratio is undefined
                # there, so report the difference alone rather than divide.
                ratio = f" ({max(vals) / min(vals):.2f}x)" if min(vals) > 0 else ""
                print(f"    spread across cases: {max(vals) - min(vals):.1f} h" + ratio)
        print(sep)

    if not args.out_dir:
        print("no --out_dir: numbers only, no figures written.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    written = _figures(cases, args, marks)
    jp = os.path.join(args.out_dir, "ensemble_horizon.json")
    with open(jp, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    written.append(jp)
    print("wrote:")
    for w in written:
        print(f"  {w}")


# Lead axes run CHRONOLOGICALLY: 0 (the anchor / the present) on the right, deeper past
# to the left, so a panel overlays the companion study's Figure 1 q/r directly -- their
# whole 66 h span is our first 11 steps on the same grid. stdout and the JSON keep the
# positive "hours into the past" convention of backward_rollout.py's schema.
_XLABEL = "hours from the anchor  (0 = the present; negative = into the past)"


def _mark_axis(ax, marks):
    for lab, hrs in marks:
        ax.axvline(-abs(hrs), color="0.35", lw=1.0, ls=(0, (4, 3)))
        ax.annotate(lab, xy=(-abs(hrs), 0.02), xycoords=("data", "axes fraction"), rotation=90,
                    fontsize=6, color="0.35", ha="right", va="bottom")


def _domain_label(box, args):
    """Describe ONE case's reduction domain for a figure title.

    The mask half is conditioned on --land_mask being GIVEN, not on --mask_want, which
    always has a value: the title otherwise announced "land only" over an unmasked mean.
    The box half is independent, so a global reduction with a mask reads "global, land only"
    rather than the flatly wrong "global". stdout and the JSON already guard this way; the
    figure is the artifact that leaves the machine, so it must agree with them.

    Takes the CASE's box, not args.box: under --case_box the cases have different domains,
    and a single title naming one of them mislabels the others.
    """
    dom = f"box {box}" if box else "global"
    return dom + (f", {args.mask_want} only" if args.land_mask else "")


def _figures(cases, args, marks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    # From ensemble_horizon, beside the conversion, so the number and its unit string are
    # produced by one table. (An earlier version guessed from channel.startswith("t"),
    # which labelled tcwv (kg/m^2) as kelvin and got t850/t500 right only by luck.)
    unit = _unit(args)
    # bracket only a KNOWN unit, so an unlisted channel reads "domain-mean foo", not "foo []"
    unit_sfx = f" [{unit}]" if unit else ""
    # under --case_box the cases have different domains, so no single title can name one
    mixed_domains = len({c["box"] for c in cases}) > 1
    ncase = len(cases)

    # --- 1. spaghetti + survival, one column per case, SHARED y so the cases compare ---
    fig, axes = plt.subplots(2, ncase, figsize=(5.2 * ncase, 6.4), squeeze=False,
                             sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    # ERA5 is drawn on this axis too, and this ensemble is documented under-dispersed
    # (SSR 0.39-0.53), so truth sitting OUTSIDE the member envelope is the expected case,
    # not a corner one -- bounding on members alone would clip the reference line out of
    # the headline figure exactly where under-dispersion is the finding.
    ylo = min(min(np.nanmin(np.where(c["alive"], c["scalar"], np.nan)),
                  np.nanmin(c["truth_scalar"])) for c in cases)
    yhi = max(max(np.nanmax(np.where(c["alive"], c["scalar"], np.nan)),
                  np.nanmax(c["truth_scalar"])) for c in cases)
    pad = 0.05 * (yhi - ylo + 1e-9)
    for j, c in enumerate(cases):
        ax, axs = axes[0][j], axes[1][j]
        leads, st_ = c["leads"], c["stats"]
        # every member, censored at its own failure lead -- the line simply ends. Nothing
        # past failure is ever drawn, which is why no log axis or clipping is needed.
        y = np.where(c["alive"], c["scalar"], np.nan)
        x = -leads
        # alpha scales with the pool: 48 members at alpha .06 are invisible and 1000 at
        # alpha .5 are a solid block. ~a 6-member equivalent opacity either way.
        a_mem = (args.member_alpha if args.member_alpha is not None
                 else float(np.clip(6.0 / max(y.shape[0], 1), 0.02, 0.6)))
        for m in range(y.shape[0]):
            ax.plot(x, y[m], "-", color="C0", lw=0.35, alpha=a_mem, zorder=1)
        q25, q50, q75 = st_["q"]
        ax.fill_between(x, q25, q75, color="C0", alpha=0.30, lw=0, zorder=2,
                        label="IQR over survivors")
        ax.plot(x, q50, "-", color="C0", lw=2.0, zorder=3, label="median over survivors")
        ax.plot(x, c["truth_scalar"], "--", color="k", lw=1.8, zorder=4, label="ERA5")
        _mark_axis(ax, marks)
        ax.set_ylim(ylo - pad, yhi + pad)
        ax.set_title(f"{c['label']}  ({y.shape[0]} members)"
                     + (f"\n{_domain_label(c['box'], args)}" if mixed_domains else ""), fontsize=9)
        ax.set_ylabel(f"domain-mean {args.channel}{unit_sfx}" if j == 0 else "")
        if j == 0:
            ax.legend(fontsize=7, loc="best")

        axs.plot(x, st_["survival"], "-o", ms=2.5, color="C3")
        axs.axhline(0.5, color="0.6", lw=0.8, ls=":")
        _mark_axis(axs, marks)
        axs.set_ylim(-0.03, 1.03)
        axs.set_xlabel(_XLABEL, fontsize=8)
        if j == 0:
            axs.set_ylabel("surviving fraction")
    fig.suptitle(f"Backward ensemble, domain-mean {args.channel}"
                 + ("  (domain per case, below)" if mixed_domains
                    else f"  ({_domain_label(cases[0]['box'], args)})")
                 + "  -- every member censored at its own divergence lead", fontsize=10)
    fig.tight_layout()
    fp = os.path.join(args.out_dir, f"ensemble_spaghetti_{args.channel}.png")
    fig.savefig(fp, dpi=args.dpi)
    plt.close(fig)
    written.append(fp)

    # --- 2. survival fraction, cases overlaid, with the threshold sensitivity ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), squeeze=True)
    for i, c in enumerate(cases):
        axes[0].plot(-c["leads"], c["stats"]["survival"], "-o", ms=3, color=f"C{i}", label=c["label"])
    axes[0].set_title("Survival fraction vs lead (both cases)", fontsize=10)
    axes[0].legend(fontsize=8)
    styles = ["-", "--", ":", "-."]
    any_sens = False
    for i, c in enumerate(cases):
        for k, (ceil, s) in enumerate(sorted(c.get("sens", {}).items())):
            any_sens = True
            axes[1].plot(-c["leads"], s, styles[k % len(styles)], color=f"C{i}", lw=1.3,
                         label=f"{c['label']}, tail ceiling {ceil:g}")
    axes[1].set_title("Sensitivity: near-parallel curves = the horizon is the model's,\n"
                      "not the threshold's", fontsize=9)
    if any_sens:
        axes[1].legend(fontsize=6)
    else:
        axes[1].text(0.5, 0.5, "no tail-ceiling sensitivity\n(needs --spectrum, and a criterion that uses it)",
                     ha="center", va="center", transform=axes[1].transAxes, fontsize=8, color="0.4")
    for ax in axes:
        ax.axhline(0.5, color="0.6", lw=0.8, ls=":")
        _mark_axis(ax, marks)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel(_XLABEL, fontsize=8)
        ax.set_ylabel("surviving fraction")
    fig.tight_layout()
    fp = os.path.join(args.out_dir, "ensemble_survival.png")
    fig.savefig(fp, dpi=args.dpi)
    plt.close(fig)
    written.append(fp)

    # --- 3. log(spread) vs lead + exponential fit, with survival alongside ---
    # The survival panel is not optional decoration: past onset the spread is computed over
    # survivors, and censoring is survivorship bias. The curve underneath is what keeps the
    # fit honest about how much of the pool it describes.
    if any(c["spread"] is not None for c in cases):
        fig, axes = plt.subplots(2, 1, figsize=(6.6, 5.6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
        ax, axs = axes
        for i, c in enumerate(cases):
            if c["spread"] is None:
                continue
            leads, sp = c["leads"], c["spread"]
            # lead 0 is the initial condition, where every member IS truth, so the spread is
            # exactly 0. Masking it keeps the log axis from drawing a spurious vertical drop
            # to the floor; the zero is real, it just has no place on a log scale.
            ax.semilogy(-leads, np.where(sp > 0, sp, np.nan), "-o", ms=3, color=f"C{i}",
                        label=c["label"])
            ok = c["stats"]["survival"] >= args.fit_min_survival
            slope, icept, nfit = eh.exponential_fit(leads[ok], sp[ok])
            if np.isfinite(slope) and nfit >= 2:
                lab = (f"  fit: x{np.exp(slope * 6.0):.2f} per 6 h, doubling {np.log(2.0) / slope:.1f} h"
                       if slope > 0 else "  fit: decaying")
                ax.semilogy(-leads[ok], np.exp(icept + slope * leads[ok]), "--", color=f"C{i}",
                            lw=1.2, label=lab)
            axs.plot(-leads, c["stats"]["survival"], "-", color=f"C{i}", lw=1.3)
        axs.axhline(args.fit_min_survival, color="0.6", lw=0.8, ls=":")
        axs.annotate(f"fit uses survival >= {args.fit_min_survival:g}", xy=(0.99, 0.08),
                     xycoords="axes fraction", fontsize=6, color="0.4", ha="right")
        axs.set_ylim(-0.03, 1.03)
        axs.set_ylabel("surviving\nfraction", fontsize=8)
        for a in axes:
            _mark_axis(a, marks)
        axs.set_xlabel(_XLABEL, fontsize=8)
        ax.set_ylabel(f"area-weighted ensemble spread of {args.channel}{unit_sfx}", fontsize=9)
        ax.set_title("Ensemble spread growth (straight line = exponential; slope = growth rate)",
                     fontsize=10)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fp = os.path.join(args.out_dir, "ensemble_spread_growth.png")
        fig.savefig(fp, dpi=args.dpi)
        plt.close(fig)
        written.append(fp)
    return written


if __name__ == "__main__":
    main()
