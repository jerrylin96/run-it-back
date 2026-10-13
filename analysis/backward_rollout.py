# SPDX-License-Identifier: Apache-2.0
#
# Multistep BACKWARD autoregressive rollout + failure-mode diagnosis.
#
# What this does
# --------------
# A backward model maps X(t) -> X(t-dt) for one 6 h step (direction confirmed single-step
# by analysis/verify_direction.py). This script rolls that map out autoregressively into
# the PAST -- feed the prediction back to get X(t-2dt), X(t-3dt), ... -- and characterizes
# HOW the rollout degrades. The two failure modes leave opposite fingerprints:
#
#   * Anti-diffusive INSTABILITY: small-scale (high spherical-harmonic-degree) power BLOWS
#       UP relative to truth and grows (super-)exponentially with lead; area-weighted RMSE
#       shoots PAST the persistence/climatology ceilings; fields develop grid-scale noise.
#   * BLUR / NON-UNIQUENESS: the rollout stays bounded but small-scale power COLLAPSES
#       below truth (a deterministic model regresses to the conditional mean of the many
#       pasts consistent with the present); RMSE saturates at the climatology ceiling FROM
#       BELOW; fields go smooth.
#
# Method (mirrors the verify_direction philosophy: read data FORWARD, impose direction here)
# ------------------------------------------------------------------------------------------
#   * We reuse verify_direction.build_inferencer, which loads the trained (backward) WEIGHTS
#     but forces a plain FORWARD single-frame loader (reverse_time=False, n_history=n_future=0).
#     So we never touch the reverse_time dataset machinery during inference (it would raise
#     anyway -- data_loader_multifiles.py: backward + return_target=False is unsupported), and
#     a hypothetical offset-swap bug in that machinery cannot contaminate this diagnosis.
#   * The backward DIRECTION comes entirely from (a) the trained weights and (b) the index walk
#     here: from a present anchor t_i we read truth+zenith at the DESCENDING indices
#     i, i-1, ..., i-R and step the model by -dt each iteration. At step k the model is fed its
#     own previous prediction as input (n_history=0 => next input = prediction) plus the
#     INPUT-side zenith for time t_{i-(k-1)} (the only unpredicted feature the model consumes);
#     it predicts X_hat(t_{i-k}), which we score against the true X(t_{i-k}).
#   * This reuses makani's exact autoregressive feedback (cache_unpredicted_features + one
#     model() call). The denorm/renorm round-trip below keeps it correct even if the input and
#     output normalizations differ (makani's own forward rollout assumes they are equal).
#
# Metrics vs lead time (reusing makani's BUILT-IN spectral diagnostic)
# --------------------------------------------------------------------
#   * Power spectra P(l, lead) per channel via makani.utils.inference.rollout_buffer.
#     SpectrumAverageBuffer -- constructed with the SAME args makani.inference uses on the
#     forward path (grid_type=model_grid_type, out-normalization scale/bias), so the output
#     HDF5 is directly comparable to a forward makani.inference spectrum file.
#     The buffer also stores the TRUTH spectrum (ensemble index 0) next to the prediction
#     (index 1) in the same file, so pred-vs-truth power per degree per lead is self-contained.
#   * Area-weighted (cos-lat) RMSE and ACC per channel vs lead (physical units), plus the
#     PERSISTENCE (hold the present fixed) and CLIMATOLOGY RMSE ceilings -- the references that
#     separate "blew up" (instability) from "saturated at climatology" (blur). NB "climatology"
#     here is time_means.npy, one all-years mean map, not a day-of-year climatology.
#   * A scalar high-degree TAIL-POWER RATIO (sum_{l>=l0} P_pred / P_truth) vs lead: >1 and
#     rising => instability; <1 and falling => blur. The one-number discriminator.
#
# Optionally (--save_fields_anchors > 0) dumps the raw pred+truth FIELD snapshots of a few
# rollouts to <scores>/backward_fields.h5 (schema: backward_fields_io.py) so that
# analysis/make_backward_video.py can render a video of any dumped variable WITHOUT another
# GPU pass. Off by default; the metric outputs are unchanged either way.
#
# ENSEMBLE mode (--ensemble_size E > 1; FCN3 + input noise)
# ---------------------------------------------------------
# The same index-walk rollout, run for E members simultaneously by folding the ensemble into
# the batch dim (the EnsembleTrainer trick): the anchor state is replicated E times, ONE fresh
# multi-scale noise state is drawn at batch E (each batch element gets an independent sample,
# seeded via the preprocessor), and each member then feeds ITSELF back -- E independent
# backward trajectories whose only difference is the noise. The model call mirrors the
# EnsembleTrainer validation loop exactly: update_state=(k != 1), so the freshly drawn state
# is consumed at the first step and AR-advanced (temporal correlation lambd) afterwards.
# Fails loud if the loaded model has no input_noise (members would be identical copies).
#   * The DETERMINISTIC metrics (rmse/acc/persistence/climatology + the compat
#     tail_power_ratio) are computed on the ENSEMBLE MEAN -- the point-prediction analog --
#     so backward_rollout.json stays schema-compatible with deterministic runs and
#     compare_backward_rollouts.py overlays ensemble and deterministic runs unchanged.
#   * Probabilistic metrics vs lead reuse makani's built-in Geometric* metric classes
#     (makani/utils/metrics/functions.py -- the same ones MetricsHandler wires for ensemble
#     validation): CRPS (--crps_type, default the fair "skillspread" estimator MetricsHandler
#     uses for eval), ensemble SPREAD, SSR (spread-skill ratio; ~1 = calibrated, <1 =
#     under-dispersed), and the area-weighted RANK HISTOGRAM (E+1 bins; U-shape =
#     under-dispersion). All in physical units, per channel, per lead.
#     NB the "ssr" key is NOT reconstructible as json["spread"] / json["rmse"]: it is a
#     finite-ensemble-DEBIASED ratio, sqrt(spread / (skill - spread/E)), pooled over anchors.
#     Dividing the two published arrays by hand gives a different (and biased) number.
#   * The spectrum buffer stores columns [truth, ensemble-mean field, member 1..E]: the
#     mean-field column is the blur diagnostic; the member columns answer the question "do
#     individual SAMPLES keep realistic small-scale power even where the mean blurs?"
#     (tail_power_ratio_members in the JSON).
#
# Single-GPU inside the container. --ensemble_size 1 is the deterministic harness.

import os
import sys
import json
import logging

import numpy as np
import torch

# Reuse the validated setup + area-weighted helpers from the direction-verification harness
# (same directory). build_inferencer loads the backward WEIGHTS but forces a forward, single-
# frame loader -- exactly what an externally-driven backward rollout needs.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_direction import build_inferencer, _make_weight, _wmean, _wrmse  # noqa: E402
from backward_fields_io import FieldDumpWriter  # noqa: E402  (torch-free; h5py imported lazily)

from makani.utils import comm  # noqa: E402
from makani.utils import argument_parser  # noqa: E402
from makani.utils.inference.rollout_buffer import SpectrumAverageBuffer  # noqa: E402
from makani.utils.dataloaders.data_helpers import get_climatology  # noqa: E402


def _wacc(pred_anom, truth_anom, wt, wsum):
    # Standard (uncentered) area-weighted anomaly correlation. Anomalies are pred-clim and
    # truth-clim; we do NOT re-subtract their spatial mean (that would be the centered variant),
    # matching the usual NWP ACC definition. Reduces over (H, W) -> (C,).
    # No eps in the denominator: ACC is unit-free and an absolute eps is not. In kg/kg, q's
    # variance product is ~1e-12 at 500 hPa and far smaller aloft, so a 1e-12 eps pulled
    # humidity ACC toward 0. A field with exactly zero anomaly gets NaN: its ACC is undefined.
    num = _wmean(pred_anom * truth_anom, wt, wsum)
    den = torch.sqrt(_wmean(pred_anom ** 2, wt, wsum) * _wmean(truth_anom ** 2, wt, wsum))
    return (num / den).reshape(-1)


def _parse_dump_channels(spec, channel_names, dyn_set):
    """--save_fields_channels: 'all', 'dynamical', or a comma-separated channel list. Fails
    loud on unknown names so a typo can't silently dump the wrong (or an empty) set."""
    if spec == "all":
        return list(channel_names)
    if spec == "dynamical":
        if not dyn_set:
            raise ValueError("--save_fields_channels=dynamical: none of the dynamical channels are in this config")
        return list(dyn_set)
    sel = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [s for s in sel if s not in channel_names]
    if unknown:
        raise ValueError(f"--save_fields_channels: unknown channel(s) {unknown}; available: {channel_names}")
    return sel


def _resolve_single_anchor(dataset, anchor_index, anchor_date, lo, hi, exact=False):
    """Resolve ONE explicit anchor from --anchor_index OR --anchor_date (mutually exclusive),
    overriding the uniform --num_anchors spread so a video can target a specific case study
    (e.g. a hurricane on a chosen date). Returns the dataset index, validated to leave room for
    a full R-step backward window (lo <= idx <= hi). Reuses the dataset's own get_index_at_time
    (a bisect over its tz-aware datestamps) rather than re-deriving the date->index map."""
    import datetime as dt

    if anchor_index is not None and anchor_date:
        raise ValueError("pass only one of --anchor_index / --anchor_date")

    if anchor_index is not None:
        idx = int(anchor_index)
        source = f"--anchor_index {idx}"
    else:
        # datestamps are tz-aware UTC (get_date_from_timestamp), so localize the request to UTC
        # before the bisect or the datetime comparison inside get_index_at_time would raise.
        target = dt.datetime.fromisoformat(anchor_date.replace("Z", "+00:00"))
        target = target.replace(tzinfo=dt.timezone.utc) if target.tzinfo is None else target.astimezone(dt.timezone.utc)
        idx = dataset.get_index_at_time(target)  # frame at/just before target; None if out of range
        if idx is None:
            raise ValueError(f"--anchor_date {anchor_date} is outside the dataset range "
                             f"[{dataset.start_date} .. {dataset.end_date}]")
        resolved = dataset.get_time_at_index(idx)
        source = f"--anchor_date {anchor_date} -> index {idx} ({resolved})"
        # "at or just before" is a SILENT snap: a bare date means 00:00 UTC, and an off-grid
        # request rounds down to the previous 6-hourly frame. Either way the run proceeds
        # happily from the wrong time. --anchor_date_exact turns that into a startup error --
        # worth it whenever the anchor is a chosen event time rather than an arbitrary sample,
        # and doubly so under a job array, where nobody reads 63 logs to notice the snap.
        if exact:
            # A bare date would PASS the frame-match check below -- 00:00 UTC is a real frame --
            # so exactness alone does not catch the footgun it exists for. Require the time to be
            # written out: if the anchor is a chosen event moment, stating it is not a burden.
            if "T" not in anchor_date and " " not in anchor_date.strip():
                raise ValueError(
                    f"--anchor_date_exact: '{anchor_date}' names no time of day, which means "
                    f"00:00 UTC. That is a valid frame, so this would run without complaint from "
                    f"a time you did not choose. Write the time explicitly "
                    f"(e.g. '{anchor_date}T12:00' for 12:00 UTC)."
                )
            got = resolved.astimezone(dt.timezone.utc) if resolved.tzinfo else resolved.replace(tzinfo=dt.timezone.utc)
            if got != target:
                raise ValueError(
                    f"--anchor_date_exact: requested {target.isoformat()} but the nearest frame "
                    f"at/before it is index {idx} = {got.isoformat()}. Frames sit on a fixed "
                    f"cadence, so an off-grid request rounds DOWN (and a bare date means 00:00 "
                    f"UTC). Pass a time that lands on a frame, or drop --anchor_date_exact to "
                    f"accept the snap."
                )

    if not (lo <= idx <= hi):
        raise ValueError(f"{source}: index {idx} is outside the rollable range [{lo}, {hi}]; a "
                         f"backward rollout needs the anchor at least R steps into the valid set so "
                         f"the full window fits (and idx <= n_total-1). Pick a later date.")
    return idx


def _signature(rmse_over_clim_final, tail_ratio_max, tail_ratio_final,
               blowup=1.5, excess=3.0, deficit=0.7, ceiling=1.2):
    """Tentative per-channel fingerprint (advisory only).
       INSTABILITY: RMSE exceeds the climatology ceiling OR small-scale power runs hot.
       BLUR:        RMSE saturates at/below the ceiling AND small-scale power is depleted."""
    # A non-finite metric (NaN or +Inf) means the rollout diverged -- the ultimate instability.
    # Guard first: NaN comparisons are all False, so without this a blow-up to NaN would fall
    # through to "INCONCLUSIVE". (Inf would be caught by the thresholds, but NaN would not.)
    if not (np.isfinite(rmse_over_clim_final) and np.isfinite(tail_ratio_max)):
        return "INSTABILITY"
    if (rmse_over_clim_final > blowup) or (tail_ratio_max > excess):
        return "INSTABILITY"
    if (tail_ratio_final < deficit) and (rmse_over_clim_final <= ceiling):
        return "BLUR"
    return "INCONCLUSIVE"


def run(args):
    # build_inferencer reads args.tf32 as a bool; normalize the on/off choice before it does.
    args.tf32 = args.tf32 == "on"

    inf, params, world_rank = build_inferencer(args)

    # The index walk below steps the dataset by 1 frame per rollout step and pairs each state
    # with the zenith of the SAME index, which is only the intended -dt walk when dt == 1.
    # Every config in this project uses dt: 1; fail loud rather than silently desynchronizing
    # the state and its solar forcing if a future target changes it.
    if int(params.dt) != 1:
        raise NotImplementedError(
            f"backward_rollout.py assumes dt == 1 (got dt={params.dt}): the rollout indexes "
            "samples and zenith angles with a hardcoded stride of 1 frame per step, so dt > 1 "
            "would desynchronize the state from its solar forcing. Stride the index walk first."
        )
    inf._set_eval()  # model.eval() + preprocessor.eval() -> uses the *_eval unpredicted caches

    model = inf.model
    preproc = inf.preprocessor
    device = inf.device
    dataset = inf.valid_dataset

    channel_names = list(params.channel_names)
    C = len(channel_names)
    R = int(args.num_steps)

    # ensemble mode: E members folded into the batch dim, differing only by input noise.
    E = max(1, int(args.ensemble_size))
    if E > 1 and not hasattr(preproc, "input_noise"):
        raise RuntimeError(
            f"--ensemble_size {E} requires a model with input noise (config key input_noise, "
            "e.g. the FCN3 targets in config/rewind_fcn3.yaml) — without it all members would "
            "be identical copies and every spread metric would be trivially 0."
        )
    # Decorrelate this run's noise draws from other runs so several E-member rollouts can be
    # POOLED into one large ensemble. set_rng builds fresh CPU+GPU torch.Generators and seeds
    # both, and reset() zeroes the correlated state, so distinct offsets give genuinely
    # independent trajectories. Offset 0 reproduces the historical default exactly.
    if E > 1 and int(args.noise_seed) != 0:
        preproc.set_rng(reset=True, seed=preproc.get_base_seed() + int(args.noise_seed))

    # --input_noise_sigma was already applied to params by build_inferencer (before the
    # preprocessor was constructed). Two things follow.
    #
    # (1) Zeroed noise makes an ENSEMBLE degenerate: members differ ONLY through the noise, so
    #     every member would be an identical copy and spread/SSR/rank histograms would be
    #     trivially 0 while still writing a plausible-looking JSON. Fail loud, in the same
    #     spirit as the missing-input_noise guard above.
    if args.input_noise_sigma is not None and E > 1 and float(args.input_noise_sigma) == 0.0:
        raise ValueError(
            "--input_noise_sigma 0 with --ensemble_size > 1: the members differ only through "
            "the input noise, so zeroing it collapses them to identical copies and every "
            "spread metric becomes trivially 0. Use --ensemble_size 1 for the noise-off "
            "stability diagnostic, or a nonzero sigma for a reduced-amplitude ensemble."
        )
    #
    # (2) VERIFY the override actually reached the constructed noise module rather than being
    #     silently swallowed by a key that nothing reads -- exactly the failure mode that would
    #     make a "noise-off" run secretly a noise-on run and invert the conclusion. Measuring
    #     the emitted field is the only honest check, so draw one state, take its RMS, and put
    #     the number in the log and the JSON. The draw is made NON-DESTRUCTIVE by saving and
    #     restoring both the RNG generators and the state tensor, so a run with the probe is
    #     bit-identical to one without it (prior runs stay reproducible).
    noise_rms = None
    if hasattr(preproc, "input_noise"):
        rng_saved = preproc.get_internal_state(tensor=False)
        state_saved = preproc.get_internal_state(tensor=True)
        preproc.update_internal_state(replace_state=True, batch_size=E)
        with torch.no_grad():
            noise_rms = preproc.input_noise().float().pow(2).mean().sqrt().item()
        preproc.set_internal_state(state_saved)
        preproc.set_internal_state(rng_saved)
        sigma_cfg = params.input_noise.get("sigma", 1.0)
        print(f"input noise: {type(preproc.input_noise).__name__} sigma={sigma_cfg} "
              f"-> emitted field rms={noise_rms:.4e}"
              + ("   [NOISE OFF]" if noise_rms == 0.0 else ""))

    if args.fields_member not in ("mean", "all"):
        m = int(args.fields_member)  # fail loud on a non-integer spec
        if not (0 <= m < E):
            raise ValueError(f"--fields_member {args.fields_member}: must be 'mean', 'all', or 0..{E-1}")
    if args.fields_member == "all" and E == 1:
        # Not an error -- a deterministic run has exactly one trajectory, which is what gets
        # dumped either way. Say so, because "all" silently meaning "one" is worth knowing when
        # the same command line is reused at E > 1.
        print("--fields_member all with --ensemble_size 1: dumping the single deterministic trajectory")

    # physical (de)normalization. Inputs read with IN stats, predictions with OUT stats; equal
    # for the prototype (one channel list) but kept distinct so this stays correct if in != out.
    in_bias, in_scale = dataset.get_input_normalization()
    out_bias, out_scale = dataset.get_output_normalization()
    ib = torch.as_tensor(in_bias, dtype=torch.float32, device=device)
    isc = torch.as_tensor(in_scale, dtype=torch.float32, device=device)
    ob = torch.as_tensor(out_bias, dtype=torch.float32, device=device)
    osc = torch.as_tensor(out_scale, dtype=torch.float32, device=device)

    def denorm_in(x):
        return x * isc + ib

    def denorm_out(x):
        return x * osc + ob

    def renorm_out(x_phys):
        return (x_phys - ob) / osc

    def renorm_in(x_phys):
        return (x_phys - ib) / isc

    # climatology in normalized (out) space, exactly as makani's ACC uses it; and in physical.
    clim_norm = torch.as_tensor(get_climatology(params), dtype=torch.float32, device=device)  # (1,C,H,W)
    clim_phys = denorm_out(clim_norm)

    H, W = dataset.img_shape[0], dataset.img_shape[1]
    wt, wsum = _make_weight(dataset.lat_lon[0], W, device)

    # makani's BUILT-IN spectral diagnostic -- same construction makani.inference uses on the
    # forward path (inferencer.py), so the output HDF5 is byte-comparable to the forward baseline.
    scores_dir = os.path.join(params.experiment_dir, "scores")
    if world_rank == 0:
        os.makedirs(scores_dir, exist_ok=True)
    spectrum_path = os.path.join(scores_dir, args.spectrum_file)
    # ensemble mode stores columns [truth, ensemble-mean field, member 1..E] (the buffer's
    # ensemble_size counts the DATA columns we pass; it adds the truth column itself), so
    # both the deterministic-comparable mean-field spectrum AND the per-member spectra land
    # in one file. Deterministic mode keeps the original [truth, prediction] layout.
    spectrum_buffer = SpectrumAverageBuffer(
        num_rollout_steps=R,
        rollout_dt=params.dt,
        img_shape=(H, W),
        ensemble_size=(E + 1 if E > 1 else params.local_ensemble_size),
        grid_type=params.model_grid_type,
        channel_names=channel_names,
        device=device,
        scale=osc,
        bias=ob,
        output_channels=channel_names,
        output_file=spectrum_path,
        spatial_distributed=False,
    )
    spectrum_buffer.zero_buffers()

    # probabilistic metrics (ensemble mode only) — reuse makani's built-in Geometric* metric
    # classes, constructed as MetricsHandler does for ensemble validation (channel_reduction
    # "none" -> per-channel values; batch_reduction "mean" is exact at our per-anchor B=1).
    # Fed PHYSICAL-unit tensors, so CRPS/spread come out in each channel's physical units.
    if E > 1:
        from makani.utils.metrics.functions import (  # noqa: E402
            GeometricCRPS, GeometricSpread, GeometricSSR, GeometricRankHistogram,
        )

        mkw = dict(
            grid_type=params.model_grid_type,
            img_shape=(H, W),
            crop_shape=(H, W),
            crop_offset=(0, 0),
            normalize=True,
            channel_reduction="none",
            batch_reduction="mean",
            spatial_distributed=False,
            ensemble_distributed=False,
        )
        crps_metric = GeometricCRPS(crps_type=args.crps_type, **mkw).to(device)
        spread_metric = GeometricSpread(**mkw).to(device)

        # The deterministic metrics (rmse/acc) area-weight with cos(lat) taken from the FILE
        # coordinate; the probabilistic ones weight with GridQuadrature, which SYNTHESIZES
        # colatitudes as linspace(0, pi, H) and never looks at the file. On the standard
        # 721x1440 ERA5 grid these coincide to ~1E-15 (sin(colat_i) == cos(lat_i) at identical
        # nodes; the dtheta convention is a scalar that cancels under normalization), so the two
        # families are interchangeable TODAY. They would diverge silently on a cropped, Gaussian,
        # or half-cell-offset grid -- and regional cropping is exactly what the case-study EOF
        # analysis wants. Assert the agreement rather than assume it survives a grid change.
        _qw = spread_metric.quadrature.quad_weight.detach().reshape(-1, W)[:, 0].double().cpu()
        _qw = (_qw / _qw.sum()).numpy()
        _cw = wt.detach().reshape(-1).double().cpu().numpy()
        _cw = _cw / _cw.sum()
        _dev = float(np.max(np.abs(_qw - _cw)) / max(np.mean(_cw), 1e-30))
        if _dev > 1e-6:
            raise RuntimeError(
                f"area-weight mismatch between the deterministic (cos-lat, from the data file) and "
                f"probabilistic (GridQuadrature, synthesized) measures: max|dw|/mean(w) = {_dev:.3e}. "
                "rmse/acc and crps/spread/ssr would be on different measures, so they could not be "
                "compared or combined. Reconcile the grid before trusting any of these numbers."
            )
        ssr_metric = GeometricSSR(**mkw).to(device)
        rank_metric = GeometricRankHistogram(**mkw).to(device)

    # anchors: present-times spread across the valid set; each rolls back R steps, so we need
    # i - R >= 0 and i <= n_total - 1.
    n_total = dataset.n_samples_total
    step_h = dataset.dhours * params.dt
    lo, hi = R, n_total - 1
    if hi < lo:
        raise RuntimeError(f"valid set too small ({n_total} frames) for an R={R} backward rollout")
    if args.anchor_index is not None or args.anchor_date:
        # explicit single anchor for a case-study video (e.g. a hurricane on a chosen date)
        anchors = [_resolve_single_anchor(dataset, args.anchor_index, args.anchor_date, lo, hi,
                                          exact=args.anchor_date_exact)]
    else:
        n_anchors = min(args.num_anchors, hi - lo + 1)
        anchors = np.unique(np.linspace(lo, hi, num=n_anchors, dtype=int)).tolist()

    dyn_set = [c for c in ["z500", "t500", "u500", "v500", "msl"] if c in channel_names]
    dyn_idx = [channel_names.index(c) for c in dyn_set]
    # A channel subset with no 500 hPa / msl field -- a surface-only ablation, say -- leaves
    # dyn_set empty, and an empty selection fed the signature NaN, which _signature reads as
    # "diverged" and reports as INSTABILITY. A healthy rollout would be labelled catastrophic.
    # Channel-subset runs train on exactly such subsets, so the SIGNATURE
    # falls back to every channel. Kept as a SEPARATE selection rather than widening dyn_set:
    # dyn_set also drives --save_fields_channels=dynamical, whose "none of the dynamical
    # channels are in this config" guard would otherwise be silently satisfied and dump all
    # ~72 channels (~25 GB/anchor) under a flag that asked for five.
    sig_set = dyn_set or list(channel_names)
    sig_idx = dyn_idx or list(range(len(channel_names)))
    if not dyn_set:
        print(f"NOTE: none of z500/t500/u500/v500/msl are in this dataset; the signature below "
              f"is computed over ALL {len(sig_set)} channels instead of the usual dynamical subset.")
        # The tail term is a MAX over channels and leads, so widening the set widens what can
        # trip it: near-zero stratospheric-q truth power makes tail_power_ratio legitimately
        # large on a perfectly stable rollout, and one such channel pins the verdict to
        # INSTABILITY. That is still strictly better than the pre-fallback behaviour (an empty
        # selection gave NaN, i.e. INSTABILITY unconditionally), but on these configs the
        # advisory is weaker than usual and the spectra are what to read.
        print("      The tail term is a max over channels, so a channel with near-zero truth "
              "power (stratospheric q) can trip INSTABILITY on its own -- read the spectra, "
              "not this line, on channel-subset runs.")

    def _date_at(idx):
        try:
            return str(dataset.get_time_at_index(int(idx)))
        except Exception:
            return ""

    # optional raw-field dump (pred + truth snapshots -> HDF5) for make_backward_video.py.
    # Rides along the metric rollout at negligible cost (the fields are already computed each
    # step; dumping is a slab write), so a video needs no separate GPU pass. Rank 0 only --
    # this harness is single-process inference, and the guard keeps a hypothetical multi-rank
    # launch from racing on one file.
    dump, dump_slots, dump_sel_idx, fields_path = None, {}, [], None
    if args.save_fields_anchors > 0 and world_rank == 0:
        dump_channels = _parse_dump_channels(args.save_fields_channels, channel_names, dyn_set)
        dump_sel_idx = [channel_names.index(c) for c in dump_channels]
        n_dump = min(args.save_fields_anchors, len(anchors))
        # anchor-list positions spread evenly, so >1 dumped rollouts sample distinct seasons
        dump_slots = {int(p): s for s, p in enumerate(np.unique(np.linspace(0, len(anchors) - 1, num=n_dump, dtype=int)))}
        fields_path = os.path.join(scores_dir, args.fields_file)
        # members RETAINED in the dump: every member for 'all', otherwise a single trajectory
        # (the ensemble mean, or one named member). E is recorded separately, so a pooled
        # analysis can always tell how many members were run vs how many were kept.
        n_dump_members = E if args.fields_member == "all" else 1
        dump = FieldDumpWriter(
            fields_path, dump_channels, len(dump_slots), R + 1,
            dataset.lat_lon[0], dataset.lat_lon[1], step_h,
            attrs={"config": args.config, "run_num": str(args.run_num), "checkpoint_path": str(params.checkpoint_path),
                   # ensemble provenance: which object the 'pred' frames are
                   "ensemble_size": str(E), "n_members_dumped": str(n_dump_members),
                   # the seed is what makes separate runs poolable -- without it, two dumps
                   # cannot be shown to hold independent members rather than duplicates
                   "noise_seed": str(args.noise_seed),
                   "input_noise_sigma": ("" if args.input_noise_sigma is None else str(args.input_noise_sigma)),
                   "pred_source": ("deterministic" if E == 1 else "all_members" if args.fields_member == "all"
                                   else f"member_{args.fields_member}" if args.fields_member != "mean" else "ensemble_mean")},
            n_members=n_dump_members,
        )

    # accumulators (summed over anchors) -> (lead 1..R) x channel
    sum_rmse = np.zeros((R, C))
    sum_acc = np.zeros((R, C))
    sum_pers = np.zeros((R, C))
    sum_clim = np.zeros((R, C))
    if E > 1:
        sum_crps = np.zeros((R, C))
        sum_spread = np.zeros((R, C))
        # SSR components, pooled over anchors (see the accumulation site for why)
        sum_ssr_spread = np.zeros((R, C))
        sum_ssr_skill = np.zeros((R, C))
        sum_ssr = np.zeros((R, C))
        sum_rank = np.zeros((R, C, E + 1))   # area-weighted rank histogram, mass 1 per anchor
    n_used = 0
    first_date, last_date = None, None

    if world_rank == 0:
        logging.info(f"channels: {channel_names}")
        logging.info(f"backward rollout: R={R} steps of -{step_h} h (deepest lead = -{R*step_h} h), "
                     f"{len(anchors)} present anchors over index [{lo}, {hi}] of {n_total} frames"
                     + (f", ensemble_size={E} (crps_type={args.crps_type})" if E > 1 else ""))
        if len(anchors) == 1:
            logging.info(f"single anchor: index {anchors[0]}  (present time {_date_at(anchors[0])})")
        logging.info(f"spectrum -> {spectrum_path}")
        if dump is not None:
            logging.info(f"field dump: {len(dump_slots)} anchor(s) x {len(dump_sel_idx)} channel(s) -> {fields_path}")

    with torch.inference_mode():
        with torch.no_grad():
            for ai, i in enumerate(anchors):
                # preload truth+zenith at the descending indices i, i-1, ..., i-R.
                # return_target=False => each sample is (frame, zenith[, timestamp]); we use [0],[1].
                samples = [dataset.get_sample_at_index(int(i) - k, return_target=False) for k in range(0, R + 1)]

                X0 = samples[0][0].unsqueeze(0).to(device)          # (1,1,C,H,W) present, IN-norm
                ic_phys = denorm_in(samples[0][0].to(device))       # (1,C,H,W) present, physical
                state = preproc.flatten_history(X0)                 # (1,C,H,W) -> fed to the model
                if E > 1:
                    # fold the ensemble into the batch dim: E copies of the anchor whose
                    # trajectories diverge only through the per-batch-sample input noise
                    state = state.repeat(E, 1, 1, 1)                # (E,C,H,W)
                # draws ONE fresh noise state at the folded batch (independent per member);
                # a no-op for a deterministic (no-input_noise) model
                preproc.update_internal_state(replace_state=True, batch_size=state.shape[0])

                slot = dump_slots.get(ai) if dump is not None else None
                if slot is not None:
                    dump.set_anchor(slot, int(i))
                    ic_np = ic_phys[0, dump_sel_idx].detach().cpu().numpy()
                    # lead 0: pred == truth == IC. Every member starts from the SAME initial
                    # condition (they diverge only through the input noise drawn at each step),
                    # so replicating the IC across the member axis is exact, not a broadcast
                    # convenience -- hence doing it explicitly here rather than in the writer.
                    ic_pred = np.repeat(ic_np[None], n_dump_members, axis=0) if n_dump_members > 1 else ic_np
                    dump.write_frame(slot, 0, ic_pred, ic_np, _date_at(i))

                for k in range(1, R + 1):
                    # input-side zenith for the CURRENT input frame (time t_{i-(k-1)}); this is
                    # the only unpredicted feature the model consumes for its single step.
                    zin = samples[k - 1][1].unsqueeze(0).to(device)  # (1,1,1,H,W)
                    if E > 1:
                        # every member sees the same (true) zenith; expand is a view, and the
                        # preprocessor's _append_channels batch check requires the match
                        zin = zin.expand(E, -1, -1, -1, -1)
                    preproc.cache_unpredicted_features(None, None, zin, None)

                    with inf.autocast():
                        # update_state=(k != 1) mirrors the EnsembleTrainer validation loop:
                        # the fresh noise state drawn above is consumed at the first step, then
                        # AR-advanced (temporal correlation) on subsequent steps. For a
                        # deterministic (no-input_noise) model update_state is a no-op, so the
                        # k==1 change is behavior-identical to the old unconditional True.
                        # replace_state=False mirrors makani's forward rollout.
                        pred = model(state, update_state=(k != 1), replace_state=False)  # (B,C,H,W) out-norm
                    pred = pred.float()

                    # truth at t_{i-k}
                    Xk = samples[k][0].to(device)                    # (1,C,H,W) IN-norm
                    truth_phys = denorm_in(Xk)
                    truth_outnorm = renorm_out(truth_phys)           # == Xk for prototype; general otherwise
                    pred_phys_members = denorm_out(pred)             # (E,C,H,W) physical members
                    # deterministic metrics score the ensemble MEAN (the point-prediction
                    # analog), keeping the JSON schema comparable to the deterministic control
                    pred_phys = pred_phys_members.mean(dim=0, keepdim=True) if E > 1 else pred_phys_members

                    # area-weighted scalar metrics (physical units), accumulated per lead
                    sum_rmse[k - 1] += _wrmse(pred_phys, truth_phys, wt, wsum).detach().cpu().numpy()
                    sum_pers[k - 1] += _wrmse(ic_phys, truth_phys, wt, wsum).detach().cpu().numpy()
                    sum_clim[k - 1] += _wrmse(clim_phys, truth_phys, wt, wsum).detach().cpu().numpy()
                    sum_acc[k - 1] += _wacc(pred_phys - clim_phys, truth_phys - clim_phys, wt, wsum).detach().cpu().numpy()

                    if E > 1:
                        # probabilistic metrics: forecasts (1,E,C,H,W) vs observations (1,C,H,W)
                        fc = pred_phys_members.unsqueeze(0)
                        sum_crps[k - 1] += crps_metric(fc, truth_phys).detach().cpu().numpy()
                        sum_spread[k - 1] += spread_metric(fc, truth_phys).detach().cpu().numpy()
                        # SSR is POOLED across anchors, not averaged: accumulate the area-weighted
                        # spread and skill components and form the ratio once, at the end. The
                        # per-anchor ratio blows up wherever skill ~ spread/E (the eps clamp takes
                        # over), so a mean of ratios is dominated by a few well-predicted anchors.
                        # components come back BEFORE the metric's batch reduction, i.e. (B, C)
                        # rather than the (C,) the reduced ratio returns. Sum over the batch dim:
                        # B == 1 here (one anchor per call), and summing is what pooling wants
                        # anyway, so this stays correct if the harness ever batches anchors.
                        sp_c, sk_c = ssr_metric(fc, truth_phys, return_components=True)
                        sum_ssr_spread[k - 1] += sp_c.sum(dim=0).detach().cpu().numpy()
                        sum_ssr_skill[k - 1] += sk_c.sum(dim=0).detach().cpu().numpy()
                        sum_rank[k - 1] += rank_metric(fc, truth_phys).detach().cpu().numpy()

                    # power spectrum via makani's buffer; expects data (B,E,C,H,W), targ (B,1,C,H,W)
                    if E > 1:
                        # columns: [ensemble-mean field, member 1..E] (buffer prepends truth)
                        ens_stack = torch.cat([pred.mean(dim=0, keepdim=True), pred], dim=0).unsqueeze(0)
                        spectrum_buffer.update(ens_stack, truth_outnorm.unsqueeze(1), idt=k - 1)
                    else:
                        spectrum_buffer.update(pred.unsqueeze(1), truth_outnorm.unsqueeze(1), idt=k - 1)

                    if slot is not None:
                        # ensemble mode: dump the mean (deterministic-comparable), one member
                        # (--fields_member N — a physically-plausible SAMPLE, where the FCN3
                        # sharpness claim is visible), or EVERY member (--fields_member all, the
                        # ensemble-analysis path: spread, EOFs, quantiles). attrs record which.
                        if E == 1 or args.fields_member == "mean":
                            pf = pred_phys[0, dump_sel_idx]                            # (Cd,H,W)
                        elif args.fields_member == "all":
                            pf = pred_phys_members[:, dump_sel_idx]                    # (E,Cd,H,W)
                        else:
                            pf = pred_phys_members[int(args.fields_member), dump_sel_idx]
                        dump.write_frame(slot, k,
                                         pf.detach().cpu().numpy(),
                                         truth_phys[0, dump_sel_idx].detach().cpu().numpy(),
                                         _date_at(int(i) - k))

                    # feed the prediction back as the next input. n_history=0, so the next input
                    # is just this frame; round-trip out-norm -> physical -> in-norm so the state
                    # stays in INPUT normalization. A no-op for the prototype (in == out norm) but
                    # correct if in != out. (makani's own AR path assumes in == out and
                    # skips this round-trip; we keep it explicit. NB: a full in != out backward
                    # rollout would need more than this -- tendency targets, n_history, etc.)
                    # Ensemble mode: each MEMBER feeds itself (E independent trajectories) —
                    # never feed the mean back, that would collapse the ensemble.
                    state = renorm_in(pred_phys_members)

                n_used += 1
                try:
                    if ai == 0:
                        first_date = str(dataset.get_time_at_index(int(i)))
                    last_date = str(dataset.get_time_at_index(int(i)))
                except Exception:
                    pass

    # finalize the spectra (writes the HDF5 in the same layout as the forward baseline)
    spectrum_buffer.finalize()
    if dump is not None:
        dump.close()

    if world_rank != 0:
        comm.cleanup()
        return

    sum_rmse /= n_used
    sum_acc /= n_used
    sum_pers /= n_used
    sum_clim /= n_used
    if E > 1:
        sum_crps /= n_used
        sum_spread /= n_used
        sum_rank /= n_used
        # pooled SSR: the anchor counts cancel in the ratio, so the sums are used directly.
        sum_ssr = np.sqrt(
            sum_ssr_spread / np.maximum(sum_ssr_skill - sum_ssr_spread / float(E), 1e-6)
        )

    # high-degree TAIL-POWER RATIO vs lead from the buffer's running mean.
    # running_mean[idt] has shape (C, cols, lmax); update() does cat([targ, data], dim=1), so
    # column 0 = truth and column 1 = the prediction (deterministic) / the ensemble-MEAN field
    # (ensemble mode; columns 2.. are the members). Either way column 1 is the point-prediction
    # analog, so tail_ratio stays comparable across deterministic and ensemble runs.
    # l0 = upper half of the spectrum = "small scales".
    mean = spectrum_buffer.running_mean.detach().cpu().numpy()  # (R, C, cols, lmax)
    lmax = mean.shape[-1]
    l0 = lmax // 2
    tail_truth = mean[:, :, 0, l0:].sum(axis=-1)               # (R, C)
    tail_pred = mean[:, :, 1, l0:].sum(axis=-1)                # (R, C)
    tail_ratio = tail_pred / np.clip(tail_truth, 1e-30, None)  # (R, C)
    if E > 1:
        # per-member tail power averaged over members: do SAMPLES keep realistic small scales
        # even where the mean blurs? (the FCN3-specific readout)
        tail_members = mean[:, :, 2:, l0:].sum(axis=-1).mean(axis=-1)     # (R, C)
        tail_ratio_members = tail_members / np.clip(tail_truth, 1e-30, None)

    rmse_over_clim = sum_rmse / np.clip(sum_clim, 1e-30, None)  # (R, C); >1 => worse than climatology

    leads_h = [int((k + 1) * step_h) for k in range(R)]

    # ---- pretty summary ----
    sep = "-" * 110
    print(sep)
    print(f"BACKWARD ROLLOUT  |  config={args.config}  run_num={args.run_num}  "
          f"checkpoint_select={args.checkpoint_select}")
    print(f"checkpoint_path={params.checkpoint_path}")
    print(f"anchors={n_used}  step=-{step_h} h  R={R} (deepest lead -{R*step_h} h)  "
          f"ensemble_size={E}  dates: {first_date} .. {last_date}")
    if E > 1:
        print(f"spectrum_file={spectrum_path}  (cols: 0 = truth, 1 = ensemble-mean field, "
              f"2..{E+1} = members; lmax={lmax}, tail l>={l0})")
    else:
        print(f"spectrum_file={spectrum_path}  (ens idx 0 = truth, 1 = prediction; lmax={lmax}, tail l>={l0})")
    if fields_path:
        print(f"fields_file={fields_path}  (raw pred/truth snapshots -> analysis/make_backward_video.py)")
    print(sep)

    # per-channel table at the deepest lead
    print(f"At deepest lead (-{R*step_h} h):")
    print(f"{'channel':<8} {'rmse':>11} {'persist':>11} {'clim':>11} {'rmse/clim':>10} "
          f"{'acc':>7} {'tail_pred/tru':>14} {'tail_max':>9}  signature")
    sigs = []
    for c in range(C):
        sig = _signature(rmse_over_clim[-1, c], float(tail_ratio[:, c].max()), tail_ratio[-1, c])
        sigs.append(sig)
        print(f"{channel_names[c]:<8} {sum_rmse[-1,c]:>11.4g} {sum_pers[-1,c]:>11.4g} "
              f"{sum_clim[-1,c]:>11.4g} {rmse_over_clim[-1,c]:>10.3f} {sum_acc[-1,c]:>7.3f} "
              f"{tail_ratio[-1,c]:>14.3g} {float(tail_ratio[:,c].max()):>9.3g}  {sig}")
    print(sep)

    if E > 1:
        # ensemble diagnostics at the deepest lead. outer_mass = fraction of area-weighted
        # truth ranks landing in the two OUTER histogram bins, over the uniform expectation
        # 2/(E+1): ~1 = calibrated, >1 = under-dispersed (U-shape),
        # <1 = over-dispersed (dome).
        uniform_outer = 2.0 / (E + 1)
        print(f"Ensemble diagnostics at deepest lead (-{R*step_h} h)  [crps_type={args.crps_type}; "
              f"rmse/acc above are on the ensemble MEAN]:")
        print(f"{'channel':<8} {'crps':>11} {'spread':>11} {'ssr':>7} {'tail_members':>13} {'outer_mass':>11}")
        for c in range(C):
            outer = (sum_rank[-1, c, 0] + sum_rank[-1, c, -1]) / uniform_outer
            print(f"{channel_names[c]:<8} {sum_crps[-1,c]:>11.4g} {sum_spread[-1,c]:>11.4g} "
                  f"{sum_ssr[-1,c]:>7.3f} {tail_ratio_members[-1,c]:>13.3g} {outer:>11.3f}")
        print("  ssr ~1 calibrated, <1 under-dispersed; outer_mass ~1 calibrated, >1 under-dispersed (U-shape).")
        print(sep)

    # lead-by-lead trace for the dynamical channels (where the signature is clearest)
    sel = sorted(set([0, max(0, R // 4 - 1), max(0, R // 2 - 1), max(0, 3 * R // 4 - 1), R - 1]))
    for c in sig_idx:
        name = channel_names[c]
        print(f"{name}:  lead-h |   rmse  | rmse/clim |  acc  | tail_pred/truth")
        for k in sel:
            print(f"      {leads_h[k]:>7} | {sum_rmse[k,c]:>6.4g} | {rmse_over_clim[k,c]:>9.3f} | "
                  f"{sum_acc[k,c]:>5.3f} | {tail_ratio[k,c]:>6.3g}")
    print(sep)

    dyn_sig = _signature(
        float(np.mean(rmse_over_clim[-1, sig_idx])),
        float(np.max(tail_ratio[:, sig_idx])),
        float(np.mean(tail_ratio[-1, sig_idx])),
    )
    print(f"signature channels {sig_set}: tentative signature = {dyn_sig}")
    print("  INSTABILITY -> small-scale power runs hot / RMSE exceeds the climatology ceiling.")
    print("  BLUR        -> small-scale power depleted; RMSE saturates at or below the ceiling.")
    print("  (Advisory only. Inspect the spectra in the HDF5 and the RMSE-vs-lead curves.)")
    print(sep)

    summary = {
        "config": args.config,
        "run_num": str(args.run_num),
        "checkpoint_select": args.checkpoint_select,
        "checkpoint_path": params.checkpoint_path,
        "direction": "backward",
        "step_hours": -int(step_h),
        "num_steps": R,
        "num_anchors": int(n_used),
        "date_first": first_date,
        "date_last": last_date,
        "lead_hours_into_past": leads_h,
        "channel_names": channel_names,
        "dynamical_channels": dyn_set,
        "spectrum_file": spectrum_path,
        "spectrum_layout": (
            f"mean[lead, channel, col, l]; col 0 = truth, 1 = ensemble-mean field, 2..{E+1} = members"
            if E > 1 else "mean[lead, channel, ens, l]; ens 0 = truth, ens 1 = prediction"
        ),
        "fields_file": fields_path,
        "tail_l0": int(l0),
        "lmax": int(lmax),
        # E > 1: rmse/acc/tail_power_ratio are computed on the ENSEMBLE MEAN (the
        # point-prediction analog), so these keys stay comparable to deterministic runs and
        # compare_backward_rollouts.py overlays both kinds unchanged.
        "ensemble_size": int(E),
        "crps_type": args.crps_type if E > 1 else None,
        # precision + seeding are recorded so a run is self-describing: the precision control
        # arm and the pooled-ensemble members are otherwise indistinguishable from a rerun.
        "tf32": bool(args.tf32),
        "amp_mode": getattr(args, "amp_mode", None),
        "noise_seed": int(args.noise_seed) if E > 1 else None,
        # input-noise amplitude actually in force. sigma is what was ASKED for (None = the
        # config value); field_rms is what the constructed noise module MEASURABLY emitted, so
        # a noise-off arm is self-certifying rather than self-asserting.
        "input_noise_sigma": (float(args.input_noise_sigma) if args.input_noise_sigma is not None else None),
        "input_noise_field_rms": noise_rms,
        "ssr_aggregation": "pooled" if E > 1 else None,
        "fields_member": (args.fields_member if (E > 1 and fields_path) else None),
        "per_channel": {
            channel_names[c]: {
                **{
                    "rmse": [float(v) for v in sum_rmse[:, c]],
                    "rmse_persistence": [float(v) for v in sum_pers[:, c]],
                    "rmse_climatology": [float(v) for v in sum_clim[:, c]],
                    "rmse_over_clim": [float(v) for v in rmse_over_clim[:, c]],
                    "acc": [float(v) for v in sum_acc[:, c]],
                    "tail_power_ratio": [float(v) for v in tail_ratio[:, c]],
                    "signature": sigs[c],
                },
                **({
                    "crps": [float(v) for v in sum_crps[:, c]],
                    "spread": [float(v) for v in sum_spread[:, c]],
                    "ssr": [float(v) for v in sum_ssr[:, c]],
                    "tail_power_ratio_members": [float(v) for v in tail_ratio_members[:, c]],
                    "rank_histogram": [[float(v) for v in sum_rank[k, c]] for k in range(R)],
                } if E > 1 else {}),
            }
            for c in range(C)
        },
        "dynamical_signature": dyn_sig,
    }

    out_json = args.output_json or os.path.join(scores_dir, "backward_rollout.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote summary -> {out_json}")

    comm.cleanup()


def main():
    parser = argument_parser.get_default_argument_parser()
    parser.add_argument("--checkpoint_path", default=None, type=str,
                        help="Directory containing the training_checkpoints files. "
                             "Defaults to <exp_dir>/<config>/<run_num>/training_checkpoints.")
    parser.add_argument("--checkpoint_select", default="best", choices=["best", "latest"],
                        help="Load best_ckpt_mp0.tar (lowest val loss) or the latest legacy version.")
    parser.add_argument("--num_steps", default=40, type=int,
                        help="Backward rollout depth R (steps of -dt). 40 * 6 h = 10 days into the past.")
    parser.add_argument("--ensemble_size", default=1, type=int,
                        help="Number of ensemble members E (FCN3 + input noise). 1 (default) = "
                             "the unchanged deterministic rollout; E > 1 folds E members into the batch "
                             "dim, differing only by input noise, and adds CRPS / spread / SSR / rank-"
                             "histogram metrics. Requires a model trained with input_noise (fails loud otherwise).")
    parser.add_argument("--crps_type", default="skillspread", choices=["skillspread", "cdf"],
                        help="CRPS estimator for the eval metric: 'skillspread' = the fair (finite-"
                             "ensemble-unbiased) estimator MetricsHandler uses for validation; 'cdf' = "
                             "the standard estimator (matches the pretrain-1 training loss flavor).")
    parser.add_argument("--noise_seed", default=0, type=int,
                        help="Offset added to the preprocessor's base noise seed. Members "
                             "within ONE run are already independent; this decorrelates SEPARATE runs "
                             "so many E-member rollouts can be POOLED into a large ensemble. Without "
                             "it every run redraws the same noise (base seed is a fixed 333 + rank) and "
                             "pooling silently yields N identical copies of the same E trajectories.")
    parser.add_argument("--input_noise_sigma", default=None, type=float,
                        help="Override the config's input_noise sigma (stationary noise std) at EVAL "
                             "time. None (default) = the config value, i.e. every prior run. The "
                             "STABILITY DIAGNOSTIC arm is '--ensemble_size 1 --input_noise_sigma 0': "
                             "FCN3 injects fresh multi-scale noise as auxiliary inputs at EVERY "
                             "autoregressive step, so a backward blow-up may be the noise seeding "
                             "small scales the map amplifies rather than the map itself. Zeroing the "
                             "amplitude (or stepping it down 1.0/0.5/0.25/0) separates the two. NB "
                             "--ensemble_size 1 alone is NOT noise-off: it is a single NOISY member.")
    parser.add_argument("--tf32", default="on", choices=["on", "off"],
                        help="TF32 matmul/cudnn precision. 'on' (default) preserves the historical "
                             "behavior of every prior run. 'off' is the PRECISION CONTROL arm: combined "
                             "with --amp_mode none it removes the rounding floor entirely, separating a "
                             "numerical high-l artifact from genuine anti-diffusive model instability. "
                             "--amp_mode none ALONE is not a clean control (TF32 stays on).")
    parser.add_argument("--fields_member", default="mean", type=str,
                        help="Ensemble mode field dump: 'mean' (default; deterministic-comparable), a "
                             "member index 0..E-1 (a physically-plausible SAMPLE — where the FCN3 "
                             "sharpness claim is visible in a video), or 'all' to keep EVERY member. "
                             "'all' is the ensemble-ANALYSIS path (spread, EOFs, quantiles) and "
                             "multiplies the dump size by E, so pair it with a deliberate "
                             "--save_fields_channels list rather than 'all' channels.")
    parser.add_argument("--num_anchors", default=128, type=int,
                        help="Number of present-times to roll back from, spread across the valid set.")
    parser.add_argument("--anchor_index", default=None, type=int,
                        help="Roll back from this single dataset index instead of the uniform "
                             "--num_anchors spread (overrides it). Must be >= R and <= n_total-1.")
    parser.add_argument("--anchor_date_exact", action="store_true",
                        help="Require --anchor_date to name a time of day AND land exactly on a "
                             "dataset frame, instead of silently snapping to the frame at/just "
                             "before it. Use whenever the anchor is a chosen event time. Without "
                             "it a bare '2021-06-28' means 00:00 UTC — 17:00 PDT on the 27th, half "
                             "a day off — and an off-grid time rounds down; either way every "
                             "member inherits the error with nothing in the output flagging it.")
    parser.add_argument("--anchor_date", default=None, type=str,
                        help="Roll back from the frame at/just before this UTC date (ISO, e.g. "
                             "2019-09-01 or 2019-09-01T06:00) as a single anchor, overriding "
                             "--num_anchors. For case-study videos (e.g. a hurricane).")
    parser.add_argument("--spectrum_file", default="backward_spectrum.h5", type=str,
                        help="Spectrum HDF5 filename (written under <experiment_dir>/scores/).")
    parser.add_argument("--save_fields_anchors", default=0, type=int,
                        help="Dump raw pred+truth FIELD snapshots for this many anchors (spread across the "
                             "anchor list) to an HDF5 consumed by make_backward_video.py. 0 (default) = off.")
    parser.add_argument("--save_fields_channels", default="all", type=str,
                        help="Which channels to dump: 'all', 'dynamical', or a comma-separated list. "
                             "'all' on the 72-ch config is ~25 GB per anchor at R=40; trim if that matters.")
    parser.add_argument("--fields_file", default="backward_fields.h5", type=str,
                        help="Field-dump HDF5 filename (written under <experiment_dir>/scores/).")
    parser.add_argument("--output_json", default=None, type=str,
                        help="Where to write the JSON summary. Defaults to <experiment_dir>/scores/backward_rollout.json.")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
