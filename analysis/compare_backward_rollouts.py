#!/usr/bin/env python3
# ============================================================================
"""Cross-run comparison of backward-rollout diagnostics.

Why this exists: the per-run plots (analysis/plot_backward_rollout.py) show ONE model's
skill/blur vs lead. Questions about a recipe -- does more data, a different noise amplitude
or a different checkpoint move the failure signature -- are about how the curves move
ACROSS runs. This overlays several runs on shared axes and prints a saturation verdict at
a chosen (intermediate) lead:

  * flat lines / vanishing run-to-run gain along the ordering  => SATURATED.
  * lines still separating at the last run                     => still moving.

It reads ONLY each run's ``backward_rollout.json`` (the exact schema backward_rollout.py
writes and plot_backward_rollout.py reads -- so producer and consumer can't drift): per channel
it already stores ``acc``, ``rmse_over_clim`` and ``tail_power_ratio`` as length-R arrays over
``lead_hours_into_past``. No HDF5 needed. Ensemble runs are scored on their ensemble MEAN
and overlay deterministic runs unchanged; their ensemble metrics are read when present.

**Matched horizons are mandatory** -- comparing runs at different R (num_steps) or different
lead grids is meaningless, so the tool FAILS LOUD if the ``lead_hours_into_past`` arrays don't
match across runs. (This is the exact footgun that silently corrupts a cross-run figure.)

Metric directions for reading the verdict:
  * acc               -- higher is better (more retrodictive skill).
  * rmse_over_clim     -- lower is better; ~1.0 = the climatology / blur floor.
  * tail_power_ratio   -- small-scale P_pred/P_truth at l>=tail_l0; **1.0 is ideal**. <1 = blur
                         (small-scale deficit); >1 = anti-diffusive excess. "Improvement" =
                         moving toward 1.0.

Usage
-----
Each RUN arg is ``label=<scores_dir_or_json>``. Pass the runs in increasing order of the
quantity being varied (the label's leading integer sets the color gradient + the
saturation delta ordering; a ``<n>yr`` token is preferred when present):

  python compare_backward_rollouts.py \
      "sigma0=$R/runs/<target>/<run>/scores_sigma0" \
      "sigma1=$R/runs/<target>/<run>/scores" \
      --lead_hours 72 --out $R/runs/compare

Prints the verdict table headless; pass --out to also write one PNG per metric. Run in the
container (matplotlib baked in), e.g. via crun:
  crun python /opt/makani/analysis/compare_backward_rollouts.py a=... b=... --out ...
"""

import argparse
import json
import math
import os
import re
import sys

# Deterministic-comparable metrics: present in EVERY backward_rollout.json (E=1 and E>1 alike,
# since ensemble runs score these on the ensemble MEAN). Only these drive the stability scan, so
# divergence onset is measured the same way regardless of ensemble size.
_DET_METRICS = ("acc", "rmse_over_clim", "tail_power_ratio")
# Ensemble-only metrics: written only when E > 1. A run that lacks them is SKIPPED for those rows
# rather than failing, so a mixed E=1 / E>1 comparison still works for the shared metrics.
_ENS_METRICS = ("crps", "spread", "ssr", "tail_power_ratio_members")
_METRICS = _DET_METRICS + _ENS_METRICS
_METRIC_REF = {"acc": 0.0, "rmse_over_clim": 1.0, "tail_power_ratio": 1.0,
               "crps": 0.0, "spread": 0.0, "ssr": 1.0, "tail_power_ratio_members": 1.0}
_METRIC_DESC = {
    "acc": "ACC (higher = more skill)",
    "rmse_over_clim": "RMSE / climatology (<1 beats climo; ~1 = blur floor)",
    "tail_power_ratio": "high-l P_pred/P_truth (1 = ideal; <1 blur, >1 anti-diffusive)",
    "crps": "CRPS, physical units (lower = better; ENSEMBLE runs only)",
    "spread": "ensemble spread, physical units (ENSEMBLE runs only)",
    "ssr": "spread-skill ratio (1 = calibrated, <1 under-dispersed; ENSEMBLE runs only)",
    "tail_power_ratio_members": "high-l P/P_truth on MEMBERS, not the mean (ENSEMBLE runs only)",
}
# below this ABSOLUTE change (in metric units) a data-doubling counts as "no change". All three
# metrics are O(0.1-1.5), so 1e-4 is ~<0.1% of a typical value — used so the SAT/MOV ratio can't
# read MOV off two sub-noise deltas (a false "data-limited" on an already-flat metric).
_FLAT_ABS = 1e-4

# divergence ceilings: a metric above this (or non-finite) means the rollout has numerically blown
# up (anti-diffusive instability), NOT that the model is merely poor. Bounded blur saturates at
# rmse_over_clim ~1 and the anti-diffusive excess in a STABLE run stays tail_power_ratio ~O(1-2)
# (prototype z500 ~1.08); these ceilings are far above that, so they trip only on real divergence.
# acc is a bounded correlation in [-1, 1] (Cauchy-Schwarz); a FINITE value past ~1 is numerical
# corruption. A real blow-up usually shows first as NaN (already caught by the finiteness test),
# so this is a secondary net — 1.5 catches an invalid finite acc while leaving margin over the
# legitimate acc ~= 1.0 at the first (lead ~0) step (roundoff there is ~1e-6, nowhere near 1.5).
# Only the deterministic-comparable metrics carry ceilings: crps/spread are in physical units with
# no universal scale, and ssr is a ratio whose blow-up behavior is not a reliable divergence signal.
# The scan uses _DET_METRICS, so an ensemble metric never needs one; plots fall back to +inf (mask
# nothing but non-finite values).
_DIVERGENCE_CEIL = {"acc": 1.5, "rmse_over_clim": 5.0, "tail_power_ratio": 10.0}
_PLOT_CEIL = dict(_DIVERGENCE_CEIL, crps=float("inf"), spread=float("inf"),
                  ssr=float("inf"), tail_power_ratio_members=float("inf"))


def _first_divergence(vals, ceil):
    """First index where vals goes non-finite or |vals| exceeds ceil, else None (stayed bounded)."""
    for i, v in enumerate(vals):
        if not math.isfinite(v) or abs(v) > ceil:
            return i
    return None


def _resolve_json(path):
    """A run arg may point at a scores DIR or the JSON itself."""
    if os.path.isdir(path):
        path = os.path.join(path, "backward_rollout.json")
    if not os.path.isfile(path):
        raise SystemExit(f"backward_rollout.json not found: {path}")
    return path


def _parse_run_arg(arg):
    """'label=path' -> (label, loaded_json_dict). label default = the run-dir name."""
    if "=" in arg:
        label, path = arg.split("=", 1)
    else:
        label, path = None, arg
    resolved = _resolve_json(path)  # normalize dir-or-json to the JSON file FIRST, so the
    if label is None:               # auto-label is consistent regardless of which form was passed
        # runs/<config>/<run_num>/scores/backward_rollout.json -> the <config> dir name (unique per
        # rung, carries the yr token). Always 3 dirnames up from the resolved JSON file.
        label = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(resolved))))) or os.path.basename(resolved)
    with open(resolved) as f:
        return label, json.load(f)


def _data_key(label):
    """Sort/color key = the training-YEARS count. Prefer an explicit '<n>yr' token (so a config
    name like 'run_72ch_10yr' keys on 10, not the 72 in '72ch'); else the first
    integer anywhere; else +inf (unknown sorts last)."""
    m = re.search(r"(\d+)\s*yr\b", label)
    if m:
        return int(m.group(1))
    m = re.search(r"\d+", label)
    if m:
        return int(m.group())
    return float("inf")


def _check_matched_leads(runs):
    """FAIL LOUD unless every run shares the same lead grid (else a comparison is meaningless)."""
    ref_label, ref = runs[0]
    ref_leads = [float(x) for x in ref["lead_hours_into_past"]]
    if not ref_leads:
        raise SystemExit(f"run '{ref_label}' has an empty lead_hours_into_past — nothing to compare.")
    for label, S in runs[1:]:
        leads = [float(x) for x in S["lead_hours_into_past"]]
        matched = len(leads) == len(ref_leads) and all(abs(a - b) <= 1e-6 + 1e-6 * abs(b) for a, b in zip(leads, ref_leads))
        if not matched:
            raise SystemExit(
                f"horizon mismatch: '{label}' has {len(leads)} leads "
                f"(num_steps={S.get('num_steps')}) vs '{ref_label}' {len(ref_leads)} "
                f"(num_steps={ref.get('num_steps')}). Re-run the odd one out at the matched R."
            )
    return ref_leads


def _resolve_channels(runs, spec):
    """Channel list from the spec, intersected with what every run actually has (fail loud)."""
    label0, S0 = runs[0]
    if spec == "dynamical":
        want = S0.get("dynamical_channels") or S0["channel_names"]
    elif spec == "all":
        want = S0["channel_names"]
    else:
        want = [c.strip() for c in spec.split(",") if c.strip()]
    for label, S in runs:
        have = set(S["per_channel"])
        missing = [c for c in want if c not in have]
        if missing:
            raise SystemExit(f"run '{label}' is missing requested channel(s): {missing}")
    return want


def _series(S, channel, metric):
    """Per-lead values, or None if this run does not carry the metric (E=1 runs have no
    crps/spread/ssr/tail_power_ratio_members). Callers SKIP a None rather than failing, so a
    mixed E=1 / E>1 comparison still works for the metrics both kinds share."""
    vals = S["per_channel"][channel].get(metric)
    if vals is None:
        return None
    return [float(v) for v in vals]


def print_verdict(runs, channels, leads, lead_hours):
    """Table of each metric at the lead closest to lead_hours, across rungs, + a saturation flag."""
    li = min(range(len(leads)), key=lambda i: abs(leads[i] - lead_hours))
    actual = leads[li]
    order = sorted(range(len(runs)), key=lambda i: _data_key(runs[i][0]))
    labels = [runs[i][0] for i in order]
    print(f"\n=== saturation verdict @ lead {actual:.0f} h into the past "
          f"(step {li + 1}/{len(leads)}; requested {lead_hours:.0f} h) ===", file=sys.stderr)
    print(f"rungs in data order: {labels}", file=sys.stderr)
    if any(_data_key(lbl) == float('inf') for lbl in labels):
        print("  (warning: a label has no integer -> data ordering / Δ may be unreliable)", file=sys.stderr)

    # STABILITY scan first — a rollout that numerically blows up (anti-diffusive instability) makes
    # the metric tables below meaningless past the blow-up lead, and is itself a first-order finding
    # (instability is a stability problem, blur a formulation problem). Report, per run, the
    # earliest lead at which ANY compared channel diverges (non-finite or past the metric ceiling).
    print("\n[stability]  earliest divergence across the compared channels (blow-up = anti-diffusive instability):", file=sys.stderr)
    onset_h = {}
    for i in order:
        label, S = runs[i]
        first = None
        for metric in _DET_METRICS:   # shared by E=1 and E>1 -> onset measured the same way
            for ch in channels:
                d = _first_divergence(_series(S, ch, metric), _DIVERGENCE_CEIL[metric])
                if d is not None and (first is None or d < first[0]):
                    first = (d, ch, metric)
        if first is None:
            onset_h[label] = None
            print(f"    {label:<8} STABLE to -{leads[-1]:.0f} h", file=sys.stderr)
        else:
            d, ch, metric = first
            onset_h[label] = leads[d]
            print(f"    {label:<8} DIVERGES at -{leads[d]:.0f} h (step {d + 1}/{len(leads)}; "
                  f"first in {ch}/{metric})", file=sys.stderr)
    if any(v is not None and v <= actual for v in onset_h.values()):
        bad = [label for label, v in onset_h.items() if v is not None and v <= actual]
        print(f"  *** WARNING: the verdict lead -{actual:.0f} h is AT/PAST divergence for {bad} — "
              f"their rows below are post-blow-up garbage. Re-run --lead_hours below the onset.", file=sys.stderr)

    for metric in _METRICS:
        # an ensemble-only metric is reported over just the runs that carry it
        has = [i for i in order if _series(runs[i][1], channels[0], metric) is not None]
        if not has:
            continue
        sub_labels = [runs[i][0] for i in has]
        print(f"\n[{metric}]  {_METRIC_DESC[metric]}", file=sys.stderr)
        if len(has) < len(order):
            skipped = [runs[i][0] for i in order if i not in has]
            print(f"  (ensemble-only metric; not present in {skipped} — those runs are E=1)", file=sys.stderr)
        header = "  " + "channel".ljust(10) + "".join(f"{lbl:>11}" for lbl in sub_labels) + "   Δlast   sat?"
        print(header, file=sys.stderr)
        for ch in channels:
            vals = [_series(runs[i][1], ch, metric)[li] for i in has]
            row = "  " + ch.ljust(10) + "".join(f"{v:>11.4f}" for v in vals)
            # saturation proxy: |change over the last data-doubling| vs |change over the prior one|
            note = ""
            if len(vals) >= 3:
                d_last = vals[-1] - vals[-2]
                d_prev = vals[-2] - vals[-3]
                if max(abs(d_last), abs(d_prev)) < _FLAT_ABS:
                    # both doublings moved less than the noise floor -> flat, not "still moving"
                    # (the ratio of two sub-noise deltas is meaningless and can read MOV spuriously)
                    flag = "SAT"
                else:
                    ratio = abs(d_last) / (abs(d_prev) + 1e-12)
                    flag = "SAT" if ratio < 0.34 else ("~" if ratio < 0.67 else "MOV")
                note = f"{d_last:>+8.4f}  {flag}"
            elif len(vals) >= 2:
                note = f"{vals[-1] - vals[-2]:>+8.4f}   --"
            print(row + "   " + note, file=sys.stderr)
    print("\n  sat? = |Δ last doubling| / |Δ previous doubling|: SAT <0.34 (flattening), "
          "MOV >0.67 (still data-limited), ~ in between. Advisory — read the curves.", file=sys.stderr)


def make_figures(runs, channels, leads, out_dir, metrics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = sorted(range(len(runs)), key=lambda i: _data_key(runs[i][0]))
    cmap = plt.get_cmap("viridis")
    colors = {order[k]: cmap(0.12 + 0.76 * (k / max(1, len(order) - 1))) for k in range(len(order))}

    os.makedirs(out_dir, exist_ok=True)
    written = []
    for metric in metrics:
        C = len(channels)
        ncol = min(4, C)
        nrow = math.ceil(C / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.0 * nrow), squeeze=False)
        ceil = _PLOT_CEIL[metric]
        any_diverged = False
        for j, ch in enumerate(channels):
            ax = axes[j // ncol][j % ncol]
            for i in order:
                label, S = runs[i]
                vals = _series(S, ch, metric)
                if vals is None:
                    continue      # ensemble-only metric, E=1 run
                # mask blow-up (non-finite or past the divergence ceiling) to NaN so the line breaks
                # cleanly AT the instability onset — otherwise autoscale keys off 1e13-1e34 values
                # and crushes the STABLE rungs to the axis floor (unreadable). The verdict's
                # [stability] section reports exactly where each rung diverged.
                masked = [v if (math.isfinite(v) and abs(v) <= ceil) else float("nan") for v in vals]
                any_diverged = any_diverged or any(math.isnan(m) for m in masked)
                ax.plot(leads, masked, "-", color=colors[i], lw=1.6, label=label)
            ref = _METRIC_REF[metric]
            ax.axhline(ref, color="0.5", ls=":", lw=1)
            ax.set_title(ch, fontsize=10)
            ax.grid(True, alpha=0.3)
            if j % ncol == 0:
                ax.set_ylabel(metric)
            if j // ncol == nrow - 1:
                ax.set_xlabel("hours into the past")
        for k in range(C, nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        axes[0][0].legend(fontsize=8, title="train years")
        suptitle = f"Backward-rollout comparison — {_METRIC_DESC[metric]}"
        if any_diverged and ceil != float("inf"):
            suptitle += f"  [lines break where a rollout blows up: |value| > {ceil:g}]"
        fig.suptitle(suptitle)
        fig.tight_layout()
        fp = os.path.join(out_dir, f"ladder_{metric}.png")
        fig.savefig(fp, dpi=120)
        plt.close(fig)
        written.append(fp)
        print(f"wrote {fp}", file=sys.stderr)
    return written


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="each: label=<scores_dir_or_backward_rollout.json>")
    p.add_argument("--metric", default="all", choices=("all",) + _METRICS, help="metric to plot (default all)")
    p.add_argument("--channels", default="dynamical", help="'dynamical' (default), 'all', or a comma list")
    p.add_argument("--lead_hours", type=float, default=72.0, help="lead (h into past) for the verdict table")
    p.add_argument("--out", default=None, help="directory for the PNGs (needs matplotlib); omit to only print")
    args = p.parse_args(argv)

    if len(args.runs) < 2:
        raise SystemExit("need at least 2 runs to compare")
    runs = [_parse_run_arg(a) for a in args.runs]
    leads = _check_matched_leads(runs)
    channels = _resolve_channels(runs, args.channels)
    if not channels:
        raise SystemExit("no channels to compare — check --channels (empty list, or the runs have no channel_names).")

    for label, S in runs:
        print(f"[{label}] num_steps={S.get('num_steps')} num_anchors={S.get('num_anchors')} "
              f"channels={len(S.get('channel_names', []))}", file=sys.stderr)
    anchors = {S.get("num_anchors") for _, S in runs}
    if len(anchors) > 1:
        print(f"  warning: runs used different anchor counts {sorted(a for a in anchors if a is not None)} "
              f"— metrics are still per-lead comparable but sampled over different ICs.", file=sys.stderr)

    print_verdict(runs, channels, leads, args.lead_hours)

    if args.out is None:
        print("\nNo --out given; printed the verdict only. Pass --out <dir> to write the overlay PNGs.", file=sys.stderr)
        return
    metrics = _METRICS if args.metric == "all" else (args.metric,)
    make_figures(runs, channels, leads, args.out, metrics)


if __name__ == "__main__":
    main()
