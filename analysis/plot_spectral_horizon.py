#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Case-study diagnostic 4: the BANDED SPECTRAL TAIL RATIO vs lead.

This is the mechanism, and it is where the physics claim lives. The other three
diagnostics (plot_ensemble_horizon.py) measure the NUMERICAL STABILITY horizon -- where
error growth explodes, which is a property of a model, not a statement about the
atmosphere. This one measures the INFORMATION
horizon: the lead beyond which a given spatial scale stops being recoverable at all.
That is tied to forward dissipation, not to our architecture, and it is the thing a
sequence-prior model structurally cannot produce -- you can only ask "how far back is
scale l recoverable" of a model that IS the backward operator.

The prediction: small scales go first, so the horizon SHRINKS WITH l. The tool prints
whether that ordering actually holds rather than assuming it.

WHAT IT READS. Only the per-seed ``backward_spectrum_<tag>_seed<NNN>.h5`` dumps -- never
the per-task JSONs, whose tail_power_ratio is computed on the E=4 ensemble MEAN and does
not pool into a 1000-member number. Each spectrum file carries per-MEMBER columns
(col 0 truth, col 1 ensemble mean, cols 2.. the members), so pooling 250 files gives 1000
independent member spectra, which DO pool exactly.

CENSORING. Same rule as diagnostics 1-3 and for the same reason: a diverged member's band
ratio is 1e10 and one of them destroys a median. Each member is censored from its own
first lead past the tail ceiling, and the surviving fraction is reported alongside every
band curve.

THRESHOLDS. The ratio level defining "no longer recoverable" is a CHOICE, not a field
standard -- unlike rmse/clim = 1 or ACC = 0.6 -- so it is exposed, defaulted to 0.5 (half
the true variance in that band still present), and reported at several levels. The
ordering of the horizons with l, which is the actual claim, does not depend on it.

Usage (globs are arguments; both the pre-tag and the case-tagged filename schemes work):

  S=$MAKANI_DATA/runs/rewind_fcn3_ens_casestudy/fcn3_ens_10yr/scores
  crun python /opt/makani/analysis/plot_spectral_horizon.py \
      "PNW heat dome=$S/backward_spectrum_20210628T1200_seed*.h5" \
      "Hurricane Ian=$S/backward_spectrum_20220930T1800_seed*.h5" \
      --channels z500,t2m,msl --out_dir $S/../diagnostics

Cheap: the spectrum dumps are kilobytes, not terabytes, so the whole 250-file pool reads
in seconds on the cpu partition. Reads inputs "r" only.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ensemble_horizon as eh  # noqa: E402

DEFAULT_MARKS = ["cbottle-video 66 h context=66"]
# Must match plot_ensemble_horizon.py's --channel default: diagnostics 1-4 censor one pool,
# so a different censoring channel here means the two figures disagree about who survived.
CENSOR_CHANNEL_DEFAULT = "t2m"
_XLABEL = "hours from the anchor  (0 = the present; negative = into the past)"


def _fmt_horizon(band, level_key, deepest):
    """Render one band's horizon, keeping the three outcomes distinct AND unambiguous.

    Horizons are quoted as a DEPTH IN HOURS BACK (a positive magnitude), never as a signed
    coordinate: on a signed axis "<= -6 h" reads as "at most -6", which admits -96 h --
    the exact opposite of the intended "no deeper than 6 h back", and the same inversion
    class this module is being fixed for. So:

        -42 h        a measured crossing
        < 6 h back   BEFORE_FIRST: already past the level at the first sampled lead
        > 180 h back NEVER: still on the near side at the deepest lead

    Both bounds name the lead that bounds them, so neither can be mistaken for the other
    or for a measurement. Status is absent on a pre-fix JSON, hence the .get fallback.
    """
    v = band["horizon_h"][level_key]
    s = band.get("horizon_status", {}).get(level_key)
    if s == eh.BEFORE_FIRST:
        return f"< {v:.0f} h back"
    if v is None:
        return f"> {deepest:.0f} h back"
    return f"-{v:.0f} h"


def ordering_verdict(values, statuses, deepest):
    """Does the horizon SHRINK with l across these bands (large-scale first)? Returns
    ``(verdict, shown)`` -- a headless string and the per-band renderings behind it.

    Every band takes part at the value the rollout bounds it to: a measured crossing at its
    crossing, BEFORE_FIRST at the first lead (shallowest), NEVER at the deepest lead. Two
    rules keep this from confirming the prediction for free, both of which it previously
    could:

      * a NEVER band is NOT dropped. Dropping it lets a mid-sequence band that outlives its
        smaller-scale neighbours vanish, and the remaining bands still read monotone -- and
        the documented anti-diffusive high-l power EXCESS makes exactly that a live
        possibility, not a hypothetical.
      * fewer than two MEASURED crossings is UNRESOLVED, never a verdict. Bands pinned to
        the same bound form a flat sequence, which is trivially monotone; without this an
        all-BEFORE_FIRST run would print the falsifiable prediction as confirmed on zero
        measurements.
    """
    if len(values) != len(statuses):
        raise ValueError("values and statuses must be the same length")
    seq, shown, n_measured = [], [], 0
    for v, s in zip(values, statuses):
        if s == eh.CROSSED:
            seq.append(float(v))
            shown.append(f"{float(v):.0f}")
            n_measured += 1
        elif s == eh.BEFORE_FIRST:
            seq.append(float(v))
            shown.append(f"<{float(v):.0f}")
        else:
            seq.append(float(deepest))
            shown.append(f">{float(deepest):.0f}")
    if n_measured < 2:
        return (f"UNRESOLVED -- only {n_measured} measured crossing(s); the rest are bounds", shown)
    shrinks = all(a >= b for a, b in zip(seq, seq[1:]))
    return ("SHRINKS with l (as predicted)" if shrinks else "is NOT monotone in l", shown)


def _parse_labeled(args, what):
    out = {}
    for a in args or []:
        label, val = a.split("=", 1) if "=" in a else (os.path.basename(a).replace("*", "").strip("_"), a)
        if label in out:
            raise SystemExit(f"duplicate {what} label {label!r}")
        out[label] = val
    return out


def _check_provenance(label, files, args, log):
    """Verify this pool is 1000 INDEPENDENT members, the same claim diagnostics 1-3 check.

    ``eh.discover`` only rejects a duplicated FILENAME seed. The stronger check --
    ``noise_seed == 0`` ("do not reseed", so two such tasks emit identical members), a
    filename seed disagreeing with the recorded one, a dump that is not a member pool, a
    mixed ensemble size -- reads file attrs that the spectrum dumps DO NOT CARRY: they are
    written by makani's SpectrumAverageBuffer, which creates datasets and no attributes at
    all. Running check_pool_provenance against them would fail every file on an unparseable
    empty noise_seed, so the attrs are read from the SIBLING FIELD dumps instead, named by
    --provenance. Only the attrs are touched, so this is seconds even for a 5 TB pool.

    Without --provenance this cannot be checked, and it says so rather than implying the
    pool was verified -- the whole point of the machinery is that independence is checkable
    rather than asserted.
    """
    globs = _parse_labeled(args.provenance, "provenance")
    if not globs:
        log(f"[{label}] provenance UNCHECKED: spectrum dumps carry no seed attrs; "
            f"pass --provenance '{label}=<field-dump glob>' to verify the pool is independent draws")
        return
    if label not in globs:
        raise SystemExit(f"--provenance given but no entry for case {label!r}; "
                         f"labels present: {sorted(globs)}")
    fld = eh.discover(globs[label], max_files=args.max_files)
    if [s for s, _ in fld] != [s for s, _ in files]:
        raise SystemExit(f"[{label}] --provenance seeds {[s for s, _ in fld][:5]}... do not match the "
                         f"spectrum seeds {[s for s, _ in files][:5]}... -- these are different pools")
    prov = eh.check_pool_provenance({s: eh.read_attrs(p) for s, p in fld},
                                    strict=not args.allow_bad_provenance)
    log(f"[{label}] provenance: {prov['n_files']} files, {prov['n_distinct_noise_seeds']} distinct "
        f"noise seeds, ensemble_size={prov['ensemble_size']}"
        + (f"  PROBLEMS: {prov['problems']}" if prov["problems"] else ""))


def _load_case(label, pattern, channels, args, log):
    """Pool every seed's per-member banded ratios for each requested channel, and censor
    each member at its own tail-ratio divergence.

    NO UNIT CONVERSION HERE, deliberately. Everything this module reports is a RATIO of
    predicted to true power in a band, so the geopotential m^2/s^2 -> m factor that
    plot_ensemble_horizon.py applies (g, or g**2 for a power spectrum) cancels exactly.
    Converting would be a no-op at best and a double-count at worst; see
    ensemble_horizon.to_display_units.
    """
    files = eh.discover(pattern, max_files=args.max_files)
    log(f"[{label}] {len(files)} spectrum dumps matched {pattern}")
    _check_provenance(label, files, args, log)
    banded = {c: [] for c in channels}
    tails, edges, lmax, nlead, censor_ch = [], None, None, None, None
    for seed, path in files:
        spec = eh.load_member_spectra(path)
        if edges is None:
            lmax = spec["lmax"]
            edges = eh.band_edges(lmax, args.n_bands, args.l_min)
            nlead = spec["mean"].shape[0]
            # Censor on the SAME channel diagnostics 1-3 use, so the survival curve printed
            # under these band figures is the one printed under the spaghetti figure. The old
            # default (channels[0], i.e. whatever --channels listed first) silently produced a
            # DIFFERENT survivor set from the identical pool -- two figures, two survival
            # curves, same data.
            censor_ch = args.censor_channel or (CENSOR_CHANNEL_DEFAULT
                                                if CENSOR_CHANNEL_DEFAULT in spec["channels"]
                                                else channels[0])
            log(f"[{label}] lmax={lmax}, {len(edges) - 1} bands at l edges {[int(e) for e in edges]}, "
                f"{spec['n_members']} members/file, {nlead} rolled leads")
            log(f"[{label}] censoring on {censor_ch!r}"
                + ("" if args.censor_channel else
                   f" (default; pass --censor_channel to match a different "
                   f"plot_ensemble_horizon.py --channel)"))
        elif spec["lmax"] != lmax or spec["mean"].shape[0] != nlead:
            raise SystemExit(f"[{label}] {path}: lmax/lead count differs from the rest of the pool "
                             f"({spec['lmax']}/{spec['mean'].shape[0]} vs {lmax}/{nlead}) -- these "
                             "dumps are not one ensemble")
        # censoring uses the SAME high-l tail ratio (l >= lmax//2) as diagnostics 1-3, so a
        # member counted dead in the spaghetti figure is dead here too
        tails.append(eh.tail_ratio_members(spec, censor_ch, args.tail_l0))
        for c in channels:
            banded[c].append(eh.banded_ratio_members(spec, c, edges))
    tails = np.concatenate(tails, axis=0)                                 # (M, R)
    alive = np.asarray([eh.make_monotone_alive(eh.first_failure(t, args.tail_ceil), tails.shape[1])
                        for t in tails])                                  # (M, R), in SPECTRUM-row space
    banded = {c: np.concatenate(v, axis=0) for c, v in banded.items()}    # (M, R, B)
    # spectrum row k is field lead k+1: hours = (k+1) * step_hours. The file's own
    # 'lead_time' is in units of the model dt (steps), not hours, so hours come from an
    # explicit --step_hours rather than from an ambiguous field.
    leads_h = (np.arange(tails.shape[1]) + 1) * float(args.step_hours)
    log(f"[{label}] pool: {tails.shape[0]} members from {len(files)} files")
    return {"label": label, "banded": banded, "alive": alive, "leads": leads_h,
            "edges": edges, "n_files": len(files), "n_members": int(tails.shape[0]),
            "survival": alive.sum(axis=0) / float(alive.shape[0])}


def main():
    p = argparse.ArgumentParser(
        description="Diagnostic 4: banded spectral power ratio vs lead -> the information horizon vs scale.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("cases", nargs="+", metavar="LABEL=SPECTRUM_GLOB",
                   help="One per case. QUOTE it so the shell does not expand the glob.")
    p.add_argument("--channels", default="z500",
                   help="Comma-separated channels to band (each must be in the dumps).")
    p.add_argument("--n_bands", default=5, type=int, help="Number of log-spaced spherical-harmonic bands.")
    p.add_argument("--l_min", default=1, type=int,
                   help="Lowest degree banded. l=0 is the domain mean, whose 'ratio' is a mean offset, "
                        "not a scale, so it is excluded by default.")
    p.add_argument("--step_hours", default=6.0, type=float,
                   help="Hours per rollout step. The dumps' lead_time is in model-dt units (steps), "
                        "not hours, so the conversion is explicit here rather than guessed.")
    p.add_argument("--censor_channel", default=None,
                   help="Channel whose high-l tail ratio decides member survival. Default: "
                        f"{CENSOR_CHANNEL_DEFAULT!r} when the dumps carry it (plot_ensemble_horizon.py's "
                        "--channel default, so the two figures censor identically), else the first of "
                        "--channels. Keep it the same as the one used for diagnostics 1-3 so a member "
                        "declared dead in one figure is dead in all of them.")
    p.add_argument("--tail_ceil", default=eh.TAIL_CEIL_DEFAULT, type=float,
                   help="High-l ratio above which a member counts as diverged and is censored.")
    p.add_argument("--tail_l0", default=None, type=int, help="First degree counted as 'small scale' for "
                                                             "censoring (default lmax//2).")
    p.add_argument("--ratio_levels", nargs="+", type=float, default=[0.75, 0.5, 0.25],
                   help="Band-power ratios defining 'no longer recoverable'. NOT a field standard -- "
                        "several are reported so the ORDERING with l, which is the claim, is visibly "
                        "independent of the choice.")
    p.add_argument("--mark", action="append", metavar="LABEL=HOURS", default=None,
                   help=f"Vertical reference line. Default: {DEFAULT_MARKS}. --mark none to draw none.")
    p.add_argument("--max_files", default=None, type=int, help="Use only the first N seeds (smoke test).")
    p.add_argument("--provenance", action="append", metavar="LABEL=FIELDS_GLOB",
                   help="Per-case glob of the SIBLING FIELD dumps, whose attrs carry the noise seeds. "
                        "The spectrum dumps carry no attrs, so without this the pool cannot be shown "
                        "to be independent draws. Repeatable; labels must match the cases.")
    p.add_argument("--allow_bad_provenance", action="store_true",
                   help="Downgrade provenance violations to a warning (they abort by default).")
    p.add_argument("--out_dir", default=None, help="Write PNGs + spectral_horizon.json here.")
    p.add_argument("--dpi", default=140, type=int)
    args = p.parse_args()

    def log(msg):
        print(msg, flush=True)

    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    if not channels:
        raise SystemExit("--channels resolved to nothing")
    marks = []
    raw = DEFAULT_MARKS if args.mark is None else args.mark
    if raw != ["none"]:
        for m in raw:
            lab, _, hrs = m.rpartition("=")
            marks.append((lab or m, float(hrs)))

    cases = [_load_case(lab, g, channels, args, log)
             for lab, g in _parse_labeled(args.cases, "case").items()]
    ref = cases[0]
    for c in cases[1:]:
        if not np.array_equal(c["edges"], ref["edges"]):
            raise SystemExit(f"band edges differ between {ref['label']!r} and {c['label']!r} "
                             f"({[int(e) for e in ref['edges']]} vs {[int(e) for e in c['edges']]}) "
                             "-- different lmax; a "
                             "shared-axis comparison would compare different scales")
        n = min(len(ref["leads"]), len(c["leads"]))
        if not np.allclose(ref["leads"][:n], c["leads"][:n]):
            raise SystemExit(f"lead grids differ between {ref['label']!r} and {c['label']!r}")

    summary = {"channels": channels, "band_edges": [int(e) for e in ref["edges"]],
               "tail_ceil": args.tail_ceil, "ratio_levels": args.ratio_levels,
               "step_hours": args.step_hours, "cases": {}}
    sep = "-" * 100
    for c in cases:
        edges, leads = c["edges"], c["leads"]
        nb = len(edges) - 1
        ent = {"n_members": c["n_members"], "n_files": c["n_files"],
               "lead_hours_into_past": leads.tolist(), "survival": c["survival"].tolist(), "bands": {}}
        print(sep)
        print(f"CASE {c['label']}  |  {c['n_members']} members from {c['n_files']} dumps  "
              f"|  bands at l = {[int(e) for e in edges]}")
        for ch in channels:
            B = c["banded"][ch]                                  # (M, R, B)
            print(f"  channel {ch}: median band power ratio P_member/P_truth over survivors")
            hdr = f"    {'band (l)':>14} " + " ".join(f"{-h:>8.0f}" for h in leads[:: max(1, len(leads) // 8)])
            print(hdr + "   <- hours from the anchor")
            band_out = {}
            for b in range(nb):
                st = eh.survivor_stats(B[:, :, b], c["alive"])
                med = st["q"][1]
                cols = " ".join(f"{med[k]:>8.3f}" for k in range(0, len(leads), max(1, len(leads) // 8)))
                lab = f"[{int(edges[b])},{int(edges[b + 1])})"
                print(f"    {lab:>14} {cols}")
                hzs = {str(lv): eh.horizon_with_status(leads, med, lv) for lv in args.ratio_levels}
                band_out[lab] = {"l_lo": int(edges[b]), "l_hi": int(edges[b + 1]),
                                 "l_center": float(np.sqrt(edges[b] * edges[b + 1])),
                                 "median_ratio": med.tolist(),
                                 "iqr_lo": st["q"][0].tolist(), "iqr_hi": st["q"][2].tolist(),
                                 "horizon_h": {k: v for k, (v, _) in hzs.items()},
                                 "horizon_status": {k: s for k, (_, s) in hzs.items()}}
            print("    recoverability horizon (lead where the median band ratio first drops below a level):")
            for lv in args.ratio_levels:
                cells = []
                for b in range(nb):
                    lab = f"[{int(edges[b])},{int(edges[b + 1])})"
                    cells.append(f"{lab}: " + _fmt_horizon(band_out[lab], str(lv), float(leads[-1])))
                print(f"      ratio < {lv:g}  ->  " + ",  ".join(cells))
            # The falsifiable ordering: small scales should stop being recoverable FIRST.
            # EVERY band takes part, each at the value the rollout actually bounds it to --
            # BEFORE_FIRST at the first lead (shallowest), NEVER at the deepest lead. A
            # NEVER band must NOT be dropped: a mid-sequence one sitting between two
            # measured crossings is a genuine violation of the predicted ordering (and the
            # documented anti-diffusive high-l power EXCESS makes that a live possibility,
            # not a hypothetical), and dropping it would let a non-monotone run print the
            # prediction as confirmed.
            deepest_lead = float(leads[-1])
            for lv in args.ratio_levels:
                labs = [f"[{int(edges[b])},{int(edges[b + 1])})" for b in range(nb)]
                vals = [band_out[lab]["horizon_h"][str(lv)] for lab in labs]
                stats = [band_out[lab]["horizon_status"][str(lv)] for lab in labs]
                verdict, shown = ordering_verdict(vals, stats, deepest_lead)
                print(f"      ordering at ratio < {lv:g}: horizon {verdict} "
                      f"[{', '.join(shown)} h back]")
            ent["bands"][ch] = band_out
        summary["cases"][c["label"]] = ent
    print(sep)

    if len(cases) > 1:
        print("REGIME CONTRAST (a dynamical information horizon must be regime-dependent)")
        for ch in channels:
            for lv in args.ratio_levels:
                for b in range(len(ref["edges"]) - 1):
                    lab = f"[{int(ref['edges'][b])},{int(ref['edges'][b + 1])})"
                    row = []
                    for c in cases:
                        bo = summary["cases"][c["label"]]["bands"][ch][lab]
                        row.append(f"{c['label']}: " + _fmt_horizon(bo, str(lv), float(c["leads"][-1])))
                    print(f"  {ch} band {lab}, ratio < {lv:g}  ->  " + ",  ".join(row))
        print(sep)

    if not args.out_dir:
        print("no --out_dir: numbers only, no figures written.")
        return
    os.makedirs(args.out_dir, exist_ok=True)
    written = _figures(cases, channels, args, marks, summary)
    jp = os.path.join(args.out_dir, "spectral_horizon.json")
    with open(jp, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    written.append(jp)
    print("wrote:")
    for w in written:
        print(f"  {w}")


def _mark_axis(ax, marks):
    for lab, hrs in marks:
        ax.axvline(-abs(hrs), color="0.35", lw=1.0, ls=(0, (4, 3)))
        ax.annotate(lab, xy=(-abs(hrs), 0.02), xycoords=("data", "axes fraction"), rotation=90,
                    fontsize=6, color="0.35", ha="right", va="bottom")


def _figures(cases, channels, args, marks, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    edges = cases[0]["edges"]
    nb = len(edges) - 1
    for ch in channels:
        # --- band ratio vs lead: one panel per band on a SHARED y axis (so "small scales
        # go first" is visible as a shape, not only as a number), over ONE survival panel
        # spanning the width -- the survival curve is per member, not per band, so drawing
        # it nb times would imply a band dependence that does not exist.
        fig = plt.figure(figsize=(3.0 * nb, 5.6))
        gs = fig.add_gridspec(2, nb, height_ratios=[3, 1], hspace=0.28)
        top = [fig.add_subplot(gs[0, b]) for b in range(nb)]
        for b in range(1, nb):
            top[b].sharey(top[0])
            # labelleft with which="both": a log axis draws MINOR tick labels too, and
            # hiding only the major ones leaves the shared axis looking un-shared
            top[b].tick_params(labelleft=False, which="both")
        axs = fig.add_subplot(gs[1, :])
        for b in range(nb):
            ax = top[b]
            for i, c in enumerate(cases):
                st = eh.survivor_stats(c["banded"][ch][:, :, b], c["alive"])
                x = -c["leads"]
                ax.fill_between(x, st["q"][0], st["q"][2], color=f"C{i}", alpha=0.22, lw=0)
                ax.plot(x, st["q"][1], "-", color=f"C{i}", lw=1.6, label=c["label"])
            ax.axhline(1.0, color="k", lw=0.8, ls="--")
            for lv in args.ratio_levels:
                ax.axhline(lv, color="0.7", lw=0.6, ls=":")
            ax.set_yscale("log")
            ax.set_title(f"l in [{int(edges[b])}, {int(edges[b + 1])})", fontsize=9)
            _mark_axis(ax, marks)
            if b == 0:
                ax.set_ylabel("band power ratio\nP_member / P_truth\n(median over survivors)", fontsize=8)
                ax.legend(fontsize=7, loc="lower left")
        for i, c in enumerate(cases):
            axs.plot(-c["leads"], c["survival"], "-", color=f"C{i}", lw=1.3, label=c["label"])
        axs.axhline(0.5, color="0.6", lw=0.8, ls=":")
        _mark_axis(axs, marks)
        axs.set_ylim(-0.03, 1.03)
        axs.set_ylabel("surviving fraction", fontsize=8)
        axs.set_xlabel(_XLABEL, fontsize=8)
        axs.set_title("member survival (shared by every band -- censoring is per member, not per scale)",
                      fontsize=8)
        fig.suptitle(f"{ch}: recoverability by spatial scale (1 = truth; below 1 = power the "
                     f"backward rollout cannot put back)", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fp = os.path.join(args.out_dir, f"spectral_band_ratio_{ch}.png")
        fig.savefig(fp, dpi=args.dpi)
        plt.close(fig)
        written.append(fp)

    # --- the money plot: horizon vs scale, per case, at several ratio levels ---
    # Unresolved bands are NOT dropped silently -- omitting them would make a partially
    # resolved curve look complete -- but the two ways of being unresolved are drawn at
    # OPPOSITE ends, because they are opposite conclusions:
    #   NEVER        -> open marker pinned to the deepest lead   ("horizon >= rollout")
    #   BEFORE_FIRST -> open marker pinned to the shallowest lead ("horizon <= first step")
    # Collapsing BEFORE_FIRST into the NEVER pin is exactly the inversion this fixes: the
    # spectra have no lead-0 row, so a band already blurred by -6 h landed on the deepest
    # pin and the fastest-dying scales plotted as the longest-surviving.
    deepest = max(float(c["leads"][-1]) for c in cases)
    shallowest = min(float(c["leads"][0]) for c in cases)
    fig, axes = plt.subplots(1, len(channels), figsize=(max(7.2, 4.8 * len(channels)), 4.6),
                             squeeze=False)
    markers = ["o", "s", "^", "D"]
    for j, ch in enumerate(channels):
        ax = axes[0][j]
        for i, c in enumerate(cases):
            for k, lv in enumerate(args.ratio_levels):
                xs, ys, censored = [], [], []
                for b in range(nb):
                    lab = f"[{int(edges[b])},{int(edges[b + 1])})"
                    d = summary["cases"][c["label"]]["bands"][ch][lab]
                    h = d["horizon_h"][str(lv)]
                    st = d.get("horizon_status", {}).get(str(lv))
                    xs.append(d["l_center"])
                    # Each case is pinned to ITS OWN rollout depth, not the global max:
                    # a 36 h case whose band never crosses is bounded at 36 h, and drawing
                    # it at another case's 180 h would make the SHORTER rollout look like
                    # it has the deepest horizon -- and would disagree with the stdout
                    # table, which is per case.
                    if st == eh.BEFORE_FIRST:
                        # the BOUND horizon_with_status actually returned, which is the first
                        # FINITE sample -- not leads[0]. They differ whenever the leading rows
                        # are non-finite, and the figure then named a different bound from the
                        # stdout table and the JSON it is supposed to summarise.
                        ys.append(float(c["leads"][0]) if h is None else h)
                        censored.append(True)
                    else:
                        ys.append(float(c["leads"][-1]) if h is None else h)
                        censored.append(h is None)
                mk = markers[k % len(markers)]
                ax.plot(xs, ys, "-", color=f"C{i}", lw=1.1, alpha=0.5 + 0.15 * k)
                cen = np.asarray(censored)
                xs, ys = np.asarray(xs), np.asarray(ys)
                ax.plot(xs[~cen], ys[~cen], mk, color=f"C{i}", ms=5, alpha=0.55 + 0.15 * k,
                        label=f"{c['label']}, ratio < {lv:g}")
                ax.plot(xs[cen], ys[cen], mk, mfc="none", mec=f"C{i}", ms=5, alpha=0.55 + 0.15 * k)
        for lab, hrs in marks:
            ax.axhline(abs(hrs), color="0.35", lw=1.0, ls=(0, (4, 3)))
            ax.annotate(lab, xy=(0.01, abs(hrs)), xycoords=("axes fraction", "data"),
                        fontsize=6, color="0.35", va="bottom")
        ax.axhline(deepest, color="0.75", lw=0.8, ls="-")
        ax.annotate("rollout depth (open markers here: never crossed, horizon >= this)",
                    xy=(0.01, deepest), xycoords=("axes fraction", "data"), fontsize=6,
                    color="0.5", va="top")
        ax.axhline(shallowest, color="0.75", lw=0.8, ls="-")
        ax.annotate("first sampled lead (open markers here: already past, horizon <= this)",
                    xy=(0.01, shallowest), xycoords=("axes fraction", "data"), fontsize=6,
                    color="0.5", va="bottom")
        ax.set_xscale("log")
        ax.set_ylim(0, deepest * 1.08)
        centers = [float(np.sqrt(edges[b] * edges[b + 1])) for b in range(nb)]
        ax.set_xticks(centers)
        ax.set_xticklabels([f"{int(edges[b])}-{int(edges[b + 1])}" for b in range(nb)], fontsize=7)
        ax.minorticks_off()
        ax.set_xlabel("spherical harmonic degree band")
        ax.set_title(ch, fontsize=10)
        if j == 0:
            ax.set_ylabel("recoverability horizon (hours into the past)")
    # one figure-level legend: an in-axes one lands exactly where the descending curves do
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=min(3, max(1, len(lab))), fontsize=6, frameon=False)
    fig.suptitle("Information horizon vs spatial scale\n"
                 "the physics claim: it should SHRINK with l", fontsize=10)
    fig.tight_layout(rect=(0, 0.12, 1, 0.90))
    fp = os.path.join(args.out_dir, "spectral_horizon_vs_scale.png")
    fig.savefig(fp, dpi=args.dpi)
    plt.close(fig)
    written.append(fp)
    return written


if __name__ == "__main__":
    main()
