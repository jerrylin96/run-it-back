# SPDX-License-Identifier: Apache-2.0
#
# plot the backward-rollout diagnosis produced by analysis/backward_rollout.py.
#
# Turns the two outputs (backward_rollout.json + backward_spectrum.h5) into the figures that
# read the failure mode:
#   1. RMSE vs lead per dynamical channel, with the PERSISTENCE and CLIMATOLOGY ceilings
#      (instability = blows past them; blur = saturates at climatology from below).
#   2. ACC vs lead.
#   3. Tail-power ratio (small-scale P_pred / P_truth) vs lead (>1 rising = instability;
#      <1 falling = blur).
#   4. Power spectra P(l) at several leads, prediction vs truth (instability = excess at high l;
#      blur = deficit at high l).
#
# ENSEMBLE runs (backward_rollout.py --ensemble_size E>1) are detected from the JSON keys —
# no flag, and deterministic inputs render exactly as before. The mean-scored curves stay the
# control-comparable baseline; on top of them:
#   1. gains the ensemble SPREAD (std) curve — spread ~ RMSE(mean) = calibrated (the spread-skill
#      picture in one panel; per-member RMSE/ACC are deliberately NOT recorded by the rollout).
#   3. gains the MEMBER-average tail ratio (dashed, same color per channel) — the FCN3-specific
#      question: do individual samples keep realistic small-scale power where the mean blurs?
#   4. gains thin per-member spectra at the deepest lead (the sharp-sample vs smooth-mean view).
#   5. backward_ensemble_skill.png — CRPS + spread-skill ratio vs lead (SSR ~ 1 = calibrated).
#   6. backward_rank_histogram.png — rank of truth among members, first vs deepest lead
#      (flat = calibrated; U-shape = underdispersive; dome = overdispersive).
#
# No GPU needed. Needs only numpy + h5py + matplotlib (all present in makani.sif). Run it
# wherever those are available, e.g. locally after copying the two files, or in the container.
#
# Usage:
#   python analysis/plot_backward_rollout.py \
#       --scores_dir $MAKANI_DATA/runs/rewind_fcn3_ens_10yr_backward/fcn3_ens_10yr/scores
#   # optional: --forward_spectrum <a forward makani.inference spectrum.h5>
#   #           overlays the forward truth/pred spectra for context.

import os
import json
import argparse

import numpy as np


def _load_spectrum(path):
    import h5py
    with h5py.File(path, "r") as f:
        mean = f["mean"][...]                       # (R, C, ens=2, lmax): ens 0 = truth, 1 = pred
        chans = [c.decode() if isinstance(c, bytes) else str(c) for c in f["channel"][...]]
        lead = f["lead_time"][...] if "lead_time" in f else np.arange(mean.shape[0]) + 1
    return mean, chans, lead


def _plot_channel_overview(chans, per, leads, out_dir, plt):
    """All-channel skill map (channel x lead) for ACC and RMSE/climatology.

    Scales from a handful of channels to the full 72-channel state: the per-DYNAMICAL-channel
    figures below only ever show ~5 channels, so on the full FCN3 state they leave the per-channel
    retrodiction skill unvisualized.
    Channel order is taken as-is from the config channel_names (already grouped: surface fields,
    then each variable by ascending pressure level), so the y-axis reads as coherent variable blocks.
    """
    C = len(chans)
    nlead = len(leads)
    acc = np.full((C, nlead), np.nan)
    rmse_clim = np.full((C, nlead), np.nan)
    for i, name in enumerate(chans):
        d = per.get(name)
        if not d:
            continue
        a = np.asarray(d.get("acc", []), dtype=float)
        r = np.asarray(d.get("rmse", []), dtype=float)
        cl = np.asarray(d.get("rmse_climatology", []), dtype=float)
        if a.size == nlead:
            acc[i] = a
        if r.size == nlead and cl.size == nlead:
            with np.errstate(divide="ignore", invalid="ignore"):
                rmse_clim[i] = np.where(cl > 0, r / cl, np.nan)

    fig_h = max(3.2, 0.16 * C + 1.4)  # ~2.7in tall at 8 ch, ~13in at 72 ch
    fig, axes = plt.subplots(1, 2, figsize=(12, fig_h), squeeze=True)
    step = max(1, nlead // 8)
    panels = [
        (axes[0], acc, "ACC (higher = more retrodictive skill)", -0.2, 1.0, "viridis"),
        (axes[1], rmse_clim, "RMSE / climatology (<1 beats climo; ~1 = blur floor)", 0.0, 1.2, "magma_r"),
    ]
    for ax, M, title, vmin, vmax, cmap in panels:
        im = ax.imshow(M, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest")
        ax.set_yticks(range(C))
        ax.set_yticklabels(chans, fontsize=(5 if C > 24 else 8))
        ax.set_xticks(range(0, nlead, step))
        ax.set_xticklabels([f"{int(leads[k])}" for k in range(0, nlead, step)], fontsize=7)
        ax.set_xlabel("hours into the past")
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"Backward rollout per-channel skill map ({C} channels)")
    fig.tight_layout()
    fp = os.path.join(out_dir, "backward_channel_overview.png")
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    return fp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores_dir", required=True,
                   help="Directory holding backward_rollout.json and backward_spectrum.h5.")
    p.add_argument("--json", default=None, help="Override path to backward_rollout.json.")
    p.add_argument("--spectrum", default=None, help="Override path to backward_spectrum.h5.")
    p.add_argument("--forward_spectrum", default=None,
                   help="Optional forward spectrum.h5 to overlay for context.")
    p.add_argument("--out_dir", default=None, help="Where to write PNGs (default: scores_dir).")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    json_path = args.json or os.path.join(args.scores_dir, "backward_rollout.json")
    spec_path = args.spectrum or os.path.join(args.scores_dir, "backward_spectrum.h5")
    out_dir = args.out_dir or args.scores_dir
    os.makedirs(out_dir, exist_ok=True)

    with open(json_path) as f:
        S = json.load(f)
    leads = np.array(S["lead_hours_into_past"], dtype=float)
    chans = S["channel_names"]
    # present-but-EMPTY, not missing, is the real case: a surface-only ablation has no
    # z500/t500/u500/v500/msl, and backward_rollout.py records that faithfully as []. A
    # .get() default only fires on a MISSING key, so the empty list flowed through to
    # plt.subplots(1, 0) below. `or` falls back on both, matching compare_backward_rollouts.py.
    dyn = S.get("dynamical_channels") or chans
    per = S["per_channel"]

    # 0. all-channel skill map (scales from 8 to 72 channels; the money plot for the full state)
    written = [_plot_channel_overview(chans, per, leads, out_dir, plt)]

    # 1. RMSE vs lead with persistence + climatology ceilings
    fig, axes = plt.subplots(1, len(dyn), figsize=(4 * len(dyn), 3.6), squeeze=False)
    for j, name in enumerate(dyn):
        ax = axes[0][j]
        d = per[name]
        ax.plot(leads, d["rmse"], "-o", ms=3, label="backward model (ensemble mean)" if "spread" in d else "backward model")
        ax.plot(leads, d["rmse_persistence"], "--", color="gray", label="persistence (hold present)")
        ax.plot(leads, d["rmse_climatology"], ":", color="k", label="climatology ceiling")
        if "spread" in d:
            # calibrated ensemble: spread tracks RMSE of the mean; spread << rmse = underdispersive
            ax.plot(leads, d["spread"], "-s", ms=2.5, color="C2", label="ensemble spread (std)")
        ax.set_title(f"{name}  ({d['signature']})")
        ax.set_xlabel("hours into the past")
        ax.set_ylabel("area-weighted RMSE")
        if j == 0:
            ax.legend(fontsize=7)
    fig.suptitle("Backward rollout RMSE vs lead (blows past ceiling = instability; "
                 "saturates at climatology = blur)")
    fig.tight_layout()
    f1 = os.path.join(out_dir, "backward_rmse_vs_lead.png")
    fig.savefig(f1, dpi=130)
    plt.close(fig)

    # 2. ACC vs lead
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in dyn:
        ax.plot(leads, per[name]["acc"], "-o", ms=3, label=name)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("hours into the past")
    ax.set_ylabel("ACC (anomaly correlation)")
    ax.set_title("Backward rollout ACC vs lead")
    ax.legend(fontsize=8)
    fig.tight_layout()
    f2 = os.path.join(out_dir, "backward_acc_vs_lead.png")
    fig.savefig(f2, dpi=130)
    plt.close(fig)

    # 3. tail-power ratio vs lead (ensemble: member-average dashed in the channel's color —
    #    members keeping ratio ~1 while the mean falls is the FCN3 sharp-samples signature)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, name in enumerate(dyn):
        col = f"C{i % 10}"
        ax.plot(leads, per[name]["tail_power_ratio"], "-o", ms=3, color=col, label=name)
        if "tail_power_ratio_members" in per[name]:
            ax.plot(leads, per[name]["tail_power_ratio_members"], "--", lw=1.0, color=col)
    if any("tail_power_ratio_members" in per[n] for n in dyn):
        ax.plot([], [], "--", color="gray", label="member average (dashed)")
    ax.axhline(1.0, color="k", lw=0.8, ls="--", label="truth (ratio = 1)")
    ax.set_yscale("log")
    ax.set_xlabel("hours into the past")
    ax.set_ylabel(f"small-scale power ratio  P_pred / P_truth  (l >= {S['tail_l0']})")
    ax.set_title("Tail-power ratio (>1 rising = instability; <1 falling = blur)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    f3 = os.path.join(out_dir, "backward_tail_power_ratio.png")
    fig.savefig(f3, dpi=130)
    plt.close(fig)

    written += [f1, f2, f3]

    # 5+6. ensemble-only figures (key-presence detection: deterministic JSONs skip both)
    ens_dyn = [n for n in dyn if "crps" in per[n]]
    if ens_dyn:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), squeeze=True)
        for i, name in enumerate(ens_dyn):
            col = f"C{i % 10}"
            axes[0].plot(leads, per[name]["crps"], "-o", ms=3, color=col, label=name)
            axes[1].plot(leads, per[name]["ssr"], "-o", ms=3, color=col, label=name)
        axes[0].set_yscale("log")  # CRPS is in each channel's physical units -> decades apart
        axes[0].set_ylabel("CRPS (per-channel physical units)")
        axes[0].set_title("CRPS vs lead (lower = better probabilistic skill)")
        axes[1].axhline(1.0, color="k", lw=0.8, ls="--", label="calibrated (spread = skill)")
        axes[1].set_ylabel("spread-skill ratio")
        axes[1].set_title("SSR vs lead (<1 = underdispersive; >1 = overdispersive)")
        for ax in axes:
            ax.set_xlabel("hours into the past")
            ax.legend(fontsize=7)
        fig.tight_layout()
        f5 = os.path.join(out_dir, "backward_ensemble_skill.png")
        fig.savefig(f5, dpi=130)
        plt.close(fig)
        written.append(f5)

        if "rank_histogram" in per[ens_dyn[0]]:
            fig, axes = plt.subplots(1, len(ens_dyn), figsize=(3.0 * len(ens_dyn), 3.2), squeeze=False)
            for j, name in enumerate(ens_dyn):
                ax = axes[0][j]
                rh = np.asarray(per[name]["rank_histogram"], dtype=float)  # (R, E+1)
                nb = rh.shape[1]
                x = np.arange(nb)
                w = 0.4
                sel_pairs = ((-w / 2, 0, f"-{int(leads[0])} h"), (w / 2, rh.shape[0] - 1, f"-{int(leads[-1])} h"))
                if rh.shape[0] == 1:  # single-step rollout: first == deepest, draw once
                    sel_pairs = ((0.0, 0, f"-{int(leads[0])} h"),)
                for off, k, lab in sel_pairs:
                    tot = max(rh[k].sum(), 1e-12)
                    ax.bar(x + off, rh[k] / tot, width=w, label=lab)
                ax.axhline(1.0 / nb, color="k", lw=0.8, ls=":")  # flat = calibrated
                ax.set_title(name, fontsize=9)
                ax.set_xlabel("rank of truth among members", fontsize=8)
                if j == 0:
                    ax.set_ylabel("frequency")
                    ax.legend(fontsize=7)
            fig.suptitle("Rank histograms, first vs deepest lead (flat = calibrated; U = underdispersive; dome = overdispersive)", fontsize=9)
            fig.tight_layout()
            f6 = os.path.join(out_dir, "backward_rank_histogram.png")
            fig.savefig(f6, dpi=130)
            plt.close(fig)
            written.append(f6)

    # 4. power spectra at several leads, pred vs truth
    if os.path.exists(spec_path):
        mean, schans, slead = _load_spectrum(spec_path)
        R = mean.shape[0]
        sel = sorted(set([0, max(0, R // 4 - 1), max(0, R // 2 - 1), R - 1]))
        ell = np.arange(mean.shape[-1])
        fwd = None
        if args.forward_spectrum and os.path.exists(args.forward_spectrum):
            fwd, fchans, _ = _load_spectrum(args.forward_spectrum)
        for name in dyn:
            if name not in schans:
                continue
            ci = schans.index(name)
            fig, ax = plt.subplots(figsize=(6, 4.4))
            # one truth curve (truth spectrum drifts negligibly across lead; show the deepest)
            ax.loglog(ell[1:], mean[sel[-1], ci, 0, 1:], "-", color="k", lw=1.5, label="truth")
            for k in sel:
                hrs = int(slead[k]) if k < len(slead) else (k + 1)
                ax.loglog(ell[1:], mean[k, ci, 1, 1:], "-", lw=1.0, label=f"pred -{hrs} (step)")
            # ensemble files carry columns [truth, mean, member 1..E] (E+2 wide; deterministic = 2).
            # Thin member curves at the deepest lead: samples holding the truth slope at high l
            # while the mean rolls off is the sharp-samples/smooth-mean signature.
            n_members = mean.shape[2] - 2
            if n_members > 0:
                for m in range(n_members):
                    ax.loglog(ell[1:], mean[sel[-1], ci, 2 + m, 1:], "-", color="C0", alpha=0.35, lw=0.7)
                ax.plot([], [], "-", color="C0", alpha=0.5, lw=0.7, label=f"{n_members} members (deepest)")
            if fwd is not None and name in fchans:
                fi = fchans.index(name)
                ax.loglog(ell[1:], fwd[min(sel[-1], fwd.shape[0]-1), fi, 1, 1:], ":", color="C3",
                          lw=1.0, label="forward pred (deepest)")
            ax.set_xlabel("spherical harmonic degree l")
            ax.set_ylabel("power")
            ax.set_title(f"{name}: backward power spectrum vs lead (pred vs truth)")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fp = os.path.join(out_dir, f"backward_spectrum_{name}.png")
            fig.savefig(fp, dpi=130)
            plt.close(fig)
            written.append(fp)
    else:
        print(f"NOTE: spectrum file not found at {spec_path}; skipped spectrum plots.")

    print("wrote:")
    for w in written:
        print(f"  {w}")


if __name__ == "__main__":
    main()
