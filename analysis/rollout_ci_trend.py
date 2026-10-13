# SPDX-License-Identifier: Apache-2.0
#
# Rollout-CI trend: "does multistep backward skill improve as training progresses?"
# Single-step convergence does NOT imply rollout stability, so rollout skill must be
# tracked DURING training, not discovered after it.
#
# Consumes the per-epoch snapshots written by analysis/rollout_ci_fcn3.sbatch:
#   <scores>/ci/epoch_NNNN/backward_rollout.json     (the standard backward_rollout.py schema)
# and produces:
#   <scores>/ci/rollout_trend.csv    one row per (epoch, channel, lead): acc, rmse_over_clim,
#                                    tail_power_ratio, crps ('' when deterministic)
#   a headless VERDICT on stdout    z500 @ the deepest CI lead across the last 3 evals:
#                                    IMPROVING / FLAT / DEGRADING (advisory, like the
#                                    compare_backward_rollouts saturation flag)
#   <out>.png (optional, --out)     z500 acc + tail_power_ratio vs epoch at mid + deepest lead
#
# stdlib-only for the parse/CSV/verdict path (matplotlib lazy under --out), matching
# plot_loss_from_log.py / compare_backward_rollouts.py. Run via crun (no sbatch needed) or
# let rollout_ci_fcn3.sbatch invoke it after each snapshot.

import argparse
import csv
import glob
import json
import os
import re
import sys

# advisory-verdict knobs (mirror the spirit of compare_backward_rollouts._FLAT_ABS)
_FLAT_ABS = 5.0e-3   # |delta acc| below this across the window counts as FLAT, not a trend
_VERDICT_WINDOW = 3  # trend judged on the last N evals
_VERDICT_CHANNEL = "z500"


def collect(ci_dir):
    """Parse every epoch_*/backward_rollout.json under ci_dir -> list of row dicts."""
    rows = []
    lead_grid = None
    for d in sorted(glob.glob(os.path.join(ci_dir, "epoch_*"))):
        m = re.search(r"epoch_(\d+)$", d)
        jp = os.path.join(d, "backward_rollout.json")
        if not m or not os.path.isfile(jp):
            continue
        epoch = int(m.group(1))
        with open(jp) as f:
            S = json.load(f)
        leads = S.get("lead_hours_into_past", [])
        if not leads:
            raise SystemExit(f"ERROR: {jp} has an empty lead grid — malformed snapshot")
        if lead_grid is None:
            lead_grid = list(leads)
        elif list(leads) != lead_grid:
            # fail loud, like compare_backward_rollouts: a horizon change silently corrupts trends
            raise SystemExit(f"ERROR: lead-grid mismatch at epoch {epoch} ({leads} vs {lead_grid}) — "
                             f"CI num_steps changed mid-run; archive the old ci/ dir and restart the trend")
        sel = sorted(set([max(0, len(leads) // 2 - 1), len(leads) - 1]))  # mid + deepest
        # `or`, not a .get default: a channel subset with no dynamical field records an
        # EMPTY list, which a default cannot catch and which silently yields zero rows.
        for ch in S.get("dynamical_channels") or list(S.get("per_channel", {})):
            p = S["per_channel"].get(ch)
            if not p:
                continue
            for k in sel:
                rows.append({
                    "epoch": epoch,
                    "channel": ch,
                    "lead_h": leads[k],
                    "acc": p["acc"][k],
                    "rmse_over_clim": p["rmse_over_clim"][k],
                    "tail_power_ratio": p["tail_power_ratio"][k],
                    "crps": p["crps"][k] if "crps" in p else "",
                })
    return rows


def verdict(rows, channel=_VERDICT_CHANNEL):
    """Advisory trend on `channel` at the DEEPEST recorded lead over the last _VERDICT_WINDOW evals."""
    ch_rows = [r for r in rows if r["channel"] == channel]
    if not ch_rows:
        return f"VERDICT: no {channel} rows — check the CI channel set"
    deepest = max(r["lead_h"] for r in ch_rows)
    series = sorted([(r["epoch"], r["acc"]) for r in ch_rows if r["lead_h"] == deepest])
    if len(series) < 2:
        return f"VERDICT: {channel}@-{deepest:g}h — only {len(series)} eval(s); trend needs >= 2"
    window = series[-_VERDICT_WINDOW:]
    d = window[-1][1] - window[0][1]
    flag = "FLAT" if abs(d) < _FLAT_ABS else ("IMPROVING" if d > 0 else "DEGRADING")
    pts = ", ".join(f"ep{e}: {a:.3f}" for e, a in window)
    return (f"VERDICT ({channel} acc @ -{deepest:g} h, last {len(window)} evals): {flag} "
            f"(delta {d:+.4f})  [{pts}]\n"
            f"  DEGRADING while training loss falls = the single-step over-sharpening signature — "
            f"inspect the tail_power_ratio column before burning more epochs. Advisory only.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ci_dir", required=True, help="The scores/ci directory holding epoch_*/ snapshots.")
    p.add_argument("--out", default=None, help="Optional PNG path for the z500 trend figure.")
    args = p.parse_args()

    rows = collect(args.ci_dir)
    if not rows:
        raise SystemExit(f"ERROR: no epoch_*/backward_rollout.json snapshots under {args.ci_dir}")

    csv_path = os.path.join(args.ci_dir, "rollout_trend.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "channel", "lead_h", "acc",
                                          "rmse_over_clim", "tail_power_ratio", "crps"])
        w.writeheader()
        w.writerows(rows)
    n_epochs = len({r["epoch"] for r in rows})
    print(f"wrote {csv_path} ({len(rows)} rows, {n_epochs} epochs)")
    print(verdict(rows))

    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ch_rows = [r for r in rows if r["channel"] == _VERDICT_CHANNEL]
        leads = sorted({r["lead_h"] for r in ch_rows})
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), squeeze=True)
        for lead in leads:
            s = sorted([(r["epoch"], r["acc"], r["tail_power_ratio"])
                        for r in ch_rows if r["lead_h"] == lead])
            eps = [x[0] for x in s]
            axes[0].plot(eps, [x[1] for x in s], "-o", ms=3, label=f"-{lead:g} h")
            axes[1].plot(eps, [x[2] for x in s], "-o", ms=3, label=f"-{lead:g} h")
        axes[0].set_ylabel(f"{_VERDICT_CHANNEL} ACC")
        axes[0].set_title("rollout skill vs training epoch")
        axes[1].axhline(1.0, color="k", lw=0.8, ls="--")
        axes[1].set_yscale("log")
        axes[1].set_ylabel("tail power ratio (1 = truth)")
        axes[1].set_title(">1 rising = instability; <1 = blur")
        for ax in axes:
            ax.set_xlabel("training epoch")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out, dpi=130)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
