#
# Acceptance checks for the normalization stats written by get_stats.py (invoked by
# stats_era5.sbatch after the stats job; standalone so it is unit-testable off-cluster).
# Fails loud (exit 1) on anything that would silently corrupt training:
#   - a missing / wrongly-shaped stats file
#   - non-finite mean/std/min/max (e.g. the all-zero-channel -> NaN-std failure mode)
#   - a zero or negative std (z-score would divide by it)
#   - min >= max on ANY channel (a constant channel; fatal for whichever channels the
#     config's normalization map sends down the minmax path — q*/tcwv in config/rewind_fcn3.yaml —
#     and checked universally so this file cannot drift out of sync with that map)
#   - wind_angle_aware not applied (u/v of a pair must share one magnitude-based std)
# time_means.npy is checked but NaN there only WARNS when imputed timesteps exist:
# get_stats deliberately lets NaN propagate into the climatology ("nan stay nan since
# they are localized in time"), while masking them out of every other reduction.

import argparse
import json
import os
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats_dir", type=str, required=True)
    parser.add_argument("--metadata_file", type=str, required=True)
    parser.add_argument("--dt", type=int, default=1)
    parser.add_argument(
        "--imputed_count_file", type=str, default=None, help="File holding the pre-scanned count of valid_data==0 entries; nonzero downgrades time_means NaN from error to warning."
    )
    args = parser.parse_args()

    with open(args.metadata_file, "r") as f:
        metadata = json.load(f)
    channels = metadata["coords"]["channel"]
    nchan = len(channels)
    nlat, nlon = len(metadata["coords"]["lat"]), len(metadata["coords"]["lon"])

    imputed = 0
    if args.imputed_count_file and os.path.isfile(args.imputed_count_file):
        with open(args.imputed_count_file) as f:
            imputed = int(f.read().strip())

    expected = {
        "global_means.npy": (1, nchan, 1, 1),
        "global_stds.npy": (1, nchan, 1, 1),
        "mins.npy": (1, nchan, 1, 1),
        "maxs.npy": (1, nchan, 1, 1),
        "time_means.npy": (1, nchan, nlat, nlon),
        f"time_diff_means_dt{args.dt}.npy": (1, nchan, 1, 1),
        f"time_diff_stds_dt{args.dt}.npy": (1, nchan, 1, 1),
    }

    errors = []
    stats = {}
    for fname, shape in expected.items():
        path = os.path.join(args.stats_dir, fname)
        if not os.path.isfile(path):
            errors.append(f"{fname}: MISSING")
            continue
        arr = np.load(path)
        if arr.shape != shape:
            errors.append(f"{fname}: shape {arr.shape}, expected {shape}")
            continue
        stats[fname] = arr

    # finiteness: everything except time_means must be finite everywhere
    for fname, arr in stats.items():
        if fname == "time_means.npy":
            continue
        bad = ~np.isfinite(arr)
        if bad.any():
            badch = sorted({channels[c] for c in np.argwhere(bad)[:, 1]})
            errors.append(f"{fname}: {int(bad.sum())} non-finite entries (channels: {badch})")

    # time_means: NaN is an error only when nothing was imputed
    if "time_means.npy" in stats:
        tm = stats["time_means.npy"]
        n_nan = int(np.isnan(tm).sum())
        if n_nan and imputed == 0:
            errors.append(f"time_means.npy: {n_nan} NaN pixels but pre-scan found 0 imputed timesteps")
        elif n_nan:
            print(
                f"WARN time_means.npy: {n_nan} NaN pixels (expected: {imputed} imputed valid_data entries; "
                f"harmless for training, but exclude these from any ACC/climatology use)"
            )

    # stds strictly positive (z-score divides by them)
    for fname in ("global_stds.npy", f"time_diff_stds_dt{args.dt}.npy"):
        if fname in stats:
            bad = stats[fname] <= 0
            if bad.any():
                badch = sorted({channels[c] for c in np.argwhere(bad)[:, 1]})
                errors.append(f"{fname}: non-positive std for channels {badch}")

    # every channel needs min < max (a degenerate range means a constant channel — fatal for
    # whichever channels the config's normalization map sends down the minmax path, and a
    # red flag for the rest); checked for ALL channels so this cannot drift out of sync with
    # the config's minmax set
    if "mins.npy" in stats and "maxs.npy" in stats:
        mins, maxs = stats["mins.npy"], stats["maxs.npy"]
        for c in range(nchan):
            if not mins[0, c, 0, 0] < maxs[0, c, 0, 0]:
                errors.append(f"{channels[c]}: degenerate range, min {mins[0, c, 0, 0]} !< max {maxs[0, c, 0, 0]}")
        # a strongly negative water min flags unit/field confusion (tiny analysis negatives are known in ERA5 q)
        for c, name in enumerate(channels):
            if (name.startswith("q") or name == "tcwv") and mins[0, c, 0, 0] < -1e-4:
                print(f"WARN {name}: min {mins[0, c, 0, 0]:.3e} is well below 0 for a water channel")

    # wind_angle_aware: u/v of each pair share one magnitude std. Mirrors get_wind_channels'
    # sorted-zip pairing (kept numpy-only: importing the helper drags in torch/mpi4py),
    # including its pair-integrity guard so an unpaired wind channel cannot silently
    # truncate the zip.
    if "global_stds.npy" in stats:
        stds = stats["global_stds.npy"]
        u_names = sorted([c for c in channels if c.startswith("u")])
        v_names = sorted([c for c in channels if c.startswith("v")])
        if len(u_names) != len(v_names) or any(u[1:] != v[1:] for u, v in zip(u_names, v_names)):
            errors.append(f"wind channels do not form u/v pairs (u={u_names}, v={v_names}) — get_stats' --wind_angle_aware would have raised")
        else:
            for u, v in zip(u_names, v_names):
                su, sv = stds[0, channels.index(u), 0, 0], stds[0, channels.index(v), 0, 0]
                if su != sv:
                    errors.append(f"wind_angle_aware not applied: std({u})={su} != std({v})={sv}")

    # physical sanity spot-values (loose bounds; catches channel-order scrambles)
    if "global_means.npy" in stats:
        means = stats["global_means.npy"][0, :, 0, 0]
        for name, lo, hi in (("t2m", 250.0, 300.0), ("z500", 4.5e4, 6.0e4), ("msl", 9.0e4, 1.1e5)):
            val = float(means[channels.index(name)])
            if not lo <= val <= hi:
                errors.append(f"global mean of {name} = {val:.4g}, outside plausible [{lo:g}, {hi:g}]")

    if errors:
        print("STATS ACCEPTANCE: FAIL")
        for e in errors:
            print(f"  ERROR {e}")
        sys.exit(1)
    print(f"STATS ACCEPTANCE: PASS ({len(expected)} files, {nchan} channels, " f"{len([c for c in channels if c.startswith('u')])} wind pairs angle-aware)")


if __name__ == "__main__":
    main()
