#!/usr/bin/env python3
"""Generate a small Makani metadata.json for prototyping.

Produces a metadata file with an exact equiangular lat/lon grid and a chosen
subset of channels, so you don't hand-transcribe hundreds of coordinates. The
grid must match the WeatherBench2/ARCO zarr you convert from (with
``--coord_mode match`` the converter selects/reorders the input onto this grid).

Atmospheric channels must form a *regular* grid: every pressure level you include
must carry the same set of variables. Makani's ``get_channel_groups`` rejects e.g.
``z500, t850`` because levels 500 and 850 would then hold different variables. To
use multiple levels, repeat the same vars at each (``z500,t500,u500,v500`` and
``z850,t850,u850,v850``).

Examples
--------
# 8-channel, 6-hourly, full 0.25 deg (matches the 0p25deg WB2 zarr):
    python make_prototype_metadata.py --output metadata_small.json

# coarser 1.5 deg grid (point the converter at a matching 1.5 deg WB2 zarr):
    python make_prototype_metadata.py --deg 1.5 --output metadata_1p5.json
"""

import argparse
import json


def equiangular(deg: float):
    # ERA5/WB2 convention: latitude 90 -> -90, longitude 0 -> 360-deg
    nlat = int(round(180.0 / deg)) + 1
    nlon = int(round(360.0 / deg))
    lat = [round(90.0 - deg * i, 4) for i in range(nlat)]
    lon = [round(deg * i, 4) for i in range(nlon)]
    return lat, lon


# FourCastNet 3 target set (paper Table 1/4): 7 surface + 5 vars x 13 levels = 72
# channels, all prognostic atmospheric state. NB: the paper drops `sp` (surface
# pressure) and keeps `msl` (mean sea level pressure); the released
# examples/metadata.json instead lists 73 (it includes `sp`) -- reconcile against
# the actual released checkpoint if you intend to warm-start. Channel ORDER here
# follows examples/metadata.json (surface, then u/v/z/t/q at each level); confirm
# it matches the checkpoint's expected order before warm-starting.
FCN3_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
FCN3_SURFACE = ["u10m", "v10m", "u100m", "v100m", "t2m", "msl", "tcwv"]
FCN3_ATMO_VARS = ["u", "v", "z", "t", "q"]


def fcn3_72_channels():
    """The exact 72-channel FourCastNet 3 prognostic-state set (Table 4)."""
    chans = list(FCN3_SURFACE)
    for var in FCN3_ATMO_VARS:
        chans += [f"{var}{lvl}" for lvl in FCN3_LEVELS]
    return chans  # 7 + 5*13 = 72


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--deg", type=float, default=0.25, help="equiangular grid spacing in degrees")
    p.add_argument("--dhours", type=int, default=6, help="output cadence in hours (subsamples the zarr)")
    p.add_argument(
        "--channels",
        nargs="+",
        default=["u10m", "v10m", "t2m", "msl", "z500", "t500", "u500", "v500"],
        help="channel names (Makani convention); ignored when --preset is given",
    )
    p.add_argument(
        "--preset",
        choices=["fcn3"],
        default=None,
        help="named channel preset: fcn3 = the exact 72-channel FourCastNet 3 target set (overrides --channels)",
    )
    p.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="channel names to drop from the selected set, e.g. channels absent from the data source "
        "(the WeatherBench2 SUBSET store lacks u100m/v100m -> --preset fcn3 --exclude u100m v100m = 70 "
        "channels; the canonical ARCO store gcp-public-data-arco-era5 HAS them, so the 72-ch default is used)",
    )
    p.add_argument("--output", default="metadata_small.json")
    a = p.parse_args()

    channels = fcn3_72_channels() if a.preset == "fcn3" else a.channels
    if a.exclude:
        unknown = [c for c in a.exclude if c not in channels]
        if unknown:
            raise SystemExit(f"--exclude names not in the selected channel set: {unknown}")
        channels = [c for c in channels if c not in a.exclude]
        print(f"excluded {a.exclude} -> {len(channels)} channels")

    lat, lon = equiangular(a.deg)
    meta = {
        "dataset_name": f"era5-{len(channels)}ch-{a.dhours}h-{a.deg}deg",
        "attrs": {"description": "Prototype slice for backward-emulator development"},
        "h5_path": "fields",
        "dims": ["time", "channel", "lat", "lon"],
        "dhours": a.dhours,
        "coords": {"grid_type": "equiangular", "lat": lat, "lon": lon, "channel": channels},
    }
    with open(a.output, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {a.output}: {len(lat)} lat x {len(lon)} lon, {len(channels)} channels, dhours={a.dhours}")


if __name__ == "__main__":
    main()
