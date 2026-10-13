# SPDX-License-Identifier: Apache-2.0
#
# cut a CONTIGUOUS time slice out of a converted makani HDF5 year file,
# preserving the schema the multifiles loader depends on.
#
# Why this exists: a full converted year at 72 ch / 6-hourly / 0.25 deg is ~0.44 TB, which is
# painful to ship to a collaborator's cluster when the science only needs a few weeks around
# one event (a 5-week window is ~42 GB). Slicing is SAFE because the loader reads timestamps
# from each file's own HDF5 dimension scale (data_loader_multifiles.py: dims[0][timestamp]),
# NOT from the filename year -- so a correctly-sliced file self-describes its dates and the
# zenith forcing stays aligned. It is only safe if the slice carries those scales, which is
# exactly what this script guarantees (schema mirrored from convert_wb2_to_makani_input.py).
#
# Two loader contracts this script upholds (both fail loud here rather than mid-training):
#   * CONTIGUOUS in time -- the loader raises "time difference between steps is not constant"
#     on any gap, so only whole ranges may be cut (no striding, no holes).
#   * DISJOINT files -- the loader raises on overlapping date ranges across globbed files, so
#     keep one slice per directory (or non-overlapping slices).
#
# Rollout window note: a backward rollout of R steps anchored at date D needs R*dhours of data
# BEFORE D inside the file. Cut the range accordingly (--check_anchor asserts it for you).
#
# Usage:
#   python data_process/subset_time_range.py \
#       --input  /work/.../makani_era5_fcn3/test/2021.h5 \
#       --output /work/.../ship/2021_pnw.h5 \
#       --start 2021-06-01 --end 2021-07-05 \
#       --check_anchor 2021-06-28 --check_steps 40
#
#   # verify a slice someone handed you (no writing):
#   python data_process/subset_time_range.py --input 2021_pnw.h5 --inspect

import argparse
import datetime as dt
import os
import sys

import numpy as np
import h5py as h5

_DEF_DATASET = "fields"


def _utc(ts):
    """Unix seconds (the converter's timestamp convention) -> aware datetime."""
    return dt.datetime.fromtimestamp(float(ts), tz=dt.timezone.utc)


def _parse_day(s, end_of_day=False):
    d = dt.datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    return d.replace(hour=23, minute=59, second=59) if end_of_day else d


def inspect(path, dataset=_DEF_DATASET):
    with h5.File(path, "r") as f:
        d = f[dataset]
        ts = f["timestamp"][...]
        dts = [_utc(t) for t in ts]
        steps = {int((dts[i + 1] - dts[i]).total_seconds() // 3600) for i in range(len(dts) - 1)}
        print(f"{path}")
        print(f"  {dataset}: shape={d.shape} dtype={d.dtype}")
        print(f"  time:   {dts[0].isoformat()} .. {dts[-1].isoformat()}  ({len(ts)} frames)")
        print(f"  cadence: {sorted(steps)} h" + ("  <-- NOT CONTIGUOUS, loader will reject" if len(steps) != 1 else ""))
        print(f"  scales attached: " + ", ".join(k for k in ("timestamp", "channel", "lat", "lon") if k in f))
        gb = d.shape[0] * d.shape[1] * d.shape[2] * d.shape[3] * d.dtype.itemsize / 1024**3
        print(f"  size: {gb:.1f} GiB")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Converted makani year file (e.g. 2021.h5).")
    p.add_argument("--output", default=None, help="Destination slice. Omitted with --inspect.")
    p.add_argument("--start", default=None, help="First day to KEEP, YYYY-MM-DD (UTC, inclusive).")
    p.add_argument("--end", default=None, help="Last day to KEEP, YYYY-MM-DD (UTC, inclusive).")
    p.add_argument("--dataset", default=_DEF_DATASET, help="Field dataset name (default: fields).")
    p.add_argument("--check_anchor", default=None,
                   help="Assert this YYYY-MM-DD anchor keeps a full backward window inside the slice.")
    p.add_argument("--check_steps", default=40, type=int, help="R for --check_anchor (default 40 = -240 h).")
    p.add_argument("--inspect", action="store_true", help="Print the file's time range/cadence and exit.")
    args = p.parse_args()

    if args.inspect:
        return inspect(args.input, args.dataset)
    if not (args.output and args.start and args.end):
        raise SystemExit("ERROR: --output, --start and --end are required unless --inspect is given")
    if os.path.exists(args.output):
        raise SystemExit(f"ERROR: {args.output} exists — refusing to overwrite")

    t0, t1 = _parse_day(args.start), _parse_day(args.end, end_of_day=True)
    if t1 <= t0:
        raise SystemExit(f"ERROR: --end ({args.end}) must be after --start ({args.start})")

    with h5.File(args.input, "r") as src:
        if args.dataset not in src:
            raise SystemExit(f"ERROR: no '{args.dataset}' dataset in {args.input} (have: {list(src)})")
        ts = src["timestamp"][...]
        dts = np.array([_utc(t) for t in ts])
        keep = np.nonzero((dts >= t0) & (dts <= t1))[0]
        if keep.size == 0:
            raise SystemExit(f"ERROR: no frames in [{args.start}, {args.end}]; file covers "
                             f"{dts[0].isoformat()} .. {dts[-1].isoformat()}")
        i0, i1 = int(keep[0]), int(keep[-1]) + 1        # half-open
        if keep.size != (i1 - i0):
            raise SystemExit("ERROR: selected frames are not contiguous in the source file")

        n = i1 - i0
        if args.check_anchor:
            a = _parse_day(args.check_anchor)
            hit = np.nonzero(dts[i0:i1] <= a)[0]
            if hit.size == 0:
                raise SystemExit(f"ERROR: anchor {args.check_anchor} precedes the slice")
            idx = int(hit[-1])                           # frame at/just before the anchor
            if idx < args.check_steps:
                need = args.check_steps - idx
                raise SystemExit(f"ERROR: anchor {args.check_anchor} sits at slice index {idx}, but a "
                                 f"{args.check_steps}-step backward rollout needs {args.check_steps} earlier "
                                 f"frames — extend --start by ~{need * 6} h")
            print(f"anchor check OK: {args.check_anchor} -> slice index {idx} "
                  f"(>= R={args.check_steps}; window fits)")

        d = src[args.dataset]
        print(f"slicing {args.input}[{i0}:{i1}] -> {args.output}  "
              f"({n} frames, {dts[i0].isoformat()} .. {dts[i1-1].isoformat()})")

        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        tmp = args.output + ".tmp"                       # atomic: never leave a partial slice
        with h5.File(tmp, "w") as dst:
            out = dst.create_dataset(args.dataset, (n,) + d.shape[1:], dtype=d.dtype)
            for k in range(n):                           # frame-at-a-time: bounded memory
                out[k] = d[i0 + k]
                if (k + 1) % 50 == 0 or k == n - 1:
                    print(f"  {k+1}/{n} frames", flush=True)

            dst.create_dataset("timestamp", data=ts[i0:i1])
            for key in ("lat", "lon"):
                dst.create_dataset(key, data=src[key][...])
            ch = src["channel"]
            dst.create_dataset("channel", data=ch[...], dtype=ch.dtype)
            if "valid_data" in src:
                dst.create_dataset("valid_data", data=src["valid_data"][i0:i1])

            for key in ("timestamp", "channel", "lat", "lon"):
                dst[key].make_scale(key)
            for dim, key, label in ((0, "timestamp", "Timestamp in seconds in UTC time zone"),
                                    (1, "channel", "Channel name"),
                                    (2, "lat", "Latitude in degrees"),
                                    (3, "lon", "Longitude in degrees")):
                out.dims[dim].label = label
                out.dims[dim].attach_scale(dst[key])
        os.replace(tmp, args.output)

    print("done. verifying:")
    return inspect(args.output, args.dataset)


if __name__ == "__main__":
    sys.exit(main())
