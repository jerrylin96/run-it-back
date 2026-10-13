# SPDX-License-Identifier: Apache-2.0
#
# render a VIDEO of the backward rollout for any dumped variable.
#
# Consumes the raw-field HDF5 written by backward_rollout.py --save_fields_anchors
# (schema: analysis/backward_fields_io.py) and renders one channel's trajectory as a
# three-panel animation -- ERA5 truth | backward prediction | (pred - truth) error --
# stepping 6 h further into the past per frame. The truth panel makes blur visually
# obvious (prediction goes smooth while truth stays sharp); the error panel localizes it.
#
# CPU-only (numpy + h5py + matplotlib; no makani/torch import), like plot_backward_rollout.py.
# Writer: ffmpeg -> .mp4 when available, else matplotlib's PillowWriter -> .gif (always
# available; Pillow is a matplotlib dependency), so it works in makani.sif either way.
#
# Usage (inside the container, e.g. via crun or make_backward_video.sbatch):
#   python analysis/make_backward_video.py \
#       --fields_file /work/.../runs/<cfg>/<run>/scores/backward_fields.h5 --channel z500
#   # every dumped channel:      --channel all
#   # a different dumped anchor: --anchor 1
#   # playback into the past (default) or reversed (past -> present): --reverse

import os
import sys
import shutil
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backward_fields_io import load_channel_trajectory  # noqa: E402


def _resolve_format(fmt):
    """ffmpeg -> .mp4 if available (or forced), else Pillow -> .gif. Fails loud up-front on
    --format=mp4 without ffmpeg, before any rendering work."""
    have_ffmpeg = shutil.which("ffmpeg") is not None
    if fmt == "mp4" and not have_ffmpeg:
        raise RuntimeError("--format=mp4 requested but no ffmpeg on PATH; use --format=gif (or auto)")
    return ".mp4" if (fmt in ("auto", "mp4") and have_ffmpeg) else ".gif"


def _make_writer(ext, fps, animation):
    """Fresh writer per video: matplotlib writers are single-use once .save has run. The pad
    filter rounds both frame dimensions UP to even -- libx264 + yuv420p rejects odd sizes, and
    figsize*dpi lands on odd pixel counts for many (--dpi, figsize) combinations."""
    if ext == ".mp4":
        return animation.FFMpegWriter(
            fps=fps, codec="h264",
            extra_args=["-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"],
        )
    return animation.PillowWriter(fps=fps)


def render_channel(traj, channel, out_path, writer, dpi, reverse, cmap, plt, animation):
    pred, truth = traj["pred"], traj["truth"]            # (L, H, W) physical units
    leads = traj["lead_hours_into_past"]
    lat, lon = traj["lat"], traj["lon"]
    ts = traj["timestamps"]
    # normalize to north-first rows so imshow(origin='upper') is upright regardless of how the
    # dataset stored latitude (an ascending lat axis would otherwise render every map mirrored)
    if len(lat) > 1 and lat[0] < lat[-1]:
        lat = lat[::-1]
        pred, truth = pred[:, ::-1, :], truth[:, ::-1, :]
    err = pred - truth

    # one fixed, robust color scale per panel across ALL frames -- a per-frame scale would
    # hide the blur (matplotlib would silently re-stretch the flattening prediction).
    vmin, vmax = np.nanpercentile(truth, [1.0, 99.0])
    if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmin == vmax:
        vmin, vmax = np.nanmin(truth), np.nanmax(truth) + 1e-6
    elim = float(np.nanpercentile(np.abs(err), 99.0))
    if not np.isfinite(elim) or elim == 0.0:
        elim = 1e-6

    order = np.arange(len(leads))[::-1] if reverse else np.arange(len(leads))
    extent = [float(lon[0]), float(lon[-1]), float(min(lat[0], lat[-1])), float(max(lat[0], lat[-1]))]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.0))
    panels = [
        (axes[0], truth, "ERA5 truth", cmap, vmin, vmax),
        (axes[1], pred, "backward prediction", cmap, vmin, vmax),
        (axes[2], err, "prediction - truth", "RdBu_r", -elim, elim),
    ]
    ims, movies = [], [truth, pred, err]
    for ax, data, title, cm, lo, hi in panels:
        im = ax.imshow(data[order[0]], origin="upper", extent=extent, aspect="auto", cmap=cm, vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
        ims.append(im)
    sup = fig.suptitle("", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    def update(fi):
        k = int(order[fi])
        for im, data in zip(ims, movies):
            im.set_data(data[k])
        date = f"   valid {ts[k]}" if ts[k] else ""
        sup.set_text(f"{channel}  backward rollout   lead −{int(leads[k])} h{date}")
        return ims

    ani = animation.FuncAnimation(fig, update, frames=len(order), blit=False)
    ani.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fields_file", required=True,
                   help="backward_fields.h5 written by backward_rollout.py --save_fields_anchors.")
    p.add_argument("--channel", required=True,
                   help="Channel to render (any channel present in the dump), or 'all' for every dumped channel.")
    p.add_argument("--anchor", default=0, type=int, help="Which dumped anchor (rollout) to render.")
    p.add_argument("--member", default=0, type=int,
                   help="Which ensemble member to render, for dumps written with --fields_member all. "
                        "Default 0. Single-member dumps (mean, one member, or deterministic) have only "
                        "member 0 — check the file's pred_source attr for what that member actually is.")
    p.add_argument("--out_dir", default=None, help="Output directory (default: alongside the fields file).")
    p.add_argument("--format", default="auto", choices=["auto", "mp4", "gif"],
                   help="auto = mp4 when ffmpeg is on PATH, else gif.")
    p.add_argument("--fps", default=4, type=int, help="Frames per second (one frame = one 6 h step).")
    p.add_argument("--dpi", default=110, type=int)
    p.add_argument("--cmap", default="viridis", help="Colormap for the truth/prediction panels.")
    p.add_argument("--reverse", action="store_true",
                   help="Play past -> present (chronological) instead of the rollout order (present -> past).")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.fields_file))
    os.makedirs(out_dir, exist_ok=True)
    ext = _resolve_format(args.format)

    # resolve the channel list up-front so 'all' and typos both fail loud before rendering
    import h5py
    with h5py.File(args.fields_file, "r") as f:
        available = [c.decode() if isinstance(c, bytes) else str(c) for c in f["channel"][...]]
    if args.channel != "all" and args.channel not in available:
        raise SystemExit(f"channel '{args.channel}' not in this dump; available: {available}")
    channels = available if args.channel == "all" else [args.channel]

    written = []
    for ch in channels:
        traj = load_channel_trajectory(args.fields_file, ch, anchor=args.anchor, member=args.member)
        writer = _make_writer(ext, args.fps, animation)
        # member goes in the filename only for multi-member dumps, so existing single-member
        # output paths (and anything that globs them) are unchanged.
        mtag = f"_m{args.member}" if traj["n_members"] > 1 else ""
        out_path = os.path.join(out_dir, f"backward_rollout_{ch}_anchor{args.anchor}{mtag}{ext}")
        render_channel(traj, ch, out_path, writer, args.dpi, args.reverse, args.cmap, plt, animation)
        written.append(out_path)
        print(f"wrote {out_path}")

    if not written:
        raise RuntimeError("nothing rendered")


if __name__ == "__main__":
    main()
