# SPDX-License-Identifier: Apache-2.0
#
# the HDF5 schema for raw backward-rollout FIELD dumps, shared between the
# GPU producer (backward_rollout.py --save_fields_anchors) and the CPU consumer
# (make_backward_video.py). Keeping writer + loader in ONE torch-free module means the two
# sides cannot drift apart, and the round-trip is unit-testable with numpy + h5py alone.
#
# File layout (backward_fields.h5):
#   attrs:  config, run_num, checkpoint_path, step_hours (negative, e.g. -6), direction
#   truth:  (A, L, C, H, W) float32 physical units; L = num_steps+1 leads, lead 0 = the present
#   pred:   (A, M, L, C, H, W) float32 physical units; pred[:, m, 0] == truth[:, 0] for every m
#           (the IC, by construction -- the rollout starts FROM truth, so frame 0 is duplicated
#           for a uniform (lead -> frame) contract on both panels)
#
#   M = ENSEMBLE MEMBERS retained (1 for a deterministic run, for --fields_member <N>, or for
#   --fields_member mean; E for --fields_member all). Truth carries NO member axis -- it is the
#   same reanalysis for every member, so duplicating it M times would be pure waste.
#
#   SCHEMA NOTE -- pred gained the member axis in Aug 2026 (the 1000-member case-study runs).
#   Dumps written before that have a 5-D pred with no member axis; load_channel_trajectory
#   reads BOTH layouts, so old files stay usable and make_backward_video.py needs no change.
#   Anything reading f["pred"] directly must branch on ndim.
#   channel:            (C,) channel names actually dumped (any subset of the model channels)
#   lead_hours_into_past: (L,) [0, 6, 12, ...] -- positive hours INTO THE PAST
#
#   CONVENTION WARNING -- this key is NOT the same array as the identically-named key in
#   backward_rollout.json. Here it has length L = num_steps+1 and STARTS AT 0, because frame 0
#   of the dump is the initial condition (pred == truth). The metrics JSON omits lead 0 (there
#   is nothing to score there), so its array has length num_steps and starts at 6. Indexing a
#   JSON metric array with an HDF5 frame index is therefore off by one. Nothing in the repo
#   joins the two today -- the dump feeds videos, the JSON feeds plots -- but a figure that
#   overlays a metric on a frame must align on the VALUE, not the position.
#   lat, lon:           (H,), (W,) degrees
#   timestamps:         (A, L) ISO date strings of the VALID time of each frame ("" if unknown)
#   anchor_index:       (A,) dataset index of each anchor (the present time rolled back from)
#
# float32, not float16: physical msl (~1.01e5 Pa) overflows fp16's 65504 max. Datasets are
# created up-front and written slab-by-slab, so the producer never holds a rollout in RAM.
#
# Why not makani's RolloutBuffer machinery (utils/inference/rollout_buffer.py): those buffers
# average over initial conditions (running means for scores/spectra) and live on torch; this
# dump keeps INDIVIDUAL trajectories (pred AND truth per anchor) and must be readable by a
# torch-free CPU renderer -- a deliberate divergence, not an oversight.

import numpy as np

FIELDS_FILENAME_DEFAULT = "backward_fields.h5"


class FieldDumpWriter:
    """Incremental writer for the backward-rollout field dump. h5py is imported lazily so
    importing this module never costs anything for runs that don't dump fields."""

    def __init__(self, path, channels, n_anchors, n_leads, lat, lon, step_hours, attrs=None, n_members=1):
        import h5py

        if int(n_members) < 1:
            raise ValueError(f"n_members must be >= 1, got {n_members}")
        self.path = path
        self.n_members = int(n_members)
        self._f = h5py.File(path, "w")
        C, H, W = len(channels), len(lat), len(lon)
        # truth has NO member axis: it is the same reanalysis for every member.
        self._truth = self._f.create_dataset("truth", (n_anchors, n_leads, C, H, W), dtype=np.float32, chunks=(1, 1, 1, H, W))
        # chunk = one (anchor, member, lead, channel) plane: matches both the write pattern (one
        # lead, all members, all dumped channels) and the read pattern (one channel + member,
        # all leads). Keeping a single plane per chunk is what lets tomorrow's analysis pull one
        # channel out of a 22-channel dump without paying for the other 21.
        self._pred = self._f.create_dataset(
            "pred", (n_anchors, self.n_members, n_leads, C, H, W), dtype=np.float32, chunks=(1, 1, 1, 1, H, W)
        )
        self._f.create_dataset("member_index", data=np.arange(self.n_members, dtype=np.int64))
        str_dt = "S64"
        self._f.create_dataset("channel", data=np.array([c.encode() for c in channels], dtype=str_dt))
        self._f.create_dataset("lead_hours_into_past", data=np.arange(n_leads, dtype=np.int64) * int(abs(step_hours)))
        self._f.create_dataset("lat", data=np.asarray(lat, dtype=np.float64))
        self._f.create_dataset("lon", data=np.asarray(lon, dtype=np.float64))
        self._ts = self._f.create_dataset("timestamps", (n_anchors, n_leads), dtype=str_dt)
        self._anchor_index = self._f.create_dataset("anchor_index", (n_anchors,), dtype=np.int64)
        self._f.attrs["step_hours"] = -int(abs(step_hours))
        self._f.attrs["direction"] = "backward"
        for k, v in (attrs or {}).items():
            self._f.attrs[k] = v

    def set_anchor(self, anchor_slot, dataset_index):
        self._anchor_index[anchor_slot] = int(dataset_index)

    def write_frame(self, anchor_slot, lead_slot, pred_chw, truth_chw, timestamp=""):
        """truth_chw: (C, H, W) in PHYSICAL units, dumped-channel order.
        pred_chw: (C, H, W) for a single-member dump, or (M, C, H, W) to write every retained
        member of this lead at once. A (C, H, W) array is accepted only when n_members == 1 --
        silently broadcasting one member across M slots would fabricate an ensemble."""
        p = np.asarray(pred_chw, dtype=np.float32)
        if p.ndim == 3:
            p = p[None]
        if p.ndim != 4 or p.shape[0] != self.n_members:
            raise ValueError(
                f"write_frame: pred must be (C,H,W) or ({self.n_members},C,H,W); got shape {np.shape(pred_chw)}"
            )
        self._pred[anchor_slot, :, lead_slot] = p
        self._truth[anchor_slot, lead_slot] = np.asarray(truth_chw, dtype=np.float32)
        self._ts[anchor_slot, lead_slot] = str(timestamp)[:64].encode()

    def close(self):
        self._f.close()


def _open_channel(f, path, channel, anchor):
    """Shared validation for the two loaders: resolve the channel index and check the anchor."""
    channels = [c.decode() if isinstance(c, bytes) else str(c) for c in f["channel"][...]]
    if channel not in channels:
        raise KeyError(f"channel '{channel}' not in this dump; available: {channels}")
    n_anchors = f["pred"].shape[0]
    if not (0 <= anchor < n_anchors):
        raise IndexError(f"anchor {anchor} out of range: dump holds {n_anchors} anchor(s)")
    # pred is (A,M,L,C,H,W) since Aug 2026; older dumps are (A,L,C,H,W) with no member axis.
    n_members = f["pred"].shape[1] if f["pred"].ndim == 6 else 1
    return channels, channels.index(channel), n_members


def _common_fields(f, channels, anchor, n_members):
    return {
        "lead_hours_into_past": f["lead_hours_into_past"][...],
        "lat": f["lat"][...],
        "lon": f["lon"][...],
        "timestamps": [t.decode() if isinstance(t, bytes) else str(t) for t in f["timestamps"][anchor]],
        "channels": channels,
        "n_members": n_members,
        "attrs": dict(f.attrs),
    }


def load_channel_trajectory(path, channel, anchor=0, member=0):
    """Load ONE channel's rollout for ONE member -> dict with pred/truth (L,H,W), leads,
    lat/lon, timestamps, n_members, and the file attrs. Reads only the requested slabs.
    Handles both the member-axis layout and pre-Aug-2026 dumps without one."""
    import h5py

    with h5py.File(path, "r") as f:
        channels, ci, n_members = _open_channel(f, path, channel, anchor)
        if not (0 <= member < n_members):
            raise IndexError(f"member {member} out of range: dump holds {n_members} member(s)")
        pred = f["pred"][anchor, member, :, ci] if f["pred"].ndim == 6 else f["pred"][anchor, :, ci]
        out = {"pred": pred, "truth": f["truth"][anchor, :, ci]}
        out.update(_common_fields(f, channels, anchor, n_members))
    return out


def load_channel_members(path, channel, anchor=0):
    """Load ONE channel's rollout for EVERY member -> dict with pred (M,L,H,W) and truth (L,H,W),
    plus the same metadata as load_channel_trajectory. This is the entry point for ensemble
    analysis (spread, EOFs, quantiles); pooling several dumps means calling it per file and
    concatenating on axis 0, which is valid because each run's members are drawn from an
    independent noise seed. Reads only this channel's slabs -- a 22-channel dump costs the
    same here as a 1-channel one."""
    import h5py

    with h5py.File(path, "r") as f:
        channels, ci, n_members = _open_channel(f, path, channel, anchor)
        pred = f["pred"][anchor, :, :, ci] if f["pred"].ndim == 6 else f["pred"][anchor, :, ci][None]
        out = {"pred": pred, "truth": f["truth"][anchor, :, ci]}
        out.update(_common_fields(f, channels, anchor, n_members))
    return out
