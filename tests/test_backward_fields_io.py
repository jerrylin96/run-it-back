# SPDX-License-Identifier: Apache-2.0
#
# round-trip tests for the backward-rollout field-dump schema
# (analysis/backward_fields_io.py). The producer (backward_rollout.py, on GPU) and the
# consumers (make_backward_video.py and the ensemble analysis, on CPU) share this one
# module precisely so they cannot drift -- these tests are what make that guarantee real.
#
# The member axis on `pred` was added Aug 2026 for the 1000-member case-study runs, so the
# coverage here deliberately includes BOTH layouts: dumps written before that change have a
# 5-D pred and must stay readable, since re-running a rollout to recover an old figure costs
# GPU-hours. torch-free by construction (numpy + h5py only), like the module under test.

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis"))

from backward_fields_io import (  # noqa: E402
    FieldDumpWriter,
    load_channel_members,
    load_channel_trajectory,
)

CHANNELS = ["t2m", "z500", "u250"]
H, W = 6, 8
LAT = np.linspace(90.0, -90.0, H)
LON = np.linspace(0.0, 360.0, W, endpoint=False)


def _write(path, n_members, n_anchors=1, n_leads=4):
    """Write a dump whose every value encodes its own index, so a mis-transposed read
    cannot pass: value = anchor*1e6 + member*1e4 + lead*1e2 + channel (+ a spatial ramp)."""
    ramp = np.arange(H * W, dtype=np.float32).reshape(H, W) * 1e-3
    w = FieldDumpWriter(path, CHANNELS, n_anchors, n_leads, LAT, LON, step_hours=6, n_members=n_members)
    for a in range(n_anchors):
        w.set_anchor(a, 100 + a)
        for lead in range(n_leads):
            truth = np.stack([np.full((H, W), -(a * 1e6 + lead * 1e2 + c), dtype=np.float32) + ramp for c in range(len(CHANNELS))])
            pred = np.stack(
                [
                    np.stack([np.full((H, W), a * 1e6 + m * 1e4 + lead * 1e2 + c, dtype=np.float32) + ramp for c in range(len(CHANNELS))])
                    for m in range(n_members)
                ]
            )
            w.write_frame(a, lead, pred if n_members > 1 else pred[0], truth, timestamp=f"2021-06-{lead + 1:02d}")
    w.close()
    return ramp


def test_roundtrip_single_member(tmp_path):
    p = str(tmp_path / "d.h5")
    ramp = _write(p, n_members=1)
    got = load_channel_trajectory(p, "z500")
    assert got["n_members"] == 1
    ci = CHANNELS.index("z500")
    for lead in range(4):
        np.testing.assert_allclose(got["pred"][lead], lead * 1e2 + ci + ramp)
        np.testing.assert_allclose(got["truth"][lead], -(lead * 1e2 + ci) + ramp)
    np.testing.assert_array_equal(got["lead_hours_into_past"], [0, 6, 12, 18])
    assert got["attrs"]["direction"] == "backward"
    assert got["attrs"]["step_hours"] == -6


def test_roundtrip_all_members(tmp_path):
    """The case-study path: every member retained, each recoverable and distinct."""
    p = str(tmp_path / "d.h5")
    ramp = _write(p, n_members=5)
    got = load_channel_members(p, "u250")
    ci = CHANNELS.index("u250")
    assert got["n_members"] == 5
    assert got["pred"].shape == (5, 4, H, W)
    assert got["truth"].shape == (4, H, W)
    for m in range(5):
        for lead in range(4):
            np.testing.assert_allclose(got["pred"][m, lead], m * 1e4 + lead * 1e2 + ci + ramp)
    # per-member single-trajectory reads must agree with the bulk read
    for m in (0, 2, 4):
        one = load_channel_trajectory(p, "u250", member=m)
        np.testing.assert_array_equal(one["pred"], got["pred"][m])
        np.testing.assert_array_equal(one["truth"], got["truth"])


def test_truth_has_no_member_axis(tmp_path):
    """Truth is the same reanalysis for every member; duplicating it M times would be waste.
    Locks the asymmetry between the pred and truth datasets."""
    import h5py

    p = str(tmp_path / "d.h5")
    _write(p, n_members=5)
    with h5py.File(p, "r") as f:
        assert f["pred"].shape == (1, 5, 4, len(CHANNELS), H, W)
        assert f["truth"].shape == (1, 4, len(CHANNELS), H, W)
        np.testing.assert_array_equal(f["member_index"][...], np.arange(5))


def test_multi_anchor_and_member_axes_are_not_transposed(tmp_path):
    """A (anchor, member) mix-up survives any test with one of them equal to 1."""
    p = str(tmp_path / "d.h5")
    ramp = _write(p, n_members=3, n_anchors=2)
    ci = CHANNELS.index("t2m")
    for a in range(2):
        got = load_channel_members(p, "t2m", anchor=a)
        for m in range(3):
            np.testing.assert_allclose(got["pred"][m, 1], a * 1e6 + m * 1e4 + 1e2 + ci + ramp)


def test_legacy_dump_without_member_axis_still_reads(tmp_path):
    """Pre-Aug-2026 dumps have a 5-D pred. Re-running a rollout to recover an old figure
    costs GPU-hours, so the loader must keep reading them."""
    import h5py

    p = str(tmp_path / "legacy.h5")
    with h5py.File(p, "w") as f:
        f.create_dataset("pred", data=np.arange(1 * 4 * 3 * H * W, dtype=np.float32).reshape(1, 4, 3, H, W))
        f.create_dataset("truth", data=np.zeros((1, 4, 3, H, W), dtype=np.float32))
        f.create_dataset("channel", data=np.array([c.encode() for c in CHANNELS], dtype="S64"))
        f.create_dataset("lead_hours_into_past", data=np.arange(4) * 6)
        f.create_dataset("lat", data=LAT)
        f.create_dataset("lon", data=LON)
        f.create_dataset("timestamps", data=np.array([[b""] * 4], dtype="S64"))
        f.attrs["direction"] = "backward"

    got = load_channel_trajectory(p, "z500")
    assert got["n_members"] == 1
    assert got["pred"].shape == (4, H, W)
    # load_channel_members promotes the legacy layout to a length-1 member axis
    bulk = load_channel_members(p, "z500")
    assert bulk["pred"].shape == (1, 4, H, W)
    np.testing.assert_array_equal(bulk["pred"][0], got["pred"])
    with pytest.raises(IndexError, match="member 1 out of range"):
        load_channel_trajectory(p, "z500", member=1)


@pytest.mark.parametrize(
    "bad_pred, match",
    [
        (np.zeros((3, H, W), np.float32), "pred must be"),      # single member into an M=4 dump
        (np.zeros((2, 3, H, W), np.float32), "pred must be"),   # wrong member count
        (np.zeros((H, W), np.float32), "pred must be"),         # missing the channel axis
    ],
)
def test_write_frame_rejects_wrong_member_shape(tmp_path, bad_pred, match):
    """Silently broadcasting one member across M slots would FABRICATE an ensemble -- every
    member identical, spread exactly zero, and no error anywhere. Must fail loud."""
    p = str(tmp_path / "d.h5")
    w = FieldDumpWriter(p, CHANNELS, 1, 4, LAT, LON, step_hours=6, n_members=4)
    with pytest.raises(ValueError, match=match):
        w.write_frame(0, 0, bad_pred, np.zeros((3, H, W), np.float32))
    w.close()


def test_bad_inputs_fail_loud(tmp_path):
    p = str(tmp_path / "d.h5")
    _write(p, n_members=2)
    with pytest.raises(KeyError, match="not in this dump"):
        load_channel_trajectory(p, "q500")
    with pytest.raises(IndexError, match="anchor 1 out of range"):
        load_channel_trajectory(p, "t2m", anchor=1)
    with pytest.raises(IndexError, match="member 2 out of range"):
        load_channel_trajectory(p, "t2m", member=2)
    with pytest.raises(ValueError, match="n_members must be >= 1"):
        FieldDumpWriter(str(tmp_path / "x.h5"), CHANNELS, 1, 4, LAT, LON, step_hours=6, n_members=0)
