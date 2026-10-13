#
# Locks the two metric corrections found by the Jul/Aug 2026 adversarial review round
# (reviewers 3 and 8, corroborated by 1). Both changed numbers we intended to publish:
#
#   1. GeometricRankHistogram sent every TIE to the top bin (searchsorted side="right"),
#      inflating outer_mass and manufacturing apparent under-dispersion. This is not an
#      exotic edge case here: clamp_water pins near-zero specific humidity to exactly 0,
#      so the water channels are tie-dense.
#   2. GeometricSSR forms its ratio per-sample; averaging those ratios over anchors is
#      unstable wherever skill ~ spread/E, because the eps clamp then dominates and a few
#      anchors produce outliers that bias the mean. forward(return_components=True) hands
#      back the area-weighted components so callers can pool instead.
#
# Both fixes are no-ops on the paths that were already correct, and that is what the
# no-tie / single-sample cases below assert.

import numpy as np
import pytest
import torch

from makani.utils.metrics.functions import GeometricRankHistogram, GeometricSSR

_GRID = dict(grid_type="equiangular", img_shape=(8, 16), crop_shape=(8, 16), crop_offset=(0, 0))


def _rankhist(members, obs, channel_reduction="none", batch_reduction="sum"):
    """(E,) members and a scalar obs, broadcast over a small grid -> the rank histogram."""
    E = len(members)
    H, W = _GRID["img_shape"]
    fc = torch.zeros(1, E, 1, H, W, dtype=torch.float64)
    for e, v in enumerate(members):
        fc[0, e] = float(v)
    ob = torch.full((1, 1, H, W), float(obs), dtype=torch.float64)
    metric = GeometricRankHistogram(channel_reduction=channel_reduction, batch_reduction=batch_reduction, **_GRID)
    return metric(fc, ob).squeeze().double().numpy()


class TestRankHistogramTies:
    def test_no_ties_is_one_hot(self):
        """Without ties the fix must be bit-identical to the old one_hot behavior."""
        for obs, expect_bin in [(0.5, 0), (2.5, 2), (9.0, 4)]:
            h = _rankhist([1.0, 2.0, 3.0, 4.0], obs)
            h = h / h.sum()
            assert h[expect_bin] == pytest.approx(1.0), f"obs={obs} -> {h}"

    def test_tie_with_one_member_splits_two_bins(self):
        h = _rankhist([1.0, 2.0, 3.0, 4.0], 3.0)
        h = h / h.sum()
        np.testing.assert_allclose(h, [0.0, 0.0, 0.5, 0.5, 0.0], atol=1e-9)

    def test_all_members_tied_is_flat_not_top_bin(self):
        """The clamp_water case. Old code put 100% in bin E and faked under-dispersion."""
        h = _rankhist([0.0, 0.0, 0.0, 0.0], 0.0)
        h = h / h.sum()
        np.testing.assert_allclose(h, np.full(5, 0.2), atol=1e-9)
        assert h[-1] < 0.5, "an observation tying every member carries no rank information"

    def test_mass_is_conserved(self):
        for members, obs in [([1, 2, 3, 4], 0.5), ([1, 2, 3, 4], 3.0), ([0, 0, 2, 3], 0.0), ([0, 0, 0, 0], 0.0)]:
            h = _rankhist([float(m) for m in members], float(obs))
            assert h.sum() > 0
            assert np.all(h >= 0.0)


class TestSSRPooling:
    def _fc_ob(self, seed, spread_scale, bias):
        g = torch.Generator().manual_seed(seed)
        H, W = _GRID["img_shape"]
        E = 4
        fc = spread_scale * torch.randn(1, E, 2, H, W, generator=g, dtype=torch.float64)
        ob = bias + torch.zeros(1, 2, H, W, dtype=torch.float64)
        return fc, ob

    def test_components_reproduce_the_ratio_on_one_sample(self):
        """Pooling over a single sample must equal that sample's own SSR."""
        metric = GeometricSSR(channel_reduction="none", batch_reduction="none", **_GRID)
        fc, ob = self._fc_ob(0, 1.0, 0.3)
        direct = metric(fc, ob).double().numpy()
        spread, skill = metric(fc, ob, return_components=True)
        E = fc.shape[1]
        pooled = torch.sqrt(spread / torch.clamp(skill - spread / float(E), min=metric.eps))
        np.testing.assert_allclose(pooled.double().numpy(), direct, rtol=1e-10)

    def test_pooled_differs_from_mean_of_ratios_under_an_outlier(self):
        """A WELL-CALIBRATED anchor drives the per-sample denominator into the eps clamp, and the
        mean of ratios then explodes while the pooled estimate stays finite and sane.

        The mechanism is worth stating precisely, because it is easy to construct backwards: the
        blow-up needs spread to stay O(1) while skill -> spread/E, which is what a perfectly
        centered ensemble does (the ensemble-mean error of E iid members is exactly spread/E in
        expectation). Shrinking BOTH spread and skill instead -- e.g. a degenerate ensemble sitting
        on the observation -- gives sqrt(tiny/eps), which the clamp pushes toward ZERO. So the
        pathology afflicts the anchors the model predicts BEST, which is precisely why averaging
        per-anchor ratios biases a skillful model's reported SSR.

        Samples 1-2 are biased (obs offset from the ensemble center) and behave normally; sample 3
        is unbiased and triggers the clamp.
        """
        metric = GeometricSSR(channel_reduction="none", batch_reduction="none", **_GRID)
        samples = [self._fc_ob(1, 1.0, 0.5), self._fc_ob(2, 1.0, 0.5), self._fc_ob(3, 1.0, 0.0)]

        ratios, num, den = [], 0.0, 0.0
        E = 4
        for fc, ob in samples:
            ratios.append(metric(fc, ob).double().numpy())
            sp, sk = metric(fc, ob, return_components=True)
            num = num + sp.double().numpy()
            den = den + sk.double().numpy()
        mean_of_ratios = np.mean(np.stack(ratios), axis=0)
        pooled = np.sqrt(num / np.maximum(den - num / float(E), metric.eps))

        assert np.all(np.isfinite(pooled))
        assert np.any(mean_of_ratios > 5.0 * pooled), (
            f"expected the eps-clamped outlier to inflate the mean of ratios: "
            f"mean_of_ratios={mean_of_ratios}, pooled={pooled}"
        )

    def test_components_keep_the_batch_dim(self):
        """The components are returned BEFORE the metric's batch reduction, so they are (B, C)
        while the reduced ratio is (C,). backward_rollout.py accumulates into per-channel arrays
        and must sum over the batch dim; this locks that contract, which an earlier version of
        this file missed by only ever comparing components against components."""
        metric = GeometricSSR(channel_reduction="none", batch_reduction="mean", **_GRID)
        fc, ob = self._fc_ob(11, 1.0, 0.3)
        C = fc.shape[2]

        reduced = metric(fc, ob)
        spread, skill = metric(fc, ob, return_components=True)

        assert reduced.shape == (C,), f"reduced ratio should be (C,), got {tuple(reduced.shape)}"
        assert spread.shape == (fc.shape[0], C), f"components should be (B, C), got {tuple(spread.shape)}"
        assert skill.shape == (fc.shape[0], C)

        # the harness's accumulation pattern must land in a (C,) bucket without broadcasting
        bucket = np.zeros(C)
        bucket += spread.sum(dim=0).double().numpy()
        assert bucket.shape == (C,)

    def test_return_components_does_not_change_the_default_path(self):
        metric = GeometricSSR(channel_reduction="none", batch_reduction="none", **_GRID)
        fc, ob = self._fc_ob(7, 1.0, 0.2)
        a = metric(fc, ob).double().numpy()
        _ = metric(fc, ob, return_components=True)
        b = metric(fc, ob).double().numpy()
        np.testing.assert_array_equal(a, b)
