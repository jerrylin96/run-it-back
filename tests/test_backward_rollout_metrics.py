# SPDX-License-Identifier: Apache-2.0
#
# The area-weighted correlation helpers of the rollout harnesses are unit-free, so rescaling a
# channel must not move them. An absolute 1e-12 guard under the square root of the denominator
# broke this for specific humidity (kg/kg): q's anomaly-variance product is ~1e-12 at 500 hPa
# and far smaller aloft, so the guard shrank q500's ACC to ~0.7x and read q200-q50 as 0.000.
# This test fails on that code and passes once the guard is gone (a zero-anomaly field now
# yields NaN, which is the correct value for an undefined correlation).

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis"))

import backward_rollout as br  # noqa: E402
import verify_direction as vd  # noqa: E402

H, W = 19, 36


@pytest.mark.parametrize("corr", [br._wacc, vd._wcorr], ids=["acc", "corr"])
def test_correlations_do_not_depend_on_units(corr):
    g = torch.Generator().manual_seed(0)
    p = torch.randn(1, 3, H, W, generator=g)
    t = 0.8 * p + 0.6 * torch.randn(1, 3, H, W, generator=g)
    wt, wsum = vd._make_weight(np.linspace(-90, 90, H), W, "cpu")
    scale = torch.tensor([1.0, 1e-6, 1e3]).view(1, 3, 1, 1)
    assert torch.allclose(corr(p * scale, t * scale, wt, wsum), corr(p, t, wt, wsum), rtol=1e-5)
    # and it is still a correlation: a perfect match is 1 at every scale
    assert torch.allclose(corr(p * scale, p * scale, wt, wsum), torch.ones(3), rtol=1e-5)
