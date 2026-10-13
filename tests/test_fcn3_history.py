# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FCN3 with ``n_history >= 2``: the dynamic-aux history offsets must come from the frame-0 list.

The loop in ``AtmoSphericNeuralOperatorNet._precompute_channel_groups`` extended the list it was
iterating, so from the second history step on it re-offset entries it had already offset: 39 aux
channels instead of 30 for ``n_history=2``, with indices past the input width, and an IndexError
at the first forward. Inert for ``n_history <= 1`` (every config here), which is why it was never hit.
"""

import unittest

import torch

from makani.models.networks.fourcastnet3 import AtmoSphericNeuralOperatorNet
from makani.utils.features import get_auxiliary_channels

from .testutils import disable_tf32, set_seed

LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
SURFACE = ["u10m", "v10m", "u100m", "v100m", "t2m", "msl", "tcwv"]
CHANNELS = SURFACE + [f"{v}{p}" for v in "uvztq" for p in LEVELS]
# the auxiliary channels our config appends after the state: zenith, 8 noise, orography, 2-channel land-sea mask
AUX = get_auxiliary_channels(add_zenith=True, n_noise_chan=8, add_orography=True, add_landmask=True, landmask_preprocessing="floor")
STATIC_AUX = {"xoro", "xlsml", "xlsms"}  # replicated per frame: everything else (zenith, noise)
N_DYN_AUX = len([a for a in AUX if a not in STATIC_AUX])
IMG_SHAPE = (36, 72)


class TestHistoryFrames(unittest.TestCase):
    def setUp(self):
        disable_tf32()
        set_seed(333)

    def _model(self, n_history):
        return AtmoSphericNeuralOperatorNet(
            inp_shape=IMG_SHAPE,
            out_shape=IMG_SHAPE,
            channel_names=CHANNELS,
            aux_channel_names=AUX,
            n_history=n_history,
            embed_dim=8,
            aux_embed_dim=4,
            num_layers=1,
            scale_factor=2,
            big_skip=False,
            clamp_water=False,
        ).eval()

    def test_two_history_frames_build_and_run(self):
        model = self._model(n_history=2)
        n = len(CHANNELS)
        frame = n + N_DYN_AUX
        n_static = len(AUX) - N_DYN_AUX
        with self.subTest(desc="one aux index per (frame, dynamic aux) plus the static aux"):
            self.assertEqual(model.n_aux_chans, 3 * N_DYN_AUX + n_static)
            self.assertEqual(model.aux_channels.numel(), model.n_aux_chans)
        # the full model input: (n_history + 1) frames of [state, dynamic aux], then the static aux channels
        x = torch.randn(2, model.n_in_chans + model.n_aux_chans, *IMG_SHAPE)
        with self.subTest(desc="every aux index lies inside the input"):
            self.assertLess(int(model.aux_channels.max()), x.shape[1])
            self.assertEqual(x.shape[1], 3 * frame + n_static)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.shape), (2, n, *IMG_SHAPE))

    def test_single_history_frame_unchanged(self):
        """n_history <= 1 never hit the defect; its channel bookkeeping must be unchanged."""
        for n_history in (0, 1):
            model = self._model(n_history=n_history)
            n_static = len(AUX) - N_DYN_AUX
            self.assertEqual(model.n_aux_chans, (n_history + 1) * N_DYN_AUX + n_static)
            self.assertEqual(model.aux_channels.numel(), model.n_aux_chans)


if __name__ == "__main__":
    unittest.main()
