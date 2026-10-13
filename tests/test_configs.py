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

"""Consistency locks for config/rewind_fcn3.yaml (torch-free).

1. The backward FCN3 config must read the SAME 72-channel state with the SAME normalization
   map as upstream config/fourcastnet3.yaml (channel order included). YParams has no
   cross-file include, so the blocks are verbatim copies; this test is the recurring guard
   the copies otherwise lack.

2. The in-repo FCN3 net (AtmoSphericNeuralOperatorNet) takes a single ``embed_dim``
   (constructor default 8) and silently absorbs unknown config keys via **kwargs
   (model_registry.get_model passes the whole params dict). The upstream
   config/fourcastnet3.yaml keys atmo_embed_dim/surf_embed_dim/encoder_mlp/mlp_mode/
   pos_embed are STALE for this implementation — a config relying on them silently builds
   a tiny model. These tests pin: embed_dim is set explicitly (not 8), every architecture
   key rewind_fcn3.yaml sets is genuinely consumed by the constructor (AST-checked, so a
   future signature change fails loud here), and none of the known-stale keys sneak in.

Torch-free by design (ruamel/yaml + ast + json only) so it runs on any node, container or
not, alongside the other CPU tests.
"""

import ast
import json
import os
import unittest

try:
    from ruamel.yaml import YAML

    def _load_yaml(path):
        with open(path) as f:
            return YAML(typ="safe").load(f)

except ImportError:  # dev boxes without ruamel; pyyaml handles the same merge-key subset
    import yaml

    def _load_yaml(path):
        with open(path) as f:
            return yaml.safe_load(f)


_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cfg(name):
    return _load_yaml(os.path.join(_REPO, "config", name))


class TestRewindConfigConsistency(unittest.TestCase):
    """The backward FCN3 config must read the same state as upstream FCN3."""

    @classmethod
    def setUpClass(cls):
        cls.fcn3 = _cfg("rewind_fcn3.yaml")
        cls.upstream = _cfg("fourcastnet3.yaml")
        with open(os.path.join(_REPO, "data_process", "examples", "metadata_fcn3_72.json")) as f:
            cls.metadata = json.load(f)

    def test_channel_names_identical_to_upstream(self):
        """72-ch channel list (order included) identical to upstream fourcastnet3.yaml."""
        ours = self.fcn3["base_config"]["channel_names"]
        self.assertEqual(len(ours), 72)
        self.assertEqual(ours, self.upstream["base_config"]["channel_names"])

    def test_normalization_map_identical_to_upstream(self):
        """min/max water-channel map identical to upstream fourcastnet3.yaml."""
        ours = self.fcn3["base_config"]["normalization"]
        self.assertEqual(ours, self.upstream["base_config"]["normalization"])

    def test_channels_resolve_against_metadata(self):
        meta_ch = self.metadata["coords"]["channel"]
        for c in self.fcn3["base_config"]["channel_names"]:
            self.assertIn(c, meta_ch)

    def test_backward_targets_wiring(self):
        """Every run target: backward + multifiles + single-step validation."""
        for target in ("rewind_fcn3_det_smoke", "rewind_fcn3_ens_smoke", "rewind_fcn3_ens_10yr_backward", "rewind_fcn3_ens_casestudy"):
            t = self.fcn3[target]
            self.assertTrue(t["reverse_time"], target)
            self.assertTrue(t["multifiles"], target)
            self.assertEqual(t["valid_autoreg_steps"], 0, target)  # trained single-step

    def test_casestudy_target_is_the_trained_architecture(self):
        """The evaluation-only case-study target must load the trained checkpoint unchanged."""
        trained, case = self.fcn3["rewind_fcn3_ens_10yr_backward"], self.fcn3["rewind_fcn3_ens_casestudy"]
        for k in TestFCN3StaleKeyTrap.ARCH_KEYS + ("nettype", "channel_names", "normalization", "input_noise", "ensemble_size"):
            self.assertEqual(trained[k], case[k], k)


class TestFCN3StaleKeyTrap(unittest.TestCase):
    """Pin the embed_dim finding: stale upstream keys must not silently size the model."""

    STALE_KEYS = ("atmo_embed_dim", "surf_embed_dim", "encoder_mlp", "mlp_mode", "pos_embed")

    # every architecture key rewind_fcn3.yaml sets for the model — must stay
    # constructor-consumed, or get_model's **kwargs pass-through silently drops it
    ARCH_KEYS = (
        "model_grid_type", "sht_grid_type", "scale_factor", "embed_dim", "aux_embed_dim",
        "pos_embed_dim", "num_layers", "sfno_block_frequency", "filter_basis_type",
        "kernel_shape", "num_groups", "normalization_layer", "hard_thresholding_fraction",
        "use_mlp", "mlp_ratio", "activation_function", "big_skip", "bias", "clamp_water",
    )

    @classmethod
    def setUpClass(cls):
        cls.base = _cfg("rewind_fcn3.yaml")["base_config"]
        src = open(os.path.join(_REPO, "makani", "models", "networks", "fourcastnet3.py")).read()
        cls.ctor_args = None
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ClassDef) and node.name == "AtmoSphericNeuralOperatorNet":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        cls.ctor_args = {a.arg for a in item.args.args + item.args.kwonlyargs}

    def test_embed_dim_explicit_and_not_default(self):
        self.assertIn("embed_dim", self.base)
        self.assertNotEqual(self.base["embed_dim"], 8)  # 8 = the silently-tiny constructor default

    def test_arch_keys_constructor_consumed(self):
        self.assertIsNotNone(self.ctor_args, "AtmoSphericNeuralOperatorNet.__init__ not found")
        for k in self.ARCH_KEYS:
            self.assertIn(k, self.ctor_args, f"config key '{k}' no longer a constructor arg — it would be **kwargs-swallowed")
            self.assertIn(k, self.base, f"config no longer sets '{k}'")

    def test_stale_keys_absent(self):
        for k in self.STALE_KEYS:
            self.assertNotIn(k, self.base, f"stale key '{k}' would be silently **kwargs-swallowed")


if __name__ == "__main__":
    unittest.main()
