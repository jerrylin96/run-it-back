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

import unittest
import torch
import torch.nn as nn

from makani.utils.training.training_helpers import clip_grads, _compute_total_grad_norm, normalize_weights, get_parameter_groups
from makani.utils.training.training_helpers import update_val_tracking, should_stop_training, should_skip_step

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import set_seed, disable_tf32


def _simple_model(*shapes):
    """Build a model whose parameters have the given shapes and synthetic grads."""
    params = nn.ParameterList([nn.Parameter(torch.zeros(*s)) for s in shapes])
    model = nn.Module()
    model.params = params
    return model


def _assign_grads(model, grads):
    for param, g in zip(model.parameters(), grads):
        param.grad = g.clone()


def _global_l2(grads):
    """Reference L2 norm of a flat list of grad tensors."""
    return torch.sqrt(sum(g.float().norm() ** 2 for g in grads))


class TestComputeTotalGradNorm(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def test_matches_reference_single_param(self):
        g = torch.randn(4, 8)
        model = _simple_model((4, 8))
        _assign_grads(model, [g])
        got = _compute_total_grad_norm(model)
        self.assertAlmostEqual(got.item(), _global_l2([g]).item(), places=5)

    def test_matches_reference_multiple_params(self):
        grads = [torch.randn(3, 5), torch.randn(7), torch.randn(2, 2, 2)]
        model = _simple_model((3, 5), (7,), (2, 2, 2))
        _assign_grads(model, grads)
        got = _compute_total_grad_norm(model)
        self.assertAlmostEqual(got.item(), _global_l2(grads).item(), places=5)

    def test_skips_params_with_no_grad(self):
        model = _simple_model((4,), (4,))
        params = list(model.parameters())
        params[0].grad = torch.ones(4)
        # params[1].grad left as None
        got = _compute_total_grad_norm(model)
        ref = _global_l2([params[0].grad])
        self.assertAlmostEqual(got.item(), ref.item(), places=5)

    def test_all_zero_grads(self):
        grads = [torch.zeros(3, 3), torch.zeros(5)]
        model = _simple_model((3, 3), (5,))
        _assign_grads(model, grads)
        got = _compute_total_grad_norm(model)
        self.assertAlmostEqual(got.item(), 0.0, places=6)

    def test_no_grads_returns_zero(self):
        model = _simple_model((4,), (4,))
        # no grads assigned at all
        got = _compute_total_grad_norm(model)
        self.assertAlmostEqual(got.item(), 0.0, places=6)

    def test_complex_grads_supported(self):
        g = torch.randn(4, 4, dtype=torch.complex64)
        model = nn.Module()
        model.p = nn.Parameter(torch.zeros(4, 4, dtype=torch.complex64))
        model.p.grad = g.clone()
        got = _compute_total_grad_norm(model)
        ref = _global_l2([g.abs()])
        self.assertAlmostEqual(got.item(), ref.item(), places=4)


class TestClipGrads(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def test_returned_norm_matches_compute(self):
        grads = [torch.randn(4, 4), torch.randn(8)]
        model = _simple_model((4, 4), (8,))
        _assign_grads(model, grads)
        returned = clip_grads(model, max_grad_norm=100.0)
        self.assertAlmostEqual(returned.item(), _global_l2(grads).item(), places=5)

    def test_no_clip_when_norm_within_bound(self):
        grads = [torch.randn(3, 3)]
        model = _simple_model((3, 3))
        _assign_grads(model, grads)
        original_grad = list(model.parameters())[0].grad.clone()
        clip_grads(model, max_grad_norm=1e6)
        # grads should be unchanged
        self.assertTrue(torch.allclose(list(model.parameters())[0].grad, original_grad))

    def test_clip_reduces_norm_to_max(self):
        grads = [torch.ones(10) * 10.0]  # large norm
        model = _simple_model((10,))
        _assign_grads(model, grads)
        max_norm = 1.0
        clip_grads(model, max_grad_norm=max_norm)
        clipped_norm = _compute_total_grad_norm(model)
        self.assertLessEqual(clipped_norm.item(), max_norm + 1e-5)

    def test_clip_preserves_direction(self):
        g = torch.randn(5, 5)
        model = _simple_model((5, 5))
        _assign_grads(model, [g])
        clip_grads(model, max_grad_norm=0.1)
        clipped = list(model.parameters())[0].grad
        # direction preserved: clipped grad is a positive scalar multiple of original
        ratio = clipped / g
        self.assertTrue(torch.allclose(ratio, ratio[0, 0].expand_as(ratio), atol=1e-5))
        self.assertGreater(ratio[0, 0].item(), 0.0)

    def test_grads_without_grad_unaffected(self):
        model = _simple_model((4,), (4,))
        params = list(model.parameters())
        params[0].grad = torch.ones(4) * 100.0
        # params[1].grad is None — should not raise
        clip_grads(model, max_grad_norm=0.01)
        self.assertIsNone(params[1].grad)

    def test_clip_with_multiple_params(self):
        grads = [torch.randn(4, 4) * 5, torch.randn(8) * 5]
        model = _simple_model((4, 4), (8,))
        _assign_grads(model, grads)
        max_norm = 1.0
        clip_grads(model, max_grad_norm=max_norm)
        clipped_norm = _compute_total_grad_norm(model)
        self.assertLessEqual(clipped_norm.item(), max_norm + 1e-5)


class TestNormalizeWeights(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def _param_l2(self, param):
        return param.data.float().norm().item()

    def test_single_param_has_unit_norm_after(self):
        model = _simple_model((4, 8))
        with torch.no_grad():
            list(model.parameters())[0].fill_(2.0)
        normalize_weights(model)
        self.assertAlmostEqual(self._param_l2(list(model.parameters())[0]), 1.0, places=5)

    def test_multiple_params_each_have_unit_norm(self):
        model = _simple_model((3, 4), (5,), (2, 2))
        with torch.no_grad():
            for i, p in enumerate(model.parameters()):
                p.fill_(float(i + 1))
        normalize_weights(model)
        for p in model.parameters():
            self.assertAlmostEqual(self._param_l2(p), 1.0, places=5)

    def test_zero_param_not_nan(self):
        model = _simple_model((4,))
        with torch.no_grad():
            list(model.parameters())[0].zero_()
        normalize_weights(model, eps=1e-5)
        p = list(model.parameters())[0]
        self.assertFalse(torch.isnan(p).any())

    def test_direction_preserved(self):
        model = _simple_model((4,))
        p = list(model.parameters())[0]
        with torch.no_grad():
            p.copy_(torch.tensor([3.0, 4.0, 1.0, 2.0]))
        original = p.data.clone()
        normalize_weights(model)
        # after normalization the parameter should be a positive scalar multiple
        # of the original — compare unit vectors to avoid dividing by zero
        unit_before = original / original.norm()
        unit_after  = p.data / p.data.norm()
        self.assertTrue(torch.allclose(unit_after, unit_before, atol=1e-5))

    def test_complex_params_supported(self):
        model = nn.Module()
        model.p = nn.Parameter(torch.randn(4, 4, dtype=torch.complex64) * 3.0)
        normalize_weights(model)
        norm = model.p.data.abs().norm().item()
        self.assertAlmostEqual(norm, 1.0, places=4)


class _WeirdNorm(nn.Module):
    """A norm-like module whose gain is 2-D and named 'weight'. Only module-type
    detection (not the name or ndim rules) can classify its gain as no-decay -- this
    proves the norm detection works beyond the simple heuristics."""

    def __init__(self, n=4):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n, n))
        self.bias = nn.Parameter(torch.zeros(n))

    def forward(self, x):
        return x


class _ParamGroupNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)                                   # weight(2D)->decay, bias(1D)->no_decay
        self.conv = nn.Conv2d(4, 4, 1)                               # weight(4D)->decay, bias(1D)->no_decay
        self.ln = nn.LayerNorm(8)                                    # weight/bias -> no_decay (module type)
        self.wnorm = _WeirdNorm(4)                                   # 2D gain -> no_decay only via module type
        self.expanded_bias = nn.Parameter(torch.zeros(1, 4, 1, 1))   # 4D bias -> no_decay BY NAME
        self.plain_weight = nn.Parameter(torch.randn(8, 8))          # 2D weight -> decay

    def forward(self, x):
        return x


class TestParameterGroups(unittest.TestCase):
    def setUp(self):
        set_seed(0)
        self.model = _ParamGroupNet()
        self.wd = 0.1

    def _named(self):
        return dict(self.model.named_parameters())

    def test_full_mode_is_single_group(self):
        groups = get_parameter_groups(self.model, self.wd, "full")
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["weight_decay"], self.wd)
        self.assertEqual(len(groups[0]["params"]), sum(1 for _ in self.model.parameters()))

    def test_transformer_mode_splits_norms_and_biases(self):
        groups = get_parameter_groups(self.model, self.wd, "transformer")
        self.assertEqual(len(groups), 2)

        decay = next(g for g in groups if g["weight_decay"] == self.wd)
        no_decay = next(g for g in groups if g["weight_decay"] == 0.0)
        decay_ids = {id(p) for p in decay["params"]}
        no_decay_ids = {id(p) for p in no_decay["params"]}

        named = self._named()

        # weight matrices keep weight decay
        for nm in ("lin.weight", "conv.weight", "plain_weight"):
            self.assertIn(id(named[nm]), decay_ids, f"{nm} should be in the decay group")

        # biases (incl. the 4-D expanded one) and all norm affine params (incl. the
        # 2-D weird-norm gain) are excluded from weight decay
        for nm in ("lin.bias", "conv.bias", "expanded_bias",
                   "ln.weight", "ln.bias", "wnorm.weight", "wnorm.bias"):
            self.assertIn(id(named[nm]), no_decay_ids, f"{nm} should be in the no-decay group")

        # partition is exact: every param classified once, no overlap
        total = sum(1 for _ in self.model.parameters())
        self.assertEqual(len(decay_ids) + len(no_decay_ids), total)
        self.assertEqual(decay_ids & no_decay_ids, set())

    def test_unknown_mode_raises(self):
        for bad in ("selective", "Transformer", "none", ""):
            with self.assertRaises(ValueError):
                get_parameter_groups(self.model, self.wd, bad)


def _drive_epochs(seq, min_delta=0.0, divergence_threshold=0.0, early_stopping_patience=0, divergence_patience=0):
    """Drive the epoch loop exactly like deterministic_trainer.Trainer.train():
    stop-check at the TOP of each iteration, then the end-of-epoch update. Returns a dict
    with the stop epoch/reason, the final counters, and the epochs where best_ckpt would
    have been (re)written (the ``improved`` flag)."""
    best, esi, streak = 1.0e6, 0, 0
    best_ckpt_saves = []
    for e, vl in enumerate(seq):
        reason = should_stop_training(esi, streak, early_stopping_patience, divergence_patience)
        if reason is not None:
            return {"stopped_epoch": e, "reason": reason, "best": best, "esi": esi, "streak": streak, "best_ckpt_saves": best_ckpt_saves}
        best, esi, streak, improved = update_val_tracking(vl, best, esi, streak, min_delta, divergence_threshold)
        if improved:
            best_ckpt_saves.append(e)
    return {"stopped_epoch": None, "reason": None, "best": best, "esi": esi, "streak": streak, "best_ckpt_saves": best_ckpt_saves}


class TestEarlyStoppingPolicy(unittest.TestCase):
    """Locks the early-stopping / divergence-guard bookkeeping (the validation-loss-spike fix).

    These are the pure decision helpers the deterministic trainer calls per epoch; testing
    them in isolation exercises the exact policy without spinning up a real training run."""

    def test_improved_tracks_true_minimum_for_best_ckpt(self):
        # improved is a strict new minimum, independent of min_delta -> best_ckpt is the true best
        best, esi, streak, improved = update_val_tracking(0.5, 1.0e6, 0, 0)
        self.assertTrue(improved)
        self.assertEqual(best, 0.5)
        # a worse epoch is not an improvement and does not lower best
        best, esi, streak, improved = update_val_tracking(0.7, best, esi, streak)
        self.assertFalse(improved)
        self.assertEqual(best, 0.5)
        self.assertEqual(esi, 1)

    def test_min_delta_decouples_ckpt_from_patience_counter(self):
        # a slow sub-min_delta drift: best_ckpt saves every epoch (true min) but patience climbs
        slow = [0.030 - 5e-5 * i for i in range(20)]  # 5e-5/epoch < min_delta 1e-4
        r = _drive_epochs(slow, min_delta=1e-4, early_stopping_patience=100)
        self.assertEqual(r["best_ckpt_saves"], list(range(20)))  # every epoch is a strict min
        self.assertAlmostEqual(r["best"], slow[-1])
        self.assertEqual(r["esi"], 19)  # epoch 0 set best; 1..19 never beat best by min_delta
        # with min_delta 0 (old strict behavior) the same drift resets the counter every epoch
        r0 = _drive_epochs(slow, min_delta=0.0, early_stopping_patience=100)
        self.assertEqual(r0["esi"], 0)

    def test_divergence_guard_stops_fast_and_preserves_best(self):
        # the p4 shape: descend, plateau, then a ~10x spike that stays high
        descend = [0.043, 0.031, 0.027, 0.025, 0.0246]
        plateau = [0.0243, 0.0242, 0.0242, 0.0242, 0.0242]
        spike = [0.24, 0.24, 0.24, 0.24]
        seq = descend + plateau + spike
        spike_epoch = len(descend) + len(plateau)
        r = _drive_epochs(seq, min_delta=1e-4, divergence_threshold=2.0, divergence_patience=2, early_stopping_patience=30)
        self.assertEqual(r["reason"], "divergence")
        # streak reaches 2 at spike_epoch+1; the top-of-loop check fires at spike_epoch+2
        self.assertEqual(r["stopped_epoch"], spike_epoch + 2)
        # best_ckpt holds the plateau minimum, never the spike
        self.assertAlmostEqual(r["best"], 0.0242)
        self.assertLess(max(r["best_ckpt_saves"]), spike_epoch)

    def test_divergence_disabled_leaves_early_stopping_only(self):
        # same spike, guard OFF (threshold 0) -> never a divergence stop; streak stays 0
        seq = [0.043, 0.03, 0.025, 0.024, 0.24, 0.24, 0.24]
        r = _drive_epochs(seq, divergence_threshold=0.0, divergence_patience=0, early_stopping_patience=30)
        self.assertEqual(r["streak"], 0)
        self.assertIn(r["reason"], (None, "early_stopping"))

    def test_single_epoch_transient_does_not_trip_patience_two(self):
        transient = [0.10, 0.05, 0.04, 0.09, 0.038, 0.037]  # one 2.25x bump then recovers
        r = _drive_epochs(transient, divergence_threshold=2.0, divergence_patience=2, early_stopping_patience=50)
        self.assertIsNone(r["reason"])

    def test_divergence_streak_resets_when_loss_recovers(self):
        recover = [0.10, 0.04, 0.20, 0.04, 0.20, 0.04]  # never two divergent epochs in a row
        r = _drive_epochs(recover, divergence_threshold=2.0, divergence_patience=2, early_stopping_patience=50)
        self.assertIsNone(r["reason"])
        self.assertEqual(r["streak"], 0)

    def test_fresh_run_sentinel_is_not_divergence(self):
        # epoch 0 compares against the 1e6 sentinel; must not read as a divergence
        best, esi, streak, improved = update_val_tracking(0.5, 1.0e6, 0, 0, divergence_threshold=2.0)
        self.assertEqual(streak, 0)
        self.assertTrue(improved)

    def test_should_stop_reports_divergence_first(self):
        # when both would trip, divergence is the reported (more urgent) reason
        self.assertEqual(should_stop_training(99, 5, early_stopping_patience=30, divergence_patience=2), "divergence")
        self.assertEqual(should_stop_training(99, 0, early_stopping_patience=30, divergence_patience=2), "early_stopping")
        self.assertIsNone(should_stop_training(5, 0, early_stopping_patience=30, divergence_patience=2))
        # both patiences opt-in: zero disables regardless of counter
        self.assertIsNone(should_stop_training(999, 999, early_stopping_patience=0, divergence_patience=0))


class TestSkipStepGuard(unittest.TestCase):
    """Locks the skip-step guard (reject an optimizer step on a pathological grad norm)."""

    def test_disabled_when_threshold_nonpositive(self):
        # threshold <= 0 is a true no-op: nothing is skipped, even a huge or non-finite norm
        for g in (0.077, 5000.0, float("nan"), float("inf")):
            self.assertFalse(should_skip_step(g, 0.0), f"threshold 0 must not skip {g}")
            self.assertFalse(should_skip_step(g, -1.0), f"negative threshold must not skip {g}")

    def test_healthy_and_coldstart_not_skipped(self):
        # the p4 run's healthy ~0.077 and epoch-1 cold-start ~2-3 are far below 150
        self.assertFalse(should_skip_step(0.077, 150))
        self.assertFalse(should_skip_step(2.7, 150))

    def test_spike_skipped(self):
        # the epoch-38 spike (~950, peak ~5000) is well above 150
        self.assertTrue(should_skip_step(950.0, 150))
        self.assertTrue(should_skip_step(5000.0, 150))

    def test_nonfinite_skipped_when_enabled(self):
        self.assertTrue(should_skip_step(float("nan"), 150))
        self.assertTrue(should_skip_step(float("inf"), 150))

    def test_boundary_is_strict_greater_than(self):
        self.assertFalse(should_skip_step(150.0, 150))     # exactly at threshold is kept
        self.assertTrue(should_skip_step(150.0001, 150))   # just above is skipped

    def test_none_norm_does_not_skip(self):
        # no grad norm available (e.g. clipping off) -> can't judge -> don't skip
        self.assertFalse(should_skip_step(None, 150))

    def test_accepts_zero_d_tensor(self):
        # the trainer passes the clip norm as a 0-d tensor; float() must handle it
        self.assertTrue(should_skip_step(torch.tensor(950.0), 150))
        self.assertFalse(should_skip_step(torch.tensor(0.077), 150))
        self.assertTrue(should_skip_step(torch.tensor(float("nan")), 150))

    def test_warmup_exempts_finite_coldstart_norms(self):
        # the devel-smoke regression (2026-07-13): fresh-model per-step norms ~260-320 must NOT be
        # skipped during warmup, or the weights freeze and the run stalls permanently
        for step in (1, 100, 499):
            self.assertFalse(should_skip_step(300.0, 150, steps_taken=step, warmup_steps=500), f"cold-start norm at iter {step} must train")
        # ...but once warmup has elapsed the same norm is pathological and IS skipped
        for step in (500, 501, 10000):
            self.assertTrue(should_skip_step(300.0, 150, steps_taken=step, warmup_steps=500), f"post-warmup norm at iter {step} must skip")

    def test_warmup_does_not_exempt_nonfinite(self):
        # a NaN/inf gradient is broken at ANY stage — warmup only covers legitimately-large finite norms
        self.assertTrue(should_skip_step(float("nan"), 150, steps_taken=1, warmup_steps=500))
        self.assertTrue(should_skip_step(float("inf"), 150, steps_taken=1, warmup_steps=500))

    def test_warmup_default_zero_preserves_old_behavior(self):
        # callers that don't pass warmup get the original semantics (guarded from step 0)
        self.assertTrue(should_skip_step(300.0, 150))
        self.assertTrue(should_skip_step(300.0, 150, steps_taken=0, warmup_steps=0))


if __name__ == "__main__":
    unittest.main()
