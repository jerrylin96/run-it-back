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

"""
Resume regression lock for the best-checkpoint clobber (UPSTREAM_ISSUES.md #5).

WHAT THIS ADDS OVER ``tests/test_trainers.py``. That file constructs each trainer, runs
``train()`` once, and restores a trainer from the checkpoint — but it never calls ``train()``
a SECOND time, so nothing there exercises what a resume does to ``best_ckpt``. That is the
whole of #5: the trainers used to initialise a LOCAL ``best_valid_loss = 1.0e6`` at the top of
every ``train()`` call, so the first epoch after a resume compared against 1e6, won, and
overwrote ``best_checkpoint_path`` with a worse model. This file runs the second leg.

All four trainers are covered, because all four save and restore the counter through the same
``driver.py`` path. ``Trainer`` and ``EnsembleTrainer`` additionally persist the
guard counters; the other two deliberately do not (see UPSTREAM_ISSUES.md #5), so the guard
assertions are parameterised off for them rather than skipped silently.

SCOPE. This runs on the synthetic 5-channel / 64x128 HDF5 fixture from ``testutils`` with an
SFNO backbone, single-process on CPU. It verifies the counter round-trip and the
best-checkpoint gate, which is where the bug lived. It does NOT stand in for a cluster run: no
FCN3 backbone, no 72-channel data, no DDP, no SLURM resume chain. The guard-related params
below are the values ``config/rewind_fcn3.yaml``'s ``rewind_fcn3_ens_smoke`` carries, so the
same branches of ``update_val_tracking`` run here as they would there.
"""

import os
import sys
import hashlib
import tempfile
from typing import Optional

import unittest
from parameterized import parameterized

import torch

from makani import Trainer, EnsembleTrainer, StochasticTrainer, AutoencoderTrainer
from makani.utils import comm
from makani.utils.checkpoint_helpers import get_latest_checkpoint_version

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import disable_tf32, set_seed, init_hdf5_dataset
from .test_trainers import init_params

# An unbeatable stored best, written into the checkpoint to force the "post-resume epoch is
# WORSE than the stored best" branch deterministically. Without it, whether that branch is ever
# taken is up to the optimizer, and the decisive assertion could go a whole run untested.
UNBEATABLE_BEST = 1.0e-12

# what the pre-fix trainers reset to on every train() call
PRE_FIX_RESET = 1.0e6

# 73 samples/year (vs testutils' 365) keeps an epoch to a few seconds; dhours follows from it,
# exactly as init_hdf5_dataset computes it, so the fixture stays self-consistent.
SAMPLES_PER_YEAR = 73
DHOURS = (365 * 24) // SAMPLES_PER_YEAR

# (label, trainer, persists the guard counters)
TRAINER_CASES = [
    ("Trainer", Trainer, True),
    ("EnsembleTrainer", EnsembleTrainer, True),
    ("StochasticTrainer", StochasticTrainer, False),
    ("AutoencoderTrainer", AutoencoderTrainer, False),
]

GUARD_COUNTERS = ("epochs_since_improvement", "divergence_streak")


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class TestResumeBestCheckpoint(unittest.TestCase):

    @classmethod
    def setUpClass(cls, path: Optional[str] = "/tmp"):
        cls.datadir = tempfile.TemporaryDirectory(dir=path)
        cls.train_path, cls.n_train_samples, cls.valid_path, cls.n_eval_samples, cls.stats_path, cls.metadata_path, _ = init_hdf5_dataset(
            cls.datadir.name, num_samples_per_year=SAMPLES_PER_YEAR
        )

    @classmethod
    def tearDownClass(cls):
        cls.datadir.cleanup()

    def setUp(self, path: Optional[str] = "/tmp"):

        disable_tf32()
        set_seed(333)

        self.tmpdir = tempfile.TemporaryDirectory(dir=path)
        exp_path = os.path.join(self.tmpdir.name, "experiment_dir")
        os.mkdir(exp_path)
        os.mkdir(os.path.join(exp_path, "training_checkpoints"))

        self.params = init_params(
            exp_path,
            self.train_path,
            self.valid_path,
            self.stats_path,
            batch_size=2,
            ensemble_size=2,
            stochastic_size=5,
            n_history=0,
            n_future=0,
            normalization="zscore",
            num_data_workers=0,
        )

        # testutils' init_params hardcodes a literal "_v0" checkpoint name. Use the versioned
        # path makani/train.py:98 actually builds, so the 3-way rotation and the
        # get_latest_checkpoint_version selection a real resume goes through are exercised too.
        self.params.checkpoint_path = os.path.join(exp_path, "training_checkpoints/ckpt_mp{mp_rank}_v{checkpoint_version}.tar")

        self.params.dhours = DHOURS
        self.params.multifiles = True
        self.params.n_train_samples = self.n_train_samples
        self.params.n_eval_samples = self.n_eval_samples
        self.params.io_grid = [1, 1, 1]
        self.params.io_rank = [0, 0, 0]
        self.params.print_timings_frequency = 0

        # mirrored from config/rewind_fcn3.yaml::rewind_fcn3_ens_smoke. early_stopping_patience
        # cannot fire — no leg here runs 12 epochs. The divergence guard CAN accrue in the two
        # doctored tests below (against a 1e-12 best, any real loss counts as divergent), but it is
        # only consulted at the TOP of the epoch loop and those legs run a single epoch, so it never
        # truncates one. The "resume checkpoint still advanced" assertion is what keeps that honest:
        # if a guard ever did cut a leg short, the epoch counter would not have moved and the
        # "best_ckpt untouched" result would be vacuous.
        self.params.early_stopping_patience = 12
        self.params.early_stopping_min_delta = 1.0e-4
        self.params.divergence_threshold = 2.0
        self.params.divergence_patience = 2

        self.device = torch.device("cpu")
        self.best_checkpoint_path = self.params.best_checkpoint_path.format(mp_rank=comm.get_rank("model"))

    def tearDown(self):
        self.tmpdir.cleanup()

    def _build(self, trainer_handle, max_epochs: int, resuming: bool):
        self.params.max_epochs = max_epochs
        self.params.resuming = resuming
        return trainer_handle(self.params, 0, device=self.device)

    def _run_leg(self, trainer_handle, max_epochs: int, resuming: bool, best_valid_loss_override: Optional[float] = None):
        """Run one leg of a resume chain and return the trainer."""
        trainer = self._build(trainer_handle, max_epochs, resuming)
        if best_valid_loss_override is not None:
            # reinstate the pre-fix code path, which discarded the restored value
            trainer.best_valid_loss = best_valid_loss_override
        trainer.train()
        return trainer

    def _checkpoint_file(self):
        """The checkpoint a resume would pick, resolved through the trainer's own helper."""
        version = get_latest_checkpoint_version(self.params.checkpoint_path)
        return self.params.checkpoint_path.format(mp_rank=comm.get_rank("model"), checkpoint_version=version)

    def _load_checkpoint(self):
        return torch.load(self._checkpoint_file(), map_location="cpu", weights_only=False)

    def _rewrite_stored_best(self, value: float):
        """Overwrite best_valid_loss in the resume checkpoint, leaving everything else alone."""
        checkpoint_file = self._checkpoint_file()
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        checkpoint["best_valid_loss"] = value
        torch.save(checkpoint, checkpoint_file)

    @parameterized.expand(TRAINER_CASES)
    def test_resume_carries_the_best_loss_and_honours_the_improvement_gate(self, _label, trainer_handle, tracks_guards):
        """Two legs, nothing doctored: the counters round-trip, and best_ckpt moves iff the
        post-resume epoch beat the stored best."""

        trainer = self._run_leg(trainer_handle, max_epochs=1, resuming=False)
        checkpoint = self._load_checkpoint()

        with self.subTest(desc="best_valid_loss persisted as a real loss"):
            self.assertIn("best_valid_loss", checkpoint)
            self.assertTrue(torch.isfinite(torch.tensor(float(checkpoint["best_valid_loss"]))))
            self.assertLess(checkpoint["best_valid_loss"], PRE_FIX_RESET)
            self.assertGreater(checkpoint["best_valid_loss"], 0.0)
            self.assertEqual(checkpoint["best_valid_loss"], trainer.best_valid_loss)

        for key in GUARD_COUNTERS:
            with self.subTest(desc=f"{key} persisted={tracks_guards}"):
                self.assertEqual(key in checkpoint, tracks_guards)
                if tracks_guards:
                    self.assertEqual(checkpoint[key], getattr(trainer, key))

        stored_best = checkpoint["best_valid_loss"]
        best_hash_before = sha256_file(self.best_checkpoint_path)

        # leg 2, construction only: the restored trainer must open at the stored best
        resumed = self._build(trainer_handle, max_epochs=2, resuming=True)
        with self.subTest(desc="counters restored, not reset"):
            self.assertEqual(resumed.best_valid_loss, stored_best)
            self.assertNotEqual(resumed.best_valid_loss, PRE_FIX_RESET)
            self.assertEqual(resumed.epoch, trainer.epoch)
            self.assertEqual(resumed.iters, trainer.iters)
            if tracks_guards:
                for key in GUARD_COUNTERS:
                    self.assertEqual(getattr(resumed, key), getattr(trainer, key))

        resumed.train()
        best_hash_after = sha256_file(self.best_checkpoint_path)

        # best_valid_loss only ever decreases, so a strict decrease is exactly "this epoch was
        # recorded as the new best" for Trainer/EnsembleTrainer, whose gate is `<`. StochasticTrainer
        # and AutoencoderTrainer gate on `<=`, so an exact tie would have them rewrite best_ckpt while
        # this reads as "not improved" — two float32 validation losses landing on the same bits is not
        # a case worth designing around, and the forced-branch test below does not depend on it.
        improved = resumed.best_valid_loss < stored_best
        with self.subTest(desc=f"best_ckpt rewritten={improved}"):
            if improved:
                self.assertNotEqual(best_hash_after, best_hash_before, msg=f"epoch improved ({resumed.best_valid_loss} < {stored_best}) but best_ckpt was not rewritten")
            else:
                self.assertEqual(best_hash_after, best_hash_before, msg=f"epoch did not improve on {stored_best} but best_ckpt was rewritten")

    @parameterized.expand(TRAINER_CASES)
    def test_worse_post_resume_epoch_leaves_best_ckpt_untouched(self, _label, trainer_handle, _tracks_guards):
        """The decisive assertion, with the "worse" branch forced so it is always exercised."""

        self._run_leg(trainer_handle, max_epochs=1, resuming=False)
        self._rewrite_stored_best(UNBEATABLE_BEST)
        best_hash_before = sha256_file(self.best_checkpoint_path)

        resumed = self._run_leg(trainer_handle, max_epochs=2, resuming=True)

        with self.subTest(desc="the unbeatable best was restored"):
            self.assertEqual(resumed.best_valid_loss, UNBEATABLE_BEST)
        with self.subTest(desc="best_ckpt untouched"):
            self.assertEqual(sha256_file(self.best_checkpoint_path), best_hash_before)
        with self.subTest(desc="the resume checkpoint still advanced"):
            self.assertEqual(self._load_checkpoint()["epoch"], 2)

    @parameterized.expand(TRAINER_CASES)
    def test_pre_fix_reset_reproduces_the_clobber(self, _label, trainer_handle, _tracks_guards):
        """Control: the same worse epoch DOES overwrite best_ckpt once best_valid_loss is reset
        to 1e6, as the pre-fix code did. Without this, a green result above could mean the
        trainer simply never writes best_ckpt on a second leg, for reasons unrelated to the fix."""

        self._run_leg(trainer_handle, max_epochs=1, resuming=False)
        self._rewrite_stored_best(UNBEATABLE_BEST)
        best_hash_before = sha256_file(self.best_checkpoint_path)

        self._run_leg(trainer_handle, max_epochs=2, resuming=True, best_valid_loss_override=PRE_FIX_RESET)

        self.assertNotEqual(sha256_file(self.best_checkpoint_path), best_hash_before)


if __name__ == "__main__":
    unittest.main()
