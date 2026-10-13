# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math

import torch
from torch import nn
import torch.distributed as dist
from makani.utils import comm


# standard torch normalization module base classes whose affine parameters
# (gain / shift) should be excluded from weight decay
_NORM_MODULE_TYPES = (
    nn.LayerNorm,
    nn.GroupNorm,
    nn.modules.batchnorm._BatchNorm,
    nn.modules.instancenorm._InstanceNorm,
)


def _is_norm_module(module: nn.Module) -> bool:
    """True for standard torch norms and makani's custom normalization layers.

    The class-name fallback catches the custom S2 / distributed norms
    (GeometricInstanceNormS2, DistributedInstanceNorm2d, DistributedGeometricInstanceNormS2,
    DistributedLayerNorm) without importing them (which would risk circular imports).
    """
    return isinstance(module, _NORM_MODULE_TYPES) or ("norm" in type(module).__name__.lower())


def get_parameter_groups(model: nn.Module, weight_decay: float, mode: str = "full") -> list:
    """Build optimizer parameter groups according to the weight-decay mode.

    - ``"full"``: a single group with ``weight_decay`` applied to every trainable
      parameter (the historical behavior).
    - ``"transformer"``: two groups -- ``weight_decay`` on the weight matrices and
      ``weight_decay = 0`` on biases and normalization-layer affine parameters, the
      standard modern-transformer convention. Biases are matched by name (robust to
      expanded ``(1, C, 1, 1)`` bias shapes), norm affine params by their owning
      module type, plus a 1-D safety net for any stray scale/gain.

    Any other ``mode`` raises ``ValueError``.
    """
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]

    if mode == "full":
        return [{"params": [p for _, p in trainable], "weight_decay": weight_decay}]

    if mode != "transformer":
        raise ValueError(f"Unknown weight_decay_mode '{mode}', expected 'full' or 'transformer'.")

    # parameters owned by normalization modules (their affine gain / shift)
    norm_param_ids = set()
    for module in model.modules():
        if _is_norm_module(module):
            for p in module.parameters(recurse=False):
                norm_param_ids.add(id(p))

    decay, no_decay = [], []
    for name, p in trainable:
        leaf = name.rsplit(".", 1)[-1]
        if ("bias" in leaf) or (id(p) in norm_param_ids) or (p.ndim < 2):
            no_decay.append(p)
        else:
            decay.append(p)

    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def get_memory_usage(device):
    free_mem, total_mem = torch.cuda.mem_get_info(device=device)
    allocated_mem_gb = (total_mem - free_mem) / (1024.0 * 1024.0 * 1024.0)
    torch_mem_gb = torch.cuda.max_memory_allocated(device=device) / (1024.0 * 1024.0 * 1024.0)

    return allocated_mem_gb, torch_mem_gb


def normalize_weights(model, eps=1e-5):
    with torch.no_grad():
        # Group params by their MP sharding signature — same pattern as clip_grads.
        groups: dict[tuple, list] = {}
        for param in model.parameters():
            key = tuple(
                g for g in getattr(param, "sharded_dims_mp", [])
                if g is not None and comm.get_size(g) > 1
            )
            groups.setdefault(key, []).append(param)

        for mp_groups, params in groups.items():
            # _foreach_norm: one fused kernel for all local norms in the group
            norms = torch._foreach_norm(params, ord=2)

            if mp_groups:
                # Sharded params: we must all_reduce the *squared* partial norms
                # (sum-of-squares is additive across shards) then re-sqrt.
                for norm, param in zip(norms, params):
                    norm.pow_(2)
                    for g in mp_groups:
                        dist.all_reduce(norm, group=comm.get_group(g))
                    norm.sqrt_()

            # per-param scale factors: 1/(norm + eps) — then one fused weight update
            scale_factors = torch._foreach_reciprocal(torch._foreach_add(norms, eps))
            torch._foreach_mul_(params, scale_factors)

    return


def _compute_total_grad_norm(model, norm_type=2.0, verbose=False):
    # Group parameters by their model-parallel sharding signature so we can
    # issue one all_reduce per process group instead of one per parameter,
    # and use _foreach_norm to collapse N small reduce kernels into one.
    groups: dict[tuple, list] = {}
    param_map: dict[tuple, list] = {}  # for verbose NaN reporting
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        key = tuple(
            g for g in getattr(param, "sharded_dims_mp", [])
            if g is not None and comm.get_size(g) > 1
        )
        groups.setdefault(key, []).append(param.grad)
        param_map.setdefault(key, []).append(name)

    ord = 2 if norm_type == 2.0 else 1
    partials = []
    for mp_groups, grads in groups.items():
        # _foreach_norm: one fused kernel instead of one per tensor
        per_tensor_norms = torch._foreach_norm(grads, ord=ord)
        if norm_type == 2.0:
            partial = torch.stack(per_tensor_norms).pow(2).sum()
        else:
            partial = torch.stack(per_tensor_norms).sum()

        # one all_reduce per process group, not per parameter
        for g in mp_groups:
            dist.all_reduce(partial, group=comm.get_group(g))

        if verbose and torch.any(torch.isnan(partial)):
            for name, norm in zip(param_map[mp_groups], per_tensor_norms):
                if torch.isnan(norm):
                    print(f"Gradient norm is NaN for parameter {name}")

        partials.append(partial)

    if partials:
        total_gnorm = torch.stack(partials).sum()
    else:
        total_gnorm = torch.tensor(0.0)

    if norm_type == 2.0:
        total_gnorm = total_gnorm.sqrt()

    return total_gnorm


def clip_grads(model, max_grad_norm, norm_type=2.0, verbose=False):
    with torch.no_grad():
        total_gnorm = _compute_total_grad_norm(model, norm_type=norm_type, verbose=verbose)

        clip_factor = torch.clamp(max_grad_norm / (total_gnorm + 1e-6), max=1.0)

        # skip scaling entirely when the norm is already within bounds
        if clip_factor < 1.0:
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            # _foreach_mul_: one fused kernel instead of one per tensor
            torch._foreach_mul_(grads, clip_factor)

    return total_gnorm


def update_val_tracking(valid_loss, best_valid_loss, epochs_since_improvement, divergence_streak, min_delta=0.0, divergence_threshold=0.0):
    """One epoch's update of the validation bookkeeping that early stopping consumes.

    Pure function (no torch, no side effects) so the stop policy is unit-testable in isolation
    from a live training run. Given this epoch's ``valid_loss`` and the running state, returns
    the updated ``(best_valid_loss, epochs_since_improvement, divergence_streak, improved)``:

    - ``improved`` (bool) tracks the TRUE running minimum and drives the best-checkpoint save,
      so ``best_ckpt`` is always the genuinely-best weights -- independent of ``min_delta``.
    - ``epochs_since_improvement`` resets only on a ``min_delta``-sized improvement over the
      previous best; otherwise it increments. This is what ``early_stopping_patience`` counts,
      so sub-threshold noise on a plateau doesn't keep resetting patience forever. NB this is
      independent of ``ReduceLROnPlateau``'s OWN internal improvement threshold (default 1e-4
      rel), which is what actually gates LR drops -- the two knobs don't interact.
    - ``divergence_streak`` counts CONSECUTIVE epochs whose loss blows past
      ``divergence_threshold * best_valid_loss`` (a genuine training divergence / loss spike).
      Gated on a real best (the 1e6 sentinel of a fresh run is skipped) and a non-zero
      ``divergence_threshold`` (0 disables); resets to 0 the moment the loss drops back under
      the bar. Measured against the previous best, before it is lowered below.
    """
    improved = valid_loss < best_valid_loss

    # early-stopping / plateau counter (min_delta measured against the PREVIOUS best)
    if valid_loss < best_valid_loss - min_delta:
        epochs_since_improvement = 0
    else:
        epochs_since_improvement += 1

    # divergence streak (also vs the previous best, so a spike is caught before best updates)
    if divergence_threshold and (best_valid_loss < 1.0e5) and (valid_loss > best_valid_loss * divergence_threshold):
        divergence_streak += 1
    else:
        divergence_streak = 0

    if improved:
        best_valid_loss = valid_loss

    return best_valid_loss, epochs_since_improvement, divergence_streak, improved


def should_stop_training(epochs_since_improvement, divergence_streak, early_stopping_patience=0, divergence_patience=0):
    """Decide whether to stop, given the counters from :func:`update_val_tracking`.

    Returns a short reason string (``"divergence"`` or ``"early_stopping"``) or ``None``. Both
    patiences are opt-in (0 disables). Checked at the TOP of the epoch loop so a stray
    resume-chain link resubmitted after a stop exits immediately instead of burning an epoch.
    Divergence is reported first (it's the more urgent, more informative signal); it can only
    trip when its streak has actually accrued (the streak stays 0 unless ``divergence_threshold``
    was set in ``update_val_tracking``), so leaving ``divergence_patience`` non-zero while the
    threshold is 0 is a safe no-op.
    """
    if divergence_patience and divergence_streak >= divergence_patience:
        return "divergence"
    if early_stopping_patience and epochs_since_improvement >= early_stopping_patience:
        return "early_stopping"
    return None


def should_skip_step(grad_norm, threshold, steps_taken=0, warmup_steps=0):
    """True if this optimizer step should be SKIPPED (the update rejected) because the pre-clip
    gradient norm is pathological.

    ``threshold <= 0`` disables the guard entirely — a true no-op, so configs that don't opt in
    behave exactly as before. When enabled (``threshold > 0``), skip a **non-finite** norm (nan/inf)
    or a **finite** norm above ``threshold``. Rejecting the update entirely — unlike clipping, which
    still steps in the same (garbage) direction, just shorter — keeps a rare bad batch from kicking
    the weights into a worse basin. NB fp16 AMP already skips inf/nan grads via ``GradScaler``, but
    bf16 has no loss scaling so that net is off; enabling this restores it AND extends it to
    finite-but-huge norms.

    **Warmup exemption (essential):** while ``steps_taken < warmup_steps``, the finite-norm check is
    OFF (non-finite still skips). A freshly initialized model produces legitimately large per-step
    gradients — measured ~260-320 on the first steps of the devel smoke (2026-07-13), where a
    threshold of 150 skipped EVERY step and froze the weights, so the norms never decayed and the
    run stalled permanently. Cold-start gradients are training, not pathology; the guard arms
    itself once the transient has decayed. Callers pass a step counter that is persisted across
    resume (e.g. the trainer's ``iters``), so only a genuinely fresh run gets the exemption.

    Pure (no torch needed to test): ``grad_norm`` may be a python float or a 0-d tensor
    (``float(...)`` handles both); None => don't skip (no norm available)."""
    if threshold <= 0.0 or grad_norm is None:
        return False
    try:
        g = float(grad_norm)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(g):
        return True
    if steps_taken < warmup_steps:
        return False
    return g > threshold


def wandb_register_activations_monitor(model: nn.Module, step: int):

    def check_eligibility(module: nn.Module) -> bool:
        is_activation = False
        is_activation = is_activation or isinstance(module, nn.ReLU)
        is_activation = is_activation or isinstance(module, nn.LeakyReLU)
        is_activation = is_activation or isinstance(module, nn.GELU)
        is_activation = is_activation or isinstance(module, nn.Sigmoid)
        is_activation = is_activation or isinstance(module, nn.SiLU)

        return is_activation

    for submodule in model.modules():
        if check_eligibility(submodule):

            def log_activation(module, step, output):
                name = module.name
                if output.is_complex:
                    normsq = output * output.conj()
                else:
                    normsq = torch.square(output)
                norm = torch.sqrt(torch.sum(normsq))
                wandb.log(f"activation {name}", norm, step=step)

            submodule.register_forward_hook(log_activation, step)

    return
