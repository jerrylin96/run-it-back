# SPDX-License-Identifier: Apache-2.0
#
# behavioral direction test for the backward-in-time emulator.
#
# Question this answers
# ---------------------
# A low *single-step* validation loss does NOT prove that the "backward" checkpoint
# predicts X(t-dt) (the past) rather than X(t+dt) (the future): a 6 h step is easy in
# either direction because persistence dominates. This script distinguishes the two
# directions behaviorally.
#
# Method
# ------
#   * We deliberately read data with reverse_time = False (a normal FORWARD loader) and
#     build the triplet (past, present, future) ourselves from raw consecutive indices.
#     We do NOT use the reverse_time dataloader to label past/future -- if an offset-swap
#     bug existed, the model and a reverse_time-based eval would share it and the bug would
#     be invisible. The checkpoint supplies only the trained WEIGHTS; direction must show up
#     in behavior, not in how we fetch data.
#   * For each anchor time t_i: feed present = X(t_i) (+ its zenith at t_i) through the
#     checkpoint -> prediction Yhat. A single forward step appends only the INPUT-side
#     zenith (stepper.SingleStepWrapper), so the model sees frame_t + zenith(t) and no
#     target-time cue: the direction lives entirely in the weights. Compare Yhat against
#     the true past X(t_i - dt) and true future X(t_i + dt).
#   * Primary metric -- increment correlation. With the present subtracted off (removing the
#     persistence baseline), weather advection makes the past-increment and future-increment
#     point in OPPOSITE directions. We report the area-weighted Pearson correlation of the
#     predicted increment dP = Yhat - present with the past increment (past - present) and
#     the future increment (future - present):
#         r_past  = corr(dP, past - present)
#         r_future= corr(dP, future - present)
#     A backward model: r_past >> r_future. A (mislabeled) forward model: the mirror image.
#     A model that merely copies the input (persistence / blur): r_past ~ r_future ~ 0.
#   * Secondary metrics -- area-weighted RMSE of Yhat to past vs future (physical units), the
#     persistence-RMSE baselines, and a normalized-space single-step MSE against the past vs
#     future target (the quantity most comparable to the training loss).
#
# The same script serves as the forward-checkpoint CONTROL: point it at the forward
# checkpoint and it should report the mirror verdict (FUTURE), validating the harness.
#
# Runs single-GPU inside the container. Architecture-agnostic: it reads the checkpoint's own config.

import os
import json
import argparse
import logging

import numpy as np
import torch

from makani.utils import logging_utils
from makani.utils.YParams import YParams
from makani.utils import comm
from makani.utils.parse_dataset_metada import parse_dataset_metadata
from makani.utils import argument_parser
from makani.utils.argument_parser import parse_odirect_config
from makani import Inferencer


# ---- area-weighted spatial statistics (cos-latitude weighting) ---------------------------

def _make_weight(lats_deg, width, device):
    w = np.cos(np.deg2rad(np.asarray(lats_deg, dtype=np.float64)))
    w = np.clip(w, 0.0, None)
    wt = torch.tensor(w, dtype=torch.float32, device=device).view(1, 1, -1, 1)  # (1,1,H,1)
    wsum = wt.sum() * float(width)
    return wt, wsum


def _wmean(x, wt, wsum):
    # area-weighted mean over the trailing (H, W) dims. Reduce over (-2, -1) -- NOT a hardcoded
    # (2, 3) -- so the channel axis is never collapsed even if a stray leading dim sneaks in.
    return (x * wt).sum(dim=(-2, -1), keepdim=True) / wsum


def _wcorr(a, b, wt, wsum):
    # no eps in the denominator: it would break unit-invariance (see backward_rollout._wacc)
    ma, mb = _wmean(a, wt, wsum), _wmean(b, wt, wsum)
    am, bm = a - ma, b - mb
    cov = _wmean(am * bm, wt, wsum)
    va = _wmean(am * am, wt, wsum)
    vb = _wmean(bm * bm, wt, wsum)
    return (cov / torch.sqrt(va * vb)).reshape(-1)  # (C,)


def _wrmse(a, b, wt, wsum):
    return torch.sqrt(_wmean((a - b) ** 2, wt, wsum)).reshape(-1)  # (C,)


# ---- setup: mirror makani/inference.py up to building the Inferencer ---------------------

def build_inferencer(args):
    params = YParams(os.path.abspath(args.yaml_config), args.config)

    # parallelism: single process, no model/ensemble parallelism
    params["ensemble_parallel_size"] = 1
    params["matmul_parallel_size"] = args.matmul_parallel_size
    params["h_parallel_size"] = args.h_parallel_size
    params["w_parallel_size"] = args.w_parallel_size
    params["data_parallel_sizes"] = [1, -1]
    params["data_parallel_names"] = ["ensemble", "batch"]
    params["model_parallel_sizes"] = [args.h_parallel_size, args.w_parallel_size, args.matmul_parallel_size]
    params["model_parallel_names"] = ["h", "w", "matmul"]
    params["load_checkpoint"] = args.load_checkpoint

    comm.init(
        model_parallel_sizes=params["model_parallel_sizes"],
        model_parallel_names=params["model_parallel_names"],
        data_parallel_sizes=params["data_parallel_sizes"],
        data_parallel_names=params["data_parallel_names"],
        verbose=False,
    )
    world_rank = comm.get_world_rank()

    params["world_size"] = comm.get_world_size()
    params["batch_size"] = 1
    params["global_batch_size"] = 1
    params["ensemble_size"] = 1
    params["local_ensemble_size"] = 1

    torch.cuda.set_device(comm.get_local_rank())
    torch.backends.cudnn.benchmark = True

    # TF32 was unconditionally forced on here, which made --amp_mode none NOT a clean fp32
    # control: TF32 matmuls keep a ~5.6E-4 RMS rounding floor even with autocast disabled.
    # That matters for the backward rollout specifically, because rounding noise is BROADBAND
    # -- it deposits energy at high l, where the truth spectrum is orders of magnitude below
    # the large scales, and an anti-diffusive backward map then amplifies it. So a precision
    # artifact and genuine model instability produce the same tail_power_ratio signature and
    # can only be separated by actually turning the floor off. Default True keeps every prior
    # run comparable; the caller passes tf32=False for the control arm.
    use_tf32 = bool(getattr(args, "tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32

    # experiment dir (only used for output bookkeeping)
    expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
    params["experiment_dir"] = os.path.abspath(expDir)
    if world_rank == 0 and not os.path.isdir(expDir):
        os.makedirs(expDir, exist_ok=True)

    # checkpoint location: a directory holding the training_checkpoints files.
    # default: <exp_dir>/<config>/<run_num>/training_checkpoints
    ckpt_dir = args.checkpoint_path or os.path.join(expDir, "training_checkpoints")
    if args.checkpoint_select == "best":
        params["checkpoint_path"] = os.path.join(ckpt_dir, "best_ckpt_mp{mp_rank}.tar")
    else:
        params["checkpoint_path"] = os.path.join(ckpt_dir, "ckpt_mp{mp_rank}_v{checkpoint_version}.tar")
    params["best_checkpoint_path"] = os.path.join(ckpt_dir, "best_ckpt_mp{mp_rank}.tar")

    params["resuming"] = False
    params["amp_mode"] = args.amp_mode
    params["jit_mode"] = "none"
    params["enable_odirect"], params["odirect_alignment"] = parse_odirect_config(None)
    params["enable_s3"] = False
    params["disable_ddp"] = True
    params["checkpointing_level"] = 0
    params["enable_synthetic_data"] = False
    params["split_data_channels"] = False
    params["mask_file"] = None
    params["climatology_file"] = None
    params["dataset_file_suffix"] = "h5"
    params["multifiles"] = True

    # ---- the crux: read data FORWARD, single-step, regardless of the config target ----
    params["reverse_time"] = False
    params["n_history"] = 0
    params["n_future"] = 0
    params["valid_autoreg_steps"] = 0

    # ---- optional INPUT-NOISE AMPLITUDE override (noise-amplitude diagnostic) -------------
    # FCN3's ensemble mechanism draws 8 correlated multi-scale noise channels and appends them
    # as auxiliary INPUTS at every autoregressive step. In a backward rollout that means fresh
    # small-scale structure is injected -720 h worth of times into a map we already know
    # amplifies small scales, so "the noise seeds what the map grows" and "the map is
    # intrinsically unstable" produce the same blow-up and can only be separated by turning the
    # amplitude down. Scaling sigma is the least invasive way to do that: it keeps the SAME
    # noise class, kT scales, channel count, seeding and tensor shapes (so the trained
    # checkpoint loads unchanged and the input layout the model was trained on is preserved),
    # and only rescales sigma_l -- at sigma=0 the state is allocated zero and every AR update
    # is state <- phi*state + 0, so the emitted field is identically zero. That gives a
    # continuous dose-response knob (1.0 / 0.5 / 0.25 / 0.0) rather than a binary switch.
    # CAVEAT worth stating in any writeup: sigma=0 is out of the training distribution (the
    # model never saw a noiseless input), so a zero-noise run diagnoses the MAP, it is not a
    # calibrated forecast. None (default) leaves the config value untouched.
    sigma_override = getattr(args, "input_noise_sigma", None)
    if sigma_override is not None:
        if params.get("input_noise", None) is None:
            raise ValueError(
                f"--input_noise_sigma {sigma_override} was requested but config target "
                f"'{args.config}' has no input_noise block, so there is no noise amplitude to "
                "override. Point this at an FCN3 ensemble target (config/rewind_fcn3.yaml)."
            )
        if float(sigma_override) < 0.0:
            raise ValueError(f"--input_noise_sigma must be >= 0, got {sigma_override}")
        # in-place on the mapping the params dict holds, so params.input_noise (the same
        # object, bound by ParamsBase.__setitem__) reflects it too
        params["input_noise"]["sigma"] = float(sigma_override)

    if params["wandb_name"] is None:
        params["wandb_name"] = args.config + "_dirverify_" + str(args.run_num)
    if params["wandb_group"] is None:
        params["wandb_group"] = "rewind"
    if not hasattr(params, "wandb_dir") or params["wandb_dir"] is None:
        params["wandb_dir"] = os.path.abspath(expDir)

    if world_rank == 0:
        logging_utils.config_logger()
        logging_utils.log_versions()

    params["log_to_wandb"] = False
    params["log_to_screen"] = (world_rank == 0) and params["log_to_screen"]

    if "metadata_json_path" in params:
        params, _ = parse_dataset_metadata(params["metadata_json_path"], params=params)
    else:
        raise RuntimeError("Error, please specify a dataset descriptor file in json format")

    inferencer = Inferencer(params, world_rank)
    return inferencer, params, world_rank


# ---- main test loop ----------------------------------------------------------------------

def run(args):
    inf, params, world_rank = build_inferencer(args)
    inf._set_eval()  # model.eval() + preprocessor.eval() -> uses the *_eval unpredicted caches

    model = inf.model
    preproc = inf.preprocessor
    device = inf.device
    dataset = inf.valid_dataset

    channel_names = list(params.channel_names)
    C = len(channel_names)

    # physical-unit (de)normalization: inputs read with IN norm, targets/preds with OUT norm
    in_bias, in_scale = dataset.get_input_normalization()      # (1,C,1,1) numpy
    out_bias, out_scale = dataset.get_output_normalization()
    ib = torch.as_tensor(in_bias, dtype=torch.float32, device=device)
    isc = torch.as_tensor(in_scale, dtype=torch.float32, device=device)
    ob = torch.as_tensor(out_bias, dtype=torch.float32, device=device)
    osc = torch.as_tensor(out_scale, dtype=torch.float32, device=device)

    def denorm_in(x):
        return x * isc + ib

    def denorm_out(x):
        return x * osc + ob

    def renorm_out(x_phys):
        return (x_phys - ob) / osc

    H = dataset.img_shape[0]
    W = dataset.img_shape[1]
    wt, wsum = _make_weight(dataset.lat_lon[0], W, device)

    # anchors: evenly spaced present-times across the validation set, each with both neighbors
    n_total = dataset.n_samples_total
    lo, hi = 1, n_total - 2
    if hi < lo:
        raise RuntimeError(f"dataset too small ({n_total} frames) to form past/present/future triplets")
    n_anchors = min(args.num_anchors, hi - lo + 1)
    anchors = np.unique(np.linspace(lo, hi, num=n_anchors, dtype=int)).tolist()

    R_past, R_fut = [], []
    RMSE_past, RMSE_fut = [], []
    RMSE_pers_past, RMSE_pers_fut = [], []
    MSE_norm_past, MSE_norm_fut = [], []
    used_dates = []

    if world_rank == 0:
        logging.info(f"channels: {channel_names}")
        logging.info(f"dataset frames: {n_total}, dhours: {dataset.dhours}, dt: {params.dt} "
                     f"(step = {dataset.dhours * params.dt} h)")
        logging.info(f"running {len(anchors)} anchors over [{lo}, {hi}]")

    for k, idx in enumerate(anchors):
        s_now = dataset.get_sample_at_index(int(idx), return_target=True)
        s_prev = dataset.get_sample_at_index(int(idx) - 1, return_target=True)
        # add_zenith + return_timestamp + return_target -> (inp, tar, zen_inp, zen_tar, t_inp, t_tar)
        inp, tar, zen_inp = s_now[0], s_now[1], s_now[2]
        t_inp = s_now[4]
        inp_prev = s_prev[0]

        # consistency check: prev-sample's target frame == this-sample's input frame
        if k == 0:
            same = torch.allclose(s_prev[1], inp, atol=1e-4, rtol=0)
            if world_rank == 0 and not same:
                logging.warning("triplet indexing check: prev.target != now.input (unexpected)")

        present = inp.unsqueeze(0).to(device)        # (1,1,C,H,W)
        zen = zen_inp.unsqueeze(0).to(device)        # (1,1,1,H,W)
        inp_flat = preproc.flatten_history(present)  # (1,C,H,W)
        preproc.cache_unpredicted_features(None, None, zen, None)

        with torch.inference_mode():
            with inf.autocast():
                pred = model(inp_flat, update_state=True, replace_state=True)  # (1,C,H,W), out-norm
        pred = pred.float()

        # Metric fields are 4D (1, C, H, W). Do NOT unsqueeze here: inp/inp_prev/tar are already
        # (1, C, H, W), and an extra leading dim would push the channel axis into the area-mean
        # reduction (collapsing channels). The model-input path above keeps its own 5D copy for
        # flatten_history; that copy is separate from these metric tensors.
        present_p = denorm_in(inp.to(device))       # (1, C, H, W) physical, IN  norm
        past_p = denorm_in(inp_prev.to(device))     # (1, C, H, W) physical, IN  norm
        future_p = denorm_out(tar.to(device))       # (1, C, H, W) physical, OUT norm
        yhat_p = denorm_out(pred)                   # (1, C, H, W) physical, OUT norm

        if k == 0:
            assert present_p.dim() == 4 and present_p.shape[1] == C, \
                f"expected (1,{C},H,W) metric tensors, got {tuple(present_p.shape)}"
            assert yhat_p.shape == present_p.shape == past_p.shape == future_p.shape, \
                f"metric shape mismatch: {tuple(yhat_p.shape)} vs {tuple(present_p.shape)}"

        dP = yhat_p - present_p
        dPast = past_p - present_p
        dFut = future_p - present_p

        R_past.append(_wcorr(dP, dPast, wt, wsum).detach().cpu().numpy())
        R_fut.append(_wcorr(dP, dFut, wt, wsum).detach().cpu().numpy())
        RMSE_past.append(_wrmse(yhat_p, past_p, wt, wsum).detach().cpu().numpy())
        RMSE_fut.append(_wrmse(yhat_p, future_p, wt, wsum).detach().cpu().numpy())
        RMSE_pers_past.append(_wrmse(present_p, past_p, wt, wsum).detach().cpu().numpy())
        RMSE_pers_fut.append(_wrmse(present_p, future_p, wt, wsum).detach().cpu().numpy())

        # normalized-space single-step MSE (closest analogue to the training loss)
        past_norm = renorm_out(past_p)
        fut_norm = tar.to(device)  # already out-normalized, (1, C, H, W)
        MSE_norm_past.append(_wmean((pred - past_norm) ** 2, wt, wsum).reshape(-1).detach().cpu().numpy())
        MSE_norm_fut.append(_wmean((pred - fut_norm) ** 2, wt, wsum).reshape(-1).detach().cpu().numpy())

        try:
            used_dates.append(str(dataset.get_time_at_index(int(idx))))
        except Exception:
            used_dates.append(str(np.asarray(t_inp).ravel()[0]))

    if world_rank != 0:
        comm.cleanup()
        return

    R_past = np.asarray(R_past)            # (N, C)
    R_fut = np.asarray(R_fut)
    RMSE_past = np.asarray(RMSE_past)
    RMSE_fut = np.asarray(RMSE_fut)
    RMSE_pers_past = np.asarray(RMSE_pers_past)
    RMSE_pers_fut = np.asarray(RMSE_pers_fut)
    MSE_norm_past = np.asarray(MSE_norm_past)
    MSE_norm_fut = np.asarray(MSE_norm_fut)

    mean_r_past = np.nanmean(R_past, axis=0)
    mean_r_fut = np.nanmean(R_fut, axis=0)
    std_r_past = np.nanstd(R_past, axis=0)
    std_r_fut = np.nanstd(R_fut, axis=0)
    frac_past = np.nanmean((R_past > R_fut).astype(np.float64), axis=0)

    dyn_set = [c for c in ["z500", "t500", "u500", "v500", "msl"] if c in channel_names]
    dyn_idx = [channel_names.index(c) for c in dyn_set]

    def per_channel_verdict(rp, rf, margin=0.05, floor=0.10):
        if (rp > rf + margin) and (rp > floor):
            return "PAST"
        if (rf > rp + margin) and (rf > floor):
            return "FUTURE"
        return "INCONCLUSIVE"

    # ---- pretty table ----
    sep = "-" * 104
    print(sep)
    print(f"DIRECTION VERIFICATION  |  config={args.config}  run_num={args.run_num}  "
          f"checkpoint_select={args.checkpoint_select}")
    print(f"checkpoint_path={params.checkpoint_path}")
    print(f"anchors={R_past.shape[0]}  step={dataset.dhours * params.dt} h  "
          f"dates: {used_dates[0]} .. {used_dates[-1]}")
    print(sep)
    print(f"{'channel':<8} {'r_past':>9} {'r_future':>9} {'frac>':>7} "
          f"{'rmse_past':>11} {'rmse_fut':>11} {'pers_past':>11} {'pers_fut':>11}  verdict")
    for c in range(C):
        print(f"{channel_names[c]:<8} {mean_r_past[c]:>9.3f} {mean_r_fut[c]:>9.3f} {frac_past[c]:>7.2f} "
              f"{np.nanmean(RMSE_past[:, c]):>11.4g} {np.nanmean(RMSE_fut[:, c]):>11.4g} "
              f"{np.nanmean(RMSE_pers_past[:, c]):>11.4g} {np.nanmean(RMSE_pers_fut[:, c]):>11.4g}  "
              f"{per_channel_verdict(mean_r_past[c], mean_r_fut[c])}")
    print(sep)

    dyn_rp = float(np.mean(mean_r_past[dyn_idx])) if dyn_idx else float("nan")
    dyn_rf = float(np.mean(mean_r_fut[dyn_idx])) if dyn_idx else float("nan")
    all_rp = float(np.mean(mean_r_past))
    all_rf = float(np.mean(mean_r_fut))
    print(f"dynamical channels {dyn_set}: mean r_past={dyn_rp:.3f}  mean r_future={dyn_rf:.3f}")
    print(f"all channels:        mean r_past={all_rp:.3f}  mean r_future={all_rf:.3f}")

    overall = per_channel_verdict(dyn_rp, dyn_rf)
    if overall == "PAST":
        msg = ("VERDICT: checkpoint predicts the PAST -> backward direction CONFIRMED "
               "(predicted increment aligns with t-dt, not t+dt).")
    elif overall == "FUTURE":
        msg = ("VERDICT: checkpoint predicts the FUTURE -> this is a FORWARD model. "
               "(Expected for the forward-control run; a RED FLAG for the backward checkpoint.)")
    else:
        msg = ("VERDICT: INCONCLUSIVE -> predicted increment is uncorrelated with both "
               "neighbors (persistence / blur). Single-step skill is near-trivial; cannot "
               "claim a direction from this test.")
    print(msg)
    print(sep)

    summary = {
        "config": args.config,
        "run_num": str(args.run_num),
        "checkpoint_select": args.checkpoint_select,
        "checkpoint_path": params.checkpoint_path,
        "step_hours": dataset.dhours * params.dt,
        "n_anchors": int(R_past.shape[0]),
        "anchor_indices": [int(a) for a in anchors],
        "date_first": used_dates[0],
        "date_last": used_dates[-1],
        "channel_names": channel_names,
        "dynamical_channels": dyn_set,
        "per_channel": {
            channel_names[c]: {
                "r_past_mean": float(mean_r_past[c]), "r_past_std": float(std_r_past[c]),
                "r_future_mean": float(mean_r_fut[c]), "r_future_std": float(std_r_fut[c]),
                "frac_r_past_gt_r_future": float(frac_past[c]),
                "rmse_past": float(np.nanmean(RMSE_past[:, c])),
                "rmse_future": float(np.nanmean(RMSE_fut[:, c])),
                "rmse_persist_past": float(np.nanmean(RMSE_pers_past[:, c])),
                "rmse_persist_future": float(np.nanmean(RMSE_pers_fut[:, c])),
                "mse_norm_past": float(np.nanmean(MSE_norm_past[:, c])),
                "mse_norm_future": float(np.nanmean(MSE_norm_fut[:, c])),
                "verdict": per_channel_verdict(mean_r_past[c], mean_r_fut[c]),
            }
            for c in range(C)
        },
        "dynamical_mean_r_past": dyn_rp,
        "dynamical_mean_r_future": dyn_rf,
        "all_mean_r_past": all_rp,
        "all_mean_r_future": all_rf,
        "overall_verdict": overall,
    }

    out_json = args.output_json or os.path.join(params.experiment_dir, "direction_verification.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote summary -> {out_json}")

    comm.cleanup()


def main():
    parser = argument_parser.get_default_argument_parser()
    parser.add_argument("--checkpoint_path", default=None, type=str,
                        help="Directory containing the training_checkpoints files. "
                             "Defaults to <exp_dir>/<config>/<run_num>/training_checkpoints.")
    parser.add_argument("--checkpoint_select", default="best", choices=["best", "latest"],
                        help="Load best_ckpt_mp0.tar (lowest val loss) or the latest legacy version.")
    parser.add_argument("--num_anchors", default=128, type=int,
                        help="Number of present-times to test, spread evenly across the valid set.")
    parser.add_argument("--output_json", default=None, type=str,
                        help="Where to write the JSON summary. Defaults to the experiment dir.")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
