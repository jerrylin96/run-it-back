#!/usr/bin/env python3
# ============================================================================
"""Root-cause diagnostic for a validation-loss spike, parsed from a makani out.log.

Why this exists: a run's val loss can descend smoothly to a plateau and then jump ~10x in a
single epoch and stay there. plot_loss_from_log.py shows the spike but not WHY. This
overlays, on a shared epoch axis, the three signals that discriminate the leading
hypotheses -- all recoverable from out.log alone:

  * gradient norm (makani logs "gradient norm: <v>" every epoch when grad clipping is on)
    -> a spike here => an optimizer / bad-batch / bf16 grad event that clipping didn't contain.
  * RESUME BOUNDARIES (each resumed job logs "Resuming from checkpoint" before its first
    epoch) drawn as vertical lines -> if the loss spike sits exactly on one, the culprit is
    the resume (optimizer / scheduler / sampler state reload), NOT the model.
  * train vs val loss -> a train-loss spike too => genuine optimization blow-up; val-only =>
    an eval/normalization/data artifact.

It also prints a headless text verdict: the largest val jump, its ratio to the running best,
whether that epoch is a resume boundary, and the gradient norm there vs the run's median --
so it is useful over ssh with no display. The LEARNING RATE is the one signal makani logs
only to wandb (not out.log); pass it via --lr_csv (epoch,lr) exported from wandb to overlay it.

NB: this reads the SAME "Epoch N summary:" blocks as plot_loss_from_log.py and imports its
regexes so the two parsers can't drift.

Usage
-----
  # one run, resume chain concatenated in order (later files win on a duplicate epoch):
  python plot_training_diagnostics.py <runs>/<target>/<run>/out.log --out <runs>/<target>/<run>/diagnostics.png

  # headless verdict only (no matplotlib needed):
  python plot_training_diagnostics.py .../out.log --csv

Run inside the container (matplotlib is baked in), e.g. via the crun wrapper:
  crun python /opt/makani/analysis/plot_training_diagnostics.py .../out.log --out .../diagnostics.png
"""

import argparse
import math
import os
import re
import sys

# reuse the exact epoch/loss regexes so this parser can't drift from plot_loss_from_log.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_loss_from_log import EPOCH_RE, TRAIN_RE, VAL_RE  # noqa: E402

# makani logs (deterministic_trainer.py log_epoch): "gradient norm: <pad><v>" when grad
# clipping is on. The resume line is logged at trainer init before the resumed job's first
# epoch. The stop lines are what the early-stopping / divergence guard emit on exit.
GRAD_RE = re.compile(r"gradient norm:\s+([-+0-9.eE]+|nan)", re.IGNORECASE)
RESUME_RE = re.compile(r"Resuming from checkpoint")
EARLYSTOP_RE = re.compile(r"Early stopping at epoch\s+(\d+)")
DIVSTOP_RE = re.compile(r"Divergence stop at epoch\s+(\d+)")


def parse_logs(paths):
    """Parse one run's log file(s) -> (data, resume_epochs, stops).

    data: {epoch: {"train":v|None, "val":v|None, "grad":v|None}}
    resume_epochs: set of epochs that are the FIRST epoch of a resumed job (a resume boundary)
    stops: list of (epoch, kind) for early-stopping / divergence-stop lines found in the log
    """
    data = {}
    resume_epochs = set()
    stops = []
    cur = None
    pending_resume = False
    for path in paths:
        with open(path, "r") as f:
            for line in f:
                if RESUME_RE.search(line):
                    # the NEXT "Epoch N summary:" is the resumed job's first computed epoch
                    pending_resume = True
                    continue
                m = EPOCH_RE.search(line)
                if m:
                    cur = int(m.group(1))
                    data.setdefault(cur, {"train": None, "val": None, "grad": None})
                    if pending_resume:
                        resume_epochs.add(cur)
                        pending_resume = False
                    continue
                ms = EARLYSTOP_RE.search(line)
                if ms:
                    stops.append((int(ms.group(1)), "early_stopping"))
                    continue
                md = DIVSTOP_RE.search(line)
                if md:
                    stops.append((int(md.group(1)), "divergence"))
                    continue
                if cur is None:
                    continue
                mt = TRAIN_RE.search(line)
                if mt:
                    data[cur]["train"] = float(mt.group(1))
                    continue
                mg = GRAD_RE.search(line)
                if mg:
                    data[cur]["grad"] = float(mg.group(1))
                    continue
                mv = VAL_RE.search(line)
                if mv:
                    data[cur]["val"] = float(mv.group(1))
    return data, resume_epochs, stops


def _series(data, key):
    """Sorted (epochs, values) for one key, dropping epochs with no value for it."""
    epochs = sorted(e for e in data if data[e][key] is not None)
    return epochs, [data[e][key] for e in epochs]


def _median(xs):
    """Median of the FINITE values in xs. Drops nan/inf first: makani's GRAD_RE/VAL_RE
    capture 'nan' (a divergence blow-up writes 'gradient norm: nan'), and Python's sort is
    undefined once a nan is in the list (all nan comparisons are False), which would corrupt
    the median. Returns None if nothing finite remains (all values were nan/inf)."""
    xs = sorted(x for x in xs if math.isfinite(x))
    n = len(xs)
    if n == 0:
        return None
    mid = n // 2
    return xs[mid] if n % 2 else 0.5 * (xs[mid - 1] + xs[mid])


def find_spike(data):
    """Anchor epoch where the val loss first broke -> a dict, or None.

    Considers two anomaly kinds and returns the EARLIER one (things break once): (a) the
    largest finite ratio jump ``val[e] / best(val[:e])`` above the running best, and (b) the
    first non-finite (nan/inf) val -- a full divergence, given ``ratio=inf`` so the caller can
    tell it apart. None if <2 epochs or nothing anomalous."""
    epochs, vals = _series(data, "val")
    if len(vals) < 2:
        return None
    best = None            # finite running minimum
    worst = None           # largest finite ratio spike
    first_nonfinite = None  # first nan/inf val
    for e, v in zip(epochs, vals):
        if not math.isfinite(v):
            if first_nonfinite is None:
                bb = best if best is not None else float("nan")
                first_nonfinite = {"epoch": e, "val": v, "ratio": float("inf"), "best_before": bb}
            continue
        if best is not None and best > 0:
            ratio = v / best
            if ratio > (worst["ratio"] if worst else 1.0):
                worst = {"epoch": e, "val": v, "ratio": ratio, "best_before": best}
        best = v if best is None else min(best, v)
    candidates = [c for c in (worst, first_nonfinite) if c is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda c: c["epoch"])


def load_lr_csv(path):
    """Optional (epoch,lr) CSV exported from wandb -> {epoch: lr}. First two columns used."""
    lr = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                lr[int(float(parts[0]))] = float(parts[1])
            except ValueError:
                continue  # header / blank
    return lr


def print_verdict(data, resume_epochs, stops, lr):
    e_val, _ = _series(data, "val")
    e_grad, grads = _series(data, "grad")
    n = len(e_val)
    print(f"[diagnostics] {n} epochs with a validation loss (through epoch {e_val[-1] if n else '-'})", file=sys.stderr)
    if resume_epochs:
        print(f"[diagnostics] resume boundaries at epochs: {sorted(resume_epochs)}", file=sys.stderr)
    else:
        print("[diagnostics] no 'Resuming from checkpoint' lines found (single job, or logs not concatenated)", file=sys.stderr)
    for ep, kind in stops:
        print(f"[diagnostics] log records a {kind.upper()} stop at epoch {ep}", file=sys.stderr)

    # a non-finite validation loss is a full numerical divergence, not just a finite ratio-spike;
    # find_spike (below) can't rank a nan ratio, so surface it explicitly.
    nonfinite_val = [ep for ep in e_val if not math.isfinite(data[ep]["val"])]
    if nonfinite_val:
        more = " ..." if len(nonfinite_val) > 5 else ""
        print(f"[diagnostics] validation loss is NaN/inf at epochs {nonfinite_val[:5]}{more} "
              f"-> a full numerical divergence (not just a finite spike)", file=sys.stderr)

    spike = find_spike(data)
    if spike is None:
        print("[diagnostics] no val-loss spike detected (need >=2 epochs).", file=sys.stderr)
        return
    e = spike["epoch"]
    on_resume = e in resume_epochs
    gmed = _median(grads) if grads else None
    ghere = data[e]["grad"]
    gprev = data[e - 1]["grad"] if (e - 1) in data else None
    if math.isfinite(spike["ratio"]):
        print(f"\n[diagnostics] LARGEST val jump: epoch {e}, val={spike['val']:.4e} "
              f"= {spike['ratio']:.2f}x the running best ({spike['best_before']:.4e})", file=sys.stderr)
    else:
        print(f"\n[diagnostics] DIVERGENCE anchor: epoch {e}, val is non-finite (nan/inf); "
              f"running best before was {spike['best_before']:.4e}", file=sys.stderr)
    print(f"[diagnostics]   on a resume boundary? {'YES -> suspect resume-state reload' if on_resume else 'no'}", file=sys.stderr)
    if grads:
        gh = f"{ghere:.3e}" if ghere is not None else "n/a"
        gp = f"{gprev:.3e}" if gprev is not None else "n/a"
        gm = f"{gmed:.3e}" if gmed is not None else "n/a (all grad norms non-finite)"
        print(f"[diagnostics]   gradient norm: this epoch={gh}, prev epoch={gp}, run median={gm}", file=sys.stderr)

        def _fin(x):
            return x is not None and math.isfinite(x)

        nonfinite = (ghere is not None and not math.isfinite(ghere)) or (gprev is not None and not math.isfinite(gprev))
        if nonfinite:
            # a NaN/inf grad norm IS the smoking gun — and note grad clipping caps a norm, it
            # cannot rescue a non-finite gradient, so tightening the clip won't fix this.
            print("[diagnostics]   -> grad norm is NaN/inf at the spike -> a numerical blow-up (bf16 overflow / bad batch); the clip can't contain a non-finite grad", file=sys.stderr)
        elif gmed is not None and gmed > 0 and ((_fin(ghere) and ghere > 5 * gmed) or (_fin(gprev) and gprev > 5 * gmed)):
            print("[diagnostics]   -> grad norm is >5x the median here -> suspect an optimizer / bad-batch / bf16 grad event", file=sys.stderr)
        else:
            print("[diagnostics]   -> grad norm is NOT anomalous here -> clipping is likely not the culprit", file=sys.stderr)
    else:
        print("[diagnostics]   no gradient-norm data (grad clipping off?) — can't test the grad-spike hypothesis", file=sys.stderr)
    if lr and e in lr:
        print(f"[diagnostics]   learning rate at epoch {e}: {lr[e]:.3e}", file=sys.stderr)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+", help="one run's out.log(s); comma/space-join a resume chain in order")
    p.add_argument("--out", type=str, default=None, help="write a PNG here (needs matplotlib)")
    p.add_argument("--csv", action="store_true", help="dump parsed epoch,train,val,grad,resume as CSV to stdout")
    p.add_argument("--lr_csv", type=str, default=None, help="optional (epoch,lr) CSV exported from wandb to overlay LR")
    p.add_argument("--logy_loss", action="store_true", help="log-scale the loss axis (grad norm is always log-scale)")
    args = p.parse_args(argv)

    # allow "a.log,b.log" as well as separate args
    files = []
    for a in args.logs:
        files.extend(a.split(","))
    for fp in files:
        if not os.path.isfile(fp):
            raise SystemExit(f"log file not found: {fp}")

    data, resume_epochs, stops = parse_logs(files)
    if not data:
        raise SystemExit("no 'Epoch N summary:' blocks found — check the log path/format")
    lr = load_lr_csv(args.lr_csv) if args.lr_csv else {}

    print_verdict(data, resume_epochs, stops, lr)

    if args.csv:
        print("epoch,train_loss,val_loss,grad_norm,resume_boundary")
        for e in sorted(data):
            d = data[e]
            print(f"{e},{'' if d['train'] is None else d['train']},{'' if d['val'] is None else d['val']},"
                  f"{'' if d['grad'] is None else d['grad']},{1 if e in resume_epochs else 0}")
        return

    if args.out is None:
        print("\nNothing plotted. Pass --out <file.png> to write the figure, or --csv to dump values.", file=sys.stderr)
        return

    import matplotlib

    matplotlib.use("Agg")  # headless (cluster / container)
    import matplotlib.pyplot as plt

    nrows = 3 if lr else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(9, 3.0 * nrows), sharex=True)
    ax_loss, ax_grad = axes[0], axes[1]

    def draw_boundaries(ax):
        for i, e in enumerate(sorted(resume_epochs)):
            ax.axvline(e, color="0.6", ls="--", lw=1, label="resume boundary" if i == 0 else None)
        for i, (e, kind) in enumerate(stops):
            ax.axvline(e, color="tab:red" if kind == "divergence" else "tab:orange", ls=":", lw=1.5,
                       label=(kind.replace("_", " ")) if i == 0 else None)

    # panel 1: losses
    et, tr = _series(data, "train")
    ev, va = _series(data, "val")
    if et:
        ax_loss.plot(et, tr, "-", color="tab:blue", label="train loss")
    if ev:
        ax_loss.plot(ev, va, "--", color="tab:green", label="val loss")
    draw_boundaries(ax_loss)
    spike = find_spike(data)
    if spike is not None:
        ax_loss.plot([spike["epoch"]], [spike["val"]], "v", color="tab:red", ms=9,
                     label=f"largest val jump ({spike['ratio']:.1f}x)")
    if args.logy_loss:
        ax_loss.set_yscale("log")
    ax_loss.set_ylabel("loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(fontsize=8)
    ax_loss.set_title("training diagnostics (parsed from out.log)")

    # panel 2: gradient norm
    eg, gr = _series(data, "grad")
    if eg:
        ax_grad.plot(eg, gr, "-", color="tab:purple", label="gradient norm")
        ax_grad.set_yscale("log")
    else:
        ax_grad.text(0.5, 0.5, "no gradient-norm data (grad clipping off?)",
                     ha="center", va="center", transform=ax_grad.transAxes)
    draw_boundaries(ax_grad)
    ax_grad.set_ylabel("grad norm")
    ax_grad.grid(True, alpha=0.3)
    ax_grad.legend(fontsize=8)

    # optional panel 3: learning rate
    if lr:
        ax_lr = axes[2]
        elr = sorted(lr)
        ax_lr.plot(elr, [lr[e] for e in elr], "-", color="tab:brown", label="learning rate")
        ax_lr.set_yscale("log")
        draw_boundaries(ax_lr)
        ax_lr.set_ylabel("lr")
        ax_lr.grid(True, alpha=0.3)
        ax_lr.legend(fontsize=8)

    axes[-1].set_xlabel("epoch (= full pass over train/)")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
