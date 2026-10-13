#!/usr/bin/env python3
# ============================================================================
"""Plot training / validation loss curves parsed from makani job logs / out.log.

Why this exists: a run trained with wandb OFF has as its ONLY loss record the per-epoch
"Epoch N summary:" block makani writes to out.log (and to the SLURM job log). This parses
those blocks and plots train + val loss vs epoch, so wandb-less runs are viewable, and runs
with wandb enabled (offline) get a log-derived cross-check against the wandb curve.

It parses the exact strings the trainers log -- "Epoch N summary:", "training loss: <v>",
"validation loss: <v>" -- which are unambiguous against the "training steps:" /
"validation steps:" lines in the same block. Robust to the INFO:root: prefix and the
alignment padding. No makani/torch import; needs only the stdlib to parse, and matplotlib
only for --out (use --csv to skip matplotlib entirely).

Usage
-----
Each RUN arg is either  <logfile>  or  <label>=<logfile>[,<logfile>...]
(comma-join a resume chain's per-job logs, in order; later files win on a duplicate epoch).

  python plot_loss_from_log.py \
      smoke=$MAKANI_DATA/runs/rewind_fcn3_ens_smoke/ens_smoke1/out.log \
      10yr=$MAKANI_DATA/runs/rewind_fcn3_ens_10yr_backward/fcn3_ens_10yr/out.log \
      --out $MAKANI_DATA/runs/loss.png

  # no matplotlib handy? dump the parsed values as CSV instead:
  python plot_loss_from_log.py 10yr=.../out.log --csv

Run it inside the container (matplotlib is baked in), e.g. via the crun wrapper:
  crun python /opt/makani/analysis/plot_loss_from_log.py 10yr=.../out.log --out .../loss.png
"""

import argparse
import os
import re
import sys

# makani logs (deterministic_trainer.py): a header "Epoch <N> summary:", then a Metrics
# block with "training loss: <pad><v>" and "validation loss: <pad><v>". Match those exactly;
# the "... steps:" lines in between must NOT match, hence "loss:" is part of the pattern.
EPOCH_RE = re.compile(r"Epoch\s+(\d+)\s+summary:")
TRAIN_RE = re.compile(r"training loss:\s+([-+0-9.eE]+|nan)", re.IGNORECASE)
VAL_RE = re.compile(r"validation loss:\s+([-+0-9.eE]+|nan)", re.IGNORECASE)


def parse_logs(paths):
    """Parse one run's log file(s) -> {epoch: [train, val]} (either may stay None).

    An 'Epoch N summary:' line sets the current epoch; the following 'training loss:' /
    'validation loss:' lines fill it. Files are read in order so a resume chain's later
    job log overrides an earlier one on a duplicated epoch (there normally are none)."""
    data = {}
    cur = None
    for path in paths:
        with open(path, "r") as f:
            for line in f:
                m = EPOCH_RE.search(line)
                if m:
                    cur = int(m.group(1))
                    data.setdefault(cur, [None, None])
                    continue
                if cur is None:
                    continue
                mt = TRAIN_RE.search(line)
                if mt:
                    data[cur][0] = float(mt.group(1))
                    continue
                mv = VAL_RE.search(line)
                if mv:
                    data[cur][1] = float(mv.group(1))
    return data


def _series(data):
    """Sorted (epochs, train, val) lists with epochs that have at least one value."""
    epochs = sorted(e for e in data if data[e][0] is not None or data[e][1] is not None)
    return epochs, [data[e][0] for e in epochs], [data[e][1] for e in epochs]


def _parse_run_arg(arg):
    """'label=f1,f2' or 'f' -> (label, [files]). Default label = the run dir name."""
    if "=" in arg:
        label, files = arg.split("=", 1)
        files = files.split(",")
    else:
        files = [arg]
        # label from the exp-dir layout runs/<config>/<run_num>/out.log -> "<run_num>"
        parent = os.path.basename(os.path.dirname(os.path.abspath(arg)))
        label = parent if parent not in ("", ".") else os.path.basename(arg)
    for fp in files:
        if not os.path.isfile(fp):
            raise SystemExit(f"log file not found: {fp}")
    return label, files


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="each: <logfile> or <label>=<logfile>[,<logfile>...]")
    p.add_argument("--out", type=str, default=None, help="write a PNG here (needs matplotlib)")
    p.add_argument("--csv", action="store_true", help="print parsed epoch,train,val as CSV to stdout (no matplotlib)")
    p.add_argument("--logy", action="store_true", help="log-scale the loss axis")
    args = p.parse_args(argv)

    runs = [_parse_run_arg(a) for a in args.runs]
    parsed = [(label, parse_logs(files)) for label, files in runs]

    # always print a compact summary so the tool is useful headless
    for label, data in parsed:
        epochs, tr, va = _series(data)
        if not epochs:
            print(f"[{label}] no 'Epoch N summary:' blocks found — check the log path/format", file=sys.stderr)
            continue
        last = epochs[-1]
        print(f"[{label}] {len(epochs)} epochs parsed (through epoch {last}); " f"last train={data[last][0]} val={data[last][1]}", file=sys.stderr)

    if args.csv:
        print("run,epoch,train_loss,val_loss")
        for label, data in parsed:
            epochs, tr, va = _series(data)
            for e, t, v in zip(epochs, tr, va):
                print(f"{label},{e},{'' if t is None else t},{'' if v is None else v}")
        return

    if args.out is None:
        print("\nNothing plotted. Pass --out <file.png> to write a figure, or --csv to dump values.", file=sys.stderr)
        return

    import matplotlib

    matplotlib.use("Agg")  # headless (cluster / container)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")
    for i, (label, data) in enumerate(parsed):
        epochs, tr, va = _series(data)
        if not epochs:
            continue
        c = cmap(i % 10)
        # drop None points per series so a partial epoch doesn't break the line
        et = [(e, t) for e, t in zip(epochs, tr) if t is not None]
        ev = [(e, v) for e, v in zip(epochs, va) if v is not None]
        if et:
            ax.plot([e for e, _ in et], [t for _, t in et], "-", color=c, label=f"{label} train")
        if ev:
            ax.plot([e for e, _ in ev], [v for _, v in ev], "--", color=c, label=f"{label} val")
    ax.set_xlabel("epoch (= full pass over train/)")
    ax.set_ylabel("loss")
    if args.logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("makani loss curves (parsed from logs)")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
