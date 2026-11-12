"""The learning curve: the U-Net's error on airfoils it has not seen against the number of
training samples, trained from random weights ('scratch') and from the free-air model
('pretrained': transfer learning).

For every fraction of the training samples (--fractions) and every seed (--seeds), one model is
trained from random weights and one from --pretrained, with the settings of train_unet.py; and
the free-air model is tested as it is ('zero-shot'). Each run has its own folder in --out (for
example f0.25_pretrained_s0/, see train_unet.py for what is in it). Runs that already have a
summary.json are not repeated, so after an interruption the same command goes on where it
stopped. At the end it writes <out>/learning_curve.csv (one line per run) and prints the table;
--plot also draws the figure.

On a GPU, from the ground_effect folder (see the README for Kaggle):
    python learning_curve.py --data <train folder> --test <test folder> --pretrained <long80k_modelG> --out <results folder>
Only the table and the figure, from finished runs (for example downloaded from Kaggle):
    ..\\.venv\\Scripts\\python.exe learning_curve.py --out data\\unet\\learning_curve --report-only --plot ..\\results\\learning_curve.png
"""
import argparse
import glob
import json
import os

import numpy as np

STARTS = ("scratch", "pretrained")
COLORS = {"scratch": "#2a78d6", "pretrained": "#eb6834"}  # categorical slots 1 and 2, as in the other reports
NAMES = {"scratch": "from random weights", "pretrained": "from the free-air model"}
COLUMNS = ["run", "start", "fraction", "n_train", "seed", "iterations", "train_minutes", "val_p", "val_v", "val_all",
           "test_p", "test_v", "test_all", "best_iteration", "best_val_all", "best_test_all"]


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", help="folder with the training samples")
    ap.add_argument("--test", help="folder with the test samples (airfoils not in --data)")
    ap.add_argument("--pretrained", help="the free-air model (the reproduction's long80k_modelG)")
    ap.add_argument("--out", required=True, help="folder for the runs and the table")
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.125, 0.25, 0.5, 1.0],
                    help="parts of the training samples (default 0.125 0.25 0.5 1)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1], help="random seeds (default 0 1)")
    ap.add_argument("--iterations", type=int, default=80000, help="training iterations of each run (default 80000)")
    ap.add_argument("--report-only", action="store_true", help="no training: only the table and the figure")
    ap.add_argument("--plot", help="draw the learning curve into this image")
    ap.add_argument("--lr", type=float, default=0.0006)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--val-percent", type=float, default=10)
    ap.add_argument("--exclude-clipped", action="store_true")
    ap.add_argument("--p-offset", choices=("fluid", "grid"), default="fluid")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int)
    args = ap.parse_args(argv)
    if not args.report_only and not args.data:
        ap.error("--data is needed to train (or use --report-only)")
    return args


def runs(args):
    """(folder name, train_unet.py arguments) of each run, the most useful first: the zero-shot
    test, then every fraction with the first seed, then with the next seed."""
    common = ["--data", args.data, "--lr", str(args.lr), "--batch", str(args.batch),
              "--val-percent", str(args.val_percent), "--p-offset", args.p_offset,
              "--eval-every", str(args.eval_every), "--device", args.device]
    common += ["--test", args.test] if args.test else []
    common += ["--threads", str(args.threads)] if args.threads else []
    common += ["--exclude-clipped"] if args.exclude_clipped else []
    todo = [("zero_shot", common + ["--pretrained", args.pretrained, "--iterations", "0"])] if args.pretrained else []
    for seed in args.seeds:
        for fraction in args.fractions:
            for start in STARTS:
                if start == "pretrained" and not args.pretrained:
                    continue
                todo.append((f"f{fraction:g}_{start}_s{seed}", common + ["--iterations", str(args.iterations),
                             "--fraction", str(fraction), "--seed", str(seed)] +
                             (["--pretrained", args.pretrained] if start == "pretrained" else [])))
    return todo


def train_all(args):
    import train_unet  # needs torch, which --report-only does not
    import unet_data
    data = test = None
    for name, argv in runs(args):
        out = os.path.join(args.out, name)
        if os.path.exists(os.path.join(out, "summary.json")):
            print(f"{name}: done before, skipped")
            continue
        if data is None:  # load the samples once, for all runs
            data = unet_data.load(args.data, args.p_offset, args.exclude_clipped)
            test = unet_data.load(args.test, args.p_offset, args.exclude_clipped) if args.test else None
        print(f"\n===== {name} =====", flush=True)
        train_unet.run(train_unet.parse(argv + ["--out", out]), data, test)


def collect(out):
    """One row per finished run in out."""
    rows = []
    for path in sorted(glob.glob(os.path.join(out, "*", "summary.json"))):
        with open(path) as f:
            s = json.load(f)
        st = s["settings"]
        start = "scratch" if not st["pretrained"] else "zero-shot" if st["iterations"] == 0 else "pretrained"
        best = s.get("best_val", {})
        row = {"run": os.path.basename(os.path.dirname(path)), "start": start, "fraction": st["fraction"],
               "n_train": 0 if start == "zero-shot" else s["n_train"], "seed": st["seed"],
               "iterations": st["iterations"], "train_minutes": s["train_seconds"] / 60,
               "best_iteration": best.get("iteration", np.nan),
               "best_val_all": best.get("val", {}).get("all", np.nan),
               "best_test_all": best.get("test", {}).get("all", np.nan)}
        for name in ("val", "test"):
            for k in ("p", "v", "all"):
                row[f"{name}_{k}"] = s[name][k] if name in s else np.nan
        rows.append(row)
    return rows


def write_csv(rows, path):
    with open(path, "w") as f:
        f.write(",".join(COLUMNS) + "\n")
        for r in rows:
            f.write(",".join(f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c]) for c in COLUMNS) + "\n")
    print("saved", path)


def groups(rows, which):
    """{start: [(n_train, [errors of the seeds]), ...]} for the error column which."""
    out = {}
    for start in STARTS:
        sizes = sorted({r["n_train"] for r in rows if r["start"] == start})
        out[start] = [(n, [r[which] for r in rows if r["start"] == start and r["n_train"] == n]) for n in sizes]
    return out


def table(rows, which):
    print(f"\nError on the {which} samples (%, mean over the seeds; the seeds' values in brackets):")
    print(f"  {'start':10s} {'samples':>7s} {'seeds':>5s} {'p':>8s} {'v':>7s} {'all':>7s}   all per seed")
    for r in rows:
        if r["start"] == "zero-shot":
            print(f"  {'zero-shot':10s} {0:7d} {'-':>5s} {100 * r[which + '_p']:8.2f} {100 * r[which + '_v']:7.2f} "
                  f"{100 * r[which + '_all']:7.2f}")
    for start in STARTS:
        for n, _ in groups(rows, f"{which}_all")[start]:
            sel = [r for r in rows if r["start"] == start and r["n_train"] == n]
            mean = lambda k: 100 * np.mean([r[f"{which}_{k}"] for r in sel])
            print(f"  {start:10s} {n:7d} {len(sel):5d} {mean('p'):8.2f} {mean('v'):7.2f} {mean('all'):7.2f}   "
                  + " ".join(f"{100 * r[which + '_all']:.2f}" for r in sel))


def plot(rows, which, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator

    plain = FuncFormatter(lambda v, _: f"{v:g}")
    sizes = sorted({r["n_train"] for r in rows if r["start"] != "zero-shot"})
    ink, grid = "#333333", "#d9d9d9"
    plt.rcParams.update({"axes.edgecolor": ink, "axes.labelcolor": ink, "xtick.color": ink,
                         "ytick.color": ink, "text.color": ink, "font.size": 9})
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    zero = [r for r in rows if r["start"] == "zero-shot"]
    for ax, k, title in zip(axes, ("all", "p", "v"), ("all three fields", "pressure", "velocity")):
        for start in STARTS:
            g = groups(rows, f"{which}_{k}")[start]
            if not g:
                continue
            n = np.array([x for x, _ in g])
            for x, errs in g:
                ax.plot([x] * len(errs), 100 * np.array(errs), "o", ms=3.5, color=COLORS[start], alpha=0.5, mec="none")
            ax.plot(n, [100 * np.mean(e) for _, e in g], "-o", lw=2, ms=6, color=COLORS[start],
                    mec="white", mew=1.2, label=NAMES[start])
        if zero:
            ax.axhline(100 * zero[0][f"{which}_{k}"], color=ink, lw=1, ls="--", label="free-air model, not trained on these")
        ax.set(xscale="log", yscale="log", xlabel="training samples", title=title)
        ax.xaxis.set_major_locator(FixedLocator(sizes))  # a tick at each number of samples
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))  # ... 1, 2, 5, 10, 20, 50 ... %
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_formatter(plain)
        ax.yaxis.set_major_formatter(plain)
        ax.grid(color=grid, lw=0.6)
    axes[0].set_ylabel(f"error on the {which} samples (%)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("U-Net error in ground effect against the amount of training data", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print("saved", path)


def main(argv=None):
    args = parse(argv)
    if not args.report_only:
        train_all(args)
    rows = collect(args.out)
    if not rows:
        raise SystemExit(f"no finished runs in {args.out}")
    write_csv(rows, os.path.join(args.out, "learning_curve.csv"))
    which = "test" if all(np.isfinite(r["test_all"]) for r in rows) else "val"
    table(rows, which)
    if which == "test":
        table(rows, "val")
    if args.plot:
        plot(rows, which, args.plot)


if __name__ == "__main__":
    main()
