#!/usr/bin/env python3
"""Test how closely the airfoil surface follows the true shape, with and without smoothing.

The true shapes of the airfoils in the database are unknown, so this test uses stand-ins
whose shape is known exactly. For each file it takes a NACA 4-digit airfoil with about the
same thickness and camber, and the same kind of trailing edge (closed or blunt), samples it
at the file's own x-stations and rounds it to the decimals the file carries
(smooth_airfoils.decimals). The stand-in's points then have the same spacing and rounding as
the file's. For each stand-in it compares three surfaces with the exact shape:
  points : the surface gmsh builds through the points as they are (what groundGen did so far)
  arc    : smooth_airfoils.smooth(), a quintic smoothing spline in arc length (the Tyrrell
           026 wing of the validation)
  sqrt   : smooth_airfoils.smooth_sqrt(), a quintic smoothing spline of y against
           xi = +-sqrt(x - x0), the smoothing used for the dataset
It reports the largest distance of each surface from the exact shape (and where), the
largest error in the inviscid Cp between 2% and 95% chord, and the largest wiggle in Cp
(panel.py). Files that smooth_airfoils leaves out are skipped.

--stand-in-decimals N rounds the stand-ins to N decimals at most, but still smooths them as
if they had the file's decimals: a file that is less precise than its digits say.

Runs on Windows with the project's Python (about 4 s per file and core):
    .venv\\Scripts\\python.exe ground_effect\\smoothing_test.py Deep-Flow-Prediction\\data\\airfoil_database --jobs 8 --csv results\\smoothing_test.csv
"""
import argparse
import csv
import functools
import multiprocessing
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from groundGen import load_airfoil, tidy_trailing_edge  # noqa: E402
from panel import cp_change, gmsh_curve, inviscid_cp, wiggle  # noqa: E402
from smooth_airfoils import curve_distance, decimals, file_list, read_points, smooth, smooth_airfoil, smooth_sqrt  # noqa: E402

METHODS = ["points", "arc", "sqrt"]
ALPHA = 3.45


def naca4(m, p, t, xc, closed):
    """Upper and lower surface points of a NACA 4-digit airfoil at the camber-line stations
    xc (max camber m at p, thickness t); closed or blunt (0.0021 t) trailing edge."""
    a4 = -0.1036 if closed else -0.1015
    yt = 5 * t * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc ** 2 + 0.2843 * xc ** 3 + a4 * xc ** 4)
    if m > 0:
        yc = np.where(xc < p, m / p ** 2 * (2 * p * xc - xc ** 2), m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * xc - xc ** 2))
        th = np.arctan(np.where(xc < p, 2 * m / p ** 2 * (p - xc), 2 * m / (1 - p) ** 2 * (p - xc)))
    else:
        yc = th = np.zeros_like(xc)
    return (np.column_stack([xc - yt * np.sin(th), yc + yt * np.cos(th)]),
            np.column_stack([xc + yt * np.sin(th), yc - yt * np.cos(th)]))


def contour(m, p, t, x_upper, x_lower, closed):
    """UIUC order: upper stations from the trailing edge to the leading edge, then lower."""
    return np.vstack([naca4(m, p, t, x_upper, closed)[0], naca4(m, p, t, x_lower, closed)[1]])


def shape_of(ar):
    """Thickness, camber and its position of the airfoil in a file (for the stand-in)."""
    le = int(np.argmin(ar[:, 0]))
    x = np.linspace(0.01, 0.99, 197)
    yu = np.interp(x, *np.array(sorted(map(tuple, ar[:le + 1]))).T)
    yl = np.interp(x, *np.array(sorted(map(tuple, ar[le:]))).T)
    camber = (yu + yl) / 2
    m = float(camber.max())
    return (0.0 if m < 0.002 else min(m, 0.09), float(np.clip(x[np.argmax(camber)], 0.15, 0.7)),
            float(np.clip((yu - yl).max(), 0.04, 0.30)))


def dedupe(ar):
    return ar[np.r_[True, np.hypot(*np.diff(ar, axis=0).T) > 1e-9]]


def as_read(ar):
    """Points written with six decimals and read back by groundGen (a repeated end point
    dropped)."""
    ar = np.round(ar, 6)
    return ar[:-1] if np.max(np.abs(ar[0] - ar[-1])) < 1e-6 else ar


def surface(ar):
    """Fine points along the surface groundGen meshes (see panel.cfd_surface)."""
    ar, blunt = tidy_trailing_edge(ar)
    c = gmsh_curve(ar, 20)
    if not blunt:
        c = np.vstack([c, ar[-1] + (ar[0] - ar[-1]) * np.linspace(0, 1, 50)[:, None]])
    return c


def test(path, stand_in_decimals=None):
    """One file: the stand-in, and the errors of each surface. Returns a dict (a CSV row)."""
    row = {"file": os.path.basename(path)}
    try:
        raw = read_points(path)
        digits = decimals(path)[1]
        why = smooth_airfoil(raw, digits)[3]  # the dataset leaves this file out
        if why:
            row["error"] = "left out: " + "; ".join(why)
            return row
        closed = np.hypot(*(raw[0] - raw[-1])) < 1e-4
        m, p, t = shape_of(load_airfoil(path))
        le = int(np.argmin(raw[:, 0]))
        xu, xl = np.clip(raw[:le + 1, 0], 0, 1), np.clip(raw[le + 1:, 0], 0, 1)
        if closed:  # the last point repeats the first one
            xu[0] = xl[-1] = 1.0
        rounded = digits if stand_in_decimals is None else min(digits, stand_in_decimals)
        pts = dedupe(np.round(contour(m, p, t, xu, xl, closed), rounded))
        row.update(points=len(raw), decimals=digits, stand_in_decimals=rounded,
                   m=round(m, 4), p=round(p, 3), t=round(t, 4), closed=int(closed))
        xd = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 20001)))
        exact = contour(m, p, t, xd[::-1], xd[1:], closed)
        xe = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 401)))
        cp_exact = inviscid_cp(as_read(contour(m, p, t, xe[::-1], xe[1:], closed)), ALPHA)
    except (OSError, ValueError, np.linalg.LinAlgError) as e:
        row["error"] = str(e)[:200]
        return row
    make = {"points": lambda: pts[:-1] if np.max(np.abs(pts[0] - pts[-1])) < 1e-6 else pts,
            "arc": lambda: as_read(smooth(pts, digits)[0]),
            "sqrt": lambda: as_read(smooth_sqrt(pts, digits)[0])}
    for k in METHODS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ar = make[k]()
            s = surface(ar)
            d = curve_distance(exact, s)
            i = int(np.argmax(d))
            cp = inviscid_cp(ar, ALPHA)
            row[f"geo_{k}"], row[f"x_geo_{k}"] = float(d[i]), float(s[i, 0])
            row[f"dcp_{k}"] = max(cp_change(cp[side], cp_exact[side])[0] for side in ("upper", "lower"))
            row[f"wiggle_{k}"] = max(wiggle(*cp[side])[0] for side in ("upper", "lower"))
        except (ValueError, np.linalg.LinAlgError) as e:
            row["error"] = (row.get("error", "") + f" {k}: {e}").strip()[:200]
    return row


COLUMNS = (["file", "points", "decimals", "stand_in_decimals", "m", "p", "t", "closed"]
           + [f"{a}_{k}" for k in METHODS for a in ("geo", "x_geo", "dcp", "wiggle")] + ["error"])


def summary(rows):
    def col(key):
        return np.array([float(r[key]) if r.get(key) not in (None, "") else np.nan for r in rows])

    out = [r for r in rows if r.get("error", "").startswith("left out")]
    rows = [r for r in rows if r not in out]
    print(f"{len(rows)} files tested ({len(out)} left out by smooth_airfoils); "
          f"{sum(1 for r in rows if r.get('error'))} with an error")
    for r in rows:
        if r.get("error"):
            print(f"  {r['file']}: {r['error']}")
    for metric, text, f in [("geo", "largest distance from the exact shape (chords)", "{:.1e}"),
                            ("dcp", "largest error in Cp, 2-95% chord", "{:.4f}"),
                            ("wiggle", "largest wiggle in Cp", "{:.4f}")]:
        print(f"{text}: median / 90% / 99% / largest")
        for k in METHODS:
            v = col(f"{metric}_{k}")
            v = v[np.isfinite(v)]
            if len(v):
                print(f"  {k:7s} " + " / ".join(f.format(q) for q in [*np.percentile(v, [50, 90, 99]), v.max()]))
    for a, b in [("sqrt", "points"), ("sqrt", "arc"), ("arc", "points")]:
        for metric in ("geo", "dcp"):
            x, y = col(f"{metric}_{a}"), col(f"{metric}_{b}")
            ok = np.isfinite(x) & np.isfinite(y)
            print(f"{a} closer than {b} ({metric}): {np.mean(x[ok] < y[ok]) * 100:.0f}% of the files; "
                  f"more than twice as far in {np.sum(x[ok] > 2 * y[ok])}")


COLORS = {"points": "#2a78d6", "arc": "#eb6834", "sqrt": "#1baf7a"}  # categorical slots 1-3
LABELS = {"points": "points as given",
          "arc": "arc length (Tyrrell)",
          "sqrt": "sqrt(x) (dataset)"}


def plot(rows, path, example):
    """Distributions of the surface and Cp errors over all files, and the nose of one
    stand-in."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, grid = "#333333", "#d9d9d9"
    plt.rcParams.update({"axes.edgecolor": ink, "axes.labelcolor": ink, "xtick.color": ink,
                         "ytick.color": ink, "text.color": ink, "font.size": 9})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    medians = {}
    for ax, metric, xlabel in [(axes[0], "geo", "largest distance from the exact shape (chords)"),
                               (axes[1], "dcp", "largest error in inviscid Cp, 2-95% chord")]:
        for k in METHODS:
            v = np.sort([float(r[f"{metric}_{k}"]) for r in rows if r.get(f"{metric}_{k}") not in (None, "")])
            ax.step(v, np.arange(1, len(v) + 1) / len(v) * 100, where="post", color=COLORS[k], lw=1.6)
            medians[k, metric] = np.median(v)
        # a few files beyond these limits (gmsh's spline and the arc-length spline go far off
        # on some) run off the right edge
        ax.set(xscale="log", xlabel=xlabel, ylabel="share of the files (%)", ylim=(0, 100),
               xlim=(1e-7, 0.1) if metric == "geo" else (1e-3, 10))
        ax.grid(color=grid, lw=0.6)
    # one legend for all three panels, below them
    handles = [plt.Line2D([], [], color=COLORS[k], lw=1.6) for k in METHODS]
    labels = [f"{LABELS[k]}: median {medians[k, 'geo']:.1e} chords, {medians[k, 'dcp']:.3f} in Cp"
              for k in METHODS]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    tested = sum(1 for r in rows if r.get("geo_sqrt") not in (None, ""))
    axes[0].set_title(f"Stand-ins at the stations of {tested} UIUC files: surface")
    axes[1].set_title("... and inviscid surface pressure")

    # the nose of the example's stand-in
    raw = read_points(example)
    digits = decimals(example)[1]
    closed = np.hypot(*(raw[0] - raw[-1])) < 1e-4
    m, p, t = shape_of(load_airfoil(example))
    le = int(np.argmin(raw[:, 0]))
    xu, xl = np.clip(raw[:le + 1, 0], 0, 1), np.clip(raw[le + 1:, 0], 0, 1)
    if closed:
        xu[0] = xl[-1] = 1.0
    pts = dedupe(np.round(contour(m, p, t, xu, xl, closed), digits))
    xd = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 20001)))
    ax = axes[2]
    ax.plot(*contour(m, p, t, xd[::-1], xd[1:], closed).T, color="#9a9a9a", lw=5, alpha=0.6,
            label="exact shape", solid_capstyle="round")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        curves = {"points": gmsh_curve(pts[:-1] if closed else pts, 40),
                  "arc": gmsh_curve(as_read(smooth(pts, digits)[0]), 10),
                  "sqrt": gmsh_curve(as_read(smooth_sqrt(pts, digits)[0]), 10)}
    for k in METHODS:
        ax.plot(*curves[k].T, color=COLORS[k], lw=1.4, ls={"points": "-", "arc": "--", "sqrt": "-"}[k])
    ax.plot(*pts.T, "o", color=ink, ms=4, label="points (stand-in)")
    ax.set(xlim=(-0.008, 0.032), ylim=(-0.022, 0.03), aspect="equal", xlabel="x/c", ylabel="y/c")
    ax.grid(color=grid, lw=0.6)
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    ax.set_title(f"Nose of the stand-in for {os.path.basename(example)}")
    fig.tight_layout(rect=(0, 0.07, 1, 1))  # room for the legend below
    fig.savefig(path, dpi=110)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="*", help="airfoil files, folders of .dat files, or wildcard patterns")
    ap.add_argument("--csv", help="write one row per file into this table")
    ap.add_argument("--jobs", type=int, default=1, help="files to test at the same time (default 1)")
    ap.add_argument("--stand-in-decimals", type=int, help="round the stand-ins to at most this many decimals")
    ap.add_argument("--plot", help="draw the distributions of the errors and an example nose into this image")
    ap.add_argument("--example", help="airfoil file whose stand-in nose is drawn (default: the first file)")
    ap.add_argument("--from-csv", help="do not test: summarize (and plot) the table of an earlier run")
    args = ap.parse_args()
    files = file_list(args.files)
    example = args.example or (files[0] if files else None)
    if args.plot and example is None:
        raise SystemExit("--plot needs an airfoil file for the example nose (--example)")
    if args.from_csv:
        with open(args.from_csv) as f:
            rows = list(csv.DictReader(f))
        summary(rows)
        if args.plot:
            plot(rows, args.plot, example)
        return
    if not files:
        raise SystemExit("no airfoil files given")
    work = functools.partial(test, stand_in_decimals=args.stand_in_decimals)
    if args.jobs > 1:
        for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = "1"  # one thread in each process: the processes share the cores
        with multiprocessing.Pool(args.jobs) as pool:
            rows = []
            for k, r in enumerate(pool.imap(work, files, chunksize=2), 1):
                rows.append(r)
                if k % 100 == 0 or k == len(files):
                    print(f"tested {k} of {len(files)} files", flush=True)
    else:
        rows = [work(f) for f in files]
    summary(rows)
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLUMNS)
            for r in rows:
                w.writerow([(f"{r[c]:.3e}" if c.startswith("geo") else f"{r[c]:.4f}")
                            if isinstance(r.get(c), float) else r.get(c, "") for c in COLUMNS])
        print("wrote", args.csv)
    if args.plot:
        plot(rows, args.plot, example)


if __name__ == "__main__":
    main()
