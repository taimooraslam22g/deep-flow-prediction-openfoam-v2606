#!/usr/bin/env python3
"""Check airfoil files for kinks in the surface that gmsh builds from them.

gmsh joins the points of an airfoil with its built-in 'Spline', a Catmull-Rom curve. Where
the spacing of the points changes abruptly, that curve bends sharply between them. The kinks
are tiny in shape (hundredths of a millimetre on a wind tunnel model), but the surface
pressure reacts to the curvature, so they show up as spikes in Cp.

This script builds the surface as groundGen.py meshes it (the same curve through the points
of each file, closed by a straight line at the trailing edge) and computes the inviscid
surface pressure on it with a panel method (panel.py: constant-strength sources and vortex,
Hess & Smith 1967). For each surface it reports, between 2% and 95% chord:
  - the largest wiggle: how far Cp departs from a smooth local fit (a parabola over +-0.01
    chords). Kinks give large values; a smooth shape gives small ones, but not zero close to
    the leading edge, where Cp itself changes too fast for the fit;
  - the largest change in Cp when the file is smoothed first (smooth_airfoils.py), and the
    wiggle left after smoothing.
It also reports how far the surface moves when smoothed, how far the smoothed curve stays
from the points of the file (in half units of the last decimal they carry), the largest jump
in the spacing of the points (the cause of the kinks), repeated points, and the reasons to
leave a file out of the dataset (smooth_airfoils.problems: the chord does not run from 0 to
1, a surface is not a function of x, or the contour crosses itself). Files that are left out
are not smoothed.

Usage (on Windows, with the project's Python; folders and wildcards are expanded here,
because the Windows shells pass them on as they are):
    .venv\\Scripts\\python.exe ground_effect\\shape_check.py <airfoil.dat> [...] [--alpha 3.45] [--plot out.png]
    .venv\\Scripts\\python.exe ground_effect\\shape_check.py Deep-Flow-Prediction\\data\\airfoil_database --jobs 8 --csv results\\airfoil_shape_scan.csv
"""
import argparse
import csv
import functools
import multiprocessing
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from groundGen import load_airfoil, tidy_trailing_edge  # noqa: E402
from panel import cp_change, gmsh_curve, inviscid_cp, wiggle  # noqa: E402
from smooth_airfoils import (MAX_WIGGLE, as_written, curve_distance, decimals, file_list,  # noqa: E402
                             read_points, smooth_airfoil)

COLUMNS = ["file", "points", "written_decimals", "decimals", "spacing_jump", "x_jump",
           "wiggle_upper", "x_upper", "wiggle_lower", "x_lower",
           "dcp_upper", "x_dcp_upper", "dcp_lower", "x_dcp_lower",
           "smooth_wiggle_upper", "smooth_wiggle_lower",
           "shape_change", "x_shape_change", "fit_max", "fit_rms", "problems"]


def spacing_jump(ar):
    """Largest ratio between the lengths of two neighbouring segments of the open contour
    (as gmsh's spline sees it), and the x/c of the point between them."""
    seg = np.hypot(*np.diff(ar, axis=0).T)
    keep = seg > 1e-9  # repeated points are reported separately
    seg, inner = seg[keep], ar[1:][keep][:-1]
    ratio = np.maximum(seg[1:] / seg[:-1], seg[:-1] / seg[1:])
    i = int(np.argmax(ratio))
    return float(ratio[i]), float(inner[i, 0])


def check(path, alpha, keep_cp=False):
    """Everything this script measures for one airfoil file: a dict with the COLUMNS, and
    the inviscid Cp of both surfaces, before and after smoothing, if keep_cp (else None)."""
    row = {"file": os.path.basename(path)}
    try:
        ar = load_airfoil(path)
        if ar.ndim != 2 or ar.shape[1] != 2 or len(ar) < 7:
            raise ValueError(f"expected rows of two numbers, found an array of shape {ar.shape}")
    except (OSError, ValueError) as e:
        row["problems"] = f"cannot read the file: {e}"
        return row, None
    row["points"] = len(ar)
    notes = []
    repeated = int((np.hypot(*np.diff(ar, axis=0).T) <= 1e-9).sum())
    if repeated:
        notes.append(f"{repeated} repeated point(s)")
    row["spacing_jump"], row["x_jump"] = spacing_jump(ar)
    cps = {}
    try:
        cps["file"] = inviscid_cp(ar, alpha)
        for s in ("upper", "lower"):
            row[f"wiggle_{s}"], row[f"x_{s}"] = wiggle(*cps["file"][s])
    except (np.linalg.LinAlgError, ValueError) as e:
        notes.append(f"panel method failed: {e}")
    new = None
    if os.path.isfile(path):  # a generated nacaXXXX has no file to smooth
        try:
            row["written_decimals"], row["decimals"] = decimals(path)
            new, dist, dense, why = smooth_airfoil(read_points(path), row["decimals"])
            if why:
                notes.append("left out: " + "; ".join(why))
        except (np.linalg.LinAlgError, ValueError) as e:
            notes.append(f"smoothing failed: {e}")
    if new is not None:
        try:
            half = 0.5 * 10.0 ** -row["decimals"]
            row["fit_max"], row["fit_rms"] = dist.max() / half, np.sqrt(np.mean(dist ** 2)) / half
            # how far the surface moves: from the surface gmsh builds now to the smoothed curve
            ar_t, blunt = tidy_trailing_edge(ar)
            now = gmsh_curve(ar_t, 20)
            if not blunt:
                now = np.vstack([now, ar_t[-1] + (ar_t[0] - ar_t[-1]) * np.linspace(0, 1, 50)[:, None]])
            gap = curve_distance(dense, now)
            i = int(np.argmax(gap))
            row["shape_change"], row["x_shape_change"] = float(gap[i]), float(now[i, 0])
            cps["smoothed"] = inviscid_cp(as_written(new), alpha)
            for s in ("upper", "lower"):
                row[f"smooth_wiggle_{s}"] = wiggle(*cps["smoothed"][s])[0]
                if "file" in cps:
                    row[f"dcp_{s}"], row[f"x_dcp_{s}"] = cp_change(cps["file"][s], cps["smoothed"][s])
        except (np.linalg.LinAlgError, ValueError) as e:
            notes.append(f"panel method failed on the smoothed points: {e}")
    row["problems"] = "; ".join(notes)
    return row, (cps if keep_cp else None)


def fmt(value, column):
    if isinstance(value, float):
        return f"{value:.2e}" if column == "shape_change" else f"{value:.{2 if column.startswith(('spacing', 'fit')) else 4}f}"
    return "" if value is None else str(value)


def summary(rows):
    """A few lines about a whole set of files."""
    def largest(key):
        return np.array([max(r.get(f"{key}_upper", np.nan), r.get(f"{key}_lower", np.nan)) for r in rows])

    print(f"\n{len(rows)} files")
    for key, text in [("wiggle", "largest wiggle in Cp (either surface)"),
                      ("dcp", "largest change in Cp when smoothed"),
                      ("smooth_wiggle", "largest wiggle after smoothing")]:
        v = largest(key)
        v = v[np.isfinite(v)]
        if len(v):
            q = np.percentile(v, [50, 90, 99])
            print(f"{text}: median {q[0]:.4f}, 90% of the files below {q[1]:.4f}, 99% below {q[2]:.4f}, "
                  f"largest {v.max():.4f}; above 0.01: {(v > 0.01).sum()}, above 0.03: {(v > 0.03).sum()}, "
                  f"above 0.1: {(v > 0.1).sum()}")
    out = [r for r in rows if "left out" in r.get("problems", "")]
    print(f"{len(out)} files left out (not smoothed)" + (":" if out else ""))
    for r in out:
        print(f"  {r['file']}: {r['problems']}")
    other = [r for r in rows if r.get("problems") and r not in out]
    print(f"{len(other)} other files with notes" + (":" if other else ""))
    for r in other:
        print(f"  {r['file']}: {r['problems']}")


def read_table(path):
    """The rows of a table written with --csv, with the numbers as numbers (empty cells left
    out)."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            row = {}
            for key, value in r.items():
                if value in ("", None):
                    continue
                try:
                    row[key] = value if key in ("file", "problems") else float(value)
                except ValueError:
                    row[key] = value
            rows.append(row)
    return rows


def wiggle_plot(rows, path, alpha):
    """Distribution of the largest wiggle in the inviscid Cp of each file that is used (not
    left out), on the curve gmsh builds through the points as given and after smoothing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kept = [r for r in rows if "smooth_wiggle_upper" in r]
    ink, grid = "#333333", "#d9d9d9"
    plt.rcParams.update({"axes.edgecolor": ink, "axes.labelcolor": ink, "xtick.color": ink,
                         "ytick.color": ink, "text.color": ink, "font.size": 9})
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for key, color, label in [("wiggle", "#2a78d6", "as given: gmsh's spline through the points"),
                              ("smooth_wiggle", "#1baf7a", "smoothed (smooth_airfoils.py)")]:
        v = np.array([max(r.get(f"{key}_upper", np.nan), r.get(f"{key}_lower", np.nan)) for r in kept])
        v = np.sort(v[np.isfinite(v)])
        ax.step(v, np.arange(1, len(v) + 1) / len(v) * 100, where="post", color=color, lw=1.8,
                label=f"{label}: median {np.median(v):.4f}")
    ax.axvline(MAX_WIGGLE, color=ink, lw=0.8, ls="--")
    ax.text(MAX_WIGGLE * 1.15, 40, "screen: smoothed shapes that\nwiggle more are left out", fontsize=8)
    ax.set(xscale="log", xlim=(1e-4, 1e2), ylim=(0, 100), ylabel="share of the files (%)",
           xlabel=f"largest wiggle in the inviscid Cp of a file ({alpha:g} deg, 2-95% chord)")
    ax.grid(color=grid, lw=0.6)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.set_title(f"Kinks in the {len(kept)} airfoils used for the dataset, before and after smoothing")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="*", help="airfoil files, folders of .dat files, or wildcard patterns")
    ap.add_argument("--alpha", type=float, default=3.45, help="incidence for the panel method, degrees")
    ap.add_argument("--plot", help="also draw the inviscid Cp of all files into this image (for a few files)")
    ap.add_argument("--csv", help="write one row per file into this table")
    ap.add_argument("--wiggle-plot", help="draw the distribution of the largest wiggle per file, as given "
                                          "and smoothed, into this image")
    ap.add_argument("--from-csv", help="do not check files: summarize (and --wiggle-plot) the table of an "
                                       "earlier run")
    ap.add_argument("--jobs", type=int, default=1,
                    help="files to check at the same time, one per processor core (default 1)")
    args = ap.parse_args()
    if args.from_csv:
        rows = read_table(args.from_csv)
        summary(rows)
        if args.wiggle_plot:
            wiggle_plot(rows, args.wiggle_plot, args.alpha)
        return
    files = file_list(args.files)
    if not files:
        raise SystemExit("no airfoil files found")
    work = functools.partial(check, alpha=args.alpha, keep_cp=bool(args.plot))
    verbose = len(files) <= 50
    results = []
    if args.jobs > 1:
        for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = "1"  # one thread in each process: the processes share the cores
        pool = multiprocessing.Pool(args.jobs)
        it = pool.imap(work, files, chunksize=4)
    else:
        pool, it = None, map(work, files)
    for k, (row, cps) in enumerate(it, 1):
        results.append((row, cps))
        if verbose:
            text = "; ".join(f"{s} surface {row[f'wiggle_{s}']:.3f} at x/c {row[f'x_{s}']:.3f}"
                             + (f" ({row[f'smooth_wiggle_{s}']:.3f} smoothed, Cp changes by up to "
                                f"{row[f'dcp_{s}']:.3f})" if f"dcp_{s}" in row else "")
                             for s in ("upper", "lower") if f"wiggle_{s}" in row)
            print(f"{row['file']} ({row.get('points', '?')} points): largest wiggle in inviscid Cp: {text}"
                  + (f" [{row['problems']}]" if row.get("problems") else ""))
        elif k % 100 == 0 or k == len(files):
            print(f"checked {k} of {len(files)} files", flush=True)
    if pool:
        pool.close()
        pool.join()
    rows = [r for r, _ in results]
    if not verbose:
        summary(rows)
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLUMNS)
            for r in rows:
                w.writerow([fmt(r.get(c), c) for c in COLUMNS])
        print("wrote", args.csv)
    if args.wiggle_plot:
        wiggle_plot(rows, args.wiggle_plot, args.alpha)
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        for k, (row, cps) in enumerate(r for r in results if r[1] and "file" in r[1]):
            for ax in axes:
                ax.plot(*cps["file"]["upper"], "-", color=f"C{k}", lw=1, label=f"{row['file']}, upper surface")
                ax.plot(*cps["file"]["lower"], "--", color=f"C{k}", lw=1, label=f"{row['file']}, lower surface")
        for ax, xlim in [(axes[0], (0, 1)), (axes[1], (0, 0.3))]:
            ax.invert_yaxis()
            ax.set(xlim=xlim, xlabel="x/c", ylabel=f"inviscid Cp, {args.alpha:g} deg")
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        axes[0].set_title("Inviscid surface pressure on the curve gmsh builds through the points")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=110)
        print("saved", args.plot)


if __name__ == "__main__":
    main()
