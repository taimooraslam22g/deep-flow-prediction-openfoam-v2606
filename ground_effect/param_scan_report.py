#!/usr/bin/env python3
"""Parameter-range scan (param_scan.sh): where does steady RANS converge?

Reads the samples.csv of every worker of a scan, the Cl history of each saved run, and the
worker logs for the runs that stopped before a sample was saved (mesh or solver failures).
Every run is put in one of these groups:
  - steady: Cl varies by less than 0.1% over the last 200 iterations (groundGen.py's
    convergence test); counted separately when nuTilda was still clipped at the end;
  - settling: Cl not steady yet, but its range shrinking (groundGen.py now continues such
    runs, up to 2000 iterations);
  - oscillating: Cl neither steady nor settling (separated, unsteady flow);
  - failed: no sample saved (the log says why).
The report gives the share of each group by ride height, angle of attack and air speed,
the share of steady runs in cells of ride height x angle of attack, run times, the range
of Cl and Cd of the steady runs, and the airfoils that failed more than once. For
comparison it also counts the runs that passed the earlier test (steady Cl and no
clipping at the end), which the scan's own 'converged' column used.
--plot draws every run in the plane of angle of attack and ride height; --csv writes one
row per run.

Usage, on Windows after the scan (default folders: ground_effect/data/param_scan and
ground_effect/logs/param_scan):
    .venv\\Scripts\\python.exe ground_effect\\param_scan_report.py --plot results\\param_scan.png --csv results\\param_scan.csv
"""
import argparse
import csv
import glob
import os
import re
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from groundGen import ALPHA_RANGE, AVERAGE, H_RANGE, STEADY_TOL, U_RANGE, settling  # noqa: E402

RUN = re.compile(r"^Run \d+: (.+?), alpha ([+-]?[\d.]+) deg, (?:h/c ([\d.]+)|free air).*?, U ([\d.]+) m/s")
GROUPS = ["steady", "steady_clipped", "settling", "oscillating", "failed"]


def number(row, key, default=np.nan):
    """A number from a row of samples.csv (older files lack some columns)."""
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def load(folder, log_folder):
    """All runs of a scan, as dicts: airfoil, alpha, h, U, group (see GROUPS, or 'unknown'
    for tables without cl_range), why (for failures), clipped, and for saved runs the
    columns of samples.csv."""
    runs = []
    for path in sorted(glob.glob(os.path.join(folder, "*", "samples.csv"))):
        with open(path) as f:
            for r in csv.DictReader(f):
                run = {"airfoil": r["airfoil"], "alpha": number(r, "alpha_deg"), "h": number(r, "h_over_c"),
                       "U": number(r, "U"), "Cl": number(r, "Cl"), "Cd": number(r, "Cd"),
                       "cl_range": number(r, "cl_range"), "seconds": number(r, "seconds"),
                       "cells": number(r, "cells"), "reversed": number(r, "reversed_flow"),
                       "iterations": number(r, "iterations"), "why": ""}
                iters = number(r, "iterations", 0)
                run["clipped"] = number(r, "bounded", 0) > 0 and number(r, "last_bounded", 0) > iters - AVERAGE
                if not np.isfinite(run["cl_range"]):
                    run["group"] = "unknown"
                elif run["cl_range"] < STEADY_TOL:
                    run["group"] = "steady_clipped" if run["clipped"] else "steady"
                else:
                    npz = os.path.join(os.path.dirname(path), r["file"])
                    cl = np.load(npz)["Cl_history"] if os.path.exists(npz) else None
                    run["group"] = "settling" if cl is not None and settling(cl) else "oscillating"
                runs.append(run)
    for path in sorted(glob.glob(os.path.join(log_folder, "*.log"))):
        current = None
        with open(path, errors="replace") as f:
            for line in f:
                m = RUN.match(line)
                if m:
                    current = {"airfoil": m.group(1), "alpha": float(m.group(2)),
                               "h": float(m.group(3)) if m.group(3) else np.inf, "U": float(m.group(4))}
                elif current is not None and "skipping" in line:
                    current["group"] = "failed"
                    current["why"] = line.strip().split(", skipping")[0]
                    current["clipped"] = False
                    runs.append(current)
                    current = None
                elif line.startswith("    saved"):
                    current = None
    return runs


def bins(runs, key, lo, hi, n, geometric=False):
    """n bins over groundGen's range of one parameter (widened to the runs if they go beyond
    it); geometric spacing for the ride height, whose small values matter most."""
    v = [r[key] for r in runs if np.isfinite(r[key])]
    lo, hi = min([lo] + v), max([hi] + v)
    e = np.geomspace(lo, hi, n + 1) if geometric and lo > 0 else np.linspace(lo, hi, n + 1)
    return np.round(e, 3)


def in_bin(value, edges, i):
    """True if value lies in bin i of edges (the last bin includes its upper end)."""
    lo, hi = edges[i], edges[i + 1]
    return lo <= value < hi or (i == len(edges) - 2 and value == hi)


def rates(runs, key, edges, label):
    """Per bin of one parameter: runs, and the share that is steady, steady or settling,
    and steady without clipping (the earlier test)."""
    print(f"by {label}:")
    print(f"  {'':>14s}{'runs':>9s}{'steady':>9s}{'+ settling':>12s}{'earlier test':>14s}")
    for i in range(len(edges) - 1):
        sel = [r for r in runs if np.isfinite(r[key]) and in_bin(r[key], edges, i)]
        if sel:
            g = Counter(r["group"] for r in sel)
            n, st = len(sel), g["steady"] + g["steady_clipped"]
            print(f"  {f'{edges[i]:g}-{edges[i + 1]:g}':>14s}{n:9d}{100 * st / n:8.0f}%"
                  f"{100 * (st + g['settling']) / n:11.0f}%{100 * g['steady'] / n:13.0f}%")


def table(runs, h_edges, a_edges):
    """Steady (or settling) / all runs in cells of ride height (rows) x angle of attack (columns)."""
    print("steady or settling / runs, ride height (rows) x angle of attack (columns):")
    print("  h/c \\ alpha " + "".join(f"{f'{a0:g}-{a1:g}':>9s}" for a0, a1 in zip(a_edges[:-1], a_edges[1:])))
    for i in range(len(h_edges) - 1):
        cells = []
        for j in range(len(a_edges) - 1):
            sel = [r for r in runs if np.isfinite(r["h"]) and in_bin(r["h"], h_edges, i) and in_bin(r["alpha"], a_edges, j)]
            ok = sum(r["group"] in ("steady", "steady_clipped", "settling") for r in sel)
            cells.append(f"{ok}/{len(sel)}" if sel else "-")
        print(f"  {f'{h_edges[i]:g}-{h_edges[i + 1]:g}':>11s} " + "".join(f"{c:>9s}" for c in cells))


def write_csv(runs, path):
    """One row per run: parameters, group and the main results (empty where a run has none)."""
    cols = ["airfoil", "alpha", "h", "U", "group", "Cl", "Cd", "cl_range", "clipped", "reversed", "iterations",
            "seconds", "why"]
    names = {"alpha": "alpha_deg", "h": "h_over_c", "reversed": "reversed_flow"}

    def text(r, k):
        v = r.get(k, "")
        if isinstance(v, (bool, np.bool_)):
            return str(int(v))
        if isinstance(v, float):
            return "" if not np.isfinite(v) else f"{v:.6g}"
        return str(v)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([names.get(c, c) for c in cols])
        for r in sorted(runs, key=lambda r: (r["airfoil"], r["h"], r["alpha"])):
            w.writerow([text(r, c) for c in cols])
    print("saved", path)


def plot(runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, grid = "#333333", "#d9d9d9"
    plt.rcParams.update({"axes.edgecolor": ink, "axes.labelcolor": ink, "xtick.color": ink,
                         "ytick.color": ink, "text.color": ink, "font.size": 9})
    fig, ax = plt.subplots(figsize=(7.5, 5))
    shown = [r for r in runs if np.isfinite(r["h"])]  # not the runs in free air
    # categorical slots 1-3 for the groups of saved runs; failures in the status colour 'critical'
    style = {"steady": ("#2a78d6", "o", "full"), "steady_clipped": ("#2a78d6", "o", "none"),
             "settling": ("#1baf7a", "^", "full"), "oscillating": ("#eb6834", "x", "full"),
             "failed": ("#d03b3b", "s", "none")}
    label = {"steady": "steady Cl", "steady_clipped": "steady Cl, nuTilda clipped to the end",
             "settling": "Cl still settling at 1000 iterations", "oscillating": "Cl oscillating",
             "failed": "mesh failed"}
    for g in GROUPS:
        sel = [r for r in shown if r["group"] == g]
        if not sel:
            continue
        color, marker, fill = style[g]
        size = [8 + r["U"] / 2 for r in sel]
        if fill == "none":
            ax.scatter([r["alpha"] for r in sel], [r["h"] for r in sel], s=size, facecolors="none",
                       edgecolors=color, marker=marker, linewidths=1.1, label=f"{label[g]} ({len(sel)})")
        else:
            ax.scatter([r["alpha"] for r in sel], [r["h"] for r in sel], s=size, c=color, marker=marker,
                       linewidths=1.2, label=f"{label[g]} ({len(sel)})")
    ax.set(xlabel="angle of attack (degrees)", ylabel="ride height h/c (log scale)", yscale="log",
           title=f"Parameter scan: {len(shown)} random cases (marker size: air speed, 10-50 m/s)")
    ticks = [0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6]
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.grid(color=grid, lw=0.6)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", nargs="?", default=os.path.join(HERE, "data", "param_scan"),
                    help="the scan's output folder (default ground_effect/data/param_scan)")
    ap.add_argument("--logs", help="its log folder (default: ground_effect/logs/<name of the folder>)")
    ap.add_argument("--plot", help="draw the runs in the alpha-h plane into this image")
    ap.add_argument("--csv", help="write one row per run (parameters, group, Cl, Cd, ...) to this file")
    args = ap.parse_args()
    logs = args.logs or os.path.join(HERE, "logs", os.path.basename(os.path.normpath(args.folder)))
    runs = load(args.folder, logs)
    if not runs:
        raise SystemExit(f"no runs found in {args.folder} or {logs}")
    g = Counter(r["group"] for r in runs)
    n = len(runs)
    print(f"{n} runs: {g['steady'] + g['steady_clipped']} with a steady Cl ({g['steady_clipped']} of them with "
          f"nuTilda clipped until the end), {g['settling']} still settling, {g['oscillating']} oscillating, "
          f"{g['failed']} failed before a sample was saved"
          + (f", {g['unknown']} from older tables without cl_range (left out below)" if g["unknown"] else ""))
    print(f"  passed the earlier test (steady Cl, no clipping at the end): {g['steady']} "
          f"({100 * g['steady'] / n:.0f}%); steady Cl: {100 * (g['steady'] + g['steady_clipped']) / n:.0f}%; "
          f"steady or settling: {100 * (n - g['oscillating'] - g['failed'] - g['unknown']) / n:.0f}%")
    for why, k in Counter(r["why"] for r in runs if r["group"] == "failed").most_common():
        print(f"  {k:3d} x {why}")
    runs = [r for r in runs if r["group"] != "unknown"]
    free = sum(not np.isfinite(r["h"]) for r in runs)
    if free:
        print(f"  ({free} runs in free air are left out of the tables by ride height)")
    rates(runs, "h", bins(runs, "h", *H_RANGE, 5, geometric=True), "ride height h/c")
    rates(runs, "alpha", bins(runs, "alpha", *ALPHA_RANGE, 5), "angle of attack (deg)")
    rates(runs, "U", bins(runs, "U", *U_RANGE, 4), "air speed (m/s)")
    table(runs, bins(runs, "h", *H_RANGE, 4, geometric=True), bins(runs, "alpha", *ALPHA_RANGE, 4))
    saved = [r for r in runs if np.isfinite(r.get("seconds", np.nan))]
    if saved:
        sec = np.array([r["seconds"] for r in saved])
        cells = np.array([r["cells"] for r in saved])
        print(f"run time per sample: median {np.median(sec) / 60:.1f} min (10-90%: {np.percentile(sec, 10) / 60:.1f}"
              f"-{np.percentile(sec, 90) / 60:.1f} min); cells: median {np.nanmedian(cells):,.0f}")
    steady = [r for r in runs if r["group"] in ("steady", "steady_clipped")]
    if steady:
        cl = np.array([r["Cl"] for r in steady])
        cd = np.array([r["Cd"] for r in steady])
        rev = np.array([r["reversed"] for r in steady])
        print(f"steady runs: Cl {np.nanmin(cl):+.3f} to {np.nanmax(cl):+.3f} (negative = downforce), Cd "
              f"{np.nanmin(cd):.4f} to {np.nanmax(cd):.4f}; reversed flow on more than 10% of the surface in "
              f"{(rev > 0.1).sum()} of {len(steady)}")
    bad = Counter(r["airfoil"] for r in runs if r["group"] in ("oscillating", "failed"))
    repeat = [(k, m) for k, m in bad.most_common() if m > 1]
    if repeat:
        print("airfoils that oscillated or failed more than once: " + ", ".join(f"{k} ({m})" for k, m in repeat))
    if args.csv:
        write_csv(runs, args.csv)
    if args.plot:
        plot(runs, args.plot)


if __name__ == "__main__":
    main()
