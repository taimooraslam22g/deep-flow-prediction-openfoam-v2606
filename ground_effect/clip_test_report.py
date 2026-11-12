#!/usr/bin/env python3
"""Results of clip_test.sh: what the nuTilda clipping changes, and whether runs whose Cl is
still settling converge when they continue.

A. Four scan runs with a steady Cl but nuTilda clipped in nearly every iteration, against
   the same run after 2000 iterations (does the flow still change?) and with first-order
   upwind convection for nuTilda (nothing to clip). Flow fields are compared like the U-Net
   test error (verification_report.field_difference): sum |a - b| / sum |b| over the
   sampled window, pressure without its mean. Also: where the clipped cells are, from the
   kept work folders (clip_cells.py).
B. Six scan runs whose Cl still changed after 1000 iterations, rerun with continuation
   (while Cl is settling, up to 2000 iterations; fxs03182 also up to 3000); one of them
   also straight to 2000 iterations, which must give the same answer as continuing.
C. The two separated cases of A with all four nuTilda schemes of groundGen.py
   (cellLimited, the default; linearUpwind, which clips far more; limitedLinear; upwind),
   against the default after 2000 iterations: is the difference made by upwind the
   clipping or the first-order scheme?
For every rerun, the first 1000 iterations are checked against the scan run (same case,
same settings: they should be identical).

Usage, on Windows after the test:
    .venv\\Scripts\\python.exe ground_effect\\clip_test_report.py --plot results\\clip_test.png
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from clip_cells import clipped_cells, summary  # noqa: E402
from groundGen import AVERAGE, STEADY_TOL, STEADY_WINDOW, rel_range  # noqa: E402

TEST = os.path.join(HERE, "data", "clip_test")
SCAN = os.path.join(HERE, "data", "param_scan")
WORK = os.path.join(HERE, "work")
NUTILDA_INF = 3e-5
# name: worker (as in clip_test.sh)
A_CASES = {"goe280": (102, 110), "e186": (103, 111), "goe281": (104, 112), "n9": (105, 113)}
B_CASES = {"goe284": 100, "fxs03182": 101, "e337": 106, "fx08s176": 107, "goe553": 108, "goe225": 109,
           "fxs03182_3000": 119}
C_CASES = {"goe280": {"cellLimited": 102, "linearUpwind": 115, "limitedLinear": 117, "upwind": 110},
           "n9": {"cellLimited": 105, "linearUpwind": 116, "limitedLinear": 118, "upwind": 113}}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # categorical slots 1-6


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def test_run(name):
    """The sample of one test run (in its folder or in not_converged/), or None."""
    files = glob.glob(os.path.join(TEST, name, "**", "*.npz"), recursive=True)
    return load(files[0]) if files else None


SCAN_FILES = None


def scan_run(r):
    """The scan run with the same airfoil, angle, ride height and speed as test run r."""
    global SCAN_FILES
    if SCAN_FILES is None:
        SCAN_FILES = glob.glob(os.path.join(SCAN, "**", "*.npz"), recursive=True)
    stem = str(r["airfoil"])
    for path in SCAN_FILES:
        if os.path.basename(path).startswith(stem + "_h"):
            s = load(path)
            if all(abs(float(s[k]) - float(r[k])) < 1e-9 for k in ("alpha", "h", "U")):
                return s
    return None


def field_difference(a, b):
    """Difference of sample a from sample b (U-Net error measure), in %: (pressure, velocity)."""
    fluid = (a[2] == 0) & (b[2] == 0)
    pa, pb = a[3][fluid] - a[3][fluid].mean(), b[3][fluid] - b[3][fluid].mean()
    dv = np.abs(a[4] - b[4]) + np.abs(a[5] - b[5])
    return (100 * np.abs(pa - pb).sum() / np.abs(pb).sum(),
            100 * dv[fluid].sum() / (np.abs(b[4]) + np.abs(b[5]))[fluid].sum())


def same_start(r, s):
    """Largest relative difference of Cl over the scan run's iterations (0: identical)."""
    n = min(len(r["Cl_history"]), len(s["Cl_history"]))
    a, b = r["Cl_history"][:n], s["Cl_history"][:n]
    return float(np.max(np.abs(a - b) / np.abs(b)))


def clip_text(r):
    if int(r["bounded"]) == 0:
        return "none"
    end = bool(r["clipped"]) if "clipped" in r else int(r["last_bounded"]) > int(r["iterations"]) - AVERAGE
    low = f" ({float(r['clip_min']) / NUTILDA_INF:.0f}x)" if "clip_min" in r else ""  # older samples lack it
    return (f"{int(r['bounded'])} its, to the end{low}" if end
            else f"{int(r['bounded'])} its, last {int(r['last_bounded'])}")


def part_a():
    print("A. Steady Cl, nuTilda clipped in nearly every iteration: the scan run (1000 iterations) "
          "against 2000 iterations\n   and against upwind convection of nuTilda. dCl, dCd and the field "
          "differences are from the scan run, in %\n   (field: pressure / velocity, measured like the U-Net "
          "test error). Clipping: iterations, and the most negative value\n   in the last "
          f"{AVERAGE} iterations as a multiple of the incoming nuTilda.")
    print(f"   {'case':10s}{'run':>16s}{'its':>6s}{'Cl':>9s}{'Cd':>8s}{'Cl range':>10s}{'dCl':>8s}{'dCd':>8s}"
          f"{'field p / v':>15s}   clipping")
    rows = []
    for name, (w2000, wup) in A_CASES.items():
        runs = {"2000 its": test_run(f"{name}_2000"), "upwind": test_run(f"{name}_upwind")}
        ref = next((scan_run(r) for r in runs.values() if r is not None), None)
        if ref is None:
            print(f"   {name}: no results yet")
            continue
        print(f"   {name:10s}{'scan':>16s}{int(ref['iterations']):6d}{float(ref['Cl']):+9.4f}{float(ref['Cd']):8.4f}"
              f"{100 * float(ref['cl_range']):9.3f}%{'':31s}   {clip_text(ref)}")
        for label, r in runs.items():
            if r is None:
                print(f"   {'':10s}{label:>16s}  missing")
                continue
            dp, dv = field_difference(r["a"], ref["a"])
            dcl = 100 * (float(r["Cl"]) / float(ref["Cl"]) - 1)
            dcd = 100 * (float(r["Cd"]) / float(ref["Cd"]) - 1)
            print(f"   {'':10s}{label:>16s}{int(r['iterations']):6d}{float(r['Cl']):+9.4f}{float(r['Cd']):8.4f}"
                  f"{100 * float(r['cl_range']):9.3f}%{dcl:+8.2f}{dcd:+8.2f}{f'{dp:.2f} / {dv:.2f}':>15s}   {clip_text(r)}")
            rows.append((name, label, dcl, dcd, dp, dv))
        r2 = runs["2000 its"]
        if r2 is not None:
            d = same_start(r2, ref)
            print(f"   {'':26s}first 1000 iterations of the 2000-iteration run vs the scan: "
                  + ("identical" if d == 0 else f"differ by up to {d:.1e} (relative)"))
        if runs["upwind"] is not None and r2 is not None:
            dp, dv = field_difference(runs["upwind"]["a"], r2["a"])
            print(f"   {'':26s}upwind vs 2000 iterations: field {dp:.2f}% / {dv:.2f}%, "
                  f"dCl {100 * (float(runs['upwind']['Cl']) / float(r2['Cl']) - 1):+.2f}%")
        for label, w in (("2000 its", w2000), ("upwind", wup)):
            case = os.path.join(WORK, f"worker{w}")
            if os.path.isdir(case) and runs[label] is not None:
                try:
                    print(f"   {'':26s}clipped cells, {label}: {summary(clipped_cells(case))}")
                except (OSError, ValueError) as e:
                    print(f"   {'':26s}clipped cells, {label}: could not read {case} ({e})")
    if rows:
        def biggest(label, i):
            v = [abs(r[i]) for r in rows if r[1] == label]
            return max(v) if v else np.nan
        print(f"   largest change after 2000 iterations: Cl {biggest('2000 its', 2):.2f}%, Cd "
              f"{biggest('2000 its', 3):.2f}%, field {biggest('2000 its', 4):.2f}% / {biggest('2000 its', 5):.2f}%")
        print(f"   largest difference to upwind: Cl {biggest('upwind', 2):.2f}%, Cd {biggest('upwind', 3):.2f}%, "
              f"field {biggest('upwind', 4):.2f}% / {biggest('upwind', 5):.2f}%")
    print()


def part_b():
    print(f"B. Cl still changing after 1000 iterations in the scan; rerun, continuing while Cl is settling, up to "
          f"2000 iterations\n   (fxs03182_3000: up to 3000). Cl range: over the last {STEADY_WINDOW} iterations; "
          f"converged below {100 * STEADY_TOL:g}%.")
    print(f"   {'case':14s}{'scan Cl range':>14s}{'its':>6s}{'Cl range':>10s}{'converged':>11s}{'Cl':>9s}"
          f"{'dCl vs scan':>12s}   first 1000 its   clipping")
    out = {}
    for name in B_CASES:
        r = test_run(name)
        if r is None:
            print(f"   {name}: no results yet")
            continue
        s = scan_run(r)
        d = same_start(r, s) if s is not None else np.nan
        same = "identical" if d == 0 else f"differ {d:.0e}"
        dcl = 100 * (float(r["Cl"]) / float(s["Cl"]) - 1) if s is not None else np.nan
        print(f"   {name:14s}{100 * float(s['cl_range']) if s is not None else np.nan:13.3f}%{int(r['iterations']):6d}"
              f"{100 * float(r['cl_range']):9.3f}%{'yes' if bool(r['converged']) else 'no':>11s}"
              f"{float(r['Cl']):+9.4f}{dcl:+11.2f}%   {same:15s}  {clip_text(r)}")
        out[name] = r
    straight = test_run("fxs03182_2000")
    if straight is not None and "fxs03182" in out:
        r = out["fxs03182"]
        n = min(len(r["Cl_history"]), len(straight["Cl_history"]))
        d = float(np.max(np.abs(r["Cl_history"][:n] / straight["Cl_history"][:n] - 1)))
        text = (f"   continued run vs the same run straight to 2000 iterations (fxs03182): Cl over iterations 1-{n} "
                + ("identical" if d == 0 else f"differs by up to {100 * d:.2g}%"))
        if int(r["iterations"]) == int(straight["iterations"]):
            dp, dv = field_difference(r["a"], straight["a"])
            text += f"; final field {dp:.3f}% / {dv:.3f}%"
        print(text)
    print()
    return out


def part_c():
    print("C. Separated flow (goe280: h/c 0.109, 6.7 deg; n9: h/c 0.052, 3.8 deg) with the four nuTilda schemes. "
          "dCl, dCd and\n   the field are from the default (cellLimited) after 2000 iterations, in %. "
          "Clipped cells: in the last iteration.")
    print(f"   {'case':8s}{'scheme':>15s}{'its':>6s}{'Cl':>9s}{'Cd':>8s}{'Cl range':>10s}{'dCl':>8s}{'dCd':>8s}"
          f"{'field p / v':>15s}   clipping; clipped cells")
    for case, schemes in C_CASES.items():
        ref = test_run(f"{case}_2000")
        if ref is None:
            print(f"   {case}: no results yet")
            continue
        for scheme, worker in schemes.items():
            r = ref if scheme == "cellLimited" else test_run(f"{case}_{scheme}")
            if r is None:
                print(f"   {case:8s}{scheme:>15s}  no results yet")
                continue
            dp, dv = field_difference(r["a"], ref["a"])
            dcl = 100 * (float(r["Cl"]) / float(ref["Cl"]) - 1)
            dcd = 100 * (float(r["Cd"]) / float(ref["Cd"]) - 1)
            cells = ""
            folder = os.path.join(WORK, f"worker{worker}")
            if os.path.isdir(folder):
                try:
                    c = clipped_cells(folder)
                    cells = f"; {len(c['idx'])} cells" + (f", {np.sum(c['xy'][:, 0] > c['wing'][:, 0].max())} "
                                                         f"behind the wing" if len(c["idx"]) else "")
                except (OSError, ValueError):
                    cells = "; (work folder unreadable)"
            print(f"   {case:8s}{scheme:>15s}{int(r['iterations']):6d}{float(r['Cl']):+9.4f}{float(r['Cd']):8.4f}"
                  f"{100 * float(r['cl_range']):9.3f}%{dcl:+8.2f}{dcd:+8.2f}{f'{dp:.2f} / {dv:.2f}':>15s}   "
                  f"{clip_text(r)}{cells}")
    print()


def window_ranges(cl, step=50):
    """Relative Cl range over the STEADY_WINDOW iterations before each iteration (every step)."""
    its = np.arange(STEADY_WINDOW, len(cl) + 1, step)
    return its, np.array([rel_range(cl[i - STEADY_WINDOW:i], 0.05) for i in its])


def plot(b_runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, grid = "#333333", "#d9d9d9"
    plt.rcParams.update({"axes.edgecolor": ink, "axes.labelcolor": ink, "xtick.color": ink,
                         "ytick.color": ink, "text.color": ink, "font.size": 9})
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    colour = {}
    for name, r in b_runs.items():
        base = name.split("_")[0]  # fxs03182_3000 is drawn dashed, in the colour of fxs03182
        colour.setdefault(base, COLORS[len(colour) % len(COLORS)])
        its, rng = window_ranges(r["Cl_history"])
        ax.plot(its, 100 * rng, color=colour[base], lw=1.6, ls="--" if base != name else "-",
                label=f"{name} ({'converged' if bool(r['converged']) else 'not converged'} "
                      f"after {int(r['iterations'])})")
    longest = max(len(r["Cl_history"]) for r in b_runs.values())
    ax.axhline(100 * STEADY_TOL, color=ink, lw=0.8, ls="--")
    ax.text(60, 100 * STEADY_TOL * 1.15, "converged below this", fontsize=8)
    ax.axvline(1000, color=grid, lw=1.2)
    ax.text(1010, 60, "the scan stopped here", fontsize=8, va="top")
    ax.set(yscale="log", xlim=(0, max(2000, longest)), ylim=(1e-3, 100), xlabel="iteration",
           ylabel=f"Cl range over the previous {STEADY_WINDOW} iterations (%)",
           title="Runs whose Cl still changed after 1000 iterations, continued while Cl settles")
    ax.grid(color=grid, lw=0.6)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plot", help="draw the Cl range of the part-B runs over the iterations into this image")
    args = ap.parse_args()
    if not os.path.isdir(TEST):
        raise SystemExit(f"no results in {TEST}: run clip_test.sh in the container first")
    part_a()
    b_runs = part_b()
    part_c()
    if args.plot and b_runs:
        plot(b_runs, args.plot)


if __name__ == "__main__":
    main()
