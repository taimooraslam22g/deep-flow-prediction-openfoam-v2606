#!/usr/bin/env python3
"""Tables and figures for the numerical verification (mesh study, round 3), after
verification.sh has run in the container.

Usage (on Windows, with the project's Python):
    .venv\\Scripts\\python.exe ground_effect\\verification_report.py

It reads the samples in data/verification/ and the solver logs and force histories in the
runs' work folders (work/worker31 to worker41; the round-2 runs: worker22 and worker24),
so keep those folders until the results are written up.

Prints:
    A. nuTilda scheme test: clipping, residuals, forces, Cf scatter, accuracy vs the LES
    B. grid convergence index (GCI) of Cl, Cd and the sampled flow field, two cases
    C. domain size at the largest ride height
    D. y+ and cell sizes of the grid convergence meshes
    E. run time: start-up (mostly the search for the sample points) and solver
Cl and Cd are averages over the last 100 iterations, because the drag flips between two
values from one iteration to the next.
Writes to results/:
    scheme_test_les.png        Cp and Cf vs the LES with the three nuTilda schemes
    scheme_test_residuals.png  nuTilda residual during the runs, three schemes
    grid_convergence.png       Cl and Cd against cell size, with the extrapolated values
"""
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")  # write to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from groundGen import (AVERAGE, STEADY_TOL, STEADY_WINDOW, bounded_iterations, cell_count,  # noqa: E402
                       final_residuals, force_summary, read_coeffs, reversed_flow)
from mesh_quality import report as mesh_report  # noqa: E402
from validate_les import rans  # noqa: E402

LES = os.path.join(ROOT, "data", "wing_les", "WingData", "naca4412.mat")
OUT = os.path.join(ROOT, "results")
FREE = "naca4412_free_a+0500_U0100.npz"
GROUND = "naca23012_h0200_a+0400_U0300.npz"
HARD = "s1223_h0100_a+0600_U0500.npz"
HIGH = "naca23012_h0600_a+0400_U0300.npz"


def path(*parts):
    return os.path.join(HERE, *parts)


def v(name, npz):
    """Sample file of a verification.sh run (a newer groundGen.py saves runs that did not
    converge in not_converged/)."""
    main, other = path("data", "verification", name, npz), path("data", "verification", name, "not_converged", npz)
    return other if not os.path.exists(main) and os.path.exists(other) else main


# name: (sample, work folder). The linearUpwind runs are from round 2 (mesh_study.sh).
RUNS = {
    "free_linearUpwind": (path("data", "validation", "r2_d20", FREE), "worker22"),
    "free_cellLimited": (v("free_cellLimited", FREE), "worker31"),
    "free_limitedLinear": (v("free_limitedLinear", FREE), "worker32"),
    "ground_linearUpwind": (path("data", "tests", "r2_d20", GROUND), "worker24"),
    "ground_medium": (v("ground_medium", GROUND), "worker33"),
    "ground_limitedLinear": (v("ground_limitedLinear", GROUND), "worker34"),
    "ground_coarse": (v("ground_coarse", GROUND), "worker35"),
    "ground_fine": (v("ground_fine", GROUND), "worker36"),
    "hard_coarse": (v("hard_coarse", HARD), "worker37"),
    "hard_medium": (v("hard_medium", HARD), "worker38"),
    "hard_fine": (v("hard_fine", HARD), "worker39"),
    "high_d20": (v("high_d20", HIGH), "worker40"),
    "high_d40": (v("high_d40", HIGH), "worker41"),
}
ROUND2 = ("free_linearUpwind", "ground_linearUpwind")
SCHEMES = {
    "free air (LES case)": [("linearUpwind (round 2)", "free_linearUpwind"), ("cellLimited", "free_cellLimited"),
                            ("limitedLinear", "free_limitedLinear")],
    "ground effect (NACA 23012)": [("linearUpwind (round 2)", "ground_linearUpwind"), ("cellLimited", "ground_medium"),
                                   ("limitedLinear", "ground_limitedLinear")]}
GCI_CASES = {"NACA 23012, 4 deg, h/c 0.2, 30 m/s": ["ground_fine", "ground_medium", "ground_coarse"],
             "S1223, 6 deg, h/c 0.1, 50 m/s": ["hard_fine", "hard_medium", "hard_coarse"]}
DOMAIN = [("20 chords", "high_d20"), ("40 chords", "high_d40")]
LEVELS = ["fine", "medium", "coarse"]
P_FORMAL = 2.0  # formal order of the discretisation
SHOW = 500      # the figure shows the range of Cl and Cd over this many last iterations


def load(name):
    """What the report needs about one run, or None (with a message) if files are missing."""
    npz, worker = RUNS[name]
    log = path("work", worker, "log.simpleFoam")
    gone = [f for f in (npz, log) if not os.path.exists(f)]
    for f in gone:
        print(f"   missing: {os.path.relpath(f, ROOT)}")
    if gone:
        return None
    z = np.load(npz)
    if "Cd_history" in z.files and len(z["Cd_history"]):
        cl_h, cd_h = z["Cl_history"], z["Cd_history"]
    else:  # samples made before Cd_history was saved: read the run's force file
        co = read_coeffs(path("work", worker))
        if co is None:
            print(f"   missing: the force coefficients in work/{worker}")
            return None
        cl_h, cd_h = co
    cl, cd, cl_range, cd_osc = force_summary(cl_h, cd_h)
    clipped, last = bounded_iterations(log)
    clipped_at_end = clipped > 0 and last > 0.9 * len(cl_h)
    cells = int(z["cells"]) if "cells" in z.files else cell_count(path("work", worker))  # round 2: not saved
    return dict(z=z, log=log, worker=worker, cl_h=cl_h, cd_h=cd_h, Cl=cl, Cd=cd, cl_range=cl_range,
                cd_osc=cd_osc, clipped=clipped, last_clipped=last, cells=cells,
                converged=cl_range < STEADY_TOL and not clipped_at_end,
                rev=reversed_flow(dict(z), float(z["alpha"])))


def residual_history(log, field="nuTilda"):
    """Initial residual of one field at every iteration, from the simpleFoam log."""
    key = f"Solving for {field}, Initial residual = "
    with open(log) as f:
        return np.array([float(line.split(key)[1].split(",")[0]) for line in f if key in line])


def lowest_clipped(log, last=100):
    """The most negative nuTilda clipped in the last `last` iterations, or None."""
    found, it = [], 0
    with open(log) as f:
        for line in f:
            if line.startswith("Time = "):
                it = int(float(line.split("=")[1]))
            elif line.startswith("bounding nuTilda"):
                found.append((it, float(line.split("min:")[1].split()[0])))
    late = [m for i, m in found if i > it - last]
    return min(late) if late else None


def execution_times(log):
    """Seconds when the first and the last iteration had finished."""
    with open(log) as f:
        t = [float(line.split()[2]) for line in f if line.startswith("ExecutionTime")]
    return t[0], t[-1]


def settled(cl, tol=0.005):
    """First iteration from which Cl stays within tol of its final value."""
    bad = np.where(np.abs(cl - cl[-1]) > tol * abs(cl[-1]))[0]
    return int(bad[-1] + 2) if len(bad) else 1


def cf_scatter(r):
    """Face-to-face scatter of the skin friction: rms deviation from a running median over
    9 faces, in % of Cf, on the suction and pressure side (5-95% chord)."""
    out = []
    for top in (True, False):
        s = (r["top_t"] == top) & (r["xt"] > 0.05) & (r["xt"] < 0.95)
        c = np.abs(r["cf"][s][np.argsort(r["xt"][s])])
        ref = np.array([np.median(c[max(0, i - 4):i + 5]) for i in range(len(c))])
        out.append(100 * np.sqrt(np.mean((c / ref - 1) ** 2)))
    return out


def gci(n, f, fs=1.25):
    """Grid convergence index from the fine, medium and coarse values f on meshes with n
    cells (Celik et al. 2008, J. Fluids Eng. 130, 078001). 2D: cell size ~ n^-1/2.
    The observed order p is used, but at most the formal order 2 (a smaller p gives a
    larger, more cautious GCI). Returns p, the extrapolated value, and the GCI (as a
    fraction) of the fine and the medium mesh."""
    h1, h2, h3 = np.asarray(n, float) ** -0.5
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = f[1] - f[0], f[2] - f[1]
    if e21 == 0 or e32 == 0:
        return dict(p=np.nan, ext=f[0], oscillating=False, gci_fine=0.0, gci_medium=abs(e21 / f[0]))
    s = np.sign(e32 / e21)  # -1: the values go up and down from mesh to mesh
    p = P_FORMAL
    for _ in range(500):  # fixed-point iteration of Celik's equations
        q = np.log((r21 ** p - s) / (r32 ** p - s))
        p_new = float(np.clip(abs(np.log(abs(e32 / e21)) + q) / np.log(r21), 0.05, 20))
        if abs(p_new - p) < 1e-10:
            break
        p = p_new
    pu = min(p, P_FORMAL)
    ea21 = abs(e21 / f[0])
    return dict(p=p, ext=(r21 ** pu * f[0] - f[1]) / (r21 ** pu - 1), oscillating=s < 0,
                gci_fine=fs * ea21 / (r21 ** pu - 1), gci_medium=fs * ea21 * r21 ** pu / (r21 ** pu - 1))


def field_difference(a, b):
    """Difference of sample a from sample b in the sampled 128 x 128 window, measured like
    the U-Net test error: sum |a - b| / sum |b|, pressure without its mean, only where both
    have flow. Returns (pressure, velocity) in %."""
    fluid = (a[2] == 0) & (b[2] == 0)
    pa, pb = a[3][fluid] - a[3][fluid].mean(), b[3][fluid] - b[3][fluid].mean()
    dv = np.abs(a[4] - b[4]) + np.abs(a[5] - b[5])
    return (100 * np.abs(pa - pb).sum() / np.abs(pb).sum(),
            100 * dv[fluid].sum() / (np.abs(b[4]) + np.abs(b[5]))[fluid].sum())


def scheme_test():
    print("A. nuTilda scheme test, 2000 iterations each. clipped: iterations in which OpenFOAM clipped negative "
          "nuTilda;\n   lowest: the most negative nuTilda clipped in the last 100 iterations (freestream 3e-5); "
          "res.: final residual;\n   Cl 0.5%: iteration from which Cl stays within 0.5% of its final value; "
          f"Cl, Cd: averages of the last {AVERAGE} iterations;\n   Cf scatter: suction / pressure side")
    print(f"   {'':24s}{'clipped':>13s}{'lowest':>10s}{'nuTilda res.':>14s}{'p res.':>9s}"
          f"{'Cl 0.5%':>9s}{'Cl':>9s}{'Cd':>8s}{'Cf scatter':>16s}")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, (case, runs) in zip(axes, SCHEMES.items()):
        print(f"   {case}")
        for k, (label, name) in enumerate(runs):
            r = load(name)
            if r is None:
                continue
            res, low = final_residuals(r["log"]), lowest_clipped(r["log"])
            scatter = cf_scatter(rans(f"{label}={RUNS[name][0]}"))
            clipped = f"{r['clipped']} of {len(r['cl_h'])}"
            print(f"   {label:24s}{clipped:>13s}{f'{low:.1e}' if low is not None else 'none':>10s}"
                  f"{res.get('nuTilda', np.nan):14.1e}{res.get('p', np.nan):9.1e}{settled(r['cl_h']):9d}"
                  f"{r['Cl']:+9.4f}{r['Cd']:8.4f}{f'{scatter[0]:.1f}% / {scatter[1]:.1f}%':>16s}")
            hist = residual_history(r["log"])
            ax.semilogy(np.arange(1, len(hist) + 1), hist, color=f"C{k}", lw=1.2, label=label)
        ax.axhline(1e-5, color="gray", ls=":", lw=1)
        ax.set(xlabel="iteration", title=f"nuTilda residual, {case}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("initial residual (dotted: stop limit 1e-5)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "scheme_test_residuals.png"), dpi=120)
    print(f"   saved {os.path.join(OUT, 'scheme_test_residuals.png')}")

    found = [f"{label}={RUNS[name][0]}" for label, name in SCHEMES["free air (LES case)"]
             if os.path.exists(RUNS[name][0])]
    if found:
        print("\n   Against the LES (free air, Re = 1M; Cl and Cd here are the last iteration's values):")
        subprocess.run([sys.executable, path("validate_les.py"), LES, "1M",
                        os.path.join(OUT, "scheme_test_les.png"), *found])


def grid_convergence():
    print(f"\nB. Grid convergence. Each mesh has about twice the cells of the next coarser one. Cl range: max - min "
          f"over the\n   last {STEADY_WINDOW} iterations; Cd flip: how much Cd changes from one iteration to the "
          "next; clipped: iterations with nuTilda\n   clipped (the last one); reversed: share of the wing surface "
          "where the flow runs backwards (separated)")
    fig, axes = plt.subplots(2, len(GCI_CASES), figsize=(12, 7.5))
    for col, (case, names) in enumerate(GCI_CASES.items()):
        print(f"   {case}")
        runs = [load(name) for name in names]
        if None in runs:
            continue
        print(f"   {'mesh':8s}{'cells':>9s}{'Cl':>10s}{'Cd':>9s}{'Cl 0.5% at':>11s}{'Cl range':>10s}"
              f"{'Cd flip':>9s}{'clipped':>13s}{'reversed':>10s}{'converged':>11s}")
        for level, r in zip(LEVELS, runs):
            clipped = f"{r['clipped']} ({r['last_clipped']})"
            print(f"   {level:8s}{r['cells']:9,d}{r['Cl']:+10.4f}{r['Cd']:9.4f}{settled(r['cl_h']):11d}"
                  f"{100 * r['cl_range']:9.3f}%{f'+-{100 * r['cd_osc']:.2f}%':>9s}{clipped:>13s}"
                  f"{100 * r['rev']:9.0f}%{'yes' if r['converged'] else 'NO':>11s}")
        n = [r["cells"] for r in runs]
        h = (np.array(n, float) / n[0]) ** -0.5  # cell size relative to the fine mesh
        steady = all(r["converged"] for r in runs)
        for row, key in enumerate(["Cl", "Cd"]):
            f = [r[key] for r in runs]
            hist = [r["cl_h" if key == "Cl" else "cd_h"][-SHOW:] for r in runs]
            ax = axes[row, col]
            ax.errorbar(h, f, yerr=[[a - x.min() for a, x in zip(f, hist)], [x.max() - a for a, x in zip(f, hist)]],
                        fmt="o-", color="C0", ms=7, lw=1.5, capsize=4,
                        label=f"meshes (fine, medium, coarse): mean of the last {AVERAGE} iterations,\n"
                              f"bar: range over the last {SHOW}")
            if steady:
                g = gci(n, f)
                note = "oscillating" if g["oscillating"] else "monotonic"
                print(f"   {key}: observed order p = {g['p']:.2f} ({note}); extrapolated {g['ext']:+.4f}; "
                      f"GCI fine {100 * g['gci_fine']:.2f}%, medium {100 * g['gci_medium']:.2f}%")
                if g["oscillating"] or not g["p"] >= 1:
                    print("      caution: the values do not converge steadily with the mesh, so this GCI is "
                          "only a rough indication")
                if g["p"] > 1.5 * P_FORMAL:
                    g3 = gci(n, f, fs=3.0)
                    print(f"      the observed order is far above the formal order 2: the coarse mesh is not yet in "
                          f"the asymptotic range.\n      With the cautious safety factor 3 (Roache): GCI fine "
                          f"{100 * g3['gci_fine']:.2f}%, medium {100 * g3['gci_medium']:.2f}%")
                ax.plot([0], [g["ext"]], "o", mfc="none", mec="C1", ms=9, mew=1.5, label="extrapolated to zero cell size")
                ax.plot([0, h[0]], [g["ext"], f[0]], ":", color="C1", lw=1)
            ax.set(xlim=(-0.1, h[-1] * 1.1), xlabel="cell size relative to the fine mesh", ylabel=key)
            ax.set_title(f"{key}, {case}" + ("" if steady else ": no steady solution"), fontsize=10)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="lower right")
        if not steady:
            bad = [level for level, r in zip(LEVELS, runs) if not r["converged"]]
            meshes = f"{', '.join(bad[:-1])} and {bad[-1]} meshes" if len(bad) > 1 else f"{bad[0]} mesh"
            print(f"   not converged on the {meshes}: the GCI needs converged solutions, so it is not computed")
        a = [r["z"]["a"] for r in runs]
        for level, k in [("medium", 1), ("coarse", 2)]:
            dp, dv = field_difference(a[k], a[0])
            print(f"   flow field, {level} vs fine mesh: pressure {dp:.2f}%, velocity {dv:.2f}% "
                  f"(measured like the U-Net test error{'' if steady else '; last iteration of unsteady runs'})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "grid_convergence.png"), dpi=120)
    print(f"   saved {os.path.join(OUT, 'grid_convergence.png')}")


def domain():
    print(f"\nC. Domain size at the largest ride height (NACA 23012, 4 deg, h/c 0.6, 30 m/s; Cl, Cd: averages of the "
          f"last {AVERAGE} iterations)")
    runs = [load(name) for _, name in DOMAIN]
    if None in runs:
        return
    for (label, _), r in zip(DOMAIN, runs):
        print(f"   {label}: Cl {r['Cl']:+.4f}  Cd {r['Cd']:.4f}  cells {r['cells']:,}")
    for key in ["Cl", "Cd"]:
        a, b = runs[0][key], runs[1][key]
        print(f"   {key} changes by {100 * abs(a - b) / abs(b):.2f}% from 20 to 40 chords")


def mesh_quality():
    print("\nD. Mesh quality of the grid convergence meshes")
    for case, names in GCI_CASES.items():
        for level, name in zip(LEVELS, names):
            npz, worker = RUNS[name]
            if os.path.exists(path("work", worker, "airfoil.msh")):
                print(f"   {case}, {level} mesh:")
                mesh_report(path("work", worker), npz if os.path.exists(npz) else None)


def run_time():
    print("\nE. Run time of the round-3 runs (4 at a time). start-up: until the first iteration had finished "
          "(reading the mesh,\n   wall distance, and finding the 16,384 sample points); solver: the 2000 iterations")
    print(f"   {'run':22s}{'cells':>9s}{'start-up':>10s}{'solver':>9s}{'per iteration':>15s}{'start-up share':>16s}")
    shares = []
    for name, (npz, worker) in RUNS.items():
        log = path("work", worker, "log.simpleFoam")
        if name in ROUND2 or not os.path.exists(log) or not os.path.exists(npz):
            continue
        first, last = execution_times(log)
        n = len(residual_history(log, "p"))
        shares.append(first / last)
        print(f"   {name:22s}{int(np.load(npz)['cells']):9,d}{first:9.0f}s{last - first:8.0f}s"
              f"{(last - first) / max(n - 1, 1):14.3f}s{100 * first / last:15.0f}%")
    if shares:
        print(f"   the start-up took {100 * min(shares):.0f}-{100 * max(shares):.0f}% of the run time")


def main():
    sys.stdout.reconfigure(line_buffering=True)  # keep our lines in order with the subprocess output
    scheme_test()
    grid_convergence()
    domain()
    mesh_quality()
    run_time()


if __name__ == "__main__":
    main()
