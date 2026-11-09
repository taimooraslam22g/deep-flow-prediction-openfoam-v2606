#!/usr/bin/env python3
"""Tables and figures for the numerical verification (mesh study, round 3), after
verification.sh has run in the container.

Usage (on Windows, with the project's Python):
    .venv\\Scripts\\python.exe ground_effect\\verification_report.py

Prints:
    A. nuTilda scheme test: clipping, residuals, forces, Cf scatter, accuracy vs the LES
    B. grid convergence index (GCI) of Cl, Cd and the sampled flow field, two cases
    C. domain size at the largest ride height
    D. y+ and cell sizes of the grid convergence meshes
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
from groundGen import bounded_iterations, final_residuals  # noqa: E402
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
    """Sample file of a verification.sh run."""
    return path("data", "verification", name, npz)


# (label, sample, work folder). The linearUpwind runs are from round 2 (mesh_study.sh).
SCHEMES = {
    "free air (LES case)": [("linearUpwind (round 2)", path("data", "validation", "r2_d20", FREE), "worker22"),
                            ("cellLimited", v("free_cellLimited", FREE), "worker31"),
                            ("limitedLinear", v("free_limitedLinear", FREE), "worker32")],
    "ground effect (NACA 23012)": [("linearUpwind (round 2)", path("data", "tests", "r2_d20", GROUND), "worker24"),
                                   ("cellLimited", v("ground_medium", GROUND), "worker33"),
                                   ("limitedLinear", v("ground_limitedLinear", GROUND), "worker34")]}
# fine, medium, coarse
GCI_CASES = {
    "NACA 23012, 4 deg, h/c 0.2, 30 m/s": [(v("ground_fine", GROUND), "worker36"),
                                           (v("ground_medium", GROUND), "worker33"),
                                           (v("ground_coarse", GROUND), "worker35")],
    "S1223, 6 deg, h/c 0.1, 50 m/s": [(v("hard_fine", HARD), "worker39"),
                                      (v("hard_medium", HARD), "worker38"),
                                      (v("hard_coarse", HARD), "worker37")]}
DOMAIN = [("20 chords", v("high_d20", HIGH)), ("40 chords", v("high_d40", HIGH))]
LEVELS = ["fine", "medium", "coarse"]
P_FORMAL = 2.0  # formal order of the discretisation


def log_of(worker):
    return path("work", worker, "log.simpleFoam")


def residual_history(log, field="nuTilda"):
    """Initial residual of one field at every iteration, from the simpleFoam log."""
    key = f"Solving for {field}, Initial residual = "
    with open(log) as f:
        return np.array([float(line.split(key)[1].split(",")[0]) for line in f if key in line])


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


def missing(files):
    gone = [f for f in files if not os.path.exists(f)]
    for f in gone:
        print(f"   missing: {os.path.relpath(f, ROOT)}")
    return bool(gone)


def scheme_test():
    print("A. nuTilda scheme test (clipped: iterations in which negative nuTilda was clipped; "
          "res.: final residual;\n   Cl 0.5%: iteration from which Cl stays within 0.5% of its final value; "
          "Cf scatter: suction / pressure side)")
    print(f"   {'':24s}{'clipped':>13s}{'nuTilda res.':>14s}{'p res.':>9s}{'iterations':>11s}"
          f"{'Cl 0.5%':>9s}{'Cl':>9s}{'Cd':>8s}{'Cf scatter':>16s}")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, (case, runs) in zip(axes, SCHEMES.items()):
        print(f"   {case}")
        for k, (label, npz, worker) in enumerate(runs):
            if missing([npz, log_of(worker)]):
                continue
            z, log = np.load(npz), log_of(worker)
            res, n = final_residuals(log), int(z["iterations"])
            scatter = cf_scatter(rans(f"{label}={npz}"))
            clipped = f"{bounded_iterations(log)[0]} of {n}"
            print(f"   {label:24s}{clipped:>13s}{res.get('nuTilda', np.nan):14.1e}{res.get('p', np.nan):9.1e}"
                  f"{n:11d}{settled(z['Cl_history']):9d}{float(z['Cl']):+9.4f}{float(z['Cd']):8.4f}"
                  f"{f'{scatter[0]:.1f}% / {scatter[1]:.1f}%':>16s}")
            hist = residual_history(log)
            ax.semilogy(np.arange(1, len(hist) + 1), hist, color=f"C{k}", lw=1.2, label=label)
        ax.axhline(1e-5, color="gray", ls=":", lw=1)
        ax.set(xlabel="iteration", title=f"nuTilda residual, {case}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("initial residual (dotted: stop limit 1e-5)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "scheme_test_residuals.png"), dpi=120)
    print(f"   saved {os.path.join(OUT, 'scheme_test_residuals.png')}")

    runs = SCHEMES["free air (LES case)"]
    found = [f"{label}={npz}" for label, npz, _ in runs if os.path.exists(npz)]
    if found:
        print("\n   Against the LES (free air, Re = 1M):")
        subprocess.run([sys.executable, path("validate_les.py"), LES, "1M",
                        os.path.join(OUT, "scheme_test_les.png"), *found])


def grid_convergence():
    print("\nB. Grid convergence (each mesh has about twice the cells of the next coarser one)")
    fig, axes = plt.subplots(2, len(GCI_CASES), figsize=(12, 7.5))
    for col, (case, runs) in enumerate(GCI_CASES.items()):
        print(f"   {case}")
        if missing([npz for npz, _ in runs]):
            continue
        z = [np.load(npz) for npz, _ in runs]
        n = [int(s["cells"]) for s in z]
        print(f"   {'mesh':8s}{'cells':>9s}{'Cl':>10s}{'Cd':>9s}{'iterations':>11s}{'Cl 0.5% at':>11s}{'clipped':>9s}")
        for level, s, cells in zip(LEVELS, z, n):
            print(f"   {level:8s}{cells:9,d}{float(s['Cl']):+10.4f}{float(s['Cd']):9.4f}{int(s['iterations']):11d}"
                  f"{settled(s['Cl_history']):11d}{int(s['bounded']):9d}")
        h = (np.array(n, float) / n[0]) ** -0.5  # cell size relative to the fine mesh
        for row, key in enumerate(["Cl", "Cd"]):
            f = [float(s[key]) for s in z]
            g = gci(n, f)
            note = "oscillating" if g["oscillating"] else "monotonic"
            print(f"   {key}: observed order p = {g['p']:.2f} ({note}); extrapolated {g['ext']:+.4f}; "
                  f"GCI fine {100 * g['gci_fine']:.2f}%, medium {100 * g['gci_medium']:.2f}%")
            if g["oscillating"] or not g["p"] >= 1:
                print("      caution: the values do not converge steadily with the mesh, so this GCI is "
                      "only a rough indication")
            ax = axes[row, col]
            ax.plot(h, f, "o-", color="C0", ms=7, lw=1.5, label="meshes (fine, medium, coarse)")
            ax.plot([0], [g["ext"]], "o", mfc="none", mec="C1", ms=9, mew=1.5, label="extrapolated to zero cell size")
            ax.plot([0, h[0]], [g["ext"], f[0]], ":", color="C1", lw=1)
            ax.set(xlim=(-0.1, h[-1] * 1.1), xlabel="cell size relative to the fine mesh", ylabel=key)
            ax.set_title(f"{key}, {case}", fontsize=10)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        a = [s["a"] for s in z]
        for level, k in [("medium", 1), ("coarse", 2)]:
            dp, dv = field_difference(a[k], a[0])
            print(f"   flow field, {level} vs fine mesh: pressure {dp:.2f}%, velocity {dv:.2f}% "
                  "(measured like the U-Net test error)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "grid_convergence.png"), dpi=120)
    print(f"   saved {os.path.join(OUT, 'grid_convergence.png')}")


def domain():
    print("\nC. Domain size at the largest ride height (NACA 23012, 4 deg, h/c 0.6, 30 m/s)")
    if missing([npz for _, npz in DOMAIN]):
        return
    z = [np.load(npz) for _, npz in DOMAIN]
    for (label, _), s in zip(DOMAIN, z):
        print(f"   {label}: Cl {float(s['Cl']):+.4f}  Cd {float(s['Cd']):.4f}  cells {int(s['cells']):,}")
    for key in ["Cl", "Cd"]:
        a, b = float(z[0][key]), float(z[1][key])
        print(f"   {key} changes by {100 * abs(a - b) / abs(b):.2f}% from 20 to 40 chords")


def mesh_quality():
    print("\nD. Mesh quality of the grid convergence meshes")
    for case, runs in GCI_CASES.items():
        for level, (npz, worker) in zip(LEVELS, runs):
            if os.path.exists(path("work", worker, "airfoil.msh")):
                print(f"   {case}, {level} mesh:")
                mesh_report(path("work", worker), npz if os.path.exists(npz) else None)


def main():
    sys.stdout.reconfigure(line_buffering=True)  # keep our lines in order with the subprocess output
    scheme_test()
    grid_convergence()
    domain()
    mesh_quality()


if __name__ == "__main__":
    main()
