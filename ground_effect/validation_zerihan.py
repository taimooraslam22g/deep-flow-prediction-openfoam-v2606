#!/usr/bin/env python3
"""Compare our RANS runs of the Tyrrell 026 wing (validation_zerihan.sh) with the wind tunnel
tests and the 2D RANS results of Zerihan (2001).

Usage (on Windows, with the project's Python):
    .venv\\Scripts\\python.exe ground_effect\\validation_zerihan.py

Reference data, all for the single element wing at the reference incidence (quoted 1 deg),
free transition, U = 30 m/s, Re = 0.43-0.46 million, moving ground:
  - Table 15 of the thesis: sectional downforce at the wing centre at eight heights.
    'Experiment' is the integral of the 45 surface pressure tappings at the wing centre
    (straight lines between them). Zerihan calls it an estimate for the trends, and
    expects it to be low because the tappings miss the suction spike near the leading edge.
    'CFL3D S-A' is his own 2D RANS result (Spalart-Allmaras, fully turbulent, y+ about 1,
    moving ground, 25,000-30,000 grid points). Its 'freestream' row: the tests were made
    3.36 chords above the moving ground (750 mm, the middle of the tunnel); the thesis does
    not say where the ground was in the CFD.
  - the tapping positions on the two surfaces (Table 5), used to integrate our pressures
    the same way as the experiment ('at tappings').
  - where the boundary layer separates on the suction surface, from oil flow
    visualisation (Table 9 and the text of Chapters 4 and 6).
Our runs: three meshes per height (coarse, medium = the default, fine). The grid
convergence index (GCI, Celik et al. 2008) of the fine mesh is the numerical uncertainty of
our downforce. If data/validation_zerihan_raw/ exists (the first run, with the 87 table
points as they are), a third table shows what smoothing the coordinates changed.

Prints three tables and writes to results/:
    validation_zerihan.png     downforce and separation point against the ride height
    validation_zerihan_cp.png  our surface pressures; table points against smoothed shape
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")  # write to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from validate_les import to_upright, top_side  # noqa: E402
from verification_report import field_difference, gci  # noqa: E402

AIRFOIL = os.path.join(ROOT, "data", "airfoils", "tyrrell026.dat")
RUNS = os.path.join(HERE, "data", "validation_zerihan")
RAW = os.path.join(HERE, "data", "validation_zerihan_raw")
OUT = os.path.join(ROOT, "results")
CHORD_MM = 223.4
FREESTREAM = 3.36  # h/c of the 'freestream' tests (thesis section 2.7)

# Table 15: h/c, experiment, CFL3D S-A. The first row is the 'freestream' case.
TABLE15 = [(FREESTREAM, 0.766, 0.885), (0.671, 0.902, 1.052), (0.448, 1.009, 1.174), (0.313, 1.145, 1.320),
           (0.224, 1.286, 1.475), (0.179, 1.362, 1.565), (0.134, 1.385, 1.604), (0.090, 1.371, 1.539)]
# Table 5: chordwise tappings at the wing centre, mm from the leading edge
TAPS_SUCTION = [0, 2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 56, 65, 75, 85, 95, 110, 130, 150, 160, 170, 180,
                190, 200, 211, 222]
TAPS_PRESSURE = [0, 2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 56, 70, 90, 110, 125, 140, 150, 160, 170, 185, 200, 222]
# Oil flow: start of the separated region on the suction surface, x/c. 0.224: 'a very small
# region (x/c about 0.99) begins to appear' (Chapter 4); 0.134: 0.91 (Table 9, Chapter 6);
# 0.090: about 0.8 and 0.067: about 0.7 (Chapter 4; Table 9: 0.80 and 0.71).
OIL_FLOW = [(0.224, 0.99), (0.134, 0.91), (0.090, 0.80), (0.067, 0.71)]
HEIGHTS = [None, FREESTREAM, 0.671, 0.448, 0.313, 0.224, 0.179, 0.134, 0.112, 0.090, 0.067]  # None: free air
MESHES = ["coarse", "medium", "fine"]


def run_name(mesh, h):
    return f"{mesh}_{'free' if h is None else f'h{round(h * 1000):04d}'}"


def label(h):
    return "free" if h is None else f"{h:.2f}" if h > 1 else f"{h:.3f}"


def find(folder, name):
    """The sample of one run (converged or not), or None."""
    files = glob.glob(os.path.join(folder, name, "tyrrell026*.npz")) + \
        glob.glob(os.path.join(folder, name, "not_converged", "tyrrell026*.npz"))
    return np.load(files[0]) if files else None


def surfaces(z):
    """Cp and Cf along the suction and pressure surfaces, in the frame of the upright wing
    (x/c from the leading edge along the chord). Returns {side: (x, cp)}, {side: (x, cf)}."""
    alpha, U = float(z["alpha"]), float(z["U"])
    q = 0.5 * U * U
    is_top = top_side(AIRFOIL)  # upright: the suction surface is on top
    xp, yp = to_upright(z["surf_xy_p"], alpha)
    xt, yt = to_upright(z["surf_xy_tau"], alpha)
    a = np.radians(alpha)
    tau = z["surf_tau"]
    along = tau[:, 0] * np.cos(a) + tau[:, 1] * np.sin(a)  # along the chord, leading -> trailing edge
    ref = (xt > 0.1) & (xt < 0.4)
    cf = np.sign(np.median(along[ref])) * along / q  # > 0: attached flow (sign convention of OpenFOAM)
    cp, cfs = {}, {}
    for side, top in [("suction", True), ("pressure", False)]:
        s = is_top(xp, yp) == top
        o = np.argsort(xp[s])
        cp[side] = (xp[s][o], z["surf_p"][s][o] / q)
        s = is_top(xt, yt) == top
        o = np.argsort(xt[s])
        cfs[side] = (xt[s][o], cf[s][o])
    return cp, cfs


def downforce_from_pressures(cp, taps=None):
    """Sectional downforce (normal to the chord) from the surface pressures: the integral of
    Cp(pressure surface) - Cp(suction surface) over x/c. With taps, Cp is first sampled at
    the tapping positions and integrated with straight lines between them, like the
    experiment."""
    total = 0.0
    for side, sign in [("pressure", 1.0), ("suction", -1.0)]:
        x, c = cp[side]
        if taps is not None:
            xs = np.array(taps[side]) / CHORD_MM
            x, c = xs, np.interp(xs, x, c)
        total += sign * np.trapezoid(c, x)
    return total


def separation(cfs):
    """Start of trailing-edge separation on the suction surface (x/c), or NaN if attached:
    where the region of backward flow next to the wall begins that reaches past 95% chord.
    The last 2% of the chord are left out: right at the trailing edge, a small corner eddy
    can turn the flow forwards again."""
    x, cf = cfs["suction"]
    s = (x > 0.3) & (x < 0.98)
    x, back = x[s], cf[s] < 0
    if not back.any() or x[np.where(back)[0][-1]] < 0.95:
        return np.nan
    i = np.where(back)[0][-1]
    while i > 0 and back[i - 1]:
        i -= 1
    return float(x[i])


def load_runs(folder):
    """{(mesh, h): summary of the run} for all runs found in folder."""
    taps = {"suction": TAPS_SUCTION, "pressure": TAPS_PRESSURE}
    rows = {}
    for mesh in MESHES:
        for h in HEIGHTS:
            z = find(folder, run_name(mesh, h))
            if z is None:
                continue
            cp, cfs = surfaces(z)
            rows[(mesh, h)] = dict(force=-float(z["Cl"]), full=downforce_from_pressures(cp),
                                   taps=downforce_from_pressures(cp, taps), sep=separation(cfs),
                                   converged=bool(z["converged"]) if "converged" in z.files else True,
                                   cells=int(z["cells"]), a=z["a"], cp=cp)
    return rows


def mesh_study(rows, h):
    """GCI of the downforce at height h, if all three meshes have converged; else None."""
    runs = [rows.get((m, h)) for m in ("fine", "medium", "coarse")]
    if any(r is None or not r["converged"] for r in runs):
        return None
    return gci([r["cells"] for r in runs], [r["force"] for r in runs])


def num(v, width, spec=".3f", suffix=""):
    """v right-aligned in width characters (spec: format without the width, '+' first if
    wanted), or '-' if there is no value."""
    if v is None or not np.isfinite(v):
        return f"{'-':>{width}s}"
    sign, spec = ("+", spec[1:]) if spec.startswith("+") else ("", spec)
    return f"{v:{sign}{width - len(suffix)}{spec}}{suffix}"


def main():
    rows = load_runs(RUNS)
    if not any(m == "fine" for m, _ in rows):
        sys.exit(f"no runs found in {RUNS}: run validation_zerihan.sh in the container first")
    raw = load_runs(RAW) if os.path.isdir(RAW) else {}
    ref = {h: (e, s) for h, e, s in TABLE15}
    oil = dict(OIL_FLOW)
    studies = {h: mesh_study(rows, h) for h in HEIGHTS}

    print("1. Sectional downforce of the Tyrrell 026 wing at the reference incidence (quoted 1 deg, true 3.45 deg),")
    print("   Re 462,000. Ours: fine mesh, force coefficients, with the GCI of the fine mesh; 'at tappings': our")
    print("   pressures sampled at the experiment's tappings and integrated like the experiment. 'vs expt' compares")
    print("   'at tappings' with the experiment, 'vs CFL3D' our forces with CFL3D. Separation: x/c on the suction")
    print("   surface. Experiment and CFL3D: Zerihan (2001), Table 15; its freestream row is shown at h/c 3.36.")
    print(f"{'h/c':>6s}{'experiment':>11s}{'CFL3D S-A':>10s}{'ours':>8s}{'GCI':>7s}{'at tappings':>12s}"
          f"{'vs expt':>9s}{'vs CFL3D':>9s}{'separation':>11s}{'oil flow':>9s}{'converged c/m/f':>17s}")
    for h in HEIGHTS:
        f = rows.get(("fine", h))
        if f is None:
            continue
        e, s = ref.get(h, (np.nan, np.nan))
        st = studies[h]
        conv = "/".join("-" if (m, h) not in rows else "yes" if rows[(m, h)]["converged"] else "NO" for m in MESHES)
        print(f"{label(h):>6s}{num(e, 11)}{num(s, 10)}{num(f['force'], 8)}"
              f"{num(100 * st['gci_fine'] if st else None, 7, '.1f', '%')}{num(f['taps'], 12)}"
              f"{num(100 * (f['taps'] / e - 1), 9, '+.1f', '%')}{num(100 * (f['force'] / s - 1), 9, '+.1f', '%')}"
              f"{num(f['sep'], 11, '.2f')}{num(oil.get(h, np.nan), 9, '.2f')}{conv:>17s}")
    fine = {h: r for (m, h), r in rows.items() if m == "fine"}
    ground = {h: r for h, r in fine.items() if h is not None and h < 1}
    if ground:
        hmax = max(ground, key=lambda h: ground[h]["force"])
        print(f"\nlargest downforce (fine mesh): {ground[hmax]['force']:.3f} at h/c {hmax:.3f}. Table 15: experiment 1.385 "
              f"and CFL3D 1.604, both at h/c 0.134. 3D wing on the balance: CL 1.72 at h/c 0.082")
    loss = [100 * (1 - r["taps"] / r["full"]) for r in fine.values()]
    full = [100 * (r["full"] / r["force"] - 1) for r in fine.values()]
    print(f"integrating at the tappings lowers our downforce by {min(loss):.1f} to {max(loss):.1f}%; the integral of all "
          f"our surface pressures is {min(full):+.1f} to {max(full):+.1f}% off the force coefficients (friction, "
          f"direction)")

    print("\n2. Mesh study: downforce on the three meshes (about 2x the cells each), observed order p, extrapolated")
    print("   value and GCI (Celik et al. 2008; p at most 2 in the GCI), and how much the sampled flow field of the")
    print("   medium mesh differs from the fine mesh (measured like the U-Net error)")
    print(f"{'h/c':>6s}{'cells: coarse':>14s}{'medium':>8s}{'fine':>8s}{'downforce: coarse':>18s}{'medium':>8s}"
          f"{'fine':>8s}{'p':>6s}{'extrap.':>9s}{'GCI fine':>9s}{'GCI medium':>11s}{'field p':>9s}{'field v':>8s}")
    for h in HEIGHTS:
        r = [rows.get((m, h)) for m in MESHES]
        if r[2] is None:
            continue
        st = studies[h]
        dp, dv = field_difference(r[1]["a"], r[2]["a"]) if r[1] is not None else (np.nan, np.nan)
        osc = st is not None and st["oscillating"]
        print(f"{label(h):>6s}" + "".join(num(x["cells"] if x else None, w, "d") for x, w in zip(r, (14, 8, 8)))
              + "".join(num(x["force"] if x else None, w) for x, w in zip(r, (18, 8, 8)))
              + num(st["p"] if st else None, 6, ".1f") + num(st["ext"] if st else None, 9)
              + num(100 * st["gci_fine"] if st else None, 9, ".2f", "%")
              + num(100 * st["gci_medium"] if st else None, 11, ".2f", "%")
              + num(dp, 9, ".2f", "%") + num(dv, 8, ".2f", "%") + ("  (oscillating)" if osc else ""))

    if raw:
        print("\n3. Effect of smoothing the coordinates (fine mesh): the first run used the 87 table points as they are")
        print(f"{'h/c':>6s}{'downforce: table points':>24s}{'smoothed':>10s}{'change':>8s}"
              f"{'separation: table points':>26s}{'smoothed':>10s}")
        for h in HEIGHTS:
            a, b = raw.get(("fine", h)), rows.get(("fine", h))
            if a is None or b is None:
                continue
            print(f"{label(h):>6s}{num(a['force'], 24)}{num(b['force'], 10)}{num(100 * (b['force'] / a['force'] - 1), 8, '+.1f', '%')}"
                  f"{num(a['sep'], 26, '.2f')}{num(b['sep'], 10, '.2f')}")

    figures(rows, raw, studies)


def figures(rows, raw, studies):
    def xpos(h):  # the ride heights in ground effect, then h/c 3.36 and free air at the right
        return 0.92 if h is None else 0.80 if h > 1 else h

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    near = [(h, e, s) for h, e, s in TABLE15 if h < 1]
    ax.plot([h for h, _, _ in near], [s for _, _, s in near], "--", color="gray", lw=1.2,
            label="CFL3D, Spalart-Allmaras (Zerihan)")
    ax.plot(xpos(FREESTREAM), TABLE15[0][2], "o", color="gray", ms=5, mfc="none")
    ax.plot([xpos(h) for h, _, _ in TABLE15], [e for _, e, _ in TABLE15], "ks", mfc="none", ms=7,
            label="experiment, integrated tappings at the wing centre")
    for mesh, style in [("fine", "-"), ("medium", ":")]:
        pts = sorted((h, r) for (m, h), r in rows.items() if m == mesh and h is not None and h < 1)
        far = [(h, r) for (m, h), r in rows.items() if m == mesh and (h is None or h > 1)]
        if not pts:
            continue
        if mesh == "fine":
            err = [0.0 if studies.get(h) is None else studies[h]["gci_fine"] * r["force"] for h, r in pts]
            ax.errorbar([h for h, _ in pts], [r["force"] for _, r in pts], yerr=err, fmt="-o", color="C0", ms=4,
                        capsize=3, label="ours, fine mesh (bars: GCI)")
            ax.plot([xpos(h) for h, _ in far], [r["force"] for _, r in far], "o", color="C0", ms=5)
            ax.plot([h for h, _ in pts], [r["taps"] for _, r in pts], "-^", color="C1", ms=5,
                    label="ours, fine mesh, integrated at the tappings")
            ax.plot([xpos(h) for h, _ in far], [r["taps"] for _, r in far], "^", color="C1", ms=5)
            bad = [(xpos(h), r["force"]) for h, r in pts + far if not r["converged"]]
            if bad:
                ax.plot(*zip(*bad), "x", color="C3", ms=10, mew=2, label="not converged")
        else:
            ax.plot([h for h, _ in pts], [r["force"] for _, r in pts], style, color="C0", lw=1.2,
                    label=f"ours, {mesh} mesh")
    ax.axvline(0.75, color="k", lw=0.6)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.80, 0.92],
                  ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "3.36", "free\nair"])
    ax.set(xlabel="ride height h/c", ylabel="sectional downforce coefficient", xlim=(0, 0.98),
           title="Tyrrell 026 wing, quoted incidence 1 deg, Re 462,000")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    ax.plot(*zip(*OIL_FLOW), "ks", mfc="none", ms=7, label="experiment, oil flow")
    for mesh, style, marker in [("fine", "-", "o"), ("medium", ":", "o"), ("coarse", "-.", ".")]:
        pts = sorted((h, r["sep"]) for (m, h), r in rows.items() if m == mesh and h is not None and h < 1)
        if pts:
            ax.plot(*zip(*pts), style, color="C0", marker=marker, ms=4, label=f"ours, {mesh} mesh")
    pts = sorted((h, r["sep"]) for (m, h), r in raw.items() if m == "fine" and h is not None and h < 1)
    if pts:
        ax.plot(*zip(*pts), "-", color="gray", lw=1, marker="o", ms=3, label="ours, fine mesh, table points (first run)")
    ax.set(xlabel="ride height h/c", ylabel="separation on the suction surface, x/c", xlim=(0, 0.35),
           ylim=(0.5, 1.02), title="Start of trailing-edge separation (none plotted: attached)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "validation_zerihan.png"), dpi=120)

    # figure 2: our surface pressures at a selection of heights; table points vs smoothed
    fig, axes = plt.subplots(1, 2 if raw else 1, figsize=(15 if raw else 8, 5.5), squeeze=False)
    ax = axes[0][0]
    for k, h in enumerate([FREESTREAM, 0.448, 0.224, 0.134, 0.090, 0.067]):
        r = rows.get(("fine", h))
        if r is None:
            continue
        for side, style in [("suction", "-"), ("pressure", "--")]:
            ax.plot(*r["cp"][side], style, color=f"C{k}", lw=1.2, label=f"h/c {label(h)}" if side == "suction" else None)
    ax.plot(np.array(TAPS_SUCTION) / CHORD_MM, np.full(len(TAPS_SUCTION), 1.15), "|", color="gray", ms=8,
            label="tappings of the experiment, suction surface")
    ax.invert_yaxis()
    ax.set(xlabel="x/c", ylabel="Cp", xlim=(0, 1),
           title="Ours, fine mesh (solid: suction surface, dashed: pressure surface)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    if raw:
        ax = axes[0][1]
        h = 0.134
        for rr, color, lw, name in [(raw, "gray", 2.5, "87 table points (first run)"),
                                    (rows, "C3", 1.0, "smoothed coordinates")]:
            r = rr.get(("fine", h))
            if r is None:
                continue
            for side, style in [("suction", "-"), ("pressure", "--")]:
                ax.plot(*r["cp"][side], style, color=color, lw=lw, label=name if side == "suction" else None)
        ax.invert_yaxis()
        ax.set(xlabel="x/c", ylabel="Cp", xlim=(0, 1), title=f"Wing shape from the table, h/c {h}, fine mesh")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "validation_zerihan_cp.png"), dpi=120)
    print(f"\nsaved {os.path.join(OUT, 'validation_zerihan.png')} and {os.path.join(OUT, 'validation_zerihan_cp.png')}")


if __name__ == "__main__":
    main()
