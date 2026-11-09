#!/usr/bin/env python3
"""Compare free-air RANS runs of the NACA 4412 with the LES boundary-layer data of
Vinuesa et al. (naca4412.mat from the WING database, shared by Prof. Vinuesa).

The LES file gives, at 50 positions along each side of the wing (15% to 98% of the chord),
the pressure and velocity profiles and the skin friction Cf. This script compares the
surface pressure coefficient Cp and Cf with runs of `groundGen.py --free-air`.

Usage (on Windows, with the project's Python; needs scipy):
    python ground_effect/validate_les.py <naca4412.mat> <Re> <out.png> run1.npz [run2.npz ...]
Each run can be given as "label=path.npz" to set its name in the plot and table.

<Re> picks the LES case: 100k, 200k, 400k or 1M. In our setup the chord is 1 m and
nu = 1e-5 m^2/s, so U = 10 m/s is Re = 1M.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")  # write to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from groundGen import load_airfoil  # noqa: E402

LES_KEYS = {"100k": "1n", "200k": "2n", "400k": "4n", "1M": "10n"}
AIRFOIL_DIR = os.path.join(HERE, "..", "Deep-Flow-Prediction", "data", "airfoil_database")


def les_side(mat, key):
    """x/c, Cp and Cf along one side of the wing, from the LES statistics."""
    st = mat[key]
    x = np.array([float(s["xa"]) for s in st])
    # The stored pressure has an arbitrary offset. Outside the boundary layer the total
    # pressure P + |U|^2/2 is constant and equals P_inf + 1/2 (U_inf = 1), which gives P_inf.
    p0 = np.mean([s["P"][-1] + 0.5 * (s["U"][-1] ** 2 + s["V"][-1] ** 2) for s in st])
    cp = np.array([2 * (s["P"][0] - (p0 - 0.5)) for s in st])
    cf = np.array([float(s["Cf"]) for s in st])
    return x, cp, cf


def rans(spec):
    """Cp and Cf on both sides of the wing from a groundGen.py --free-air sample."""
    label, path = spec.split("=", 1) if "=" in spec else (None, spec)
    z = np.load(path)
    if "surf_p" not in z.files or "surf_tau" not in z.files:
        sys.exit(f"{path} has no surface data")
    a, U = np.radians(float(z["alpha"])), float(z["U"])
    q = 0.5 * U * U  # dynamic pressure (kinematic, as in OpenFOAM)

    def upright(xy):
        """Undo groundGen.place_airfoil: rotate back about the quarter chord, flip again."""
        dx, y = xy[:, 0] - 0.25, xy[:, 1]
        return 0.25 + dx * np.cos(a) + y * np.sin(a), dx * np.sin(a) - y * np.cos(a)

    # camber line of the upright airfoil, to tell the two sides apart
    name = str(z["airfoil"])
    file = os.path.join(AIRFOIL_DIR, name + ".dat")
    ar = load_airfoil(file if os.path.isfile(file) else os.path.join(AIRFOIL_DIR, name))
    le = np.argmin(ar[:, 0])
    up, lo = ar[:le + 1][::-1], ar[le:]  # both from the leading to the trailing edge

    def is_top(x, y):
        return y > 0.5 * (np.interp(x, up[:, 0], up[:, 1]) + np.interp(x, lo[:, 0], lo[:, 1]))

    xp, yp = upright(z["surf_xy_p"])
    xt, yt = upright(z["surf_xy_tau"])
    tau = z["surf_tau"]
    tx = tau[:, 0] * np.cos(a) + tau[:, 1] * np.sin(a)  # x component in the upright frame
    cf = np.sign(tx) * np.hypot(tau[:, 0], tau[:, 1]) / q
    top_t = is_top(xt, yt)
    # OpenFOAM's sign convention for the wall shear stress can be the opposite of the usual
    # one. Attached flow must have Cf > 0: check on the pressure side, 30-60% chord.
    ref = ~top_t & (xt > 0.3) & (xt < 0.6)
    if np.median(cf[ref]) < 0:
        cf = -cf
    if label is None:
        label = f"RANS, nuTilda {float(z['nutilda']):g}"
        if "bl" in z.files:
            label += f", {'BL' if bool(z['bl']) else 'original'} mesh, {float(z['domain']):g}c"
    return dict(label=label, Cl=abs(float(z["Cl"])), Cd=float(z["Cd"]),
                xp=xp, cp=z["surf_p"] / q, top_p=is_top(xp, yp), xt=xt, cf=cf, top_t=top_t)


def at(x_les, x, v):
    """RANS values interpolated to the LES positions."""
    order = np.argsort(x)
    return np.interp(x_les, x[order], v[order])


def main():
    if len(sys.argv) < 5 or sys.argv[2] not in LES_KEYS:
        sys.exit(__doc__)
    mat_path, re_key, out, runs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    mat = loadmat(mat_path, simplify_cells=True)
    sides = {"suction side": les_side(mat, "top" + LES_KEYS[re_key]),
             "pressure side": les_side(mat, "bot" + LES_KEYS[re_key])}
    results = [rans(r) for r in runs]

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for side, (xl, cpl, cfl) in sides.items():
        mk = "o" if side == "suction side" else "s"
        axes[0].plot(xl, cpl, mk, mfc="none", color="k", ms=4, label=f"LES, {side}")
        ax = axes[1] if side == "suction side" else axes[2]
        ax.plot(xl, cfl, mk, mfc="none", color="k", ms=4, label="LES")
    print(f"Mean absolute difference from the LES at Re = {re_key}, 15-98% chord:")
    width = max(34, max(len(r["label"]) for r in results) + 2)
    print(f"{'run':{width}s}{'|Cl|':>8s}{'Cd':>8s}{'Cp suct':>9s}{'Cp pres':>9s}{'Cf suct':>9s}{'Cf pres':>9s}")
    for k, r in enumerate(results):
        color = f"C{k}"
        errs = []
        for side, (xl, cpl, cfl) in sides.items():
            top = side == "suction side"
            sel_p, sel_t = r["top_p"] == top, r["top_t"] == top
            order = np.argsort(r["xp"][sel_p])
            axes[0].plot(r["xp"][sel_p][order], r["cp"][sel_p][order], "-" if top else "--", color=color,
                         label=f"{r['label']}, {side}")
            order = np.argsort(r["xt"][sel_t])
            (axes[1] if top else axes[2]).plot(r["xt"][sel_t][order], r["cf"][sel_t][order], "-",
                                               color=color, label=r["label"])
            errs.append(np.mean(np.abs(at(xl, r["xp"][sel_p], r["cp"][sel_p]) - cpl)))
            errs.append(np.mean(np.abs(at(xl, r["xt"][sel_t], r["cf"][sel_t]) - cfl)))
        print(f"{r['label']:{width}s}{r['Cl']:8.3f}{r['Cd']:8.4f}{errs[0]:9.3f}{errs[2]:9.3f}{errs[1]:9.5f}{errs[3]:9.5f}")

    axes[0].set(xlabel="x/c", ylabel="Cp", title=f"Surface pressure, NACA 4412, Re = {re_key}")
    axes[0].invert_yaxis()
    axes[1].set(xlabel="x/c", ylabel="Cf", title="Skin friction, suction side")
    axes[2].set(xlabel="x/c", ylabel="Cf", title="Skin friction, pressure side")
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes[1:]:
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print("saved", out)


if __name__ == "__main__":
    main()
