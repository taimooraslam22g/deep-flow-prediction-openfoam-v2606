#!/usr/bin/env python3
"""Check airfoil files for kinks in the surface that gmsh builds from them.

gmsh joins the points of an airfoil with its built-in 'Spline', a Catmull-Rom curve. Where
the spacing of the points changes abruptly, that curve bends sharply between them. The kinks
are tiny in shape (hundredths of a millimetre on a wind tunnel model), but the surface
pressure reacts to the curvature, so they show up as spikes in Cp.

This script builds the same curve through the points of each file and computes the
inviscid surface pressure on it with a panel method (constant-strength sources and vortex,
Hess & Smith 1967). It reports the largest wiggle on each surface: how far Cp departs from a
smooth local fit (a parabola over +-0.01 chords), between 2% and 95% chord. A narrow but
smooth feature of the shape also gives a small value; kinks give large ones.

Usage (on Windows, with the project's Python):
    .venv\\Scripts\\python.exe ground_effect\\shape_check.py <airfoil.dat> [...] [--alpha 3.45] [--plot out.png]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from groundGen import load_airfoil  # noqa: E402


def gmsh_curve(ar, per_segment):
    """Points along the uniform Catmull-Rom spline through ar, as gmsh's built-in kernel
    builds it (end segments: mirrored end points)."""
    p = np.vstack([2 * ar[0] - ar[1], ar, 2 * ar[-1] - ar[-2]])
    t = np.linspace(0, 1, per_segment, endpoint=False)[:, None]
    out = [0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3)
           for p0, p1, p2, p3 in zip(p[:-3], p[1:-2], p[2:-1], p[3:])]
    return np.vstack(out + [ar[-1:]])


def panel_cp(pts, alpha_deg):
    """Inviscid Cp at the panel midpoints of the contour pts (UIUC order: trailing edge ->
    upper surface -> leading edge -> lower surface; a thick trailing edge stays open).
    Returns the midpoints and Cp, in the same order."""
    pts = pts[::-1]  # clockwise, so that the left normal points into the flow
    p1, p2 = pts[:-1], pts[1:]
    d = p2 - p1
    length = np.hypot(d[:, 0], d[:, 1])
    t = d / length[:, None]
    n = np.column_stack([-t[:, 1], t[:, 0]])
    mid = 0.5 * (p1 + p2)
    rx, ry = mid[:, None, 0] - p1[None, :, 0], mid[:, None, 1] - p1[None, :, 1]
    xi = rx * t[None, :, 0] + ry * t[None, :, 1]   # midpoint i in the frame of panel j
    eta = rx * n[None, :, 0] + ry * n[None, :, 1]
    log_r = np.log(np.hypot(xi, eta) / np.hypot(xi - length[None, :], eta))
    angle = np.arctan2(eta, xi - length[None, :]) - np.arctan2(eta, xi)
    np.fill_diagonal(log_r, 0.0)
    np.fill_diagonal(angle, np.pi)  # on the panel itself, seen from the flow side
    k = 1 / (2 * np.pi)

    def project(along, across, onto):  # velocity in panel j's frame -> component along onto[i]
        vx = along * t[None, :, 0] + across * n[None, :, 0]
        vy = along * t[None, :, 1] + across * n[None, :, 1]
        return vx * onto[:, None, 0] + vy * onto[:, None, 1]

    src_n, src_t = project(k * log_r, k * angle, n), project(k * log_r, k * angle, t)
    vor_n, vor_t = project(k * angle, -k * log_r, n).sum(1), project(k * angle, -k * log_r, t).sum(1)
    a = np.radians(alpha_deg)
    v_inf = np.array([np.cos(a), np.sin(a)])
    m = len(length)
    lhs, rhs = np.zeros((m + 1, m + 1)), np.zeros(m + 1)
    lhs[:m, :m], lhs[:m, m], rhs[:m] = src_n, vor_n, -(n @ v_inf)        # no flow through the wall
    lhs[m, :m], lhs[m, m] = src_t[0] + src_t[-1], vor_t[0] + vor_t[-1]  # Kutta condition
    rhs[m] = -(t[0] + t[-1]) @ v_inf
    sol = np.linalg.solve(lhs, rhs)
    vt = src_t @ sol[:m] + vor_t * sol[m] + t @ v_inf
    return mid[::-1], (1 - vt ** 2)[::-1]


def surfaces(mid, cp):
    """{side: (x, cp)} sorted by x; upper = from the trailing edge to the leading edge."""
    le = np.argmin(mid[:, 0])
    out = {}
    for side, sl in [("upper", slice(0, le + 1)), ("lower", slice(le, None))]:
        o = np.argsort(mid[sl, 0])
        out[side] = (mid[sl, 0][o], cp[sl][o])
    return out


def wiggle(x, cp, lo=0.02, hi=0.95, half_width=0.01):
    """Largest departure of cp from a parabola fitted over +-half_width around each point."""
    best = (0.0, np.nan)
    for i in np.where((x > lo) & (x < hi))[0]:
        s = np.abs(x - x[i]) < half_width
        dev = abs(cp[i] - np.polyval(np.polyfit(x[s], cp[s], 2), x[i]))
        if dev > best[0]:
            best = (dev, float(x[i]))
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--alpha", type=float, default=3.45, help="incidence for the panel method, degrees")
    ap.add_argument("--plot", help="also draw the inviscid Cp of all files into this image")
    args = ap.parse_args()
    results = []
    for f in args.files:
        ar = load_airfoil(f)
        curve = gmsh_curve(ar, max(2, 1600 // len(ar)))
        sides = surfaces(*panel_cp(curve, args.alpha))
        text = "; ".join(f"{side} surface {w:.3f} at x/c {x:.3f}" for side, (w, x) in
                         ((s, wiggle(*sides[s])) for s in ("upper", "lower")))
        print(f"{os.path.basename(f)} ({len(ar)} points): largest wiggle in inviscid Cp: {text}")
        results.append((os.path.basename(f), sides))
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        for k, (name, sides) in enumerate(results):
            for ax in axes:
                ax.plot(*sides["upper"], "-", color=f"C{k}", lw=1, label=f"{name}, upper surface")
                ax.plot(*sides["lower"], "--", color=f"C{k}", lw=1, label=f"{name}, lower surface")
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
