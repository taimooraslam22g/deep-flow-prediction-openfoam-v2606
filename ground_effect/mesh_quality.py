#!/usr/bin/env python3
"""Mesh quality checks for a groundGen.py run.

  * y+ of the first cell along both sides of the wing: is the boundary-layer mesh fine
    enough near the wall? (Needs the run's sample .npz for the wall shear stress.)
  * median cell size at increasing distance from the wing, inside the sampled window:
    is the region we sample for the training data resolved well?

Usage (on Windows, with the project's Python):
    python ground_effect/mesh_quality.py <run folder> [<sample.npz>]
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from groundGen import NU  # noqa: E402
from plot_mesh import read_msh2  # noqa: E402
from validate_les import to_upright, top_side  # noqa: E402

BINS = [0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5]  # distance from the wing, chords


def centres_and_sizes(polys):
    """Centre and size of each 2D cell. The size is the edge of an equilateral triangle
    with the same area, which is what gmsh's mesh size means."""
    centres = np.array([np.mean(p, axis=0) for p in polys])
    areas = np.array([0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))
                      for p in map(np.array, polys)])
    return centres, np.sqrt(4 * areas / np.sqrt(3))


def size_vs_distance(case):
    """Number of cells, cells in the sampled window, and the median cell size in the window
    for each distance band in BINS."""
    polys, wall = read_msh2(os.path.join(case, "airfoil.msh"))
    c, size = centres_and_sizes(polys)
    d = cKDTree(wall).query(c)[0]
    window = (c[:, 0] > -0.5) & (c[:, 0] < 1.5) & (c[:, 1] > -1) & (c[:, 1] < 1)
    medians = [float(np.median(size[m])) if m.any() else np.nan
               for m in (window & (d >= lo) & (d < hi) for lo, hi in zip(BINS[:-1], BINS[1:]))]
    return len(polys), int(window.sum()), medians


def yplus(case, npz):
    """y+ of the first cell centre at each airfoil wall face.
    Returns x/c (upright), y+ and whether the face is on the suction side."""
    polys, wall = read_msh2(os.path.join(case, "airfoil.msh"))
    on_wall = {tuple(np.round(p, 9)) for p in wall}
    first = []  # centre x, y and wall distance of every cell touching the airfoil
    for p in map(np.array, polys):
        on = np.array([tuple(np.round(v, 9)) in on_wall for v in p])
        if on.sum() >= 2:
            w, c = p[on], p.mean(axis=0)
            t = w[1] - w[0]
            n = np.array([-t[1], t[0]]) / np.hypot(*t)
            first.append((c[0], c[1], abs(np.dot(c - w[0], n))))
    first = np.array(first)
    z = np.load(npz)
    xy, tau = z["surf_xy_tau"], z["surf_tau"]
    k = cKDTree(first[:, :2]).query(xy)[1]  # the cell next to each wall face
    u_tau = np.sqrt(np.hypot(tau[:, 0], tau[:, 1]))  # tau is kinematic (divided by density)
    x, y = to_upright(xy, float(z["alpha"]))
    return x, first[k, 2] * u_tau / NU, top_side(str(z["airfoil"]))(x, y)


def report(case, npz=None):
    n, nwin, med = size_vs_distance(case)
    print(f"{os.path.basename(os.path.normpath(case))}: {n:,} cells, {nwin:,} in the sampled window")
    print("  median cell size (chords) at this distance from the wing:")
    print("   " + "  ".join(f"{lo:.2f}-{hi:.2f}" for lo, hi in zip(BINS[:-1], BINS[1:])))
    print("   " + "  ".join(f"{v:9.4f}" for v in med))
    if npz:
        x, yp, top = yplus(case, npz)
        inner = (x > 0.05) & (x < 0.95)  # skip the stagnation point and the trailing edge
        for side, sel in [("suction side", top & inner), ("pressure side", ~top & inner)]:
            v = yp[sel]
            print(f"  y+ of the first cell, {side} (5-95% chord): min {v.min():.2f}, "
                  f"median {np.median(v):.2f}, max {v.max():.2f}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    report(*sys.argv[1:])
