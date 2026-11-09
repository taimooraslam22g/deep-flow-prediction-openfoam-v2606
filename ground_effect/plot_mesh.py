#!/usr/bin/env python3
"""Draw the mesh around the airfoil from a run folder (its gmsh file airfoil.msh):
the whole wing, a close-up of the leading edge and a close-up of the trailing edge.

Usage (on Windows, with the project's Python):
    python ground_effect/plot_mesh.py ground_effect/work/worker6 results/mesh.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")  # write to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection


def read_msh2(path):
    """Cells of the 2D slice (as polygons) and the airfoil surface points, from a gmsh
    version 2 ASCII file written by groundGen.py."""
    with open(path) as f:
        lines = f.read().split("\n")
    names = {}
    if "$PhysicalNames" in lines:
        i = lines.index("$PhysicalNames")
        for line in lines[i + 2:i + 2 + int(lines[i + 1])]:
            _, tag, name = line.split(maxsplit=2)
            names[name.strip('"')] = int(tag)
    i = lines.index("$Nodes")
    nodes = {}
    for line in lines[i + 2:i + 2 + int(lines[i + 1])]:
        t = line.split()
        nodes[int(t[0])] = (float(t[1]), float(t[2]), float(t[3]))
    j = lines.index("$Elements")
    polys, wall = [], []
    for line in lines[j + 2:j + 2 + int(lines[j + 1])]:
        t = list(map(int, line.split()))
        kind, ntags = t[1], t[2]
        v = t[3 + ntags:]
        if kind in (5, 6):  # hexahedron or prism: draw its face at z = 0
            half = len(v) // 2
            face = v[:half] if abs(nodes[v[0]][2]) < 1e-9 else v[half:]
            polys.append([nodes[k][:2] for k in face])
        elif kind == 3 and t[3] == names.get("aerofoil"):  # face on the airfoil surface
            wall += [nodes[k][:2] for k in v]
    return polys, np.array(wall)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    case, out = sys.argv[1], sys.argv[2]
    polys, wall = read_msh2(os.path.join(case, "airfoil.msh"))
    quads = sum(len(p) == 4 for p in polys)
    le, te = wall[np.argmin(wall[:, 0])], wall[np.argmax(wall[:, 0])]
    views = [("whole wing", (le[0] - 0.3, te[0] + 0.3), (wall[:, 1].min() - 0.35, wall[:, 1].max() + 0.25)),
             ("leading edge", (le[0] - 0.03, le[0] + 0.07), (le[1] - 0.04, le[1] + 0.04)),
             ("trailing edge", (te[0] - 0.06, te[0] + 0.04), (te[1] - 0.04, te[1] + 0.04))]
    centres = np.array([np.mean(p, axis=0) for p in polys])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (title, xl, yl) in zip(axes, views):
        near = ((centres[:, 0] > xl[0] - 0.3) & (centres[:, 0] < xl[1] + 0.3)
                & (centres[:, 1] > yl[0] - 0.3) & (centres[:, 1] < yl[1] + 0.3))
        ax.add_collection(PolyCollection([polys[k] for k in np.where(near)[0]],
                                         facecolors="none", edgecolors="k", linewidths=0.2))
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.set_aspect("equal")
        ax.set_title(title)
    fig.suptitle(f"{os.path.basename(os.path.normpath(case))}: {len(polys):,} cells in the 2D slice, "
                 f"{quads:,} of them quadrilaterals (boundary layer)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print("saved", out)


if __name__ == "__main__":
    main()
