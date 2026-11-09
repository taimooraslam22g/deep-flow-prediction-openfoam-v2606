"""Plot one training sample (.npz) produced by dataGen.py or ground_effect/groundGen.py.

Usage:
    python3 scripts/view_sample.py <path/to/sample.npz> [output.png]

Each sample is a (6, 128, 128) array, indexed [channel][x][y]:
    0  freestream velocity X (input)   - constant everywhere outside the airfoil
    1  freestream velocity Y (input)
    2  mask (input)                    - 1 inside the airfoil (and below the ground), 0 in the fluid
    3  pressure p (target)
    4  velocity Ux (target)
    5  velocity Uy (target)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")  # write to a file; no window needed (works inside Docker)
import matplotlib.pyplot as plt
import numpy as np

NAMES = [
    ("Input: freestream Ux", "viridis"),
    ("Input: freestream Uy", "viridis"),
    ("Input: airfoil mask", "gray_r"),
    ("Target: pressure p", "RdBu_r"),
    ("Target: velocity Ux", "RdBu_r"),
    ("Target: velocity Uy", "RdBu_r"),
]

path = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(path)[0] + "_view.png"

data = np.load(path)
a = data["a"]
mask = a[2] > 0.5
ground = "ground_y" in data.files  # sample from ground_effect/groundGen.py
gy = float(data["ground_y"]) if ground else np.nan  # NaN in --free-air samples
if ground and np.isfinite(gy):
    NAMES[2] = ("Input: mask (airfoil + ground)", "gray_r")

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for i, ax in enumerate(axes.flat):
    title, cmap = NAMES[i]
    field = a[i].T  # stored as [x][y]; transpose so x is horizontal
    if i != 2:
        field = np.ma.masked_where(mask.T, field)  # hide the airfoil interior
    # targets: centre the colour scale on zero so +/- regions are easy to compare
    if i >= 3:
        lim = np.abs(field).max()
        im = ax.imshow(field, origin="lower", cmap=cmap, vmin=-lim, vmax=lim,
                       extent=[-0.5, 1.5, -1, 1])
    else:
        im = ax.imshow(field, origin="lower", cmap=cmap, extent=[-0.5, 1.5, -1, 1])
    if -1 < gy < 1:
        ax.axhline(gy, color="k", lw=0.8)  # the ground
    ax.set_title(title)
    ax.set_xlabel("x (chord lengths)")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, shrink=0.8)

if ground:
    where = f"h/c = {float(data['h']):.3f}" if np.isfinite(gy) else "free air"
    fig.suptitle(f"{os.path.basename(path)}  |  {data['airfoil']} upside down, "
                 f"alpha = {float(data['alpha']):+.1f} deg, {where}, "
                 f"U = {float(data['U']):.1f} m/s  |  Cl = {float(data['Cl']):+.3f}, "
                 f"Cd = {float(data['Cd']):.4f}")
else:
    fsx, fsy = a[0][~mask].mean(), a[1][~mask].mean()
    aoa = np.degrees(np.arctan2(-fsy, fsx))
    fig.suptitle(f"{os.path.basename(path)}  |  freestream = ({fsx:.1f}, {fsy:.1f}) m/s, "
                 f"|U| = {np.hypot(fsx, fsy):.1f} m/s, angle = {aoa:+.1f} deg")
fig.tight_layout()
fig.savefig(out, dpi=110)
print("saved", out)
