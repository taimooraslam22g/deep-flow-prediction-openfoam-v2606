#!/usr/bin/env python3
"""Turn the Tyrrell 026 front-wing main element of Zerihan (2001) into an airfoil file for
groundGen.py.

Table 2 of the thesis (J. Zerihan, "An Investigation into the Aerodynamics of Wings in
Ground Effect", PhD thesis, University of Southampton, 2001) lists 44 points per surface
for the single element wing at its reference incidence: upside down, with the suction
surface (facing the ground) at negative y, at a true incidence of 3.45 deg to the chord
line. This script turns it back to zero incidence (chord line from the leading edge to the
middle of the 0.007c thick trailing edge), scales it to a chord of 1, turns it upright
(suction surface on top, like the UIUC files) and writes it in UIUC order: trailing edge
-> upper surface -> leading edge -> lower surface -> trailing edge.

Two files come out:
  - tyrrell026_raw.dat: the 87 table points as they are.
  - tyrrell026.dat: a smooth version, for the runs. gmsh joins the points of an airfoil
    with a Catmull-Rom spline, which bends sharply wherever the spacing of the points
    changes (in the table: 0.001 -> 0.005 -> 0.01 -> 0.02 -> 0.05 chords). With only 44
    points per surface, the first runs had spikes in the surface pressure at those places
    (the largest at x/c 0.18, next to the suction peak). Here a quintic smoothing spline
    in arc length passes the table points within their rounding (four decimals), and is
    resampled with 200 points per surface, closely spaced near the leading and trailing
    edges (smooth_airfoils.smooth). The dataset airfoils are smoothed in sqrt(x) instead
    (smooth_airfoils.smooth_sqrt); for this table the two curves agree within 21 um.

The table itself is not part of this repository. Put its 44 rows (columns: x/c suction,
y/c suction, x/c pressure, y/c pressure; '#' lines are comments) in
data/airfoils/tyrrell026_table2.txt, then run (on Windows, with the project's Python):
    .venv\\Scripts\\python.exe ground_effect\\make_tyrrell026.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smooth_airfoils import smooth, write  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "airfoils", "tyrrell026_table2.txt")
OUT = os.path.join(ROOT, "data", "airfoils", "tyrrell026.dat")
OUT_RAW = os.path.join(ROOT, "data", "airfoils", "tyrrell026_raw.dat")
TITLE = "TYRRELL 026 FRONT WING MAIN ELEMENT (ZERIHAN 2001, TABLE 2), ZERO INCIDENCE"
CHORD_MM = 223.4  # the wind tunnel model
TABLE_DECIMALS = 4  # each number of the table is off by up to 0.00005


def main():
    t = np.loadtxt(SRC, comments="#")
    if t.shape != (44, 4):
        raise SystemExit(f"expected 44 rows of 4 numbers in {SRC}, found {t.shape}")
    suction, pressure = t[:, :2], t[:, 2:]
    for name, s in [("suction", suction), ("pressure", pressure)]:
        if not (np.diff(s[:, 0]) > 0).all():
            raise SystemExit(f"x/c of the {name} surface does not increase steadily: check the table")
    le = suction[0]
    te = 0.5 * (suction[-1] + pressure[-1])  # middle of the blunt trailing edge
    angle = np.arctan2(te[1] - le[1], te[0] - le[0])
    chord = np.hypot(*(te - le))
    c, s = np.cos(angle), np.sin(angle)

    def upright(p):
        d = p - le
        x = (d[:, 0] * c + d[:, 1] * s) / chord   # along the chord
        y = (-d[:, 0] * s + d[:, 1] * c) / chord  # across it, suction side negative
        return np.column_stack([x, -y])           # upright: suction side on top

    up, lo = upright(suction), upright(pressure)
    ar = np.vstack([up[::-1], lo[1:]])  # TE -> upper -> LE -> lower -> TE (LE point once)
    write(OUT_RAW, ar, TITLE + ", TABLE POINTS")
    new, dist, _ = smooth(ar, TABLE_DECIMALS)
    write(OUT, new, TITLE + ", SMOOTHED")

    thick = np.interp(up[:, 0], lo[:, 0], lo[:, 1])
    t_all = up[:, 1] - thick
    low = suction[np.argmin(suction[:, 1])]
    print(f"chord line in the table: {np.degrees(angle):.3f} deg to the x axis (the thesis: true incidence 3.45 deg), "
          f"length {chord:.5f}")
    print(f"trailing edge thickness {up[-1, 1] - lo[-1, 1]:.4f} c (the thesis: 0.007c); "
          f"largest thickness {t_all.max():.4f} c at x/c {up[np.argmax(t_all), 0]:.3f}")
    print(f"lowest point of the suction surface in the table: x/c {low[0]:.4f}")
    print(f"wrote the {len(ar)} table points to {OUT_RAW}")
    print(f"smoothing spline: the table points are {1e3 * np.sqrt(np.mean(dist ** 2)) * CHORD_MM:.0f} um "
          f"(rms) and at most {1e3 * dist.max() * CHORD_MM:.0f} um from it on the {CHORD_MM:g} mm model "
          f"({dist.max():.1e} c; rounding of the table: up to 0.5e-4 c in x and in y)")
    print(f"wrote {len(new)} points of the smoothed wing to {OUT}")


if __name__ == "__main__":
    main()
