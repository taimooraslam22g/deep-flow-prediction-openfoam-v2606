#!/usr/bin/env python3
"""Generate training samples for an inverted airfoil in ground effect (a 2D F1 front wing).

Adapted from data/dataGen.py of Deep-Flow-Prediction (Thuerey et al., Apache-2.0).
What is different from the original:
  * the airfoil is turned upside down, so it makes downforce instead of lift
  * the angle of attack is set by rotating the airfoil (positive = trailing edge up =
    more downforce). The air always flows parallel to the ground.
  * a ground is placed a distance h (the ride height) below the lowest point of the
    airfoil. It moves at the same speed as the air, like the road seen from a moving car.
  * every run starts from a fresh copy of case_template/, so a failed run can never
    leave old results behind that get saved by mistake
  * the lift (downforce) and drag coefficients (averaged over the last 100 iterations),
    the residuals and the surface pressure / wall shear stress are saved too
  * every run is checked for convergence. A run whose Cl is still settling after
    --iterations continues, up to --max-iterations. Runs whose Cl does not become steady
    (separated, unsteady flow) are saved in not_converged/, so they cannot end up in the
    training data by mistake. How much nuTilda OpenFOAM had to clip is recorded too
  * the ride height is drawn log-uniformly (as many cases between h/c 0.05 and 0.1 as
    between 0.3 and 0.6), because the flow changes fastest close to the ground
  * --free-air removes the ground (for validation against free-flight reference data)
  * a boundary-layer mesh (thin cells along the airfoil, first cell at y+ of about 1,
    steady cell growth away from it) in a 20-chord domain; --refine makes every cell
    smaller or larger (grid convergence study)
  * the flow is sampled along the 128 rows of the output grid instead of as one cloud
    of points. That gives the same points, without the slow search for every point
    before the first iteration
  * NACA 4-digit airfoils (e.g. naca4412) are generated from their formula if they are
    not in the airfoil folder

Defaults are the settings chosen by the validation and the mesh study (see README):
incoming nuTilda = 3e-5 (3 x nu) with a limited gradient for its convection, SIMPLEC,
1000 iterations (up to 3000 while Cl is still settling), boundary-layer mesh, 20-chord
domain. The original dataset used --nutilda 0.14 --nutilda-scheme linearUpwind
--solver simple --iterations 500 --max-iterations 500 --no-bl --domain 5 --sampling cloud
(and had no ground; the ride height was not drawn).

Each sample is saved in the same 6 x 128 x 128 layout as the original dataset, so the same
U-Net code can train on it. The ground appears in the mask channel (2), like the airfoil.

Run it inside the Docker container, from /work/ground_effect:
    python3 groundGen.py --airfoil naca23012.dat --alpha 4 --h 0.2 --U 30   # one chosen case
    python3 groundGen.py --samples 50                                       # 50 random cases
    python3 groundGen.py --free-air --airfoil naca4412 --alpha 5 --U 10     # no ground
    python3 groundGen.py --mesh-only --airfoil naca4412 --alpha 5 --h 0.2   # mesh only
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Ranges for random samples (kept after the parameter scan, see README). The angle of
# attack and the air speed are drawn uniformly; the ride height log-uniformly by default
# (--h-sampling), like the heights of the wind-tunnel tests (Zerihan, 2001).
ALPHA_RANGE = (0.0, 10.0)   # angle of attack, degrees
H_RANGE = (0.05, 0.6)       # ride height as a fraction of the chord
U_RANGE = (10.0, 50.0)      # air speed, m/s (chord Reynolds number 1-5 million)

RES = 128  # output grid: RES x RES points over x in [-0.5, 1.5], y in [-1, 1]

# Convergence checks (mesh study, round 3; parameter scan). The drag flips between two
# values from one iteration to the next (by up to 2% on coarse meshes), so Cl and Cd are
# averaged over the last AVERAGE iterations. A run counts as converged if Cl varies by
# less than STEADY_TOL over the last STEADY_WINDOW iterations (steady runs: below 0.03%
# after 1000 iterations; runs with an unsteady, separated flow: 0.6-12%).
# In the parameter scan, the Cl of some runs still changed after 1000 iterations but
# settled steadily, its range shrinking 2-5 times every 200 iterations; others
# oscillated without settling. A run whose Cl range over the last STEADY_WINDOW
# iterations is below SETTLING times the range over the window before therefore
# continues, EXTEND iterations at a time, up to --max-iterations (clip_test.sh: of five
# such runs, three converged by 2000 iterations, one was still settling at 2000, and one
# stopped settling at 1500 just above the limit).
# Whether nuTilda was still clipped at the end is recorded (clipped, clip_min), but no
# longer counts against convergence: in the scan it was clipped in nearly every iteration
# in 45% of the runs, also in runs with a steady Cl, in a few up to about a hundred cells
# at the outer edge of the wake (README, parameter scan; clip_test.sh measures the effect).
AVERAGE = 100
STEADY_WINDOW = 200
STEADY_TOL = 0.001
SETTLING = 0.7
EXTEND = 500

# Solver settings: (consistent, relaxation p, U, nuTilda)
SOLVER = {"simple": ("no", 0.3, 0.7, 0.7),     # the original case
          "simplec": ("yes", 1.0, 0.9, 0.9)}   # SIMPLEC: same answer, about 3.5x fewer iterations

# Convection scheme for nuTilda: (line added to gradSchemes, div(phi,nuTilda) scheme).
# With a realistic (low) incoming nuTilda, the original scheme (linearUpwind with an
# unlimited gradient) undershoots below zero at the edge of the boundary layer, and
# OpenFOAM clips nuTilda in almost every iteration ("bounding nuTilda" in the log). The
# nuTilda residual then stalls near 1e-3 (mesh study, round 2). Limiting the gradient
# (cellLimited) stops the clipping; the TVD scheme limitedLinear does not (round 3).
NUTILDA_SCHEMES = {
    "linearUpwind": ("", "bounded Gauss linearUpwind grad(nuTilda)"),  # the original case
    "cellLimited": ("grad(nuTilda)   cellLimited Gauss linear 1;",
                    "bounded Gauss linearUpwind grad(nuTilda)"),
    "limitedLinear": ("", "bounded Gauss limitedLinear 1"),
    # first order: cannot undershoot, so nothing to clip, but more numerical diffusion.
    # Only to measure what the clipping changes (clip_test.sh)
    "upwind": ("", "bounded Gauss upwind"),
}

# Boundary-layer mesh (--bl). The first cell is thin enough to put its centre at y+ of
# about 1, so the Spalart-Allmaras model resolves the wall region itself. (A first test
# with a fixed 2e-4 put it at y+ 4-8, in the buffer layer, and gave a jagged skin
# friction.) Its height comes from a flat-plate estimate of the skin friction at the
# sample's Reynolds number (first_layer). The layers grow by BL_RATIO up to BL_THICK
# chords, where they match the WALL_SIZE cells on the airfoil surface.
BL_RATIO, BL_THICK = 1.15, 0.03
WALL_SIZE = 0.005  # cell size along the airfoil surface, chords (as in the original mesh)
NU = 1e-5  # kinematic viscosity, as in constant/transportProperties
# Away from the wing, cells grow as WALL_SIZE + NEAR_GROWTH * distance (in chords),
# whatever the domain size. (A first test showed that a larger domain otherwise coarsens
# the cells near the wing, where we sample the flow.)
NEAR_GROWTH = 0.05


def first_layer(U, yplus=1.0):
    """Height (in chords) of the first boundary-layer cell, so its centre is at about y+."""
    cf = 0.026 * (U * 1.0 / NU) ** (-1 / 7)  # turbulent flat plate, chord Reynolds number
    u_tau = 1.5 * U * np.sqrt(cf / 2)        # 1.5: faster flow near the suction peak
    return 2 * yplus * NU / u_tau


def bl_layers(U, refine=1.0):
    """First cell height and growth ratio of the boundary layer. With --refine R every
    cell is R times smaller, also across the boundary layer (R times more layers)."""
    return first_layer(U) / refine, BL_RATIO ** (1 / refine)

# gmsh text for the two airfoil/mesh variants
ORIGINAL_EXTRUDE = """Extrude {0, 0, 1} {
  Surface{1};
  Layers{1};
  Recombine;
}
Physical Surface("back") = {1027};
Physical Surface("front") = {1};
Physical Surface("top") = {1022};
Physical Surface("exit") = {1010};
Physical Surface("ground") = {1014};
Physical Surface("inlet") = {1018};
Physical Surface("aerofoil") = {1026};
Physical Volume("internal") = {1};"""

BL_EXTRUDE = """// ex[0] = back face, ex[1] = volume, then the side faces in the order of the
// boundary curves: 1 (exit), 2 (ground), 3 (inlet), 4 (top), 1000 and 1001 (airfoil)
ex[] = Extrude {0, 0, 1} { Surface{1}; Layers{1}; Recombine; };
Physical Surface("back") = {ex[0]};
Physical Surface("front") = {1};
Physical Surface("exit") = {ex[2]};
Physical Surface("ground") = {ex[3]};
Physical Surface("inlet") = {ex[4]};
Physical Surface("top") = {ex[5]};
Physical Surface("aerofoil") = {ex[6], ex[7]};
Physical Volume("internal") = {ex[1]};"""

SIZE_FIELDS = """// Cells grow steadily with distance from the wing ({wall} + {g} * distance),
// whatever the domain size, so the region we sample stays well resolved.
Field[3] = Distance;
Field[3].CurvesList = {{1000, 1001}};
Field[3].Sampling = {sampling};
Field[4] = MathEval;
Field[4].F = "{wall} + {g} * F3";
Field[5] = Min;
Field[5].FieldsList = {{1, 4}};"""

BL_FIELD = """// Boundary layer: thin quadrilateral cells along the airfoil. The first one is
// {first} chords thick (y+ about 1), each next one {ratio} times thicker, up to
// {thick} chords. At the trailing-edge corners the layers fan out.
Field[2] = BoundaryLayer;
Field[2].CurvesList = {{1000, 1001}};
Field[2].FanPointsList = {{{fans}}};
Field[2].Size = {first};
Field[2].Ratio = {ratio};
Field[2].Thickness = {thick};
Field[2].Quads = 1;
BoundaryLayer Field = 2;"""


# ---------------------------------------------------------------- geometry

def naca4(code, n=100):
    """Coordinates of a NACA 4-digit airfoil such as '4412', in the same order as the UIUC
    files: trailing edge -> upper surface -> leading edge -> lower surface -> trailing edge.
    Uses the closed trailing edge form of the thickness formula."""
    m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))  # more points near both edges
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    if m > 0:
        yc = np.where(x < p, m / p**2 * (2 * p * x - x**2),
                      m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2))
        dyc = np.where(x < p, 2 * m / p**2 * (p - x), 2 * m / (1 - p)**2 * (p - x))
    else:
        yc = dyc = np.zeros_like(x)
    th = np.arctan(dyc)
    upper = np.column_stack([x - yt * np.sin(th), yc + yt * np.cos(th)])[::-1]  # TE -> LE
    lower = np.column_stack([x + yt * np.sin(th), yc - yt * np.cos(th)])[1:]    # LE -> TE
    return np.vstack([upper, lower])


def load_airfoil(path):
    """Read a UIUC airfoil file (one title line, then x y pairs, chord from 0 to 1).
    If the file does not exist and the name is nacaXXXX, generate the shape instead."""
    if os.path.isfile(path):
        ar = np.loadtxt(path, skiprows=1)
    else:
        m = re.fullmatch(r"naca(\d{4})(\.dat)?", os.path.basename(path).lower())
        if not m:
            raise OSError(f"{path} not found")
        ar = naca4(m.group(1))
    if np.max(np.abs(ar[0] - ar[-1])) < 1e-6:  # drop a repeated end point
        ar = ar[:-1]
    return ar


def tidy_trailing_edge(ar):
    """For the boundary-layer mesh: decide how the trailing edge closes.

    The first point is the trailing edge. The airfoil is closed by a straight segment from
    the last point back to the first. If that segment is steep, it is a blunt trailing-edge
    base with a corner at each end; otherwise it is just the last bit of the lower surface,
    with one sharp corner at the trailing edge. Returns the points and True if blunt."""
    gap = ar[0] - ar[-1]
    if np.hypot(*gap) < 1e-4:  # two trailing-edge points almost on top of each other
        return ar[:-1], False
    return ar, abs(gap[1]) > abs(gap[0])


def place_airfoil(ar, alpha_deg, h, h_alpha_deg=None):
    """Turn the airfoil upside down, rotate it, and work out where the ground goes.

    The ride height h is the gap between the lowest point and the ground. By default it is
    measured with the wing at alpha. With h_alpha_deg, it is measured with the wing at that
    incidence, and the wing is then rotated to alpha about the quarter chord, as in wind
    tunnel tests that set the height first (Zerihan 2001).
    Returns the new (x, y) points and the y position of the ground."""
    x = ar[::-1, 0]   # reversed order keeps the points running anticlockwise, as before
    y = -ar[::-1, 1]  # upside down: the suction side now faces the ground

    def rotate(deg):  # about the quarter chord; positive alpha lifts the trailing edge
        a = np.radians(deg)
        return 0.25 + (x - 0.25) * np.cos(a) - y * np.sin(a), (x - 0.25) * np.sin(a) + y * np.cos(a)

    xr, yr = rotate(alpha_deg)
    lowest = yr.min() if h_alpha_deg is None else rotate(h_alpha_deg)[1].min()
    return np.column_stack([xr, yr]), lowest - h


# ---------------------------------------------------------------- case files

def fill(path, values, out=None):
    """Replace @NAME@ placeholders in a file (in place, unless out is given)."""
    with open(path) as f:
        text = f.read()
    for key, val in values.items():
        text = text.replace("@" + key + "@", str(val))
    left = re.findall(r"@[A-Z_]+@", text)
    if left:
        raise RuntimeError(f"unfilled placeholders {sorted(set(left))} in {path}")
    with open(out or path, "w") as f:
        f.write(text)


def ground_like_top(path):
    """Free air: give the bottom boundary ('ground') the same far-field condition as 'top'."""
    with open(path) as f:
        text = f.read()
    top = re.search(r"\n\s*top\s*\{([^}]*)\}", text)
    if not top:
        raise RuntimeError(f"no 'top' entry in {path}")
    text, n = re.subn(r"(\n\s*ground\s*\{)[^}]*(\})", lambda m: m.group(1) + top.group(1) + m.group(2), text)
    if n != 1:
        raise RuntimeError(f"no single 'ground' entry in {path}")
    with open(path, "w") as f:
        f.write(text)


def run(cmd, case, log):
    """Run a command inside the case folder, sending all its output to a log file."""
    with open(os.path.join(case, log), "w") as f:
        return subprocess.call(cmd, shell=True, cwd=case, stdout=f, stderr=subprocess.STDOUT)


def set_patch_types(boundary, types):
    """gmshToFoam makes every boundary a plain 'patch'. Set the right types:
    'empty' for front and back (this makes the case 2D), 'wall' for solid surfaces."""
    with open(boundary) as f:
        text = f.read()
    for name, new in types.items():
        pattern = re.compile(r"(^\s*" + name + r"\s*\{[^}]*?^\s*type\s+)\w+", re.M)
        text, n = pattern.subn(r"\g<1>" + new, text, count=1)
        if n != 1:
            raise RuntimeError(f"boundary '{name}' not found in {boundary}")
    with open(boundary, "w") as f:
        f.write(text)


def setup_case(case, U, iterations, nutilda, free_air=False, solver="simplec", scheme="cellLimited"):
    """Fill in the flow conditions and solver settings. This must happen before meshing,
    because gmshToFoam already reads system/controlDict."""
    fields = [os.path.join(case, "0", name) for name in ("U", "p", "nut", "nuTilda")]
    if free_air:
        for path in fields:
            ground_like_top(path)
    fill(fields[0], {"VEL_X": U})
    fill(fields[1], {})
    fill(fields[2], {"NUTILDA": nutilda})
    fill(fields[3], {"NUTILDA": nutilda})
    fill(os.path.join(case, "system", "controlDict"), {"ITERATIONS": iterations})
    fill(os.path.join(case, "system", "forceCoeffs"), {"MAG_U": U})
    consistent, rp, ru, rnt = SOLVER[solver]
    fill(os.path.join(case, "system", "fvSolution"),
         {"CONSISTENT": consistent, "RELAX_P": rp, "RELAX_U": ru, "RELAX_NUTILDA": rnt})
    grad, div = NUTILDA_SCHEMES[scheme]
    fill(os.path.join(case, "system", "fvSchemes"), {"GRAD_NUTILDA": grad, "DIV_NUTILDA": div})


def make_mesh(case, pts, ground_y, h, free_air=False, domain=5.0, bl=False, blunt=False, U=10.0,
              refine=1.0):
    """Write airfoil.geo, mesh it with gmsh and convert it to OpenFOAM. refine > 1 makes
    every cell that many times smaller. Returns None on success, or a short error message."""
    last = 1000 + len(pts) - 1
    wall = WALL_SIZE / refine
    points = "".join(f"Point({1000 + i}) = {{ {px:.8f}, {py:.8f}, 0.00000000, {wall:.6g}}};\n"
                     for i, (px, py) in enumerate(pts))
    # about 10 cells across the gap; in free air the box is switched off (same size as outside)
    edge_lc = 0.04 * domain / refine  # 0.2 for the original 5-chord domain
    lc_gap = edge_lc if free_air else np.clip(h / 10, 0.005, 0.02) / refine
    if bl:
        curves = f"Spline(1000) = {{1000:{last}}};\nLine(1001) = {{{last}, 1000}};"
        loop, extrude = "1000, 1001", BL_EXTRUDE
        # the trailing edge is the last point (place_airfoil reverses the order)
        fans = f"{last}, 1000" if blunt else f"{last}"
        first, ratio = bl_layers(U, refine)
        bl_field = BL_FIELD.format(first=f"{first:.3g}", ratio=f"{ratio:.4g}", thick=BL_THICK, fans=fans)
        size_fields = SIZE_FIELDS.format(wall=f"{wall:.6g}", g=f"{NEAR_GROWTH / refine:.6g}",
                                         sampling=round(400 * refine))
        background = 5
    else:
        curves = f"Spline(1000) = {{1000:{last},1000}};"
        loop, extrude, bl_field = "1000", ORIGINAL_EXTRUDE, ""
        size_fields, background = "", 1
    fill(os.path.join(case, "ground_template.geo"), {
        "POINTS": points,
        "AIRFOIL_CURVES": curves,
        "AIRFOIL_LOOP": loop,
        "EXTRUDE_AND_PHYSICAL": extrude,
        "SIZE_FIELDS": size_fields,
        "BACKGROUND": background,
        "BL_FIELD": bl_field,
        "EDGE_LC": f"{edge_lc:g}",
        "X_IN": f"{-domain:g}",
        "X_OUT": f"{domain:g}",
        "Y_TOP": f"{domain:g}",
        "GROUND_Y": f"{ground_y:.8f}",
        "LC_GAP": f"{lc_gap:.5f}",
        "BOX_YMIN": f"{ground_y - 0.01:.8f}",
        "BOX_YMAX": f"{pts[:, 1].min() + 0.15:.8f}",
    }, out=os.path.join(case, "airfoil.geo"))

    if run("gmsh airfoil.geo -3 -format msh2 -o airfoil.msh", case, "log.gmsh") != 0:
        return "gmsh failed"
    if run("gmshToFoam airfoil.msh", case, "log.gmshToFoam") != 0:
        return "gmshToFoam failed"
    set_patch_types(os.path.join(case, "constant", "polyMesh", "boundary"),
                    {"front": "empty", "back": "empty", "aerofoil": "wall",
                     "ground": "patch" if free_air else "wall"})
    return None


def check_mesh(case):
    """Run OpenFOAM's checkMesh. Returns (ok, one-line summary)."""
    run("checkMesh", case, "log.checkMesh")
    with open(os.path.join(case, "log.checkMesh")) as f:
        text = f.read()
    ok = "Mesh OK." in text
    failed = re.search(r"Failed (\d+) mesh checks", text)
    parts = ["Mesh OK" if ok else f"{failed.group(1) if failed else '?'} mesh check(s) FAILED"]
    for label, pattern, fmt in [("max non-orthogonality", r"Mesh non-orthogonality Max: ([\d.eE+-]+)", "{:.0f}"),
                                ("max skewness", r"Max skewness = ([\d.eE+-]+)", "{:.2f}"),
                                ("max aspect ratio", r"Max aspect ratio = ([\d.eE+-]+)", "{:.0f}")]:
        m = re.search(pattern, text)
        if m:
            parts.append(f"{label} {fmt.format(float(m.group(1)))}")
    return ok, ", ".join(parts)


def patch_faces(case):
    """Number of faces on each boundary, from constant/polyMesh/boundary."""
    with open(os.path.join(case, "constant", "polyMesh", "boundary")) as f:
        text = f.read()
    return {m.group(1): int(m.group(2))
            for m in re.finditer(r"\n\s*(\w+)\s*\{[^}]*?nFaces\s+(\d+);", text)}


# ---------------------------------------------------------------- results

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def latest_time_dir(folder):
    """The latest time folder inside folder (folder names are times), or None."""
    times = [d for d in glob.glob(os.path.join(folder, "*")) if is_number(os.path.basename(d))]
    return max(times, key=lambda d: float(os.path.basename(d))) if times else None


def cell_count(case):
    try:
        with open(os.path.join(case, "constant", "polyMesh", "owner"), "rb") as f:
            m = re.search(rb"nCells:\s*(\d+)", f.read(4000))
        return int(m.group(1)) if m else -1
    except OSError:
        return -1


# Sampling of the RES x RES output grid. The original setup samples it as one 'cloud' of
# RES^2 points (case_template/system/internalCloud). Before the first iteration, OpenFOAM
# then searches the mesh for every point. On our boundary-layer meshes this took 2.5 to
# 20 minutes per run (mesh study, round 3), often longer than the solver itself. Sampling
# the same points as RES straight lines, one per grid row ('uniform' sets), needs only a
# search for where each line starts; OpenFOAM then walks along it from cell to cell.
# Same points, same interpolation (cellPoint).
LINE_SETS = """/*--------------------------------*- C++ -*----------------------------------*\\
Description
    Samples p and U on the {res} x {res} training grid (x from -0.5 to {x1:g},
    y from -1 to {y1:g}, z = 0.5), as one straight line per grid row. OpenFOAM then
    locates only the start of each line and walks along it, instead of searching for
    every point separately. Written by groundGen.py (--sampling lines).
\\*---------------------------------------------------------------------------*/

type            sets;
libs            ("libsampling.so");

fields          (p U);
interpolationScheme cellPoint;
setFormat       raw;

executeControl  writeTime;
writeControl    writeTime;

sets
(
{rows}
);

// ************************************************************************* //
"""


def grid_x(i, res=RES):
    return (i / res - 0.5) * 2 + 0.5


def grid_y(j, res=RES):
    return (j / res - 0.5) * 2


def line_sets(res=RES):
    """The internalCloud function for --sampling lines: grid row j is the line
    y = grid_y(j) from grid_x(0) to grid_x(res - 1) with res points, so the points are
    exactly those of the original cloud."""
    rows = [f"    row{j:03d} {{ type uniform; axis xyz; start ({grid_x(0):.10g} {grid_y(j):.10g} 0.5); "
            f"end ({grid_x(res - 1):.10g} {grid_y(j):.10g} 0.5); nPoints {res}; }}" for j in range(res)]
    return LINE_SETS.format(res=res, x1=grid_x(res - 1), y1=grid_y(res - 1), rows="\n".join(rows))


def read_rows(folder, res=RES):
    """The sampled grid points of the line sets in one time folder, in grid order (row by
    row, x increasing), as rows of x y z p Ux Uy Uz. Also returns the number of extra
    points that are not grid points (where a line crosses the airfoil or the ground),
    which are dropped. A row completely below the ground has no points (and maybe no file)."""
    parts, extra = [], 0
    for j in range(res):
        files = glob.glob(os.path.join(folder, f"row{j:03d}_*.xy"))
        if not files:
            continue
        if len(files) > 1:
            raise RuntimeError(f"more than one file for row {j}: {files}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # an empty file (row below the ground) is fine
            data = np.loadtxt(files[0], ndmin=2)
        if not data.size:
            continue
        # OpenFOAM writes 6 significant digits (1.453125 -> 1.45312), so allow 2e-5
        k = np.round((data[:, 0] + 0.5) * res / 2).astype(int)  # grid index along the row
        dist = np.abs(data[:, 0] - grid_x(k, res)) + np.abs(data[:, 1] - grid_y(j, res))
        keep = {}
        for n in np.where((dist < 2e-5) & (k >= 0) & (k < res))[0]:
            if k[n] not in keep or dist[n] < dist[keep[k[n]]]:  # the closest point per grid index
                keep[k[n]] = n
        extra += len(data) - len(keep)
        parts.append(data[[keep[i] for i in sorted(keep)]])
    return (np.vstack(parts) if parts else np.zeros((0, 7))), extra


def to_grid(samples, U):
    """Turn the sampled points into the 6 x RES x RES training array. samples is an array,
    or a file, with the columns x y z p Ux Uy Uz, in grid order (row by row, x increasing).

    Points inside the airfoil or below the ground are not in the mesh, so OpenFOAM
    skips them. Those become mask = 1, exactly like the airfoil in the original."""
    ar = np.loadtxt(samples) if isinstance(samples, str) else np.asarray(samples)
    out = np.zeros((6, RES, RES))
    i = 0
    for y in range(RES):
        for x in range(RES):
            xf, yf = grid_x(x), grid_y(y)
            if i < len(ar) and abs(ar[i][0] - xf) < 1e-4 and abs(ar[i][1] - yf) < 1e-4:
                out[0][x][y] = U          # input: freestream x
                out[1][x][y] = 0.0        # input: freestream y (always 0 here)
                out[3][x][y] = ar[i][3]   # target: pressure
                out[4][x][y] = ar[i][4]   # target: Ux
                out[5][x][y] = ar[i][5]   # target: Uy
                i += 1
            else:
                out[2][x][y] = 1.0        # input: mask (airfoil or ground)
    if i != len(ar):
        raise RuntimeError(f"matched {i} of {len(ar)} sampled points; the grid does not line up")
    return out


def read_coeffs(case):
    """Cl and Cd for every iteration, from the forceCoeffs output. A run that was continued
    (see continue_run) has one folder per start time (0, 1000, 1500, ...): they are joined
    in order of time."""
    base = os.path.join(case, "postProcessing", "forceCoeffs1")
    starts = sorted((d for d in glob.glob(os.path.join(base, "*")) if is_number(os.path.basename(d))),
                    key=lambda d: float(os.path.basename(d)))
    cl, cd, last = [], [], -np.inf
    for folder in starts:
        files = sorted(p for p in glob.glob(os.path.join(folder, "*.dat")) if "oeff" in os.path.basename(p))
        if not files:
            continue
        header, rows = None, []
        with open(files[-1]) as f:
            for line in f:
                if line.startswith("#"):
                    words = line[1:].split()
                    if words and words[0] == "Time":
                        header = words
                elif line.strip():
                    rows.append([float(v) for v in line.split()])
        if header is None or not rows or "Cl" not in header or "Cd" not in header:
            return None
        data = np.array(rows)
        new = data[:, 0] > last  # a time that is already there (should not happen) is kept once
        cl.extend(data[new, header.index("Cl")])
        cd.extend(data[new, header.index("Cd")])
        last = max(last, data[:, 0].max())
    if not cl:
        return None
    return np.array(cl), np.array(cd)


def rel_range(a, floor):
    """(max - min) / |mean| of the values a; the mean counts as at least floor."""
    return float((a.max() - a.min()) / max(abs(a.mean()), floor))


def force_summary(cl, cd):
    """From the Cl and Cd histories: Cl and Cd averaged over the last AVERAGE iterations;
    how much Cl varies over the last STEADY_WINDOW iterations (max - min, relative); and
    how much Cd flips from one iteration to the next (half its range over the last
    AVERAGE iterations, relative)."""
    return (float(cl[-AVERAGE:].mean()), float(cd[-AVERAGE:].mean()),
            rel_range(cl[-STEADY_WINDOW:], 0.05), 0.5 * rel_range(cd[-AVERAGE:], 0.001))


def reversed_flow(surface, alpha):
    """Fraction of the wing surface (2-98% of the chord) where the flow next to the wall runs
    backwards, from the trailing towards the leading edge: separated flow. Uses the wall
    shear stress along the chord. NaN without surface data."""
    if "surf_tau" not in surface:
        return float("nan")
    a = np.radians(alpha)
    chord = np.array([np.cos(a), np.sin(a)])  # leading -> trailing edge (see place_airfoil)
    xc = 0.25 + (surface["surf_xy_tau"] - [0.25, 0.0]) @ chord  # position along the chord
    along = surface["surf_tau"][:, :2] @ chord
    # The flow is attached at 10-40% chord in all but extreme cases: that sets the sign of
    # attached flow (OpenFOAM's sign convention for the wall shear stress can be either).
    sign = np.sign(np.median(along[(xc > 0.1) & (xc < 0.4)]))
    sel = (xc > 0.02) & (xc < 0.98)
    return float(np.mean(sign * along[sel] < 0))


def startup_seconds(log):
    """Seconds until the first iteration had finished: reading the mesh, the wall distance
    and setting up the sampling (the slow part with --sampling cloud)."""
    with open(log) as f:
        for line in f:
            if line.startswith("ExecutionTime"):
                return float(line.split()[2])
    return float("nan")


def read_surface(case):
    """Pressure and wall shear stress at the airfoil face centres, from system/surfaceData.
    Returns a dict for the .npz file (empty if nothing was written)."""
    tdir = latest_time_dir(os.path.join(case, "postProcessing", "surfaceData"))
    if tdir is None:
        return {}
    out = {}
    for path in glob.glob(os.path.join(tdir, "**", "*.raw"), recursive=True):
        names = []
        with open(path) as f:
            for line in f:
                if not line.startswith("#"):
                    break
                words = line[1:].split()
                if words[:3] == ["x", "y", "z"]:
                    names = words
        data = np.loadtxt(path, comments="#", ndmin=2)
        if data.size == 0:
            continue
        stem = os.path.basename(path)
        cols = names if len(names) == data.shape[1] else None
        if (cols and "p" in cols) or (not cols and stem.startswith("p") and data.shape[1] == 4):
            j = cols.index("p") if cols else 3
            out["surf_xy_p"], out["surf_p"] = data[:, :2], data[:, j]
        tau = [i for i, c in enumerate(cols or []) if c.startswith("wallShearStress")]
        if tau or (not cols and "wallShearStress" in stem and data.shape[1] == 6):
            j = tau[0] if tau else 3
            out["surf_xy_tau"], out["surf_tau"] = data[:, :2], data[:, j:j + 3]
    return out


def final_residuals(log):
    """The last initial residual of each solved field in the simpleFoam log."""
    res = {}
    with open(log) as f:
        for line in f:
            m = re.search(r"Solving for (\w+), Initial residual = ([0-9.eE+-]+)", line)
            if m:
                res[m.group(1)] = float(m.group(2))
    return res


def clipping(logs):
    """From the simpleFoam logs of a run (several if it was continued): in how many
    iterations OpenFOAM had to clip negative nuTilda values, the last iteration in which
    it did, and the most negative value it clipped in the last AVERAGE iterations (0 if
    none). Clipping in the first few hundred iterations, while the flow develops, is
    common. In some runs it goes on in nearly every iteration until the end: then a few
    cells at the outer edge of the wake go negative again in every iteration."""
    count, last, it, lows = 0, 0, 0, {}
    for log in logs:
        with open(log) as f:
            for line in f:
                if line.startswith("Time = "):
                    it = int(float(line.split("=")[1]))
                elif line.startswith("bounding nuTilda"):
                    count, last = count + 1, it
                    m = re.search(r"min: ([-+0-9.eE]+)", line)
                    if m:
                        lows[it] = float(m.group(1))
    recent = [v for k, v in lows.items() if k > it - AVERAGE]
    return count, last, min(recent) if recent else 0.0


def bounded_iterations(log):
    """Iterations with clipping and the last of them, for one log (verification_report.py)."""
    return clipping([log])[:2]


def settling(cl):
    """True if Cl is not steady yet but settling: its range over the last STEADY_WINDOW
    iterations is at least STEADY_TOL, and below SETTLING times its range over the
    STEADY_WINDOW iterations before (same measure as cl_range in force_summary)."""
    if len(cl) < 2 * STEADY_WINDOW:
        return False
    now = rel_range(cl[-STEADY_WINDOW:], 0.05)
    before = rel_range(cl[-2 * STEADY_WINDOW:-STEADY_WINDOW], 0.05)
    return STEADY_TOL <= now < SETTLING * before


def continue_run(case, end, step):
    """Prepare the case to continue from the last written time (controlDict starts from
    the latest time) to iteration `end`, writing every `step` iterations, so that the
    new end is written and the sampling and surface data are made there."""
    path = os.path.join(case, "system", "controlDict")
    with open(path) as f:
        text = f.read()
    text, n1 = re.subn(r"^(endTime\s+)\S+;", rf"\g<1>{end};", text, flags=re.M)
    text, n2 = re.subn(r"^(writeInterval\s+)\S+;", rf"\g<1>{step};", text, flags=re.M)
    if n1 != 1 or n2 != 1:
        raise RuntimeError(f"could not set endTime/writeInterval in {path}")
    with open(path, "w") as f:
        f.write(text)


def last_iteration(log):
    last = 0
    with open(log) as f:
        for line in f:
            m = re.match(r"^Time = (\d+)", line)
            if m:
                last = int(m.group(1))
    return last


def tail(path, n=15):
    try:
        with open(path) as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return "(no log)"


# ---------------------------------------------------------------- main

CSV_HEADER = ("file,airfoil,alpha_deg,h_over_c,U,Cl,Cd,cl_range,cd_oscillation,converged,reversed_flow,"
              "max_residual,iterations,cells,seconds,startup_seconds,solver,nutilda,bl,domain,refine,"
              "nutilda_scheme,sampling,bounded,last_bounded,clipped,clip_min\n")


def draw_h(u, sampling):
    """Ride height from a uniform draw u in H_RANGE: unchanged ('uniform') or mapped to a
    log-uniform value ('log'). The same seed then gives the same airfoils, angles and
    speeds with either kind of sampling."""
    lo, hi = H_RANGE
    return u if sampling == "uniform" else lo * (hi / lo) ** ((u - lo) / (hi - lo))


def main():
    sys.stdout.reconfigure(line_buffering=True)  # show each line at once, also when sent to a log file
    p = argparse.ArgumentParser(description="Inverted airfoil in ground effect: OpenFOAM data generation.")
    p.add_argument("--samples", type=int, default=1, help="number of simulations (default 1)")
    p.add_argument("--airfoil", help="airfoil file, e.g. naca23012.dat, or nacaXXXX (default: random)")
    p.add_argument("--alpha", type=float, help="angle of attack in degrees (default: random)")
    p.add_argument("--h", type=float, help="ride height as a fraction of the chord (default: random)")
    p.add_argument("--h-alpha", type=float,
                   help="measure the ride height with the wing at this incidence (degrees), then rotate it "
                        "to --alpha about the quarter chord, as in wind tunnel tests (default: at --alpha)")
    p.add_argument("--U", type=float, help="air speed in m/s (default: random)")
    p.add_argument("--free-air", action="store_true", help="no ground (for validation)")
    p.add_argument("--bl", action=argparse.BooleanOptionalAction, default=True,
                   help="boundary-layer mesh: thin cells along the airfoil (first cell at y+ about 1) "
                        "and steady cell growth away from it (default; --no-bl gives the original mesh)")
    p.add_argument("--domain", type=float, default=20.0,
                   help="distance from the airfoil to the inlet, exit and top, in chords (default 20; the original used 5)")
    p.add_argument("--refine", type=float, default=1.0,
                   help="make every cell this many times smaller (grid convergence study; default 1)")
    p.add_argument("--solver", choices=sorted(SOLVER), default="simplec",
                   help="simplec (default) or simple (the original settings)")
    p.add_argument("--iterations", type=int, default=1000, help="solver iterations (default 1000)")
    p.add_argument("--max-iterations", type=int, default=3000,
                   help=f"continue a run whose Cl is still settling after --iterations, {EXTEND} iterations "
                        "at a time, up to this many (default 3000; the same as --iterations: never)")
    p.add_argument("--h-sampling", choices=["log", "uniform"], default="log",
                   help="draw random ride heights log-uniformly (default) or uniformly (as in the parameter scan)")
    p.add_argument("--nutilda", type=float, default=3e-5,
                   help="incoming turbulent viscosity (default 3e-5 = 3 nu; the original dataset used 0.14)")
    p.add_argument("--nutilda-scheme", choices=list(NUTILDA_SCHEMES), default="cellLimited",
                   help="convection scheme for nuTilda (default cellLimited; the original used linearUpwind)")
    p.add_argument("--sampling", choices=["lines", "cloud"], default="lines",
                   help="sample the output grid row by row (default) or as one cloud of points "
                        "(the original; searching for every point makes the start slow)")
    p.add_argument("--mesh-only", action="store_true", help="only make and check the mesh, do not solve")
    p.add_argument("--seed", type=int, help="random seed (default: random)")
    p.add_argument("--airfoil-dir",
                   default=os.path.join(HERE, "..", "Deep-Flow-Prediction", "data", "airfoil_database"),
                   help="folder of airfoil files to draw from (default: the UIUC files as they are; the dataset "
                        "uses the smoothed files from smooth_airfoils.py, /work/data/airfoil_database_smooth)")
    p.add_argument("--out", default=os.path.join(HERE, "data", "train"), help="where to save the samples")
    p.add_argument("--worker", type=int, default=0,
                   help="run in work/worker<N>; give each copy a different number to run several at once")
    args = p.parse_args()
    if args.max_iterations > args.iterations and (args.iterations % EXTEND or args.max_iterations % EXTEND):
        # continue_run writes every EXTEND iterations: the ends must fall on those
        p.error(f"--iterations and --max-iterations must be multiples of {EXTEND} when runs may be continued")

    seed = args.seed if args.seed is not None else int(np.random.randint(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    print(f"Seed: {seed}")

    files = sorted(os.listdir(args.airfoil_dir))
    os.makedirs(args.out, exist_ok=True)
    template = os.path.join(HERE, "case_template")
    case = os.path.join(HERE, "work", f"worker{args.worker}")
    csv = os.path.join(args.out, "samples.csv")
    if os.path.exists(csv):
        with open(csv) as f:
            if f.readline() != CSV_HEADER:  # written by an older version: start a new file
                csv = os.path.join(args.out, "samples2.csv")
    mesh_text = (f"{'boundary-layer' if args.bl else 'original'} mesh"
                 + (f" refined x{args.refine:g}" if args.refine != 1 else "")
                 + f", domain {args.domain:g} chords")

    for n in range(args.samples):
        # draw all random values every time, so a given seed always gives the same cases
        r_airfoil, r_alpha = files[rng.integers(len(files))], rng.uniform(*ALPHA_RANGE)
        r_h, r_U = draw_h(rng.uniform(*H_RANGE), args.h_sampling), rng.uniform(*U_RANGE)
        airfoil = args.airfoil or r_airfoil
        alpha = args.alpha if args.alpha is not None else r_alpha
        h = np.inf if args.free_air else (args.h if args.h is not None else r_h)
        U = args.U if args.U is not None else r_U

        basename = os.path.splitext(os.path.basename(airfoil))[0]
        where = "free air" if args.free_air else f"h/c {h:.3f}" + (
            f" (measured at {args.h_alpha:g} deg)" if args.h_alpha is not None else "")
        print(f"Run {n}: {basename}, alpha {alpha:+.2f} deg, {where}, U {U:.1f} m/s"
              f" ({args.solver}, nuTilda {args.nutilda:g} with {args.nutilda_scheme}, {mesh_text})")
        start = time.time()

        # fresh copy of the case, so nothing from an earlier run can be reused by mistake
        if os.path.exists(case):
            shutil.rmtree(case)
        shutil.copytree(template, case)
        setup_case(case, U, args.iterations, args.nutilda, args.free_air, args.solver, args.nutilda_scheme)
        if args.sampling == "lines":
            with open(os.path.join(case, "system", "internalCloud"), "w") as f:
                f.write(line_sets())

        path = airfoil if os.path.isfile(airfoil) else os.path.join(args.airfoil_dir, airfoil)
        try:
            ar = load_airfoil(path)
        except (OSError, ValueError) as e:
            print(f"    could not read the airfoil ({e}), skipping")
            continue
        blunt = False
        if args.bl:
            ar, blunt = tidy_trailing_edge(ar)
        pts, ground_y = place_airfoil(ar, alpha, 0.0 if args.free_air else h, args.h_alpha)
        if args.free_air:
            ground_y = -args.domain
        elif ground_y < -0.95:
            print("    warning: the ground is below the sampled window (y = -1), "
                  "so it will not show up in the mask")

        err = make_mesh(case, pts, ground_y, h, args.free_air, args.domain, args.bl, blunt, U, args.refine)
        if err:
            print(f"    {err}, skipping. End of the log:\n{tail(os.path.join(case, 'log.' + err.split()[0]))}")
            continue
        ok, summary = check_mesh(case)
        faces = patch_faces(case)
        print(f"    mesh: {cell_count(case):,} cells ({summary})")
        print("    boundary faces: " + ", ".join(f"{k} {v}" for k, v in faces.items() if k not in ("front", "back"))
              + (f"; trailing edge {'blunt' if blunt else 'sharp'}, first cell {bl_layers(U, args.refine)[0]:.2g} chords"
                 if args.bl else ""))
        if args.mesh_only:
            print(f"    mesh only: the case is in {os.path.relpath(case, HERE)}")
            continue
        print("    solving (this takes a few minutes)...")

        logs = [os.path.join(case, "log.simpleFoam")]
        if run("simpleFoam", case, "log.simpleFoam") != 0:
            print(f"    simpleFoam failed, skipping. End of the log:\n{tail(logs[0])}")
            continue
        # Cl not steady yet but settling? Then continue, EXTEND iterations at a time, up
        # to --max-iterations (not if the solver already stopped on its residuals)
        end, failed = args.iterations, False
        while end < args.max_iterations and last_iteration(logs[-1]) == end:
            coeffs = read_coeffs(case)
            if coeffs is None or not settling(coeffs[0]):
                break
            new_end = min(end + EXTEND, args.max_iterations)
            print(f"    Cl still settling after {end} iterations (varies by "
                  f"{100 * rel_range(coeffs[0][-STEADY_WINDOW:], 0.05):.2f}%): continuing to {new_end}")
            continue_run(case, new_end, new_end - end)
            end = new_end
            logs.append(os.path.join(case, f"log.simpleFoam.{end}"))
            if run("simpleFoam", case, os.path.basename(logs[-1])) != 0:
                print(f"    simpleFoam failed, skipping. End of the log:\n{tail(logs[-1])}")
                failed = True
                break
        if failed:
            continue
        iters = last_iteration(logs[-1])

        folder = latest_time_dir(os.path.join(case, "postProcessing", "internalCloud"))
        if folder is None:
            samples = np.zeros((0, 7))
        elif float(os.path.basename(folder)) != iters:  # never save the flow of an earlier iteration
            print(f"    the flow was sampled at iteration {os.path.basename(folder)}, not at the last "
                  f"iteration {iters}, skipping")
            continue
        elif args.sampling == "lines":
            samples = read_rows(folder)[0]
        else:
            cloud = os.path.join(folder, "cloud_p_U.xy")
            samples = np.loadtxt(cloud, ndmin=2) if os.path.exists(cloud) else np.zeros((0, 7))
        if len(samples) < RES * RES // 4:  # at most about half the grid is airfoil or ground
            print(f"    only {len(samples)} of the {RES * RES} grid points were sampled, skipping. "
                  f"End of the log:\n{tail(logs[-1])}")
            continue
        grid = to_grid(samples, U)

        coeffs = read_coeffs(case)
        if coeffs is None:
            print("    warning: could not read the force coefficients")
            cl = cd = cl_range = cd_osc = float("nan")
        else:
            cl, cd, cl_range, cd_osc = force_summary(*coeffs)
        surface = read_surface(case)
        if "surf_p" not in surface or "surf_tau" not in surface:
            print("    warning: no surface pressure / wall shear stress found")
        rev = reversed_flow(surface, alpha)
        res = final_residuals(logs[-1])
        max_res = max(res.values()) if res else float("nan")
        bounded, last_bounded, clip_min = clipping(logs)
        clipped = bounded > 0 and last_bounded > iters - AVERAGE  # still clipped at the end
        converged = bool(cl_range < STEADY_TOL)  # False if Cl is missing
        cells = cell_count(case)
        startup = startup_seconds(logs[0])
        seconds = time.time() - start

        tag = "free" if args.free_air else f"h{round(h * 1000):04d}"
        name = f"{basename}_{tag}_a{round(alpha * 100):+05d}_U{round(U * 10):04d}"
        sub = "" if converged else "not_converged"
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
        saved = os.path.join(args.out, sub, name + ".npz")
        np.savez_compressed(saved, a=grid,
                            airfoil=basename, alpha=alpha, h=h, U=U,
                            h_alpha=np.nan if args.h_alpha is None else args.h_alpha,
                            ground_y=np.nan if args.free_air else ground_y,
                            Cl=cl, Cd=cd, cl_range=cl_range, cd_oscillation=cd_osc,
                            converged=converged, reversed_flow=rev, max_residual=max_res,
                            iterations=iters, nutilda=args.nutilda, solver=args.solver,
                            bl=args.bl, domain=args.domain, refine=args.refine,
                            nutilda_scheme=args.nutilda_scheme, sampling=args.sampling,
                            bounded=bounded, last_bounded=last_bounded, clipped=clipped,
                            clip_min=clip_min, max_iterations=args.max_iterations, cells=cells,
                            seconds=seconds, startup_seconds=startup,
                            mesh_ok=ok, bl_first=bl_layers(U, args.refine)[0] if args.bl else np.nan,
                            Cl_history=coeffs[0] if coeffs else np.array([]),
                            Cd_history=coeffs[1] if coeffs else np.array([]), **surface)
        new_csv = not os.path.exists(csv)
        with open(csv, "a") as f:
            if new_csv:
                f.write(CSV_HEADER)
            f.write(f"{os.path.join(sub, name + '.npz').replace(os.sep, '/')},{basename},{alpha:.4f},{h:.4f},"
                    f"{U:.3f},{cl:.5f},{cd:.5f},{cl_range:.6f},{cd_osc:.5f},{int(converged)},{rev:.4f},"
                    f"{max_res:.3e},{iters},{cells},{seconds:.0f},{startup:.0f},"
                    f"{args.solver},{args.nutilda:g},{int(args.bl)},{args.domain:g},{args.refine:g},"
                    f"{args.nutilda_scheme},{args.sampling},{bounded},{last_bounded},{int(clipped)},"
                    f"{clip_min:.3e}\n")

        print(f"    done in {seconds:.0f} s after {iters} iterations "
              f"(the first iteration ended after {startup:.0f} s)")
        print(f"    Cl = {cl:+.4f}  (negative = downforce),  Cd = {cd:.4f}  "
              f"(both averaged over the last {AVERAGE} iterations)")
        print("    final residuals: " + ", ".join(f"{k} {v:.1e}" for k, v in res.items()))
        print(f"    nuTilda clipped below zero in {bounded} of {iters} iterations"
              + (f" (last in iteration {last_bounded})" if bounded else "")
              + (f"  <- still at the end, down to {clip_min:.1e} ({clip_min / args.nutilda:.0f} x the "
                 f"incoming value) in the last {AVERAGE} iterations" if clipped else ""))
        print(f"    Cl varies by {100 * cl_range:.3f}% over the last {STEADY_WINDOW} iterations; "
              f"Cd flips by +-{100 * cd_osc:.2f}% from one iteration to the next")
        print(f"    flow reversed (separated) on {100 * rev:.0f}% of the wing surface")
        if converged:
            print("    converged")
        else:
            print(f"    NOT converged (Cl still varies by {100 * cl_range:.2f}%): saved in not_converged/, "
                  "not for training")
        print(f"    saved {os.path.relpath(saved, HERE)}")


if __name__ == "__main__":
    main()
