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
  * the lift (downforce) and drag coefficients, the residuals, a convergence check and
    the surface pressure / wall shear stress are saved too
  * --free-air removes the ground (for validation against free-flight reference data)
  * a boundary-layer mesh (thin cells along the airfoil, first cell at y+ of about 1,
    steady cell growth away from it) in a 20-chord domain; --refine makes every cell
    smaller or larger (grid convergence study)
  * NACA 4-digit airfoils (e.g. naca4412) are generated from their formula if they are
    not in the airfoil folder

Defaults are the settings chosen by the validation and the mesh study (see README):
incoming nuTilda = 3e-5 (3 x nu) with a limited convection scheme, SIMPLEC, 1000
iterations, boundary-layer mesh, 20-chord domain. The original dataset used
--nutilda 0.14 --nutilda-scheme linearUpwind --solver simple --iterations 500 --no-bl --domain 5.

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
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Ranges for random samples. These are a first guess; the final ranges get decided
# after the validation runs.
ALPHA_RANGE = (0.0, 10.0)   # angle of attack, degrees
H_RANGE = (0.05, 0.6)       # ride height as a fraction of the chord
U_RANGE = (10.0, 50.0)      # air speed, m/s

RES = 128  # output grid: RES x RES points over x in [-0.5, 1.5], y in [-1, 1]

# Solver settings: (consistent, relaxation p, U, nuTilda)
SOLVER = {"simple": ("no", 0.3, 0.7, 0.7),     # the original case
          "simplec": ("yes", 1.0, 0.9, 0.9)}   # SIMPLEC: same answer, about 3.5x fewer iterations

# Convection scheme for nuTilda: (line added to gradSchemes, div(phi,nuTilda) scheme).
# With a realistic (low) incoming nuTilda, the original scheme (linearUpwind with an
# unlimited gradient) undershoots below zero at the edge of the boundary layer, and
# OpenFOAM clips nuTilda in almost every iteration ("bounding nuTilda" in the log). The
# nuTilda residual then stalls near 1e-3, and the skin friction gets small steps
# (mesh study, round 2). Limiting the gradient or the interpolation prevents the undershoot.
NUTILDA_SCHEMES = {
    "linearUpwind": ("", "bounded Gauss linearUpwind grad(nuTilda)"),  # the original case
    "cellLimited": ("grad(nuTilda)   cellLimited Gauss linear 1;",
                    "bounded Gauss linearUpwind grad(nuTilda)"),
    "limitedLinear": ("", "bounded Gauss limitedLinear 1"),
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


def place_airfoil(ar, alpha_deg, h):
    """Turn the airfoil upside down, rotate it, and work out where the ground goes.

    Returns the new (x, y) points and the y position of the ground."""
    x = ar[::-1, 0]   # reversed order keeps the points running anticlockwise, as before
    y = -ar[::-1, 1]  # upside down: the suction side now faces the ground
    a = np.radians(alpha_deg)
    # rotate about the quarter chord; positive alpha lifts the trailing edge
    xr = 0.25 + (x - 0.25) * np.cos(a) - y * np.sin(a)
    yr = (x - 0.25) * np.sin(a) + y * np.cos(a)
    ground_y = yr.min() - h  # ride height = gap between the lowest point and the ground
    return np.column_stack([xr, yr]), ground_y


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


def latest(pattern):
    """The file matching pattern in the latest time folder (folder names are times)."""
    paths = glob.glob(pattern)
    if not paths:
        return None
    return max(paths, key=lambda p: float(os.path.basename(os.path.dirname(p))))


def cell_count(case):
    try:
        with open(os.path.join(case, "constant", "polyMesh", "owner"), "rb") as f:
            m = re.search(rb"nCells:\s*(\d+)", f.read(4000))
        return int(m.group(1)) if m else -1
    except OSError:
        return -1


def to_grid(cloud_file, U):
    """Turn the sampled points into the 6 x RES x RES training array.

    Points inside the airfoil or below the ground are not in the mesh, so OpenFOAM
    skips them. Those become mask = 1, exactly like the airfoil in the original."""
    ar = np.loadtxt(cloud_file)  # columns: x y z p Ux Uy Uz
    out = np.zeros((6, RES, RES))
    i = 0
    for y in range(RES):
        for x in range(RES):
            xf = (x / RES - 0.5) * 2 + 0.5
            yf = (y / RES - 0.5) * 2
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
    """Cl and Cd for every iteration, from the forceCoeffs output."""
    files = [p for p in glob.glob(os.path.join(case, "postProcessing", "forceCoeffs1", "*", "*.dat"))
             if "oeff" in os.path.basename(p)]
    if not files:
        return None
    header, rows = None, []
    with open(sorted(files)[-1]) as f:
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
    return data[:, header.index("Cl")], data[:, header.index("Cd")]


def read_surface(case):
    """Pressure and wall shear stress at the airfoil face centres, from system/surfaceData.
    Returns a dict for the .npz file (empty if nothing was written)."""
    times = [d for d in glob.glob(os.path.join(case, "postProcessing", "surfaceData", "*"))
             if is_number(os.path.basename(d))]
    if not times:
        return {}
    tdir = max(times, key=lambda d: float(os.path.basename(d)))
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


def bounded_iterations(log):
    """In how many iterations OpenFOAM had to clip negative nuTilda values, and the last
    iteration in which it did. Some clipping at the start is harmless; clipping that goes
    on until the end stops the residual from converging."""
    count, last, it = 0, 0, 0
    with open(log) as f:
        for line in f:
            if line.startswith("Time = "):
                it = int(float(line.split("=")[1]))
            elif line.startswith("bounding nuTilda"):
                count, last = count + 1, it
    return count, last


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

CSV_HEADER = ("file,airfoil,alpha_deg,h_over_c,U,Cl,Cd,cl_change_last50,max_residual,"
              "iterations,cells,seconds,solver,nutilda,bl,domain,refine,nutilda_scheme,bounded,last_bounded\n")


def main():
    p = argparse.ArgumentParser(description="Inverted airfoil in ground effect: OpenFOAM data generation.")
    p.add_argument("--samples", type=int, default=1, help="number of simulations (default 1)")
    p.add_argument("--airfoil", help="airfoil file, e.g. naca23012.dat, or nacaXXXX (default: random)")
    p.add_argument("--alpha", type=float, help="angle of attack in degrees (default: random)")
    p.add_argument("--h", type=float, help="ride height as a fraction of the chord (default: random)")
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
    p.add_argument("--iterations", type=int, default=1000, help="maximum solver iterations (default 1000)")
    p.add_argument("--nutilda", type=float, default=3e-5,
                   help="incoming turbulent viscosity (default 3e-5 = 3 nu; the original dataset used 0.14)")
    p.add_argument("--nutilda-scheme", choices=list(NUTILDA_SCHEMES), default="cellLimited",
                   help="convection scheme for nuTilda (default cellLimited; the original used linearUpwind)")
    p.add_argument("--mesh-only", action="store_true", help="only make and check the mesh, do not solve")
    p.add_argument("--seed", type=int, help="random seed (default: random)")
    p.add_argument("--airfoil-dir",
                   default=os.path.join(HERE, "..", "Deep-Flow-Prediction", "data", "airfoil_database"))
    p.add_argument("--out", default=os.path.join(HERE, "data", "train"), help="where to save the samples")
    p.add_argument("--worker", type=int, default=0,
                   help="run in work/worker<N>; give each copy a different number to run several at once")
    args = p.parse_args()

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
        r_h, r_U = rng.uniform(*H_RANGE), rng.uniform(*U_RANGE)
        airfoil = args.airfoil or r_airfoil
        alpha = args.alpha if args.alpha is not None else r_alpha
        h = np.inf if args.free_air else (args.h if args.h is not None else r_h)
        U = args.U if args.U is not None else r_U

        basename = os.path.splitext(os.path.basename(airfoil))[0]
        where = "free air" if args.free_air else f"h/c {h:.3f}"
        print(f"Run {n}: {basename}, alpha {alpha:+.2f} deg, {where}, U {U:.1f} m/s"
              f" ({args.solver}, nuTilda {args.nutilda:g} with {args.nutilda_scheme}, {mesh_text})")
        start = time.time()

        # fresh copy of the case, so nothing from an earlier run can be reused by mistake
        if os.path.exists(case):
            shutil.rmtree(case)
        shutil.copytree(template, case)
        setup_case(case, U, args.iterations, args.nutilda, args.free_air, args.solver, args.nutilda_scheme)

        path = airfoil if os.path.isfile(airfoil) else os.path.join(args.airfoil_dir, airfoil)
        try:
            ar = load_airfoil(path)
        except (OSError, ValueError) as e:
            print(f"    could not read the airfoil ({e}), skipping")
            continue
        blunt = False
        if args.bl:
            ar, blunt = tidy_trailing_edge(ar)
        pts, ground_y = place_airfoil(ar, alpha, 0.0 if args.free_air else h)
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

        log = os.path.join(case, "log.simpleFoam")
        if run("simpleFoam", case, "log.simpleFoam") != 0:
            print(f"    simpleFoam failed, skipping. End of the log:\n{tail(log)}")
            continue

        cloud = latest(os.path.join(case, "postProcessing", "internalCloud", "*", "cloud_p_U.xy"))
        if cloud is None:
            print(f"    no sampled flow field found, skipping. End of the log:\n{tail(log)}")
            continue
        grid = to_grid(cloud, U)

        coeffs = read_coeffs(case)
        if coeffs is None:
            print("    warning: could not read the force coefficients")
            cl = cd = cl_change = float("nan")
        else:
            cl, cd = float(coeffs[0][-1]), float(coeffs[1][-1])
            last = coeffs[0][-50:]
            cl_change = float((last.max() - last.min()) / max(abs(last.mean()), 0.05))
        surface = read_surface(case)
        if "surf_p" not in surface or "surf_tau" not in surface:
            print("    warning: no surface pressure / wall shear stress found")
        res = final_residuals(log)
        max_res = max(res.values()) if res else float("nan")
        iters = last_iteration(log)
        bounded, last_bounded = bounded_iterations(log)
        cells = cell_count(case)
        seconds = time.time() - start

        tag = "free" if args.free_air else f"h{round(h * 1000):04d}"
        name = f"{basename}_{tag}_a{round(alpha * 100):+05d}_U{round(U * 10):04d}"
        np.savez_compressed(os.path.join(args.out, name + ".npz"), a=grid,
                            airfoil=basename, alpha=alpha, h=h, U=U,
                            ground_y=np.nan if args.free_air else ground_y,
                            Cl=cl, Cd=cd, cl_change=cl_change, max_residual=max_res,
                            iterations=iters, nutilda=args.nutilda, solver=args.solver,
                            bl=args.bl, domain=args.domain, refine=args.refine,
                            nutilda_scheme=args.nutilda_scheme, bounded=bounded,
                            last_bounded=last_bounded, cells=cells,
                            mesh_ok=ok, bl_first=bl_layers(U, args.refine)[0] if args.bl else np.nan,
                            Cl_history=coeffs[0] if coeffs else np.array([]), **surface)
        new_csv = not os.path.exists(csv)
        with open(csv, "a") as f:
            if new_csv:
                f.write(CSV_HEADER)
            f.write(f"{name}.npz,{basename},{alpha:.4f},{h:.4f},{U:.3f},{cl:.5f},{cd:.5f},"
                    f"{cl_change:.5f},{max_res:.3e},{iters},{cells},{seconds:.0f},"
                    f"{args.solver},{args.nutilda:g},{int(args.bl)},{args.domain:g},"
                    f"{args.refine:g},{args.nutilda_scheme},{bounded},{last_bounded}\n")

        print(f"    done in {seconds:.0f} s after {iters} iterations")
        print(f"    Cl = {cl:+.4f}  (negative = downforce),  Cd = {cd:.4f}")
        print("    final residuals: " + ", ".join(f"{k} {v:.1e}" for k, v in res.items()))
        print(f"    nuTilda clipped below zero in {bounded} of {iters} iterations"
              + (f" (last in iteration {last_bounded})" if bounded else "")
              + ("  <- still at the end: the nuTilda residual cannot converge"
                 if bounded and last_bounded > 0.9 * iters else ""))
        verdict = "steady" if cl_change < 0.01 else "NOT steady - look at this case before using it"
        print(f"    Cl changed by {100 * cl_change:.2f}% over the last 50 iterations: {verdict}")
        print(f"    saved {os.path.relpath(os.path.join(args.out, name + '.npz'), HERE)}")


if __name__ == "__main__":
    main()
