#!/usr/bin/env python3
"""Smooth airfoil files before meshing, so that the surface gmsh builds from them is smooth
and close to the airfoil's true shape.

gmsh joins the points of an airfoil with a uniform Catmull-Rom spline. It bends sharply
wherever the spacing of the points changes abruptly (shape_check.py measures the effect on
the surface pressure), and with the few points most files have around the leading edge it
gets the nose wrong. This script replaces the points of each file with a smooth curve
through them (smooth_sqrt):
  - y as a function of xi = +-sqrt(x - x0) (+ on the upper surface, - on the lower), where x0
    is the nose. Behind a round leading edge y grows like sqrt(x - x0), so in xi the surface
    runs smoothly over the nose and a spline gets it right from few points;
  - a quintic smoothing spline that stays within the rounding of the points (decimals: the
    decimals their y values carry; many files are padded with zeros at the end);
  - resampled with 200 points per surface whose spacing changes gradually (a cosine in arc
    length from the trailing edge to the nose and back). The trailing-edge points stay where
    they are, and a closed trailing edge stays closed.
Files that would not give a proper surface are left out (smooth_airfoil): the chord does not
run from 0 to 1, a surface is not a function of x, a point is misplaced, the contour crosses
itself (as given, or after smoothing), or the smoothed shape still makes the inviscid surface
pressure wiggle (panel.py). smoothing_test.py compares the curves with the exact shapes of
stand-in airfoils.

It runs on Windows with the project's Python, because it needs scipy (the container has no
scipy). Folders and wildcards are expanded here:
    .venv\\Scripts\\python.exe ground_effect\\smooth_airfoils.py Deep-Flow-Prediction\\data\\airfoil_database --out data\\airfoil_database_smooth --jobs 8
Then pass the new folder to groundGen.py in the container: --airfoil-dir /work/data/airfoil_database_smooth
"""
import argparse
import glob
import multiprocessing
import os
import sys
import warnings
from decimal import Decimal, InvalidOperation

import numpy as np
from scipy.interpolate import BSpline, splev, splprep
from scipy.linalg import lstsq
from scipy.optimize import brentq
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from groundGen import tidy_trailing_edge  # noqa: E402
from panel import inviscid_cp, wiggle  # noqa: E402

POINTS_PER_SIDE = 200
MAX_TURN = 30.0     # degrees: a sharper corner between 10% and 90% chord is a misplaced point
MAX_WIGGLE = 0.1    # in inviscid Cp after smoothing (the raw Tyrrell 026 table had 0.103)
SCREEN_ALPHA = 3.45  # incidence of that screen, degrees (the Tyrrell validation's)


def file_list(names):
    """Airfoil files from the command line: files, folders (all .dat files in them) and
    wildcard patterns. A name that is none of these is kept (it may be nacaXXXX)."""
    out = []
    for name in names:
        if os.path.isdir(name):
            out += sorted(glob.glob(os.path.join(name, "*.dat")))
        elif any(c in name for c in "*?["):
            out += sorted(glob.glob(name))
        else:
            out.append(name)
    return out


def read_points(path):
    """The points of a UIUC file (one title line, then x y pairs), without repeated points
    next to each other. A last point on the base of a blunt trailing edge (on the straight
    line from the point before it to the first point: a point that closes the contour, or
    one on the camber line) is dropped. groundGen.py closes a blunt trailing edge with a
    straight base anyway, and as the end of the lower surface the point would bend that
    surface sharply. A last point that repeats the first one at a sharp trailing edge is
    kept: both surfaces end there."""
    ar = np.loadtxt(path, skiprows=1, ndmin=2)
    if ar.shape[1] != 2:
        raise ValueError(f"expected two numbers per line, found {ar.shape[1]}")
    ar = ar[np.r_[True, np.hypot(*np.diff(ar, axis=0).T) > 1e-9]]
    if len(ar) > 3:
        base = ar[0] - ar[-2]  # the trailing edge without the last point
        length = np.hypot(*base)
        if length >= 1e-4 and abs(base[1]) > abs(base[0]):  # blunt, as tidy_trailing_edge decides
            d = ar[-1] - ar[-2]
            along = np.dot(d, base) / length ** 2
            off = abs(d[0] * base[1] - d[1] * base[0]) / length
            if -1e-9 <= along <= 1 + 1e-9 and off < 1e-4:
                ar = ar[:-1]
    return ar


def decimals(path):
    """Decimals of the coordinates in a file: (as written, as carried). As written: the
    median over all numbers. As carried: the median over the nonzero y values, without the
    zeros at the end. Many files are padded with zeros (0.1235000 carries four decimals), and
    the x values are often round stations (0.05), so only the y values tell how precisely the
    points were given."""
    written, carried = [], []
    with open(path) as f:
        next(f)  # title line
        for line in f:
            words = line.split()
            for j, word in enumerate(words[:2]):
                try:
                    d = Decimal(word)
                except InvalidOperation:
                    continue
                written.append(max(0, -d.as_tuple().exponent))
                if j == 1 and d != 0:
                    carried.append(max(0, -d.normalize().as_tuple().exponent))
    if not written:
        return None, None
    return int(np.median(written)), int(np.median(carried)) if carried else int(np.median(written))


def curve_distance(curve, pts):
    """Distance of each point from a polyline that follows a curve closely (exact for the
    polyline: the nearest vertex is found first, then the two segments next to it)."""
    j = cKDTree(curve).query(pts)[1]
    best = np.full(len(pts), np.inf)
    for a, b in [(np.maximum(j - 1, 0), j), (j, np.minimum(j + 1, len(curve) - 1))]:
        p, d = curve[a], curve[b] - curve[a]
        length2 = np.einsum("ij,ij->i", d, d)
        t = np.clip(np.einsum("ij,ij->i", pts - p, d) / np.where(length2 > 0, length2, 1), 0, 1)
        best = np.minimum(best, np.hypot(*(p + t[:, None] * d - pts).T))
    return best


def rounding(digits):
    """Standard deviation of the rounding error of a number written with this many decimals
    (an error spread evenly over +-half a unit of the last decimal)."""
    return 0.5 * 10.0 ** -digits / np.sqrt(3)


def arc_spline(ar, digits):
    """The quintic smoothing spline of smooth() (FITPACK), and the parameter of each point."""
    m = len(ar)
    u = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(ar, axis=0).T))])
    u /= u[-1]
    w = np.ones(m)
    w[[0, -1]] = 100  # keep the two trailing-edge corners in place
    tck, _ = splprep([ar[:, 0], ar[:, 1]], u=u, w=w, k=5, s=m * 2 * rounding(digits) ** 2)
    return tck, u


def smooth(ar, digits, n=POINTS_PER_SIDE):
    """Quintic smoothing spline in arc length through the points ar (UIUC order: trailing
    edge -> upper surface -> leading edge -> lower surface -> trailing edge), resampled with
    n points per surface: the smoothing of the Tyrrell 026 wing (make_tyrrell026.py). The
    spline may stay away from the points by the rounding of their digits on average, in x
    and in y, and no more (the usual choice, Reinsch 1967); the two end points are kept in
    place. The spacing follows a cosine from the trailing edge to the leading edge and back,
    so it changes smoothly everywhere. Returns the new points, the distance of each old point
    from the curve (in chords) and a fine polyline of the curve."""
    tck, u = arc_spline(ar, digits)
    u_le = u[np.argmin(ar[:, 0])]
    b = np.linspace(0, np.pi, n)
    uu = np.concatenate([u_le * (1 - np.cos(b)) / 2, u_le + (1 - u_le) * (1 - np.cos(b[1:])) / 2])
    new = np.column_stack(splev(uu, tck))
    dense = np.column_stack(splev(np.linspace(0, 1, 200001), tck))
    return new, curve_distance(dense, ar), dense


def smoothing_spline(x, y, sigma):
    """Quintic smoothing spline f of the points (x, y), x increasing, that stays away from
    them by about their errors sigma (one per point): of all curves with
    sum(((y - f(x)) / sigma)^2) = len(x), the one with the least integral of f'''^2 (Reinsch
    1967; Schoenberg 1964). It has a knot at every point, so it cannot swing out between
    points that are far apart, and its slope and curvature are free at the ends (those of a
    cubic smoothing spline, which penalises f''^2 instead, are pulled to zero there)."""
    k = 5
    span = x[-1] - x[0]
    inner = [x[0]]
    for v in x[1:-1]:  # one knot for points closer than 1e-9 of the span
        if v - inner[-1] > 1e-9 * span and x[-1] - v > 1e-9 * span:
            inner.append(v)
    inner = np.array(inner[1:])
    t = np.r_[[x[0]] * (k + 1), inner, [x[-1]] * (k + 1)]
    nb = len(t) - k - 1
    breaks = np.r_[x[0], inner, x[-1]]
    # integral of f'''^2: f''' is a parabola between knots, so 3 Gauss points per piece are exact
    g, gw = np.polynomial.legendre.leggauss(3)
    h = np.diff(breaks)
    xg = (breaks[:-1, None] + h[:, None] * (g + 1) / 2).ravel()
    wg = (h[:, None] * gw / 2).ravel()
    third = BSpline(t, np.eye(nb), k).derivative(3)(xg) * np.sqrt(wg)[:, None]
    scale = np.median(sigma)
    w = scale / sigma
    a, b = BSpline.design_matrix(x, t, k).toarray() * w[:, None], y * w
    target = len(x) * scale ** 2
    lam0 = np.sum(a ** 2) / np.sum(third ** 2)  # weighs the two terms alike, whatever the file

    def coefficients(log_lam):  # least squares of the points and of sqrt(lam) f''' together
        m = np.vstack([a, np.sqrt(lam0 * 10.0 ** log_lam) * third])
        return lstsq(m, np.r_[b, np.zeros(len(third))], lapack_driver="gelsy")[0]

    def excess(log_lam):
        return np.sum((a @ coefficients(log_lam) - b) ** 2) - target

    lo, hi = -14.0, 10.0  # from interpolation to the least-squares parabola
    if excess(lo) >= 0:
        log_lam = lo
    elif excess(hi) <= 0:
        log_lam = hi
    else:
        log_lam = brentq(excess, lo, hi, xtol=1e-3)
    return BSpline(t, coefficients(log_lam), k)


def nose(ar, digits):
    """The nose: the leftmost point of the curve of smooth() (FITPACK's quintic smoothing
    spline in arc length, which adds knots where the curve turns fast), between the two
    points on either side of the leftmost point of the file. On stand-ins with known shapes,
    this put the nose closer than smoothing splines with a knot at every point did (cubic or
    quintic, in arc length), or the smoothest curve in xi. FITPACK warns when it cannot meet
    the smoothing exactly (very precise files); the curve is still close to the points."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tck, u = arc_spline(ar, digits)
    le = int(np.argmin(ar[:, 0]))
    x, y = splev(np.linspace(u[max(le - 2, 0)], u[min(le + 2, len(ar) - 1)], 20001), tck)
    i = int(np.argmin(x))
    return float(x[i]), float(y[i])


def smooth_sqrt(ar, digits, n=POINTS_PER_SIDE):
    """Smoothing spline of y against xi = +-sqrt(x - x0) (+ on the upper surface, - on the
    lower), where (x0, y0) is the nose. Behind a round leading edge, y grows like the square
    root of the distance from the nose, so as a function of xi the surface runs smoothly over
    the nose: the spline gets the nose right even from the few points most files have there.
    A quintic smoothing spline that stays within the rounding of the points (the rounding of
    x enters through its effect on xi), resampled with n points per surface, with a cosine
    spacing in arc length from the trailing edge to the nose and back. Returns the new
    points, the distance of each old point from the curve (in chords) and a fine polyline of
    the curve; raises a ValueError if a surface is not a function of x."""
    x0, y0 = nose(ar, digits)
    le = int(np.argmin(ar[:, 0]))
    pair = le + 1 < len(ar) and ar[le + 1, 0] == ar[le, 0]  # two leading-edge points, one per surface
    if x0 >= ar[le, 0]:  # the curve passes just right of the leftmost point: the nose is that point
        if pair:
            raise ValueError("the nose is not to the left of the two leading-edge points")
        x0 = ar[le, 0]
    upper = np.r_[np.ones(le, bool), [pair or ar[le, 1] >= y0], np.zeros(len(ar) - le - 1, bool)]
    xi = np.where(upper, 1.0, -1.0) * np.sqrt(np.maximum(ar[:, 0] - x0, 0))
    xi, y = xi[::-1], ar[::-1, 1]  # lower trailing edge -> nose -> upper trailing edge
    if np.any(np.diff(xi) <= 0):
        raise ValueError("a surface is not a function of x")
    # an error dx in x moves xi by dx / (2 |xi|), and y at a given xi by the slope times that
    slope = np.gradient(y, xi)
    sigma = rounding(digits) * np.sqrt(1 + (slope / (2 * np.maximum(np.abs(xi), 1e-3))) ** 2)
    sigma[[0, -1]] /= 100  # keep the two trailing-edge end points in place
    spl = smoothing_spline(xi, y, sigma)
    dense, new = [], []
    for a, b in [(xi[-1], 0.0), (0.0, xi[0])]:  # upper surface from the trailing edge, then lower
        q = np.linspace(a, b, 100001)
        xy = np.column_stack([x0 + q ** 2, spl(q)])
        s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
        target = s[-1] * (1 - np.cos(np.linspace(0, np.pi, n))) / 2
        dense.append(xy)
        new.append(np.column_stack([np.interp(target, s, xy[:, 0]), np.interp(target, s, xy[:, 1])]))
    dense = np.vstack([dense[0], dense[1][1:]])
    return np.vstack([new[0], new[1][1:]]), curve_distance(dense, ar), dense


def as_meshed(ar):
    """The points groundGen.py meshes from ar: a last point that repeats the first one
    dropped (load_airfoil), then the trailing edge tidied (tidy_trailing_edge). Returns the
    points and True if the trailing edge is blunt."""
    if np.max(np.abs(ar[0] - ar[-1])) < 1e-6:
        ar = ar[:-1]
    return tidy_trailing_edge(ar)


def crosses(ar):
    """True if two segments of the contour (closed from the last point to the first) cross."""
    a = ar
    b = np.roll(ar, -1, axis=0)

    def side(p, q, r):  # sign of the turn p -> q -> r
        return np.sign((q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) -
                       (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0]))

    A, B, C, D = a[:, None], b[:, None], a[None, :], b[None, :]
    hit = (side(A, B, C) * side(A, B, D) < 0) & (side(C, D, A) * side(C, D, B) < 0)
    return bool(np.triu(hit, 2).any())  # neighbours share a point and never cross properly


def problems(ar):
    """Reasons to leave an airfoil out, as short texts (none for a good file): the chord does
    not run from 0 to 1 (within 2%), a surface is not a function of x (smooth_sqrt needs
    one), a misplaced point (the contour turns by more than MAX_TURN degrees at one point
    between 10% and 90% chord: 99% of the files stay below 14 degrees there, and the files
    above 30 have points that are clearly misplaced), or the contour crosses itself as
    groundGen.py closes it."""
    out = []
    x = ar[:, 0]
    if not (abs(x.min()) < 0.02 and abs(x.max() - 1) < 0.02):
        out.append(f"chord not from 0 to 1 (x/c {x.min():.3f} to {x.max():.3f})")
    le = int(np.argmin(x))
    # the upper surface runs from the trailing edge to the leading edge and the lower surface
    # back; the lower surface may start at the leading edge's x (two leading-edge points)
    bad = int((np.diff(x[:le + 1]) >= 0).sum() + (np.diff(x[le + 1:]) <= 0).sum())
    if bad:
        out.append(f"a surface is not a function of x (x/c turns back or repeats at {bad} point(s))")
    d = np.diff(ar, axis=0)
    heading = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    turn = np.abs((np.diff(heading) + 180) % 360 - 180)  # at each point between two segments
    mid = (ar[1:-1, 0] > 0.1) & (ar[1:-1, 0] < 0.9)
    if (mid & (turn > MAX_TURN)).any():
        out.append(f"a misplaced point (the contour turns by {turn[mid].max():.0f} degrees at x/c "
                   f"{ar[1:-1][mid][np.argmax(turn[mid]), 0]:.3f})")
    if crosses(as_meshed(ar)[0]):
        out.append("the contour crosses itself")
    return out


def as_written(new):
    """Points as groundGen.py reads them back from a file written by write(): six decimals,
    and the last point dropped if it repeats the first (a closed trailing edge)."""
    new = np.round(new, 6)
    return new[:-1] if np.max(np.abs(new[0] - new[-1])) < 1e-6 else new


def smooth_airfoil(ar, digits):
    """smooth_sqrt(ar, digits) for an airfoil that problems() does not leave out, and whose
    smoothed shape, as groundGen.py meshes it from the written file, is good: the contour
    does not cross itself (it can where both surfaces run into a thin trailing edge closer
    together than the rounding allows to resolve), and the inviscid Cp at SCREEN_ALPHA does
    not wiggle by more than MAX_WIGGLE between 2% and 95% chord (errors in the points, or
    features finer than the points resolve, that the smoothing within their rounding cannot
    remove). Returns the same as smooth_sqrt, and the reasons to leave the airfoil out (then
    the rest is None)."""
    why = problems(ar)
    if why:
        return None, None, None, why
    new, dist, dense = smooth_sqrt(ar, digits)
    written = as_written(new)
    if crosses(as_meshed(written)[0]):
        return None, None, None, ["the smoothed contour crosses itself"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # polyfit warns about poor conditioning near the nose
        cp = inviscid_cp(written, SCREEN_ALPHA)
        side, (w, xw) = max(((s, wiggle(*cp[s])) for s in ("upper", "lower")), key=lambda t: t[1][0])
    if w > MAX_WIGGLE:
        return None, None, None, [f"the smoothed shape's inviscid Cp still wiggles by {w:.2f} "
                                  f"({side} surface, x/c {xw:.3f})"]
    return new, dist, dense, []


def smooth_file(path):
    """Smooth one airfoil file. Returns the title, the new points (None if the file is left
    out), the old points, their distance from the curve, the decimals used (as carried) and
    the reasons to leave the file out."""
    with open(path) as f:
        title = f.readline().strip()
    ar = read_points(path)
    digits = decimals(path)[1]
    new, dist, _, why = smooth_airfoil(ar, digits)
    return title, new, ar, dist, digits, why


def write(path, ar, title):
    with open(path, "w") as f:
        f.write(title + "\n")
        for x, y in ar:
            f.write(f"{x:10.6f} {y:10.6f}\n")


def smooth_one(path):
    """smooth_file() for one process of the pool: (file name, its result, error text)."""
    try:
        return os.path.basename(path), smooth_file(path), None
    except (OSError, ValueError, np.linalg.LinAlgError) as e:
        return os.path.basename(path), None, str(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+", help="airfoil files, folders of .dat files, or wildcard patterns")
    ap.add_argument("--out", required=True, help="folder for the smoothed files (same names)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="files to smooth at the same time, one per processor core (default 1)")
    args = ap.parse_args()
    files = file_list(args.files)
    os.makedirs(args.out, exist_ok=True)
    old = set(os.listdir(args.out))
    if args.jobs > 1:
        for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = "1"  # one thread in each process: the processes share the cores
        with multiprocessing.Pool(args.jobs) as pool:
            results = list(pool.imap(smooth_one, files, chunksize=4))
    else:
        results = [smooth_one(path) for path in files]
    fits, left_out, failed = [], [], []
    for name, result, error in results:
        if error:
            failed.append(f"{name}: {error}")
            continue
        title, new, ar, dist, digits, why = result
        if why:
            left_out.append(f"{name}: {'; '.join(why)}")
            continue
        write(os.path.join(args.out, name), new, f"{title} (smoothed, {POINTS_PER_SIDE} points per surface)")
        old.discard(name)
        # how far the curve stays from the points, in half units of their last decimal
        fits.append((dist.max() / (0.5 * 10.0 ** -digits), name))
    print(f"smoothed {len(fits)} of {len(files)} files into {args.out}")
    if fits:
        r = np.array([f[0] for f in fits])
        print(f"largest distance of a file's points from its curve, in half units of the last decimal "
              f"they carry: median {np.median(r):.2f}, 90% of the files below {np.percentile(r, 90):.2f}, "
              f"largest {r.max():.2f} ({max(fits)[1]})")
    if left_out:
        print(f"left out {len(left_out)} files:")
        for text in left_out:
            print("  " + text)
    for text in failed:
        print("  could not smooth", text)
    stale = sorted(f for f in old if f.endswith(".dat"))
    if stale:
        print(f"note: {args.out} also holds {len(stale)} .dat files that this run did not write "
              f"(from an earlier run?): {', '.join(stale[:10])}{' ...' if len(stale) > 10 else ''}")


if __name__ == "__main__":
    main()
