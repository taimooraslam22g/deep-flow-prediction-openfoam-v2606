#!/usr/bin/env python3
"""Find the cells where OpenFOAM clipped negative nuTilda in the last iteration of a run.

When the nuTilda equation gives a negative value in a cell, OpenFOAM's bound() replaces it
with fvc::average(max(nuTilda, 0)): the face-area-weighted mean of the linearly
interpolated face values, in which that cell (and any other clipped cell) counts as 0.
So in the field written after the last iteration, a clipped cell holds exactly that mean
of its neighbours. With the mesh geometry computed as OpenFOAM does (face centres from
triangles, cell centres = centroids of these extruded 2D cells), the equality holds to
round-off (relative 1e-12 or so) in clipped cells and practically never by chance
elsewhere (none in runs without clipping). Cells are flagged at relative 1e-6.

Works on the work/worker<N> folders that groundGen.py leaves behind (2D meshes, binary
fields, LSB with 32-bit labels and 64-bit scalars). Usage, on Windows:
    .venv\\Scripts\\python.exe ground_effect\\clip_cells.py ground_effect\\work\\worker102
"""
import argparse
import os
import re

import numpy as np

TOL = 1e-6


def _lists(data, start, dtype, width=1, count=1):
    """Read `count` binary lists ('N (' + N items + ')') from data, searching from start."""
    out, pos = [], start
    for _ in range(count):
        m = re.compile(rb"(\d+)\s*\(").search(data, pos)
        n, b0 = int(m.group(1)), m.end()
        nbytes = n * width * np.dtype(dtype).itemsize
        arr = np.frombuffer(data[b0:b0 + nbytes], dtype=dtype)
        out.append(arr.reshape(n, width) if width > 1 else arr)
        pos = b0 + nbytes + 1
    return out


def _body(data):
    """Position just after the FoamFile header."""
    if b"format      binary" not in data[:2000] and b"format binary" not in data[:2000]:
        raise ValueError("expected a binary OpenFOAM file (controlDict writeFormat binary)")
    i = data.index(b"FoamFile")
    return data.index(b"}", i) + 1


def read_mesh(case):
    pm = os.path.join(case, "constant", "polyMesh")
    raw = {n: open(os.path.join(pm, n), "rb").read() for n in ("points", "faces", "owner", "neighbour")}
    points = _lists(raw["points"], _body(raw["points"]), "<f8", 3)[0]
    offsets, flat = _lists(raw["faces"], _body(raw["faces"]), "<i4", 1, 2)
    owner = _lists(raw["owner"], _body(raw["owner"]), "<i4")[0]
    neighbour = _lists(raw["neighbour"], _body(raw["neighbour"]), "<i4")[0]
    ncells = int(re.search(rb"nCells:\s*(\d+)", raw["owner"]).group(1))
    text = open(os.path.join(pm, "boundary")).read()
    patches = {m.group(1): (int(m.group(2)), int(m.group(3))) for m in re.finditer(
        r"\n\s*(\w+)\s*\{[^}]*?nFaces\s+(\d+);[^}]*?startFace\s+(\d+);", text)}
    return dict(points=points, offsets=offsets, flat=flat, owner=owner, neighbour=neighbour,
                ncells=ncells, patches=patches)


def read_scalar(case, time, name):
    data = open(os.path.join(case, str(time), name), "rb").read()
    i = data.index(b"nonuniform", data.index(b"internalField"))
    return _lists(data, i, "<f8")[0]


def geometry(m):
    """Face centres and area vectors (triangles about the mean point, as OpenFOAM), and
    cell centres (centroid of each cell's 'front' face: the cells are extruded polygons)."""
    P, off, flat = m["points"], m["offsets"], m["flat"]
    npts = np.diff(off)
    fc, fa = np.zeros((len(npts), 3)), np.zeros((len(npts), 3))
    for k in np.unique(npts):
        idx = np.where(npts == k)[0]
        v = P[flat[off[idx][:, None] + np.arange(k)]]  # faces x k x 3
        c = v.mean(1)
        area, cen, tot = np.zeros((len(idx), 3)), np.zeros((len(idx), 3)), np.zeros(len(idx))
        for j in range(k):
            a, b = v[:, j], v[:, (j + 1) % k]
            tri = 0.5 * np.cross(a - c, b - c)
            mag = np.linalg.norm(tri, axis=1)
            area += tri
            cen += mag[:, None] * (a + b + c) / 3
            tot += mag
        fa[idx], fc[idx] = area, cen / np.maximum(tot, 1e-300)[:, None]
    n, s = m["patches"]["front"]
    cc = np.zeros((m["ncells"], 3))
    for k in np.unique(npts[s:s + n]):
        idx = s + np.where(npts[s:s + n] == k)[0]
        v = P[flat[off[idx][:, None] + np.arange(k)]][:, :, :2]
        x, y = v[:, :, 0], v[:, :, 1]
        x1, y1 = np.roll(x, -1, axis=1), np.roll(y, -1, axis=1)
        cr = x * y1 - x1 * y
        A = cr.sum(1) / 2
        cells = m["owner"][idx]
        cc[cells, 0] = ((x + x1) * cr).sum(1) / (6 * A)
        cc[cells, 1] = ((y + y1) * cr).sum(1) / (6 * A)
    cc[:, 2] = 0.5 * (P[:, 2].min() + P[:, 2].max())
    return fc, fa, cc


def latest_time(case):
    times = [d for d in os.listdir(case) if re.fullmatch(r"\d+(\.\d+)?", d) and float(d) > 0]
    return max(times, key=float) if times else None


def clipped_cells(case, time=None, nutilda_inf=3e-5):
    """The cells clipped in the last iteration before `time` (default: the latest), with
    their centres, nuTilda and the largest nuTilda among their neighbours, plus the wing
    surface (face centres) and the ground height. Returns a dict."""
    time = time or latest_time(case)
    m = read_mesh(case)
    fc, fa, cc = geometry(m)
    mag = np.linalg.norm(fa, axis=1)
    nt = read_scalar(case, time, "nuTilda")
    owner, nb, nc = m["owner"], m["neighbour"], m["ncells"]
    f = np.arange(len(nb))
    d_own = np.abs(np.einsum("ij,ij->i", fa[f], fc[f] - cc[owner[f]]))
    d_nei = np.abs(np.einsum("ij,ij->i", fa[f], cc[nb[f]] - fc[f]))
    w = d_nei / (d_own + d_nei)  # OpenFOAM's linear weight of the owner cell
    den, bnum = np.zeros(nc), np.zeros(nc)
    np.add.at(den, owner[f], mag[f])
    np.add.at(den, nb[f], mag[f])
    for name, (n, s) in m["patches"].items():
        if name in ("front", "back"):  # empty patches take no part
            continue
        faces = np.arange(s, s + n)
        np.add.at(den, owner[faces], mag[faces])
        if name in ("inlet", "top"):  # freestream value coming in; walls are 0; the exit takes the cell's
            np.add.at(bnum, owner[faces], mag[faces] * nutilda_inf)
    reset = np.zeros(nc, bool)
    for _ in range(20):  # neighbouring clipped cells count as 0 as well: repeat until nothing changes
        val = np.where(reset, 0.0, np.maximum(nt, 0))
        num = bnum.copy()
        np.add.at(num, owner[f], mag[f] * (1 - w) * val[nb[f]])
        np.add.at(num, nb[f], mag[f] * w * val[owner[f]])
        new = np.abs(nt - num / den) <= TOL * np.abs(nt)
        if (new == reset).all():
            break
        reset = new
    nbmax = np.zeros(nc)
    np.maximum.at(nbmax, owner[f], nt[nb[f]])
    np.maximum.at(nbmax, nb[f], nt[owner[f]])
    n, s = m["patches"]["aerofoil"]
    wing = fc[s:s + n, :2]
    g = m["patches"].get("ground", (0, 0))
    ground = fc[g[1]:g[1] + g[0], 1].mean() if g[0] else np.nan
    idx = np.where(reset)[0]
    return dict(time=time, cells=nc, idx=idx, xy=cc[idx, :2], nutilda=nt[idx], nb_max=nbmax[idx],
                nutilda_max=nt.max(), wing=wing, ground_y=ground, nutilda_inf=nutilda_inf)


def summary(r):
    """One line: how many cells, where (behind the wing, near the ground, in the sampled
    window), and how large nuTilda is around them."""
    xy = r["xy"]
    if not len(xy):
        return f"no clipped cells in the last iteration (time {r['time']}, {r['cells']:,} cells)"
    dw = np.min(np.linalg.norm(r["wing"][None] - xy[:, None], axis=2), axis=1)
    te = r["wing"][np.argmax(r["wing"][:, 0])]
    window = (xy[:, 0] >= -0.5) & (xy[:, 0] <= 1.5) & (xy[:, 1] >= -1) & (xy[:, 1] <= 1)
    behind = xy[:, 0] > te[0]
    near_ground = xy[:, 1] - r["ground_y"] < 0.1
    inf = r["nutilda_inf"]
    return (f"{len(xy)} of {r['cells']:,} cells (time {r['time']}): {behind.sum()} behind the trailing edge "
            f"(x - x_TE {np.min(xy[:, 0] - te[0]):.2f} to {np.max(xy[:, 0] - te[0]):.1f} chords), "
            f"{near_ground.sum()} within 0.1 chord of the ground, {window.sum()} in the sampled window; "
            f"distance to the wing {np.median(dw):.2f} c (median, min {dw.min():.3f}); nuTilda there "
            f"{np.median(r['nutilda']) / inf:.1f} x incoming (median), largest neighbour "
            f"{np.median(r['nb_max']) / inf:.0f} x incoming (median; the field's maximum is "
            f"{r['nutilda_max'] / inf:.0f} x)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cases", nargs="+", help="work/worker<N> folders")
    ap.add_argument("--time", help="time folder (default: the latest)")
    ap.add_argument("--list", action="store_true", help="also list every clipped cell")
    args = ap.parse_args()
    for case in args.cases:
        r = clipped_cells(case, args.time)
        print(f"{case}: {summary(r)}")
        if args.list:
            for (x, y), v, n in zip(r["xy"], r["nutilda"], r["nb_max"]):
                print(f"    x {x:8.3f}  y {y:7.3f}  height above ground {y - r['ground_y']:7.3f}  nuTilda "
                      f"{v / r['nutilda_inf']:7.2f} x incoming, largest neighbour {n / r['nutilda_inf']:7.1f} x")


if __name__ == "__main__":
    main()
