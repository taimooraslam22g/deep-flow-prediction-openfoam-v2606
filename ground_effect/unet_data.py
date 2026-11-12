"""Ground-effect samples for the U-Net: loading, normalisation, the validation split and the
error measure of Thuerey et al. Used by train_unet.py and learning_curve.py.

A sample (groundGen.py) is an .npz file whose array 'a' (6 x 128 x 128, on the grid x -0.5..1.5,
y -1..1 chords) holds the inputs 0 freestream x velocity (U in the air, 0 inside the wing and
below the ground), 1 freestream y velocity (always 0: the wing is turned by the angle of attack
and the air comes horizontally), 2 mask (1 inside the wing and below the ground), and the
targets 3 pressure p (kinematic, p/rho), 4 Ux, 5 Uy. Runs whose Cl did not become steady are in
not_converged/ folders, which are skipped. The original free-air samples
(Deep-Flow-Prediction/data/train, test) have the same array, so they load too.

The normalisation is that of the original code (Deep-Flow-Prediction/train/dataset.py) with its
fixed maxima, so that the model trained on the free-air data can be used as the starting point:
  1. pressure: subtract the mean, then set p = 0 inside the mask;
  2. make dimensionless with the freestream speed U: p / U^2, Ux / U, Uy / U;
  3. divide by the fixed maxima: inputs 100, 38.12, 1; targets 4.65, 2.04, 2.37.
One change: in step 1 the original subtracts the mean over all 128 x 128 points, including the
masked points, where p is 0. For an airfoil alone (1-3% of the points masked) that is almost the
mean over the air, but the ground masks 16-47% of the points, so the original mean would shift
each pressure field by an amount set by the size of the masked area. Here the mean is taken over
the air only (p_offset="fluid"); p_offset="grid" gives the original.
"""
import glob
import os
import zlib

import numpy as np

RES = 128
MAX_INPUTS = (100.0, 38.12, 1.0)  # the original fixed maxima: freestream x, freestream y, mask
MAX_TARGETS = (4.65, 2.04, 2.37)  # p / U^2, Ux / U, Uy / U
H_EDGES = tuple(0.05 * 12 ** (np.arange(6) / 5))  # 5 geometric bins over h/c 0.05-0.6, as param_scan_report.py
ALPHA_EDGES = (0, 2, 4, 6, 8, 10)
U_EDGES = (10, 20, 30, 40, 50)


def sample_files(folder):
    """The .npz samples in folder and its subfolders (w0, w1, ...), without the not_converged/
    folders, sorted by their path in folder (so the order does not depend on the file system)."""
    rel = [os.path.relpath(f, folder) for f in glob.glob(os.path.join(folder, "**", "*.npz"), recursive=True)]
    rel = sorted(r.replace(os.sep, "/") for r in rel if "not_converged" not in r.split(os.sep))
    return [os.path.join(folder, r) for r in rel]


def normalise(a, p_offset="fluid"):
    """The normalised inputs and targets (each 3 x 128 x 128, float32) of one sample array a
    (6 x 128 x 128), as described at the top, and the freestream speed U."""
    inputs = np.array(a[:3], dtype=np.float64)
    targets = np.array(a[3:6], dtype=np.float64)
    if p_offset == "fluid":
        targets[0] -= targets[0][inputs[2] == 0].mean()
    elif p_offset == "grid":
        targets[0] -= targets[0].mean()
    else:
        raise ValueError(f"p_offset must be 'fluid' or 'grid', not {p_offset!r}")
    targets[0] -= targets[0] * inputs[2]  # p = 0 inside the mask
    speed = np.hypot(np.abs(inputs[0]).max(), np.abs(inputs[1]).max())  # the original's v_norm
    targets[0] /= speed ** 2
    targets[1:] /= speed
    for c in range(3):
        inputs[c] /= MAX_INPUTS[c]
        targets[c] /= MAX_TARGETS[c]
    return inputs.astype(np.float32), targets.astype(np.float32), speed


def denormalise(targets, speed):
    """Undo steps 2 and 3 of the normalisation for one sample (3 x 128 x 128): p in m^2/s^2
    (p/rho, mean over the air removed), Ux and Uy in m/s."""
    out = np.array(targets, dtype=np.float64)
    for c in range(3):
        out[c] *= MAX_TARGETS[c]
    out[0] *= speed ** 2
    out[1:] *= speed
    return out


def load(folder, p_offset="fluid", exclude_clipped=False):
    """All samples in folder (see sample_files), normalised. Returns a dict with 'inputs' and
    'targets' (N x 3 x 128 x 128, float32), 'file' (path in folder), 'airfoil', 'h' (h/c),
    'alpha' (deg), 'U' (m/s) and 'clipped' (nuTilda still clipped at the end of the run). The
    original free-air files have only the array: there h and alpha are NaN, clipped is False and
    the airfoil name comes from the file name."""
    files = sample_files(folder)
    if not files:
        raise SystemExit(f"no samples (.npz files) in {folder}")
    n = len(files)
    data = {"inputs": np.empty((n, 3, RES, RES), np.float32), "targets": np.empty((n, 3, RES, RES), np.float32),
            "file": [], "airfoil": [], "h": np.full(n, np.nan), "alpha": np.full(n, np.nan),
            "U": np.empty(n), "clipped": np.zeros(n, bool)}
    for i, f in enumerate(files):
        with np.load(f) as d:
            data["inputs"][i], data["targets"][i], data["U"][i] = normalise(d["a"], p_offset)
            data["file"].append(os.path.relpath(f, folder).replace(os.sep, "/"))
            data["airfoil"].append(str(d["airfoil"]) if "airfoil" in d.files else os.path.basename(f).split("_")[0])
            for key in ("h", "alpha"):
                if key in d.files:
                    data[key][i] = float(d[key])
            if "clipped" in d.files:
                data["clipped"][i] = bool(d["clipped"])
    return subset(data, ~data["clipped"]) if exclude_clipped else data


def subset(data, which):
    """The samples which (indices or a boolean array) of a dict from load()."""
    which = np.asarray(which)
    idx = np.flatnonzero(which) if which.dtype == bool else which.astype(int)
    return {k: [v[i] for i in idx] if isinstance(v, list) else v[idx] for k, v in data.items()}


def is_validation(file, percent):
    """True for about percent % of the samples, chosen by a checksum (CRC-32) of the file name,
    so the same samples stay in the validation set when more samples are added later."""
    return zlib.crc32(os.path.basename(file).encode()) % 100 < percent


def relative_errors(pred, target):
    """The relative L1 errors of each sample, as in the original test script (runTest.py), in
    normalised units over the whole 128 x 128 grid (the target is 0 in the mask, so any output
    there counts as error): pressure sum|p - p_cfd| / sum|p_cfd|, velocity (sum|dUx| + sum|dUy|) /
    (sum|Ux| + sum|Uy|), and all three channels together ('all'). The paper's numbers are the
    means over the test samples. pred, target: N x 3 x 128 x 128."""
    pred, target = np.asarray(pred, np.float64), np.asarray(target, np.float64)
    diff = np.abs(pred - target).sum(axis=(2, 3))
    size = np.abs(target).sum(axis=(2, 3))
    return {"p": diff[:, 0] / size[:, 0], "v": (diff[:, 1] + diff[:, 2]) / (size[:, 1] + size[:, 2]),
            "all": diff.sum(axis=1) / size.sum(axis=1)}


def by_bins(errors, values, edges):
    """The mean errors and the number of samples in each bin of values (the last bin includes
    its upper end), as a list of dicts with lo, hi, n, p, v, all."""
    values = np.asarray(values, float)
    rows = []
    for i in range(len(edges) - 1):
        sel = (values >= edges[i]) & ((values < edges[i + 1]) | ((i == len(edges) - 2) & (values == edges[-1])))
        rows.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": int(sel.sum()),
                     **{k: float(errors[k][sel].mean()) if sel.any() else float("nan") for k in ("p", "v", "all")}})
    return rows
