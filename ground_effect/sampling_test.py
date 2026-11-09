#!/usr/bin/env python3
"""Check the row-by-row sampling (groundGen.py --sampling lines) on finished runs.

The original setup samples the 128 x 128 output grid as one 'cloud' of 16,384 points.
Before the first iteration, OpenFOAM then searches the mesh for every point, which took
2.5 to 20 minutes per run in the mesh study (round 3), often longer than the solver.
groundGen.py now samples the same points as 128 straight lines, one per grid row
(OpenFOAM's 'uniform' set): OpenFOAM searches only for where each line starts and then
walks along it. The interpolation (cellPoint) is the same.

For each finished run given (a work folder and the sample it produced with the cloud),
this script copies the case, swaps in the line sampling, runs postProcess on the final
fields, times it, and compares the resulting grid with the stored one. Run inside the
container:
    cd /work/ground_effect
    python3 sampling_test.py work/worker24 data/tests/r2_d20/naca23012_h0200_a+0400_U0300.npz \\
                             work/worker22 data/validation/r2_d20/naca4412_free_a+0500_U0100.npz
"""
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from groundGen import latest_time_dir, line_sets, read_rows, startup_seconds, tail, to_grid  # noqa: E402


def test(case, npz, work):
    name = os.path.basename(os.path.normpath(case))
    copy = os.path.join(work, name)
    shutil.copytree(case, copy, ignore=shutil.ignore_patterns("postProcessing"))
    with open(os.path.join(copy, "system", "internalCloud"), "w") as f:
        f.write(line_sets())
    start = time.time()
    with open(os.path.join(copy, "log.postProcess"), "w") as log:
        code = subprocess.call("postProcess -func internalCloud -latestTime", shell=True, cwd=copy,
                               stdout=log, stderr=subprocess.STDOUT)
    seconds = time.time() - start
    folder = latest_time_dir(os.path.join(copy, "postProcessing", "internalCloud"))
    if code != 0 or folder is None:
        print(f"{name}: postProcess failed, end of its log:\n{tail(os.path.join(copy, 'log.postProcess'), 20)}")
        return
    rows, extra = read_rows(folder)
    z = np.load(npz)
    new, old = to_grid(rows, float(z["U"])), z["a"]
    print(f"{name}: line sampling took {seconds:.1f} s for {len(rows):,} grid points"
          f"{f' (and {extra} extra points, dropped)' if extra else ''}; at the start of the "
          f"original run, the first iteration ended after {startup_seconds(os.path.join(case, 'log.simpleFoam')):.0f} s")
    print(f"    mask identical: {np.array_equal(new[2], old[2])} "
          f"({int(new[2].sum())} vs {int(old[2].sum())} masked points)")
    fluid = (new[2] == 0) & (old[2] == 0)
    for c, label in [(3, "p"), (4, "Ux"), (5, "Uy")]:
        d = np.abs(new[c] - old[c])[fluid]
        scale = np.abs(old[c][fluid]).max()
        print(f"    {label}: largest difference {d.max():.3g} ({100 * d.max() / scale:.2g}% of the largest value), "
              f"mean {d.mean():.3g}")


def main():
    args = sys.argv[1:]
    if not args or len(args) % 2:
        sys.exit(__doc__)
    work = os.path.join(HERE, "work", "sampling_test")
    if os.path.exists(work):
        shutil.rmtree(work)
    os.makedirs(work)
    for case, npz in zip(args[::2], args[1::2]):
        test(case, npz, work)


if __name__ == "__main__":
    main()
