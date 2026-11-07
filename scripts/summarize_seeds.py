"""Collect test errors from seed*_testout.txt files and print mean +/- std.

Usage:
    python scripts/summarize_seeds.py <path/to/Deep-Flow-Prediction/train>
"""
import glob
import os
import re
import sys

import numpy as np

folder = sys.argv[1] if len(sys.argv) > 1 else "."
pattern = re.compile(r"Loss percentage \(p, v, combined\):\s*([\d.]+) %\s+([\d.]+) %\s+([\d.]+) %")

rows = []
for path in sorted(glob.glob(os.path.join(folder, "seed*_testout.txt"))):
    matches = pattern.findall(open(path).read())
    if matches:
        name = os.path.basename(path).split("_")[0]
        rows.append((name, *map(float, matches[-1])))

if not rows:
    sys.exit("no seed*_testout.txt results found in " + folder)

print(f"{'run':<8}{'pressure %':>12}{'velocity %':>12}{'combined %':>12}")
for name, p, v, c in rows:
    print(f"{name:<8}{p:>12.2f}{v:>12.2f}{c:>12.2f}")

a = np.array([r[1:] for r in rows])
mean = a.mean(axis=0)
print("-" * 44)
print(f"{'mean':<8}{mean[0]:>12.2f}{mean[1]:>12.2f}{mean[2]:>12.2f}")
if len(a) > 1:
    std = a.std(axis=0, ddof=1)  # sample standard deviation across runs
    print(f"{'std':<8}{std[0]:>12.2f}{std[1]:>12.2f}{std[2]:>12.2f}")
print(f"\n{len(rows)} runs. Paper (largest model, 12.8k samples): pressure 14.76, combined 2.6")
