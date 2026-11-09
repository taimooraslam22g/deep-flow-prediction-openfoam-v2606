#!/usr/bin/env python3
"""Plot how Cl and Cd settle during groundGen.py runs, and say when Cl got within 0.5%
of its final value.

Usage:
    python ground_effect/plot_convergence.py out.png "label=path" ["label=path" ...]

path is a run folder (ground_effect/work/workerN, has Cl and Cd) or a saved sample
(.npz, has Cl only).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")  # write to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from groundGen import read_coeffs  # noqa: E402


def history(path):
    if path.endswith(".npz"):
        return np.load(path)["Cl_history"], None
    coeffs = read_coeffs(path)
    if coeffs is None:
        sys.exit(f"no force coefficients found in {path}")
    return coeffs


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out = sys.argv[1]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    cl_range, cd_range = [], []
    for k, spec in enumerate(sys.argv[2:]):
        label, path = spec.split("=", 1)
        cl, cd = history(path)
        it = np.arange(1, len(cl) + 1)
        err = np.abs(cl - cl[-1]) / abs(cl[-1])
        bad = np.where(err > 0.005)[0]
        settled = int(bad[-1] + 2) if len(bad) else 1
        print(f"{label}: {len(cl)} iterations, final Cl {cl[-1]:+.4f}"
              + (f", Cd {cd[-1]:.4f}" if cd is not None else "")
              + f"; Cl within 0.5% of its final value from iteration {settled}")
        axes[0].plot(it, cl, color=f"C{k}", label=label)
        axes[0].axvline(settled, color=f"C{k}", ls=":", lw=1)
        axes[1].semilogy(it, np.maximum(err, 1e-6), color=f"C{k}", label=label)
        cl_range += [cl[50:].min(), cl[50:].max()]
        if cd is not None:
            axes[2].plot(it, cd, color=f"C{k}", label=label)
            cd_range += [cd[50:].min(), cd[50:].max()]
    pad = 0.1 * (max(cl_range) - min(cl_range))
    axes[0].set_ylim(min(cl_range) - pad, max(cl_range) + pad)
    axes[0].set(xlabel="iteration", ylabel="Cl", title="Cl (dotted: within 0.5% of final from here)")
    axes[1].axhline(0.005, color="gray", ls=":")
    axes[1].set(xlabel="iteration", ylabel="|Cl - final| / |final|", title="Distance from the final Cl")
    if cd_range:
        axes[2].set_ylim(0, 1.2 * max(cd_range))
    axes[2].set(xlabel="iteration", ylabel="Cd", title="Cd")
    for ax in axes:
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:  # the Cd panel is empty for .npz-only input
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print("saved", out)


if __name__ == "__main__":
    main()
