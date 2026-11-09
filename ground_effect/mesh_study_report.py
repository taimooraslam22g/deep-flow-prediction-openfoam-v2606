#!/usr/bin/env python3
"""Figures and tables for the mesh study, round 2, after mesh_study.sh has run in the
container. Round 1 results (if present) are included for comparison.

Usage (on Windows, with the project's Python):
    .venv\\Scripts\\python.exe ground_effect\\mesh_study_report.py

Writes to results/:
    validation_mesh_progress.png  Cp and Cf vs the LES: original mesh, round 1, round 2
    validation_domain_study.png   Cp and Cf vs the LES for 5, 20 and 40-chord domains
    mesh_bl2_naca4412.png         round-2 boundary-layer mesh around the NACA 4412
    mesh_bl2_ground.png           round-2 boundary-layer mesh in ground effect
    mesh_bl2_s1223_U50.png        mesh test at the highest speed (thinnest first cell)
    convergence_bl2_ground.png    Cl history in ground effect
and prints y+ and cell sizes for each mesh (mesh_quality.py).
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from mesh_quality import report as mesh_report  # noqa: E402

LES = os.path.join(ROOT, "data", "wing_les", "WingData", "naca4412.mat")
FREE = "naca4412_free_a+0500_U0100.npz"
GROUND = "naca23012_h0200_a+0400_U0300.npz"
OUT = os.path.join(ROOT, "results")


def path(*parts):
    return os.path.join(HERE, *parts)


def run(script, *args):
    print(f"\n>>> {script}")
    subprocess.run([sys.executable, path(script), *args])


def existing(runs):
    found = {k: v for k, v in runs.items() if os.path.exists(v)}
    for k in runs.keys() - found.keys():
        print(f"missing: {k} ({runs[k]})")
    return found


def validate(png, runs):
    found = existing(runs)
    if found:
        run("validate_les.py", LES, "1M", os.path.join(OUT, png), *[f"{k}={v}" for k, v in found.items()])


def main():
    validate("validation_mesh_progress.png", {
        "original mesh, 5c": path("data", "validation", "orig_d5", FREE),
        "round 1: BL y+ 5, 20c": path("data", "validation", "bl_d20", FREE),
        "round 2: BL y+ 1, 20c": path("data", "validation", "r2_d20", FREE)})
    validate("validation_domain_study.png", {
        "round 2, 5c": path("data", "validation", "r2_d5", FREE),
        "round 2, 20c": path("data", "validation", "r2_d20", FREE),
        "round 2, 40c": path("data", "validation", "r2_d40", FREE)})

    print("\n>>> mesh quality")
    for worker, npz in [("worker21", path("data", "validation", "r2_d5", FREE)),
                        ("worker22", path("data", "validation", "r2_d20", FREE)),
                        ("worker23", path("data", "validation", "r2_d40", FREE)),
                        ("worker24", path("data", "tests", "r2_d20", GROUND)),
                        ("worker25", path("data", "tests", "r2_d40", GROUND)),
                        ("worker20", None)]:
        if os.path.exists(path("work", worker, "airfoil.msh")):
            mesh_report(path("work", worker), npz if npz and os.path.exists(npz) else None)

    for worker, png in [("worker22", "mesh_bl2_naca4412.png"), ("worker24", "mesh_bl2_ground.png"),
                        ("worker20", "mesh_bl2_s1223_U50.png")]:
        if os.path.exists(path("work", worker, "airfoil.msh")):
            run("plot_mesh.py", path("work", worker), os.path.join(OUT, png))

    ground = existing({"original mesh, 5c": path("data", "tests", "turb_realistic_simplec", GROUND),
                       "round 1: BL y+ 5, 20c": path("data", "tests", "bl_d20", GROUND),
                       "round 2: BL y+ 1, 20c": path("data", "tests", "r2_d20", GROUND),
                       "round 2: BL y+ 1, 40c": path("data", "tests", "r2_d40", GROUND)})
    if ground:
        run("plot_convergence.py", os.path.join(OUT, "convergence_bl2_ground.png"),
            *[f"{k}={v}" for k, v in ground.items()])
        print("\nGround-effect test case (NACA 23012, 4 deg, h/c 0.2, 30 m/s):")
        for label, p in ground.items():
            z = np.load(p)
            print(f"   {label:24s} Cl {float(z['Cl']):+.4f}  Cd {float(z['Cd']):.4f}  "
                  f"iterations {int(z['iterations'])}")


if __name__ == "__main__":
    main()
