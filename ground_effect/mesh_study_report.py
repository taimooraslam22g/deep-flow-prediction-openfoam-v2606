#!/usr/bin/env python3
"""Figures and tables for the mesh study, after mesh_study.sh has run in the container.

Usage (on Windows, with the project's Python):
    .venv\\Scripts\\python.exe ground_effect\\mesh_study_report.py

Writes to results/:
    validation_mesh_study.png   Cp and Cf vs the LES for the 2x2 mesh/domain study
    mesh_bl_naca4412.png        boundary-layer mesh around the NACA 4412
    mesh_bl_ground.png          boundary-layer mesh in ground effect
    mesh_bl_clarky_h005.png     hardest mesh test: blunt trailing edge, ride height 0.05
    convergence_bl_ground.png   Cl history in ground effect, original vs boundary-layer mesh
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LES = os.path.join(ROOT, "data", "wing_les", "WingData", "naca4412.mat")
NAME = "naca4412_free_a+0500_U0100.npz"
GROUND = "naca23012_h0200_a+0400_U0300.npz"


def path(*parts):
    return os.path.join(HERE, *parts)


def run(script, *args):
    print(f"\n>>> {script}")
    subprocess.run([sys.executable, path(script), *args])


def main():
    out = os.path.join(ROOT, "results")
    study = {"original mesh, 5c": path("data", "validation", "orig_d5", NAME),
             "original mesh, 20c": path("data", "validation", "orig_d20", NAME),
             "BL mesh, 5c": path("data", "validation", "bl_d5", NAME),
             "BL mesh, 20c": path("data", "validation", "bl_d20", NAME)}
    found = {k: v for k, v in study.items() if os.path.exists(v)}
    for k in study.keys() - found.keys():
        print(f"missing: {k} ({study[k]})")
    if found:
        run("validate_les.py", LES, "1M", os.path.join(out, "validation_mesh_study.png"),
            *[f"{k}={v}" for k, v in found.items()])

    for worker, png in [("worker14", "mesh_bl_naca4412.png"), ("worker15", "mesh_bl_ground.png"),
                        ("worker10", "mesh_bl_clarky_h005.png")]:
        if os.path.exists(path("work", worker, "airfoil.msh")):
            run("plot_mesh.py", path("work", worker), os.path.join(out, png))

    before = path("data", "tests", "turb_realistic_simplec", GROUND)
    after = path("data", "tests", "bl_d20", GROUND)
    if os.path.exists(before) and os.path.exists(after):
        run("plot_convergence.py", os.path.join(out, "convergence_bl_ground.png"),
            f"original mesh, 5c={before}", f"BL mesh, 20c={after}")
        print("\nGround-effect test case (NACA 23012, 4 deg, h/c 0.2, 30 m/s):")
        for label, p in [("original mesh, 5c", before), ("BL mesh, 20c", after)]:
            z = np.load(p)
            print(f"   {label:20s} Cl {float(z['Cl']):+.4f}  Cd {float(z['Cd']):.4f}  "
                  f"iterations {int(z['iterations'])}")


if __name__ == "__main__":
    main()
