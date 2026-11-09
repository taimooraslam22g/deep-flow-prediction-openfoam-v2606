#!/bin/bash
# Mesh study: does a boundary-layer mesh and/or a larger domain bring the RANS results
# closer to the LES data? Run inside the Docker container:
#     cd /work/ground_effect
#     bash mesh_study.sh
# Then, on Windows: .venv\Scripts\python.exe ground_effect\mesh_study_report.py
#
# All runs use the chosen settings (incoming nuTilda 3e-5, SIMPLEC) and up to 2000
# iterations, so we can also see how fast each mesh converges.
#   1   mesh test only: blunt trailing edge at a very small ride height (hardest case)
#   2-5 free-air NACA 4412 at 5 deg, U = 10 m/s (Re = 1M, the LES case), in a 2x2 study:
#       original / boundary-layer mesh  x  5-chord / 20-chord domain
#   6   the ground-effect test case with the boundary-layer mesh and 20-chord domain
set -u
cd "$(dirname "$0")"

FREE="--free-air --airfoil naca4412 --alpha 5 --U 10 --iterations 2000"

echo "=== 1/6 mesh test: blunt trailing edge, ride height 0.05 ==="
python3 groundGen.py --mesh-only --bl --airfoil clarky.dat --alpha 8 --h 0.05 --worker 10

echo "=== 2/6 free air: original mesh, 5-chord domain ==="
python3 groundGen.py $FREE --worker 11 --out data/validation/orig_d5

echo "=== 3/6 free air: original mesh, 20-chord domain ==="
python3 groundGen.py $FREE --domain 20 --worker 12 --out data/validation/orig_d20

echo "=== 4/6 free air: boundary-layer mesh, 5-chord domain ==="
python3 groundGen.py $FREE --bl --worker 13 --out data/validation/bl_d5

echo "=== 5/6 free air: boundary-layer mesh, 20-chord domain ==="
python3 groundGen.py $FREE --bl --domain 20 --worker 14 --out data/validation/bl_d20

echo "=== 6/6 ground effect: boundary-layer mesh, 20-chord domain ==="
python3 groundGen.py --airfoil naca23012.dat --alpha 4 --h 0.2 --U 30 --iterations 2000 \
    --bl --domain 20 --worker 15 --out data/tests/bl_d20

echo "=== mesh study finished ==="
