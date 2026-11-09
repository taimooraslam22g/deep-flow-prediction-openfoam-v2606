#!/bin/bash
# Mesh study, round 2. Run inside the Docker container:
#     cd /work/ground_effect
#     bash mesh_study.sh
# Then, on Windows: .venv\Scripts\python.exe ground_effect\mesh_study_report.py
#
# Round 1 (see README) showed that a boundary-layer mesh brings the surface pressure
# much closer to the LES, but its first cell sat at y+ 4-8 (jagged skin friction), and
# that a larger domain coarsened the cells near the wing. Round 2 uses the fixed
# boundary-layer mesh (first cell at y+ of about 1, steady cell growth away from the
# wing) and checks the domain size: 5, 20 and 40 chords.
# All runs: incoming nuTilda 3e-5, SIMPLEC, up to 2000 iterations, and the original
# nuTilda scheme (linearUpwind; the default changed after this round, see verification.sh).
set -u
cd "$(dirname "$0")"

FREE="--free-air --airfoil naca4412 --alpha 5 --U 10 --iterations 2000 --bl --nutilda-scheme linearUpwind"
GROUND="--airfoil naca23012.dat --alpha 4 --h 0.2 --U 30 --iterations 2000 --bl --nutilda-scheme linearUpwind"

echo "=== 1/6 mesh test: highest speed (thinnest first cell), S1223, ride height 0.1 ==="
python3 groundGen.py --mesh-only --bl --airfoil s1223.dat --alpha 6 --h 0.1 --U 50 --domain 20 --worker 20

echo "=== 2/6 free air (LES case), 5-chord domain ==="
python3 groundGen.py $FREE --domain 5 --worker 21 --out data/validation/r2_d5

echo "=== 3/6 free air (LES case), 20-chord domain ==="
python3 groundGen.py $FREE --domain 20 --worker 22 --out data/validation/r2_d20

echo "=== 4/6 free air (LES case), 40-chord domain ==="
python3 groundGen.py $FREE --domain 40 --worker 23 --out data/validation/r2_d40

echo "=== 5/6 ground effect, 20-chord domain ==="
python3 groundGen.py $GROUND --domain 20 --worker 24 --out data/tests/r2_d20

echo "=== 6/6 ground effect, 40-chord domain ==="
python3 groundGen.py $GROUND --domain 40 --worker 25 --out data/tests/r2_d40

echo "=== mesh study round 2 finished ==="
