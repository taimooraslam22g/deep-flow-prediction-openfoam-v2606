#!/bin/bash
# Mesh study, round 3: numerical verification. Run inside the Docker container:
#     cd /work/ground_effect
#     bash verification.sh
# Then, on Windows: .venv\Scripts\python.exe ground_effect\verification_report.py
#
# All runs use the settings chosen in round 2 (boundary-layer mesh with the first cell at
# y+ of about 1, 20-chord domain, incoming nuTilda 3e-5, SIMPLEC), up to 2000 iterations.
# Three questions:
#  A. nuTilda scheme. In round 2, OpenFOAM clipped negative nuTilda values in almost every
#     iteration, so the nuTilda residual never converged. Does a limited scheme
#     (cellLimited, the new default, or limitedLinear) fix that, and does the answer change?
#  B. Grid convergence. Two cases, each on three meshes whose cells are sqrt(2) times
#     smaller from one to the next (--refine 0.7071, 1, 1.4142: about twice the cells per
#     step). This gives the grid convergence index (GCI) of Cl, Cd and the flow field.
#     Cases: the usual test case, and a hard one (the very cambered S1223 close to the
#     ground at the highest speed).
#  C. Domain size at the largest ride height (0.6 chords): 20 vs 40 chords.
#
# The 11 runs are independent, so they run JOBS at a time (default 4), each on its own CPU
# core and in its own work/workerN folder. Each run writes its output to
# logs/verification/<name>.log; this window only shows when runs start and finish.
# About an hour with 4 at a time. For fewer at a time: JOBS=2 bash verification.sh
set -u
cd "$(dirname "$0")"
JOBS=${JOBS:-4}
LOGS=logs/verification
mkdir -p "$LOGS"

SET="--bl --domain 20 --solver simplec --nutilda 3e-5 --iterations 2000"
FREE="--free-air --airfoil naca4412 --alpha 5 --U 10"       # the LES case (Re = 1M)
GROUND="--airfoil naca23012.dat --alpha 4 --h 0.2 --U 30"   # the usual test case
HARD="--airfoil s1223.dat --alpha 6 --h 0.1 --U 50"         # the hard case
HIGH="--airfoil naca23012.dat --alpha 4 --h 0.6 --U 30"     # the largest ride height

# name | worker | options. The longest runs come first, so the batch ends sooner.
JOBLIST="
hard_fine|39|$HARD --refine 1.4142 --nutilda-scheme cellLimited
ground_fine|36|$GROUND --refine 1.4142 --nutilda-scheme cellLimited
hard_medium|38|$HARD --nutilda-scheme cellLimited
ground_medium|33|$GROUND --nutilda-scheme cellLimited
ground_limitedLinear|34|$GROUND --nutilda-scheme limitedLinear
high_d40|41|$HIGH --domain 40 --nutilda-scheme cellLimited
high_d20|40|$HIGH --nutilda-scheme cellLimited
free_cellLimited|31|$FREE --nutilda-scheme cellLimited
free_limitedLinear|32|$FREE --nutilda-scheme limitedLinear
hard_coarse|37|$HARD --refine 0.7071 --nutilda-scheme cellLimited
ground_coarse|35|$GROUND --refine 0.7071 --nutilda-scheme cellLimited
"

run_one() {  # name, worker, then the options for groundGen.py
    local name=$1 worker=$2
    shift 2
    python3 groundGen.py $SET "$@" --worker "$worker" --out "data/verification/$name" \
        > "$LOGS/$name.log" 2>&1
    echo "[$(date +%H:%M)] finished $name: $(grep -m1 -E 'done in|skipping' "$LOGS/$name.log" | sed 's/^ *//')"
}

echo "[$(date +%H:%M)] verification: $(grep -c '|' <<< "$JOBLIST") runs, $JOBS at a time"
while IFS='|' read -r name worker opts; do
    [ -z "$name" ] && continue
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do
        wait -n  # wait until one of the running jobs has finished
    done
    echo "[$(date +%H:%M)] start $name"
    run_one "$name" "$worker" $opts &
done <<< "$JOBLIST"
wait
echo "=== verification finished ==="
