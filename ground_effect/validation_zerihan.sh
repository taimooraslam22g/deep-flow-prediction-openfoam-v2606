#!/bin/bash
# Validation in ground effect against the wind tunnel tests of Zerihan (2001) and Zerihan &
# Zhang (2000): the Tyrrell 026 front-wing main element above a moving ground. Run inside the
# Docker container:
#     cd /work/ground_effect
#     bash validation_zerihan.sh
# Then, on Windows: .venv\Scripts\python.exe ground_effect\validation_zerihan.py
#
# Needs data/airfoils/tyrrell026.dat (made by make_tyrrell026.py from Table 2 of the thesis:
# the table points, smoothed within their rounding). Conditions as in the experiment:
#  - reference incidence: quoted 1 deg = true incidence 3.45 deg to the chord line
#  - the ride height is set with the wing at quoted 0 deg (true 2.45 deg); the wing is then
#    rotated about the quarter chord (--h-alpha 2.45)
#  - Reynolds number 462,000 (thesis CFD; the tests were at 430,000-462,000): with our 1 m
#    chord and nu = 1e-5, U = 4.62 m/s
#  - heights: h/c from 0.671 down to 0.067, including the eight of Table 15; h/c 3.36, where
#    the 'freestream' tests were made (750 mm above the moving ground, in the middle of the
#    tunnel); and free air (no ground)
# Every height runs on three meshes, each with about twice the cells of the one before
# (--refine 0.7071, 1 = the default, 1.4142), for the grid convergence index. 2000
# iterations (the fine mesh needs about 1500).
#
# The 33 runs are independent, so they run JOBS at a time (default 4), each in its own
# work/workerN folder (60-92). Each run writes its output to $LOGS/<name>.log.
# About 1.5 hours with 4 at a time.
#
# To repeat the first run, with the 87 table points as they are:
#     AIRFOIL=../data/airfoils/tyrrell026_raw.dat OUT=data/validation_zerihan_raw bash validation_zerihan.sh
set -u
cd "$(dirname "$0")"
JOBS=${JOBS:-4}
AIRFOIL=${AIRFOIL:-../data/airfoils/tyrrell026.dat}
OUT=${OUT:-data/validation_zerihan}
LOGS=logs/$(basename "$OUT")
mkdir -p "$LOGS"

WING="--airfoil $AIRFOIL --alpha 3.45 --h-alpha 2.45 --U 4.62 --iterations 2000"
if [ ! -f "$AIRFOIL" ]; then
    echo "missing $AIRFOIL: run make_tyrrell026.py first"
    exit 1
fi

# name | worker | options. The longest runs (fine mesh, close to the ground) come first.
FINE="--refine 1.4142"
COARSE="--refine 0.7071"
JOBLIST="
fine_h0067|60|--h 0.067 $FINE
fine_h0090|61|--h 0.090 $FINE
fine_h0112|62|--h 0.112 $FINE
fine_h0134|63|--h 0.134 $FINE
fine_h3360|64|--h 3.36 $FINE
fine_free|65|--free-air $FINE
fine_h0671|66|--h 0.671 $FINE
fine_h0448|67|--h 0.448 $FINE
fine_h0179|68|--h 0.179 $FINE
fine_h0224|69|--h 0.224 $FINE
fine_h0313|70|--h 0.313 $FINE
medium_h0067|71|--h 0.067
medium_h0090|72|--h 0.090
medium_h0112|73|--h 0.112
medium_h0134|74|--h 0.134
medium_h0179|75|--h 0.179
medium_h0224|76|--h 0.224
medium_h0313|77|--h 0.313
medium_h0448|78|--h 0.448
medium_h0671|79|--h 0.671
medium_h3360|80|--h 3.36
medium_free|81|--free-air
coarse_h0067|82|--h 0.067 $COARSE
coarse_h0090|83|--h 0.090 $COARSE
coarse_h0112|84|--h 0.112 $COARSE
coarse_h0134|85|--h 0.134 $COARSE
coarse_h0179|86|--h 0.179 $COARSE
coarse_h0224|87|--h 0.224 $COARSE
coarse_h0313|88|--h 0.313 $COARSE
coarse_h0448|89|--h 0.448 $COARSE
coarse_h0671|90|--h 0.671 $COARSE
coarse_h3360|91|--h 3.36 $COARSE
coarse_free|92|--free-air $COARSE
"

run_one() {  # name, worker, then the options for groundGen.py
    local name=$1 worker=$2
    shift 2
    python3 groundGen.py $WING "$@" --worker "$worker" --out "$OUT/$name" > "$LOGS/$name.log" 2>&1
    echo "[$(date +%H:%M)] finished $name: $(grep -m1 -E 'done in|skipping' "$LOGS/$name.log" | sed 's/^ *//')" \
         "$(grep -m1 -E '^    (converged|NOT converged)' "$LOGS/$name.log" | sed 's/^ *//' | cut -c1-60)"
}

echo "[$(date +%H:%M)] validation against Zerihan (2001), $AIRFOIL: $(grep -c '|' <<< "$JOBLIST") runs, $JOBS at a time"
while IFS='|' read -r name worker opts; do
    [ -z "$name" ] && continue
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do
        wait -n  # wait until one of the running jobs has finished
    done
    echo "[$(date +%H:%M)] start $name"
    run_one "$name" "$worker" $opts &
done <<< "$JOBLIST"
wait
echo "=== validation finished ==="
