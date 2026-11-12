#!/bin/bash
# Parameter-range scan: random cases exactly as the dataset will draw them (a random airfoil
# from the smoothed UIUC files, and angle of attack, ride height and air speed in the ranges
# of groundGen.py), to see where steady RANS converges and how long a sample takes, before
# the dataset is made. Run inside the Docker container:
#     cd /work/ground_effect
#     bash param_scan.sh
# Then, on Windows: .venv\Scripts\python.exe ground_effect\param_scan_report.py --plot results\param_scan.png
#
# JOBS copies of groundGen.py (default 4) run at the same time, each SAMPLES cases (default
# 50) with its own seed (SEED + its number: the scan can be repeated exactly) in its own
# work/workerN folder. Each writes its samples to $OUT/w<N>/ (samples.csv, .npz files; runs
# that do not converge go to not_converged/) and its output to $LOGS/w<N>.log. 4 x 50
# samples took 4 hours on a Ryzen 7 7730U laptop. Keep the container window open (and
# the laptop awake) until it says "scan finished". To see how far it is, in a second window
# (docker exec, or a new docker run with the same -v option):
#     grep -c saved /work/ground_effect/logs/param_scan/w*.log
# To run more samples later without repeating these, use another SEED and OUT, e.g.
#     SEED=2000 OUT=data/param_scan2 bash param_scan.sh
#
# The scan of 2026-11-10/11 (README) drew the ride height uniformly and stopped every run
# at 1000 iterations, before groundGen.py had --h-sampling and --max-iterations; SCAN_OPTS
# keeps those settings, so the same seeds give the same runs. Its 'converged' column
# still counted nuTilda clipped at the end against a run (param_scan_report.py reports
# both criteria).
SCAN_OPTS=${SCAN_OPTS:---h-sampling uniform --max-iterations 1000}
set -u
cd "$(dirname "$0")"
JOBS=${JOBS:-4}
SAMPLES=${SAMPLES:-50}
SEED=${SEED:-1000}
AIRFOILS=${AIRFOILS:-../data/airfoil_database_smooth}
OUT=${OUT:-data/param_scan}
LOGS=logs/$(basename "$OUT")

if [ ! -d "$AIRFOILS" ]; then
    echo "missing $AIRFOILS: run smooth_airfoils.py on Windows first (see the README)"
    exit 1
fi
mkdir -p "$LOGS"
echo "[$(date +%H:%M)] parameter scan: $JOBS x $SAMPLES samples, airfoils from $AIRFOILS ($(ls "$AIRFOILS" | wc -l) files)"
for w in $(seq 0 $((JOBS - 1))); do
    python3 groundGen.py --samples "$SAMPLES" --seed $((SEED + w)) --worker $((10 + w)) $SCAN_OPTS \
        --airfoil-dir "$AIRFOILS" --out "$OUT/w$w" > "$LOGS/w$w.log" 2>&1 &
    echo "[$(date +%H:%M)] started worker $w (seed $((SEED + w))), output in $LOGS/w$w.log"
done
wait
total() { grep -h -c "$1" "$LOGS"/w*.log | awk '{s += $1} END {print s}'; }  # count over all logs
echo "[$(date +%H:%M)] scan finished: $(total '^    saved') samples saved, $(total 'NOT converged') of them" \
     "not converged; $(total 'skipping') runs failed before a sample was saved"
