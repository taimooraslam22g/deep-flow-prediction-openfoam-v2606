#!/bin/bash
# The ground-effect dataset: random cases, each a random airfoil from the smoothed UIUC
# files at a random angle of attack (0-10 deg), ride height (h/c 0.05-0.6, drawn
# log-uniformly) and air speed (10-50 m/s), with groundGen.py's settings from the
# validation, the mesh study and the parameter scan (README). Run inside the Docker
# container:
#     cd /work/ground_effect
#     JOBS=8 bash dataset.sh                 # training set: 1000 runs, 8 at a time
#     OUT=data/dataset/test AIRFOILS=../data/airfoil_database_test_smooth RUNS=120 SEED=9000 WORKER=300 bash dataset.sh
#                                            # test set: the 30 airfoils kept out of training
#
# RUNS runs in all (default 1000), shared by JOBS copies of groundGen.py running at the same
# time (default 4). Copy w uses the random seed SEED + w and its own work folder
# (work/worker<WORKER + w>), writes its samples to $OUT/w<w>/ (samples.csv and one .npz
# per run; runs whose Cl does not become steady go to not_converged/) and its output to
# logs/dataset_<name of OUT>/s<seed>.log. In the parameter scan, a run took 4.7 minutes
# (median) with 4 at a time, about 47 runs an hour; about 80% of the runs should give a
# sample. Keep the container window open and the laptop awake and plugged in.
#
# Progress, from a second window (docker exec, or a new docker run with the same -v):
#     grep -c "^    saved" /work/ground_effect/logs/dataset_train/*.log
# To stop: Ctrl+C stops this script but not the runs in the background, so then type exit
# to close the container, which stops them (the runs that are not finished are lost, the
# saved samples stay). To go on later, or to add more samples, run it again with a NEW
# SEED (e.g. SEED=3000): the same seed would repeat the same cases.
set -u
cd "$(dirname "$0")"
JOBS=${JOBS:-4}
RUNS=${RUNS:-1000}
SEED=${SEED:-2000}
WORKER=${WORKER:-200}
AIRFOILS=${AIRFOILS:-../data/airfoil_database_smooth}
OUT=${OUT:-data/dataset/train}
LOGS=logs/dataset_$(basename "$OUT")
# the dataset's settings, written out so that later changes of the defaults cannot change it
DATA_OPTS="--h-sampling log --iterations 1000 --max-iterations 3000"
SAMPLES=$(( (RUNS + JOBS - 1) / JOBS ))

if [ ! -d "$AIRFOILS" ]; then
    echo "missing $AIRFOILS: run smooth_airfoils.py on Windows first (see the README)"
    exit 1
fi
mkdir -p "$LOGS"
for w in $(seq 0 $((JOBS - 1))); do
    if [ -f "$LOGS/s$((SEED + w)).log" ]; then
        echo "$LOGS/s$((SEED + w)).log exists: these seeds were used before. Use a new SEED (e.g. SEED=$((SEED + 1000)))."
        exit 1
    fi
done
echo "[$(date +%H:%M)] dataset: $JOBS x $SAMPLES runs (seeds $SEED-$((SEED + JOBS - 1))), airfoils from" \
     "$AIRFOILS ($(ls "$AIRFOILS" | wc -l) files), samples to $OUT"
for w in $(seq 0 $((JOBS - 1))); do
    python3 groundGen.py --samples "$SAMPLES" --seed $((SEED + w)) --worker $((WORKER + w)) $DATA_OPTS \
        --airfoil-dir "$AIRFOILS" --out "$OUT/w$w" > "$LOGS/s$((SEED + w)).log" 2>&1 &
    echo "[$(date +%H:%M)] started copy $w (seed $((SEED + w))), output in $LOGS/s$((SEED + w)).log"
done
wait
total() {  # count a line pattern over the logs of this start
    for w in $(seq 0 $((JOBS - 1))); do grep -c "$1" "$LOGS/s$((SEED + w)).log"; done | awk '{s += $1} END {print s}'
}
echo "[$(date +%H:%M)] finished: $(total '^    saved') runs saved, $(total 'NOT converged') of them not converged" \
     "(in not_converged/); $(total 'skipping') runs failed before a sample was saved"
