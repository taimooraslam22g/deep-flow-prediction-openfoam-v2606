#!/bin/bash
# Checks after the parameter scan (param_scan.sh), on ten of its cases, rerun with exactly
# the same airfoil, angle of attack, ride height and air speed. Run inside the Docker
# container:
#     cd /work/ground_effect
#     bash clip_test.sh
# Then, on Windows: .venv\Scripts\python.exe ground_effect\clip_test_report.py --plot results\clip_test.png
#
# A. What does the nuTilda clipping change? Four runs whose Cl was steady while OpenFOAM
#    clipped negative nuTilda in nearly every iteration (h/c 0.05 to 0.4, attached and
#    separated flow). Each runs twice more:
#    - <name>_2000: the same settings for 2000 iterations. Does the flow still change after
#      the 1000 iterations of the scan? (The first 1000 iterations repeat the scan run
#      exactly.)
#    - <name>_upwind: first-order upwind convection for nuTilda. It cannot undershoot, so
#      there is nothing to clip. How far is the clipped flow from it?
# B. Do runs whose Cl was still changing after 1000 iterations converge when they continue
#    (up to 2000 iterations while Cl is settling, the first setting)? Four runs that were
#    settling and two that oscillated (those should stop at 1000). One of them also runs
#    straight to 2000 iterations (fxs03182_2000): continuing a run must give the same
#    answer as running it longer from the start.
# C. (added after A and B) In the two separated cases of A, upwind gave 5-8% less downforce.
#    Is that the clipping, or the first-order scheme? The two other second-order schemes
#    of groundGen.py: linearUpwind with an unlimited gradient (the original case; it clips
#    far more) and the TVD scheme limitedLinear. And fxs03182, which was still settling
#    at 2000 iterations, with up to 3000 (the new default).
#
# Runs that already have results are skipped, so running the script again only runs what
# is missing (part C after A and B). JOBS at a time (default 4), each in its own work folder
# (work/worker100-119, kept for clip_test_report.py, which also finds the clipped cells
# there). Logs in logs/clip_test/<name>.log. A and B (15 runs): 30 minutes with 4 at a
# time; C (5 runs): about 15 minutes.
set -u
cd "$(dirname "$0")"
JOBS=${JOBS:-4}
OUT=${OUT:-data/clip_test}
LOGS=logs/$(basename "$OUT")
AIRFOILS=${AIRFOILS:-../data/airfoil_database_smooth}
if [ ! -d "$AIRFOILS" ]; then
    echo "missing $AIRFOILS: run smooth_airfoils.py on Windows first (see the README)"
    exit 1
fi
mkdir -p "$LOGS"

# the scan cases (values as stored in the scan's .npz files)
GOE280="--airfoil goe280.dat --alpha 6.70139967701536 --h 0.10904474517394186 --U 32.63979546072337"
E186="--airfoil e186.dat --alpha 3.479132017067148 --h 0.25127455670988635 --U 49.59010755233223"
GOE281="--airfoil goe281.dat --alpha 1.4273570037339678 --h 0.4000607827603648 --U 27.161941797537793"
N9="--airfoil n9.dat --alpha 3.8180206306540256 --h 0.051724584260071184 --U 25.03760470021577"
GOE553="--airfoil goe553.dat --alpha 5.287590256200526 --h 0.1550699540444338 --U 21.261823945674067"
E337="--airfoil e337.dat --alpha 4.404498352449824 --h 0.1093691351898457 --U 32.679363051843524"
FX08S176="--airfoil fx08s176.dat --alpha 0.5770396696096025 --h 0.14404090212555298 --U 44.2597561603522"
FXS03182="--airfoil fxs03182.dat --alpha 8.355254093543387 --h 0.15147857111239954 --U 17.35007970068219"
GOE284="--airfoil goe284.dat --alpha 8.298203543403416 --h 0.1101576421823104 --U 48.80454036720638"
GOE225="--airfoil goe225.dat --alpha 3.8886987310444376 --h 0.07493597694583254 --U 24.32778638419107"
FIRST="--max-iterations 2000"  # A and B ran before the default became 3000

# name | worker | options. The longest runs come first.
JOBLIST="
goe284|100|$GOE284 $FIRST
fxs03182|101|$FXS03182 $FIRST
fxs03182_2000|114|$FXS03182 --iterations 2000 $FIRST
goe280_2000|102|$GOE280 --iterations 2000 $FIRST
e186_2000|103|$E186 --iterations 2000 $FIRST
goe281_2000|104|$GOE281 --iterations 2000 $FIRST
n9_2000|105|$N9 --iterations 2000 $FIRST
e337|106|$E337 $FIRST
fx08s176|107|$FX08S176 $FIRST
goe553|108|$GOE553 $FIRST
goe225|109|$GOE225 $FIRST
goe280_upwind|110|$GOE280 --nutilda-scheme upwind $FIRST
e186_upwind|111|$E186 --nutilda-scheme upwind $FIRST
goe281_upwind|112|$GOE281 --nutilda-scheme upwind $FIRST
n9_upwind|113|$N9 --nutilda-scheme upwind $FIRST
fxs03182_3000|119|$FXS03182 --max-iterations 3000
goe280_linearUpwind|115|$GOE280 --nutilda-scheme linearUpwind
n9_linearUpwind|116|$N9 --nutilda-scheme linearUpwind
goe280_limitedLinear|117|$GOE280 --nutilda-scheme limitedLinear
n9_limitedLinear|118|$N9 --nutilda-scheme limitedLinear
"

run_one() {  # name, worker, then the options for groundGen.py
    local name=$1 worker=$2
    shift 2
    python3 groundGen.py --airfoil-dir "$AIRFOILS" "$@" --worker "$worker" --out "$OUT/$name" \
        > "$LOGS/$name.log" 2>&1
    echo "[$(date +%H:%M)] finished $name: $(grep -m1 -E 'done in|skipping' "$LOGS/$name.log" | sed 's/^ *//')" \
         "$(grep -m1 -E '^    (converged|NOT converged)' "$LOGS/$name.log" | sed 's/^ *//' | cut -c1-60)"
}

todo=$(while IFS='|' read -r name worker opts; do
    [ -n "$name" ] && [ ! -f "$OUT/$name/samples.csv" ] && echo "$name|$worker|$opts"
done <<< "$JOBLIST")
echo "[$(date +%H:%M)] clipping and continuation test: $(grep -c '|' <<< "$todo") of" \
     "$(grep -c '|' <<< "$JOBLIST") runs still to do, $JOBS at a time"
while IFS='|' read -r name worker opts; do
    [ -z "$name" ] && continue
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do
        wait -n  # wait until one of the running jobs has finished
    done
    echo "[$(date +%H:%M)] start $name"
    run_one "$name" "$worker" $opts &
done <<< "$todo"
wait
echo "[$(date +%H:%M)] === test finished ==="
