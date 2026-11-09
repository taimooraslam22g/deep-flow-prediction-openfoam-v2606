# Deep Flow Prediction: Reproduction on OpenFOAM v2606

A reproduction of **Thuerey et al., "Deep Learning Methods for Reynolds-Averaged
Navier-Stokes Simulations of Airfoil Flows"** (*AIAA Journal*, 2020) on current
software: OpenFOAM v2606, gmsh 4 and Docker on Windows. It is the first step
towards an extension to **inverted airfoils in ground effect** (F1 front wings).

- Paper: https://arxiv.org/abs/1810.08217
- Original code: https://github.com/thunil/Deep-Flow-Prediction (Apache-2.0)

---

## Progress

| Stage | Status |
|---|---|
| 1. Environment (Docker + OpenFOAM v2606 + gmsh + Python) | Done |
| 2. Port the data generation pipeline to OpenFOAM v2606 | Done |
| 3. First simulation and training sample | Done (about 95 s per sample on a laptop) |
| 4. Download the authors' precomputed dataset | Done (`data_6k`: 6,400 train / 90 test, SHA-512 verified) |
| 5. Train the U-Net and reproduce the paper's error numbers | Done: 1 run + 5 repeated seeds (CPU, about 47 min each). See [Results](#results) |
| 6. Compare my results with the paper | Done: 5-seed mean ± std, plus an 80k-iteration run that matches the paper's pressure and velocity errors |
| 7. Extension: inverted airfoil in ground effect | In progress: setup works. Turbulence level and solver chosen by validation against LES. Next: mesh study, then the dataset. See [Extension](#extension-inverted-airfoil-in-ground-effect-in-progress) |

### Log
- **2026-11-07**: Set up the Docker environment. Found and fixed four
  incompatibilities between the original OpenFOAM 5 setup and OpenFOAM v2606
  (see below). Generated and plotted the first training sample.
- **2026-11-07**: Downloaded the authors' `data_6k` dataset
  ([TUM data service](https://dataserv.ub.tum.de/index.php/s/m1470791),
  1.3 GB): 6,400 training and 90 test samples. Checksum verified. The training
  code expects it in `Deep-Flow-Prediction/data/train` and `data/test`.
- **2026-11-07**: First training run on CPU (Ryzen 7 7730U, PyTorch 2.15 CPU).
  U-Net with `expo = 5` (2.33M parameters) trained for 10k iterations in 48 min.
  Test error on 90 unseen airfoils: 3.17% overall, 2.61% velocity, 16.6%
  pressure, close to the paper (see Results).
- **2026-11-07**: Repeated the same training with 5 fixed random seeds
  (`scripts/run_seeds.ps1`, 3 h 55 min unattended). Combined test error
  3.42 ± 0.18%, velocity 2.85 ± 0.13%, pressure 17.06 ± 0.94%. Run 1 (3.17%)
  was at the lucky end of the spread.
- **2026-11-08**: Overnight run with the paper's training length (80k
  iterations, 6 h on CPU). Combined error 2.96%, pressure 14.86%, velocity
  2.45%. Pressure and velocity now match the paper's best model.
- **2026-11-08**: Started the ground-effect extension
  ([`ground_effect/`](ground_effect/)): an inverted airfoil above a moving
  ground, saved in the same sample format as the original dataset.
- **2026-11-09**: First ground-effect simulations. 500 solver iterations (the
  original setting) leave Cl about 14% short of its converged value. The
  incoming turbulence level changes downforce by 9% and halves drag (see
  [Extension](#extension-inverted-airfoil-in-ground-effect-in-progress)).
- **2026-11-09**: Validated the setup against LES of a NACA 4412 at Re = 1M.
  The realistic turbulence level (3ν) predicts surface pressure clearly better
  than the original 0.14. Skin friction and drag are off with both, because of
  the mesh. SIMPLEC gives the same answer as SIMPLE in 3.5× fewer iterations.
  Both choices are now the defaults. Next: a mesh study.

---

## What the model learns

Each training sample is a 6 × 128 × 128 array covering a 2 × 2 chord-length
region around the airfoil:

| Channel | Meaning | Role |
|---|---|---|
| 0 | Freestream velocity X | Input |
| 1 | Freestream velocity Y | Input |
| 2 | Airfoil mask (1 = inside the airfoil) | Input |
| 3 | Pressure *p* | Target |
| 4 | Velocity *Ux* | Target |
| 5 | Velocity *Uy* | Target |

The network, a U-Net, takes the airfoil shape and inflow conditions and
predicts the full RANS flow field in milliseconds. The CFD simulation takes
about 95 s.

![First generated sample](results/first_sample.png)

*First sample generated with this setup: airfoil `s4061` at about 22° to the
flow, |U| = 30 m/s. The flow separates under the airfoil, visible as reversed
flow (blue in Ux) and a strong low-pressure region.*

---

## Results

### Run 1: U-Net, `expo = 5`, `data_6k`, CPU

| Setting | This run | Paper (best model, Sec. 4) |
|---|---|---|
| Trainable parameters | 2.33M | 30.9M |
| Training samples | 6,000 (+400 validation) | 12,800 |
| Training iterations | 10,000 (repo default) | 80,000 |
| Hardware / time | Laptop CPU, 48 min | GPU |

Relative error on the **90 unseen test airfoils**
(`runTestCpu.py`, mean of |prediction − CFD| / |CFD|):

| Quantity | This run | Paper, largest model, 12.8k samples |
|---|---|---|
| Velocity | **2.61%** | 2.15% (Ux), 2.6% (Uy) |
| Pressure | **16.6%** | 14.76% |
| Combined | **3.17%** | 2.6% |

For reference, the paper's *smallest* model (123k parameters) reached 3.3% with
12.8k samples.

**Interpretation:** the first run lands where the paper's scaling study says it
should. It is between the paper's smallest and largest models, while using half
the data, an eighth of the iterations, and a laptop CPU. Pressure has a much
higher relative error than velocity, as in the paper: pressure values are often
small, so small absolute errors become large relative errors.

**Caveat:** a single training run can be lucky or unlucky. See the repeated
seeds below.

### Run 2: same setup, 5 random seeds

Same settings as Run 1, trained 5 times with seeds 1–5
(`scripts/run_seeds.ps1`, summary from `scripts/summarize_seeds.py`). The paper
also reports averages over 5 runs.

| Seed | Pressure % | Velocity % | Combined % |
|---|---|---|---|
| 1 | 17.41 | 2.89 | 3.50 |
| 2 | 17.12 | 2.75 | 3.33 |
| 3 | 15.71 | 2.68 | 3.17 |
| 4 | 18.29 | 2.99 | 3.63 |
| 5 | 16.78 | 2.91 | 3.48 |
| **Mean ± std** | **17.06 ± 0.94** | **2.85 ± 0.13** | **3.42 ± 0.18** |
| Paper, largest model | 14.76 | about 2.4 | 2.6 |
| Paper, smallest model | | | 3.3 |

(std = sample standard deviation over the 5 runs.)

**What this shows:**
- The result is **stable**. The combined error varies by only ±0.18 percentage
  points between seeds, so the gap to the paper is real and not seed noise.
- Run 1 (3.17%) was on the good side of the spread. The honest number for this
  setup is **3.42 ± 0.18%**.
- That is about 0.8 points above the paper's best model, which has 13× more
  parameters, 2× more data and 8× more training iterations. It is close to the
  paper's smallest model.
- Pressure varies most between seeds (±0.94), as expected from its higher
  relative error.

### Run 3: same setup, 80k iterations (paper's training length)

Same model, data and seed (1) as above. Only the training length changes, from
10k to 80k iterations (133 epochs, 6 h 3 min on the laptop CPU).

| Quantity | 10k iterations, seed 1 | 10k, 5-seed mean | **80k iterations, seed 1** | Paper, largest model |
|---|---|---|---|---|
| Pressure | 17.41% | 17.06% | **14.86%** | 14.76% |
| Velocity | 2.89% | 2.85% | **2.45%** | about 2.4% |
| Combined | 3.50% | 3.42% | **2.96%** | 2.6% |

**What this shows:**
- Training 8× longer cuts the combined error by about 15% (3.50 → 2.96%).
  Pressure and velocity now **match the paper's best model** almost exactly,
  with 13× fewer parameters and half the training data.
- The remaining gap in the combined number (2.96 vs 2.6%) is small.
- The learning curve below shows the limit of `data_6k`: after about epoch 90
  the validation loss flattens (0.0043) while the training loss keeps falling
  (0.0031). The model is starting to memorise the training set, so **more
  data**, not more training, is the next lever.

![Learning curve, 80k iterations](results/learning_curve_expo5_80k.png)

*Training and validation L1 loss over 133 epochs (80k iterations). A gap
opens after about epoch 90: mild overfitting on 6,000 training samples.*

**Caveat:** this is one seed. Given the ±0.18 spread between seeds at 10k, the
improvement from 3.42% to 2.96% is well beyond seed noise.

---

## Extension: inverted airfoil in ground effect (in progress)

This is the next step towards F1 front wings: the same kind of 2D RANS data,
but with the airfoil upside down above a moving ground. The code is in
[`ground_effect/`](ground_effect/).

| Change from the original setup | Why |
|---|---|
| Airfoil turned upside down | It makes downforce, like a front wing |
| Angle of attack set by rotating the airfoil (positive = trailing edge up), with the air flowing parallel to the ground | With a ground, the incoming air must flow parallel to it |
| Ground placed a distance *h* (the ride height) below the airfoil's lowest point | The new design parameter |
| Ground moves at the air speed | Seen from the car, the road moves backwards at the same speed as the air, so the ground has no boundary layer |
| Finer mesh in the gap (gmsh Box field, about 10 cells across it) | The strongest flow is in the gap |
| Same 6 × 128 × 128 sample format, with the ground in the mask channel | The same U-Net code can train on it |
| Cl, Cd, residuals, Cl history and surface pressure / wall shear stress stored with every sample | Physics checks and data quality control |

![First ground-effect sample](results/ground_first_sample.png)

*NACA 23012 upside down at 4°, ride height 0.2 chords, 30 m/s, converged,
incoming nuTilda = 3e-5. The air speeds up to about 55 m/s in the gap, and the
low pressure there creates the downforce (Cl = −1.31).*

### Convergence and turbulence-level test (one case)

The same case run with two levels of incoming turbulence, up to 3000 iterations:

| Incoming turbulence | Stopped at | Cl at 500 iterations | Final Cl | Final Cd | Cl within 0.5% of its last value from |
|---|---|---|---|---|---|
| nuTilda = 0.14 m²/s (original dataset) | 1342 (residuals below 1e-5) | −1.023 | −1.194 | 0.0241 | iteration 1210 |
| nuTilda = 3e-5 m²/s (3 ν, the usual Spalart–Allmaras value) | 3000 (the nuTilda residual stays at 1e-3) | −1.137 | −1.306 | 0.0126 | iteration 1429 |

![Convergence test](results/ground_convergence_test.png)

**What this shows:**
- **500 iterations is not enough in ground effect.** The original dataset uses
  500, but Cl is then still 13–14% from its converged value. Getting within 0.5%
  takes about 1,450 iterations, unless faster solver settings (SIMPLEC, being
  tested) reach the same answer sooner.
- **The solver's residual limit is not a reliable stop signal.** One run stopped
  while Cl was still drifting. The other never stopped, although Cl had been
  steady for a long time. The Cl history is used to judge convergence instead.
- **The incoming turbulence level matters.** The original value
  (nuTilda = 0.14 m²/s) is 14,000 times the molecular viscosity (ν = 1e-5 m²/s).
  The usual recommendation for this model is 3–5 ν (Spalart & Rumsey, 2007).
  Switching changes the downforce by 9% and halves the drag. We are checking
  which setting is closer to reality against high-fidelity LES data for a
  NACA 4412 in free air (Vinuesa et al., 2018), using `groundGen.py --free-air`
  and `validate_les.py` (see below).

### Validation against LES: NACA 4412 in free air, Re = 1M

Before generating data, the CFD setup is checked against well-resolved LES of
a NACA 4412 at 5° and Re_c = 1,000,000 (Vinuesa et al., 2018). The comparison
uses the surface pressure Cp and skin friction Cf along the wing, between 15%
and 98% of the chord, where the LES data is. The RANS runs use the same mesh and
solver as the ground-effect runs, but with no ground (`groundGen.py --free-air`).
Our 10 m/s, 1 m chord and ν = 1e-5 m²/s give exactly Re = 1M.

| Incoming nuTilda | Cl | Cd | Mean \|ΔCp\| suction side | Mean \|ΔCp\| pressure side | Mean \|ΔCf\| suction side | Mean \|ΔCf\| pressure side |
|---|---|---|---|---|---|---|
| 0.14 m²/s (original) | 0.818 | 0.0488 | 0.053 | 0.025 | 0.00073 | 0.00144 |
| 3e-5 m²/s (3ν) | 0.876 | 0.0275 | **0.033** | **0.021** | 0.00089 | 0.00175 |

![Validation against LES](results/validation_naca4412_re1M.png)

- **The realistic turbulence level (3ν) predicts the pressure clearly better.**
  The suction-side error drops by 38% and the suction peak moves closer to the
  LES. The pressure at the nose (the stagnation point) is Cp ≈ 1.0, as it must
  be. With 0.14 it is 1.25, which only happens in an unrealistically viscous
  flow.
- **Neither setting predicts skin friction or drag well.** Cf is 30–50% off and
  Cd is 2–4 times too high for this airfoil. The turbulence level is not the
  cause; the mesh is. Only triangles about 0.005 chords in size touch the wall
  (picture below), so the thin boundary layer (0.005–0.04 chords) gets 1–6
  cells across it. The domain also reaches only 5 chords from the wing.
- Note on the LES data: the stored pressure has an arbitrary offset. It is
  removed using the total pressure outside the boundary layer, which is
  constant along the wing (P∞ = −0.058 for Re ≥ 200k).

![Original mesh](results/mesh_original_naca23012.png)

*The original mesh near the wing: triangles only, about 0.005 chords at the
surface.*

### Faster convergence: SIMPLEC

The same ground-effect case, solved with the original SIMPLE settings and with
SIMPLEC (`consistent yes`, relaxation 0.9):

| Solver | Final Cl | Final Cd | Cl within 0.5% of final from |
|---|---|---|---|
| SIMPLE (original settings) | −1.3057 | 0.0126 | iteration 1429 |
| SIMPLEC | −1.3035 | 0.0126 | **iteration 406** |

Both give the same answer (0.17% apart), but SIMPLEC needs 3.5 times fewer
iterations: about 1 minute per sample instead of 4.

![SIMPLE vs SIMPLEC](results/simplec_test.png)

### Decisions so far, and next step

- **Incoming nuTilda = 3ν = 3e-5 m²/s, SIMPLEC and 1000 iterations** are now
  the defaults of `groundGen.py`. The original setup is still available with
  `--nutilda 0.14 --solver simple --iterations 500`.
- **Next: a mesh study.** It compares the original mesh with a boundary-layer
  mesh (`--bl`: 23 thin quadrilateral layers along the wing, the first one
  0.0002 chords thick), in a 5-chord and a 20-chord domain (`--domain 20`). Each
  combination is checked against the LES again (`mesh_study.sh`, then
  `mesh_study_report.py`).

![Learning curve](results/learning_curve_expo5_6k.png)

*Training and validation L1 loss per epoch. Both fall steadily and end close
together (0.0060 / 0.0053), so the model is not overfitting.*

---

## Changes needed for OpenFOAM v2606

The original code targets OpenFOAM 5 (openfoam.org). Running it on OpenFOAM
v2606 (openfoam.com) with gmsh 4 required these fixes, collected in
[`patches/openfoam-v2606.patch`](patches/openfoam-v2606.patch):

| # | Problem | Error seen | Fix |
|---|---|---|---|
| 1 | gmsh 4 writes the MSH 4 mesh format by default | mesh conversion fails | add `-format msh2` to the gmsh call in `dataGen.py` |
| 2 | v2606's `freestreamPressure` requires an explicit far-field value | `Required entry 'freestreamValue' missing in dictionary "0/p/boundaryField/inlet"` | add `freestreamValue uniform 0;` to the four far-field patches in `0/p` |
| 3 | v2606 writes sampled fields to one file, not one per field | `cloud_p.xy not found` | read `cloud_p_U.xy` |
| 4 | the combined file's columns are `x y z p Ux Uy Uz` | (wrong values, no error) | read Ux and Uy from columns 4 and 5 instead of 3 and 4 |

Windows-specific issues:
- Clone with `core.autocrlf=false`. Otherwise the shell scripts get Windows
  line endings and fail inside Linux.
- If pasted commands show up as `[200~...`, run
  `bind 'set enable-bracketed-paste off'` inside the container.

---

## How to run it

**Requirements:** Docker Desktop. No local OpenFOAM install is needed.

```powershell
# 1. Build the image: OpenFOAM v2606 + gmsh + Python
docker build -t openfoam-ml .

# 2. Get the original code with Linux line endings, then apply the v2606 fixes
git clone --config core.autocrlf=false https://github.com/thunil/Deep-Flow-Prediction.git
cd Deep-Flow-Prediction
git apply ../patches/openfoam-v2606.patch
cd ..

# 3. Start the container with this folder shared as /work
docker run -it --rm -v ${PWD}:/work openfoam-ml
```

Inside the container:
```bash
cd /work/Deep-Flow-Prediction/data
bash download_airfoils.sh        # about 1,500 training and 30 test airfoils (UIUC database)
python3 dataGen.py               # number of simulations set by `samples` in dataGen.py
python3 /work/scripts/view_sample.py train/<file>.npz /work/results/sample.png
```

Ground-effect samples, also inside the container (`python3 groundGen.py --help`
lists all options):
```bash
cd /work/ground_effect
python3 groundGen.py --airfoil naca23012.dat --alpha 4 --h 0.2 --U 30   # one case
python3 groundGen.py --free-air --airfoil naca4412 --alpha 5 --U 10      # no ground
python3 groundGen.py --mesh-only --bl --airfoil naca4412 --alpha 5 --h 0.2   # check a mesh
python3 /work/scripts/view_sample.py data/train/<file>.npz /work/results/<name>.png
bash mesh_study.sh                                                        # the mesh study
```

## Repository layout

```
Dockerfile                  OpenFOAM v2606 + gmsh + Python environment
patches/openfoam-v2606.patch  fixes to the original code for OpenFOAM v2606
scripts/view_sample.py      plots all six channels of a training sample
scripts/plot_learning_curve.py  plots training/validation loss
scripts/run_seeds.ps1       trains + tests with several seeds, unattended
scripts/summarize_seeds.py  mean ± std table over the seed runs
ground_effect/groundGen.py        data generation: inverted airfoil above a moving ground
ground_effect/case_template/      OpenFOAM case and gmsh template for ground effect
ground_effect/plot_convergence.py Cl and Cd histories of runs
ground_effect/plot_mesh.py        draws the mesh around the wing
ground_effect/validate_les.py     free-air runs vs LES surface pressure and skin friction
ground_effect/mesh_study.sh       runs the mesh study (inside the container)
ground_effect/mesh_study_report.py  figures and tables of the mesh study
results/                    figures and reproduction results
Deep-Flow-Prediction/       the original repo, cloned locally (not committed)
```

---

## Reference

```bibtex
@article{thuerey2020deepFlowPred,
  title   = {Deep learning methods for Reynolds-averaged Navier--Stokes simulations of airfoil flows},
  author  = {Thuerey, Nils and Weissenow, Konstantin and Prantl, Lukas and Hu, Xiangyu},
  journal = {AIAA Journal},
  volume  = {58},
  number  = {1},
  pages   = {25--36},
  year    = {2020}
}
```

Other references used in the extension:
- P. R. Spalart and C. L. Rumsey, "Effective inflow conditions for turbulence
  models in aerodynamic calculations", *AIAA Journal* 45(10), 2007.
- R. Vinuesa et al., "Turbulent boundary layers around wing sections up to
  Re_c = 1,000,000", *International Journal of Heat and Fluid Flow* 72, 2018.

Airfoil coordinates come from the UIUC Airfoil Coordinates Database
(https://m-selig.ae.illinois.edu/ads/coord_database.html).

## License

[Apache License 2.0](LICENSE), the same license as the original
[Deep-Flow-Prediction](https://github.com/thunil/Deep-Flow-Prediction) code
that the patch in this repo modifies.
