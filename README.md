# Deep Flow Prediction: Reproduction on OpenFOAM v2606

[![DOI](https://zenodo.org/badge/1454795941.svg)](https://doi.org/10.5281/zenodo.24176980)

A reproduction of **Thuerey et al., "Deep Learning Methods for Reynolds-Averaged
Navier-Stokes Simulations of Airfoil Flows"** (*AIAA Journal*, 2020) on current
software: OpenFOAM v2606, gmsh 4 and Docker on Windows (about 3% test error,
as in the paper). It is extended here to **inverted airfoils in ground effect**
(F1 front wings). The ground-effect simulations are checked against
high-resolution LES (skin friction within 6–7%), wind-tunnel tests of an F1
wing (downforce gain near the ground within 4%) and finer meshes (0.04–3%).
Trained on 789 of these simulations, the U-Net predicts the flow around 30
airfoils it has not seen with a 1.8% error, against 24% for the free-air
model used as it is. The downforce computed from its pressure is off by 11%,
and by 7.7% with a loss that counts the cells next to the wing more. More
simulations are being made.

![A ground-effect sample: the network's three inputs and the three simulated fields it learns](results/ground_first_sample.png)

*One ground-effect sample: a NACA 23012 upside down at 4°, 0.2 chords above
the moving ground, at 30 m/s. Top: what the network gets (the oncoming air,
and where the wing and the ground are). Bottom: what it learns to predict,
the simulated pressure and velocity.*

- Paper: https://arxiv.org/abs/1810.08217
- Original code: https://github.com/thunil/Deep-Flow-Prediction (Apache-2.0)

## Main results

| | Result | Details |
|---|---|---|
| Reproduction, free air | 2.96% test error with the paper's training length (paper: 2.6%); pressure and velocity errors as in the paper's best model | [Reproduction](docs/reproduction.md#results) |
| CFD against high-resolution LES | NACA 4412 in free air: skin friction within 6–7% on average; surface pressure also close (mean \|ΔCp\| 0.009–0.014) | [Mesh study, round 2](docs/cfd_validation.md#mesh-study-round-2) |
| CFD against the wind tunnel | Tyrrell 026 F1 wing: the downforce gained near the ground within 4% | [Validation in ground effect](docs/cfd_validation.md#validation-in-ground-effect-the-tyrrell-026-wing) |
| Mesh study | numerical uncertainty of the downforce 0.04–3%, largest closest to the ground | [Mesh study, round 3](docs/cfd_validation.md#mesh-study-round-3-numerical-verification) |
| Airfoil shapes | 1,462 of the 1,497 UIUC files smoothed: pressure wiggles above 0.1 in 453 files before, in none after | [Airfoil shapes](docs/dataset.md#airfoil-shapes-for-the-dataset) |
| Where steady RANS converges | Cl becomes steady in 86% of the dataset's runs; close to the ground the flow separates, and part of the range has no steady solution | [Parameter scan](docs/dataset.md#parameter-scan-where-does-steady-rans-converge) |
| U-Net in ground effect | 1.8% error on 30 unseen airfoils with 789 training samples (the free-air model as it is: 24%) | [Learning curve](docs/unet.md#learning-curve) |
| Downforce from the U-Net | 11% error; 7.7% with a loss weighted next to the wing | [Downforce](docs/unet.md#downforce-from-the-predicted-pressure) |

## Pages

| Page | What it covers |
|---|---|
| [Reproduction](docs/reproduction.md) | the original free-air study on OpenFOAM v2606: three training runs, and the fixes the original code needed |
| [Ground-effect CFD](docs/cfd_validation.md) | the setup, the solver settings, three rounds of mesh study, and the validation against high-resolution LES and the wind tunnel |
| [Dataset](docs/dataset.md) | smoothing the airfoil files, the parameter scan (where steady RANS converges), and the samples |
| [U-Net in ground effect](docs/unet.md) | training, the learning curve, and the downforce from the predicted pressure |
| [How to run it](docs/howto.md) | every command, from building the Docker image to training on Kaggle |

<a id="words-used-here"></a>
<details>
<summary><b>Words used in these pages</b> (click to open)</summary>

| Word | Meaning |
|---|---|
| CFD | Computational fluid dynamics: computing airflow on a computer |
| RANS | Reynolds-averaged Navier–Stokes: computing the average flow, with a turbulence model for the turbulent eddies |
| OpenFOAM | The free CFD software used here (version v2606) |
| Mesh, cells | The grid of small cells on which the flow is computed |
| Boundary layer | The thin layer of slow air next to the wing's surface |
| y+ | How thin the first cell at the wall is, in the boundary layer's own units. y+ ≈ 1 resolves the layer right down to the wall |
| Spalart–Allmaras (SA) | The turbulence model used: one extra equation for the turbulent mixing |
| nuTilda | The quantity that equation solves for. "Clipping" means OpenFOAM resets values that come out negative |
| Iterations, converged | The solver improves its answer step by step (iterations); a run has converged when the answer stops changing |
| GCI | Grid convergence index: how much a result could still change on an infinitely fine mesh |
| LES | Large-eddy simulation: a far more detailed and costly simulation. Our reference is a high-resolution LES, close to DNS (direct numerical simulation, which computes all the turbulent eddies) |
| Chord (c) | The length of the airfoil from front to back |
| Ride height (h/c) | The gap between the wing's lowest point and the ground, divided by the chord |
| Angle of attack (α) | How far the wing is tilted against the oncoming air |
| Cl, Cd | Downforce (lift) and drag, divided by ½ × air density × speed² × chord, so that cases can be compared. Negative Cl means downforce |
| Reynolds number (Re) | Speed × chord ÷ the air's viscosity; here 1–5 million |
| Ground effect | The extra downforce a wing makes close to the ground |
| Moving ground | The ground moves at the air's speed, as the road does under a car |
| Separation | Where the air stops following the wing's surface; behind it the flow turns back and becomes unsteady |
| Sample | One simulation, stored as six 128 × 128 images: 3 inputs (the air's speed, and where the wing and the ground are) and 3 outputs (pressure and the two velocity components) |
| U-Net | The type of neural network used: it turns input images into output images |
| Surrogate | A fast model that replaces the slow simulation |
| Loss (L1) | What training makes small: how far the network's output is from the simulation. L1 is the mean absolute difference |

</details>

---

## Progress

| Stage | Status |
|---|---|
| 1. Environment (Docker + OpenFOAM v2606 + gmsh + Python) | Done |
| 2. Port the data generation pipeline to OpenFOAM v2606 | Done |
| 3. First simulation and training sample | Done (about 95 s per sample on a laptop) |
| 4. Download the authors' precomputed dataset | Done: `data_6k` from the [TUM data service](https://dataserv.ub.tum.de/index.php/s/m1470791) (1.3 GB, 6,400 train / 90 test, SHA-512 verified). The training code expects it in `Deep-Flow-Prediction/data/train` and `data/test` |
| 5. Train the U-Net and reproduce the paper's error numbers | Done: 1 run + 5 repeated seeds (CPU, about 47 min each). See [Reproduction](docs/reproduction.md#results) |
| 6. Compare our results with the paper | Done: 5-seed mean ± std, plus an 80k-iteration run that matches the paper's pressure and velocity errors |
| 7. Extension: inverted airfoil in ground effect | In progress. Done: simulations checked against high-resolution LES, wind-tunnel data and finer meshes; 1,462 airfoils smoothed; parameter scan (which cases converge, and the rules for keeping them); 2000 training runs (1,716 samples) and the test set (102 samples); learning curve (1.8% test error with 789 training samples); downforce from the predicted pressure (11% error, 7.7% with a loss weighted next to the wing). Now: 2000 more runs. See [Ground-effect CFD](docs/cfd_validation.md), [Dataset](docs/dataset.md) and [U-Net](docs/unet.md) |

---

## Quick start

You need Docker Desktop: OpenFOAM runs inside the container. In this folder,
in PowerShell:
```powershell
docker build -t openfoam-ml .     # OpenFOAM v2606 + gmsh + Python
git clone --config core.autocrlf=false https://github.com/thunil/Deep-Flow-Prediction.git
docker run -it --rm -v ${PWD}:/work openfoam-ml
```
Inside the container: the airfoil files, one ground-effect simulation (a few
minutes) and a picture of its sample:
```bash
cd /work/Deep-Flow-Prediction/data && bash download_airfoils.sh
cd /work/ground_effect
python3 groundGen.py --airfoil naca23012.dat --alpha 4 --h 0.2 --U 30
python3 /work/scripts/view_sample.py data/train/<file>.npz /work/results/sample.png
```
Everything else (the free-air pipeline with the fixes in `patches/`, the CFD
checks, the dataset and the training) is in [How to run it](docs/howto.md).

## Repository layout

```
README.md                   this page: summary, results, progress, quick start
docs/                       the detailed pages: reproduction, ground-effect CFD, dataset, U-Net, how to run it
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
ground_effect/mesh_quality.py     y+ of the first cell, and cell sizes in the sampled window
ground_effect/validate_les.py     free-air runs vs LES surface pressure and skin friction
ground_effect/mesh_study.sh       runs the mesh study, round 2 (inside the container)
ground_effect/mesh_study_report.py  figures and tables of round 2
ground_effect/verification.sh     round 3: nuTilda scheme, grid convergence, domain (inside the container)
ground_effect/verification_report.py  tables and figures of round 3, including the GCI
ground_effect/sampling_test.py    checks the row-by-row sampling on finished runs (inside the container)
ground_effect/make_tyrrell026.py  Tyrrell 026 wing coordinates (Zerihan 2001, Table 2) -> airfoil files (smoothed, and as tabulated)
ground_effect/panel.py            inviscid surface pressure on the curve gmsh builds through airfoil points (panel method)
ground_effect/shape_check.py      kinks in that curve, before and after smoothing, for many airfoil files
ground_effect/smooth_airfoils.py  smooths the airfoil files for the dataset (quintic spline in sqrt(x)); leaves out files with errors
ground_effect/smoothing_test.py   tests the smoothing against stand-in airfoils whose shape is known exactly
ground_effect/validation_zerihan.sh   runs the ground-effect validation (inside the container)
ground_effect/validation_zerihan.py   compares those runs with Zerihan's experiments and CFD
ground_effect/param_scan.sh       parameter scan: 200 random runs drawn like the dataset (inside the container)
ground_effect/param_scan_report.py  sorts the runs (steady, clipped, settling, oscillating, failed) by ride height, angle and speed
ground_effect/clip_test.sh        what the nuTilda clipping changes, and whether settling runs converge (inside the container)
ground_effect/clip_test_report.py tables and figure of that test
ground_effect/clip_cells.py       finds the cells where OpenFOAM clipped nuTilda in a finished run
ground_effect/dataset.sh          makes the dataset with several runs at a time (inside the container)
ground_effect/unet.py             the U-Net of the original code (copied from DfpNet.py, unchanged)
ground_effect/unet_data.py        loads and normalises the ground-effect samples; the original error measure and the downforce from the pressure
ground_effect/train_unet.py       trains one U-Net (from random weights or from the free-air model) and tests it
ground_effect/learning_curve.py   trains and tests U-Nets on parts of the data, then the table and figure
ground_effect/loss_study.py       compares training losses for the downforce: table and figure
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

To cite this repository: M. T. A. Sial, *Deep Flow Prediction on OpenFOAM
v2606, extended to inverted airfoils in ground effect*, Zenodo,
https://doi.org/10.5281/zenodo.24176980 (all versions; v0.1.0 is
https://doi.org/10.5281/zenodo.24176981). GitHub's "Cite this repository"
button gives the same in other formats.

Other references used in the extension:
- P. R. Spalart and C. L. Rumsey, "Effective inflow conditions for turbulence
  models in aerodynamic calculations", *AIAA Journal* 45(10), 2007.
- R. Vinuesa et al., "Turbulent boundary layers around wing sections up to
  Re_c = 1,000,000", *International Journal of Heat and Fluid Flow* 72, 2018,
  https://doi.org/10.1016/j.ijheatfluidflow.2018.04.017 (the high-resolution
  LES data used for validation: NACA 4412 at 5°, provided by R. Vinuesa).
- I. B. Celik, U. Ghia, P. J. Roache, C. J. Freitas, H. Coleman and P. E. Raad,
  "Procedure for estimation and reporting of uncertainty due to discretization
  in CFD applications", *Journal of Fluids Engineering* 130(7), 078001, 2008
  (the grid convergence index used in round 3).
- P. J. Roache, *Verification and Validation in Computational Science and
  Engineering*, Hermosa Publishers, 1998 (safety factor 3 for the GCI when
  the observed order is unreliable).
- J. Zerihan and X. Zhang, "Aerodynamics of a single element wing in ground
  effect", *Journal of Aircraft* 37(6), 2000 (wind-tunnel tests of an
  inverted wing above a moving ground; downforce reduction from flow
  separation at small ride heights).
- J. Zerihan, *An Investigation into the Aerodynamics of Wings in Ground
  Effect*, PhD thesis, University of Southampton, 2001 (the Tyrrell 026 wing
  coordinates, test conditions, sectional downforce, oil flow and 2D RANS
  results used for the ground-effect validation).
- C. H. Reinsch, "Smoothing by spline functions", *Numerische Mathematik*
  10, 1967 (how much a smoothing spline may depart from the data points;
  used for the Tyrrell 026 coordinates and the dataset airfoils).
- I. J. Schoenberg, "Spline functions and the problem of graduation",
  *Proceedings of the National Academy of Sciences* 52(4), 1964 (smoothing
  splines of higher order: the quintic spline used for the dataset airfoils).
- P. Dierckx, *Curve and Surface Fitting with Splines*, Oxford University
  Press, 1993 (FITPACK, used for the arc-length splines).
- J. L. Hess and A. M. O. Smith, "Calculation of potential flow about
  arbitrary bodies", *Progress in Aerospace Sciences* 8, 1967 (the panel
  method in `panel.py`).
- S. R. Allmaras, F. T. Johnson and P. R. Spalart, "Modifications and
  clarifications for the implementation of the Spalart-Allmaras turbulence
  model", Seventh International Conference on Computational Fluid Dynamics
  (ICCFD7), paper ICCFD7-1902, 2012 (negative values of the model's variable,
  and the "SA-neg" form that handles them; see the parameter scan).

Airfoil coordinates come from the UIUC Airfoil Coordinates Database
(https://m-selig.ae.illinois.edu/ads/coord_database.html).

## License

[Apache License 2.0](LICENSE), the same license as the original
[Deep-Flow-Prediction](https://github.com/thunil/Deep-Flow-Prediction) code
that the patch in this repo modifies.
