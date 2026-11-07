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
| 4. Download the authors' precomputed dataset | Next |
| 5. Train the U-Net and reproduce the paper's error numbers | To do |
| 6. Compare my results with the paper | To do |
| 7. Extension: inverted airfoil in ground effect | Planned |

### Log
- **2026-11-07**: Set up the Docker environment. Found and fixed four
  incompatibilities between the original OpenFOAM 5 setup and OpenFOAM v2606
  (see below). Generated and plotted the first training sample.

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

## Repository layout

```
Dockerfile                  OpenFOAM v2606 + gmsh + Python environment
patches/openfoam-v2606.patch  fixes to the original code for OpenFOAM v2606
scripts/view_sample.py      plots all six channels of a training sample
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

Airfoil coordinates come from the UIUC Airfoil Coordinates Database
(https://m-selig.ae.illinois.edu/ads/coord_database.html).

## License

[Apache License 2.0](LICENSE), the same license as the original
[Deep-Flow-Prediction](https://github.com/thunil/Deep-Flow-Prediction) code
that the patch in this repo modifies.
