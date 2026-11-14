# Glossary

[README](../README.md) · [All steps](../README.md#how-we-got-here)

Words used on these pages.

---

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
