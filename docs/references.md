# References

[README](../README.md) · [All steps](../README.md#how-we-got-here)

The papers and data used on these pages. How to cite this repository is in the
[README](../README.md).

---

- N. Thuerey, K. Weissenow, L. Prantl and X. Hu, "Deep learning methods for Reynolds-averaged
  Navier–Stokes simulations of airfoil flows", *AIAA Journal* 58(1), 25–36, 2020 (the study
  reproduced here).

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
