# 7. The dataset

[README](../README.md) · ← [6. Parameter scan](06-parameter-scan.md) · [8. U-Net learning curve](08-unet-learning-curve.md) → · [Workflow](../README.md#workflow)

The training data: random airfoils at random angles of attack (0–10°), ride
heights (h/c 0.05–0.6) and speeds (10–50 m/s), and a test set on 30 airfoils
that are never used for training. 2000 runs so far gave 1,716 samples; 2000 more
are running.

*Words used here: [glossary](glossary.md). References: [list](references.md).*

---

**The dataset** is made by `dataset.sh`: 1000 runs (about 21–22 hours), and a
test set from the 30 test airfoils. Running 8 at a time instead of 4 does not
help on this laptop (Ryzen 7 7730U, 8 cores): each run then takes twice as
long (9.4 instead of 4.7 minutes for 1000 iterations, with the same mesh
sizes), so the laptop finishes the same number of runs per hour.

Each new 1000 runs use new random seeds, so no case is repeated. Every run is
listed in `samples.csv`. Runs whose Cl does not become steady go to
`not_converged/` and are not used for training.

| | Runs | Samples (Cl steady) | Not converged | Failed before a sample |
|---|---|---|---|---|
| Training, first 1000 runs | 1000 | 867 | 120 | 13 |
| Training, second 1000 runs | 1000 | 849 | 136 | 15 |
| Test set (the 30 test airfoils) | 120 | 102 | 13 | 5 |

So 86% of the runs give a sample, more than the 75–80% expected from the
parameter scan. About 10% of the training samples are kept for validation,
chosen by a checksum of the file name. 2000 more training runs are being
made.

---

[README](../README.md) · ← [6. Parameter scan](06-parameter-scan.md) · [8. U-Net learning curve](08-unet-learning-curve.md) → · [Workflow](../README.md#workflow)
