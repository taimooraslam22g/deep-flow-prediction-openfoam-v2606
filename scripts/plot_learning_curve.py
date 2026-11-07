"""Plot training and validation L1 loss per epoch from runTrain*.py logs.

Usage:
    python scripts/plot_learning_curve.py <L1.txt> <L1val.txt> [output.png]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

train = np.loadtxt(sys.argv[1])
val = np.loadtxt(sys.argv[2])
out = sys.argv[3] if len(sys.argv) > 3 else "learning_curve.png"

epochs = np.arange(1, len(train) + 1)
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(epochs, train, "o-", label="training", color="#2a6fdb")
ax.plot(epochs, val, "s-", label="validation (400 held-out samples)", color="#e07b00")
ax.set_xlabel("epoch")
ax.set_ylabel("mean L1 loss (normalized fields)")
ax.set_yscale("log")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
ax.set_title(f"U-Net (expo 5) on data_6k: final train {train[-1]:.4f}, val {val[-1]:.4f}")
fig.tight_layout()
fig.savefig(out, dpi=120)
print("saved", out)
