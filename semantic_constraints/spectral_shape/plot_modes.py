"""Readable mode figure from saved arrays; no refitting or metric changes."""
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hippo-spectral-mode-plot")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(sys.argv[1])
    modes = np.load(root / "modes_first8.npz")
    mean = modes["shape_mean"]
    z = int(np.argmax((mean < 0).sum((0, 1))))
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    for i in range(3):
        mode = modes["shape_modes"][i] * modes["shape_sd"][i]
        for scale, color, style in ((-2, "royalblue", "-"), (0, "black", "--"), (2, "darkorange", "-")):
            axes[0, i].contour((mean + scale * mode)[:, :, z].T, levels=[0], colors=[color], linestyles=[style])
        axes[0, i].set_title(f"Shape mode {i+1}")
        error = modes["error_modes"][i] * modes["error_sd"][i]
        limit = max(float(abs(error[:, :, z]).max()), .001)
        im = axes[1, i].imshow(error[:, :, z].T, origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit)
        axes[1, i].contour(mean[:, :, z].T, levels=[0], colors=["black"], linewidths=.8)
        axes[1, i].set_title(f"Error mode {i+1}")
        colorbar = fig.colorbar(im, ax=axes[1, i], shrink=.85)
        colorbar.set_label("Distance change, mm / SD", fontsize=9)
        for ax in axes[:, i]:
            ax.set(xlim=(10, 70), ylim=(10, 70), aspect="equal")
            ax.tick_params(labelsize=9)
    fig.suptitle(f"Training-derived modes, slice {z}\nShape contours: mean (black), -2 SD (blue), +2 SD (orange). Mode signs are arbitrary.", fontsize=12)
    fig.savefig(root / "modes.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
