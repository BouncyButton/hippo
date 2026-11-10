"""Draw conceptual SAM prompting examples; no inference or research data used."""

from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "thesis-sam-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as ShapePath


INK = "#27313c"
BLUE = "#24699b"
ORANGE = "#cd752a"


def body_path() -> ShapePath:
    """An invented salt-body outline, reused exactly in image and mask."""
    vertices = [
        (0.34, 0.12), (0.25, 0.18), (0.24, 0.31), (0.34, 0.40),
        (0.43, 0.51), (0.37, 0.62), (0.40, 0.75),
        (0.42, 0.91), (0.66, 0.94), (0.72, 0.80),
        (0.78, 0.67), (0.65, 0.58), (0.64, 0.47),
        (0.63, 0.35), (0.79, 0.24), (0.68, 0.14),
        (0.60, 0.05), (0.43, 0.08), (0.34, 0.12),
        (0.34, 0.12),
    ]
    return ShapePath(vertices, [ShapePath.MOVETO] + [ShapePath.CURVE4] * 18
                     + [ShapePath.CLOSEPOLY])


def scene(ax, mask: bool, point: bool = False) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_facecolor("#f5f7f9")
    if not mask:
        x = np.linspace(0, 1, 200)
        for index, y in enumerate(np.linspace(-0.1, 1.1, 17)):
            ax.plot(x, y + 0.035 * np.sin(5 * x + 0.6 * index),
                    color="#9da5ad", lw=2.2 if index % 3 == 0 else 0.8)
    ax.add_patch(PathPatch(body_path(), facecolor=BLUE if mask else "#dce0e4",
                           edgecolor=BLUE if mask else "#717b85", lw=1.1,
                           zorder=3))
    if point:
        ax.plot(0.54, 0.70, "o", ms=8, color=ORANGE,
                markeredgecolor="white", markeredgewidth=1, zorder=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#a8b1ba")
        spine.set_linewidth(0.6)


def arrow(fig, start, end, colour=INK) -> None:
    fig.add_artist(FancyArrowPatch(start, end, transform=fig.transFigure,
                                  arrowstyle="-|>", mutation_scale=11,
                                  lw=1, color=colour))


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10,
                         "pdf.fonttype": 42})
    fig = plt.figure(figsize=(6.3, 4.0), facecolor="white")
    for row, bottom in enumerate((0.585, 0.12)):
        input_ax = fig.add_axes([0.035, bottom, 0.27, 0.29])
        output_ax = fig.add_axes([0.705, bottom, 0.27, 0.29])
        scene(input_ax, mask=False, point=(row == 0))
        scene(output_ax, mask=True)
        centre_y = bottom + 0.145
        fig.add_artist(FancyBboxPatch(
            (0.405, centre_y - 0.048), 0.19, 0.096,
            boxstyle="round,pad=0.008,rounding_size=0.007",
            transform=fig.transFigure, facecolor="#ebf3f9",
            edgecolor=BLUE, lw=0.8,
        ))
        fig.text(0.5, centre_y, "SAM" if row == 0 else "SAM3",
                 ha="center", va="center", color=INK, fontsize=12)
        arrow(fig, (0.295, centre_y), (0.391, centre_y))
        arrow(fig, (0.609, centre_y), (0.705, centre_y))
        fig.text(0.17, bottom - 0.025,
                 "Image + point" if row == 0 else "Same image",
                 ha="center", va="top", color=INK, fontsize=10)
        fig.text(0.84, bottom - 0.025, "Illustrative mask",
                 ha="center", va="top", color=INK, fontsize=10)
        if row == 0:
            fig.text(0.5, centre_y - 0.095, "The point locates\nthe target region.",
                     ha="center", va="top", color=INK, fontsize=9)
        else:
            fig.text(0.5, centre_y - 0.18, 'Text: “salt body”',
                     ha="center", va="center", color=ORANGE, fontsize=10,
                     bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fcf2e8",
                           "edgecolor": ORANGE, "linewidth": 0.7})
            arrow(fig, (0.5, centre_y - 0.135), (0.5, centre_y - 0.06), ORANGE)

    fig.text(0.04, 0.94, "(a) A spatial prompt: original SAM", fontsize=11, color=INK)
    fig.text(0.04, 0.475, "(b) A concept prompt: SAM3", fontsize=11, color=INK)
    fig.text(0.5, 0.012, "Schematic scene and masks; no model predictions shown.",
             ha="center", color="#6d7782", fontsize=9)
    output = Path(__file__).resolve().parents[1] / "compiled" / "sam-prompting.pdf"
    fig.savefig(output, bbox_inches="tight", pad_inches=0.04)
    fig.savefig("/private/tmp/thesis-sam-prompting-preview.png", dpi=200,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    main()
