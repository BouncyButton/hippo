"""Render a synthetic index-grid / physical-space example for the thesis."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
    })
    ink = "#27313c"
    blue = "#24699b"
    orange = "#cd752a"
    fig = plt.figure(figsize=(6.3, 3.7), facecolor="white")
    axes = [
        fig.add_axes([0.08, 0.23, 0.35, 0.57]),
        fig.add_axes([0.60, 0.23, 0.35, 0.57]),
    ]
    for ax, spacing in zip(axes, (1, 2)):
        for x in range(3):
            for y in range(3):
                colour = "#f5f7f9"
                if (x, y) == (2, 1):
                    colour = "#ebf3f9"
                elif (x, y) == (1, 2):
                    colour = "#fcf2e8"
                ax.add_patch(Rectangle(
                    (x - 0.5, (y - 0.5) * spacing), 1, spacing,
                    facecolor=colour, edgecolor="#a8b1ba", linewidth=0.6,
                ))
        centre = (1, spacing)
        for x, y, colour in ((0, 1, "#a8b1ba"), (1, 0, "#a8b1ba"),
                             (2, 1, blue), (1, 2, orange)):
            endpoint = (x, y * spacing)
            ax.plot([centre[0], endpoint[0]], [centre[1], endpoint[1]],
                    color=colour, linewidth=1.8, solid_capstyle="round")
            ax.plot(*endpoint, marker="o", markersize=4, color=colour)
        ax.plot(*centre, marker="o", markersize=5, color=ink)
        ax.annotate("1 step" if spacing == 1 else "1 mm",
                    (1.5, spacing), xytext=(0, -13),
                    textcoords="offset points", ha="center", color=blue,
                    fontsize=9)
        ax.annotate("1 step" if spacing == 1 else "2 mm",
                    (1, 1.5 * spacing), xytext=(-7, 0),
                    textcoords="offset points", ha="right", va="center",
                    color=orange, fontsize=9)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5 * spacing, 2.5 * spacing)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, spacing, 2 * spacing])
        ax.tick_params(length=0, pad=4, labelsize=9, colors=ink)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel(r"$v_1$" if spacing == 1 else r"$r_1$ (mm)", labelpad=4)
        ax.set_ylabel(r"$v_2$" if spacing == 1 else r"$r_2$ (mm)", labelpad=3)

    fig.text(0.255, 0.90, "(a) Voxel indices", ha="center", color=ink, fontsize=11)
    fig.text(0.775, 0.90, "(b) Physical positions", ha="center", color=ink, fontsize=11)
    fig.text(0.515, 0.57, r"$\longrightarrow$", ha="center", color=ink, fontsize=22)
    fig.text(0.515, 0.48, r"$r=Qv$", ha="center", color=ink, fontsize=11)
    fig.text(0.50, 0.055, "Same neighbours; different physical step lengths.",
             ha="center", color=ink, fontsize=10)

    output = Path(__file__).resolve().parents[1] / "compiled" / "voxel-coordinates.pdf"
    fig.savefig(output, bbox_inches="tight", pad_inches=0.04)
    fig.savefig("/private/tmp/thesis-voxel-coordinates-preview.png", dpi=180,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    main()
