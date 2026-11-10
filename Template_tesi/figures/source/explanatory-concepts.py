"""Render four mathematical teaching figures, independent of experiment results.

Run from the repository with .venv/bin/python. PDFs are standalone figure
assets; previews are written to the temporary directory for visual review.
"""

from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "thesis-sam-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


INK = "#27313c"
BLUE = "#24699b"
ORANGE = "#cd752a"
GRAY = "#6d7782"
PALE_BLUE = "#ebf3f9"
PALE_ORANGE = "#fcf2e8"
PALE_GRAY = "#f5f7f9"
OUTPUT = Path(__file__).resolve().parents[1] / "compiled"
PREVIEWS = Path("/private/tmp/thesis-concept-figures")


def setup() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 10,
        "mathtext.fontset": "stix", "pdf.fonttype": 42,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
    })
    PREVIEWS.mkdir(exist_ok=True)


def save(fig, name: str) -> None:
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.055)
    fig.savefig(PREVIEWS / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.055)
    plt.close(fig)


def canvas(height: float):
    fig = plt.figure(figsize=(6.3, height), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    return fig, ax


def box(ax, x, y, width, height, fill=PALE_BLUE, edge=BLUE):
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.006,rounding_size=0.008",
        facecolor=fill, edgecolor=edge, linewidth=0.8,
    ))


def arrow(ax, start, end, colour=GRAY):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                 mutation_scale=11, lw=0.9, color=colour))


def ltn_example() -> None:
    truths = np.array([0.8, 0.9])
    logs = np.log(truths)
    loss = -logs.mean()
    assert np.isclose(loss, 0.164252033486018)
    fig, ax = canvas(2.5)
    ax.text(0.5, 0.95, r"Statement $P(v)$: voxel $v$ belongs to class $c$",
            ha="center", va="top", fontsize=11)
    spans = [(0.025, 0.28), (0.365, 0.27), (0.695, 0.28)]
    headings = ["1. Ground the statement", "2. Take the logarithm", "3. Form the loss"]
    for (x, width), title in zip(spans, headings):
        box(ax, x, 0.23, width, 0.52,
            fill=PALE_ORANGE if x > 0.65 else PALE_BLUE,
            edge=ORANGE if x > 0.65 else BLUE)
        ax.text(x + width / 2, 0.80, title, ha="center", va="bottom", fontsize=9.5)
    for y, voxel, probability, logtruth in zip((0.60, 0.39), ("u", "v"), truths, logs):
        ax.add_patch(Rectangle((0.055, y - 0.055), 0.044, 0.11,
                               facecolor=BLUE, edgecolor=INK, lw=0.5))
        ax.text(0.079, y, voxel, color="white", ha="center", va="center", fontsize=9)
        ax.text(0.12, y, rf"$p_c({voxel})={probability:.1f}$", va="center", fontsize=12)
        ax.text(0.50, y, rf"$\log({probability:.1f})\simeq {logtruth:.3f}$",
                ha="center", va="center", fontsize=12)
    arrow(ax, (0.312, 0.49), (0.355, 0.49))
    arrow(ax, (0.643, 0.49), (0.685, 0.49))
    ax.text(0.835, 0.62, "Negative mean", ha="center", fontsize=10)
    ax.text(0.835, 0.45, r"$-\frac{\log(0.8)+\log(0.9)}{2}$",
            ha="center", va="center", fontsize=15)
    ax.text(0.835, 0.29, rf"$\simeq {loss:.3f}$", ha="center", fontsize=14, color=ORANGE)
    ax.text(0.165, 0.14, "Two reference voxels of class c", ha="center", fontsize=8.5)
    ax.text(0.50, 0.14, "Local log-truths", ha="center", fontsize=9)
    ax.text(0.835, 0.14, "Equal voxel weights", ha="center", fontsize=9)
    ax.text(0.5, 0.025, "Higher membership probability gives a smaller local penalty.",
            ha="center", fontsize=9, color=GRAY)
    save(fig, "ltn-grounding-example")


def probability_pair(ax, x, y, values, height=0.15):
    width = 0.062
    for index, value in enumerate(values):
        left = x + index * width
        ax.add_patch(Rectangle((left, y), width, height,
                               facecolor=PALE_GRAY, edgecolor="none"))
        ax.add_patch(Rectangle((left, y), width, height * value,
                               facecolor=BLUE, edgecolor="none"))
        ax.add_patch(Rectangle((left, y), width, height,
                               fill=False, edgecolor=INK, lw=0.7))
        ax.text(left + width / 2, y - 0.025, f"{value:g}", ha="center", va="top", fontsize=10)


def edge_example() -> None:
    fig, ax = canvas(3.25)
    # Each pair is assumed to belong to the selected band-support graph.
    examples = [
        (0.02, "(a) Within one region", (1.0, 1.0), [(0.8, 0.8), (0.8, 0.3)]),
        (0.535, "(b) Across the boundary", (1.0, 0.0), [(0.9, 0.1), (0.5, 0.5)]),
    ]
    for start, heading, reference, predictions in examples:
        ax.text(start, 0.99, heading, va="top", fontsize=10.5)
        ax.text(start, 0.755, "Reference", va="center", fontsize=9.5)
        probability_pair(ax, start + 0.16, 0.68, reference)
        target = reference[0] - reference[1]
        ax.text(start + 0.32, 0.755, r"$h(u)-h(v)$" + "\n" + rf"$={target:g}$",
                va="center", fontsize=10)
        ax.text(start + 0.16 + 0.031, 0.87, r"$u$", ha="center", fontsize=10)
        ax.text(start + 0.16 + 0.093, 0.87, r"$v$", ha="center", fontsize=10)
        for y, prediction in zip((0.42, 0.17), predictions):
            discrepancy = (prediction[0] - prediction[1]) - target
            penalty = discrepancy ** 2
            ax.text(start, y + 0.075, "Prediction", va="center", fontsize=9.5)
            probability_pair(ax, start + 0.16, y, prediction)
            ax.text(start + 0.32, y + 0.075, rf"$e^E={penalty:.2f}$",
                    va="center", color=ORANGE, fontsize=12)
    assert np.isclose(((0.9 - 0.1) - (1 - 0)) ** 2, 0.04)
    ax.plot([0.50, 0.50], [0.12, 0.96], color="#c6cbd0", lw=0.7)
    ax.text(0.5, 0.02, r"$e^E=[(q(u)-q(v))-(h(u)-h(v))]^2$",
            ha="center", va="center", fontsize=13)
    save(fig, "edge-consistency-example")


def depth_example() -> None:
    counts = np.array([32, 128, 256])
    uniform = counts / counts.sum()
    masses = np.minimum(counts / 64, 1)
    weights = masses / masses.sum()
    assert np.allclose(weights, [0.2, 0.4, 0.4])
    assert np.isclose(weights.sum(), 1)

    fig = plt.figure(figsize=(6.3, 3.6), facecolor="white")
    geometry = fig.add_axes([-0.035, 0.13, 0.57, 0.73], projection="3d")
    # Local coordinates: x is the long axis, y is the plane normal/depth,
    # and z lies in the plane. The object is schematic, not anatomical data.
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 36), np.linspace(0, np.pi, 24))
    x = 11 * np.cos(v)
    y = 3.2 * np.sin(v) * np.cos(u)
    z = 2.2 * np.sin(v) * np.sin(u)
    geometry.plot_surface(x, y, z, color="#a8b1ba", alpha=0.22,
                          linewidth=0, shade=False)
    plane = [(-12, 0, -3), (12, 0, -3), (12, 0, 3), (-12, 0, 3)]
    geometry.add_collection3d(Poly3DCollection([plane], facecolors=BLUE,
                                               edgecolors=BLUE, alpha=0.17, linewidths=0.6))
    for offset in (-2, 2):
        geometry.plot([-12, 12], [offset, offset], [0, 0], color=ORANGE,
                       alpha=0.65, linestyle=":", linewidth=1)
    geometry.quiver(-13, 0, 0, 27, 0, 0, color=INK, arrow_length_ratio=0.045, linewidth=1.2)
    geometry.quiver(0, 0, 0, 0, 7.4, 0, color=ORANGE,
                     arrow_length_ratio=0.16, linewidth=1.8)
    geometry.scatter([0], [0], [0], s=12, color=INK)
    geometry.plot([1.8, 1.8, 0], [0, 1.8, 1.8], [0, 0, 0], color=INK, linewidth=0.7)
    geometry.text(10, 0, 3.7, r"Plane $\Pi$", fontsize=10, color=BLUE)
    geometry.text(9.5, -1.6, -2.7, r"Long axis $\ell$", fontsize=9.5)
    geometry.text(0, 7.5, 0.2, r"Depth $n$", fontsize=10, color=ORANGE)
    geometry.text(-1, -0.5, -1.2, r"$\mu$", fontsize=11)
    geometry.set(xlim=(-14, 14), ylim=(-5, 9), zlim=(-4, 4))
    geometry.set_box_aspect((2.4, 1.2, 0.7))
    geometry.view_init(elev=24, azim=-63)
    geometry.set_axis_off()

    bars = fig.add_axes([0.62, 0.29, 0.36, 0.49])
    positions = np.arange(3)
    for offset, values, colour, label in [
        (-0.18, uniform, "#a8b1ba", "Uniform voxel weights"),
        (0.18, weights, BLUE, "Depth weighting"),
    ]:
        bars.bar(positions + offset, values * 100, width=0.32, color=colour, label=label)
        for px, val in zip(positions + offset, values):
            value_label = f"{100 * val:.1f}" if offset < 0 else f"{100 * val:.0f}"
            bars.text(px, val * 100 + 1.8, value_label,
                      ha="center", fontsize=8.5)
    bars.set(ylim=(0, 75), xticks=positions, xticklabels=[str(n) for n in counts],
             yticks=[0, 20, 40, 60], ylabel="Total stratum weight (%)", xlabel="Voxels in stratum")
    bars.tick_params(labelsize=9, length=0)
    bars.spines[["top", "right"]].set_visible(False)
    bars.spines[["left", "bottom"]].set_color("#a8b1ba")
    bars.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25),
                frameon=False, fontsize=8.5)
    fig.text(0.24, 0.94, "(a) Plane and depth direction", ha="center", fontsize=10.5)
    fig.text(0.80, 0.94, "(b) Weighting three strata", ha="center", fontsize=10.5)
    fig.text(0.24, 0.18, "The plane contains the long axis.\nDepth is measured perpendicular to it.",
             ha="center", va="center", fontsize=9)
    fig.text(0.5, 0.015, "Schematic geometry and illustrative voxel counts; threshold T = 64.",
             ha="center", fontsize=8.5, color=GRAY)
    save(fig, "depth-weighting-example")


def budget_example() -> None:
    fig, ax = canvas(3.8)
    ax.text(0.025, 0.97, "(a) One optimization ceiling, different update counts", fontsize=11, va="top")
    left, right = 0.30, 0.84
    unit = (right - left) / 12
    rows = [
        (0.80, "Lower-cost model", 6, 2, 0),
        (0.65, "Higher-cost model", 3, 4, 0),
        (0.50, "Lower-cost model\n+ auxiliary loss", 4, 2, 1),
    ]
    for y, label, updates, base, auxiliary in rows:
        ax.text(0.025, y + 0.037, label, va="center", fontsize=9.5)
        for index in range(updates):
            start = left + index * (base + auxiliary) * unit
            ax.add_patch(Rectangle((start, y), base * unit, 0.074,
                                   facecolor=BLUE, edgecolor="none"))
            if auxiliary:
                ax.add_patch(Rectangle((start + base * unit, y), auxiliary * unit, 0.074,
                                       facecolor=ORANGE, edgecolor="none"))
            ax.add_patch(Rectangle((start, y), (base + auxiliary) * unit, 0.074,
                                   fill=False, edgecolor="white", lw=1.2))
        ax.text(0.87, y + 0.037, f"{updates} updates", va="center", fontsize=9.5)
        assert updates * (base + auxiliary) == 12
    ax.plot([right, right], [0.48, 0.90], linestyle="--", color=INK, lw=0.9)
    ax.text(right, 0.91, "Ceiling", ha="center", fontsize=9)
    ax.add_patch(Rectangle((0.30, 0.435), 0.023, 0.025, facecolor=BLUE, edgecolor="none"))
    ax.text(0.333, 0.448, "Baseline computation", va="center", fontsize=8.5)
    ax.add_patch(Rectangle((0.62, 0.435), 0.023, 0.025, facecolor=ORANGE, edgecolor="none"))
    ax.text(0.653, 0.448, "Auxiliary computation", va="center", fontsize=8.5)

    ax.text(0.025, 0.36, "(b) Where calibration is counted", fontsize=11, va="center")
    # Containers show accounting scope, not equal measured costs across studies.
    for y, label in ((0.21, "Medical"), (0.065, "Geological")):
        ax.text(0.025, y + 0.045, label, fontsize=9.5, va="center")
        box(ax, 0.24, y, 0.18, 0.09, fill=PALE_GRAY, edge=GRAY)
        ax.text(0.33, y + 0.045, "Calibration", ha="center", va="center", fontsize=9)
        box(ax, 0.51, y, 0.39, 0.09)
        ax.text(0.705, y + 0.045, "Optimization", ha="center", va="center", fontsize=9)
    ax.add_patch(Rectangle((0.49, 0.195), 0.43, 0.12,
                           fill=False, edgecolor=INK, linewidth=0.8))
    ax.text(0.71, 0.324, "Inside the medical ceiling", ha="center", fontsize=8.5)
    ax.add_patch(Rectangle((0.22, 0.05), 0.70, 0.12,
                           fill=False, edgecolor=INK, linewidth=0.8))
    ax.text(0.57, 0.015, "Both inside the geological ceiling", ha="center", fontsize=8.5)
    save(fig, "flop-budget-example")


if __name__ == "__main__":
    setup()
    ltn_example()
    edge_example()
    depth_example()
    budget_example()
