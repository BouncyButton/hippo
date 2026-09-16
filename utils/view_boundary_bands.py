#!/usr/bin/env python3
"""Visualise the two-step outer-boundary bands B_in / B_out on a hippocampus case.

The bands are the guard domains of the LogLTN boundary constraint
(`thesis/new_constraints/bands/outer_boundary.py`):

    B_in  = F \\ E_k(F)        supervised toward foreground
    B_out = D_k(F) \\ F        supervised toward background

with F the ground-truth foreground union {anterior, posterior}, and D/E the
6-connected cross dilation/erosion applied k times (k = 2 in every trained run).

The masks are built from the label *after* the same symmetric pad / centre crop
to 64^3 that the training pipeline applies, so what is drawn is exactly the
region the auxiliary loss sees.

Examples
--------
    # interactive, three orthogonal planes with sliders
    python utils/view_boundary_bands.py --patient 003

    # static figure to disk
    python utils/view_boundary_bands.py --patient 003 --save bands_003.png

    # how the band grows with the number of morphological steps
    python utils/view_boundary_bands.py --patient 003 --sweep 1 2 3 --save sweep.png

    # cohort statistics, no figure (this is where the 70.5% number comes from)
    python utils/view_boundary_bands.py --stats-only --all-cases

Locally use the anaconda interpreter explicitly -- the system python3 has no numpy:

    /Users/filippofocaccia/anaconda3/bin/python3 utils/view_boundary_bands.py --patient 003
"""

from __future__ import annotations

import argparse
import gzip
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MSD_ROOT = REPOSITORY_ROOT / "datasets" / "Dataset101_MSD"

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

# Categorical slots 1-3 of the project palette; validated all-pairs for
# colour-vision deficiency, so the three overlay classes stay distinguishable.
COLOUR_CORE = "#1baf7a"   # F \ B_in  -- foreground the constraint does NOT touch
COLOUR_INNER = "#eb6834"  # B_in      -- pushed toward foreground
COLOUR_OUTER = "#2a78d6"  # B_out     -- pushed toward background

_NIFTI_DTYPES = {
    2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32,
    64: np.float64, 256: np.int8, 512: np.uint16, 768: np.uint32,
}


# --------------------------------------------------------------------------
# data loading -- copied from band_locality_audit/run_audit.py, which is the
# parser verified against the recorded Dice of 0.8723.
# --------------------------------------------------------------------------

def read_nifti(path: Path) -> np.ndarray:
    """Minimal NIfTI-1 reader; nibabel is not installed locally."""
    raw = gzip.open(path, "rb").read()
    endian = "<" if struct.unpack("<i", raw[:4])[0] == 348 else ">"
    dim = struct.unpack(endian + "8h", raw[40:56])
    datatype = struct.unpack(endian + "h", raw[70:72])[0]
    offset = int(struct.unpack(endian + "f", raw[108:112])[0])
    shape = tuple(int(d) for d in dim[1:1 + dim[0]])
    flat = np.frombuffer(
        raw[offset:], dtype=np.dtype(_NIFTI_DTYPES[datatype]).newbyteorder(endian)
    )
    return flat[:int(np.prod(shape))].reshape(shape, order="F")


def pad_symmetric(volume: np.ndarray, target: int = 64) -> np.ndarray:
    """Match MONAI SpatialPad(method='symmetric') then CenterSpatialCrop."""
    pads = [
        ((target - s) // 2, max(0, target - s) - (target - s) // 2) if s < target else (0, 0)
        for s in volume.shape
    ]
    padded = np.pad(volume, pads, mode="constant")
    crop = tuple(
        slice((s - target) // 2, (s - target) // 2 + target) if s > target else slice(0, target)
        for s in padded.shape
    )
    return padded[crop]


def normalise_case_name(patient: str) -> str:
    text = patient.strip()
    if text.startswith("hippocampus_"):
        return text
    if not text.isdigit():
        raise ValueError(f"Cannot interpret patient identifier {patient!r}.")
    return f"hippocampus_{int(text):03d}"


# --------------------------------------------------------------------------
# band construction
# --------------------------------------------------------------------------

def _cross_neighbour_count(mask: np.ndarray) -> np.ndarray:
    """Count the 7 cross positions (centre + 6 faces) inside `mask`.

    Positions outside the array count as 0, matching the zero-padded conv3d in
    `outer_boundary._dilate_6` / `_erode_6`.
    """
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    total = padded[1:-1, 1:-1, 1:-1].astype(np.int16).copy()
    total += padded[:-2, 1:-1, 1:-1]
    total += padded[2:, 1:-1, 1:-1]
    total += padded[1:-1, :-2, 1:-1]
    total += padded[1:-1, 2:, 1:-1]
    total += padded[1:-1, 1:-1, :-2]
    total += padded[1:-1, 1:-1, 2:]
    return total


def dilate_6(mask: np.ndarray, iterations: int) -> np.ndarray:
    current = mask.astype(bool)
    for _ in range(iterations):
        current = _cross_neighbour_count(current) > 0
    return current


def erode_6(mask: np.ndarray, iterations: int) -> np.ndarray:
    current = mask.astype(bool)
    for _ in range(iterations):
        current = _cross_neighbour_count(current) == 7
    return current


def build_bands_numpy(foreground: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    foreground = foreground.astype(bool)
    inner = foreground & ~erode_6(foreground, steps)
    outer = dilate_6(foreground, steps) & ~foreground
    return inner, outer


def build_bands_torch(foreground: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Call the repository's own loss-side implementation."""
    import torch

    from thesis.new_constraints.bands.outer_boundary import build_boundary_bands

    tensor = torch.as_tensor(foreground.astype(bool))[None, None]
    inner, outer = build_boundary_bands(tensor, steps=steps)
    return inner[0, 0].numpy(), outer[0, 0].numpy()


def build_bands(
    foreground: np.ndarray, steps: int, *, backend: str = "auto", verify: bool = False
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (B_in, B_out, backend_used).

    Prefers the repository implementation so the figure is provably the
    supervision region; falls back to the equivalent numpy morphology when torch
    is unavailable.
    """
    if backend not in {"auto", "torch", "numpy"}:
        raise ValueError("backend must be 'auto', 'torch' or 'numpy'.")

    if backend in {"auto", "torch"}:
        try:
            inner, outer = build_bands_torch(foreground, steps)
        except ImportError:
            if backend == "torch":
                raise
        else:
            if verify:
                inner_np, outer_np = build_bands_numpy(foreground, steps)
                if not (np.array_equal(inner, inner_np) and np.array_equal(outer, outer_np)):
                    raise AssertionError(
                        "numpy morphology disagrees with build_boundary_bands."
                    )
            return inner, outer, "torch (thesis.new_constraints.bands)"

    inner, outer = build_bands_numpy(foreground, steps)
    return inner, outer, "numpy fallback"


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BandStats:
    case: str
    steps: int
    foreground: int
    inner: int
    outer: int
    eroded: int
    edge_touching: bool

    @property
    def valid(self) -> bool:
        return self.inner > 0 and self.outer > 0

    def as_row(self) -> str:
        if self.foreground == 0:
            return f"{self.case}  steps={self.steps}  EMPTY FOREGROUND"
        return (
            f"{self.case}  steps={self.steps}  "
            f"|F|={self.foreground:6d}  "
            f"|B_in|={self.inner:6d} ({100 * self.inner / self.foreground:5.1f}% of F)  "
            f"|B_out|={self.outer:6d} ({100 * self.outer / self.foreground:5.1f}% of F)  "
            f"band/F={(self.inner + self.outer) / self.foreground:4.2f}x  "
            f"core={100 * self.eroded / self.foreground:5.1f}%"
            + ("  EDGE" if self.edge_touching else "")
            + ("" if self.valid else "  INVALID")
        )


def band_stats(case: str, labels: np.ndarray, steps: int) -> BandStats:
    foreground = labels > 0
    inner, outer, _ = build_bands(foreground, steps)
    eroded = erode_6(foreground, steps)
    edge = bool(
        foreground[0].any() or foreground[-1].any()
        or foreground[:, 0].any() or foreground[:, -1].any()
        or foreground[:, :, 0].any() or foreground[:, :, -1].any()
    )
    return BandStats(
        case=case,
        steps=steps,
        foreground=int(foreground.sum()),
        inner=int(inner.sum()),
        outer=int(outer.sum()),
        eroded=int(eroded.sum()),
        edge_touching=edge,
    )


def print_stats_block(stats: BandStats, backend: str) -> None:
    fg = stats.foreground
    print(f"\ncase            {stats.case}")
    print(f"band steps      {stats.steps}   (every trained run uses 2)")
    print(f"backend         {backend}")
    print(f"grid            64 x 64 x 64  (symmetric pad + centre crop)")
    print(f"\n  |F|      ground-truth foreground        {fg:7d} voxels")
    if fg:
        print(f"  |B_in|   inner band -> foreground       {stats.inner:7d}"
              f"   {100 * stats.inner / fg:5.1f}% of F")
        print(f"  |B_out|  outer band -> background       {stats.outer:7d}"
              f"   {100 * stats.outer / fg:5.1f}% of F")
        print(f"  |E_k(F)| unsupervised core              {stats.eroded:7d}"
              f"   {100 * stats.eroded / fg:5.1f}% of F")
        print(f"\n  supervised region / structure volume   "
              f"{(stats.inner + stats.outer) / fg:5.2f}x")
    print(f"  edge-touching foreground               {stats.edge_touching}")
    print(f"  valid for the loss                     {stats.valid}")


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def display_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.rot90(np.take(volume, index, axis=axis))


def initial_indices(mask: np.ndarray) -> list[int]:
    foreground = np.argwhere(mask > 0)
    if foreground.size == 0:
        return [size // 2 for size in mask.shape]
    return [int(np.median(foreground[:, axis])) for axis in range(3)]


def intensity_window(image: np.ndarray) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    nonzero = finite[finite != 0]
    values = nonzero if nonzero.size else finite
    if not values.size:
        return 0.0, 1.0
    low, high = np.percentile(values, (1, 99))
    if low == high:
        high = low + 1.0
    return float(low), float(high)


def overlay_volume(foreground: np.ndarray, inner: np.ndarray, outer: np.ndarray) -> np.ndarray:
    """0 nothing, 1 unsupervised core, 2 inner band, 3 outer band."""
    volume = np.zeros(foreground.shape, dtype=np.uint8)
    volume[foreground & ~inner] = 1
    volume[inner] = 2
    volume[outer] = 3
    return volume


def _overlay_artists(plot_axis, image_plane, overlay_plane, vmin, vmax, cmap, norm):
    image_artist = plot_axis.imshow(
        image_plane, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest"
    )
    overlay_artist = plot_axis.imshow(
        overlay_plane, cmap=cmap, norm=norm, interpolation="nearest"
    )
    plot_axis.set_xticks([])
    plot_axis.set_yticks([])
    return image_artist, overlay_artist


def _build_cmap():
    from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgba

    colours = [
        (0.0, 0.0, 0.0, 0.0),
        to_rgba(COLOUR_CORE, 0.45),
        to_rgba(COLOUR_INNER, 0.70),
        to_rgba(COLOUR_OUTER, 0.60),
    ]
    cmap = ListedColormap(colours)
    return cmap, BoundaryNorm(np.arange(-0.5, 4.5), cmap.N)


def _legend_handles(stats: BandStats, counts: bool = True):
    """Class legend. With several steps per figure the counts differ per row,
    so they are dropped and the per-row axis labels carry them instead."""
    from matplotlib.patches import Patch

    fg = max(stats.foreground, 1)

    def label(base: str, value: int) -> str:
        if not counts:
            return base
        return f"{base}   {value:,} vox ({100 * value / fg:.1f}% of F)"

    return [
        Patch(facecolor=COLOUR_INNER, alpha=0.70,
              label=label("$B_{in}$  → foreground", stats.inner)),
        Patch(facecolor=COLOUR_OUTER, alpha=0.60,
              label=label("$B_{out}$ → background", stats.outer)),
        Patch(facecolor=COLOUR_CORE, alpha=0.45,
              label=label("core $F\\setminus B_{in}$ (no band gradient)", stats.eroded)),
    ]


def crop_box(mask: np.ndarray, margin: int) -> tuple[slice, slice, slice]:
    """Tight box around `mask`, padded by `margin`, so the panels are not
    dominated by the zero padding of the 64^3 crop."""
    indices = np.argwhere(mask)
    if indices.size == 0:
        return tuple(slice(0, s) for s in mask.shape)
    low = np.maximum(indices.min(axis=0) - margin, 0)
    high = np.minimum(indices.max(axis=0) + margin + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(low, high))


AXIS_NAMES = ("Sagittal", "Coronal", "Axial")


def show_interactive(
    image, labels, inner, outer, stats: BandStats, backend: str, offset=(0, 0, 0)
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    foreground = labels > 0
    overlay = overlay_volume(foreground, inner, outer)
    indices = initial_indices(foreground)
    vmin, vmax = intensity_window(image)
    cmap, norm = _build_cmap()

    figure, axes = plt.subplots(1, 3, figsize=(15, 6.4))
    figure.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.26, wspace=0.08)
    figure.suptitle(
        f"{stats.case} — outer-boundary bands, steps = {stats.steps}\n"
        f"supervised region is {(stats.inner + stats.outer) / max(stats.foreground, 1):.2f}× "
        f"the structure volume",
        fontsize=14,
    )

    image_artists, overlay_artists, sliders = [], [], []
    for axis, (plot_axis, name, index) in enumerate(zip(axes, AXIS_NAMES, indices)):
        img, ovl = _overlay_artists(
            plot_axis,
            display_slice(image, axis, index),
            display_slice(overlay, axis, index),
            vmin, vmax, cmap, norm,
        )
        plot_axis.set_title(f"{name}  [{index + offset[axis]}]", fontsize=11)
        image_artists.append(img)
        overlay_artists.append(ovl)

        slider_axis = figure.add_axes([0.06 + axis * 0.32, 0.14, 0.26, 0.03])
        slider = Slider(
            slider_axis, name, 0, image.shape[axis] - 1, valinit=index, valstep=1
        )
        sliders.append(slider)

    def make_update(axis: int):
        def update(value: float) -> None:
            index = int(value)
            image_artists[axis].set_data(display_slice(image, axis, index))
            overlay_artists[axis].set_data(display_slice(overlay, axis, index))
            axes[axis].set_title(
                f"{AXIS_NAMES[axis]}  [{index + offset[axis]}]", fontsize=11
            )
            figure.canvas.draw_idle()
        return update

    for axis, slider in enumerate(sliders):
        slider.on_changed(make_update(axis))

    figure.legend(
        handles=_legend_handles(stats),
        loc="lower center", ncol=1, frameon=False, fontsize=10,
        bbox_to_anchor=(0.5, 0.005),
    )
    plt.show()


def render_static(
    image, labels, band_sets, stats_list, output: Path, title: str, offset=(0, 0, 0)
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    foreground = labels > 0
    indices = initial_indices(foreground)
    vmin, vmax = intensity_window(image)
    cmap, norm = _build_cmap()
    rows = len(band_sets)

    figure, axes = plt.subplots(
        rows, 3, figsize=(13.5, 4.6 * rows + 1.2), squeeze=False
    )
    figure.suptitle(title, fontsize=14)

    for row, ((inner, outer), stats) in enumerate(zip(band_sets, stats_list)):
        overlay = overlay_volume(foreground, inner, outer)
        for axis, name in enumerate(AXIS_NAMES):
            plot_axis = axes[row][axis]
            _overlay_artists(
                plot_axis,
                display_slice(image, axis, indices[axis]),
                display_slice(overlay, axis, indices[axis]),
                vmin, vmax, cmap, norm,
            )
            if row == 0:
                plot_axis.set_title(
                    f"{name}  [{indices[axis] + offset[axis]}]", fontsize=11
                )
            if axis == 0 and rows > 1:
                fg = max(stats.foreground, 1)
                plot_axis.set_ylabel(
                    f"steps = {stats.steps}\n"
                    f"$B_{{in}}$ {100 * stats.inner / fg:.0f}% · "
                    f"$B_{{out}}$ {100 * stats.outer / fg:.0f}%\n"
                    f"core {100 * stats.eroded / fg:.0f}% · "
                    f"band/F {(stats.inner + stats.outer) / fg:.2f}×",
                    fontsize=10,
                )

    figure.legend(
        handles=_legend_handles(stats_list[0], counts=rows == 1),
        loc="lower center", ncol=1, frameon=False, fontsize=10,
    )
    figure.subplots_adjust(
        left=0.10, right=0.98, top=0.90 if rows > 1 else 0.84,
        bottom=0.16 / rows + 0.04, wspace=0.06, hspace=0.10,
    )
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)
    print(f"\nwrote {output}")


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def load_case(case: str, spatial_size: int) -> tuple[np.ndarray, np.ndarray]:
    label_path = MSD_ROOT / "labelsTr" / f"{case}.nii.gz"
    image_path = MSD_ROOT / "imagesTr" / f"{case}_0000.nii.gz"
    for path in (label_path, image_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
    labels = pad_symmetric(np.asarray(read_nifti(label_path)), spatial_size)
    image = pad_symmetric(
        np.asarray(read_nifti(image_path)).astype(np.float32), spatial_size
    )
    return image, labels


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise the outer-boundary band guard domains on an MSD case.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--patient", help="MSD case ID, e.g. 003 or hippocampus_003.")
    parser.add_argument(
        "--steps", type=int, default=2,
        help="Morphological steps; the trained runs all use 2 (default: 2).",
    )
    parser.add_argument(
        "--sweep", type=int, nargs="+", metavar="K",
        help="Render one row per steps value, e.g. --sweep 1 2 3. Implies a static figure.",
    )
    parser.add_argument("--save", type=Path, help="Write a PNG instead of opening a window.")
    parser.add_argument(
        "--backend", choices=("auto", "torch", "numpy"), default="auto",
        help="'torch' uses the repository loss code (default: auto).",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Assert the numpy morphology matches build_boundary_bands exactly.",
    )
    parser.add_argument("--stats-only", action="store_true", help="Print numbers, draw nothing.")
    parser.add_argument(
        "--all-cases", action="store_true",
        help="With --stats-only, aggregate over every case in labelsTr.",
    )
    parser.add_argument("--spatial-size", type=int, default=64)
    parser.add_argument(
        "--margin", type=int, default=4,
        help="Voxels of context kept around the dilated foreground (default: 4).",
    )
    parser.add_argument(
        "--full-fov", action="store_true",
        help="Draw the whole 64^3 crop instead of cropping to the structure.",
    )
    return parser.parse_args(argv)


def run_cohort(steps: int, spatial_size: int) -> int:
    paths = sorted((MSD_ROOT / "labelsTr").glob("hippocampus_*.nii.gz"))
    if not paths:
        print("No labels found.", file=sys.stderr)
        return 1
    rows = []
    for path in paths:
        labels = pad_symmetric(np.asarray(read_nifti(path)), spatial_size)
        rows.append(band_stats(path.name.split(".")[0], labels, steps))

    usable = [r for r in rows if r.foreground > 0]
    inner_share = np.array([r.inner / r.foreground for r in usable])
    outer_share = np.array([r.outer / r.foreground for r in usable])
    core_share = np.array([r.eroded / r.foreground for r in usable])
    ratio = np.array([(r.inner + r.outer) / r.foreground for r in usable])

    print(f"\ncohort: {len(rows)} cases, steps = {steps}, grid {spatial_size}^3\n")
    print(f"  mean |F|                      {np.mean([r.foreground for r in usable]):8.1f} voxels")
    print(f"  mean |B_in|                   {np.mean([r.inner for r in usable]):8.1f}"
          f"   ({100 * inner_share.mean():5.1f}% of F)")
    print(f"  mean |B_out|                  {np.mean([r.outer for r in usable]):8.1f}"
          f"   ({100 * outer_share.mean():5.1f}% of F)")
    print(f"  mean unsupervised core        {np.mean([r.eroded for r in usable]):8.1f}"
          f"   ({100 * core_share.mean():5.1f}% of F)")
    print(f"  mean supervised / structure   {ratio.mean():8.2f}x"
          f"   (range {ratio.min():.2f}-{ratio.max():.2f})")
    print(f"  edge-touching cases           {sum(r.edge_touching for r in rows):8d}")
    print(f"  invalid cases (band empty)    {sum(not r.valid for r in rows):8d}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.stats_only and args.all_cases:
        return run_cohort(args.steps, args.spatial_size)

    if not args.patient:
        print("--patient is required (or use --stats-only --all-cases).", file=sys.stderr)
        return 2

    case = normalise_case_name(args.patient)
    image, labels = load_case(case, args.spatial_size)
    foreground = labels > 0

    steps_list = args.sweep if args.sweep else [args.steps]
    band_sets, stats_list, backend = [], [], ""
    for steps in steps_list:
        inner, outer, backend = build_bands(
            foreground, steps, backend=args.backend, verify=args.verify
        )
        band_sets.append((inner, outer))
        stats_list.append(band_stats(case, labels, steps))

    for stats in stats_list:
        print_stats_block(stats, backend)

    if args.stats_only:
        return 0

    # Statistics are always computed on the full 64^3 grid the loss sees; only
    # the drawing is cropped, so the panels are not mostly zero padding.
    offset = (0, 0, 0)
    if not args.full_fov:
        extent = foreground.copy()
        for _, outer in band_sets:
            extent |= outer
        box = crop_box(extent, args.margin)
        offset = tuple(s.start for s in box)
        image, labels = image[box], labels[box]
        band_sets = [(inner[box], outer[box]) for inner, outer in band_sets]

    if args.save or args.sweep:
        output = args.save or Path(f"{case}_bands.png")
        title = (
            f"{case} — outer-boundary band guard domains"
            + (f", steps ∈ {{{', '.join(str(s) for s in steps_list)}}}"
               if len(steps_list) > 1 else f", steps = {steps_list[0]}")
        )
        render_static(image, labels, band_sets, stats_list, output, title, offset)
        return 0

    inner, outer = band_sets[0]
    show_interactive(image, labels, inner, outer, stats_list[0], backend, offset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
