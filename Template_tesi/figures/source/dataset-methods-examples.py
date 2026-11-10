"""Matched input/mask/overlay figures from the thesis's native dataset cases.

These display reference annotations, never model predictions. Array transforms
and slice selection follow the earlier introductory and ADNI illustrations.
"""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/thesis-dataset-methods-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "Template_tesi/figures/compiled"
REPORT = ROOT / "reports/thesis_dataset_methods_examples_20261103"
BLUE = np.array([36, 105, 155]) / 255
ORANGE = np.array([205, 117, 42]) / 255
INK = "#27313c"
GRAY = "#6d7782"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_case(dataset, case):
    base = ROOT / "datasets" / dataset
    ipath = base / "imagesTr" / f"{case}_0000.nii.gz"
    ypath = base / "labelsTr" / f"{case}.nii.gz"
    im, lab = nib.load(ipath), nib.load(ypath)
    x, y = im.get_fdata(), lab.get_fdata()
    assert x.shape == y.shape and np.array_equal(im.affine, lab.affine)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert np.array_equal(y, y.astype(np.uint8))
    y = y.astype(np.uint8)
    k = int(np.argmax((y > 0).sum(axis=(1, 2))))
    geo = dataset == "Dataset203_GeoFWI3D"
    limits = np.array([-1., 1.]) if geo else np.percentile(x, [1, 99])
    g = np.clip((x[k].T - limits[0]) / (limits[1] - limits[0]), 0, 1)
    g = np.rint(g * 31) / 31
    z = y[k].T
    if not geo:
        g, z = g[::-1], z[::-1]
    assert set(np.unique(z)) == ({0, 1, 2} if dataset == "Dataset101_MSD" else {0, 1})
    image = np.repeat(g[..., None], 3, axis=2)
    mask, overlay = np.ones_like(image), image.copy()
    for label, color in [(1, BLUE), (2, ORANGE)]:
        mask[z == label] = color
        overlay[z == label] = .55 * image[z == label] + .45 * color
    record = dict(dataset=dataset, case=case, image=str(ipath.relative_to(ROOT)),
                  labels=str(ypath.relative_to(ROOT)), shape=list(x.shape),
                  plane_axis=0, slice_index_zero_based=k,
                  selection="maximum foreground area; lowest index breaks ties",
                  display_gray_limits=limits.tolist(), display_gray_levels=32,
                  display_transform="transpose axes 1/2" + ("" if geo else "; flip displayed rows"),
                  overlay_alpha=.45, labels_in_slice=np.unique(z).tolist(),
                  image_sha256=sha(ipath), labels_sha256=sha(ypath),
                  model_predictions=False)
    return (image, mask, overlay), record


def render(name, dataset, cases, row_titles, first_title, legend):
    rows = [load_case(dataset, case) for case in cases]
    width, panel_width = 6.10, 1.82
    panel_height = panel_width * rows[0][0][0].shape[0] / rows[0][0][0].shape[1]
    row_gap, header, footer = .35, .52, .38
    height = header + len(rows) * panel_height + (len(rows) - 1) * row_gap + footer
    fig = plt.figure(figsize=(width, height))
    columns = [.06, 2.14, 4.22]
    for left, title in zip(columns, [first_title, "Reference mask", "Annotation overlay"]):
        fig.text((left + panel_width / 2) / width, 1 - .09 / height, title,
                 ha="center", va="top", fontsize=9, fontweight="bold")
    records = []
    for r, ((panels, record), title) in enumerate(zip(rows, row_titles)):
        top = height - header - r * (panel_height + row_gap)
        fig.text(columns[0] / width, (top + .10) / height, title,
                 fontsize=8.5, color=GRAY, va="bottom")
        for left, panel in zip(columns, panels):
            ax = fig.add_axes([left / width, (top - panel_height) / height,
                               panel_width / width, panel_height / height])
            ax.imshow(panel, origin="upper", interpolation="nearest", aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(.5)
                spine.set_edgecolor(GRAY)
        records.append(record)
    handles = [Patch(facecolor=color, edgecolor=GRAY if color == "white" else "none",
                     linewidth=.5, label=label) for color, label in legend]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .015),
               fontsize=8, ncol=len(handles), frameon=False,
               handlelength=1.1, columnspacing=1.6)
    path = OUTPUT / f"{name}.pdf"
    fig.savefig(path, dpi=300, facecolor="white")
    fig.savefig(REPORT / f"{name}.png", dpi=180, facecolor="white")
    plt.close(fig)
    return dict(figure=str(path.relative_to(ROOT)), sha256=sha(path), cases=records)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9,
                         "pdf.fonttype": 42, "text.color": INK})
    records = [
        render("geo-data-examples", "Dataset203_GeoFWI3D", ["model_0001"],
               ["One cross-section of a three-dimensional scene"], "Reflectivity image",
               [("#24699b", "Salt"), ("white", "Background")]),
        render("msd-data-examples", "Dataset101_MSD", ["hippocampus_001"],
               ["One cross-section of a hippocampal crop"], "MRI crop",
               [("#24699b", "Anterior"), ("#cd752a", "Posterior"), ("white", "Background")]),
        render("adni-data-examples", "Dataset103_ADNI",
               ["hippocampus_adni_10007_L", "hippocampus_adni_10007_R"],
               ["(a) Left hippocampal crop", "(b) Right hippocampal crop"], "MRI crop",
               [("#24699b", "Hippocampus"), ("white", "Background")]),
    ]
    provenance = dict(generator_sha256=sha(Path(__file__)), figures=records)
    (REPORT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"figures": [r["figure"] for r in records], "cases": 4}))


if __name__ == "__main__":
    main()
