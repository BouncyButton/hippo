"""Render observed image/label examples for the supervisor briefing."""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hippo-supervisor-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIGURES = HERE / "figures"
TEAL = "#008696"
INK = "#192e3b"
GRAY = "#526371"
MASK_CMAP = ListedColormap(["#101820", TEAL])
OVERLAY_CMAP = ListedColormap([TEAL])
sources = {}


def source(path: Path) -> Path:
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def render_examples(examples, title, subtitle, stem, footer):
    """Use identical slices for raw input, reference label and overlay columns."""
    fig = plt.figure(figsize=(13.33, 8.4), dpi=160, facecolor="white")
    fig.text(.045, .945, title, fontsize=24, weight="bold", color=INK)
    fig.text(.045, .897, subtitle, fontsize=12.5, color=GRAY)
    grid = fig.add_gridspec(2, 3, left=.165, right=.95, top=.81, bottom=.16,
                           hspace=.24, wspace=.075)
    axes = []
    for row, example in enumerate(examples):
        image = np.take(example["image"], example["slice"], axis=example["axis"]).T
        mask = np.take(example["mask"], example["slice"], axis=example["axis"]).T
        assert image.shape == mask.shape and mask.any()
        for column in range(3):
            ax = fig.add_subplot(grid[row, column])
            if column == 1:
                ax.imshow(mask, cmap=MASK_CMAP, vmin=0, vmax=1,
                          interpolation="nearest", origin=example["origin"])
            else:
                ax.imshow(image, cmap="gray", vmin=example["limits"][0],
                          vmax=example["limits"][1], interpolation="nearest",
                          origin=example["origin"])
                if column == 2:
                    ax.imshow(np.ma.masked_where(~mask, mask), cmap=OVERLAY_CMAP,
                              vmin=0, vmax=1, alpha=.48, interpolation="nearest",
                              origin=example["origin"])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(["Model input", "Reference target", "Target on input"][column],
                             fontsize=15, color=INK, pad=13)
            if column == 0:
                axes.append(ax)
    fig.canvas.draw()
    for ax, example in zip(axes, examples):
        box = ax.get_position()
        fig.text(.045, box.y0 + box.height / 2, example["display_label"],
                 va="center", fontsize=13, color=INK, linespacing=1.55)
    fig.text(.045, .085, footer, fontsize=11.5, color=GRAY, linespacing=1.5)
    fig.text(.045, .035, "Reference labels shown; these panels are dataset examples, not model predictions.",
             fontsize=10.5, color=GRAY)
    for suffix in ("png", "svg"):
        fig.savefig(FIGURES / f"{stem}.{suffix}", facecolor="white")
    plt.close(fig)


def record(example):
    return {key: value for key, value in example.items() if key not in {"image", "mask"}}


geo_root = ROOT / "reports/geofwi3d_preview"
geo_metadata = json.loads(source(geo_root / "provenance.json").read_text())
geo_examples = []
for index in (2, 1):
    folder = geo_root / f"model_{index:04d}"
    image, rgt, velocity = [np.fromfile(source(folder / f"{name}.bin"), dtype="<f4").reshape(96, 96, 96)
                            for name in ("image3d", "rgt3d", "vp3d")]
    metadata = geo_metadata["models"][str(index)]
    physical_salt = velocity == np.float32(float(metadata["salt_vel"]))
    assert not np.any(physical_salt & (rgt != 0))
    mask = (rgt == 0) & physical_salt
    audit = json.loads(source(folder / "label_audit.json").read_text())
    assert int(mask.sum()) == audit["velocity_salt_voxels"]
    assert np.isfinite(image).all() and 0 < mask.mean() < 1
    limit = float(np.percentile(np.abs(image), 99.5))
    nsalt = int(metadata["nsalt"])
    geo_examples.append({
        "id": folder.name, "image": image, "mask": mask,
        "axis": 1, "slice": int(np.argmax(mask.sum(axis=(0, 2)))), "origin": "upper",
        "limits": [-limit, limit], "native_shape": list(image.shape),
        "foreground_voxels": int(mask.sum()),
        "removed_null_rgt_voxels": int(np.sum((rgt == 0) & ~physical_salt)),
        "display_label": f"Example {'A' if index == 2 else 'B'}\n{nsalt} generated salt\n{'body' if nsalt == 1 else 'bodies'}"
                         + (" + faults" if metadata["yn_fault"] == "1" else ""),
    })
render_examples(geo_examples, "GeoFWI3D: seismic images and salt targets",
                "Two simulated geological scenes | Native volumes: 96 × 96 × 96 voxels | Teal = salt",
                "08_geofwi3d_dataset_examples",
                "Each row shows one X–Z slice; depth increases downward. All salt belongs to one foreground class.\n"
                "Velocity and relative geological time define the reference mask; the model receives only the seismic image.")


adni_root = ROOT / "datasets/Dataset103_ADNI"
adni_metadata = json.loads(source(adni_root / "dataset.json").read_text())
assert adni_metadata["labels"] == {"background": 0, "hippocampus": 1}
splits = json.loads(source(adni_root / "splits_final.json").read_text())
adni_examples = []
for partition, hemisphere, letter in (("train", "L", "A"), ("val", "R", "B")):
    case = sorted(case for case in splits[0][partition] if case.endswith(f"_{hemisphere}"))[0]
    image_nii = nib.load(source(adni_root / "imagesTr" / f"{case}_0000.nii.gz"))
    label_nii = nib.load(source(adni_root / "labelsTr" / f"{case}.nii.gz"))
    image = image_nii.get_fdata()
    label = label_nii.get_fdata()
    assert image.shape == label.shape == (32, 48, 40)
    assert np.array_equal(image_nii.affine, label_nii.affine)
    assert set(np.unique(label)) == {0., 1.}
    assert np.isfinite(image).all()
    mask = label == 1
    adni_examples.append({
        "id": case, "fold0_partition": partition, "image": image, "mask": mask,
        "axis": 0, "slice": int(np.argmax(mask.sum(axis=(1, 2)))), "origin": "lower",
        "limits": np.percentile(image, [1, 99]).tolist(), "native_shape": list(image.shape),
        "foreground_voxels": int(mask.sum()), "export_affine": image_nii.affine.tolist(),
        "display_label": f"Example {letter}\n{'Left' if hemisphere == 'L' else 'Right'} hippocampus\nMRI crop",
    })
render_examples(adni_examples, "ADNI: MRI crops and hippocampus targets",
                "Two cases from the prepared dataset | Crops: 32 × 48 × 40 voxels | Teal = hippocampus",
                "09_adni_dataset_examples",
                "Each row shows a native-array slice through the hippocampus. The target is binary whole hippocampus.\n"
                "These are reference-localized crops; anatomical orientation and physical spacing require the original geometry.")

provenance = {
    "purpose": "Dataset illustrations for the supervisor meeting; no model predictions or performance-based selection.",
    "geofwi3d": {
        "cases": [record(example) for example in geo_examples],
        "selection": "Both previously downloaded public sample volumes; maximal foreground-area Y slice, first maximum.",
        "mask_rule": "(rgt3d == 0) & (vp3d == float32(metadata.salt_vel)); matches the experiment preparation rule.",
        "display": "Native voxels; symmetric clipping at each volume's 99.5th absolute amplitude percentile; depth down.",
    },
    "adni": {
        "cases": [record(example) for example in adni_examples],
        "selection": "First lexicographic left training crop and right validation crop in stored fold 0; maximal foreground-area axis-0 slice, first maximum.",
        "display": "Native voxels; per-volume 1st/99th percentile intensity window; transpose displayed slice only; no anatomical axis claim.",
    },
    "source_sha256": sources,
}
(HERE / "dataset_examples_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
print(json.dumps({"figures": ["08_geofwi3d_dataset_examples", "09_adni_dataset_examples"],
                  "geo_cases": [record(example) for example in geo_examples],
                  "adni_cases": [record(example) for example in adni_examples]}, indent=2))
