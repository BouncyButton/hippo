"""Native-volume labels, reproducible splits and I/O for GeoFWI3D."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SHAPE = (96, 96, 96)
FRACTIONS = {"5pct": 0.05, "20pct": 0.20, "100pct": 1.0}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def archive_metadata(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if 0 <= int(row["sample_index"]) < 1000]
    if len(rows) != 1000 or {int(row["sample_index"]) for row in rows} != set(range(1000)):
        raise ValueError("Archive 0 metadata must contain exactly model_0000 through model_0999")
    return rows


def make_splits(rows: list[dict], seed: int = 0) -> dict:
    """Stratify the positive cohort by fault presence; subsets share one ordering."""
    eligible = [row for row in rows if int(float(row["yn_salt"])) == 1]
    if len({row["seed"] for row in eligible}) != len(eligible):
        raise ValueError("Repeated simulation seeds require grouped splitting, not case-level shuffling")
    rng = np.random.default_rng(seed)
    train_groups, validation = [], []
    for fault in [0, 1]:
        ids = sorted(f"model_{int(row['sample_index']):04d}" for row in eligible
                     if int(float(row["yn_fault"])) == fault)
        rng.shuffle(ids)
        count = int(math.floor(len(ids) * 0.20 + 0.5))
        validation.extend(ids[:count])
        train_groups.append(ids[count:])
    # Interleave strata proportionally so every prefix has a similar class mix.
    order, used = [], [0, 0]
    total = sum(map(len, train_groups))
    for position in range(total):
        candidates = [i for i in range(2) if used[i] < len(train_groups[i])]
        group = max(candidates, key=lambda i: (position + 1) * len(train_groups[i]) / total - used[i])
        order.append(train_groups[group][used[group]])
        used[group] += 1
    subsets = {name: order[:max(1, int(math.floor(len(order) * fraction + 0.5)))]
               for name, fraction in FRACTIONS.items()}
    splits = {"seed": seed, "train_pool": order, "validation": sorted(validation),
              "training_subsets": subsets, "fraction_denominator": "eligible training pool",
              "cohort": "salt-containing scenes from archive 0", "stratification": "fault presence",
              "test_set": None}
    validate_splits(splits, {f"model_{int(row['sample_index']):04d}": row for row in eligible})
    return splits


def validate_splits(splits: dict, cases: dict) -> None:
    train, validation = splits["train_pool"], splits["validation"]
    if len(set(train)) != len(train) or len(set(validation)) != len(validation):
        raise ValueError("Duplicate case ID within a split")
    if set(train) & set(validation) or set(train) | set(validation) != set(cases):
        raise ValueError("Overlapping or incomplete training/validation split")
    if not train or not validation or splits.get("test_set") is not None:
        raise ValueError("This experiment requires nonempty training and validation, without a test split")
    previous = set()
    for name in FRACTIONS:
        subset = splits["training_subsets"][name]
        expected = max(1, int(math.floor(len(train) * FRACTIONS[name] + .5)))
        if len(subset) != expected or len(set(subset)) != len(subset) or not previous <= set(subset) <= set(train):
            raise ValueError(f"Invalid nested training subset: {name}")
        previous = set(subset)
    if previous != set(train):
        raise ValueError("The 100% subset must use the entire training pool")
    train_seeds = {str(cases[name]["seed"]) for name in train}
    validation_seeds = {str(cases[name]["seed"]) for name in validation}
    if train_seeds & validation_seeds:
        raise ValueError("A simulation seed occurs in training and validation")


def salt_mask(rgt: np.ndarray, vp: np.ndarray, salt_velocity: float) -> tuple[np.ndarray, dict]:
    """Remove null-RGT background artifacts using the independent physical model."""
    if rgt.shape != vp.shape or not np.isfinite(rgt).all() or not np.isfinite(vp).all():
        raise ValueError("Physical and RGT arrays must be aligned and finite")
    implicit = rgt == 0
    physical = vp == np.float32(salt_velocity)
    missing = physical & ~implicit
    if missing.any():
        raise ValueError("Physical salt appears outside the documented RGT-zero mask")
    mask = implicit & physical
    if not mask.any():
        raise ValueError("A salt-containing case has an empty physical salt mask")
    return mask.astype(np.uint8), {"implicit_salt_voxels": int(implicit.sum()),
                                   "salt_voxels": int(mask.sum()),
                                   "removed_null_rgt_background_voxels": int((implicit & ~physical).sum())}


def normalized_image(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Label-independent amplitude normalization on the native grid."""
    if not np.isfinite(image).all():
        raise ValueError("Nonfinite seismic amplitude")
    scale = float(np.percentile(np.abs(image), 99.5))
    if scale <= 0:
        raise ValueError("Seismic image has no usable amplitude")
    return np.clip(image / scale, -1, 1).astype(np.float32), scale


def load_case(root: Path, case: dict) -> tuple[np.ndarray, np.ndarray]:
    import nibabel as nib
    image = np.asarray(nib.load(root / case["image"]).dataobj, dtype=np.float32)
    label = np.asarray(nib.load(root / case["label"]).dataobj, dtype=np.uint8)
    if image.shape != SHAPE or label.shape != SHAPE:
        raise ValueError("Case is not a native 96³ volume")
    if not np.isfinite(image).all() or not np.isin(label, [0, 1]).all() or not label.any():
        raise ValueError("Invalid normalized seismic image or salt mask")
    return image, label
