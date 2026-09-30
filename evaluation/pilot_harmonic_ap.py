"""Training-screen and frozen-development pilot for harmonic A/P partitioning.

Run --stage screen before --stage evaluate. Selection uses synthetic errors on
training cases only; evaluation cannot change it. All output is research data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.audit_ap_partition_topology import plane_projection, score, validate_labels
from thesis.new_constraints import harmonic_partition as harmonic


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def load_native(dataset: Path, name: str, manifest: dict) -> tuple[np.ndarray, np.ndarray]:
    image_path = dataset / "imagesTr" / f"{name}_0000.nii.gz"
    label_path = dataset / "labelsTr" / f"{name}.nii.gz"
    image, labels = nib.load(image_path), nib.load(label_path)
    if image.shape != labels.shape or not np.allclose(image.affine, labels.affine):
        raise ValueError(f"Image/label geometry mismatch: {name}")
    if not np.allclose(image.affine[:3, :3], np.eye(3)):
        raise ValueError(f"Pilot requires native axis-aligned 1-mm RAS: {name}")
    manifest[str(image_path)] = sha256(image_path)
    manifest[str(label_path)] = sha256(label_path)
    return np.asarray(image.dataobj, dtype=np.float64), validate_labels(np.asarray(labels.dataobj))


def fixed_cut_labels(support: np.ndarray, cut: int) -> np.ndarray:
    return np.where(support, np.where(np.arange(support.shape[1])[None, :, None] >= cut, 1, 2), 0).astype(np.uint8)


def measure(prediction: np.ndarray, truth: np.ndarray, target_cut: int) -> dict:
    _, fit = plane_projection(prediction)
    return {**score(prediction, truth), "cut": fit["cut"], "cut_valid": fit["valid"],
            "cut_mae": abs(fit["cut"] - target_cut) if fit["valid"] else truth.shape[1]}


def run_field(image: np.ndarray, prediction: np.ndarray, *, mode: str, beta: float,
              cut: int, shuffle_seed: int | None = None) -> tuple[harmonic.HarmonicResult, np.ndarray]:
    support = prediction > 0
    seeds = harmonic.make_cores(support, mode=mode, cut=cut)
    if shuffle_seed is not None:
        image = image.copy()
        image[support] = np.random.default_rng(shuffle_seed).permutation(image[support])
    field = harmonic.harmonic_partition(image, support, seeds, (prediction == 2).astype(float), beta=beta)
    fitted = harmonic.field_labels(field, prediction)
    if not np.array_equal(fitted > 0, support):
        raise AssertionError("Harmonic partition changed foreground support")
    return field, fitted


def field_measurements(field: harmonic.HarmonicResult, fitted: np.ndarray,
                       prediction: np.ndarray, truth: np.ndarray, target_cut: int) -> dict:
    projected, fit = plane_projection(fitted)
    if not fit["valid"]:
        projected = fitted.copy()
    if not np.array_equal(projected > 0, prediction > 0):
        raise AssertionError("Projection changed foreground")
    active = field.active_seeds >= 0
    expected = np.where(field.active_seeds == 0, 1, 2)
    return {
        "raw": measure(fitted, truth, target_cut),
        "plane": measure(projected, truth, target_cut),
        "solver": field.diagnostics,
        "seed_wrong_ap": int((active & (truth > 0) & (expected != truth)).sum()),
        "seed_on_background": int((active & (truth == 0)).sum()),
        "changed": int((fitted != prediction).sum()),
        "corrected": int(((prediction != truth) & (fitted == truth)).sum()),
        "introduced": int(((prediction == truth) & (fitted != truth)).sum()),
    }


def screen(args, split, provenance):
    # Sampling depends on identifiers, never on labels, images, or results.
    names = sorted(split["train"], key=lambda name: hashlib.sha256(name.encode()).hexdigest())[:32]
    candidates = [(mode, beta) for mode in ("ends", "band") for beta in (0., 1., 10.)]
    rows, inputs = [], {}
    for index, name in enumerate(names):
        image, truth = load_native(args.dataset_root, name, inputs)
        _, truth_fit = plane_projection(truth)
        if not truth_fit["valid"] or truth_fit["left_label"] != 2:
            raise ValueError(f"Invalid RAS target plane: {name}")
        target = truth_fit["cut"]
        for displacement in (-2, 0, 2):
            prediction = fixed_cut_labels(truth > 0, target + displacement)
            for mode, beta in candidates:
                field, fitted = run_field(image, prediction, mode=mode, beta=beta, cut=target+displacement)
                rows.append({"case": name, "displacement": displacement, "mode": mode, "beta": beta,
                             "baseline": measure(prediction, truth, target),
                             **field_measurements(field, fitted, prediction, truth, target)})
        if (index+1) % 8 == 0:
            print(f"Training screen: {index+1}/{len(names)} cases", flush=True)
    summaries = []
    for mode, beta in candidates:
        selected = [r for r in rows if r["mode"] == mode and r["beta"] == beta]
        summaries.append({"mode": mode, "beta": beta, "cut_mae": float(np.mean([r["raw"]["cut_mae"] for r in selected])),
                          "macro_dice": float(np.mean([r["raw"]["macro_dice"] for r in selected])),
                          "by_displacement": {str(d): {
                              "cut_mae": float(np.mean([r["raw"]["cut_mae"] for r in selected if r["displacement"] == d])),
                              "macro_dice": float(np.mean([r["raw"]["macro_dice"] for r in selected if r["displacement"] == d])),
                          } for d in (-2, 0, 2)}})
    chosen = min((r for r in summaries if r["beta"] > 0), key=lambda r: (r["cut_mae"], -r["macro_dice"], r["mode"], r["beta"]))
    selection = {"provenance": provenance, "train_cases": names, "candidates": summaries,
                 "chosen": {"mode": chosen["mode"], "beta": chosen["beta"]}, "input_hashes": inputs,
                 "interpretation": "Selected on synthetic displaced reference-union training cases; no validation labels, prediction caches, or oracle validation seeds used. This clean-support screen is not real model validation."}
    write_json(args.output_dir / "training_cases.json", rows)
    write_json(args.output_dir / "selection.json", selection)
    print(json.dumps({"chosen": selection["chosen"], "candidates": summaries}, indent=2))


def paired(values: list[float]) -> dict:
    values = np.asarray(values)
    rng = np.random.default_rng(20260923)
    samples = rng.choice(values, (5000, len(values)), replace=True).mean(axis=1)
    return {"mean": float(values.mean()), "case_bootstrap_95": np.quantile(samples, [.025, .975]).tolist(),
            "positive_zero_negative": [int((values > 1e-12).sum()), int((np.abs(values) <= 1e-12).sum()), int((values < -1e-12).sum())]}


def summarize(rows: list[dict]) -> dict:
    result = {"cases": len(rows), "methods": {}}
    for name in ("baseline", "baseline_plane", "uniform_raw", "uniform_plane", "image_raw", "image_plane", "shuffled_raw", "shuffled_plane"):
        def get(row):
            if name in ("baseline", "baseline_plane"):
                return row[name]
            arm, kind = name.split("_")
            return row[arm][kind]
        selected = [get(row) for row in rows]
        result["methods"][name] = {
            "macro_dice": float(np.mean([r["macro_dice"] for r in selected])),
            "cut_mae": float(np.mean([r["cut_mae"] for r in selected])),
            "ap_swaps": sum(r["ap_swaps"] for r in selected),
            "dice_delta_vs_baseline": paired([r["macro_dice"]-row["baseline"]["macro_dice"] for r, row in zip(selected, rows)]),
        }
    result["image_minus_uniform_raw_dice"] = paired([r["image"]["raw"]["macro_dice"]-r["uniform"]["raw"]["macro_dice"] for r in rows])
    result["image_minus_shuffled_raw_dice"] = paired([r["image"]["raw"]["macro_dice"]-r["shuffled"]["raw"]["macro_dice"] for r in rows])
    result["image_field_diagnostics"] = {
        "max_linear_residual": max(r["image"]["solver"]["max_linear_residual"] for r in rows),
        "seed_wrong_ap": sum(r["image"]["seed_wrong_ap"] for r in rows),
        "seed_on_background": sum(r["image"]["seed_on_background"] for r in rows),
        **{key: sum(r["image"]["solver"][key] for r in rows) for key in ("active_seed_voxels", "proposed_seed_voxels", "solved_voxels", "support_voxels", "components", "solved_components")},
        **{key: sum(r["image"][key] for r in rows) for key in ("changed", "corrected", "introduced")},
    }
    return result


def evaluate(args, split, provenance):
    selection_path = args.output_dir / "selection.json"
    selection = json.loads(selection_path.read_text())
    for key in ("split_sha256", "solver_sha256", "pilot_sha256", "metrics_sha256", "protocol_sha256", "fold"):
        if selection["provenance"][key] != provenance[key]:
            raise ValueError(f"Training-screen provenance changed: {key}; rerun the screen")
    if not set(selection["train_cases"]).issubset(split["train"]) or set(selection["train_cases"]) & set(split["val"]):
        raise ValueError("Invalid screening membership")
    for path, expected_hash in selection["input_hashes"].items():
        if sha256(Path(path)) != expected_hash:
            raise ValueError(f"Screening input changed: {path}")
    chosen = selection["chosen"]
    inputs, summaries, all_rows = {}, {}, {}
    for arm in ("baseline_seed0", "augmentation_seed0"):
        directory = args.audit_root / arm / "error_maps"
        if {p.stem for p in directory.glob("*.npz")} != set(split["val"]):
            raise ValueError(f"Cache does not match development split: {arm}")
        rows = []
        for index, name in enumerate(split["val"]):
            native_image, native_truth = load_native(args.dataset_root, name, inputs)
            path = directory / f"{name}.npz"
            inputs[str(path)] = sha256(path)
            with np.load(path, allow_pickle=False) as cache:
                truth = validate_labels(cache["ground_truth"])
                prediction = validate_labels(cache["prediction"])
            if prediction.shape != truth.shape or any(a > b for a, b in zip(native_truth.shape, truth.shape)):
                raise ValueError(f"Invalid cache shape: {name}")
            padding = [((b-a)//2, (b-a+1)//2) for a, b in zip(native_truth.shape, truth.shape)]
            if not np.array_equal(np.pad(native_truth, padding), truth):
                raise ValueError(f"Native/cache label mismatch: {name}")
            image = np.pad(native_image, padding)
            projected, prediction_fit = plane_projection(prediction)
            _, truth_fit = plane_projection(truth)
            if not prediction_fit["valid"] or not truth_fit["valid"] or prediction_fit["left_label"] != 2:
                raise ValueError(f"Invalid A/P plane for {name}")
            target = truth_fit["cut"]
            row = {"case": name, "baseline": measure(prediction, truth, target), "baseline_plane": measure(projected, truth, target)}
            for method in ("uniform", "image", "shuffled"):
                shuffle_seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) if method == "shuffled" else None
                field, fitted = run_field(image, prediction, mode=chosen["mode"],
                                          beta=0. if method == "uniform" else chosen["beta"],
                                          cut=prediction_fit["cut"], shuffle_seed=shuffle_seed)
                row[method] = field_measurements(field, fitted, prediction, truth, target)
            rows.append(row)
            if (index+1) % 13 == 0:
                print(f"Development {arm}: {index+1}/{len(split['val'])}", flush=True)
        source_path = directory.parent / "summary.json"
        source = json.loads(source_path.read_text())
        inputs[str(source_path)] = sha256(source_path)
        summary = summarize(rows)
        if not np.isclose(summary["methods"]["baseline"]["macro_dice"], source["mean_dice"], rtol=0, atol=1e-12):
            raise ValueError(f"Baseline Dice does not reproduce source cache: {arm}")
        if summary["methods"]["baseline"]["ap_swaps"] != source["grouped_error_counts"]["anterior_posterior_swap"]:
            raise ValueError(f"Baseline swaps do not reproduce source cache: {arm}")
        summary["source_cache_config"] = source["config"]
        summaries[arm], all_rows[arm] = summary, rows
    output = {"provenance": provenance, "selection_sha256": sha256(selection_path), "chosen": chosen,
              "interpretation": "Exploratory repeated-use fold-0 local CPU caches; no new model training or inference. Cores do not see validation labels; labels only score predictions. Case-bootstrap intervals do not imply participant independence. Raw-field Dice is primary; planes secondary.",
              "models": summaries, "input_hashes": inputs}
    write_json(args.output_dir / "development_cases.json", all_rows)
    write_json(args.output_dir / "development_summary.json", output)
    for arm, summary in summaries.items():
        print(arm, json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("screen", "evaluate"), required=True)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/Dataset101_MSD")
    parser.add_argument("--audit-root", type=Path, default=ROOT / "experiments/uncal_fold_early_stopping_20260921/voxel_audit")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/experiments/harmonic_ap_20260923")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    split_path = args.dataset_root / "splits_final.json"
    split = json.loads(split_path.read_text())[args.fold]
    if set(split["train"]) & set(split["val"]):
        raise ValueError("Train/development overlap")
    provenance = {"fold": args.fold, "split_sha256": sha256(split_path),
                  "solver_sha256": sha256(Path(harmonic.__file__)), "pilot_sha256": sha256(Path(__file__)),
                  "metrics_sha256": sha256(ROOT / "evaluation/audit_ap_partition_topology.py"),
                  "protocol_sha256": sha256(ROOT / "docs/experiments/harmonic_ap_20260923/PROTOCOL.md"),
                  "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (screen if args.stage == "screen" else evaluate)(args, split, provenance)


if __name__ == "__main__":
    main()
