"""Frozen, subject-split test of boundary context conditional on model output.

Run extract, select, then evaluate. See the dated protocol for interpretation.
No reference-dependent quantity enters feature construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage, special
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.audit_ap_partition_topology import score
from evaluation.pilot_harmonic_ap import fixed_cut_labels
from thesis.new_constraints.uncal_fold.audit_predicted_foreground import largest_foreground_component
from thesis.new_constraints.uncal_fold.feature_probe import pool_layer_sequence
from thesis.new_constraints.uncal_fold.foldedness import best_fit_first_anterior_slice

SEED = 20260923
ARMS = ("base", "base_mri", "base_decoder", "base_shuffled_mri")
CHECKPOINTS = {"unaugmented": "baseline_seed0_checkpoint_best.pt",
               "augmented": "augmentation_seed0_checkpoint_best.pt"}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def context_maps(image: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Fixed multiscale MRI maps, normalized without reference labels."""
    values = image[support]
    center = np.median(values)
    scale = max(float(np.quantile(values, .75) - np.quantile(values, .25)), 1e-6)
    image = np.clip((image - center) / scale, -8, 8).astype(np.float32)
    maps = [image]
    for sigma in (1., 2.):
        mean = ndimage.gaussian_filter(image, sigma)
        variance = np.maximum(ndimage.gaussian_filter(image * image, sigma) - mean * mean, 0)
        maps.extend((mean, np.sqrt(variance)))
    maps.append(ndimage.gaussian_gradient_magnitude(image, 1.))
    return np.stack(maps)


def prediction_features(logits: np.ndarray, prediction: np.ndarray, union: np.ndarray,
                        candidates: np.ndarray, original_cut: int) -> np.ndarray:
    """Coordinates, geometry and conditional probabilities; no labels argument."""
    q = special.expit(logits[1].astype(np.float64) - logits[2])
    q = np.clip(q, 1e-7, 1 - 1e-7)
    # Cut likelihood includes the entire predicted foreground, including small components.
    support = prediction > 0
    log_a = (np.log(q) * support).sum(axis=(0, 2))
    log_p = (np.log1p(-q) * support).sum(axis=(0, 2))
    prefix_a, prefix_p = np.r_[0., log_a.cumsum()], np.r_[0., log_p.cumsum()]
    low, high = candidates[0] - 1, candidates[-1]
    length = max(high - low, 1)
    rows = []
    for cut in candidates:
        relative = (cut - low) / length
        offset = (cut - original_cut) / length
        likelihood = (prefix_p[cut] + prefix_a[-1] - prefix_a[cut]) / support.sum()
        row = [cut / prediction.shape[1], relative, relative**2, offset,
               abs(offset), offset**2, likelihood, length / prediction.shape[1]]
        for y in (cut - 1, cut):
            mask = union[:, y, :]
            coordinates = np.argwhere(mask)
            if not len(coordinates):
                raise ValueError("Candidate touches empty slice")
            values = q[:, y, :][mask]
            row.extend([values.mean(), values.std(), np.mean(values > .5),
                        len(values) / union.sum(), *coordinates.mean(axis=0) / 64,
                        *np.ptp(coordinates, axis=0) / 64])
        rows.append(row)
    matrix = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite prediction features")
    return matrix


def design(case: dict, arm: str) -> np.ndarray:
    if arm not in ARMS:
        raise ValueError(arm)
    blocks = [case["base"]]
    if arm == "base_decoder":
        blocks.append(case["decoder"])
    elif arm in ("base_mri", "base_shuffled_mri"):
        context = case["mri"]
        if arm == "base_shuffled_mri":
            seed = int.from_bytes(hashlib.sha256(str(case["name"].item()).encode()).digest()[:8], "little")
            context = np.random.default_rng(seed).permutation(context, axis=0)
        blocks.append(context)
    return np.column_stack(blocks)


def fit(cases: list[dict], arm: str, c_value: float):
    features, labels, weights = [], [], []
    for case in cases:
        positive = case["candidates"] == int(case["target"])
        if positive.sum() != 1:
            raise ValueError("Target outside candidate support; do not silently drop subjects")
        features.append(design(case, arm))
        labels.append(positive)
        weights.append(np.where(positive, .5, .5 / (len(positive) - 1)))
    estimator = make_pipeline(StandardScaler(), LogisticRegression(C=c_value, max_iter=3000,
                                                                   solver="lbfgs", tol=1e-7))
    estimator.fit(np.concatenate(features), np.concatenate(labels),
                  logisticregression__sample_weight=np.concatenate(weights))
    if estimator[-1].n_iter_.max() >= 3000:
        raise RuntimeError("Ranker did not converge")
    return estimator


def predict(estimator, case: dict, arm: str) -> int:
    return int(case["candidates"][estimator.decision_function(design(case, arm)).argmax()])


def load_cases(cache: Path, names: list[str]) -> list[dict]:
    result = []
    for name in names:
        with np.load(cache / f"{name}.npz", allow_pickle=False) as data:
            result.append(dict(data))
    return result


def extract(args, names: list[str]) -> None:
    import torch
    from baselines.swin_unetr.swin_unetr import _build_monai_dataset_from_pkl, _load_pkl_dataframe
    from thesis.new_constraints.train_swinunetr_constraints import build_swinunetr

    torch.set_num_threads(args.threads)
    pkl = ROOT / "datasets/Dataset101_MSD/msd_hippocampus_full.pkl"
    dataset = _build_monai_dataset_from_pkl(_load_pkl_dataframe(pkl), "MSD", 3,
                                           spatial_size=(64, 64, 64), do_resize=False)
    indices = {entry["case_name"]: i for i, entry in enumerate(dataset.data)}
    for model_name in args.models:
        cache = args.cache / model_name
        cache.mkdir(parents=True, exist_ok=True)
        checkpoint_path = ROOT / "experiments/augmentation_family_b_20260907/checkpoints" / CHECKPOINTS[model_name]
        provenance = {"checkpoint_sha256": digest(checkpoint_path), "pkl_sha256": digest(pkl),
                      "script_sha256": digest(Path(__file__)), "device": "cpu", "threads": args.threads,
                      "torch": torch.__version__, "split_sha256": digest(args.split)}
        manifest_path = cache / "manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != provenance:
            raise ValueError("Cache provenance differs; choose a new cache directory")
        save_json(manifest_path, provenance)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("run", {}).get("fold") != 0 or checkpoint.get("run", {}).get("seed") != 0:
            raise ValueError("Expected fold 0 / seed 0 checkpoint")
        model = build_swinunetr((64, 64, 64), 3, torch.device("cpu"), activation_checkpointing=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        for index, name in enumerate(names):
            path = cache / f"{name}.npz"
            if path.exists():
                continue
            item = dataset[indices[name]]
            image = item["image"][0].numpy()
            truth = item["label"][0].numpy().astype(np.uint8)
            with torch.inference_mode():
                logits = model(item["image"][None].float())[0].numpy()
            prediction = logits.argmax(axis=0).astype(np.uint8)
            arm = "baseline_seed0" if model_name == "unaugmented" else "augmentation_seed0"
            audit = ROOT / "experiments/uncal_fold_early_stopping_20260921/voxel_audit" / arm / "error_maps" / f"{name}.npz"
            if audit.exists():
                with np.load(audit) as reference:
                    if not np.array_equal(truth, reference["ground_truth"]) or not np.array_equal(prediction, reference["prediction"]):
                        raise AssertionError(f"CPU prediction/label cache mismatch: {name}")
            union, _, _ = largest_foreground_component(prediction > 0)
            occupied = np.flatnonzero(union.any(axis=(0, 2)))
            if len(occupied) < 3 or np.any(np.diff(occupied) != 1):
                raise ValueError("Disconnected A/P extent")
            low, high = int(occupied[0]), int(occupied[-1])
            candidates = np.arange(low + 1, high + 1)
            target, _, _ = best_fit_first_anterior_slice(truth)
            original_cut, _, _ = best_fit_first_anterior_slice(prediction)
            base = prediction_features(logits, prediction, union, candidates, original_cut)
            mri, _ = pool_layer_sequence(torch.from_numpy(context_maps(image, union)), union,
                                          low=low, high=high, layer_name="mri")
            old_path = ROOT / "experiments/uncal_feature_probe_20260921/cache" / model_name / f"{name}.npz"
            with np.load(old_path) as old:
                if not np.array_equal(candidates, old["candidates"]) or target != int(old["target"]):
                    raise AssertionError(f"Frozen decoder cache mismatch: {name}")
                decoder = old["features__decoder1"]
            metrics = [score(fixed_cut_labels(prediction > 0, int(cut)), truth) for cut in candidates]
            original = score(prediction, truth)
            with path.with_suffix(".tmp").open("wb") as stream:
                np.savez_compressed(stream, name=name, candidates=candidates, target=target,
                                    original_cut=original_cut, original_dice=original["macro_dice"],
                                    original_swaps=original["ap_swaps"], base=base, mri=mri, decoder=decoder,
                                    cut_dice=[row["macro_dice"] for row in metrics],
                                    cut_swaps=[row["ap_swaps"] for row in metrics],
                                    decoder_cache_sha256=digest(old_path))
            path.with_suffix(".tmp").replace(path)
            if index % 10 == 0 or index == len(names) - 1:
                print(f"{model_name}: {index + 1}/{len(names)}", flush=True)


def select(args, train_names: list[str]) -> None:
    selection = {"protocol_sha256": digest(args.output / "PROTOCOL.md"),
                 "script_sha256": digest(Path(__file__)), "train_names": train_names, "models": {}}
    for model_name in args.models:
        cases = load_cases(args.cache / model_name, train_names)
        original_mae = float(np.mean([abs(int(c["original_cut"]) - int(c["target"])) for c in cases]))
        result = {"original_cut_mae": original_mae, "arms": {},
                  "manifest": json.loads((args.cache / model_name / "manifest.json").read_text())}
        for arm in ARMS:
            grid = []
            for c_value in (.01, .1, 1.):
                errors = []
                for train, held in KFold(4, shuffle=True, random_state=SEED).split(cases):
                    estimator = fit([cases[i] for i in train], arm, c_value)
                    errors.extend(abs(predict(estimator, cases[i], arm) - int(cases[i]["target"])) for i in held)
                grid.append({"C": c_value, "cv_mae": float(np.mean(errors))})
            best = min(grid, key=lambda row: (row["cv_mae"], row["C"]))
            result["arms"][arm] = {**best, "passes_training_gate": best["cv_mae"] < original_mae, "grid": grid}
            print(model_name, arm, result["arms"][arm], flush=True)
        selection["models"][model_name] = result
    save_json(args.output / "selection.json", selection)


def paired_interval(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    draws = np.random.default_rng(SEED).integers(len(values), size=(10000, len(values)))
    low, high = np.quantile(values[draws].mean(axis=1), [.025, .975])
    return {"mean": float(values.mean()), "ci95": [float(low), float(high)]}


def evaluate(args, train_names: list[str], val_names: list[str]) -> None:
    selection = json.loads((args.output / "selection.json").read_text())
    if selection["train_names"] != train_names or set(train_names) & set(val_names):
        raise ValueError("Split mismatch or leakage")
    if selection["protocol_sha256"] != digest(args.output / "PROTOCOL.md") or selection["script_sha256"] != digest(Path(__file__)):
        raise ValueError("Protocol or implementation changed after selection")
    report, all_rows = {}, {}
    for model_name in args.models:
        chosen = selection["models"][model_name]
        if chosen["manifest"] != json.loads((args.cache / model_name / "manifest.json").read_text()):
            raise ValueError("Cache changed after selection")
        train = load_cases(args.cache / model_name, train_names)
        estimators = {arm: fit(train, arm, chosen["arms"][arm]["C"]) for arm in ARMS}
        # Development cases are opened only after the choices above are frozen.
        cases = load_cases(args.cache / model_name, val_names)
        rows = []
        for case in cases:
            target, original_cut = int(case["target"]), int(case["original_cut"])
            row = {"name": str(case["name"].item()), "target": target,
                   "original": {"cut": original_cut, "mae": abs(original_cut - target),
                                "dice": float(case["original_dice"]), "swaps": int(case["original_swaps"])}}
            for arm in ("plane", *ARMS):
                cut = original_cut if arm == "plane" else predict(estimators[arm], case, arm)
                index = np.flatnonzero(case["candidates"] == cut)
                if len(index) != 1:
                    raise ValueError("Cut outside cached candidates")
                k = int(index[0])
                row[arm] = {"cut": cut, "mae": abs(cut - target),
                            "dice": float(case["cut_dice"][k]), "swaps": int(case["cut_swaps"][k])}
                if arm != "plane":
                    row[f"gated_{arm}"] = row[arm] if chosen["arms"][arm]["passes_training_gate"] else row["original"]
            rows.append(row)
        methods = ("original", "plane", *ARMS, *(f"gated_{arm}" for arm in ARMS))
        summaries = {method: {"mae": float(np.mean([r[method]["mae"] for r in rows])),
                              "dice": float(np.mean([r[method]["dice"] for r in rows])),
                              "swaps": int(sum(r[method]["swaps"] for r in rows))} for method in methods}
        comparisons = {}
        for arm in ARMS:
            for control in ("base", "original", "base_shuffled_mri"):
                if arm == control:
                    continue
                delta = np.array([r[arm]["mae"] - r[control]["mae"] for r in rows])
                comparisons[f"{arm}_minus_{control}"] = {
                    "mae": paired_interval(delta),
                    "dice_pp": paired_interval(100 * np.array([r[arm]["dice"] - r[control]["dice"] for r in rows])),
                    "cut_improved": int((delta < 0).sum()), "cut_worsened": int((delta > 0).sum())}
        report[model_name] = {"n": len(rows), "metrics": summaries, "comparisons": comparisons}
        all_rows[model_name] = rows
    save_json(args.output / "development_summary.json", report)
    save_json(args.output / "development_cases.json", all_rows)
    print(json.dumps({model: value["metrics"] for model, value in report.items()}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("extract", "select", "evaluate"), required=True)
    parser.add_argument("--models", nargs="+", choices=tuple(CHECKPOINTS), default=list(CHECKPOINTS))
    parser.add_argument("--cache", type=Path, default=ROOT / "experiments/context_increment_20260923/cache")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/context_increment_20260923")
    parser.add_argument("--split", type=Path, default=ROOT / "datasets/Dataset101_MSD/splits_final.json")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    split = json.loads(args.split.read_text())[0]
    train, val = sorted(split["train"]), sorted(split["val"])
    if set(train) & set(val):
        raise ValueError("Train/development overlap")
    if args.stage == "extract":
        extract(args, sorted(train + val))
    elif args.stage == "select":
        select(args, train)
    else:
        evaluate(args, train, val)


if __name__ == "__main__":
    main()
