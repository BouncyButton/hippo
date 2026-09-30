#!/usr/bin/env python3
"""Audit whether frozen SwinUNETR features localise the uncal-apex cut.

Layer, spatial pooling region, and regularisation are chosen by subject-wise
cross-validation using only fold-0 training subjects.  The 52 fold-0
validation subjects are loaded only after model selection and are evaluated
once.  The segmentation checkpoints remain frozen throughout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import wilcoxon

from .audit_predicted_foreground import largest_foreground_component
from .feature_probe import (
    REGION_NAMES,
    ProbeCase,
    cut_metrics,
    fit_ranker,
    pool_layer_sequence,
    position_statistics,
    predict_ranker,
)
from .foldedness import best_fit_first_anterior_slice


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT_DIR = (
    REPOSITORY_ROOT / "experiments" / "augmentation_family_b_20260907" / "checkpoints"
)
DEFAULT_CACHE_ROOT = (
    REPOSITORY_ROOT / "experiments" / "uncal_feature_probe_20260921" / "cache"
)
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT / "docs" / "experiments" / "uncal_feature_probe_20260921"
)
CHECKPOINTS = {
    "unaugmented": "baseline_seed0_checkpoint_best.pt",
    "augmented": "augmentation_seed0_checkpoint_best.pt",
}
LAYERS = ("encoder1", "decoder2", "decoder1")
REGION_CONFIGS = {
    "union": ("union_mean", "union_max"),
    "union_plus_superior": ("union_mean", "union_max", "superior_band_mean"),
    "all_regions": REGION_NAMES,
}
C_VALUES = (0.01, 0.1, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("extract", "analyse", "all"), default="all")
    parser.add_argument("--models", nargs="+", choices=tuple(CHECKPOINTS), default=list(CHECKPOINTS))
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--pkl",
        type=Path,
        default=REPOSITORY_ROOT / "datasets" / "Dataset101_MSD" / "msd_hippocampus_full.pkl",
    )
    parser.add_argument(
        "--splits-json",
        type=Path,
        default=REPOSITORY_ROOT / "datasets" / "Dataset101_MSD" / "splits_final.json",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--overwrite-cache", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_names(path: Path) -> tuple[list[str], list[str]]:
    split = json.loads(path.read_text(encoding="utf-8"))[0]
    return sorted(split["train"]), sorted(split["val"])


def _foreground_dice(first: np.ndarray, second: np.ndarray) -> float:
    denominator = int(first.sum() + second.sum())
    return 2.0 * float(np.logical_and(first, second).sum()) / max(denominator, 1)


def _save_case(
    path: Path,
    *,
    name: str,
    candidates: np.ndarray,
    target: int,
    low: int,
    high: int,
    layer_features: dict[str, np.ndarray],
    feature_names: dict[str, tuple[str, ...]],
    foreground_dice: float,
    component_count: int,
    removed_voxels: int,
) -> None:
    payload: dict[str, np.ndarray] = {
        "name": np.asarray(name),
        "candidates": candidates,
        "target": np.asarray(target, dtype=np.int16),
        "low": np.asarray(low, dtype=np.int16),
        "high": np.asarray(high, dtype=np.int16),
        "foreground_dice": np.asarray(foreground_dice, dtype=np.float32),
        "component_count": np.asarray(component_count, dtype=np.int16),
        "removed_voxels": np.asarray(removed_voxels, dtype=np.int32),
    }
    for layer in LAYERS:
        payload[f"features__{layer}"] = layer_features[layer]
        payload[f"names__{layer}"] = np.asarray(feature_names[layer])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def _load_case(path: Path) -> tuple[ProbeCase, dict[str, float | int]]:
    with np.load(path, allow_pickle=False) as archive:
        layer_features = {
            layer: archive[f"features__{layer}"].astype(np.float32, copy=False)
            for layer in LAYERS
        }
        feature_names = {
            layer: tuple(str(value) for value in archive[f"names__{layer}"].tolist())
            for layer in LAYERS
        }
        case = ProbeCase(
            name=str(archive["name"].item()),
            candidates=archive["candidates"].astype(np.int16),
            target=int(archive["target"].item()),
            low=int(archive["low"].item()),
            high=int(archive["high"].item()),
            layer_features=layer_features,
            feature_names=feature_names,
        )
        metadata = {
            "foreground_dice": float(archive["foreground_dice"].item()),
            "component_count": int(archive["component_count"].item()),
            "removed_voxels": int(archive["removed_voxels"].item()),
        }
    return case, metadata


def extract_model_cache(args: argparse.Namespace, model_name: str, names: list[str]) -> None:
    import torch
    from torch.utils.data import DataLoader, Subset

    from baselines.swin_unetr.swin_unetr import (
        _build_monai_dataset_from_pkl,
        _load_pkl_dataframe,
    )
    from thesis.new_constraints.train_swinunetr_constraints import build_swinunetr

    checkpoint_path = args.checkpoint_dir / CHECKPOINTS[model_name]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    run = checkpoint.get("run", {})
    if int(run.get("fold", -1)) != 0 or int(run.get("seed", -1)) != 0:
        raise ValueError(f"{checkpoint_path}: expected fold 0, seed 0")
    dataframe = _load_pkl_dataframe(args.pkl)
    dataset = _build_monai_dataset_from_pkl(
        dataframe,
        "MSD",
        3,
        spatial_size=(64, 64, 64),
        do_resize=False,
    )
    index_by_name = {item["case_name"]: index for index, item in enumerate(dataset.data)}
    missing = sorted(set(names) - set(index_by_name))
    if missing:
        raise ValueError(f"dataset is missing {missing[:3]}")
    selected = names[: args.max_cases] if args.max_cases else names
    indices = [index_by_name[name] for name in selected]
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = build_swinunetr((64, 64, 64), 3, device, activation_checkpointing=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    activations: dict[str, torch.Tensor] = {}
    hooks = []
    for layer in LAYERS:
        module = getattr(model, layer)
        hooks.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, layer=layer: activations.__setitem__(layer, output)
            )
        )

    cache_dir = args.cache_root / model_name
    completed = 0
    try:
        with torch.no_grad():
            for batch in loader:
                batch_names = [str(name) for name in batch["case_name"]]
                pending = [
                    index
                    for index, name in enumerate(batch_names)
                    if args.overwrite_cache or not (cache_dir / f"{name}.npz").is_file()
                ]
                if not pending:
                    completed += len(batch_names)
                    continue
                images = batch["image"].float().to(device)
                labels = batch["label"].long().cpu().numpy()[:, 0]
                logits = model(images)
                predictions = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
                for batch_index in pending:
                    name = batch_names[batch_index]
                    label = labels[batch_index].astype(np.uint8, copy=False)
                    raw_union = predictions[batch_index] != 0
                    union, component_count, removed_voxels = largest_foreground_component(raw_union)
                    occupied = np.flatnonzero(union.sum(axis=(0, 2)))
                    if occupied.size < 3 or np.any(np.diff(occupied) != 1):
                        raise ValueError(f"{name}: predicted support lacks contiguous A/P extent")
                    low, high = int(occupied[0]), int(occupied[-1])
                    target, _, _ = best_fit_first_anterior_slice(label)
                    layer_features, feature_names = {}, {}
                    for layer in LAYERS:
                        matrix, names_for_layer = pool_layer_sequence(
                            activations[layer][batch_index],
                            union,
                            low=low,
                            high=high,
                            layer_name=layer,
                        )
                        layer_features[layer] = matrix
                        feature_names[layer] = names_for_layer
                    _save_case(
                        cache_dir / f"{name}.npz",
                        name=name,
                        candidates=np.arange(low + 1, high + 1, dtype=np.int16),
                        target=target,
                        low=low,
                        high=high,
                        layer_features=layer_features,
                        feature_names=feature_names,
                        foreground_dice=_foreground_dice(label != 0, union),
                        component_count=component_count,
                        removed_voxels=removed_voxels,
                    )
                completed += len(batch_names)
                activations.clear()
                if completed % 20 < len(batch_names) or completed == len(selected):
                    print(f"[{model_name}] cached {completed}/{len(selected)} cases", flush=True)
    finally:
        for hook in hooks:
            hook.remove()


def _load_cases(cache_dir: Path, names: list[str]) -> tuple[list[ProbeCase], dict[str, dict[str, float | int]]]:
    cases, metadata = [], {}
    for name in names:
        path = cache_dir / f"{name}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing feature cache: {path}")
        case, details = _load_case(path)
        cases.append(case)
        metadata[name] = details
    return cases, metadata


def _cross_validated_predictions(
    cases: list[ProbeCase],
    *,
    layer: str | None,
    regions: tuple[str, ...],
    include_position: bool,
    c_value: float,
) -> dict[str, int]:
    splits = KFold(n_splits=4, shuffle=True, random_state=0)
    predictions = {}
    case_array = np.asarray(cases, dtype=object)
    for train_indices, test_indices in splits.split(case_array):
        training = case_array[train_indices].tolist()
        held_out = case_array[test_indices].tolist()
        statistics = position_statistics(training)
        model, _, _ = fit_ranker(
            training,
            layer=layer,
            regions=regions,
            include_position=include_position,
            c_value=c_value,
            statistics=statistics,
        )
        fold_predictions, _ = predict_ranker(
            model,
            held_out,
            layer=layer,
            regions=regions,
            include_position=include_position,
            statistics=statistics,
        )
        predictions.update(fold_predictions)
    return predictions


def _selection_key(record: dict[str, object]) -> tuple[float, float, float, int]:
    metrics = record["metrics"]
    assert isinstance(metrics, dict)
    return (
        float(metrics["mae_slices"]),
        -float(metrics["within_1_fraction"]),
        float(metrics["p90_absolute_error"]),
        int(metrics["maximum_absolute_error"]),
    )


def _select_feature_configuration(train_cases: list[ProbeCase]) -> tuple[dict[str, object], list[dict[str, object]]]:
    records = []
    for layer in LAYERS:
        for region_name, regions in REGION_CONFIGS.items():
            for c_value in C_VALUES:
                predictions = _cross_validated_predictions(
                    train_cases,
                    layer=layer,
                    regions=regions,
                    include_position=False,
                    c_value=c_value,
                )
                records.append(
                    {
                        "layer": layer,
                        "region_configuration": region_name,
                        "regions": list(regions),
                        "c_value": c_value,
                        "metrics": cut_metrics(predictions, train_cases),
                    }
                )
    records.sort(key=_selection_key)
    return records[0], records


def _select_c(
    train_cases: list[ProbeCase],
    *,
    layer: str | None,
    regions: tuple[str, ...],
    include_position: bool,
) -> tuple[float, list[dict[str, object]]]:
    records = []
    for c_value in C_VALUES:
        predictions = _cross_validated_predictions(
            train_cases,
            layer=layer,
            regions=regions,
            include_position=include_position,
            c_value=c_value,
        )
        records.append(
            {"c_value": c_value, "metrics": cut_metrics(predictions, train_cases)}
        )
    records.sort(key=_selection_key)
    return float(records[0]["c_value"]), records


def _fit_and_evaluate(
    train_cases: list[ProbeCase],
    validation_cases: list[ProbeCase],
    *,
    layer: str | None,
    regions: tuple[str, ...],
    include_position: bool,
    c_value: float,
) -> tuple[dict[str, object], dict[str, int], dict[str, np.ndarray]]:
    statistics = position_statistics(train_cases)
    model, names, used_names = fit_ranker(
        train_cases,
        layer=layer,
        regions=regions,
        include_position=include_position,
        c_value=c_value,
        statistics=statistics,
    )
    predictions, scores = predict_ranker(
        model,
        validation_cases,
        layer=layer,
        regions=regions,
        include_position=include_position,
        statistics=statistics,
    )
    coefficients = model.named_steps["logisticregression"].coef_[0]
    order = np.argsort(np.abs(coefficients))[::-1][:20]
    top = [
        {"feature": names[index], "standardized_coefficient": float(coefficients[index])}
        for index in order
    ]
    attribution = {}
    for region in (*REGION_NAMES, "position"):
        indices = [
            index
            for index, name in enumerate(names)
            if (region == "position" and name.startswith("position__"))
            or (region != "position" and f"__{region}__" in name)
        ]
        if indices:
            attribution[region] = float(np.abs(coefficients[indices]).sum())
    total = sum(attribution.values())
    attribution = {name: value / max(total, 1e-12) for name, value in attribution.items()}
    operation_attribution = {}
    for operation in ("current", "delta", "next_delta", "local_contrast"):
        indices = [
            index for index, name in enumerate(names) if f"__{operation}__" in name
        ]
        if indices:
            operation_attribution[operation] = float(np.abs(coefficients[indices]).sum())
    operation_total = sum(operation_attribution.values())
    operation_attribution = {
        name: value / max(operation_total, 1e-12)
        for name, value in operation_attribution.items()
    }
    result = {
        "layer": layer,
        "regions": list(regions),
        "include_position": include_position,
        "c_value": c_value,
        "training_cases_used": len(used_names),
        "training_cases_excluded_target_outside_prediction": len(train_cases) - len(used_names),
        "position_statistics": {"median": statistics[0], "robust_scale": statistics[1]},
        "validation_metrics": cut_metrics(predictions, validation_cases),
        "coefficient_mass_by_region": attribution,
        "coefficient_mass_by_operation": operation_attribution,
        "top_coefficients": top,
    }
    return result, predictions, scores


def _paired_errors(
    first: dict[str, int],
    second: dict[str, int],
    cases: list[ProbeCase],
) -> dict[str, float | int]:
    targets = {case.name: case.target for case in cases}
    names = sorted(first)
    first_errors = np.asarray([abs(first[name] - targets[name]) for name in names])
    second_errors = np.asarray([abs(second[name] - targets[name]) for name in names])
    differences = second_errors - first_errors
    try:
        p_value = float(wilcoxon(second_errors, first_errors).pvalue)
    except ValueError:
        p_value = 1.0
    return {
        "second_better_cases": int((differences < 0).sum()),
        "equal_cases": int((differences == 0).sum()),
        "second_worse_cases": int((differences > 0).sum()),
        "mean_absolute_error_change_second_minus_first": float(differences.mean()),
        "wilcoxon_two_sided_p_value_exploratory": p_value,
        "cut_agreement_fraction": float(
            np.mean([first[name] == second[name] for name in names])
        ),
    }


def _support_summary(metadata: dict[str, dict[str, float | int]]) -> dict[str, float | int]:
    values = list(metadata.values())
    dice = np.asarray([value["foreground_dice"] for value in values], dtype=float)
    return {
        "mean_foreground_dice": float(dice.mean()),
        "median_foreground_dice": float(np.median(dice)),
        "cases_with_disconnected_islands": int(sum(value["component_count"] > 1 for value in values)),
        "total_removed_island_voxels": int(sum(value["removed_voxels"] for value in values)),
    }


def analyse_model(
    args: argparse.Namespace,
    model_name: str,
    train_names: list[str],
    validation_names: list[str],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    cache_dir = args.cache_root / model_name
    train_cases, train_metadata = _load_cases(cache_dir, train_names)
    validation_cases, validation_metadata = _load_cases(cache_dir, validation_names)

    best, search = _select_feature_configuration(train_cases)
    layer = str(best["layer"])
    regions = tuple(str(value) for value in best["regions"])
    feature_c = float(best["c_value"])
    combined_c, combined_search = _select_c(
        train_cases,
        layer=layer,
        regions=regions,
        include_position=True,
    )
    position_c, position_search = _select_c(
        train_cases,
        layer=None,
        regions=(),
        include_position=True,
    )

    feature_result, feature_predictions, feature_scores = _fit_and_evaluate(
        train_cases,
        validation_cases,
        layer=layer,
        regions=regions,
        include_position=False,
        c_value=feature_c,
    )
    combined_result, combined_predictions, combined_scores = _fit_and_evaluate(
        train_cases,
        validation_cases,
        layer=layer,
        regions=regions,
        include_position=True,
        c_value=combined_c,
    )
    position_result, position_predictions, position_scores = _fit_and_evaluate(
        train_cases,
        validation_cases,
        layer=None,
        regions=(),
        include_position=True,
        c_value=position_c,
    )
    by_name = {case.name: case for case in validation_cases}
    rows = []
    for name in validation_names:
        case = by_name[name]
        row: dict[str, object] = {
            "model": model_name,
            "case": name,
            "target_cut": case.target,
            "predicted_low": case.low,
            "predicted_high": case.high,
            "foreground_dice": validation_metadata[name]["foreground_dice"],
        }
        for mode, predictions, scores in (
            ("feature_only", feature_predictions, feature_scores),
            ("feature_plus_position", combined_predictions, combined_scores),
            ("position_only", position_predictions, position_scores),
        ):
            prediction = predictions[name]
            row[f"{mode}_cut"] = prediction
            row[f"{mode}_absolute_error"] = abs(prediction - case.target)
            row[f"{mode}_peak_probability"] = float(np.max(scores[name]))
        rows.append(row)

    return (
        {
            "checkpoint": str((args.checkpoint_dir / CHECKPOINTS[model_name]).resolve()),
            "checkpoint_sha256": _sha256(args.checkpoint_dir / CHECKPOINTS[model_name]),
            "selected_feature_configuration": best,
            "nested_feature_search": search,
            "combined_c_search": combined_search,
            "position_c_search": position_search,
            "results": {
                "feature_only": feature_result,
                "feature_plus_position": combined_result,
                "position_only": position_result,
            },
            "paired_position_only_vs_feature_only": _paired_errors(
                position_predictions,
                feature_predictions,
                validation_cases,
            ),
            "support": {
                "training": _support_summary(train_metadata),
                "validation": _support_summary(validation_metadata),
            },
        },
        rows,
    )


def _write_report(args: argparse.Namespace, summary: dict[str, object], rows: list[dict[str, object]]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "per_case.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Frozen feature probe for the uncal-apex cut",
        "",
        "The segmentation models were frozen. Layer, pooling region, and logistic",
        "regularisation were selected by four-fold subject-wise cross-validation on",
        "the 208 fold-0 training subjects. The 52 validation subjects were evaluated",
        "once after selection. `feature_only` contains no explicit A/P coordinate.",
        "",
        "| model | probe | MAE | exact | within 1 | within 2 | p90 | max | endpoint |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    models = summary["models"]
    assert isinstance(models, dict)
    for model_name, model_summary in models.items():
        for mode, result in model_summary["results"].items():
            metrics = result["validation_metrics"]
            lines.append(
                f"| {model_name} | {mode} | {metrics['mae_slices']:.3f} | "
                f"{metrics['exact_fraction']:.3f} | {metrics['within_1_fraction']:.3f} | "
                f"{metrics['within_2_fraction']:.3f} | {metrics['p90_absolute_error']:.1f} | "
                f"{metrics['maximum_absolute_error']} | {metrics['endpoint_prediction_fraction']:.3f} |"
            )
    lines.extend(["", "## Selected hidden representation", ""])
    for model_name, model_summary in models.items():
        selected = model_summary["selected_feature_configuration"]
        paired = model_summary["paired_position_only_vs_feature_only"]
        lines.append(
            f"- **{model_name}:** `{selected['layer']}`, "
            f"`{selected['region_configuration']}`, C={selected['c_value']}. "
            f"Feature-only was better/equal/worse than position-only in "
            f"{paired['second_better_cases']}/{paired['equal_cases']}/"
            f"{paired['second_worse_cases']} cases (exploratory paired Wilcoxon "
            f"p={paired['wilcoxon_two_sided_p_value_exploratory']:.4f})."
        )
    lines.extend(
        [
            "",
            "For both checkpoints the selected layer was the final full-resolution",
            "decoder. Position contributed less than 1.4% of total absolute coefficient",
            "mass and did not change any validation prediction when added to the hidden",
            "features.",
            "",
            "## Handcrafted predicted-foreground comparator",
            "",
            "| model | MAE | exact | within 1 | within 2 | p90 | max |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    handcrafted = summary["handcrafted_external_comparator"]
    for model_name, key in (
        ("unaugmented", "unaugmented_prediction"),
        ("augmented", "augmented_prediction"),
    ):
        metrics = handcrafted[key]
        lines.append(
            f"| {model_name} | {metrics['mae_slices']:.3f} | "
            f"{metrics['exact_fraction']:.3f} | {metrics['within_1_fraction']:.3f} | "
            f"{metrics['within_2_fraction']:.3f} | {metrics['p90_absolute_error']:.1f} | "
            f"{metrics['maximum_absolute_error']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "A feature-only improvement over position-only is evidence that the frozen",
            "network contains local visual information about the fold. Improvement only",
            "after adding position means the representation is useful mainly when anchored",
            "by a dataset prior. Validation results must not be used to retune the probe.",
            "",
            "The handcrafted comparator is reported in the adjacent predicted-foreground",
            "audit; it was not used to select this probe.",
            "",
            "## Limitations",
            "",
            "The segmentation checkpoints were trained on the same 208 subjects used for",
            "inner probe selection, so the very low training-CV errors are optimistic and",
            "are not performance estimates. Only the untouched 52-case validation metrics",
            "should be interpreted as generalisation. The logistic peak values are ranking",
            "scores, not calibrated probabilities. Finally, pooling still depends on the",
            "baseline's predicted foreground support.",
        ]
    )
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    train_names, validation_names = _split_names(args.splits_json)
    all_names = train_names + validation_names
    if args.stage in ("extract", "all"):
        for model_name in args.models:
            extract_model_cache(args, model_name, all_names)
    if args.stage in ("analyse", "all"):
        if set(args.models) != set(CHECKPOINTS):
            raise ValueError("analysis requires both canonical models")
        model_results, all_rows = {}, []
        for model_name in CHECKPOINTS:
            result, rows = analyse_model(
                args,
                model_name,
                train_names,
                validation_names,
            )
            model_results[model_name] = result
            all_rows.extend(rows)
        handcrafted_path = (
            REPOSITORY_ROOT
            / "docs"
            / "experiments"
            / "uncal_foldedness_predicted_foreground_early_stopping_20260921"
            / "summary.json"
        )
        handcrafted = json.loads(handcrafted_path.read_text(encoding="utf-8"))
        summary = {
            "schema": "uncal_fold.frozen_feature_probe.v1",
            "fold": 0,
            "seed": 0,
            "training_case_count": len(train_names),
            "validation_case_count": len(validation_names),
            "protocol": {
                "network_updates": "none; both SwinUNETR checkpoints frozen",
                "selection": "four-fold subject-wise CV on 208 training cases",
                "validation": "single untouched evaluation after model selection",
                "candidate_support": "largest 26-connected predicted foreground component",
                "feature_only_position_input": False,
            },
            "models": model_results,
            "handcrafted_external_comparator": handcrafted["cut_metrics"],
        }
        by_model = {
            model_name: {
                row["case"]: int(row["feature_only_absolute_error"])
                for row in all_rows
                if row["model"] == model_name
            }
            for model_name in CHECKPOINTS
        }
        names = sorted(by_model["unaugmented"])
        unaugmented_errors = np.asarray(
            [by_model["unaugmented"][name] for name in names]
        )
        augmented_errors = np.asarray([by_model["augmented"][name] for name in names])
        differences = augmented_errors - unaugmented_errors
        summary["paired_unaugmented_vs_augmented_feature_only"] = {
            "augmented_better_cases": int((differences < 0).sum()),
            "equal_cases": int((differences == 0).sum()),
            "augmented_worse_cases": int((differences > 0).sum()),
            "mean_absolute_error_change_augmented_minus_unaugmented": float(
                differences.mean()
            ),
            "wilcoxon_two_sided_p_value_exploratory": float(
                wilcoxon(augmented_errors, unaugmented_errors).pvalue
            ),
        }
        _write_report(args, summary, all_rows)


if __name__ == "__main__":
    main()
