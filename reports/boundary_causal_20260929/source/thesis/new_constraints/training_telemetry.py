"""Non-optimizing diagnostics for boundary-band training runs."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .bands.outer_boundary import build_boundary_bands


TELEMETRY_EPOCH_FIELDS = [
    "epoch",
    "loss_type",
    "inner_component_label",
    "outer_component_label",
    "voxel_count",
    "case_count",
    "gt_foreground_voxels",
    "pred_foreground_voxels",
    "foreground_volume_delta",
    "foreground_volume_ratio",
    "false_positive_voxels",
    "false_negative_voxels",
    "foreground_swap_voxels",
    "total_error_voxels",
    *[f"confusion_{truth}_{prediction}" for truth in range(3) for prediction in range(3)],
    "mean_foreground_probability",
    "mean_normalized_entropy",
    "foreground_brier",
    "margin_abs_lt_0_1_fraction",
    "margin_abs_lt_0_25_fraction",
    "margin_abs_lt_0_5_fraction",
    "margin_abs_lt_1_fraction",
    "inner_band_voxels",
    "outer_band_voxels",
    "inner_band_error_voxels",
    "outer_band_error_voxels",
    "inner_mean_truth_probability",
    "outer_mean_truth_probability",
    "inner_mean_focal_weight",
    "outer_mean_focal_weight",
    "inner_focal_weight_ess_fraction",
    "outer_focal_weight_ess_fraction",
    "inner_focal_weight_top10_mass_fraction",
    "outer_focal_weight_top10_mass_fraction",
    "inner_gradient_proxy_ess_fraction",
    "outer_gradient_proxy_ess_fraction",
    "inner_gradient_proxy_top10_mass_fraction",
    "outer_gradient_proxy_top10_mass_fraction",
    "inner_mean_objective_weight",
    "outer_mean_objective_weight",
    "inner_objective_weight_ess_fraction",
    "outer_objective_weight_ess_fraction",
    "inner_objective_weight_top10_mass_fraction",
    "outer_objective_weight_top10_mass_fraction",
]

TELEMETRY_GRADIENT_FIELDS = [
    "epoch",
    "case_count",
    "parameter_group",
    "inner_component_label",
    "outer_component_label",
    "supervised_gradient_norm",
    "inner_gradient_norm",
    "outer_gradient_norm",
    "inner_to_supervised_norm_ratio",
    "outer_to_supervised_norm_ratio",
    "supervised_inner_cosine",
    "supervised_outer_cosine",
    "inner_outer_cosine",
]


def _labels_without_channel(labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim == 5 and labels.shape[1] == 1:
        return labels[:, 0].long()
    if labels.ndim == 4:
        return labels.long()
    raise ValueError("labels must have shape [B, 1, X, Y, Z] or [B, X, Y, Z].")


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _mass_statistics(values: torch.Tensor) -> tuple[float, float]:
    values = values.float().reshape(-1)
    if values.numel() == 0:
        return 0.0, 0.0
    total = values.sum()
    if not bool(total > 0):
        return 0.0, 0.0
    ess = total.square() / values.square().sum().clamp_min(torch.finfo(values.dtype).tiny)
    count = max(1, math.ceil(0.10 * values.numel()))
    top_mass = torch.topk(values, count, sorted=False).values.sum() / total
    return float((ess / values.numel()).cpu()), float(top_mass.cpu())


def focal_gradient_magnitude(truth_probability: torch.Tensor, gamma: float) -> torch.Tensor:
    """Exact magnitude of d[focal BCE]/d[signed grouped logit]."""

    probability = truth_probability.float().clamp(
        torch.finfo(torch.float32).eps,
        1.0 - torch.finfo(torch.float32).eps,
    )
    one_minus = 1.0 - probability
    if gamma == 0.0:
        return one_minus
    weight = one_minus.pow(gamma)
    return weight * (one_minus + gamma * probability * (-torch.log(probability)))


class BoundaryTelemetryAccumulator:
    """Accumulate validation behavior without another model forward pass."""

    def __init__(
        self,
        *,
        foreground_class_ids: tuple[int, ...],
        complement_class_ids: tuple[int, ...],
        band_steps: int,
        inner_gamma: float,
        outer_gamma: float,
        num_classes: int,
        loss_type: str = "focal_bce",
        tversky_false_positive_weight: float = 0.60,
        tversky_false_negative_weight: float = 0.40,
    ) -> None:
        if num_classes != 3:
            raise ValueError("The current telemetry schema requires exactly three classes.")
        self.foreground_class_ids = foreground_class_ids
        self.complement_class_ids = complement_class_ids
        self.band_steps = band_steps
        self.inner_gamma = inner_gamma
        self.outer_gamma = outer_gamma
        self.num_classes = num_classes
        if loss_type not in {"focal_bce", "class_tversky"}:
            raise ValueError("Unknown telemetry boundary loss type.")
        self.loss_type = loss_type
        self.tversky_false_positive_weight = tversky_false_positive_weight
        self.tversky_false_negative_weight = tversky_false_negative_weight
        if loss_type == "class_tversky":
            self.component_labels = (
                f"class_{foreground_class_ids[0]}_tversky",
                f"class_{foreground_class_ids[1]}_tversky",
            )
        else:
            self.component_labels = ("inner_foreground_focal_bce", "outer_background_focal_bce")
        self.case_count = 0
        self.voxel_count = 0
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        self.probability_sum = 0.0
        self.entropy_sum = 0.0
        self.brier_sum = 0.0
        self.margin_near = {0.1: 0, 0.25: 0, 0.5: 0, 1.0: 0}
        self.band_values: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.band_errors = {"inner": 0, "outer": 0}

    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        labels = _labels_without_channel(labels)
        probabilities = torch.softmax(logits.float(), dim=1)
        prediction = logits.argmax(dim=1)
        truth_flat = labels.reshape(-1)
        prediction_flat = prediction.reshape(-1)
        counts = torch.bincount(
            truth_flat * self.num_classes + prediction_flat,
            minlength=self.num_classes * self.num_classes,
        ).reshape(self.num_classes, self.num_classes)
        self.confusion += counts.cpu()

        p_foreground = probabilities[:, self.foreground_class_ids].sum(dim=1)
        target_foreground = torch.zeros_like(labels, dtype=torch.bool)
        for class_id in self.foreground_class_ids:
            target_foreground |= labels == class_id
        pred_foreground = torch.zeros_like(prediction, dtype=torch.bool)
        for class_id in self.foreground_class_ids:
            pred_foreground |= prediction == class_id

        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
        entropy = entropy / math.log(self.num_classes)
        target_float = target_foreground.float()
        foreground_margin = torch.logsumexp(
            logits.float()[:, self.foreground_class_ids], dim=1
        ) - torch.logsumexp(logits.float()[:, self.complement_class_ids], dim=1)

        self.case_count += int(logits.shape[0])
        self.voxel_count += int(labels.numel())
        self.probability_sum += float(p_foreground.sum().cpu())
        self.entropy_sum += float(entropy.sum().cpu())
        self.brier_sum += float((p_foreground - target_float).square().sum().cpu())
        absolute_margin = foreground_margin.abs()
        for threshold in self.margin_near:
            self.margin_near[threshold] += int((absolute_margin < threshold).sum().item())

        if self.loss_type == "focal_bce":
            inner, outer = build_boundary_bands(
                target_foreground.unsqueeze(1), steps=self.band_steps
            )
            components = (
                ("inner", inner[:, 0], self.inner_gamma, p_foreground, ~pred_foreground),
                ("outer", outer[:, 0], self.outer_gamma, 1.0 - p_foreground, pred_foreground),
            )
            for side, mask, gamma, truth_probability, error in components:
                selected_probability = truth_probability[mask]
                objective_weight = (1.0 - selected_probability).clamp_min(
                    torch.finfo(torch.float32).eps
                ).pow(gamma)
                gradient_proxy = focal_gradient_magnitude(selected_probability, gamma)
                self._append_component(
                    side, selected_probability, objective_weight, gradient_proxy
                )
                self.band_errors[side] += int((error & mask).sum().item())
        else:
            for side, class_id in zip(("inner", "outer"), self.foreground_class_ids):
                target = labels == class_id
                class_inner, class_outer = build_boundary_bands(
                    target.unsqueeze(1), steps=self.band_steps
                )
                mask = class_inner[:, 0] | class_outer[:, 0]
                probability = probabilities[:, class_id]
                for case_index in range(labels.shape[0]):
                    case_mask = mask[case_index]
                    case_target = target[case_index].float()
                    case_probability = probability[case_index]
                    selected_probability = torch.where(
                        target[case_index], case_probability, 1.0 - case_probability
                    )[case_mask]
                    true_positive = (case_probability * case_target * case_mask).sum()
                    false_positive = (
                        case_probability * (1.0 - case_target) * case_mask
                    ).sum()
                    false_negative = (
                        (1.0 - case_probability) * case_target * case_mask
                    ).sum()
                    numerator = true_positive + 1e-6
                    denominator = (
                        true_positive
                        + self.tversky_false_positive_weight * false_positive
                        + self.tversky_false_negative_weight * false_negative
                        + 1e-6
                    )
                    positive_derivative = (
                        denominator
                        - numerator * (1.0 - self.tversky_false_negative_weight)
                    ) / denominator.square()
                    negative_derivative = (
                        numerator * self.tversky_false_positive_weight
                    ) / denominator.square()
                    objective_weight = torch.where(
                        target[case_index][case_mask],
                        positive_derivative,
                        negative_derivative,
                    ).abs()
                    selected_class_probability = case_probability[case_mask]
                    gradient_proxy = objective_weight * selected_class_probability * (
                        1.0 - selected_class_probability
                    )
                    self._append_component(
                        side, selected_probability, objective_weight, gradient_proxy
                    )
                self.band_errors[side] += int(
                    ((prediction != labels) & mask).sum().item()
                )

    def _append_component(
        self,
        side: str,
        truth_probability: torch.Tensor,
        objective_weight: torch.Tensor,
        gradient_proxy: torch.Tensor,
    ) -> None:
        self.band_values[f"{side}_truth_probability"].append(
            truth_probability.detach().float().cpu()
        )
        self.band_values[f"{side}_focal_weight"].append(
            objective_weight.detach().float().cpu()
        )
        self.band_values[f"{side}_gradient_proxy"].append(
            gradient_proxy.detach().float().cpu()
        )

    def finalize(self, epoch: int) -> dict[str, int | float]:
        confusion = self.confusion
        gt_foreground = int(confusion[1:, :].sum().item())
        pred_foreground = int(confusion[:, 1:].sum().item())
        false_positive = int(confusion[0, 1:].sum().item())
        false_negative = int(confusion[1:, 0].sum().item())
        swaps = int((confusion[1, 2] + confusion[2, 1]).item())
        row: dict[str, int | float] = {
            "epoch": epoch,
            "loss_type": self.loss_type,
            "inner_component_label": self.component_labels[0],
            "outer_component_label": self.component_labels[1],
            "voxel_count": self.voxel_count,
            "case_count": self.case_count,
            "gt_foreground_voxels": gt_foreground,
            "pred_foreground_voxels": pred_foreground,
            "foreground_volume_delta": pred_foreground - gt_foreground,
            "foreground_volume_ratio": _safe_ratio(pred_foreground, gt_foreground),
            "false_positive_voxels": false_positive,
            "false_negative_voxels": false_negative,
            "foreground_swap_voxels": swaps,
            "total_error_voxels": false_positive + false_negative + swaps,
            "mean_foreground_probability": _safe_ratio(
                self.probability_sum, self.voxel_count
            ),
            "mean_normalized_entropy": _safe_ratio(self.entropy_sum, self.voxel_count),
            "foreground_brier": _safe_ratio(self.brier_sum, self.voxel_count),
        }
        for truth in range(self.num_classes):
            for prediction in range(self.num_classes):
                row[f"confusion_{truth}_{prediction}"] = int(
                    confusion[truth, prediction].item()
                )
        for threshold, count in self.margin_near.items():
            suffix = "1" if threshold == 1.0 else str(threshold).replace(".", "_")
            row[f"margin_abs_lt_{suffix}_fraction"] = _safe_ratio(
                count, self.voxel_count
            )
        for side in ("inner", "outer"):
            probability = torch.cat(self.band_values[f"{side}_truth_probability"])
            weight = torch.cat(self.band_values[f"{side}_focal_weight"])
            gradient = torch.cat(self.band_values[f"{side}_gradient_proxy"])
            weight_ess, weight_top = _mass_statistics(weight)
            gradient_ess, gradient_top = _mass_statistics(gradient)
            row[f"{side}_band_voxels"] = int(probability.numel())
            row[f"{side}_band_error_voxels"] = self.band_errors[side]
            row[f"{side}_mean_truth_probability"] = float(probability.mean())
            row[f"{side}_mean_focal_weight"] = float(weight.mean())
            row[f"{side}_focal_weight_ess_fraction"] = weight_ess
            row[f"{side}_focal_weight_top10_mass_fraction"] = weight_top
            row[f"{side}_gradient_proxy_ess_fraction"] = gradient_ess
            row[f"{side}_gradient_proxy_top10_mass_fraction"] = gradient_top
            row[f"{side}_mean_objective_weight"] = float(weight.mean())
            row[f"{side}_objective_weight_ess_fraction"] = weight_ess
            row[f"{side}_objective_weight_top10_mass_fraction"] = weight_top
        return row


def parameter_group(name: str) -> str:
    """Map SwinUNETR parameters into interpretable coarse stages."""

    lower = name.lower()
    if lower.startswith("out") or ".out." in lower:
        return "segmentation_head"
    if "decoder" in lower:
        return "decoder"
    if "swinvit.layers1" in lower or "swinvit.layers2" in lower:
        return "early_transformer"
    if "swinvit.layers3" in lower or "swinvit.layers4" in lower:
        return "deep_transformer"
    if "encoder" in lower:
        return "convolutional_encoder"
    if "swinvit" in lower:
        return "transformer_other"
    return "other"


def _accumulate_gradient_pair_stats(
    accumulator: dict[str, dict[str, float]],
    names: list[str],
    supervised: Iterable[torch.Tensor | None],
    inner: Iterable[torch.Tensor | None],
    outer: Iterable[torch.Tensor | None],
) -> None:
    for name, supervised_grad, inner_grad, outer_grad in zip(
        names, supervised, inner, outer
    ):
        group = parameter_group(name)
        for target in ("all", group):
            values = accumulator[target]
            for key, gradient in (
                ("supervised_sq", supervised_grad),
                ("inner_sq", inner_grad),
                ("outer_sq", outer_grad),
            ):
                if gradient is not None:
                    values[key] += float(gradient.detach().float().square().sum().cpu())
            for key, left, right in (
                ("supervised_inner_dot", supervised_grad, inner_grad),
                ("supervised_outer_dot", supervised_grad, outer_grad),
                ("inner_outer_dot", inner_grad, outer_grad),
            ):
                if left is not None and right is not None:
                    values[key] += float(
                        (left.detach().float() * right.detach().float()).sum().cpu()
                    )


def _cosine(dot: float, left_sq: float, right_sq: float) -> float:
    denominator = math.sqrt(left_sq * right_sq)
    if denominator == 0:
        return 0.0
    return max(-1.0, min(1.0, dot / denominator))


def probe_component_gradients(
    model: nn.Module,
    loader: Iterable[dict[str, Any]],
    objective: nn.Module,
    supervised_loss_function: nn.Module,
    device: torch.device,
    *,
    epoch: int,
    constraint_scale: float,
    max_cases: int,
    spatial_cases: int,
) -> tuple[list[dict[str, int | float | str]], list[dict[str, Any]]]:
    """Measure component gradients without populating ``parameter.grad`` or stepping."""

    if max_cases < 1 or spatial_cases < 0 or spatial_cases > max_cases:
        raise ValueError("Telemetry case counts are inconsistent.")
    names_and_parameters = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    names = [name for name, _ in names_and_parameters]
    parameters = [parameter for _, parameter in names_and_parameters]
    previous_training = model.training
    checkpoint_flags = [
        (module, bool(module.use_checkpoint))
        for module in model.modules()
        if hasattr(module, "use_checkpoint")
    ]
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    accumulator: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    spatial_payloads: list[dict[str, Any]] = []
    cases_seen = 0
    model.eval()
    for module, _ in checkpoint_flags:
        module.use_checkpoint = False
    try:
        with torch.enable_grad():
            for batch in loader:
                if cases_seen >= max_cases:
                    break
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                case_names = [str(value) for value in batch.get("case_name", [])]
                for batch_index in range(images.shape[0]):
                    if cases_seen >= max_cases:
                        break
                    image = images[batch_index : batch_index + 1]
                    label = labels[batch_index : batch_index + 1]
                    logits = model(image).float()
                    supervised_loss = supervised_loss_function(logits, label)
                    band_result = objective.bands(logits, label)
                    valid = band_result.details["valid"]
                    if not bool(valid.any()):
                        continue
                    coefficient = constraint_scale * objective.config.bands_weight * 0.5
                    component_labels = band_result.details.get(
                        "component_labels", ("inner_band", "outer_band")
                    )
                    inner_loss = coefficient * band_result.details["inner_loss"][valid].mean()
                    outer_loss = coefficient * band_result.details["outer_loss"][valid].mean()
                    supervised_grads = torch.autograd.grad(
                        supervised_loss,
                        [*parameters, logits],
                        retain_graph=True,
                        allow_unused=True,
                    )
                    inner_grads = torch.autograd.grad(
                        inner_loss,
                        [*parameters, logits],
                        retain_graph=True,
                        allow_unused=True,
                    )
                    outer_grads = torch.autograd.grad(
                        outer_loss,
                        [*parameters, logits],
                        allow_unused=True,
                    )
                    _accumulate_gradient_pair_stats(
                        accumulator,
                        names,
                        supervised_grads[:-1],
                        inner_grads[:-1],
                        outer_grads[:-1],
                    )
                    if len(spatial_payloads) < spatial_cases:
                        label_no_channel = _labels_without_channel(label)[0]
                        probabilities = torch.softmax(logits.detach(), dim=1)
                        foreground_probability = probabilities[
                            0, objective.config.foreground_class_ids
                        ].sum(dim=0)
                        foreground = torch.zeros_like(label_no_channel, dtype=torch.bool)
                        for class_id in objective.config.foreground_class_ids:
                            foreground |= label_no_channel == class_id
                        if objective.config.bands_loss_type == "class_tversky":
                            component_masks = []
                            for class_id in objective.config.foreground_class_ids:
                                class_inner, class_outer = build_boundary_bands(
                                    (label_no_channel == class_id)[None, None],
                                    steps=objective.config.band_steps,
                                )
                                component_masks.append(class_inner | class_outer)
                            inner_mask, outer_mask = component_masks
                        else:
                            inner_mask, outer_mask = build_boundary_bands(
                                foreground[None, None], steps=objective.config.band_steps
                            )
                        spatial_payloads.append(
                            {
                                "epoch": epoch,
                                "case_name": (
                                    case_names[batch_index]
                                    if batch_index < len(case_names)
                                    else f"validation_case_{cases_seen}"
                                ),
                                "label": label_no_channel.detach().to(torch.uint8).cpu(),
                                "prediction": logits.detach().argmax(dim=1)[0].to(torch.uint8).cpu(),
                                "foreground_probability": foreground_probability.to(torch.float16).cpu(),
                                "inner_mask": inner_mask[0, 0].cpu(),
                                "outer_mask": outer_mask[0, 0].cpu(),
                                "inner_component_label": component_labels[0],
                                "outer_component_label": component_labels[1],
                                "supervised_logit_gradient": supervised_grads[-1][0].detach().to(torch.float16).cpu(),
                                "inner_logit_gradient": inner_grads[-1][0].detach().to(torch.float16).cpu(),
                                "outer_logit_gradient": outer_grads[-1][0].detach().to(torch.float16).cpu(),
                            }
                        )
                    cases_seen += 1
                    del supervised_grads, inner_grads, outer_grads, logits
    finally:
        for module, enabled in checkpoint_flags:
            module.use_checkpoint = enabled
        model.train(previous_training)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)

    if cases_seen == 0:
        raise ValueError("No valid fixed validation cases were available for telemetry.")
    rows: list[dict[str, int | float | str]] = []
    for group in sorted(accumulator, key=lambda value: (value != "all", value)):
        values = accumulator[group]
        supervised_sq = values["supervised_sq"]
        inner_sq = values["inner_sq"]
        outer_sq = values["outer_sq"]
        supervised_norm = math.sqrt(supervised_sq)
        inner_norm = math.sqrt(inner_sq)
        outer_norm = math.sqrt(outer_sq)
        rows.append(
            {
                "epoch": epoch,
                "case_count": cases_seen,
                "parameter_group": group,
                "inner_component_label": component_labels[0],
                "outer_component_label": component_labels[1],
                "supervised_gradient_norm": supervised_norm,
                "inner_gradient_norm": inner_norm,
                "outer_gradient_norm": outer_norm,
                "inner_to_supervised_norm_ratio": _safe_ratio(inner_norm, supervised_norm),
                "outer_to_supervised_norm_ratio": _safe_ratio(outer_norm, supervised_norm),
                "supervised_inner_cosine": _cosine(
                    values["supervised_inner_dot"], supervised_sq, inner_sq
                ),
                "supervised_outer_cosine": _cosine(
                    values["supervised_outer_dot"], supervised_sq, outer_sq
                ),
                "inner_outer_cosine": _cosine(
                    values["inner_outer_dot"], inner_sq, outer_sq
                ),
            }
        )
    return rows, spatial_payloads
