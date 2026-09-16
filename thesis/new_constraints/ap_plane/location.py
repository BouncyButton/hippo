"""Patient-specific supervision of the protocol-derived A/P annotation plane."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..constraint_result import ConstraintResult, differentiable_zero
from .existential import (
    _case_plane_losses,
    _legal_cuts,
    _normalise_labels,
    _validate_logits,
)


AP_PLANE_LOCATION_METRICS = (
    "raw_loss",
    "target_cut",
    "target_cost",
    "target_cost_gap",
    "target_cost_gap_fraction",
    "selected_cut",
    "cut_abs_error",
    "exact_cut",
    "raw_selected_cut",
    "raw_cut_abs_error",
    "raw_exact_cut",
    "raw_within_one_cut",
    "raw_plane_disagreement_fraction",
    "raw_has_both_ap_classes",
    "raw_cut_valid",
    "foreground_union_dice",
    "ap_swap_voxels",
    "ap_swap_fraction",
    "displaced_slab_swap_voxels",
    "displaced_slab_swap_fraction",
    "gt_plane_disagreement_fraction",
    "target_cut_count",
    "candidate_count",
    "valid_patient",
    "skipped_patient",
)


def _ground_truth_plane_costs(
    label: torch.Tensor,
    *,
    axis: int,
    anterior_high: bool,
    require_both: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return legal cuts and their hard A/P disagreement counts."""

    moved = label.movedim(axis, 0).reshape(label.shape[axis], -1)
    support = moved != 0
    cuts = _legal_cuts(support.sum(dim=1), require_both=require_both)
    if cuts.numel() == 0:
        return cuts, cuts.to(dtype=torch.float32)

    anterior = (moved == 1).sum(dim=1)
    posterior = (moved == 2).sum(dim=1)
    if anterior_high:
        costs = (
            anterior.cumsum(dim=0)[cuts - 1]
            + posterior.flip(0).cumsum(dim=0).flip(0)[cuts]
        )
    else:
        costs = (
            posterior.cumsum(dim=0)[cuts - 1]
            + anterior.flip(0).cumsum(dim=0).flip(0)[cuts]
        )
    return cuts, costs.to(dtype=torch.float32)


class BestFitAPPlaneLocationLoss(nn.Module):
    """Teach the existing A/P logits the best label-derived cut for each case.

    The target cut minimizes hard A/P disagreements in the released label.  This
    keeps every case usable when one or two coronal slices contain a small class
    mixture.  If multiple cuts tie, the loss accepts whichever tied target is
    cheapest under the current logits; Dataset101 has a unique optimum in all
    260 released labels, but the implementation does not assume that fact.
    """

    def __init__(
        self,
        *,
        axis: int = 1,
        anterior_high: bool = True,
        margin: float = 0.0,
        require_both: bool = True,
        adherence_threshold: float = 0.95,
    ) -> None:
        super().__init__()
        if type(axis) is not int or axis not in (0, 1, 2):
            raise ValueError("axis must be one of the three spatial tensor axes.")
        if type(anterior_high) is not bool or type(require_both) is not bool:
            raise ValueError("anterior_high and require_both must be booleans.")
        if not math.isfinite(margin) or margin < 0:
            raise ValueError("margin must be finite and non-negative.")
        if not 0 <= adherence_threshold <= 1:
            raise ValueError("adherence_threshold must lie in [0,1].")
        self.axis = axis
        self.anterior_high = anterior_high
        self.margin = float(margin)
        self.require_both = require_both
        self.adherence_threshold = float(adherence_threshold)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> ConstraintResult:
        _validate_logits(logits, self.axis)
        labels = _normalise_labels(labels, logits)
        ap_margin = (logits[:, 1] - logits[:, 2]).float()
        support = labels != 0
        zero = differentiable_zero(ap_margin)

        losses: list[torch.Tensor] = []
        target_cuts: list[int] = []
        selected_cuts: list[int] = []
        cut_errors: list[int] = []
        exact_cuts: list[float] = []
        disagreement_fractions: list[float] = []
        target_costs: list[float] = []
        target_cost_gaps: list[float] = []
        target_cost_gap_fractions: list[float] = []
        target_counts: list[int] = []
        candidate_counts: list[int] = []
        raw_selected_cuts: list[int] = []
        raw_cut_errors: list[int] = []
        raw_exact_cuts: list[float] = []
        raw_within_one_cuts: list[float] = []
        raw_plane_disagreement_fractions: list[float] = []
        raw_has_both_classes: list[float] = []
        raw_cut_valids: list[float] = []
        foreground_union_dices: list[float] = []
        ap_swap_voxels: list[int] = []
        ap_swap_fractions: list[float] = []
        displaced_slab_swap_voxels: list[int] = []
        displaced_slab_swap_fractions: list[float] = []
        valid: list[bool] = []

        # These hard-prediction diagnostics do not enter the loss.  They are
        # deliberately computed from raw argmax support, because a cut fitted
        # on ground-truth foreground is an oracle-support diagnostic rather
        # than the cut produced by the deployed segmenter.
        prediction = logits.argmax(dim=1)

        for case_index in range(logits.shape[0]):
            candidates, predicted_costs = _case_plane_losses(
                ap_margin[case_index],
                support[case_index],
                axis=self.axis,
                anterior_high=self.anterior_high,
                margin=self.margin,
                require_both=self.require_both,
            )
            gt_candidates, gt_costs = _ground_truth_plane_costs(
                labels[case_index],
                axis=self.axis,
                anterior_high=self.anterior_high,
                require_both=self.require_both,
            )
            is_valid = bool(candidates.numel()) and torch.equal(candidates, gt_candidates)
            valid.append(is_valid)
            candidate_counts.append(int(candidates.numel()))
            if not is_valid:
                losses.append(zero)
                target_cuts.append(-1)
                selected_cuts.append(-1)
                cut_errors.append(0)
                exact_cuts.append(0.0)
                disagreement_fractions.append(0.0)
                target_costs.append(0.0)
                target_cost_gaps.append(0.0)
                target_cost_gap_fractions.append(0.0)
                target_counts.append(0)
                raw_selected_cuts.append(-1)
                raw_cut_errors.append(0)
                raw_exact_cuts.append(0.0)
                raw_within_one_cuts.append(0.0)
                raw_plane_disagreement_fractions.append(0.0)
                raw_has_both_classes.append(0.0)
                raw_cut_valids.append(0.0)
                foreground_union_dices.append(0.0)
                ap_swap_voxels.append(0)
                ap_swap_fractions.append(0.0)
                displaced_slab_swap_voxels.append(0)
                displaced_slab_swap_fractions.append(0.0)
                continue

            minimum_gt_cost = gt_costs.min()
            target_mask = gt_costs == minimum_gt_cost
            target_indices = torch.nonzero(target_mask, as_tuple=False).flatten()
            target_candidates = candidates[target_indices]
            selected_index = predicted_costs.argmin()
            selected_cut = candidates[selected_index]
            distances = (target_candidates - selected_cut).abs()
            nearest_target_index = distances.argmin()
            nearest_target_cut = target_candidates[nearest_target_index]

            # Supervise the best current member of the label-optimal target set.
            target_predicted_costs = predicted_costs[target_indices]
            losses.append(target_predicted_costs.min())
            target_cuts.append(int(nearest_target_cut))
            selected_cuts.append(int(selected_cut))
            cut_error = int(distances[nearest_target_index])
            cut_errors.append(cut_error)
            exact_cuts.append(float(cut_error == 0))
            foreground_count = int(support[case_index].sum())
            minimum_gt_cost_value = float(minimum_gt_cost)
            disagreement_fractions.append(minimum_gt_cost_value / foreground_count)
            target_costs.append(minimum_gt_cost_value)
            non_target_costs = gt_costs[~target_mask]
            target_gap = (
                float(non_target_costs.min() - minimum_gt_cost)
                if bool(non_target_costs.numel())
                else 0.0
            )
            target_cost_gaps.append(target_gap)
            target_cost_gap_fractions.append(target_gap / foreground_count)
            target_counts.append(int(target_indices.numel()))

            case_prediction = prediction[case_index]
            predicted_foreground = case_prediction != 0
            predicted_foreground_count = int(predicted_foreground.sum())
            has_both_classes = bool((case_prediction == 1).any()) and bool(
                (case_prediction == 2).any()
            )
            raw_candidates, raw_costs = _ground_truth_plane_costs(
                case_prediction,
                axis=self.axis,
                anterior_high=self.anterior_high,
                require_both=self.require_both,
            )
            raw_valid = has_both_classes and bool(raw_candidates.numel())
            raw_has_both_classes.append(float(has_both_classes))
            raw_cut_valids.append(float(raw_valid))
            if raw_valid:
                raw_index = raw_costs.argmin()
                raw_cut = raw_candidates[raw_index]
                raw_error = int((target_candidates - raw_cut).abs().min())
                raw_selected_cuts.append(int(raw_cut))
                raw_cut_errors.append(raw_error)
                raw_exact_cuts.append(float(raw_error == 0))
                raw_within_one_cuts.append(float(raw_error <= 1))
                raw_plane_disagreement_fractions.append(
                    float(raw_costs[raw_index]) / predicted_foreground_count
                )
            else:
                raw_cut = None
                raw_selected_cuts.append(-1)
                raw_cut_errors.append(0)
                raw_exact_cuts.append(0.0)
                raw_within_one_cuts.append(0.0)
                raw_plane_disagreement_fractions.append(0.0)

            case_support = support[case_index]
            shared_foreground = predicted_foreground & case_support
            union_denominator = predicted_foreground_count + foreground_count
            foreground_union_dices.append(
                2.0 * float(shared_foreground.sum()) / union_denominator
                if union_denominator
                else 1.0
            )
            swaps = shared_foreground & (case_prediction != labels[case_index])
            swap_count = int(swaps.sum())
            ap_swap_voxels.append(swap_count)
            ap_swap_fractions.append(swap_count / foreground_count)

            if raw_cut is not None:
                target_cut = int(nearest_target_cut)
                raw_cut_value = int(raw_cut)
                low = min(target_cut, raw_cut_value)
                high = max(target_cut, raw_cut_value)
                coordinates = torch.arange(
                    labels.shape[self.axis + 1], device=labels.device
                )
                coordinate_shape = [1, 1, 1]
                coordinate_shape[self.axis] = -1
                displaced_slab = (coordinates.reshape(coordinate_shape) >= low) & (
                    coordinates.reshape(coordinate_shape) < high
                )
                slab_swap_count = int((swaps & displaced_slab).sum())
            else:
                slab_swap_count = 0
            displaced_slab_swap_voxels.append(slab_swap_count)
            displaced_slab_swap_fractions.append(
                slab_swap_count / swap_count if swap_count else 0.0
            )

        case_loss = torch.stack(losses)
        valid_tensor = torch.tensor(valid, dtype=torch.bool, device=logits.device)
        target_tensor = torch.tensor(target_cuts, dtype=torch.long, device=logits.device)
        selected_tensor = torch.tensor(selected_cuts, dtype=torch.long, device=logits.device)
        error_tensor = torch.tensor(cut_errors, dtype=torch.float32, device=logits.device)
        exact_tensor = torch.tensor(exact_cuts, dtype=torch.float32, device=logits.device)
        disagreement_tensor = torch.tensor(
            disagreement_fractions, dtype=torch.float32, device=logits.device
        )
        target_cost_tensor = torch.tensor(
            target_costs, dtype=torch.float32, device=logits.device
        )
        target_gap_tensor = torch.tensor(
            target_cost_gaps, dtype=torch.float32, device=logits.device
        )
        target_gap_fraction_tensor = torch.tensor(
            target_cost_gap_fractions, dtype=torch.float32, device=logits.device
        )
        target_count_tensor = torch.tensor(
            target_counts, dtype=torch.float32, device=logits.device
        )
        candidate_count_tensor = torch.tensor(
            candidate_counts, dtype=torch.float32, device=logits.device
        )
        raw_selected_tensor = torch.tensor(
            raw_selected_cuts, dtype=torch.long, device=logits.device
        )
        raw_error_tensor = torch.tensor(
            raw_cut_errors, dtype=torch.float32, device=logits.device
        )
        raw_exact_tensor = torch.tensor(
            raw_exact_cuts, dtype=torch.float32, device=logits.device
        )
        raw_within_one_tensor = torch.tensor(
            raw_within_one_cuts, dtype=torch.float32, device=logits.device
        )
        raw_plane_disagreement_tensor = torch.tensor(
            raw_plane_disagreement_fractions, dtype=torch.float32, device=logits.device
        )
        raw_has_both_tensor = torch.tensor(
            raw_has_both_classes, dtype=torch.float32, device=logits.device
        )
        raw_valid_tensor = torch.tensor(
            raw_cut_valids, dtype=torch.bool, device=logits.device
        )
        union_dice_tensor = torch.tensor(
            foreground_union_dices, dtype=torch.float32, device=logits.device
        )
        swap_voxel_tensor = torch.tensor(
            ap_swap_voxels, dtype=torch.float32, device=logits.device
        )
        swap_fraction_tensor = torch.tensor(
            ap_swap_fractions, dtype=torch.float32, device=logits.device
        )
        slab_swap_voxel_tensor = torch.tensor(
            displaced_slab_swap_voxels, dtype=torch.float32, device=logits.device
        )
        slab_swap_fraction_tensor = torch.tensor(
            displaced_slab_swap_fractions, dtype=torch.float32, device=logits.device
        )
        loss = case_loss[valid_tensor].mean() if bool(valid_tensor.any()) else zero
        truth_all = torch.exp(-case_loss).clamp(0.0, 1.0)
        truth = truth_all[valid_tensor]
        return ConstraintResult(
            loss=loss,
            truth=truth,
            value=error_tensor[valid_tensor],
            details={
                "confidence_weighted_agreement": truth,
                "confidence_adherent": truth >= self.adherence_threshold,
                "valid": valid_tensor,
                "case_loss": case_loss,
                "target_cut": target_tensor,
                "selected_cut": selected_tensor,
                "cut_abs_error": error_tensor,
                "exact_cut": exact_tensor,
                "target_cost": target_cost_tensor,
                "target_cost_gap": target_gap_tensor,
                "target_cost_gap_fraction": target_gap_fraction_tensor,
                "gt_plane_disagreement_fraction": disagreement_tensor,
                "target_cut_count": target_count_tensor,
                "candidate_count": candidate_count_tensor,
                "raw_selected_cut": raw_selected_tensor,
                "raw_cut_abs_error": raw_error_tensor,
                "raw_exact_cut": raw_exact_tensor,
                "raw_within_one_cut": raw_within_one_tensor,
                "raw_plane_disagreement_fraction": raw_plane_disagreement_tensor,
                "raw_has_both_ap_classes": raw_has_both_tensor,
                "raw_cut_valid": raw_valid_tensor,
                "foreground_union_dice": union_dice_tensor,
                "ap_swap_voxels": swap_voxel_tensor,
                "ap_swap_fraction": swap_fraction_tensor,
                "displaced_slab_swap_voxels": slab_swap_voxel_tensor,
                "displaced_slab_swap_fraction": slab_swap_fraction_tensor,
                "metrics": {
                    "raw_loss": case_loss[valid_tensor],
                    "target_cut": target_tensor[valid_tensor].float(),
                    "target_cost": target_cost_tensor[valid_tensor],
                    "target_cost_gap": target_gap_tensor[valid_tensor],
                    "target_cost_gap_fraction": target_gap_fraction_tensor[valid_tensor],
                    "selected_cut": selected_tensor[valid_tensor].float(),
                    "cut_abs_error": error_tensor[valid_tensor],
                    "exact_cut": exact_tensor[valid_tensor],
                    "raw_selected_cut": raw_selected_tensor[raw_valid_tensor].float(),
                    "raw_cut_abs_error": raw_error_tensor[raw_valid_tensor],
                    "raw_exact_cut": raw_exact_tensor[raw_valid_tensor],
                    "raw_within_one_cut": raw_within_one_tensor[raw_valid_tensor],
                    "raw_plane_disagreement_fraction": (
                        raw_plane_disagreement_tensor[raw_valid_tensor]
                    ),
                    "raw_has_both_ap_classes": raw_has_both_tensor[valid_tensor],
                    "raw_cut_valid": raw_valid_tensor[valid_tensor].float(),
                    "foreground_union_dice": union_dice_tensor[valid_tensor],
                    "ap_swap_voxels": swap_voxel_tensor[valid_tensor],
                    "ap_swap_fraction": swap_fraction_tensor[valid_tensor],
                    "displaced_slab_swap_voxels": (
                        slab_swap_voxel_tensor[raw_valid_tensor]
                    ),
                    "displaced_slab_swap_fraction": (
                        slab_swap_fraction_tensor[raw_valid_tensor]
                    ),
                    "gt_plane_disagreement_fraction": disagreement_tensor[valid_tensor],
                    "target_cut_count": target_count_tensor[valid_tensor],
                    "candidate_count": candidate_count_tensor[valid_tensor],
                    "valid_patient": valid_tensor.float(),
                    "skipped_patient": (~valid_tensor).float(),
                },
            },
        )
