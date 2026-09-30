"""Geometry falsifiers and gradient behavior for the structured A/P objective."""

import pytest
import torch
import torch.nn.functional as F

from .posterior import APCutConfig, APCutPosteriorLoss, compute_ap_cut_posterior


def _example(axis=1, anterior_low=False, predicted_cut=2, magnitude=6.0):
    labels = torch.zeros((1, 3, 7, 3), dtype=torch.long)
    low, high = (1, 2) if anterior_low else (2, 1)
    labels[:, 1, 1:3, 1] = low
    labels[:, 1, 3:6, 1] = high
    logits = torch.zeros((1, 3, 3, 7, 3))
    for j in range(1, 6):
        winner = low if j <= predicted_cut else high
        logits[:, winner, 1, j, 1] = magnitude
        logits[:, 3 - winner, 1, j, 1] = -magnitude
    return logits.movedim(3, axis + 2), labels.movedim(2, axis + 1)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("anterior_low", [True, False])
def test_explicit_geometry_and_complete_candidate_interval(axis, anterior_low):
    logits, labels = _example(axis, anterior_low)
    config = APCutConfig(axis=axis, anterior_low=anterior_low)
    posterior = compute_ap_cut_posterior(logits, labels, config)[0]
    assert posterior.valid
    assert posterior.cut_indices.tolist() == [1, 2, 3, 4]
    assert posterior.gt_cut == 2
    assert posterior.gt_index == 1
    assert int(posterior.cut_indices[posterior.log_probs.argmax()]) == 2
    assert torch.allclose(posterior.log_probs.exp().sum(), torch.tensor(1.0))


def test_saturated_wrong_cut_has_corrective_gradient_and_can_change_argmax():
    logits, labels = _example(predicted_cut=3, magnitude=40.0)
    logits.requires_grad_()
    loss = APCutPosteriorLoss(axis=1, anterior_low=False)
    before = loss(logits, labels)
    gradient = torch.autograd.grad(before.loss, logits)[0]
    assert torch.isfinite(before.loss)
    assert torch.isfinite(gradient).all()
    # At slice 3, class 2 is confidently wrong and class 1 is the annotation.
    assert gradient[0, 2, 1, 3, 1] > 0.1
    assert gradient[0, 1, 1, 3, 1] < -0.1
    # This deliberately large logit-space step is a reachability counterexample,
    # not a proposed network learning rate or evidence of generalization.
    updated = logits.detach() - 250.0 * gradient
    after = loss(updated, labels)
    assert updated.argmax(dim=1)[0, 1, 3, 1] == 1
    assert after.loss < before.loss
    assert after.details["metrics"]["exact_cut"].item() == 1


def test_no_background_channel_or_outside_gt_support_gradient():
    logits, labels = _example(predicted_cut=3)
    logits.requires_grad_()
    result = APCutPosteriorLoss(axis=1, anterior_low=False)(logits, labels)
    gradient = torch.autograd.grad(result.loss, logits)[0]
    assert gradient[:, 0].count_nonzero() == 0
    assert gradient[:, 1:3].masked_select((labels == 0).unsqueeze(1)).count_nonzero() == 0
    altered = logits.detach().clone()
    altered[:, 0] = 1000.0
    altered[:, 1:3] += torch.randn_like(altered[:, :1]) * 100.0
    assert torch.allclose(
        result.loss, APCutPosteriorLoss(axis=1, anterior_low=False)(altered, labels).loss,
        atol=1e-5,
    )


def test_slice_area_and_background_padding_do_not_change_posterior():
    logits, labels = _example(predicted_cut=3)
    config = APCutConfig(axis=1, anterior_low=False)
    original = compute_ap_cut_posterior(logits, labels, config)[0]
    # Replicate all support within the transverse dimensions; score is per slice.
    larger_logits = logits.repeat_interleave(3, dim=2).repeat_interleave(2, dim=4)
    larger_labels = labels.repeat_interleave(3, dim=1).repeat_interleave(2, dim=3)
    expanded = compute_ap_cut_posterior(larger_logits, larger_labels, config)[0]
    padded = compute_ap_cut_posterior(
        F.pad(logits, (1, 2, 3, 4, 1, 1)), F.pad(labels, (1, 2, 3, 4, 1, 1)), config
    )[0]
    assert torch.allclose(original.log_probs, expanded.log_probs)
    assert torch.allclose(original.log_probs, padded.log_probs)
    assert padded.gt_cut == original.gt_cut + 3


def test_temperature_changes_confidence_without_changing_candidate_ranking():
    logits, labels = _example(predicted_cut=3)
    cold = compute_ap_cut_posterior(logits, labels, APCutConfig(1, False, 0.5))[0]
    hot = compute_ap_cut_posterior(logits, labels, APCutConfig(1, False, 2.0))[0]
    assert cold.log_probs.argmax() == hot.log_probs.argmax()
    assert cold.log_probs.exp().max() > hot.log_probs.exp().max()


@pytest.mark.parametrize("kind", ["mixed", "orientation", "multiple", "missing", "gap"])
def test_invalid_geometry_is_rejected_and_skip_reason_is_explicit(kind):
    logits, labels = _example()
    if kind == "mixed":
        labels[0, 0, 2, 1] = 1
    elif kind == "orientation":
        labels = torch.where(labels == 1, 2, torch.where(labels == 2, 1, 0))
    elif kind == "multiple":
        labels[0, 1, 5, 1] = 2
    elif kind == "missing":
        labels[labels == 2] = 0
    else:
        labels[:, :, 2, :] = 0
    with pytest.raises(ValueError, match="Invalid A/P geometry in batch case 0"):
        APCutPosteriorLoss(axis=1, anterior_low=False)(logits, labels)
    skipped = compute_ap_cut_posterior(logits, labels, APCutConfig(1, False, 1.0, "skip"))[0]
    assert not skipped.valid
    assert skipped.reason
    assert skipped.log_probs.numel() == 0


def test_all_skipped_batch_has_finite_zero_and_backward():
    logits, labels = _example()
    logits.requires_grad_()
    labels.zero_()
    result = APCutPosteriorLoss(axis=1, anterior_low=False, invalid_policy="skip")(logits, labels)
    result.loss.backward()
    assert result.loss.item() == 0.0
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.count_nonzero() == 0
    assert result.truth.numel() == 0
    assert result.details["metrics"]["skipped_patient"].tolist() == [1.0]


def test_case_average_excludes_skipped_cases_and_preserves_label_detachment():
    logits, labels = _example(predicted_cut=3)
    objective = APCutPosteriorLoss(axis=1, anterior_low=False, invalid_policy="skip")
    single = objective(logits, labels).loss
    combined_logits = logits.repeat(3, 1, 1, 1, 1).requires_grad_()
    combined_labels = labels.repeat(3, 1, 1, 1).float()
    combined_labels[2].zero_()
    combined_labels.requires_grad_()
    result = objective(combined_logits, combined_labels.unsqueeze(1))
    assert torch.allclose(single, result.loss)
    result.loss.backward()
    assert combined_labels.grad is None
    assert result.details["valid"].tolist() == [True, True, False]


def test_fp16_extreme_logits_use_finite_fp32_log_probabilities():
    logits, labels = _example(predicted_cut=3, magnitude=1000.0)
    logits = logits.half().requires_grad_()
    result = APCutPosteriorLoss(axis=1, anterior_low=False)(logits, labels)
    result.loss.backward()
    assert result.loss.dtype == torch.float32
    assert torch.isfinite(result.loss)
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().max() > 0.1


@pytest.mark.parametrize("config", [
    {"axis": 3, "anterior_low": False},
    {"axis": 1, "anterior_low": "high"},
    {"axis": 1, "anterior_low": False, "temperature": 0.0},
    {"axis": 1, "anterior_low": False, "temperature": float("nan")},
    {"axis": 1, "anterior_low": False, "invalid_policy": "ignore"},
])
def test_invalid_configuration_fails(config):
    with pytest.raises(ValueError):
        APCutPosteriorLoss(**config)
