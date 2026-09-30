import numpy as np
import pytest
import torch

from .interval_constraint import UncalIntervalLogLTNLoss
from .interval_teacher import FoldIntervalTeacher, calibration_radius, label_transition_interval


def inputs(batch=1, dtype=torch.float32):
    logits = torch.zeros(batch, 3, 2, 12, 2, dtype=dtype, requires_grad=True)
    coords = torch.arange(12, dtype=dtype)[None, None, :, None].expand(batch, 2, 12, 2).clone()
    support = torch.ones_like(coords, dtype=torch.bool)
    interval = torch.tensor([[4.5, 6.5]], dtype=dtype).expand(batch, 2).clone()
    return logits, dict(support=support, anterior_coordinate_mm=coords, interval_mm=interval)


def test_reference_interval_retains_mixed_slices_without_contradictions():
    labels = np.full((2, 12, 2), 2)
    labels[:, 6:] = 1
    labels[0, 5, 0] = 1
    labels[1, 6, 1] = 2
    assert label_transition_interval(labels) == (4.5, 6.5)
    logits, args = inputs()
    logits = torch.tensor(np.stack([labels == c for c in (0, 1, 2)])[None], dtype=torch.float32) * 40
    result = UncalIntervalLogLTNLoss()(logits, **args)
    assert result.loss < 1e-12


def test_zero_background_outside_support_and_uncertain_interval_gradients():
    logits, args = inputs()
    args["support"][:, 0] = False
    result = UncalIntervalLogLTNLoss()(logits, **args)
    result.loss.backward()
    assert torch.count_nonzero(logits.grad[:, 0]) == 0
    assert torch.count_nonzero(logits.grad[:, :, 0]) == 0
    assert torch.count_nonzero(logits.grad[:, :, :, 5:7]) == 0
    assert logits.grad[0, 1, 1, 8, 1] < 0
    assert logits.grad[0, 2, 1, 3, 1] < 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_saturated_wrong_logits_have_corrective_gradient(dtype):
    logits, args = inputs(dtype=dtype)
    with torch.no_grad():
        logits[:, 1, :, :5] = 100
        logits[:, 2, :, 7:] = 100
    loss = UncalIntervalLogLTNLoss()(logits, **args).loss
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad[0, 1, 0, 8, 0] < 0
    assert logits.grad[0, 2, 0, 3, 0] < 0


def test_groundings_detached_and_reliability_scales_loss():
    logits, args = inputs()
    args["anterior_coordinate_mm"].requires_grad_()
    args["interval_mm"].requires_grad_()
    reliability = torch.tensor([.25], requires_grad=True)
    full = UncalIntervalLogLTNLoss()(logits, **args).loss
    quarter = UncalIntervalLogLTNLoss()(logits, **args, reliability=reliability).loss
    assert torch.allclose(quarter, full*.25)
    quarter.backward()
    assert reliability.grad is None and args["interval_mm"].grad is None
    assert args["anterior_coordinate_mm"].grad is None


def test_no_evidence_is_not_reported_as_perfect_anatomy():
    logits, args = inputs()
    args["interval_mm"][:] = torch.tensor([-100., 100.])
    result = UncalIntervalLogLTNLoss()(logits, **args)
    assert result.loss == 0 and not result.details["valid"].any()
    assert torch.isnan(result.truth).all()
    result.loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_coordinate_frame_translation_flip_and_permutation_invariance():
    logits, args = inputs()
    logits = torch.randn_like(logits)
    loss = UncalIntervalLogLTNLoss()
    reference = loss(logits, **args).loss
    translated = {**args, "anterior_coordinate_mm": args["anterior_coordinate_mm"]+20,
                  "interval_mm": args["interval_mm"]+20}
    assert torch.allclose(reference, loss(logits, **translated).loss)
    flipped = {**args, "support": args["support"].flip(2), "anterior_coordinate_mm": args["anterior_coordinate_mm"].flip(2)}
    assert torch.allclose(reference, loss(logits.flip(3), **flipped).loss)
    permuted = {**args, "support": args["support"].permute(0, 2, 1, 3),
                "anterior_coordinate_mm": args["anterior_coordinate_mm"].permute(0, 2, 1, 3)}
    assert torch.allclose(reference, loss(logits.permute(0, 1, 3, 2, 4), **permuted).loss)


def test_padding_and_inactive_batch_member_do_not_dilute_loss():
    logits, args = inputs()
    objective = UncalIntervalLogLTNLoss()
    expected = objective(logits, **args).loss
    other, batch_args = inputs(batch=2)
    batch_args["support"][1] = False
    assert torch.allclose(expected, objective(other, **batch_args).loss)
    pad = (2, 2, 2, 2, 2, 2)
    padded_args = {**args, "support": torch.nn.functional.pad(args["support"], pad),
                   "anterior_coordinate_mm": torch.nn.functional.pad(args["anterior_coordinate_mm"], pad)}
    assert torch.allclose(expected, objective(torch.nn.functional.pad(logits, pad), **padded_args).loss)


def test_finite_sample_calibration_quantile_is_not_interpolated():
    centres = np.zeros(9)
    intervals = np.column_stack((np.arange(9), np.arange(9)))
    assert calibration_radius(centres, intervals, alpha=.2) == 7
    with pytest.raises(ValueError):
        calibration_radius(centres, intervals, alpha=.01)


def test_invalid_grounding_rejected():
    logits, args = inputs()
    args["interval_mm"][:] = torch.tensor([7., 4.])
    with pytest.raises(ValueError):
        UncalIntervalLogLTNLoss()(logits, **args)


def test_label_free_native_grounding_and_affine_rejection(monkeypatch):
    teacher = FoldIntervalTeacher()
    monkeypatch.setattr(teacher, "predict_cut", lambda case: 3)
    image = np.ones((6, 8, 6))
    support = np.zeros_like(image, dtype=bool)
    support[1:5, 1:7, 1:5] = True
    anchors = teacher.ground_native(image, support, np.eye(4), radius_mm=2)
    np.testing.assert_array_equal(anchors["support"], support)
    np.testing.assert_array_equal(anchors["interval_mm"], [.5, 4.5])
    np.testing.assert_array_equal(anchors["anterior_coordinate_mm"][0, :, 0], np.arange(8))
    affine = np.eye(4)
    affine[1, 1] = -1
    with pytest.raises(ValueError):
        teacher.ground_native(image, support, affine, radius_mm=2)
