"""Augmented calibration parity and transformed boundary supervision."""

import torch
from monai.data import MetaTensor

from thesis.new_constraints.bands import OuterBoundaryBandLoss, build_boundary_bands
from thesis.new_constraints.bands.calibrate_weight import prepare_calibration_view
from thesis.new_constraints.bands.outer_boundary import _cross_kernel, _edge_touching
from thesis.new_constraints.objective import NewConstraintConfig, NewConstraintObjective
from thesis.new_constraints.training_augmentation import augment_mild, warp_pair


def sample():
    labels = torch.zeros(1, 1, 32, 32, 32, dtype=torch.long)
    labels[:, :, 9:16, 10:23, 8:22] = 1
    labels[:, :, 16:23, 10:23, 8:22] = 2
    return (labels > 0).float(), labels


def test_morphology_kernel_is_reused_and_edge_checks_preserve_all_faces():
    device = torch.device('cpu')
    assert _cross_kernel(device).data_ptr() == _cross_kernel(device).data_ptr()
    masks = torch.zeros(8, 1, 5, 7, 9, dtype=torch.bool)
    for row, coordinate in enumerate(((0, 3, 4), (4, 3, 4), (2, 0, 4),
                                       (2, 6, 4), (2, 3, 0), (2, 3, 8))):
        masks[row, 0, *coordinate] = True
    masks[6, 0, 2, 3, 4] = True
    assert _edge_touching(masks).tolist() == [True] * 6 + [False, False]


def test_calibration_views_match_production_rng_and_strip_metadata():
    images, labels = sample()
    calibration_rng = torch.Generator().manual_seed(1)
    production_rng = torch.Generator().manual_seed(1)
    for _ in range(12):
        actual = prepare_calibration_view(
            MetaTensor(images), MetaTensor(labels), device=torch.device('cpu'),
            augmentation='mild_v1', generator=calibration_rng,
        )
        expected = augment_mild(images, labels, generator=production_rng)
        assert type(actual[0]) is torch.Tensor and type(actual[1]) is torch.Tensor
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])
        assert actual[2] == expected[2]
    assert torch.equal(calibration_rng.get_state(), production_rng.get_state())


def test_transformed_bands_use_current_labels_and_reuse_logits():
    images, labels = sample()
    images, transformed, applied = warp_pair(
        images, labels, angles=(0, 0, 0.17), scale=1,
        translation_xyz=(1, 0, 0),
    )
    assert applied and not torch.equal(transformed, labels)
    logits = torch.randn(1, 3, 32, 32, 32, requires_grad=True)

    class NoForward(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError('Bands must reuse the supervised forward pass')

    objective = NewConstraintObjective(NewConstraintConfig(equivariance_weight=0, bands_weight=0.1))
    result = objective(NoForward(), images, logits, transformed)
    expected = 0.1 * OuterBoundaryBandLoss()(logits, transformed).loss
    assert torch.allclose(result['loss'], expected)
    result['loss'].backward()
    inner, outer = build_boundary_bands(transformed > 0)
    active = (inner | outer).expand_as(logits)
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[~active]) == 0
    assert torch.count_nonzero(logits.grad[active]) > 0
