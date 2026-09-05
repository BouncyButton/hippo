"""Geometry, estimator and reproducibility checks for the translation follow-up."""
from itertools import combinations

import pytest
import torch

from thesis.new_constraints.equivariance import translate_3d
from thesis.new_constraints.teacher import DEFAULT_TEACHER_SHIFTS, TranslationTeacherKLLoss
from thesis.new_constraints.translation_augmentation import augment_translation
from thesis.new_constraints.teacher.calibrate_training import choose_weight, regional_pressure


def test_augmentation_preserves_image_label_alignment_and_global_rng():
    labels = torch.arange(8**3).reshape(1, 1, 8, 8, 8)
    images = labels.float().repeat(2, 1, 1, 1, 1)
    labels = labels.repeat(2, 1, 1, 1, 1)
    before = torch.get_rng_state().clone()
    seen = set()
    for batch in range(40):
        a, b, shifts = augment_translation(images, labels, seed=0, epoch=2, batch_index=batch)
        assert torch.equal(a, b.float())
        replay = augment_translation(images, labels, seed=0, epoch=2, batch_index=batch)
        assert torch.equal(a, replay[0]) and shifts == replay[2]
        seen.update(shifts)
    assert len(seen) == 7
    assert torch.equal(before, torch.get_rng_state())


def test_common_support_is_fixed_across_draws_and_excludes_all_padding():
    loss = TranslationTeacherKLLoss(support='common')
    base = torch.randn(1, 3, 8, 9, 10)
    expected = torch.zeros(1, 1, 8, 9, 10, dtype=torch.bool)
    expected[:, :, 2:-2, 2:-2, 2:-2] = True
    for shift in DEFAULT_TEACHER_SHIFTS:
        teacher = loss.build_from_cached([translate_3d(base, shift)], [shift])
        assert torch.equal(teacher.valid_mask, expected)
        assert torch.allclose(teacher.probabilities, base.softmax(1) * expected, atol=1e-7)
        assert torch.equal(teacher.view_counts, expected.float())
    with pytest.raises(ValueError, match='configured set'):
        loss.build_from_cached([base], [(3, 0, 0)])


def test_uniform_two_view_gradient_equals_full_teacher_in_expectation():
    # Exact enumeration, not a loose Monte Carlo approximation. KL VALUES need
    # not agree: the teacher entropy depends on which views are averaged.
    generator = torch.Generator().manual_seed(41)
    z = torch.randn(1, 3, 6, 6, 6, generator=generator).requires_grad_()
    views = [torch.randn(z.shape, generator=generator) for _ in DEFAULT_TEACHER_SHIFTS]
    loss = TranslationTeacherKLLoss(support='common')
    full = loss.build_from_cached(views, DEFAULT_TEACHER_SHIFTS)
    full_gradient, = torch.autograd.grad(loss.loss_from_teacher(z, full).loss, z)
    gradients = []
    for indices in combinations(range(12), 2):
        teacher = loss.build_from_cached([views[i] for i in indices],
                                        [DEFAULT_TEACHER_SHIFTS[i] for i in indices])
        gradient, = torch.autograd.grad(loss.loss_from_teacher(z, teacher).loss, z)
        gradients.append(gradient)
    assert torch.allclose(torch.stack(gradients).mean(0), full_gradient, atol=2e-8, rtol=1e-5)
    assert (torch.stack(gradients).var(0) > 0).any()  # Unbiased does not mean low variance.


def test_common_teacher_has_corrective_gradient_on_saturated_wrong_student():
    z = torch.zeros(1, 3, 6, 6, 6, dtype=torch.float16)
    z[:, 0] = 40
    z.requires_grad_()
    target = torch.zeros_like(z)
    target[:, 1] = 40
    loss = TranslationTeacherKLLoss(support='common')
    teacher = loss.build_from_cached([target], [DEFAULT_TEACHER_SHIFTS[0]])
    gradient, = torch.autograd.grad(loss.loss_from_teacher(z, teacher).loss, z)
    assert torch.isfinite(gradient).all()
    assert gradient[0, 0, 2, 2, 2] > 0 and gradient[0, 1, 2, 2, 2] < 0
    assert torch.count_nonzero(gradient[:, :, :2]) == 0


def test_training_only_scale_policy_respects_median_and_tail_caps():
    import numpy as np
    for ratios in ([1.0] * 32, [1.0] * 28 + [100.0] * 4):
        weight = choose_weight(ratios)
        assert weight * np.median(ratios) <= 0.1 + 1e-10
        assert weight * np.quantile(ratios, .95) <= 0.5 + 1e-10
    for ratios in ([], [0.0], [-1.0], [float('nan')], [float('inf')]):
        with pytest.raises(ValueError):
            choose_weight(ratios)


def test_error_regions_partition_gradient_energy():
    labels = torch.tensor([0, 1, 1, 2]).reshape(1, 1, 1, 1, 4)
    prediction = torch.tensor([1, 0, 2, 2]).reshape(1, 1, 1, 4)
    logits = torch.nn.functional.one_hot(prediction, 3).movedim(-1, 1).float()
    gradient = torch.ones_like(logits)
    result = regional_pressure(logits, labels, gradient, gradient)
    assert set(result) == {'false_positive', 'false_negative', 'ap_swap', 'correct'}
    assert sum(row['voxels'] for row in result.values()) == 4
    assert sum(row['gradient_energy_fraction'] for row in result.values()) == pytest.approx(1)
    assert all(row['supervised_cosine'] == pytest.approx(1) for row in result.values())
