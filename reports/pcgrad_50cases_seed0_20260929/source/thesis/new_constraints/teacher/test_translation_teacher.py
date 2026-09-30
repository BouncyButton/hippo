"""CPU checks of teacher geometry, stop-gradient behavior, and RNG isolation."""

import pytest
import torch
from torch import nn

from thesis.new_constraints.equivariance import translate_3d, translation_valid_mask
from thesis.new_constraints.teacher import (
    DEFAULT_TEACHER_SHIFTS,
    TranslationTeacherKLLoss,
)


class RecordingPointwiseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv3d(1, 3, 1)
        self.outputs = []

    def forward(self, images):
        logits = self.projection(images)
        self.outputs.append(logits)
        return logits


class StatefulModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv3d(1, 3, 1)
        self.batch_norm = nn.BatchNorm3d(3)
        self.dropout = nn.Dropout3d(0.5)
        self.register_buffer("calls", torch.tensor(0))
        self.raise_error = False

    def forward(self, images):
        self.calls.add_(1)
        # Simulate a custom evaluation-time stochastic layer as well as dropout.
        noise = torch.rand(())
        if self.raise_error:
            raise RuntimeError("intentional failure")
        return self.dropout(self.batch_norm(self.projection(images))) + noise


def test_default_shifts_are_twelve_nonzero_exact_translations():
    assert len(DEFAULT_TEACHER_SHIFTS) == 12
    assert len(set(DEFAULT_TEACHER_SHIFTS)) == 12
    for shift in DEFAULT_TEACHER_SHIFTS:
        assert sum(amount != 0 for amount in shift) == 1
        assert sum(abs(amount) for amount in shift) in (1, 2)


def test_cached_logits_inverse_map_and_normalize_by_valid_view_count():
    torch.manual_seed(7)
    logits = torch.randn(2, 3, 5, 6, 7)
    shifts = ((1, -2, 0), (-2, 0, 1))
    loss = TranslationTeacherKLLoss(temperature=2.0)
    cached = [translate_3d(logits, shift) for shift in shifts]
    teacher = loss.build_from_cached(cached, shifts)
    expected_counts = sum(
        translation_valid_mask(logits.shape[-3:], shift) for shift in shifts
    ).expand(2, -1, -1, -1, -1)
    expected = torch.softmax(logits / 2.0, dim=1) * (expected_counts > 0)
    assert torch.equal(teacher.view_counts, expected_counts)
    assert torch.allclose(teacher.probabilities, expected, atol=1e-7)
    assert torch.allclose(
        teacher.probabilities.sum(dim=1, keepdim=True), teacher.valid_mask.float()
    )
    assert (expected_counts == 0).any()
    assert (expected_counts == 1).any()
    assert (expected_counts == 2).any()


def test_each_covered_voxel_uses_only_available_views():
    loss = TranslationTeacherKLLoss()
    first = torch.zeros(1, 3, 3, 4, 5)
    first[:, 0] = 1.0
    second = torch.zeros_like(first)
    second[:, 2] = 1.0
    first_mask = translation_valid_mask(first.shape[-3:], (1, 0, 0))
    second_mask = translation_valid_mask(first.shape[-3:], (-1, 0, 0))
    teacher = loss.build_from_aligned([first, second], [first_mask, second_mask])
    assert torch.equal(teacher.probabilities[:, :, 0], first[:, :, 0])
    assert torch.equal(teacher.probabilities[:, :, 2], second[:, :, 2])
    assert torch.equal(teacher.probabilities[:, :, 1], 0.5 * (first + second)[:, :, 1])


def test_pointwise_equivariant_model_has_zero_loss_and_teacher_has_no_graph():
    torch.manual_seed(9)
    model = RecordingPointwiseModel()
    images = torch.randn(2, 1, 5, 6, 7)
    logits = model(images)
    logits.retain_grad()
    constraint = TranslationTeacherKLLoss()
    result = constraint(model, images, logits, shifts=((1, 0, 0), (0, -2, 0)))
    assert torch.allclose(result.loss, torch.tensor(0.0), atol=1e-7)
    assert torch.allclose(result.truth, torch.ones(2), atol=1e-7)
    assert model.outputs[0].requires_grad
    assert all(not output.requires_grad and output.grad_fn is None for output in model.outputs[1:])
    result.loss.backward()
    assert logits.grad is not None
    assert model.projection.weight.grad is not None
    assert torch.isfinite(model.projection.weight.grad).all()


def test_cached_teacher_detaches_even_when_cached_logits_require_grad():
    constraint = TranslationTeacherKLLoss()
    cached = torch.randn(1, 3, 4, 5, 6, requires_grad=True)
    teacher = constraint.build_from_cached([cached], [(1, 0, 0)])
    student = torch.randn_like(cached, requires_grad=True)
    constraint.loss_from_teacher(student, teacher).loss.backward()
    assert not teacher.probabilities.requires_grad
    assert cached.grad is None
    assert student.grad is not None
    assert student.grad.abs().sum() > 0
    assert student.grad[:, :, -1].abs().sum() == 0


def test_saturated_wrong_logits_receive_finite_corrective_gradient():
    constraint = TranslationTeacherKLLoss()
    target = torch.zeros(1, 3, 2, 3, 4)
    target[:, 1] = 1.0
    mask = torch.ones(1, 1, 2, 3, 4)
    teacher = constraint.build_from_aligned([target], [mask])
    student = torch.full_like(target, -1000.0)
    student[:, 0] = 1000.0
    student.requires_grad_()
    result = constraint.loss_from_teacher(student, teacher)
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert torch.allclose(result.loss, torch.tensor(2000.0))
    assert torch.isfinite(student.grad).all()
    assert (student.grad[:, 0] > 0).all()
    assert (student.grad[:, 1] < 0).all()
    assert torch.allclose(student.grad[:, 1], torch.full_like(student.grad[:, 1], -1 / 24))


def test_temperature_compensation_has_expected_logit_gradient():
    constraint = TranslationTeacherKLLoss(temperature=2.5)
    target = torch.tensor([0.2, 0.7, 0.1]).reshape(1, 3, 1, 1, 1).expand(1, 3, 2, 2, 2)
    teacher = constraint.build_from_aligned([target], [torch.ones(1, 1, 2, 2, 2)])
    student = torch.tensor([0.4, -0.8, 0.2]).reshape(1, 3, 1, 1, 1).expand_as(target).clone().requires_grad_()
    result = constraint.loss_from_teacher(student, teacher)
    result.loss.backward()
    expected = 2.5 * (torch.softmax(student.detach() / 2.5, dim=1) - target) / 8
    assert torch.allclose(student.grad, expected, atol=1e-7)
    assert torch.allclose(result.value, result.details["kl_unscaled"] * 2.5**2)
    assert torch.allclose(result.truth, torch.exp(-result.value))


def test_cases_are_averaged_equally_despite_different_valid_support():
    constraint = TranslationTeacherKLLoss()
    target = torch.zeros(2, 3, 2, 2, 2)
    target[:, 1] = 1.0
    mask = torch.ones(2, 1, 2, 2, 2)
    mask[1] = 0
    mask[1, :, 0, 0, 0] = 1
    teacher = constraint.build_from_aligned([target], [mask])
    student = torch.zeros_like(target)
    student[1, 1] = 2.0
    student.requires_grad_()
    result = constraint.loss_from_teacher(student, teacher)
    expected = -torch.log_softmax(student[:, :, 0, 0, 0], dim=1)[:, 1]
    assert torch.equal(result.details["valid_voxels"], torch.tensor([8.0, 1.0]))
    assert torch.allclose(result.value, expected)
    assert torch.allclose(result.loss, expected.mean())
    result.loss.backward()
    assert student.grad[1, :, 1].abs().sum() == 0


def test_sampling_is_distinct_reproducible_and_does_not_consume_global_rng():
    constraint = TranslationTeacherKLLoss()
    first = torch.Generator().manual_seed(81)
    second = torch.Generator().manual_seed(81)
    global_before = torch.random.get_rng_state().clone()
    sequence = [constraint.select_shifts(generator=first) for _ in range(12)]
    repeated = [constraint.select_shifts(generator=second) for _ in range(12)]
    assert sequence == repeated
    assert all(len(set(shifts)) == constraint.num_views for shifts in sequence)
    assert len(set(sequence)) > 1
    assert torch.equal(torch.random.get_rng_state(), global_before)
    before = first.get_state().clone()
    assert constraint.select_shifts(shifts=DEFAULT_TEACHER_SHIFTS, generator=first) == DEFAULT_TEACHER_SHIFTS
    assert torch.equal(first.get_state(), before)
    with pytest.raises(ValueError, match="independent CPU generator"):
        constraint.select_shifts()


@pytest.mark.parametrize("raises", [False, True])
def test_teacher_restores_every_module_flag_buffers_and_rng_even_on_exception(raises):
    torch.manual_seed(15)
    model = StatefulModel()
    model.train()
    model.batch_norm.eval()  # Preserve mixed modes, not only the root flag.
    images = torch.randn(2, 1, 4, 5, 6)
    flags = [module.training for module in model.modules()]
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    rng_before = torch.random.get_rng_state().clone()
    model.raise_error = raises
    constraint = TranslationTeacherKLLoss()
    if raises:
        with pytest.raises(RuntimeError, match="intentional failure"):
            constraint.build_teacher(model, images, shifts=((1, 0, 0), (0, -1, 0)))
    else:
        first = constraint.build_teacher(model, images, shifts=((1, 0, 0), (0, -1, 0)))
        second = constraint.build_teacher(model, images, shifts=((1, 0, 0), (0, -1, 0)))
        assert torch.equal(first.probabilities, second.probabilities)
    assert [module.training for module in model.modules()] == flags
    assert all(torch.equal(value, buffers[name]) for name, value in model.named_buffers())
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_batch_norm_student_backward_remains_valid_after_teacher():
    torch.manual_seed(5)
    model = nn.Sequential(nn.Conv3d(1, 3, 1), nn.BatchNorm3d(3), nn.Dropout3d(0.2))
    model.train()
    images = torch.randn(2, 1, 4, 5, 6)
    student = model(images)
    expected_buffers = {name: value.clone() for name, value in model.named_buffers()}
    constraint = TranslationTeacherKLLoss()
    result = constraint(model, images, student, shifts=((1, 0, 0), (-1, 0, 0)))
    result.loss.backward()
    assert torch.isfinite(model[0].weight.grad).all()
    assert all(torch.equal(value, expected_buffers[name]) for name, value in model.named_buffers())


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_invalid_temperature_is_rejected(temperature):
    with pytest.raises(ValueError, match="temperature"):
        TranslationTeacherKLLoss(temperature=temperature)


@pytest.mark.parametrize("kwargs", [
    {"num_views": 0}, {"num_views": 13}, {"num_views": 1.5},
    {"shifts": [(0, 0, 0)]}, {"shifts": [(1, 0, 0), (1, 0, 0)]},
    {"shifts": [(0.5, 0, 0)]},
])
def test_invalid_shift_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        TranslationTeacherKLLoss(**kwargs)


def test_teacher_can_agree_confidently_on_the_wrong_class():
    # A concrete counterexample to any claim that the teacher cannot saturate.
    constraint = TranslationTeacherKLLoss()
    wrong_logits = torch.full((1, 3, 4, 5, 6), -1000.0)
    wrong_logits[:, 0] = 1000.0
    teacher = constraint.build_from_cached([wrong_logits], [(1, 0, 0)])
    result = constraint.loss_from_teacher(wrong_logits, teacher)
    assert result.loss == 0
    assert torch.equal(result.truth, torch.ones(1))


def test_float16_student_is_evaluated_in_float32():
    constraint = TranslationTeacherKLLoss()
    target = torch.zeros(1, 3, 2, 2, 2)
    target[:, 1] = 1
    teacher = constraint.build_from_aligned([target], [torch.ones(1, 1, 2, 2, 2)])
    student = torch.full(target.shape, -100.0, dtype=torch.float16)
    student[:, 0] = 100
    student.requires_grad_()
    result = constraint.loss_from_teacher(student, teacher)
    result.loss.backward()
    assert result.loss.dtype == torch.float32
    assert result.loss == 200
    assert torch.isfinite(student.grad).all()
