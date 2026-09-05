from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn

from thesis.new_constraints.objective import NewConstraintConfig, NewConstraintObjective
from thesis.new_constraints.training_telemetry import (
    TELEMETRY_EPOCH_FIELDS,
    BoundaryTelemetryAccumulator,
    focal_gradient_magnitude,
    probe_component_gradients,
)


class _TinySegmentationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv3d(1, 4, kernel_size=3, padding=1)
        self.out = nn.Conv3d(4, 3, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.out(torch.relu(self.encoder(image)))


class _SimpleSupervisedLoss(nn.Module):
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if labels.ndim == 5:
            labels = labels[:, 0]
        return nn.functional.cross_entropy(logits, labels.long())


def _sample() -> dict[str, object]:
    image = torch.linspace(-1.0, 1.0, 8**3).reshape(1, 1, 8, 8, 8)
    label = torch.zeros((1, 1, 8, 8, 8), dtype=torch.long)
    label[:, :, 2:6, 2:6, 2:6] = 1
    label[:, :, 4:6, 3:5, 3:5] = 2
    return {"image": image, "label": label, "case_name": ["fixed_case"]}


def test_boundary_telemetry_reports_confusion_and_focus_mass() -> None:
    sample = _sample()
    labels = sample["label"]
    logits = torch.zeros((1, 3, 8, 8, 8))
    logits[:, 0] = 1.0
    accumulator = BoundaryTelemetryAccumulator(
        foreground_class_ids=(1, 2),
        complement_class_ids=(0,),
        band_steps=2,
        inner_gamma=1.0,
        outer_gamma=0.0,
        num_classes=3,
    )
    accumulator.update(logits, labels)
    row = accumulator.finalize(epoch=1)

    assert set(row) == set(TELEMETRY_EPOCH_FIELDS)
    assert row["voxel_count"] == 8**3
    assert row["pred_foreground_voxels"] == 0
    assert row["false_positive_voxels"] == 0
    assert row["false_negative_voxels"] == int((labels != 0).sum())
    assert 0 < row["inner_mean_focal_weight"] < 1
    assert row["outer_mean_focal_weight"] == 1.0
    assert 0 < row["inner_gradient_proxy_ess_fraction"] <= 1


def test_focal_gradient_gamma_zero_is_bce_gradient() -> None:
    truth_probability = torch.tensor([0.1, 0.5, 0.9])
    assert torch.allclose(
        focal_gradient_magnitude(truth_probability, 0.0),
        1.0 - truth_probability,
    )
    focal = focal_gradient_magnitude(truth_probability, 1.0)
    bce = focal_gradient_magnitude(truth_probability, 0.0)
    assert focal[0] > bce[0]
    assert torch.all(focal[1:] < bce[1:])


def test_tversky_telemetry_labels_class_components_and_focus() -> None:
    sample = _sample()
    accumulator = BoundaryTelemetryAccumulator(
        foreground_class_ids=(1, 2),
        complement_class_ids=(0,),
        band_steps=2,
        inner_gamma=0.0,
        outer_gamma=0.0,
        num_classes=3,
        loss_type="class_tversky",
        tversky_false_positive_weight=0.60,
        tversky_false_negative_weight=0.40,
    )
    accumulator.update(torch.zeros((1, 3, 8, 8, 8)), sample["label"])
    row = accumulator.finalize(epoch=1)

    assert set(row) == set(TELEMETRY_EPOCH_FIELDS)
    assert row["loss_type"] == "class_tversky"
    assert row["inner_component_label"] == "class_1_tversky"
    assert row["outer_component_label"] == "class_2_tversky"
    assert row["inner_mean_objective_weight"] > 0
    assert row["outer_gradient_proxy_top10_mass_fraction"] > 0


def test_gradient_probe_is_non_optimizing_and_restores_state() -> None:
    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    model = _TinySegmentationModel()
    model.encoder.use_checkpoint = True
    model.train()
    objective = NewConstraintObjective(
        NewConstraintConfig(
            equivariance_weight=0.0,
            bands_weight=0.01,
            bands_focal_gamma=0.0,
            bands_inner_focal_gamma=1.0,
            bands_outer_focal_gamma=0.0,
        )
    )
    before_parameters = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    before_torch_rng = torch.get_rng_state().clone()
    before_numpy_rng = np.random.get_state()
    before_python_rng = random.getstate()

    rows, maps = probe_component_gradients(
        model,
        [_sample()],
        objective,
        _SimpleSupervisedLoss(),
        torch.device("cpu"),
        epoch=5,
        constraint_scale=1.0,
        max_cases=1,
        spatial_cases=1,
    )

    assert model.training
    assert model.encoder.use_checkpoint is True
    assert torch.equal(torch.get_rng_state(), before_torch_rng)
    assert np.array_equal(np.random.get_state()[1], before_numpy_rng[1])
    assert random.getstate() == before_python_rng
    assert all(parameter.grad is None for parameter in model.parameters())
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before_parameters[name])
    all_row = next(row for row in rows if row["parameter_group"] == "all")
    assert all_row["supervised_gradient_norm"] > 0
    assert all_row["inner_gradient_norm"] > 0
    assert all_row["outer_gradient_norm"] > 0
    for key in (
        "supervised_inner_cosine",
        "supervised_outer_cosine",
        "inner_outer_cosine",
    ):
        assert -1 <= all_row[key] <= 1
    assert maps[0]["case_name"] == "fixed_case"
    assert maps[0]["inner_logit_gradient"].shape == (3, 8, 8, 8)
