"""Numerical, gradient, and pipeline tests for translation equivariance."""

from types import SimpleNamespace

import torch
import torch.nn as nn
import thesis.new_constraints.train_swinunetr_constraints as trainer

from thesis.new_constraints import (
    NewConstraintConfig,
    NewConstraintObjective,
    TranslationEquivarianceLoss,
    restore_translation,
    translate_3d,
    translation_valid_mask,
)
from thesis.new_constraints.constraint_result import ConstraintResult
from thesis.new_constraints.train_swinunetr_constraints import (
    RunSpec,
    averaged_constraint_metrics,
    build_experiment_generators,
    build_swinunetr,
    constraint_warmup_scale,
    evaluate_constraint_metrics,
    evaluate_segmentation_metrics,
    patient_id_from_case,
    resolve_constraint_config,
    resolve_resume_run_spec,
    restore_rng_state,
    rng_state,
    save_checkpoint,
    update_constraint_totals,
)
from thesis.new_constraints.equivariance.translation_equivariance import (
    _classwise_soft_dice,
)


class PointwiseSegmentationModel(nn.Module):
    """A translation-equivariant model used to test the loss contract."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv3d(1, 3, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images)


class PositionBiasedModel(nn.Module):
    """A deliberately non-equivariant model with a trainable position prior."""

    def __init__(self, spatial_size: int) -> None:
        super().__init__()
        coordinate = torch.linspace(-1.0, 1.0, spatial_size)
        prior = coordinate[:, None, None].expand(
            spatial_size,
            spatial_size,
            spatial_size,
        )
        self.register_buffer("prior", prior[None, None])
        self.position_scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        foreground = images + self.position_scale * self.prior
        return torch.cat((-foreground, foreground, -0.5 * foreground), dim=1)


class RecordingPositionBiasedModel(PositionBiasedModel):
    def __init__(self, spatial_size: int) -> None:
        super().__init__(spatial_size)
        self.outputs: list[torch.Tensor] = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = super().forward(images)
        output.retain_grad()
        self.outputs.append(output)
        return output


class FixedLogitModel(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[0] != self.logits.shape[0]:
            raise ValueError("Test model requires the configured batch size.")
        return self.logits


def _assert_value_error(function, expected_text: str) -> None:
    try:
        function()
    except ValueError as error:
        assert expected_text in str(error)
    else:
        raise AssertionError("Expected ValueError was not raised.")


def test_real_swinunetr_constructor_matches_installed_monai() -> None:
    model = build_swinunetr((64, 64, 64), 3, torch.device("cpu"))
    assert model.out.conv.out_channels == 3


def test_swinunetr_constructor_supports_both_monai_signatures() -> None:
    calls: list[dict[str, object]] = []

    class LegacySwinUNETR:
        def __init__(self, img_size, in_channels, out_channels, use_checkpoint):
            calls.append(locals())

        def to(self, device):
            return self

    class ModernSwinUNETR:
        def __init__(self, in_channels, out_channels, use_checkpoint):
            calls.append(locals())

        def to(self, device):
            return self

    installed_constructor = trainer.SwinUNETR
    try:
        trainer.SwinUNETR = LegacySwinUNETR
        trainer.build_swinunetr((64, 64, 64), 3, torch.device("cpu"))
        assert calls[-1]["img_size"] == (64, 64, 64)

        trainer.SwinUNETR = ModernSwinUNETR
        trainer.build_swinunetr((64, 64, 64), 3, torch.device("cpu"))
        assert "img_size" not in calls[-1]
    finally:
        trainer.SwinUNETR = installed_constructor


def test_translate_restore_and_valid_mask() -> None:
    tensor = torch.arange(5 * 6 * 7).reshape(1, 1, 5, 6, 7).float()
    shift = (1, -2, 0)
    restored = restore_translation(translate_3d(tensor, shift), shift)
    valid = translation_valid_mask(tensor.shape[-3:], shift)

    assert torch.equal(restored * valid, tensor * valid)
    expected_fraction = torch.tensor((4 * 4 * 7) / (5 * 6 * 7))
    assert torch.isclose(valid.mean(), expected_fraction)


def test_all_six_signed_axis_translations() -> None:
    tensor = torch.zeros(1, 1, 7, 8, 9)
    origin = (3, 3, 4)
    tensor[(0, 0, *origin)] = 1.0
    shifts = (
        (2, 0, 0),
        (-2, 0, 0),
        (0, 2, 0),
        (0, -2, 0),
        (0, 0, 2),
        (0, 0, -2),
    )

    for shift in shifts:
        translated = translate_3d(tensor, shift)
        destination = tuple(index + amount for index, amount in zip(origin, shift))
        assert translated[(0, 0, *destination)].item() == 1.0
        assert translated.sum().item() == 1.0
        valid = translation_valid_mask(tensor.shape[-3:], shift)
        restored = restore_translation(translated, shift)
        assert torch.equal(restored * valid, tensor * valid)
        expected_fraction = 1.0
        for length, amount in zip(tensor.shape[-3:], shift):
            expected_fraction *= (length - abs(amount)) / length
        assert torch.isclose(valid.mean(), torch.tensor(expected_fraction))


def test_identity_optimization_satisfaction_is_confidence_invariant() -> None:
    valid = torch.ones(1, 1, 3, 4, 5)
    for confidence in (0.05, 0.50, 0.95):
        probabilities = torch.full_like(valid, confidence)
        optimization = _classwise_soft_dice(
            probabilities,
            probabilities,
            valid,
            (0,),
            1e-6,
            squared_denominator=True,
        )
        legacy_linear = _classwise_soft_dice(
            probabilities,
            probabilities,
            valid,
            (0,),
            1e-6,
            squared_denominator=False,
        )
        assert torch.allclose(optimization, torch.ones_like(optimization))
        assert torch.allclose(
            legacy_linear,
            torch.full_like(legacy_linear, confidence),
        )


def test_pointwise_model_is_translation_equivariant() -> None:
    torch.manual_seed(3)
    images = torch.randn(2, 1, 8, 9, 10)
    model = PointwiseSegmentationModel()
    constraint = TranslationEquivarianceLoss()

    result = constraint(model, images, shift=(2, 0, 0))

    assert torch.allclose(result.truth, torch.ones_like(result.truth), atol=1e-5)
    assert torch.allclose(
        result.details["confidence_weighted_agreement"],
        result.value.mean(dim=1),
    )
    assert torch.all(result.details["confidence_weighted_agreement"] <= result.truth)


def test_explicit_zero_shift_is_not_randomized() -> None:
    torch.manual_seed(31)
    images = torch.randn(1, 1, 8, 8, 8)
    model = PositionBiasedModel(spatial_size=8)
    constraint = TranslationEquivarianceLoss()

    result = constraint(model, images, shift=(0, 0, 0))

    assert result.details["shift"] == (0, 0, 0)
    assert torch.allclose(result.truth, torch.ones_like(result.truth), atol=1e-5)


def test_translation_loss_backpropagates_through_model() -> None:
    torch.manual_seed(4)
    images = torch.randn(2, 1, 8, 8, 8)
    model = PositionBiasedModel(spatial_size=8)
    constraint = TranslationEquivarianceLoss()

    result = constraint(model, images, shift=(2, 0, 0))
    result.loss.backward()

    assert result.truth.mean() < 0.99
    assert model.position_scale.grad is not None
    assert model.position_scale.grad.abs() > 0


def test_gradients_reach_original_and_translated_prediction_branches() -> None:
    torch.manual_seed(41)
    images = torch.randn(1, 1, 8, 8, 8)
    model = RecordingPositionBiasedModel(spatial_size=8)
    constraint = TranslationEquivarianceLoss()
    base_logits = model(images)

    result = constraint(model, images, base_logits, shift=(-2, 0, 0))
    result.loss.backward()

    assert len(model.outputs) == 2
    assert model.outputs[0].grad is not None
    assert model.outputs[1].grad is not None
    assert model.outputs[0].grad.abs().sum() > 0
    assert model.outputs[1].grad.abs().sum() > 0


def test_objective_returns_trainable_scalar() -> None:
    torch.manual_seed(5)
    images = torch.randn(1, 1, 12, 12, 12)
    model = PositionBiasedModel(spatial_size=12)
    objective = NewConstraintObjective(
        NewConstraintConfig(equivariance_weight=0.1)
    )
    logits = model(images)

    output = objective(model, images, logits, shift=(2, 0, 0))
    output["loss"].backward()

    assert output["loss"].ndim == 0
    assert set(output["results"]) == {"translation_equivariance"}
    assert model.position_scale.grad is not None
    assert torch.isfinite(model.position_scale.grad)


def test_weight_and_max_samples_apply_to_the_actual_batch_loss() -> None:
    torch.manual_seed(51)
    images = torch.randn(3, 1, 8, 8, 8)
    model = PositionBiasedModel(spatial_size=8)
    objective = NewConstraintObjective(
        NewConstraintConfig(
            equivariance_weight=0.25,
            equivariance_max_samples=1,
        )
    )
    logits = model(images)

    output = objective(model, images, logits, shift=(2, 0, 0))
    result = output["results"]["translation_equivariance"]

    assert result.truth.shape == (1,)
    assert torch.allclose(output["loss"], 0.25 * result.loss)


def test_constraint_warmup_scales_the_weighted_loss() -> None:
    assert constraint_warmup_scale(1, 5) == 0.2
    assert constraint_warmup_scale(5, 5) == 1.0
    assert constraint_warmup_scale(6, 5) == 1.0
    assert constraint_warmup_scale(1, 0) == 1.0
    _assert_value_error(lambda: constraint_warmup_scale(0, 5), "positive")


def test_objective_rejects_invalid_configuration() -> None:
    _assert_value_error(
        lambda: NewConstraintObjective(
            NewConstraintConfig(equivariance_weight=-0.1)
        ),
        "non-negative",
    )
    for invalid_weight in (float("nan"), float("inf"), -float("inf")):
        _assert_value_error(
            lambda invalid_weight=invalid_weight: NewConstraintObjective(
                NewConstraintConfig(equivariance_weight=invalid_weight)
            ),
            "finite",
        )
    _assert_value_error(
        lambda: TranslationEquivarianceLoss(confidence_agreement_threshold=1.1),
        "between zero and one",
    )
    for invalid_epsilon in (float("nan"), float("inf"), -float("inf")):
        _assert_value_error(
            lambda invalid_epsilon=invalid_epsilon: TranslationEquivarianceLoss(
                epsilon=invalid_epsilon
            ),
            "finite",
        )
    _assert_value_error(
        lambda: TranslationEquivarianceLoss(shifts=((1, 2),)),
        "dx, dy, dz",
    )


def test_none_control_has_zero_constraint_loss() -> None:
    images = torch.randn(1, 1, 8, 8, 8)
    model = PointwiseSegmentationModel()
    logits = model(images)
    objective = NewConstraintObjective(NewConstraintConfig(equivariance_weight=0.0))

    output = objective(model, images, logits)

    assert output["loss"].item() == 0.0
    assert output["results"] == {}


def test_none_control_zero_does_not_overflow_with_fp16_logits() -> None:
    images = torch.zeros(1, 1, 32, 32, 32, dtype=torch.float16)
    logits = torch.full((1, 3, 32, 32, 32), 10.0, dtype=torch.float16)
    objective = NewConstraintObjective(NewConstraintConfig(equivariance_weight=0.0))

    # This is the failure mode of the old ``logits.sum() * 0.0`` expression:
    # every logit is finite, but the FP16 reduction overflows before * 0.
    assert torch.isfinite(logits).all()
    assert not torch.isfinite(logits.sum() * 0.0)

    output = objective(PointwiseSegmentationModel(), images, logits)

    assert output["loss"].item() == 0.0
    assert torch.isfinite(output["loss"])


def test_constraint_set_and_weight_cannot_disagree() -> None:
    base = {
        "translation_size": 2,
        "equivariance_max_samples": 1,
    }
    none_args = SimpleNamespace(
        **base,
        constraint_set="none",
        equivariance_weight=0.0,
    )
    assert resolve_constraint_config(none_args).equivariance_weight == 0.0
    _assert_value_error(
        lambda: resolve_constraint_config(
            SimpleNamespace(
                **base,
                constraint_set="none",
                equivariance_weight=0.1,
            )
        ),
        "cannot be combined",
    )
    _assert_value_error(
        lambda: resolve_constraint_config(
            SimpleNamespace(
                **base,
                constraint_set="translation",
                equivariance_weight=0.0,
            )
        ),
        "requires a finite positive",
    )
    for invalid_weight in (float("nan"), float("inf"), -float("inf")):
        _assert_value_error(
            lambda invalid_weight=invalid_weight: resolve_constraint_config(
                SimpleNamespace(
                    **base,
                    constraint_set="translation",
                    equivariance_weight=invalid_weight,
                )
            ),
            "finite positive",
        )


def test_data_order_is_independent_of_translation_sampling() -> None:
    control_data, _, _ = build_experiment_generators(61)
    translation_data, translation_shifts, _ = build_experiment_generators(61)

    assert torch.equal(
        torch.randperm(23, generator=control_data),
        torch.randperm(23, generator=translation_data),
    )
    for _ in range(17):
        torch.randint(6, (), generator=translation_shifts)
    assert torch.equal(
        torch.randperm(23, generator=control_data),
        torch.randperm(23, generator=translation_data),
    )


def test_generator_states_are_restored_exactly() -> None:
    data_generator, translation_generator, _ = build_experiment_generators(71)
    state = rng_state(data_generator, translation_generator)
    expected_data = torch.randperm(13, generator=data_generator)
    expected_shift = torch.randint(6, (), generator=translation_generator)

    restore_rng_state(state, data_generator, translation_generator)

    assert torch.equal(expected_data, torch.randperm(13, generator=data_generator))
    assert torch.equal(
        expected_shift,
        torch.randint(6, (), generator=translation_generator),
    )


def test_none_control_skips_constraint_evaluation_and_extra_forwards() -> None:
    class FailOnForward(nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            raise AssertionError("none must not perform a constraint forward")

    batch = {
        "image": torch.randn(2, 1, 7, 8, 9),
        "case_name": ["case-a", "case-b"],
    }
    objective = NewConstraintObjective(
        NewConstraintConfig(equivariance_weight=0.0)
    )
    details: list[dict[str, object]] = []

    metrics = evaluate_constraint_metrics(
        FailOnForward(),
        [batch],
        objective,
        torch.device("cpu"),
        amp=False,
        all_translation_shifts=True,
        detail_rows=details,
    )

    assert metrics == {}
    assert details == []


def test_segmentation_soft_and_hard_dice_are_distinct_and_valid() -> None:
    labels = torch.tensor([1, 1, 2, 2]).reshape(1, 1, 1, 1, 4)
    images = torch.zeros_like(labels, dtype=torch.float32)

    def logits_with_margin(margin: float) -> torch.Tensor:
        logits = torch.zeros(1, 3, 1, 1, 4)
        logits.scatter_(1, labels, margin)
        return logits

    weak_soft, weak_hard = evaluate_segmentation_metrics(
        FixedLogitModel(logits_with_margin(1.0)),
        [{"image": images, "label": labels}],
        torch.device("cpu"),
        num_classes=3,
    )
    strong_soft, strong_hard = evaluate_segmentation_metrics(
        FixedLogitModel(logits_with_margin(10.0)),
        [{"image": images, "label": labels}],
        torch.device("cpu"),
        num_classes=3,
    )

    assert weak_hard == 1.0
    assert strong_hard == 1.0
    assert 0.0 < weak_soft < strong_soft < 1.0


def test_constraint_aggregation_uses_case_direction_denominator() -> None:
    totals: dict[str, dict[str, float]] = {}
    update_constraint_totals(
        totals,
        {
            "translation_equivariance": ConstraintResult(
                loss=torch.tensor(0.5),
                truth=torch.tensor([0.2, 0.8]),
                value=torch.tensor([[0.1, 0.3], [0.7, 0.9]]),
                details={
                    "confidence_weighted_agreement": torch.tensor([0.2, 0.8]),
                    "confidence_adherent": torch.tensor([False, True]),
                },
            )
        },
    )

    metrics = averaged_constraint_metrics(totals, "val")

    assert abs(metrics["val_translation_equivariance_truth"] - 0.5) < 1e-7
    assert (
        abs(metrics["val_translation_equivariance_confidence_weighted_agreement"] - 0.5)
        < 1e-7
    )
    assert abs(metrics["val_translation_equivariance_confidence_adherent"] - 0.5) < 1e-7


def test_translation_defaults() -> None:
    config = NewConstraintConfig()

    assert config.equivariance_weight == 0.10
    assert config.translation_size == 2
    assert config.equivariance_max_samples is None


def test_resume_restores_initial_checkpoint_provenance() -> None:
    run_spec = RunSpec(
        dataset="MSD",
        fold=0,
        seed=0,
        translation_seed=1,
        epochs=50,
        batch_size=1,
        spatial_size=(64, 64, 64),
        resize=False,
        optimizer_mode="adamw_0.01",
        learning_rate=1e-4,
        weight_decay=1e-5,
        step_size=20,
        adamw_gamma=0.5,
        constraint_set="bands",
        constraint_config={"equivariance_weight": 0.0, "bands_weight": 0.04},
        constraint_warmup_epochs=5,
        constraint_eval_every=5,
        amp=False,
        initial_checkpoint=None,
    )
    saved = {
        **run_spec.__dict__,
        "spatial_size": [64, 64, 64],
        "initial_checkpoint": "/models/initial.pt",
        "initial_checkpoint_sha256": "a" * 64,
    }

    restored = resolve_resume_run_spec(run_spec, saved)

    assert restored.initial_checkpoint == "/models/initial.pt"
    assert restored.initial_checkpoint_sha256 == "a" * 64


def test_resume_still_rejects_real_configuration_changes() -> None:
    run_spec = RunSpec(
        dataset="MSD",
        fold=0,
        seed=0,
        translation_seed=1,
        epochs=50,
        batch_size=1,
        spatial_size=(64, 64, 64),
        resize=False,
        optimizer_mode="adamw_0.01",
        learning_rate=1e-4,
        weight_decay=1e-5,
        step_size=20,
        adamw_gamma=0.5,
        constraint_set="none",
        constraint_config={"equivariance_weight": 0.0, "bands_weight": 0.0},
        constraint_warmup_epochs=5,
        constraint_eval_every=5,
        amp=False,
        initial_checkpoint=None,
    )
    saved = {**run_spec.__dict__, "spatial_size": [64, 64, 64], "seed": 99}

    _assert_value_error(
        lambda: resolve_resume_run_spec(run_spec, saved),
        "does not match",
    )


def test_patient_group_extraction_handles_multicase_datasets() -> None:
    assert patient_id_from_case("hippocampus_001", "MSD") == "hippocampus_001"
    assert patient_id_from_case("hippocampus_mni_s042_left", "MNI") == "s042"
    assert patient_id_from_case("hippocampus_adni_123_left", "ADNI") == "adni_123"
    assert patient_id_from_case("hippocampus_cobra_77_right", "COBRA") == "cobra_77"


def test_checkpoint_embeds_run_provenance(tmp_path) -> None:
    model = PointwiseSegmentationModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    data_generator, translation_generator, _ = build_experiment_generators(3)
    run_spec = RunSpec(
        dataset="MSD",
        fold=0,
        seed=3,
        translation_seed=4,
        epochs=5,
        batch_size=1,
        spatial_size=(64, 64, 64),
        resize=False,
        optimizer_mode="adamw_0.01",
        learning_rate=1e-4,
        weight_decay=1e-5,
        step_size=20,
        adamw_gamma=0.5,
        constraint_set="none",
        constraint_config={"equivariance_weight": 0.0, "bands_weight": 0.0},
        constraint_warmup_epochs=5,
        constraint_eval_every=5,
        amp=False,
        initial_checkpoint=None,
    )
    path = tmp_path / "checkpoint_latest.pt"

    save_checkpoint(
        path,
        epoch=5,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        best_hard_dice=0.8,
        best_epoch=5,
        data_generator=data_generator,
        translation_generator=translation_generator,
        run_spec=run_spec,
    )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["run"]["constraint_set"] == "none"
    assert payload["run"]["spatial_size"] == (64, 64, 64)
