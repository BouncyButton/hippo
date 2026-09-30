"""Verify execution optimizations keep tensor values and model weights unchanged."""
import pytest
import torch
from monai.data import MetaTensor

from thesis.new_constraints import train_swinunetr_constraints as trainer
from thesis.new_constraints.objective import NewConstraintConfig, NewConstraintObjective
from thesis.new_constraints.constraint_result import differentiable_zero


@pytest.mark.parametrize('size', [0, 1, 3])
def test_constraint_logging_batches_transfers_and_preserves_totals(monkeypatch, size):
    from thesis.new_constraints.constraint_result import ConstraintResult
    values = torch.linspace(0.1, 0.9, steps=max(size, 1))[:size].requires_grad_(True)
    result = ConstraintResult(loss=values.sum(), truth=values, value=values, details={
        'confidence_weighted_agreement': values,
        'confidence_adherent': values > 0.5,
        'metrics': {'raw_loss': values, 'valid_patient': torch.ones(size)},
    })
    calls = []
    original = torch.Tensor.cpu
    def counted(tensor, *args, **kwargs):
        calls.append(tensor.numel())
        return original(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'cpu', counted)
    totals = {}
    trainer.update_constraint_totals(totals, {'test': result})
    trainer.update_constraint_totals(totals, {'test': result})
    assert calls == [5, 5]
    row = totals['test']
    expected = float(values.detach().sum()) * 2
    assert row['truth'] == expected == row['confidence_weighted_agreement']
    assert row['confidence_adherent'] == float((values > 0.5).float().sum()) * 2
    assert row['count'] == 2 * size
    assert row['metric_sums'] == {'raw_loss': expected, 'valid_patient': float(2 * size)}
    assert row['metric_counts'] == {'raw_loss': float(2 * size), 'valid_patient': float(2 * size)}


def test_plain_transform_preserves_values_names_and_rng():
    image = MetaTensor(torch.randn(1, 8, 8, 8), meta={"example": "retained until preprocessing ends"})
    label = MetaTensor(torch.ones(1, 8, 8, 8, dtype=torch.long))
    sample = {"image": image, "label": label, "case_name": "case"}
    state = torch.get_rng_state()
    output = trainer.PlainTensorTransform(lambda x: dict(x))(sample)
    assert type(output["image"]) is torch.Tensor
    assert type(output["label"]) is torch.Tensor
    assert torch.equal(output["image"], image.as_tensor())
    assert torch.equal(output["label"], label.as_tensor())
    assert output["image"].data_ptr() == image.data_ptr()
    assert output["case_name"] == "case"
    assert torch.equal(state, torch.get_rng_state())
    assert isinstance(sample["image"], MetaTensor)


def test_activation_checkpointing_keeps_initial_parameters_and_dropout():
    torch.manual_seed(0)
    original = trainer.build_swinunetr((64,64,64), 3, torch.device("cpu"), drop_rate=0.1)
    torch.manual_seed(0)
    fast = trainer.build_swinunetr((64,64,64), 3, torch.device("cpu"), drop_rate=0.1,
                                 activation_checkpointing=False)
    assert all(torch.equal(v, fast.state_dict()[k]) for k,v in original.state_dict().items())
    blocks = [m for m in fast.modules() if hasattr(m, "use_checkpoint")]
    assert blocks and all(not m.use_checkpoint for m in blocks)
    assert fast.swinViT.pos_drop.p == 0.1


def test_disabled_constraint_never_decollates_flattened_meta_tensor(monkeypatch):
    import monai.data.meta_tensor as meta_module

    raw = torch.randn(1, 3, 8, 8, 8, requires_grad=True)
    logits = MetaTensor(raw)
    logits.is_batch = True
    objective = NewConstraintObjective(NewConstraintConfig(equivariance_weight=0))

    def reject_decollation(*args, **kwargs):
        raise AssertionError("Zero constraint loss must not decollate metadata.")

    monkeypatch.setattr(meta_module, "decollate_batch", reject_decollation)
    result = objective(torch.nn.Identity(), raw, logits)
    assert type(result["loss"]) is torch.Tensor
    assert result["loss"].requires_grad and result["loss"].item() == 0
    assert result["results"] == {}
    result["loss"].backward()
    assert raw.grad is not None and torch.count_nonzero(raw.grad) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shared_zero_preserves_dtype_and_gradient_without_metadata_splitting(monkeypatch, dtype):
    import monai.data.meta_tensor as meta_module

    raw = torch.randn(2, 3, 8, 8, 8, dtype=dtype, requires_grad=True)
    wrapped = MetaTensor(raw)
    wrapped.is_batch = True

    def reject_decollation(*args, **kwargs):
        raise AssertionError("Scalar zero must not split metadata.")

    monkeypatch.setattr(meta_module, "decollate_batch", reject_decollation)
    zero = differentiable_zero(wrapped)
    assert type(zero) is torch.Tensor and zero.dtype == dtype and zero.item() == 0
    zero.backward()
    torch.testing.assert_close(raw.grad, torch.zeros_like(raw), rtol=0, atol=0)


@pytest.mark.parametrize("name", ["ap_cut", "plane", "location", "plane_ce", "band", "tversky", "onecut"])
def test_invalid_geometry_zero_on_batched_metadata_preserves_backward(name):
    from thesis.new_constraints.ap_cut import APCutPosteriorLoss
    from thesis.new_constraints.ap_plane import ExistentialAPPlaneLoss, BestFitAPPlaneLocationLoss, OriginalLabelAPConditionalCELoss
    from thesis.new_constraints.bands import OuterBoundaryBandLoss, ClassAwareBoundaryTverskyLoss
    from thesis.new_constraints.onecut import OuterOneCutLogLTNLoss

    factories = {
        "ap_cut": lambda: APCutPosteriorLoss(axis=1, anterior_low=False, invalid_policy="skip"),
        "plane": ExistentialAPPlaneLoss, "location": BestFitAPPlaneLocationLoss,
        "plane_ce": OriginalLabelAPConditionalCELoss, "band": OuterBoundaryBandLoss,
        "tversky": ClassAwareBoundaryTverskyLoss, "onecut": OuterOneCutLogLTNLoss,
    }
    raw = torch.randn(1, 3, 8, 8, 8, requires_grad=True)
    wrapped = MetaTensor(raw)
    wrapped.is_batch = True
    labels = torch.zeros(1, 1, 8, 8, 8, dtype=torch.long)
    result = factories[name]()(wrapped, labels)
    assert result.loss.item() == 0 and result.loss.requires_grad
    result.loss.backward()
    torch.testing.assert_close(raw.grad, torch.zeros_like(raw), rtol=0, atol=0)
