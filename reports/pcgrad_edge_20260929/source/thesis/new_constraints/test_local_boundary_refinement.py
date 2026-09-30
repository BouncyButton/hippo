import copy

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from thesis.new_constraints.local_boundary_refinement import (
    FrozenBackboneBoundaryRefiner, LocalBoundaryConfig, LocalBoundaryCorrection)


def example():
    logits = torch.zeros(1,3,11,11,11)
    logits[:,1:] = -3
    logits[:,1,3:8,3:8,3:8] = .2
    features = torch.randn(1,4,11,11,11)
    head = LocalBoundaryCorrection(LocalBoundaryConfig(feature_channels=4, hidden_channels=4))
    return logits, features, head


def test_exact_identity_then_local_bounded_changes_and_unchanged_class_odds():
    logits, features, head = example()
    initial = head(logits,features)
    assert torch.equal(initial['refined'],logits)
    with torch.no_grad():
        head.correction.bias.fill_(100)
    out = head(logits,features)
    outside = ~out['support'].expand_as(logits)
    assert torch.equal(out['refined'][outside],logits[outside])
    assert torch.equal(out['refined'][:,1:],logits[:,1:])
    assert out['delta'].abs().max() <= 2
    assert not out['support'][0,0,5,5,5]  # Body interior protected.
    assert not out['support'][0,0,0,0,0]  # Distant background protected.


def test_physical_band_matches_euclidean_distance_not_cube_and_ignores_padding():
    logits, _, head = example()
    logits[:,1:] = -3
    logits[:,1,5,5,5] = .2
    support = head.support(logits)
    assert support[0,0,5,5,7]
    assert not support[0,0,5,7,7]  # sqrt(8) > 2 mm.
    anisotropic = LocalBoundaryCorrection(LocalBoundaryConfig(feature_channels=4, spacing_mm=(3,1,1)))
    assert not anisotropic.support(logits)[0,0,6,5,5]
    for value in (-3,3):
        logits[:,1] = value
        assert not head.support(logits).any()  # No synthetic crop-face boundary.


def test_neighboring_empty_ray_is_editable_but_distant_ray_is_not():
    logits, _, head = example()
    logits[:,1:] = -3
    logits[:,1,5,5,5] = .2
    band = head.support(logits)
    assert not (logits.argmax(1)[0,6,5] > 0).any()
    assert band[0,0,6,5,5]
    assert not band[0,0,8,5].any()


def test_thin_foreground_is_not_falsely_claimed_protected():
    logits, features, head = example()
    logits[:,1:] = -3
    logits[:,1,5,5,5] = .2
    with torch.no_grad():
        head.correction.bias.fill_(-100)
    out = head(logits,features)
    assert logits.argmax(1)[0,5,5,5] == 1
    assert out['refined'].argmax(1)[0,5,5,5] == 0


def test_detached_head_inputs_preserve_only_direct_logit_gradient():
    logits, features, head = example()
    logits.requires_grad_()
    features.requires_grad_()
    with torch.no_grad():
        head.correction.weight.fill_(.01)
    out = head(logits,features)
    grad_logits, grad_features = torch.autograd.grad(out['delta'].sum(), (logits, features), allow_unused=True)
    assert grad_logits is None and grad_features is None
    out = head(logits,features)
    grad = torch.autograd.grad(out['refined'].sum(), logits)[0]
    assert torch.equal(grad,torch.ones_like(logits))


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Conv3d(1,4,1)
        self.out = nn.Conv3d(4,3,1)
    def forward(self,image):
        return self.out(self.decoder(image))


def test_optimizer_step_checkpoint_roundtrip_and_frozen_backbone():
    torch.manual_seed(3)
    model = FrozenBackboneBoundaryRefiner(TinyBackbone(),LocalBoundaryConfig(feature_channels=4,hidden_channels=4))
    with torch.no_grad():
        model.backbone.decoder.weight.zero_()
        model.backbone.decoder.bias.zero_()
        model.backbone.decoder.weight[0,0] = 1
        model.backbone.out.weight.zero_()
        model.backbone.out.bias.copy_(torch.tensor([0.,0.,-4.]))
        model.backbone.out.weight[1,0] = 1
    image = torch.randn(1,1,7,7,7)
    labels = torch.ones(1,7,7,7,dtype=torch.long)
    original = copy.deepcopy(model.backbone.state_dict())
    optimizer = torch.optim.AdamW(model.refiner.parameters(),lr=.01)
    model.train()
    losses=[]
    for _ in range(4):
        optimizer.zero_grad()
        out = model(image)
        loss = F.cross_entropy(out['refined'],labels)
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0]
    assert not model.backbone.training
    assert all(p.grad is None for p in model.backbone.parameters())
    for k,v in original.items():
        assert torch.equal(model.backbone.state_dict()[k],v)
    restored = copy.deepcopy(model)
    restored.load_state_dict(model.state_dict())
    assert torch.equal(restored(image)['refined'],model(image)['refined'])
    assert len(model.backbone.out._forward_pre_hooks) == 0


@pytest.mark.parametrize('kwargs',[{'radius_mm':0},{'max_logit_shift':float('nan')},{'spacing_mm':(1,0,1)}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        LocalBoundaryConfig(**kwargs)
