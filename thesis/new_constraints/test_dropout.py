"""Check real MONAI wiring, inference behavior, and invalid dropout settings."""

import pytest
import torch

from thesis.new_constraints import train_swinunetr_constraints as trainer


@pytest.mark.parametrize("rate", [-0.1, 1.0, float("nan"), float("inf")])
def test_invalid_dropout_is_rejected(rate):
    with pytest.raises(ValueError, match="drop-rate"):
        trainer.build_swinunetr((64, 64, 64), 3, torch.device("cpu"), drop_rate=rate)


def test_dropout_uses_same_initial_weights_and_is_disabled_at_evaluation():
    torch.manual_seed(0)
    baseline = trainer.build_swinunetr((64, 64, 64), 3, torch.device("cpu"))
    torch.manual_seed(0)
    candidate = trainer.build_swinunetr((64, 64, 64), 3, torch.device("cpu"), drop_rate=0.1)
    assert all(torch.equal(value, candidate.state_dict()[key])
               for key, value in baseline.state_dict().items())
    assert candidate.swinViT.pos_drop.p == 0.1
    block = candidate.swinViT.layers1[0].blocks[0]
    assert block.attn.attn_drop.p == 0.0
    assert block.attn.proj_drop.p == 0.1
    assert isinstance(block.drop_path, torch.nn.Identity)
    active = [m for m in block.mlp.modules() if isinstance(m, torch.nn.Dropout)]
    assert active and all(m.p == 0.1 for m in active)
    x = torch.ones(2, 8, 24)
    candidate.train()
    assert not torch.equal(block.mlp(x), block.mlp(x))
    candidate.eval()
    assert torch.equal(block.mlp(x), block.mlp(x))
    baseline.eval()
    assert torch.equal(block.mlp(x), baseline.swinViT.layers1[0].blocks[0].mlp(x))
