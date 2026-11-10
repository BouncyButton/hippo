"""Check that the window changes only auxiliary gradients and charges its work."""
import ast
from pathlib import Path
import torch

from scripts.mni_constraints import objectives
from scripts.mni_constraints.unetr_constraint_window import envelope, account, instrument_source
from scripts.cross_dataset_campaign.common import read
from scripts.capped_depth_delay.schedule import training_flops


def run():
    torch.set_num_threads(1)
    factors = envelope(409, 205, 307)
    assert factors[:206] == [1.] * 206 and factors[256] == .5
    assert factors[307:] == [0.] * 102
    assert all(a >= b for a,b in zip(factors, factors[1:]))
    torch.manual_seed(811)
    z = torch.randn(1,4,6,6,6, requires_grad=True)
    label = torch.zeros(1,1,6,6,6,dtype=torch.long)
    label[:,:,1:5,1:5,1:3]=1;label[:,:,1:5,1:5,3:4]=2;label[:,:,1:5,1:5,4:5]=3
    sdf = objectives.signed_distance_maps(label);depth = torch.ones(6,6,6)/216
    base = objectives.base_loss(z,label,'unetr_vit_b16')
    gb = torch.autograd.grad(base,z,retain_graph=True)[0]
    for arm in ('dice_bands','dice_bands_edge_sdf_depth'):
        weights = {n:.02 for n in objectives.TERMS if n in arm}
        def loss(factor):
            return objectives.training_loss(z,label,arm,weights,.7*factor,sdf,None,depth,
                                            .4*factor,identity='unetr_vit_b16')
        zero = loss(0.)
        assert torch.equal(zero,base)
        assert torch.equal(torch.autograd.grad(zero,z,retain_graph=True)[0],gb)
        full = torch.autograd.grad(loss(1.),z,retain_graph=True)[0]
        half = torch.autograd.grad(loss(.5),z,retain_graph=True)[0]
        assert torch.allclose(half-gb,.5*(full-gb),atol=2e-8,rtol=1e-4)
    root = Path(__file__).resolve().parents[2]
    parent = root/'experiments/mni_unetr_delayed_combinations53t_20261108'
    cap = 53305109592450
    for arm in ('dice_bands_edge','dice_bands_edge_sdf_depth'):
        s = read(parent/'protocols'/arm/'protocol.json')['models']['unet']['objectives'][arm]
        adjusted = account(s,'sdf' in arm,cap)
        expected = 409 + (409-s['sdf_prefix_updates'] if 'sdf' in arm else 0)
        assert training_flops(409,adjusted)-training_flops(409,s)==expected
        assert adjusted['max_steps']==409
    source = (parent/'source/scripts/cross_dataset_campaign/capped_core.py').read_text()
    changed = instrument_source(source);ast.parse(changed)
    assert changed.count('loss.backward()') == source.count('loss.backward()') == 1
    assert changed.count('optimizer.step()') == source.count('optimizer.step()') == 1
    print('Window boundaries, base-only gradients, linear auxiliary scaling, and FLOPs checks passed.')


if __name__ == '__main__':
    run()
