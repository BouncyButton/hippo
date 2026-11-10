"""Scientific invariants for prediction-dependent band supervision."""
import torch
import torch.nn.functional as F
from scripts.mni_constraints.adaptive_bands import loss, masks
from thesis.new_constraints.bands.outer_boundary import build_boundary_bands, foreground_log_odds


def checks():
    torch.set_num_threads(4)
    y = torch.zeros(1, 1, 16, 16, 16, dtype=torch.long)
    y[:, :, 3:13, 3:13, 3:13] = 1
    y[:, :, 8:13, 3:13, 3:13] = 2
    torch.manual_seed(13)
    z = torch.randn(1, 4, 16, 16, 16, requires_grad=True)
    anchor = torch.stack(build_boundary_bands(y > 0)).any(0)
    for variant in ('weighted', 'adaptive'):
        i, o, w, e = masks(z, y, variant)
        assert not w.requires_grad and torch.equal(e, (y > 0) ^ (z.argmax(1, keepdim=True) > 0))
        assert torch.equal(w, 1. + 2. * e.float())
        assert torch.equal(i | o, anchor) if variant == 'weighted' else bool(((i | o)[e | anchor]).all())
        voxel = F.binary_cross_entropy_with_logits(foreground_log_odds(z, (1, 2, 3), (0,)), (y[:, 0] > 0).float(), reduction='none')
        expected = .5 * ((voxel * (i*w)[:, 0]).sum() / ((i*w).sum()+1e-6) + (voxel * (o*w)[:, 0]).sum() / ((o*w).sum()+1e-6))
        value = loss(z, y, variant)
        assert torch.allclose(value, expected)
        grad = torch.autograd.grad(value, z)[0]
        assert torch.isfinite(grad).all() and not grad[~(i | o).expand_as(grad)].count_nonzero()
        assert torch.allclose(loss(torch.cat([z,z]), torch.cat([y,y]), variant), value)
        for k in range(4):
            for flip in (False, True):
                a, b = torch.rot90(z,k,(2,3)), torch.rot90(y,k,(2,3))
                if flip: a,b = a.flip(2),b.flip(2)
                assert torch.allclose(loss(a,b,variant), value, atol=1e-6)
    # Summed foreground probability exceeds .5, yet BG is the largest class.
    q = torch.zeros_like(z); q[:, 0] = .5
    assert (q.softmax(1)[:, 1:].sum(1) > .5).all()
    assert torch.equal(masks(q,y,'adaptive')[3], y > 0)
    # Correct union with subfield swaps is not an external error.
    q = F.one_hot(torch.where(y[:,0] > 0, 3, 0),4).permute(0,4,1,2,3).float()*10
    assert not masks(q,y,'adaptive')[3].any()
    assert torch.equal(masks(q,y,'adaptive')[0] | masks(q,y,'adaptive')[1], anchor)
    # An isolated remote FP and a deep FN must be included with six neighbours.
    q[0,0,0,0,0]=0; q[0,1,0,0,0]=10
    q[0,:,7,7,7]=0; q[0,0,7,7,7]=10
    i,o,w,e=masks(q,y,'adaptive')
    assert o[0,0,0,0,1] and i[0,0,7,7,8] and e.sum()==2
    assert torch.isfinite(loss(z,torch.zeros_like(y),'weighted'))
    return dict(status='passed',checks=['actual argmax union errors','detached 3x weights','original support A',
        'anchor plus six-connected error halo B','side normalization','zero off-support gradient',
        'D4 equivariance','batch averaging','internal swaps excluded','remote FP/deep FN included'])


if __name__ == '__main__':
    import json
    print(json.dumps(checks()))
