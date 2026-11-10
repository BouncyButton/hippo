"""Scientific invariants for the class-aware outer-band ablation."""
import torch
import torch.nn.functional as F
from scripts.mni_constraints.class_aware_bands import loss
from scripts.unetr_subfields.geo_recipe import apply_transform
from thesis.new_constraints.bands.outer_boundary import build_boundary_bands, OuterBoundaryBandLoss


def checks():
    torch.set_num_threads(4)
    label = torch.zeros((1, 1, 12, 12, 12), dtype=torch.long)
    label[:, :, 3:9, 3:9, 3:9] = 1
    label[:, :, 3:9, 6:9, 3:9] = 2
    label[:, :, 6:9, 3:6, 3:9] = 3
    inner, outer = build_boundary_bands(label > 0)
    torch.manual_seed(7)
    logits = torch.randn((1, 4, 12, 12, 12), requires_grad=True)
    actual = loss(logits, label)
    ce = F.cross_entropy(logits, label[:, 0], reduction='none')
    manual = .5 * (ce[inner[:, 0]].sum() / (inner.sum()+1e-6) + ce[outer[:, 0]].sum() / (outer.sum()+1e-6))
    assert torch.allclose(actual, manual, atol=1e-6)
    gradient = torch.autograd.grad(actual, logits)[0]
    support = (inner | outer).expand_as(gradient)
    assert torch.count_nonzero(gradient[~support]) == 0
    # At an inner-band true class1 voxel dominated by class2, descent must
    # increase the true-minus-wrong margin; pooled bands cannot enforce this.
    z = torch.zeros_like(logits, requires_grad=True)
    with torch.no_grad():
        z[:, 2] = 3.
    g = torch.autograd.grad(loss(z, label), z)[0]
    mask = inner[:, 0] & (label[:, 0] == 1)
    assert torch.all((-g[:, 1] + g[:, 2])[mask] > 0)
    pooled = OuterBoundaryBandLoss(foreground_class_ids=(1, 2, 3), complement_class_ids=(0,))
    pg = torch.autograd.grad(pooled(z, label).loss, z)[0]
    assert torch.all((-pg[:, 1] + pg[:, 2])[mask] < 0)
    # Moving probability between foreground classes leaves pooled bands
    # invariant but changes the new term when it changes the correct label.
    swapped = z.detach()[:, [0, 2, 1, 3]].clone()
    assert torch.allclose(pooled(z, label).loss, pooled(swapped, label).loss)
    assert not torch.allclose(loss(z, label), loss(swapped, label))
    for index in range(8):
        channels = []
        for channel in range(4):
            transformed, target = apply_transform(logits[:, channel:channel+1], label, index)
            channels.append(transformed)
        assert torch.allclose(loss(torch.cat(channels, 1), target), actual, atol=1e-6)
    for empty_label in (torch.zeros_like(label), torch.ones_like(label)):
        value = loss(logits, empty_label)
        assert torch.isfinite(value) and value == 0
    doubled = loss(torch.cat([logits, logits], 0), torch.cat([label, label], 0))
    assert torch.allclose(doubled, actual)
    return dict(status='passed', checks=['exact original union-band support', 'equal inner/outer and case averaging',
        'zero gradients off band', 'corrective true/wrong margin', 'categorical versus union distinction',
        'all eight paired D4 transforms', 'empty-side differentiable zero', 'batch averaging'])


if __name__ == '__main__':
    import json
    print(json.dumps(checks()))
