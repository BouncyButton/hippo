import torch
import pytest

from thesis.new_constraints.edge_consistency import BoundaryEdgeConsistencyLoss, signed_face_error


def test_explicit_face_enumeration_and_gradient():
    torch.manual_seed(4)
    p = torch.rand(2, 1, 4, 5, 6, dtype=torch.double, requires_grad=True)
    y = torch.rand_like(p) > .5
    support = torch.rand_like(p) > .2
    total, count = signed_face_error(p, y, support)
    expected = []
    counts = []
    for b in range(2):
        terms = []
        for x in range(4):
            for yy in range(5):
                for z in range(6):
                    for dx, dy, dz in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                        j = (x + dx, yy + dy, z + dz)
                        if any(a >= n for a, n in zip(j, (4, 5, 6))):
                            continue
                        i = (b, 0, x, yy, z)
                        j = (b, 0, *j)
                        if support[i] and support[j]:
                            terms.append(((p[i] - p[j]) - (y[i].double() - y[j].double())).square())
        expected.append(torch.stack(terms).sum())
        counts.append(len(terms))
    expected = torch.stack(expected)
    torch.testing.assert_close(total, expected)
    assert count.tolist() == counts
    torch.testing.assert_close(torch.autograd.grad(total.sum(), p, retain_graph=True)[0],
                               torch.autograd.grad(expected.sum(), p)[0])


def test_correct_orientation_beats_reversed_boundary_and_constant_offset_is_invisible():
    y = torch.tensor([1., 0.]).reshape(1, 1, 1, 1, 2)
    support = torch.ones_like(y, dtype=torch.bool)
    correct, n = signed_face_error(y, y, support)
    reverse, _ = signed_face_error(1-y, y, support)
    offset, _ = signed_face_error(y + .1, y, support)
    assert n.item() == 1 and correct.item() == 0 and reverse.item() == 4
    torch.testing.assert_close(offset, torch.zeros_like(offset), atol=1e-12, rtol=0)


def test_empty_case_skipped_and_ap_invariance():
    labels = torch.zeros(2, 1, 9, 9, 9, dtype=torch.long)
    labels[0, :, 2:7, 2:7, 2:7] = 1
    labels[0, :, 5:7, 2:7, 2:7] = 2
    logits = torch.randn(2, 3, 9, 9, 9, requires_grad=True)
    loss = BoundaryEdgeConsistencyLoss()
    torch.testing.assert_close(loss(logits, labels), loss(logits[:1], labels[:1]))
    swapped = torch.where(labels == 1, 2, torch.where(labels == 2, 1, 0))
    torch.testing.assert_close(loss(logits, labels), loss(logits[:, [0, 2, 1]], swapped))
    empty = loss(logits, torch.zeros_like(labels))
    assert empty.item() == 0
    assert torch.isfinite(torch.autograd.grad(empty, logits)[0]).all()


def test_shape_validation():
    with pytest.raises(ValueError):
        BoundaryEdgeConsistencyLoss()(torch.zeros(1, 2, 4, 4, 4), torch.zeros(1, 1, 4, 4, 4))
