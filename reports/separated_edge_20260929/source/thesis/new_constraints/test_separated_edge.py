import numpy as np
import pytest
import torch

from thesis.new_constraints.edge_consistency import BoundaryEdgeConsistencyLoss
from thesis.new_constraints.separated_edge import rule_statistics, rule_losses, SeparatedEdgeLoss, matched_budget


def example():
    torch.manual_seed(29)
    z = torch.randn(2, 3, 9, 9, 9, requires_grad=True)
    y = torch.zeros(2, 1, 9, 9, 9, dtype=torch.long)
    y[0, :, 2:7, 2:7, 2:7] = 1
    y[1, :, 3:6, 3:6, 3:6] = 2
    return z, y


def test_disjoint_partition_recovers_original_value_and_gradient():
    z, y = example()
    sums, counts = rule_statistics(z, y)
    reconstructed = (sum(sums.values()) / sum(counts.values())).mean()
    original = BoundaryEdgeConsistencyLoss()(z, y)
    torch.testing.assert_close(reconstructed, original)
    torch.testing.assert_close(torch.autograd.grad(reconstructed, z, retain_graph=True)[0],
                               torch.autograd.grad(original, z)[0])


def test_equal_rules_patient_balance_and_empty_case():
    z, y = example()
    loss = SeparatedEdgeLoss()
    expected = sum(loss(z[i:i+1], y[i:i+1]) for i in range(2))/2
    torch.testing.assert_close(loss(z, y), expected)
    terms, counts = rule_losses(z, y)
    assert all((v > 0).all() for v in counts.values())
    torch.testing.assert_close(loss(z, y), sum(terms.values())/3)
    expanded = loss(torch.cat([z, z[:1]]), torch.cat([y, torch.zeros_like(y[:1])]))
    torch.testing.assert_close(expanded, loss(z, y))
    empty = loss(z, torch.zeros_like(y))
    assert empty.item() == 0
    assert torch.autograd.grad(empty, z)[0].abs().max() == 0


def test_perfect_foreground_and_reversed_transition():
    _, y = example()
    z = torch.full((2, 3, 9, 9, 9), -30.)
    z[:, :1] = torch.where(y == 0, 30., -30.)
    z[:, 1:2] = torch.where(y > 0, 30., -30.)
    terms, _ = rule_losses(z, y)
    assert all(v.item() < 1e-12 for v in terms.values())
    reverse, _ = rule_losses(z[:, [1, 0, 2]], y)
    assert reverse['cross'].item() == pytest.approx(4)
    assert reverse['inner'].item() == 0 and reverse['outer'].item() == 0


def test_matches_independent_numpy_face_audit():
    from scripts.audit_edge_coherence import metrics
    z, y = example()
    terms, _ = rule_losses(z, y)
    p = z.detach().softmax(1).numpy()
    rows = [metrics(p[i, 1:].sum(0), p[i].argmax(0), y[i,0].numpy()) for i in range(2)]
    for k in terms:
        assert terms[k].item() == pytest.approx(np.mean([r[k+'_soft_squared_error'] for r in rows]), rel=1e-5)


@pytest.mark.parametrize('ratios', [dict(pooled=[1]*10, separated=[2]*10),
                                    dict(pooled=[1]*10, separated=[1]*9+[100])])
def test_budget_is_matched_even_when_one_arm_hits_cap(ratios):
    result = matched_budget(ratios)
    medians = []
    for arm, values in ratios.items():
        weighted = np.array(values) * result['weights'][arm]
        medians.append(np.median(weighted))
        assert np.quantile(weighted, .95) <= .5 + 1e-12
    assert medians[0] == pytest.approx(medians[1])
    assert medians[0] <= .1 + 1e-12


def test_budget_rejects_missing_or_invalid_gradients():
    with pytest.raises(ValueError):
        matched_budget(dict(pooled=[0], separated=[1]))
    with pytest.raises(ValueError):
        matched_budget(dict(pooled=[1]))


def test_historical_coefficient_is_preserved_or_matching_fails():
    result = matched_budget(dict(pooled=[2]*10,separated=[4]*10),reference_weight=.04)
    assert result['weights']['pooled'] == .04
    assert result['weights']['separated'] == .02
    assert result['matched_median_budget'] == .08
    with pytest.raises(ValueError,match='stop for review'):
        matched_budget(dict(pooled=[1]*10,separated=[1]*9+[100]),reference_weight=.1)
