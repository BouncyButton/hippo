"""Synthetic checks that coherence is distinct from overlap and smoothing."""
import numpy as np
from audit_edge_coherence import metrics, volume_match


def cube():
    y = np.zeros((9, 9, 9), dtype=np.uint8)
    y[2:7, 2:7, 2:7] = 1
    return y


def test_perfect_and_reversed_boundaries():
    y = cube()
    perfect = metrics(y.astype(float), y, y)
    assert perfect['all_soft_squared_error'] == 0
    assert perfect['inner_disagree'] == perfect['outer_disagree'] == 0
    assert perfect['cross_correct_transition'] == 1
    reversed_ = metrics((1-y).astype(float), 1-y, y)
    assert reversed_['cross_soft_squared_error'] == 4
    assert reversed_['cross_reversed_transition'] == 1


def test_constant_is_coherent_but_incorrect():
    y = cube()
    m = metrics(np.zeros_like(y, dtype=float), np.zeros_like(y), y)
    assert m['inner_disagree'] == m['outer_disagree'] == 0
    assert m['inner_both_wrong'] == 1
    assert m['cross_correct_transition'] == 0
    assert m['cross_soft_squared_error'] == 1


def test_equal_dice_can_have_different_coherence():
    y = cube()
    scattered = y.copy()
    clustered = y.copy()
    scattered[2, 3, 3] = scattered[2, 5, 5] = 0
    clustered[2, 3, 3] = clustered[2, 3, 4] = 0
    a = metrics(scattered.astype(float), scattered, y)
    b = metrics(clustered.astype(float), clustered, y)
    assert a['union_dice'] == b['union_dice']
    assert a['inner_disagree'] > b['inner_disagree']


def test_volume_match_counts_and_ranks():
    p = np.arange(27).reshape(3, 3, 3) / 26
    for n in (0, 1, 13, 27):
        h = volume_match(p, n)
        assert h.sum() == n
        if 0 < n < 27:
            assert p[h > 0].min() > p[h == 0].max()
