import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_edge_generalization import METRICS, compare_splits


def records(prefix, count, effect):
    return {model: [{'case_name': f'{prefix}{i}', 'metrics': {
        key: .3 + (effect if model == 'edge' else 0) for key in METRICS}}
        for i in range(count)] for model in ('dice', 'bands', 'edge')}


def test_training_only_effect_uses_case_means_and_correct_metric_directions():
    result = compare_splits(records('train', 8, -.1), records('val', 2, .05))
    m = result['edge_effect_relative_to']['bands']['balanced_boundary_error']
    assert m['training_only_improvement_point_estimate']
    assert m['train_edge_minus_reference'] == pytest.approx(-.1)
    assert m['validation_edge_minus_reference'] == pytest.approx(.05)
    assert m['effect_gap_validation_minus_train'] == pytest.approx(.15)
    assert m['effect_gap_bootstrap_95_interval'] == pytest.approx([.15, .15])
    assert not result['edge_effect_relative_to']['bands']['macro_dice']['training_only_improvement_point_estimate']


def test_rejects_overlap_or_unpaired_cases():
    with pytest.raises(AssertionError):
        compare_splits(records('same', 8, 0), records('same', 2, 0))
    training = records('train', 8, 0)
    training['dice'].reverse()
    with pytest.raises(AssertionError):
        compare_splits(training, records('val', 2, 0))
