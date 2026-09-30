import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from edge_lowdata_metrics import crossing, compare_histories


def rows(scores):
    return [{'epoch': i + 1, 'val_dice_hard': v} for i, v in enumerate(scores)]


def test_crossing_requires_sustained_observations_and_reports_unreached():
    r = rows([.8, .6, .8, .81, .82])
    assert crossing(r, .8)['first_epoch'] == 1
    assert crossing(r, .8, 3) == {'first_epoch': 3, 'confirmed_epoch': 5, 'updates_to_confirmation': 50}
    assert crossing(r, .9) is None


def test_comparison_truncates_common_horizon_without_filling_after_stop():
    result = compare_histories({'dice':rows([.4,.5,.6]), 'bands':rows([.5,.6]), 'edge':rows([.6,.7,.8,.9])})
    assert result['common_observed_epochs'] == 2
    assert result['models']['edge']['mean_dice_over_common_epochs'] == (.6+.7)/2
    assert result['thresholds']['bands_reference_best'] == .6
