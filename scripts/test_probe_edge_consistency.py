from scripts.probe_edge_consistency import decide_gate


def aggregate(fn=10, fp=10, classes=25):
    return {step: {'base': {'inner_fn': 10, 'outer_fp': 10, 'class_errors': 25},
                   'combined': {'inner_fn': fn, 'outer_fp': fp, 'class_errors': classes}}
            for step in ('0.25', '1.0')}


def test_gate_requires_strict_gain_without_trading_fn_for_fp():
    assert decide_gate(aggregate(fn=9, classes=24), True, .2)['passed']
    assert not decide_gate(aggregate(fn=5, fp=11, classes=21), True, .2)['passed']
    assert not decide_gate(aggregate(), True, .2)['passed']


def test_gate_rejects_invalid_gradients_or_class_harm():
    assert not decide_gate(aggregate(fn=9), False, .2)['passed']
    assert not decide_gate(aggregate(fn=9), True, -.1)['passed']
    assert not decide_gate(aggregate(fn=9, classes=26), True, .2)['passed']
