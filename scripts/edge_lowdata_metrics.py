"""Prespecified learning-curve comparisons; epochs are paired within seeds."""
import math


def crossing(rows, threshold, consecutive=1, updates_per_epoch=10):
    scores = [float(r['val_dice_hard']) for r in rows]
    for end in range(consecutive - 1, len(rows)):
        start = end - consecutive + 1
        if all(s >= threshold for s in scores[start:end + 1]):
            return {'first_epoch': int(rows[start]['epoch']), 'confirmed_epoch': int(rows[end]['epoch']),
                    'updates_to_confirmation': int(rows[end]['epoch']) * updates_per_epoch}
    return None


def compare_histories(histories, *, candidate='edge', references=('dice', 'bands')):
    """No extrapolation after early stopping; unreached targets remain null."""
    assert set(histories) == {candidate, *references}
    for rows in histories.values():
        assert [int(r['epoch']) for r in rows] == list(range(1, len(rows) + 1))
        assert all(math.isfinite(float(r['val_dice_hard'])) for r in rows)
    targets = {f'{name}_reference_best': max(float(r['val_dice_hard']) for r in histories[name]) for name in references}
    targets.update({f'fixed_{v}pct': v / 100 for v in (70, 74, 76, 78, 80)})
    common = min(map(len, histories.values()))
    result = {'common_observed_epochs': common, 'common_observed_updates': common * 10,
              'thresholds': targets, 'models': {},
              'timing_scope': 'Epoch/update crossings compare the same ten cases and LR schedule. Historical controls used 4g.40gb; candidate uses 3g.40gb, so no direct wall-clock speedup claim. Calibration overhead is recorded separately.'}
    for name, rows in histories.items():
        best = max(rows, key=lambda r: float(r['val_dice_hard']))
        result['models'][name] = {
            'best_dice': float(best['val_dice_hard']), 'best_epoch': int(best['epoch']),
            'stopped_epoch': int(rows[-1]['epoch']), 'final_dice': float(rows[-1]['val_dice_hard']),
            'mean_dice_over_common_epochs': sum(float(r['val_dice_hard']) for r in rows[:common]) / common,
            'dice_at_common_epochs': {str(e): float(rows[e - 1]['val_dice_hard']) for e in (10,20,40,60,75) if e <= common},
            'crossings': {key: {'first': crossing(rows, value), 'sustained_three': crossing(rows, value, 3)}
                          for key, value in targets.items()}}
    result[f'{candidate}_minus_reference'] = {name: {
        'best_dice': result['models'][candidate]['best_dice'] - result['models'][name]['best_dice'],
        'mean_dice_over_common_epochs': result['models'][candidate]['mean_dice_over_common_epochs'] - result['models'][name]['mean_dice_over_common_epochs']}
        for name in references}
    return result
