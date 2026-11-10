"""Reconstruct validation records from aggregate evidence and frozen run order."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def read(path):
    return json.loads(path.read_text())


live = read(HERE / 'progression_live_20261012.json')['rows']
live_by_id = {r['objective']: r for r in live}
base_protocol = read(HERE / 'protocol.json')
sources = [
    ('Original edge study', ROOT / 'edge_ablation_20261008/final_results/protocol.json',
     ROOT / 'edge_ablation_20261008/final_results/results.json'),
    ('Band / Chamfer study', ROOT / 'band_chamfer_20261008/protocol.json',
     ROOT / 'band_chamfer_20261008/results.json'),
    ('Independent endpoints / bands study', ROOT / 'band_evolution_20261010_recovery/protocol.json',
     ROOT / 'band_evolution_20261010_recovery/final_results/aggregate_summary.json'),
    ('Focus / width study', HERE / 'protocol.json', HERE / 'progression_live_20261012.json'),
]
descriptions = {
    'dice': 'Dice only',
    'bands': 'Original inner/outer bands; balanced BCE',
    'legacy_edge': 'Original bands + original edge constraint',
    'split_equal': 'Original bands + equal weighting of same-side and crossing edges',
    'probability_1mm_anchors_outer': 'Original bands + 1 mm probability comparisons, balanced per anchor',
    'probability_1and3mm_pairs_outer': 'Original bands + 1+3 mm probability comparisons',
    'pair_common_3': 'Original bands + 3 mm comparisons on common eligible support',
    'pair_independent13': 'Original bands + separate inside/outside targets at 1+3 mm',
    'combo_sdf8_in50': 'Independent 1+3 mm targets + original bands + signed-distance term',
    'band_bce1_pairs_in50': 'Independent 1+3 mm targets + 1/1 mm BCE bands',
    'band_focus_top25_in50': '3/3 mm Brier bands; hardest 25%, emphasis 50%; independent 1+3 mm targets',
    'hard_q20_mix50': '3/3 mm bands; hardest 20%, emphasis 50%',
    'hard_q35_mix75': '3/3 mm bands; hardest 35%, emphasis 75%',
    'width_i2_o2_q25': '2/2 mm bands; hardest 25%, emphasis 50%',
    'width_i3_o2_q25': '3/2 mm bands; hardest 25%, emphasis 50%',
    'a32_hard_q35_mix75': '3/2 mm bands; hardest 35%, emphasis 75%',
    'a32_neighbor_k3_q35_mix75_point50': '3/2 mm bands; 35%/75% focus + mixed point and 3×3×3 neighborhood ranking',
    'a32_budget_b1p0_p1p5': '3/2 mm bands; 25%/50% focus; endpoint gradient-budget target ×1.5',
    'a32_side_qi40_qo15_mix75': '3/2 mm bands; hardest 40% inside / 15% outside; emphasis 75%',
    'a32_side_qi40_qo25_mix75': '3/2 mm bands; hardest 40% inside / 25% outside; emphasis 75%',
    'a32_side_qi60_qo15_mix75': '3/2 mm bands; hardest 60% inside / 15% outside; emphasis 75%',
}

records = []
best = float('-inf')


def consider(row, campaign, source):
    global best
    score = row['macro_dice'] * 100
    if score <= best + 1e-6:
        return
    name = row['objective']
    assert name in descriptions, f'New record requires an accurate description: {name}'
    records.append(dict(step=len(records)+1, objective=name, description=descriptions[name],
                        dice_percent=score, gain_pp=None if not records else score-best,
                        campaign=campaign, source=str(source.relative_to(ROOT)),
                        best_epoch=row['best_epoch']))
    best = score


consider(live_by_id['dice'], 'Matched Dice-only reference', HERE / 'progression_live_20261012.json')
for campaign, protocol_path, results_path in sources:
    protocol = read(protocol_path)
    assert protocol['splits']['5pct'] == base_protocol['splits']['5pct']
    for key in ('fold', 'seed', 'max_epochs', 'step_size', 'lr_gamma', 'learning_rate',
                'early_stopping', 'model_parameters', 'auxiliary_branch'):
        assert protocol[key] == base_protocol[key], (protocol_path, key)
    rows = read(results_path)['rows']
    lookup = {r['objective']: r for r in rows}
    for entry in protocol['runs']:
        if entry.get('training') or entry['id'] not in lookup:
            continue
        consider(lookup[entry['id']], campaign, results_path)

scope = read(HERE / 'asym32_scope.json')
for entry in scope['added_runs']:
    if entry['id'] in live_by_id:
        consider(live_by_id[entry['id']], 'Asymmetric-band combinations',
                 HERE / 'progression_live_20261012.json')

branch = read(ROOT / 'native_branch_constraints_20261008/results.json')['rows']
branch_best = max(branch, key=lambda r: r['macro_dice'])
branch_score = 100 * branch_best['macro_dice']
assert len(records) == 21
assert all(a['dice_percent'] < b['dice_percent'] for a, b in zip(records, records[1:]))
assert abs(records[-1]['dice_percent'] - 80.72518839502469) < 1e-10
summary = dict(metric='Validation anterior/posterior macro Dice (%)', fold=0, seed=0,
               training_cases=10, validation_cases=52, max_epochs=75,
               ordering='Matched Dice reference, then new records in archived campaign execution order; repeated controls omitted.',
               interpretation='Exploratory records on one reused validation split. Successive records are not cumulative additions of every earlier loss.',
               records=records, branch_reference=dict(objective=branch_best['objective'],
                   dice_percent=branch_score, separate_architecture=True),
               gain_over_dice_pp=best-records[0]['dice_percent'],
               gain_over_original_bands_pp=best-records[1]['dice_percent'],
               gap_to_branch_pp=branch_score-best)
(HERE / 'progression_records_20261012.json').write_text(json.dumps(summary, indent=2)+'\n')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(11, 6.2))
x = np.arange(1, len(records)+1)
y = np.array([r['dice_percent'] for r in records])
ax.plot(x, y, color='#176b87', lw=2, marker='o', ms=4)
ax.axhline(branch_score, color='#8c4f2d', lw=1.5, ls='--')
ax.text(1, branch_score+0.13, f'Native hippocampus-only auxiliary branch: {branch_score:.3f}%',
        color='#62361f', fontsize=11)
labels = {
    1: ('Dice only\n74.304%', (15, 10), 'left'),
    2: ('Original bands\n76.727%', (13, -38), 'left'),
    8: ('Separate 1+3 mm targets\n78.132%', (-14, 34), 'right'),
    11: ('Hard-voxel focus\n78.937%', (2, -55), 'center'),
    15: ('3/2 mm bands\n79.263%', (10, -47), 'left'),
    21: ('60% inside / 15% outside\n80.725%', (-9, 18), 'right'),
}
for step, (label, offset, align) in labels.items():
    ax.annotate(label, (step, y[step-1]), xytext=offset, textcoords='offset points',
                ha=align, fontsize=10.5,
                arrowprops=dict(arrowstyle='-', color='#858585', lw=.7))
ax.set_xlim(.5, 21.7)
ax.set_ylim(73.8, 82.9)
ax.set_xticks([1, 2, 5, 8, 11, 15, 18, 21])
ax.set_yticks(np.arange(74, 83, 1))
ax.set_xlabel('Record-setting variant (numbered in the table)')
ax.set_ylabel('Validation A/P Dice (%)')
ax.grid(axis='y', color='#dedede', lw=.6)
ax.set_axisbelow(True)
ax.set_title('Hippocampus segmentation: progression of validation records', loc='left', pad=18)
fig.text(.1, .035, '5% data · 10 training / 52 validation cases · fold 0 · seed 0 · max 75 epochs\n'
         'Each point is a completed variant; the sweep is ongoing. Records are exploratory, not independent confirmations.',
         fontsize=10, color='#444444')
fig.subplots_adjust(left=.1, right=.98, bottom=.2, top=.89)
for suffix in ('png', 'svg'):
    fig.savefig(HERE / f'progression_20261012.{suffix}', dpi=180)
plt.close(fig)
for row in records:
    delta='—' if row['gain_pp'] is None else f"+{row['gain_pp']:.3f}"
    print(f"| {row['step']} | {row['description']} | {row['dice_percent']:.3f}% | {delta} |")
print(json.dumps({k:summary[k] for k in ('gain_over_dice_pp','gain_over_original_bands_pp','gap_to_branch_pp')}))
