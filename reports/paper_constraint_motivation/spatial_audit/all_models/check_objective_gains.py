"""Compute within-batch matched Dice comparisons for the paper's twenty cells."""
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
with ZipFile(sys.argv[1]) as archive:
    runs = pd.concat([
        pd.DataFrame(json.loads(archive.read(source))['rows'])
        for source in ('data/results.json', 'data/hfh_mni_extension.json')
    ], ignore_index=True)
runs['model'] = runs.model.str.lower()
hfh_med = pd.read_csv(
    ROOT / 'reports/hfh_colab_exploration_20261111/medsam_adaptive_matrix_replication/validation_dice_all_126.csv'
)
hfh_scratch = pd.read_csv(
    ROOT / 'reports/hfh_nnunet_matrix_reconciliation_20261114/final_a100_matrix_126.csv'
)
runs = runs.loc[~((runs.dataset == 'HFH') & (runs.model == 'nnunet'))]
for source, model, dice in (
    (hfh_med, 'medsam3', hfh_med.validation_dice_pct),
    (hfh_scratch, 'nnunet', 100 * hfh_scratch.final_dice),
):
    extra = source.assign(
        model=model, dataset='HFH', study='HFH_' + model,
        arm=source.objective.str.replace(r'^dice_', '', regex=True).replace({'dice': 'baseline'}),
        dice=dice,
    )
    runs = pd.concat([runs, extra], ignore_index=True)

rows = []
for (study, dataset, model), group in runs.groupby(['study', 'dataset', 'model']):
    wide = group.pivot(index=['fold', 'seed'], columns='arm', values='dice')
    for a, b in [('bands', 'baseline'), ('bands_edge', 'bands'), ('depth', 'baseline'), ('sdf', 'baseline')]:
        if a not in wide or b not in wide:
            continue
        diff = wide[a] - wide[b]
        assert len(diff) == 9 and diff.notna().all()
        rows.append(dict(model=model, dataset=dataset, study=study, comparison=a + '_minus_' + b,
                         matched_runs=9, mean_dice_gain_pp=diff.mean(), positive_pairs=int((diff > 0).sum())))
result = pd.DataFrame(rows)
assert len(result) == 80 and not result.duplicated(['model', 'dataset', 'comparison']).any()
result.to_csv(HERE / 'original_objective_checks_20_cells.csv', index=False)
print(result.groupby('comparison').mean_dice_gain_pp.agg(['mean',lambda values: int((values > 0).sum())]))
