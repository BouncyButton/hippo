"""Recompute independent checks from archived predictions and per-case counts."""
import csv
import json
from pathlib import Path
import numpy as np
from run import OUT, ROOT, counts, recipe


def save(name, rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


archive=ROOT/'reports/mni_winning_recipes_matrix2030_53t_0519_20261109'
inventory=json.loads((archive/'final_results.json').read_text())['rows']
rows=[]
for entry in inventory:
    if entry['arm']!='dice':
        continue
    for stage in (25,100):
        path=archive/'frozen/reused_runs'/entry['id']/f'spatial_{stage:03d}.json'
        cases=[c for c in json.loads(path.read_text())['cases'] if c['split']=='validation']
        assert len(cases)==10
        total=outer=far=band_errors=disconnected=0
        for c in cases:
            for k,e in c['errors'].items():
                assert sum(e['distance_bins'].values())==e['total']
                total+=e['total']
                if k!='internal_swaps':
                    outer+=e['total']
                    band_errors+=e['original_band']
                    far+=sum(e['distance_bins'][b] for b in ['3','4','>4'])
            disconnected+=c['predicted_components']['disconnected_count']>0
        rows.append(dict(run=entry['id'],percent_budget=stage,crops=len(cases),errors=total,
                         external_errors=outer,external_in_original_band=band_errors,
                         external_beyond_two_face_steps=far,
                         cases_with_nonoverlapping_predicted_component=disconnected))
assert len(rows)==72
save('archive_mni.csv',rows)

base=ROOT/'reports/thesis_qualitative_predictions_20261103/recovered'
rows=[]
for name,dirname,fold in [('MSD','MSD_MedSAM3_baseline_f0s83',0),
                          ('ADNI','ADNI_MedSAM3_2sample_dice_f1s2',1)]:
    protocol,_=recipe(name,fold)
    for p in sorted((base/dirname/'predictions/validation').glob('*.npz')):
        with np.load(p) as payload:
            a=payload['affine']
            if a.shape!=(4,4):
                a=np.eye(4)  # ADNI archived identity-grid convention
            rows.append(dict(dataset=name,case=p.stem,source=dirname,**counts(payload['truth'],
                             payload['prediction'],a,protocol['num_classes'])))
save('archive_medsam.csv',rows)
