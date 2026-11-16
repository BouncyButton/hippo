"""Identify the nine exact HFH MedSAM3 controls and summarize their archived spatial counts."""
from pathlib import Path
import csv
import hashlib
import json
import statistics
import sys
import zipfile

HERE=Path(__file__).resolve().parent
HFH=HERE.parents[3]/'reports/hfh_colab_exploration_20261111'
MATRIX=HFH/'medsam_adaptive_matrix_replication/validation_dice_all_126.csv'
refs={(int(r['fold']),int(r['seed'])):r for r in csv.DictReader(MATRIX.open()) if r['objective']=='dice'}
rows={}
archives=list(HFH.glob('**/*__dice__f?_s?.zip'))+[Path(sys.argv[1])]
for p in archives:
    with zipfile.ZipFile(p) as z:
        if 'results/evaluation.json' not in z.namelist():
            continue
        config=json.loads(z.read('results/protocol.json'))
        ident=(config.get('fold'),config.get('seed'))
        if ident not in refs or config.get('updates')!=125:
            continue
        val=[c for c in json.loads(z.read('results/evaluation.json'))['records'] if c['split']=='validation']
        ref=refs[ident]
        if len(val)!=10 or any('spatial' not in c for c in val):
            continue
        dice=100*statistics.mean(c['spatial']['dice'] for c in val)
        if abs(dice-float(ref['validation_dice_pct']))>1e-7:
            continue
        values=[c['spatial'] for c in val]
        fp=sum(x['fp'] for x in values);fn=sum(x['fn'] for x in values)
        assert fp==int(ref['validation_fp_voxels']) and fn==int(ref['validation_fn_voxels'])
        edge=sum(x['errors_by_region']['outer']+x['errors_by_region']['inner'] for x in values)
        detached=sum(x['detached_component_fp'] for x in values)
        rows[ident]=dict(fold=ident[0],seed=ident[1],cases=len(val),dice_pct=dice,fp=fp,fn=fn,
                    band_errors=edge,band_share_of_all_errors_pct=100*edge/(fp+fn),
                    crops_with_detached_fp_component=sum(x['detached_component_fp']>0 for x in values),
                    detached_component_share_of_fp_pct=100*detached/fp,archive=p.name,
                    archive_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
assert len(rows)==9,rows.keys()
with (HERE/'archived_hfh_medsam3_nine_controls.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(next(iter(rows.values()))));w.writeheader();w.writerows(rows[k] for k in sorted(rows))
for key in ['dice_pct','band_share_of_all_errors_pct','crops_with_detached_fp_component','detached_component_share_of_fp_pct']:
    print(key,statistics.mean(r[key] for r in rows.values()))
