"""Check all twenty published baseline blocks and quantify foreground/background label confusion."""
import csv
import json
from pathlib import Path
from statistics import mean
import sys
import zipfile
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
with zipfile.ZipFile(sys.argv[1]) as z:
    original=json.loads(z.read('data/results.json'))['rows']
    extension=json.loads(z.read('data/hfh_mni_extension.json'))['rows']
hfh_med=pd.read_csv(HERE/'archived_hfh_medsam3_nine_controls.csv')
a100=pd.read_csv(ROOT/'reports/hfh_nnunet_matrix_reconciliation_20261114/final_a100_matrix_126.csv')
a100=a100[a100.objective=='dice'].set_index(['fold','seed'])
records=[]
for r in original+extension:
    if r['arm']!='baseline' or r['study']=='MSD_MedSAM3_original':
        continue
    ds,model=r['dataset'],r['model'].lower()
    dice,reason=r['dice'],r['study']
    if ds=='MNI' and model=='medsam3':
        name=r['source_id']
        f=ROOT/'reports/mni_medsam3_n5_adaptive60_2gpu_20261110/archive/runs'/name/'evaluation.json'
        if not f.exists():
            f=ROOT/'reports/mni_medsam3_n5_third_seed_search_20261110/archive/candidates/candidates/seed2/baseline/runs'/name/'evaluation.json'
        val=[v for v in json.loads(f.read_text())['records'] if v['split']=='validation']
        assert len(val)==10 and abs(100*mean(v['macro_hard_dice'] for v in val)-dice)<1e-7
        c=np.sum([v['confusion'] for v in val],axis=0)
        ext=c[0,1:].sum()+c[1:,0].sum()
        frac=100*ext/(c.sum()-c.trace())
        reason=str(f.relative_to(ROOT))
    elif ds in ['ADNI','HFH']:
        assert dice<100
        frac=100.
        if ds=='HFH' and model=='nnunet':
            assert abs(100*a100.loc[(r['fold'],r['seed']),'final_dice']-dice)<1e-7
            reason='final_a100_matrix_126.csv'
    else:
        ext=r['fn']+r['fp']
        frac=100*ext/(ext+r['swaps'])
    records.append(dict(dataset=ds,model=model,fold=r['fold'],seed=r['seed'],dice_pct=dice,external_error_pct=frac,source=reason))
for r in hfh_med.itertuples():
    records.append(dict(dataset='HFH',model='medsam3',fold=r.fold,seed=r.seed,dice_pct=r.dice_pct,
                        external_error_pct=100.,source=r.archive))
df=pd.DataFrame(records)
assert len(df)==180 and not df.duplicated(['dataset','model','fold','seed']).any()
df.to_csv(HERE/'original_baselines_180.csv',index=False)
groups=df.groupby(['model','dataset'])
assert groups.size().eq(9).all() and len(groups)==20
out=groups.agg(runs=('dice_pct','size'),dice_pct=('dice_pct','mean'),external_error_pct=('external_error_pct','mean'))
out.to_csv(HERE/'original_baselines_20_cells.csv')
print(out.external_error_pct.unstack().round(2))
