"""Read-only CPU audit of every packaged image, mask, and split."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.stats import ks_2samp

HERE=Path(__file__).resolve().parent
BUNDLE=HERE.parent/'geofwi3d_colab_seed_search_20261018/bundle'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=read(BUNDLE/'data/fold_0_manifest.json')
splits=read(BUNDLE/'splits.json')
ids=sorted({c for s in splits for c in s['train']+s['val']})
rows=[]
for name in ids:
    c=manifest['cases'][name]
    ip,lp=BUNDLE/'data'/c['image'], BUNDLE/'data'/c['label']
    assert sha(ip)==c['image_sha256'] and sha(lp)==c['label_sha256'],name
    ni,nl=nib.load(ip),nib.load(lp)
    x=np.asarray(ni.dataobj,dtype=np.float32);y=np.asarray(nl.dataobj)
    assert x.shape==y.shape==(96,96,96),name
    assert np.array_equal(ni.affine,nl.affine),name
    assert np.isfinite(x).all() and x.min()>=-1 and x.max()<=1,name
    assert set(np.unique(y))=={0,1},name
    mask=y==1;count=int(mask.sum());assert count==c['salt_voxels'],name
    rows.append(dict(case=name,salt_voxels=count,salt_fraction=count/y.size,
        mean=float(x.mean()),std=float(x.std()),clip_fraction=float((np.abs(x)>=1).mean()),
        salt_abs_mean=float(np.abs(x[mask]).mean()),background_abs_mean=float(np.abs(x[~mask]).mean()),
        centroid_z=float(np.dot(mask.sum(axis=(0,1)),np.arange(96))/count),
        fault_present=c['fault_present'],body_count=c['generated_salt_body_count'],
        normalization_scale=c['normalization_abs_percentile_99_5']))
    if len(rows)%50==0:print('Verified',len(rows),'of',len(ids),flush=True)
by={r['case']:r for r in rows}; comparisons=[]
for fold,s in enumerate(splits):
    tr,va=set(s['train']),set(s['val'])
    assert len(tr)==19 and len(va)==96 and not tr&va
    assert not {manifest['cases'][c]['seed'] for c in tr}&{manifest['cases'][c]['seed'] for c in va}
    item={'fold':fold,'train_count':len(tr),'val_count':len(va),'features':{}}
    for field in rows[0]:
        if field=='case':continue
        a=np.array([by[c][field] for c in sorted(tr)]);b=np.array([by[c][field] for c in sorted(va)])
        k=ks_2samp(a,b)
        item['features'][field]=dict(train_mean=float(a.mean()),val_mean=float(b.mean()),
            train_range=[float(a.min()),float(a.max())],val_range=[float(b.min()),float(b.max())],
            ks_statistic=float(k.statistic),ks_pvalue=float(k.pvalue))
    comparisons.append(item)
assert len({c for s in splits for c in s['val']})==288
for field in ('image_sha256','label_sha256'):
    assert len({manifest['cases'][c][field] for c in ids})==len(ids),field
with (HERE/'data_case_features.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
report=dict(cases=len(ids),file_hashes_verified=2*len(ids),
    shapes_finite_binary_masks_and_matching_affines=True,mask_counts_match_manifest=True,
    no_duplicate_images_or_masks=True,no_train_val_case_or_simulation_seed_overlap=True,
    validation_folds_disjoint=True,comparisons=comparisons,
    limitation='Raw rgt3d/vp3d archives are not in this bundle; physical label derivation verified against recorded provenance and manifest counts, not regenerated.')
(HERE/'data_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
