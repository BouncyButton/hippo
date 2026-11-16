"""Cross-check diagnostic NIfTI inputs against the underlying campaign arrays."""
import zipfile
import hashlib
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from run import ROOT, OUT, HFH, recipe

inputs={
    'MSD':'reports/msd_colab_seeds0220_recovery_20261018/package/seed20_bundle/data/msd_hippocampus_full.pkl',
    'MNI':'reports/mni_winning_recipes_matrix2030_53t_0519_20261109/frozen/data/MNI.pkl',
    'ADNI':'datasets/Dataset103_ADNI/adni_hippocampus_full.pkl',
}
checked={}
for name, path in inputs.items():
    cases=set()
    for fold in range(3):
        protocol,split=recipe(name,fold)
        cases.update(split['train']+split['val'])
    raw=ROOT/path
    raw_hash=hashlib.sha256(raw.read_bytes()).hexdigest()
    matched_hash=(raw_hash==protocol.get('pkl_sha256'))
    folder=next((ROOT/'datasets').glob('Dataset*_'+name))
    n=0
    for _,row in pd.read_pickle(raw,compression='gzip').iterrows():
        c=('hippocampus_adni_'+row.image_id+'_'+row.direction) if name=='ADNI' else row.subject_id
        if c not in cases:
            continue
        x=np.asarray(nib.load(folder/'imagesTr'/(c+'_0000.nii.gz')).dataobj)
        y=np.asarray(nib.load(folder/'labelsTr'/(c+'.nii.gz')).dataobj)
        assert np.allclose(x,row.image_data) and np.array_equal(y,row['data' if name=='ADNI' else 'label_data']),c
        n+=1
    assert n==len(cases)
    checked[name]=dict(validated_native_image_mask_pairs=n,pickle=path,sha256=raw_hash,
                       exact_protocol_pickle_hash=matched_hash)
cases=set()
for fold in range(3):
    _,split=recipe('HFH',fold)
    cases.update(split['train']+split['val'])
index=json.loads((HFH.parent/'data_index.json').read_text())
checked['HFH']=dict(direct_archived_crops=len(cases),source=str(HFH.relative_to(ROOT)),case_sha256={
    c:hashlib.sha256((HFH/(c+'.npz')).read_bytes()).hexdigest() for c in sorted(cases)})
assert cases <= set(index['cases'])
archive=ROOT/'reports/full_data_baselines_20261115/matched_controls/HFH_matched.zip'
with zipfile.ZipFile(archive) as z:
    expected=json.loads(z.read('hfh_data_sha256.json'))
assert all(expected[c+'.npz']==h for c,h in checked['HFH']['case_sha256'].items())
checked['HFH']['verified_against_archive']=str(archive.relative_to(ROOT))
(OUT/'input_verification.json').write_text(json.dumps(checked,indent=2)+'\n')
print({k:v.get('validated_native_image_mask_pairs',v.get('direct_archived_crops')) for k,v in checked.items()})
