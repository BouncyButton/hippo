"""Validate a reusable nnU-Net preprocessing cache without modifying it."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_cases(raw, splits, fold):
    split = json.loads(splits.read_text())[fold]
    assert len(split['train']) == 208 and len(split['val']) == 52
    assert not set(split['train']) & set(split['val'])
    expected = set(split['train']) | set(split['val'])
    assert expected == {p.name.removesuffix('.nii.gz') for p in (raw/'labelsTr').glob('*.nii.gz') if not p.name.startswith('._')}
    return expected


def verify(cache, raw, splits, fold):
    import numpy as np
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
    expected = expected_cases(raw,splits,fold)
    plans=json.loads((cache/'nnUNetPlans.json').read_text())
    cfg=plans['configurations']['3d_fullres']
    data=cache/cfg['data_identifier']
    assert json.loads((cache/'dataset.json').read_text()) == json.loads((raw/'dataset.json').read_text())
    reader=nnUNetDatasetBlosc2(str(data))
    assert set(reader.identifiers)==expected, 'Partial or wrong cache'
    hashes={}
    for name in sorted(expected):
        for suffix in ('.b2nd','_seg.b2nd','.pkl'):
            p=data/(name+suffix)
            hashes[str(p.relative_to(cache))]=sha(p)
        gt=cache/'gt_segmentations'/(name+'.nii.gz')
        assert sha(gt)==sha(raw/'labelsTr'/gt.name), name
        hashes[str(gt.relative_to(cache))]=sha(gt)
        image,seg,prev,props=reader.load_case(name)
        image,seg=np.asarray(image[:]),np.asarray(seg[:])
        assert image.shape[1:]==seg.shape[1:] and np.isfinite(image).all()
        assert set(np.unique(seg)).issubset({-1,0,1,2})
        assert (seg==1).any() and (seg==2).any()
    for name in ('nnUNetPlans.json','dataset.json','dataset_fingerprint.json'):
        hashes[name]=sha(cache/name)
    return dict(status='verified',cases=len(expected),fold=fold,train_cases=208,validation_cases=52,
                cache=str(cache),raw=str(raw),splits_sha256=sha(splits),files=hashes,
                patch_size=cfg['patch_size'],batch_size=cfg['batch_size'])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--splits',type=Path,required=True)
    p.add_argument('--fold',type=int,default=0)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=verify(a.cache,a.raw,a.splits,a.fold)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files'}),flush=True)
