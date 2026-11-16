"""Export the original few-label partitions and the minimum necessary aligned inputs."""
from pathlib import Path
import json
import hashlib
import io
import zipfile
import sys
import nibabel as nib
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
INVENTORY=json.loads((ROOT/'reports/full_data_baselines_20261115/manifest.json').read_text())
MODELS=('nnunet','unetr_vit_b16','swinunetr','medsam3')
DATASETS=('MSD','MNI','ADNI','HFH')


def verified(desc):
    data=(ROOT/desc['file']).read_bytes()
    assert hashlib.sha256(data).hexdigest()==desc['sha256']
    return json.loads(data)


def build(target):
    jobs={}
    crops={ds:set() for ds in DATASETS}
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=2) as z:
        for ds in DATASETS:
            for model in MODELS:
                refs=[v for v in INVENTORY.values() if v['dataset']==ds and v['model']==model]
                for f in sorted({v['partition'] for v in refs}):
                    ref=next(v for v in refs if v['partition']==f)
                    cfg=verified(ref['archived_protocol'])
                    if model=='medsam3':
                        train,val=cfg['train_cases'],cfg['evaluation_cases']
                    else:
                        split=verified(ref['archived_validation_source'])[f]
                        train,val=split['train'],split['val']
                    assert train and val and not set(train)&set(val)
                    key=f'{ds}_{model}_f{f}'
                    jobs[key]=dict(dataset=ds,model=model,fold=f,seed=1741,train=train,val=val,
                                   recipe_sha256=ref['archived_protocol']['sha256'])
                    z.writestr('configs/'+key+'.json',json.dumps(cfg))
                    if model=='nnunet':
                        a=(ROOT/ref['archived_protocol']['file']).with_name('nnunet_architecture.json').read_bytes()
                        assert hashlib.sha256(a).hexdigest()==cfg['nnunet_architecture_sha256']
                        z.writestr('configs/'+key+'_arch.json',a)
                    crops[ds].update(train+val)
            for case in sorted(crops[ds]):
                if ds=='HFH':
                    f=ROOT/'reports/hfh_colab_exploration_20261111/baseline_v2/bundle/data'/(case+'.npz')
                    z.write(f,'data/'+ds+'/'+case+'.npz')
                    continue
                d=next((ROOT/'datasets').glob('Dataset*_'+ds))
                img=nib.load(d/'imagesTr'/(case+'_0000.nii.gz'))
                lab=nib.load(d/'labelsTr'/(case+'.nii.gz'))
                assert np.array_equal(img.affine,lab.affine)
                # Store the native scratch grid and the orientation transform for foundation inference.
                orient=nib.orientations.io_orientation(img.affine)
                buf=io.BytesIO()
                np.savez(buf,image=np.asanyarray(img.dataobj).astype('float32'),
                         label=np.asanyarray(lab.dataobj).astype('uint8'),orientation=orient,affine=img.affine)
                z.writestr('data/'+ds+'/'+case+'.npz',buf.getvalue())
        z.writestr('jobs.json',json.dumps(jobs,indent=2))
        (HERE/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
        for f in (HERE/'worker.py',HERE/'finish_remaining.py',HERE/'audit_hfh_augmentation.py',
                  HERE/'audit.sbatch',HERE.parent/'spatial_metrics.py'):
            z.write(f,f.name)
        z.write(ROOT/'scripts/depth_supervision/geometry.py','geometry_reference.py')
        assert len(jobs)==48
    print(len(jobs),'new runs;', {ds:len(c) for ds,c in crops.items()},target.stat().st_size,'bytes')


if __name__=='__main__':
    build(Path(sys.argv[1]))
