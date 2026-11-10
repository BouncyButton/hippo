"""Bind reusable frozen inputs and record their content before submission."""
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    root=Path(__file__).resolve().parents[3]
    p=json.loads((root/'protocol.json').read_text());prior=Path(p['prior_root'])
    assert root.name==p['name'] and os.getuid()==1395
    if (root/'submission.json').exists() or (root/'submission_attempt.json').exists():
        raise RuntimeError('Cannot restage after a submission attempt')
    assert json.loads((prior/'status.json').read_text())['state']=='complete'
    old_protocol=prior/'source/docs/experiments/unwrapped_swin_20261006/protocol.json'
    assert sha(old_protocol)==p['prior_protocol_sha256']
    files={str(old_protocol):sha(old_protocol),str(prior/'comparison.json'):sha(prior/'comparison.json')}
    for folder in ('deps','atlas','loss_geometry'):
        destination=root/folder;target=(prior/folder).resolve()
        assert target.is_dir()
        if destination.is_symlink():assert destination.resolve()==target
        elif destination.exists():raise ValueError('Expected a shared input link: '+str(destination))
        else:destination.symlink_to(target,target_is_directory=True)
    for fraction,arms in p['pairs'].items():
        manifest_path=prior/'atlas'/fraction/'manifest.json'
        manifest=json.loads(manifest_path.read_text())
        assert manifest['complete'] and manifest['protocol_sha256']==p['atlas_protocol_sha256']
        assert set(manifest['cases'])==set(p['splits'][fraction]['train']+p['splits'][fraction]['val'])
        files[str(manifest_path)]=sha(manifest_path)
        for name,expected in manifest['artifacts'].items():
            path=manifest_path.parent/name
            assert sha(path)==expected
            files[str(path)]=expected
        for arm in ('dice','depth',*arms):
            run=prior/'runs'/fraction/arm;record_path=run/'completion.json'
            record=json.loads(record_path.read_text());files[str(record_path)]=sha(record_path)
            for name,expected in record['artifacts'].items():
                if name in ('validation_best.json','metrics.csv','config.json','validation_predictions.npz') or (name=='best.pt' and arm in arms):
                    path=run/name
                    assert sha(path)==expected
                    files[str(path)]=expected
    for path in (prior/'loss_geometry').glob('*.npz'):files[str(path)]=sha(path)
    record=dict(prior_root=str(prior),files=files,geometry_reused=True,completed_training_repeated=False)
    (root/'reference_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(verified_reference_files=len(files),prior_root=str(prior),shared_inputs=['deps','atlas','loss_geometry'])))


if __name__=='__main__':main()
