"""Bound new-campaign checkpoint storage, preserving every result and prediction."""
import json
import re
import subprocess
import time
from scripts.depth_supervision.utils import atomic_json, digest
from .train import completed_record, identity


def quota_free():
    value=subprocess.check_output(['beegfs','quota','list-usage','--uids','current','--pool','pool_home'],text=True)
    match=re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB',value)
    if not match:raise RuntimeError('Cannot verify quota')
    return float(match[2])-float(match[1])


def retain_checkpoints(p,root,results):
    if root.name != p['name'] or root.is_symlink():raise ValueError('Unexpected run root')
    controls={e['id'] for e in p['runs'] if e['family']=='control'}
    top={r['objective'] for r in sorted(results['rows'],key=lambda r:(-r['macro_dice'],r['objective']))[:5]}
    keep=controls|top
    for entry in p['runs']:
        folder=root/'runs'/entry['id']
        record=completed_record(folder,identity(p,entry))
        if record is None or entry['id'] in keep:continue
        path=folder/'best.pt'
        if not path.exists():continue
        if path.is_symlink():raise ValueError('Unexpected checkpoint symlink')
        if 'best.pt' in record['artifacts']:
            artifact={'best.pt':record['artifacts'].pop('best.pt')}
            if digest(path)!=artifact['best.pt']:raise ValueError('Checkpoint changed')
            # Commit metadata before deletion, so a crash only leaves an extra file.
            atomic_json(folder/'checkpoint_retired.json',dict(identity=record['identity'],artifacts=artifact,
                reason='Outside fixed controls and current top five; all predictions/results retained',unix=time.time()))
            record['retired_artifacts']=artifact
            atomic_json(folder/'completion.json',record)
        path.unlink()
    atomic_json(root/'checkpoint_retention.json',dict(retained=sorted(keep),completed=results['completed'],
        policy='Three fixed controls plus top five by validation A/P Dice; shared warmup and active resume checkpoint. All metrics, logs, configuration and predictions retained.'))
