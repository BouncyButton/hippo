"""Read-only metadata audit of the protected paused Swin baseline checkpoint."""
import json
from pathlib import Path
import subprocess

from scripts.launch_unetr_subfields import remote
from scripts.cross_dataset_campaign.common import REMOTE, write

NAME='mni_common0519_shared53t_3fold_20261108'
CODE='''import json,sys,hashlib,torch
from pathlib import Path
p=Path(sys.argv[1]);read=lambda f:json.loads(f.read_text());assert read(p/'control.json')['pause']
identifier='MNI_swinunetr_dice_f1s19';run=p/'runs'/identifier
cp=run/'checkpoint_latest.pt';sha=hashlib.sha256(cp.read_bytes()).hexdigest()
assert sha=='892147f16c4da309e50069af8f5370dbe7cd5cc53a13a93c435cd2faf795de4e'
checkpoint=torch.load(cp,map_location='cpu',weights_only=False);cfg=read(run/'config.json')
assert checkpoint['configuration']==cfg
task=next(t for t in read(p/'plan.json')['tasks'] if t['id']==identifier)
folder=p/'protocols'/task['protocol'];protocol=read(folder/'protocol.json')
assert hashlib.sha256((folder/'protocol.json').read_bytes()).hexdigest()==cfg['protocol_sha256']
missing=not (run/'budget_ledger.json').exists()
if missing:
 assert checkpoint['step']==checkpoint['charged_attempts']==0
 ledger=dict(charged_attempts=0,training_flops=0,calibration_flops=0,synthetic_zero_only=True)
else:ledger=read(run/'budget_ledger.json')
core_sha=hashlib.sha256((p/'source/scripts/cross_dataset_campaign/capped_core.py').read_bytes()).hexdigest()
assert core_sha==cfg['script_sha256']
print(json.dumps(dict(status='read_only_paused_recovery_verified',task=task,configuration=cfg,
 checkpoint_sha256=sha,step=checkpoint['step'],charged_attempts=checkpoint['charged_attempts'],
 ledger=ledger,missing_ledger_at_zero_updates=missing,checkpoint_keys=list(checkpoint),
 protocol_dir=str(folder),protocol_data=protocol['data'],core_sha256=core_sha,
 files=[dict(name=f.name,bytes=f.stat().st_size) for f in run.iterdir() if f.is_file()])))'''


def main():
    try:
        result=json.loads(remote(['/mnt/beegfsstudents/home/3160552/.conda/envs/hippocampus/bin/python',
                                  '-c',CODE,REMOTE/NAME],timeout=300))
    except subprocess.CalledProcessError as error:
        print(error.stderr)
        raise
    write(Path('reports/mni_unetr_constraint_goal_20261108/paused_swin_recovery_metadata.json'),result)
    print(json.dumps({k:result[k] for k in ('status','step','charged_attempts','ledger','missing_ledger_at_zero_updates','core_sha256')}))


if __name__=='__main__':
    main()
