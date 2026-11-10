"""Read-only completion, storage, queue, and recovery audit after a UNETR stage."""
import argparse
import json

from scripts.launch_unetr_subfields import remote
from scripts.cross_dataset_campaign.common import REMOTE, read, write
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.unetr_subfields.contracts import ROOT


CODE = '''import json,sys,subprocess,hashlib
from pathlib import Path
from scripts.cross_dataset_campaign.storage import quota
p=Path(sys.argv[1]);read=lambda f:json.loads(f.read_text());state=read(p/'status.json')
assert state['status']=='complete';runs=[]
for folder in sorted((p/'runs').iterdir()):
 if not folder.is_dir():continue
 cleanup=read(folder/'cleanup.json');assert cleanup['status']=='complete'
 assert not list(folder.glob('*.pt')),folder.name
 runs.append(dict(id=folder.name,cleanup=cleanup))
assert len(runs)==state['completed']
old=p.parent/'mni_common0519_shared53t_3fold_20261108';assert read(old/'control.json')['pause']
recovery=old/'runs/MNI_swinunetr_dice_f1s19/checkpoint_latest.pt'
sha=hashlib.sha256(recovery.read_bytes()).hexdigest()
assert sha=='892147f16c4da309e50069af8f5370dbe7cd5cc53a13a93c435cd2faf795de4e'
queue=subprocess.check_output(['squeue','-h','-u','3160552','-o','%i|%T|%j'],text=True).strip()
print(json.dumps(dict(status='complete',stage_state=state,queue=queue,quota=quota(),runs=runs,
 broader_sweep_paused=True,swin_recovery_sha256=sha,
 jobs=subprocess.check_output(['sacct','-j',sys.argv[2],'--format=JobIDRaw,State,AllocTRES','-P','-n','-X'],text=True))))'''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',required=True)
    parser.add_argument('--jobs',required=True)
    parser.add_argument('--result-file',required=True)
    args=parser.parse_args()
    assert args.name.startswith('mni_unetr_') and '/' not in args.name
    report=ROOT/'reports'/args.name
    result=read(report/args.result_file);assert result['status']=='complete'
    stage=REMOTE/args.name
    env=':'.join(str(stage/part) for part in ('source','source/scripts','source/baselines/nnUNet'))
    audit=json.loads(remote(['env','PYTHONPATH='+env,'python3','-c',CODE,stage,args.jobs]))
    archived=result.get('archived_runs',result['rows'])
    assert {r['id'] for r in audit['runs']}=={r['id'] for r in archived}
    for row in audit['runs']:
        folder=report/'runs'/row['id']
        verify_archive(folder,read(folder/'archive_manifest.json'))
        assert read(folder/'cleanup.json')==row['cleanup']
    reuse_file=report/'frozen/reused_baselines.json'
    reused={r['id'] for r in read(reuse_file)} if reuse_file.exists() else set()
    assert reused<={r['id'] for r in audit['runs']}
    audit['reused_baselines']=sorted(reused)
    audit['newly_cleaned_runs']=len(audit['runs'])-len(reused)
    audit['newly_released_gib']=sum(r['cleanup']['released_file_bytes'] for r in audit['runs'] if r['id'] not in reused)/2**30
    write(report/'final_cluster_audit.json',audit)
    print(json.dumps(dict(queue=audit['queue'],quota=audit['quota'],cleaned=len(audit['runs']),
        reused_baselines=len(reused),newly_cleaned=audit['newly_cleaned_runs'],
        released_gib=audit['newly_released_gib'],jobs=audit['jobs'])))


if __name__=='__main__':main()
