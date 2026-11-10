"""Deploy the gated all-model matrix, with persistent verified archive cleanup."""
import json
import subprocess
import time

from scripts.launch_unetr_subfields import remote
from scripts.cross_dataset_campaign.common import read,write,REMOTE
from scripts.mni_constraints import verified_unetr_full_matrix as study
from scripts.mni_constraints.stage_seed5_remote import assemble


def main():
    stage=study.ROOT/'experiments'/study.NAME
    dest=REMOTE/study.NAME;report=study.ROOT/'reports'/study.NAME
    plan=study.verify(stage)
    report.mkdir(parents=True,exist_ok=True)
    assert not (report/'submission.json').exists(),'Already submitted; inspect existing job instead of relaunching'
    prior=REMOTE/study.confirmation.NAME
    oldenv=':'.join(str(prior/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
    code='''import json,sys,subprocess,hashlib
from pathlib import Path
from scripts.cross_dataset_campaign.storage import quota
p=Path(sys.argv[1]);read=lambda f:json.loads(f.read_text())
assert read(p/'status.json')['status']=='complete'
for folder in (p/'runs').iterdir():
 if folder.is_dir():
  assert read(folder/'cleanup.json')['status']=='complete' and not list(folder.glob('*.pt'))
old=p.parent/'mni_common0519_shared53t_3fold_20261108';assert read(old/'control.json')['pause']
cp=old/'runs/MNI_swinunetr_dice_f1s19/checkpoint_latest.pt'
assert hashlib.sha256(cp.read_bytes()).hexdigest()=='892147f16c4da309e50069af8f5370dbe7cd5cc53a13a93c435cd2faf795de4e'
assert not subprocess.check_output(['squeue','-h','-u','3160552'],text=True).strip()
q=quota();assert q['free_gib']>=16;print(json.dumps(q))'''
    write(report/'initial_quota.json',json.loads(remote(['env','PYTHONPATH='+oldenv,'python3','-c',code,prior])))
    remote(['python3','-c','import sys;from pathlib import Path;p=Path(sys.argv[1]);assert not (p/"submission.json").exists();p.mkdir(exist_ok=True)',dest])
    donors=read(stage/'deployment_known_donors.json')
    # This matrix inherits its confirmation stage, not the seed-sweep helper's
    # historical default parent. assemble verifies the donor against our manifest.
    donors['parent_manifest.json']=str(prior/'manifest.json')
    print(json.dumps(assemble(stage,dest,report,donors)),flush=True)
    env=':'.join(str(dest/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
    command=['env','PYTHONPATH='+env,plan['conventional_python'],'-m',study.MODULE]
    output=remote(command+['preflight','--stage',dest],timeout=900)
    write(report/'preflight.json',json.loads(output.strip().splitlines()[-1]))
    print('Remote actual-model/loss/LR preflight passed',flush=True)
    with (report/'collector.log').open('a') as log:
        process=subprocess.Popen(['caffeinate','-i',str(study.ROOT/'.venv/bin/python'),'-u','-m',
            'scripts.mni_constraints.collect_verified_matrix','--stage',str(dest),'--local-stage',str(stage),
            '--destination',str(report)],cwd=study.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(report/'collector_process.json',dict(pid=process.pid,started_at=time.time(),
        module='scripts.mni_constraints.collect_verified_matrix'))
    receipt=json.loads(remote(command+['submit','--stage',dest],timeout=900))
    write(report/'submission.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    try:main()
    except subprocess.CalledProcessError as error:
        print(error.stderr)
        raise
