"""Prepare, verify and queue the independent 20:30 matrix handoff."""
import argparse
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import time

from scripts.cross_dataset_campaign.common import read,write,digest,REMOTE
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.launch_unetr_subfields import remote
from scripts.mni_constraints import winning_matrix2030 as study

ROOT=Path(__file__).resolve().parents[2]
BUNDLE_NAME='mni_matrix2030_handoff_bundle_20261109'
BUNDLE=ROOT/'experiments'/BUNDLE_NAME
DEST=REMOTE/BUNDLE_NAME
REPORT=ROOT/'reports'/study.NAME


def prepare():
    assert not (REPORT/'timed_submission.json').exists()
    BUNDLE.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    confirmation='mni_unet_delayed10_confirm53t_0519_20261109'
    evidence=read(ROOT/'reports'/confirmation/'confirmation_results.json')
    assert evidence['confirmation_passed'] and len(evidence['rows'])==9
    rows=[]
    for r in evidence['rows']:
        d=Path(r['archive']);verify_archive(d,read(d/'archive_manifest.json'))
        assert r['gain_pp']>0 and digest(d/'archive_manifest.json')==r['archive_sha256']
        relative=d.relative_to(ROOT/'experiments') if d.is_relative_to(ROOT/'experiments') else d.relative_to(ROOT/'reports')
        rows.append(dict(seed=r['seed'],fold=r['fold'],gain_pp=r['gain_pp'],source=str(REMOTE/relative),archive_sha256=r['archive_sha256']))
    protocol=ROOT/'experiments'/confirmation/'protocols/late10'
    parent=ROOT/'experiments'/study.PARENT
    spec=dict(start_at=study.START,parent=str(REMOTE/study.PARENT),stage=str(REMOTE/study.NAME),
        parent_manifest_sha256=digest(parent/'manifest.json'),budget_flops=study.CAP,
        python=read(parent/'plan.json')['conventional_python'],local_report=str(REPORT),
        recovery=read(REPORT/'recovery_audit.json'),
        unet_confirmation=dict(candidate='late10',protocol=str(REMOTE/confirmation/'protocols/late10'),
            protocol_sha256=digest(protocol/'protocol.json'),rows=rows))
    write(BUNDLE/'spec.json',spec)
    control=read(BUNDLE/'control.json') if (BUNDLE/'control.json').exists() else dict(pause=False,refinement_stages=[str(REMOTE/'mni_nnunet_swin_refine2030_round2_53t_20261109')],refinement_queue=[])
    write(BUNDLE/'control.json',control)
    for n in ('winning_matrix2030.py','winning_matrix_runner.py','matrix2030_dispatch.py'):
        shutil.copy2(ROOT/'scripts/mni_constraints'/n,BUNDLE/n)
    script=(parent/'run.sbatch').read_text().replace('scripts.mni_constraints.verified_lower_lr_full_matrix',study.MODULE).replace('hippo-verified53','hippo-winning53')
    (BUNDLE/'matrix_run.sbatch').write_text(script)
    # The dispatch starts from the verified parent's runtime, then execs the new frozen source.
    dispatch=script.replace('campaign="${1:?Pass the frozen campaign directory}"','campaign="'+str(REMOTE/study.PARENT)+'"')
    dispatch=dispatch.replace('"$runtime" -m '+study.MODULE+' run --stage "$campaign" &',
        '"$runtime" "${1:?Pass bundle}/matrix2030_dispatch.py" "$1" "${2:?Pass manifest hash}" &')
    assert 'matrix2030_dispatch.py' in dispatch
    (BUNDLE/'dispatch.sbatch').write_text(dispatch.replace('hippo-winning53','hippo-2030-handoff'))
    write(BUNDLE/'manifest.json',{f.name:digest(f) for f in sorted(BUNDLE.iterdir()) if f.is_file() and f.name not in ('manifest.json','control.json')})
    shutil.copytree(BUNDLE,REPORT/'handoff_bundle',dirs_exist_ok=True)
    print(json.dumps(dict(bundle=str(BUNDLE),manifest_sha256=digest(BUNDLE/'manifest.json'))))


def deploy():
    assert not (REPORT/'timed_submission.json').exists()
    remote(['python3','-c','from pathlib import Path;import sys;p=Path(sys.argv[1]);assert not (p/"timed_submission.json").exists();p.mkdir(exist_ok=True)',DEST])
    blob=io.BytesIO()
    with tarfile.open(fileobj=blob,mode='w:gz') as tar:
        for f in BUNDLE.iterdir():tar.add(f,arcname=f.name,recursive=False)
    subprocess.run(['ssh','-o','BatchMode=yes','bocconi-cluster',shlex.join(['tar','-xzf','-','-C',str(DEST)])],input=blob.getvalue(),check=True,timeout=90)
    code='''import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);assert hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest()==sys.argv[2]
for n,sha in json.loads((p/'manifest.json').read_text()).items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==sha,n
print(json.dumps(dict(status='verified',manifest_sha256=sys.argv[2])))'''
    result=json.loads(remote(['python3','-c',code,DEST,digest(BUNDLE/'manifest.json')]))
    write(REPORT/'handoff_deployment.json',result);print(json.dumps(result))


def preflight():
    code='''import importlib.util,json,os,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
bundle=Path(sys.argv[1]);loading=importlib.util.spec_from_file_location('builder',bundle/'winning_matrix2030.py');m=importlib.util.module_from_spec(loading);loading.loader.exec_module(m)
assert m.digest(bundle/'manifest.json')==sys.argv[2]
m.NAME += '_preflight_'+sys.argv[2][:8];stage=m.REMOTE/m.NAME;m.time=SimpleNamespace(time=lambda:m.START+1)
plan=m.build(bundle,stage)
env={**os.environ,'PYTHONPATH':':'.join(str(stage/p) for p in ('source','source/scripts','source/baselines/nnUNet')),'MPLCONFIGDIR':'/tmp/hippo-handoff-preflight','MPLBACKEND':'Agg'}
python=m.read(bundle/'spec.json')['python']
result=subprocess.run([python,'-m',m.MODULE,'preflight','--stage',str(stage)],cwd=stage/'source',env=env,text=True,capture_output=True)
if result.returncode:print(result.stdout,result.stderr);raise SystemExit(result.returncode)
r=dict(status='passed',bundle_manifest_sha256=m.digest(bundle/'manifest.json'),stage=str(stage),preflight=m.read(stage/'preflight.json'),manifest=m.read(stage/'manifest.json'),plan=plan,recovery=m.read(stage/'recovery.json'),selection=m.read(stage/'selection.json'))
m.write(bundle/'handoff_preflight.json',r);print(json.dumps(r))'''
    parent=REMOTE/study.PARENT;env=':'.join(str(parent/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
    r=json.loads(remote(['env','PYTHONPATH='+env,'MPLCONFIGDIR=/tmp/hippo-handoff-preflight',read(BUNDLE/'spec.json')['python'],'-c',code,DEST,digest(BUNDLE/'manifest.json')],timeout=1200))
    write(REPORT/'handoff_preflight.json',r)
    print(json.dumps(dict(status=r['status'],reused=r['plan']['reused_runs'],remaining=r['plan']['num_runs'],stage=r['stage'],checks=len(r['preflight']['actual_calibration_checks']))))


def submit():
    assert read(REPORT/'handoff_preflight.json')['status']=='passed'
    assert read(REPORT/'handoff_preflight.json')['bundle_manifest_sha256']==digest(BUNDLE/'manifest.json')
    assert not (REPORT/'timed_submission.json').exists()
    code='''import json,sys,subprocess,time,hashlib,fcntl
from pathlib import Path
p=Path(sys.argv[1]);sha=sys.argv[2];read=lambda f:json.loads(f.read_text());spec=read(p/'spec.json')
sys.path.insert(0,str(Path(spec['parent'])/'source'))
from scripts.cross_dataset_campaign.storage import quota
assert hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest()==sha
assert time.time()<spec['start_at'] and not read(p/'control.json')['pause']
assert read(Path(spec['parent'])/'control.json')['pause']
with (p/'submission.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (p/'timed_submission.json').exists()
 q=subprocess.check_output(['squeue','-h','-u','3160552','-o','%i|%T'],text=True).strip().splitlines()
 assert len(q)<=1 and all(r.split('|')[1] in ('RUNNING','COMPLETING','CONFIGURING') for r in q),q
 usage=quota();assert usage['free_gib']>=24,usage
 assert read(p/'control.json').get('refinement_queue'),'Prepare the next refinement before queuing the combined handoff.'
 command=['sbatch','--parsable','--chdir='+str(p),'--output='+str(p/'slurm-%j.out'),'--error='+str(p/'slurm-%j.err')]
 if q:command+=['--dependency=afterany:'+q[0].split('|')[0]]
 command += [str(p/'dispatch.sbatch'),str(p),sha]
 job=subprocess.check_output(command,text=True).strip().split(';')[0];assert job.isdigit()
 r=dict(job_id=job,scheduled_at=spec['start_at'],submitted_at=time.time(),command=command,bundle_manifest_sha256=sha,laptop_required=False,quota=usage)
 (p/'timed_submission.json').write_text(json.dumps(r));print(json.dumps(r))'''
    result=json.loads(remote(['python3','-c',code,DEST,digest(BUNDLE/'manifest.json')],timeout=90))
    write(REPORT/'timed_submission.json',result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','deploy','preflight','submit']);args=p.parse_args()
    try:{'prepare':prepare,'deploy':deploy,'preflight':preflight,'submit':submit}[args.action]()
    except subprocess.CalledProcessError as error:print(error.output,error.stderr,flush=True);raise
