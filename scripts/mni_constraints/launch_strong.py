"""Deploy target60 pilots and move only abandoned recovery artifacts for retirement."""
import argparse
import json
import subprocess
import time
from scripts.launch_unetr_subfields import remote
from scripts.cross_dataset_campaign.common import read,write,REMOTE
from scripts.mni_constraints.strong_pilot import ROOT,NAME,MODULE,PARENT,verify
COLLECTOR_MODULE='scripts.cross_dataset_campaign.collect'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage-only',action='store_true');parser.add_argument('--submit-only',action='store_true')
    args=parser.parse_args();assert not(args.stage_only and args.submit_only)
    stage=ROOT/'experiments'/NAME;dest=REMOTE/NAME;report=ROOT/'reports'/NAME
    plan=verify(stage);report.mkdir(parents=True,exist_ok=True)
    if (report/'submission.json').exists():raise ValueError('Already submitted')
    if remote(['squeue','-h','-u','3160552']).strip():raise ValueError('Wait for old allocation to leave queue')
    assert json.loads(remote(['cat',REMOTE/PARENT/'control.json']))['pause']
    env=':'.join(str(dest/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
    command=['env','PYTHONPATH='+env,plan['conventional_python'],'-m',MODULE]
    if not args.submit_only:
        remote(['mkdir','-p',dest])
        subprocess.run(['rsync','-az','--exclude=__pycache__',str(stage)+'/',f'bocconi-cluster:{dest}/'],check=True)
        print(remote(command+['preflight','--stage',dest],timeout=900),flush=True)
        code=r'''import json,sys,hashlib,shutil,os,time
from pathlib import Path
p=Path(sys.argv[1]);plan=json.loads((p/'plan.json').read_text());parent=Path(plan['parent_stage'])
recovery=Path(plan.get('recovery_stage',str(parent)))
assert json.loads((parent/'control.json').read_text())['pause']
def digest(f):
 h=hashlib.sha256()
 with f.open('rb') as s:
  for block in iter(lambda:s.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
receipts=[]
for item in plan['retirements']:
 src=recovery/'runs'/(item['id'] if 'recovery_stage' in plan else item['original_run']);dst=p/'runs'/item['id']
 assert src.is_dir() and not src.is_symlink()
 assert not (src/'completion.json').exists() and not (src/'cleanup.json').exists()
 dst.mkdir(parents=True,exist_ok=True)
 for name,info in item['files'].items():
  assert Path(name).name==name
  f=src/name if (src/name).exists() else dst/name
  assert f.is_file() and not f.is_symlink() and f.stat().st_size==info['bytes'] and digest(f)==info['sha256'],name
 for name,info in item['files'].items():
  source,target=src/name,dst/name
  if target.exists():assert digest(target)==info['sha256'];continue
  if source.suffix in ('.pt','.tmp'):os.rename(source,target)
  else:shutil.copy2(source,target)
  assert digest(target)==info['sha256'],name
 previous=json.loads((src/'migration.json').read_text()) if (src/'migration.json').exists() else None
 receipt=dict(status='moved_for_evaluation_and_verified_archive',original=str(src),destination=str(dst),time=time.time(),files=item['files'],previous_migration=previous)
 (src/'migration.json').write_text(json.dumps(receipt,indent=2)+'\n');(dst/'migration.json').write_text(json.dumps(receipt,indent=2)+'\n')
 receipts.append(receipt)
(p/'migration.json').write_text(json.dumps(receipts,indent=2)+'\n')
print(json.dumps(dict(migrated=len(receipts),bulky_files_remaining_in_old_stage=[str(f) for f in (parent/'runs').rglob('*.pt')])) )'''
        print(remote(['python3','-c',code,dest],timeout=600),flush=True)
    write(report/'preflight.json',json.loads(remote(['cat',dest/'preflight.json'])))
    write(report/'migration.json',json.loads(remote(['cat',dest/'migration.json'])))
    if args.stage_only:
        print('Staged and validated; abandoned checkpoints preserved for evaluation; no job submitted.',flush=True);return
    with (report/'collector.log').open('a') as log:
        collector_args=(['--stage',str(dest),'--local-stage',str(stage),'--destination',str(report)]
                        if COLLECTOR_MODULE=='scripts.cross_dataset_campaign.collect' else [])
        process=subprocess.Popen(['caffeinate','-i',str(ROOT/'.venv/bin/python'),'-u','-m',COLLECTOR_MODULE,*collector_args],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(report/'collector_process.json',dict(pid=process.pid,time=time.time()))
    receipt=json.loads(remote(command+['submit','--stage',dest]));write(report/'submission.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
