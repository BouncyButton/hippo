"""Compact live check for the finite target search and abandoned-run cleanup."""
import json

from scripts.launch_unetr_subfields import remote
from scripts.cross_dataset_campaign.common import REMOTE, write
from scripts.mni_constraints.target_search import NAME, ROOT


def main():
    code = r'''import json,subprocess,sys,time
from pathlib import Path
from scripts.cross_dataset_campaign.storage import quota
p=Path(sys.argv[1]);d={'checked_at':time.time(),'queue':subprocess.check_output(['squeue','-h','-u','3160552','-o','%i|%T|%M|%R'],text=True).splitlines(),'quota':quota()}
for name in ('status','failure','retirement_complete','successor'):
 f=p/(name+'.json')
 if f.exists():
  v=json.loads(f.read_text());d[name]={k:w for k,w in v.items() if k!='runs'}
job=json.loads((p/'submission.json').read_text())['job_id']
f=p/('slurm-'+job+'.err');d['current_errors']='\n'.join(line for line in f.read_text().splitlines() if not line.startswith('INFO: Quota usage information'))[-4000:] if f.exists() else ''
d['runs']=[]
for r in sorted((p/'runs').glob('*')):
 row={'id':r.name,'checkpoints':[f.name for f in r.glob('*.pt')]}
 for name in ('budget_ledger','calibration','cleanup','abandonment'):
  f=r/(name+'.json')
  if f.exists():
   v=json.loads(f.read_text());row[name]={k:v[k] for k in ('status','charged_attempts','weights','released_file_bytes','additional_optimizer_updates','completed_updates') if k in v}
 row['evaluations']=[f.name for f in r.glob('*interrupted.json')]
 f=r/'train.log'
 if f.exists() and not r.name.startswith('abandoned_'):row['log_tail']=f.read_text()[-800:]
 if not (r/'cleanup.json').exists() or r.name.startswith('abandoned_'):d['runs'].append(row)
print(json.dumps(d))'''
    stage = REMOTE / NAME
    env = ':'.join(str(stage / p) for p in ('source', 'source/scripts', 'source/baselines/nnUNet'))
    result = json.loads(remote(['env', 'PYTHONPATH=' + env, 'python3', '-c', code, stage]))
    write(ROOT / 'reports' / NAME / 'latest_search_check.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
