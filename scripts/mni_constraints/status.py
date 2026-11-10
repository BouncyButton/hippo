"""One-shot status snapshot of the finite MNI study."""
import json
from scripts.launch_unetr_subfields import remote
from scripts.cross_dataset_campaign.common import write, REMOTE
from scripts.mni_constraints.campaign import NAME,ROOT


def main():
    code='''import json,sys,subprocess,time
from pathlib import Path
p=Path(sys.argv[1]);result={'checked_at':time.time(),'queue':subprocess.check_output(['squeue','-h','-u','3160552','-o','%i|%T|%j|%M|%R'],text=True).splitlines()}
for name in ('status','failure','successor'):
 f=p/(name+'.json')
 if f.exists():
  v=json.loads(f.read_text());result[name]={k:w for k,w in v.items() if k!='runs'}
result['runs']=[]
for d in sorted((p/'runs').glob('*')):
 row={'id':d.name,'remaining_checkpoints':[f.name for f in d.glob('*.pt')]}
 for name in ('budget_ledger','calibration','completion','cleanup'):
  f=d/(name+'.json')
  if f.exists():
   v=json.loads(f.read_text());row[name]={k:v[k] for k in ('status','charged_attempts','training_flops','weights','updates','released_file_bytes','best_validation_hard_dice') if k in v}
 log=d/'train.log'
 if log.exists():row['log_tail']=log.read_text()[-1400:]
 if not (d/'cleanup.json').exists() or len(result['runs'])<4:result['runs'].append(row)
result['slurm_errors']={f.name:f.read_text()[-2000:] for f in p.glob('slurm-*.err') if f.stat().st_size}
print(json.dumps(result))'''
    result=json.loads(remote(['python3','-c',code,REMOTE/NAME]))
    write(ROOT/'reports'/NAME/'latest_cluster_check.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
