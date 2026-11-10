"""Compact progress and training-fit checks for a finite shared-cap pilot."""
import argparse
import json
from pathlib import Path

from scripts.launch_unetr_subfields import remote
from scripts.cross_dataset_campaign.common import REMOTE,write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);args=parser.parse_args()
    code=r'''import csv,json,subprocess,sys
from pathlib import Path
p=Path(sys.argv[1]);result={'queue':subprocess.check_output(['squeue','-h','-u','3160552','-o','%i|%T|%M|%R'],text=True).splitlines()}
for name in ('status','failure'):
 f=p/(name+'.json')
 if f.exists():result[name]=json.loads(f.read_text())
result['runs']=[]
for r in sorted((p/'runs').glob('*')):
 row={'id':r.name,'cleaned':(r/'cleanup.json').exists()}
 f=r/'budget_ledger.json'
 if f.exists():row['charged_updates']=json.loads(f.read_text())['charged_attempts']
 f=r/'metrics.csv'
 if f.exists():
  data=list(csv.DictReader(f.open()));last=data[-1]
  row['latest']={k:last[k] for k in ('step','compute_fraction','training_macro_hard_dice','macro_hard_dice','learning_rate')}
 f=r/'completion.json'
 if f.exists():row['final_training_dice']=json.loads(f.read_text())['final_training_hard_dice']
 f=r/'train.log'
 if f.exists() and 'Traceback' in f.read_text():row['error']=f.read_text()[-3000:]
 elif f.exists() and 'latest' not in row:row['startup_log']=f.read_text()[-1000:]
 result['runs'].append(row)
result['slurm_errors']={f.name:'\n'.join(line for line in f.read_text().splitlines() if not line.startswith('INFO: Quota usage'))[-3000:] for f in p.glob('slurm-*.err') if f.stat().st_size}
print(json.dumps(result))'''
    result=json.loads(remote(['python3','-c',code,REMOTE/args.name]))
    write(Path('reports')/args.name/'latest_check.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
