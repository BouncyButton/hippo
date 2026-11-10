"""Compact read-only status for the two finite recovery investigations."""
import argparse
import json
from scripts.launch_unetr_subfields import remote
from scripts.cross_dataset_campaign.common import REMOTE


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',default='mni_seed_recovery_shared43t_20261108');args=parser.parse_args()
    code='''import json,sys,subprocess
from pathlib import Path
p=Path(sys.argv[1]);read=lambda f:json.loads(f.read_text()) if f.exists() else {}
state=read(p/'status.json');result=dict(status=state,queue=subprocess.check_output(['squeue','-h','-u','3160552','-o','%i|%T|%j'],text=True).splitlines(),active=[])
for identifier in state.get('active',[]) or []:
 out=p/'runs'/identifier
 row=dict(id=identifier,files=sorted(f.name for f in out.glob('*')),ledger=read(out/'budget_ledger.json'))
 if (out/'train.log').exists():row['log_tail']=(out/'train.log').read_text()[-1800:]
 result['active'].append(row)
selection=read(p/'selection.json')
result['new_seed_results']=[r for r in selection.get('rows',[]) if r.get('seed',0)>=3]
result['strength_summary']=selection.get('summary',[])
result['cleanup_count']=len(list((p/'runs').glob('*/cleanup.json')))
result['failure']=read(p/'failure.json')
print(json.dumps(result))'''
    print(remote(['python3','-c',code,REMOTE/args.name]),flush=True)


if __name__=='__main__':main()
