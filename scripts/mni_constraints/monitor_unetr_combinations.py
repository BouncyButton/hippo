"""Current-session monitoring and independent archive checks for UNETR trials."""
import contextlib
import io
import json
import subprocess
import time

from scripts.launch_unetr_subfields import remote
from scripts.cross_dataset_campaign.common import read, write, REMOTE
from scripts.mni_constraints.unetr_delayed_combinations import ROOT, NAME
from scripts.mni_constraints import analyze_unetr_delayed_combinations as analysis

RESULT_FILENAME = 'combination_results.json'


def compact_result(row):
    return dict(arm=row['arm'],candidate=row.get('candidate',row['arm']),seed=row['seed'],fold=row['fold'],
                dice=row['current']['validation_percent'],gain_pp=row['gain_pp'])


REMOTE_SNAPSHOT = '''import json,sys,csv,subprocess
from pathlib import Path
p=Path(sys.argv[1]);read=lambda f:json.loads(f.read_text())
s=read(p/'status.json') if (p/'status.json').exists() else {}
out={'status':s,'active':[],'queue':subprocess.check_output(['squeue','-h','-u','3160552','-o','%i|%T|%j|%R'],text=True).strip()}
for name in (s.get('active') or []):
 f=p/'runs'/name;row={'id':name}
 if (f/'metrics.csv').exists():
  with (f/'metrics.csv').open() as stream:curve=list(csv.DictReader(stream))
  if curve:row.update(step=int(curve[-1]['step']),dice=100*float(curve[-1]['macro_hard_dice']))
 if (f/'train.log').exists():
  tail=(f/'train.log').read_text()[-1600:]
  if 'Traceback' in tail or 'Error:' in tail:row['error_tail']=tail
 out['active'].append(row)
if (p/'failure.json').exists():out['failure']=read(p/'failure.json')
print(json.dumps(out))'''


def main():
    report = ROOT/'reports'/NAME
    seen, analyzed, last, errors = set(), -1, None, 0
    while True:
        try:
            snapshot = json.loads(remote(['python3', '-c', REMOTE_SNAPSHOT, REMOTE/NAME]))
            write(report/'monitor_latest.json', snapshot)
            status = snapshot['status']
            compact = dict(status=status.get('status'), completed=status.get('completed'),
                           selected=status.get('selected'), active=snapshot['active'])
            if compact != last:
                print(json.dumps(dict(progress=compact)), flush=True)
                last = compact
            if snapshot.get('failure') or status.get('status') == 'failed':
                raise RuntimeError(json.dumps(snapshot))
            count = len(list((report/'runs').glob('*/cleanup.json')))
            ready_for_analysis=(report/'frozen/plan.json').exists() or status.get('status') is not None
            if ready_for_analysis and (count != analyzed or status.get('status') == 'complete'):
                with contextlib.redirect_stdout(io.StringIO()):
                    analysis.main()
                result = read(report/RESULT_FILENAME)
                added = [compact_result(r) for r in result['rows'] if r['id'] not in seen]
                if added:
                    print(json.dumps(dict(verified_results=added, summary=result['summary'])), flush=True)
                seen.update(r['id'] for r in result['rows'])
                analyzed = count
                if result['status'] == 'complete':
                    print(json.dumps(dict(stage_complete=True, outcome=result['outcome'],
                                          winner=result['winner'], summary=result['summary'])), flush=True)
                    return
            errors = 0
        except Exception as error:
            errors += 1
            detail=error.stderr.strip() if isinstance(error,subprocess.CalledProcessError) and error.stderr else str(error)
            print(json.dumps(dict(monitor_error=detail, consecutive_errors=errors)), flush=True)
            if errors >= 3:
                raise
        time.sleep(45)


if __name__ == '__main__':
    main()
