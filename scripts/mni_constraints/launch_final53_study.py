"""Deploy and collect one bounded, authorized conditional full53T matrix."""
import argparse
import json
import subprocess
import time
from scripts.launch_unetr_subfields import remote
from scripts.cross_dataset_campaign.common import read, write, REMOTE
from scripts.mni_constraints.final53_study import ROOT, MODULE, PARENT, NAME, verify
from scripts.mni_constraints.stage_seed5_remote import assemble


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', choices=(NAME,), default=NAME)
    args = parser.parse_args()
    stage, dest, report = ROOT / 'experiments' / args.name, REMOTE / args.name, ROOT / 'reports' / args.name
    plan = verify(stage)
    report.mkdir(parents=True, exist_ok=True)
    assert not (report / 'submission.json').exists(), 'Already submitted'
    prior = REMOTE / PARENT
    oldenv = ':'.join(str(prior / p) for p in ('source', 'source/scripts', 'source/baselines/nnUNet'))
    code = '''import json,sys,subprocess
from pathlib import Path
from scripts.cross_dataset_campaign.storage import quota
p=Path(sys.argv[1]);assert json.loads((p/'control.json').read_text())['pause']
assert not subprocess.check_output(['squeue','-h','-u','3160552'],text=True).strip()
q=quota();assert q['free_gib']>=16;print(json.dumps(q))'''
    write(report / 'initial_quota.json', json.loads(remote(['env', 'PYTHONPATH=' + oldenv, 'python3', '-c', code, prior])))
    remote(['python3', '-c', 'import sys;from pathlib import Path;p=Path(sys.argv[1]);assert not (p/"submission.json").exists();p.mkdir(exist_ok=True)', dest])
    print(json.dumps(assemble(stage, dest, report, read(stage / 'deployment_known_donors.json'))), flush=True)
    env = ':'.join(str(dest / p) for p in ('source', 'source/scripts', 'source/baselines/nnUNet'))
    command = ['env', 'PYTHONPATH=' + env, plan['conventional_python'], '-m', MODULE]
    output = remote(command + ['preflight', '--stage', dest], timeout=900)
    write(report / 'preflight.json', json.loads(output.strip().splitlines()[-1]))
    print('Remote preflight passed', flush=True)
    with (report / 'collector.log').open('a') as log:
        process = subprocess.Popen(['caffeinate', '-i', str(ROOT / '.venv/bin/python'), '-u', '-m',
            'scripts.cross_dataset_campaign.collect', '--stage', str(dest), '--local-stage', str(stage),
            '--destination', str(report)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(report / 'collector_process.json', dict(pid=process.pid, started_at=time.time()))
    receipt = json.loads(remote(command + ['submit', '--stage', dest]))
    write(report / 'submission.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        raise
