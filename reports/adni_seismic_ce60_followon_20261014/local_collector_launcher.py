"""Local control of the authorized four-model MSD target60 sweep."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from scripts.cross_dataset_campaign.common import read, write, REMOTE
from scripts.cross_dataset_campaign.msd_ce_constraints_job import NAME, MODULE, ROOT


def collect(name=NAME):
    report=ROOT/'reports'/name
    report.mkdir(parents=True,exist_ok=True)
    receipt=report/'collector_process.json'
    if receipt.exists():
        try:
            old=read(receipt)
            command=subprocess.check_output(['ps','-p',str(old['pid']),'-o','command='],text=True)
            if 'scripts.cross_dataset_campaign.collect' in command and name in command:
                return old
        except (subprocess.CalledProcessError,ProcessLookupError):
            pass
    with (report/'collector.log').open('a') as log:
        runtime=Path(sys.executable)
        if sys.version_info<(3,12):
            runtime=Path('/Library/Frameworks/Python.framework/Versions/3.12/bin/python3')
            if not runtime.exists():raise RuntimeError('The archive collector requires Python 3.12 for safe tar extraction')
        process=subprocess.Popen([str(runtime),'-m','scripts.cross_dataset_campaign.collect',
            '--stage',str(REMOTE/name),'--local-stage',str(ROOT/'experiments'/name),
            '--destination',str(report)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    # Keep this archive service awake while the Mac remains open and connected.
    awake=subprocess.Popen(['caffeinate','-i','-w',str(process.pid)],start_new_session=True,
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    value=dict(pid=process.pid,caffeinate_pid=awake.pid,started_at=time.time(),python=str(runtime),log=str(report/'collector.log'))
    write(receipt,value)
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['status','pause','resume','collect','report'])
    args=parser.parse_args()
    if args.command=='collect':
        print(collect());return
    if args.command=='report':
        from scripts.cross_dataset_campaign.report_msd_ce_constraints import report
        result=report(ROOT/'reports'/NAME)
        print({k:result[k] for k in ('status','completed','total','baseline_scores')});return
    stage=REMOTE/NAME
    env=':'.join(str(stage/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
    command=['env','PYTHONPATH='+env,'python3','-m',MODULE,args.command,'--stage',str(stage)]
    subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','bocconi-cluster',shlex.join(command)],check=True)
    if args.command=='resume':
        print(collect())


if __name__=='__main__':
    main()
