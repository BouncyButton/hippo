"""Local status, pause/resume and durable-archive service controls."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from scripts.cross_dataset_campaign.common import NAME, REMOTE, read, write

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports'/NAME


def start_collector():
    REPORT.mkdir(parents=True,exist_ok=True)
    receipt=REPORT/'collector_process.json'
    if receipt.exists():
        try:
            os.kill(read(receipt)['pid'],0)
            return read(receipt)
        except ProcessLookupError:pass
    with (REPORT/'collector.log').open('a') as log:
        process=subprocess.Popen([sys.executable,'-m','scripts.cross_dataset_campaign.collect',
            '--local-stage',str(ROOT/'experiments'/NAME),'--destination',str(REPORT)],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    value=dict(pid=process.pid,started_at=time.time(),log=str(REPORT/'collector.log'))
    write(receipt,value)
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['status','pause','resume','collect'])
    args=parser.parse_args()
    if args.command=='collect':print(start_collector());return
    stage=REMOTE/NAME
    command=['env','PYTHONPATH='+':'.join(str(stage/p) for p in ('source','source/scripts','source/baselines/nnUNet')),
             'python3','-m','scripts.cross_dataset_campaign.controller',args.command,'--stage',str(stage)]
    subprocess.run(['ssh','-o','BatchMode=yes','bocconi-cluster',shlex.join(command)],check=True)
    if args.command=='resume':print(start_collector())


if __name__=='__main__':main()
