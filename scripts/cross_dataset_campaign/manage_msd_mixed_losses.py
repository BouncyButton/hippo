"""Local status, pause, resume and archive controls for the mixed-loss sweep."""
import argparse
import shlex
import subprocess

from scripts.cross_dataset_campaign.common import REMOTE
from scripts.cross_dataset_campaign.msd_mixed_losses_job import NAME, MODULE
from scripts.cross_dataset_campaign.manage_msd_ce_constraints import collect


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['status','pause','resume','collect','report'])
    args=parser.parse_args()
    if args.command=='collect':
        print(collect(name=NAME));return
    if args.command=='report':
        from scripts.cross_dataset_campaign.report_msd_mixed_losses import report
        result=report()
        print({k:result[k] for k in ('status','completed','total','baseline_scores')});return
    stage=REMOTE/NAME
    env=':'.join(str(stage/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
    command=['env','PYTHONPATH='+env,'python3','-m',MODULE,args.command,'--stage',str(stage)]
    subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','bocconi-cluster',shlex.join(command)],check=True)
    if args.command=='resume':print(collect(name=NAME))


if __name__=='__main__':main()
