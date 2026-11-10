"""Start only after verified MSD completion; control and collect ADNI separately."""
import argparse
import ast
import shlex
import subprocess

from scripts.cross_dataset_campaign.common import REMOTE, read, write
from scripts.cross_dataset_campaign.adni_mixed_losses_job import NAME, MODULE, ROOT
from scripts.cross_dataset_campaign import msd_mixed_losses_job as msd
from scripts.cross_dataset_campaign.manage_msd_ce_constraints import collect


def remote(command):
    stage=REMOTE/NAME
    env=':'.join(str(stage/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
    args=['env','PYTHONPATH='+env,'python3','-m',MODULE,command,'--stage',str(stage)]
    return subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','bocconi-cluster',shlex.join(args)],text=True)


def start():
    from scripts.cross_dataset_campaign.report_msd_mixed_losses import report
    previous=report()
    if previous['status']!='complete':
        return dict(status='waiting_for_msd',completed=previous['completed'],total=previous['total'])
    # Submission is gated again on the cluster, including all acknowledgments
    # and cleanup receipts. This never replaces MSD's pending continuation.
    receipt=ast.literal_eval(remote('submit').strip())
    write(ROOT/'experiments'/NAME/'submission.json',receipt)
    write(ROOT/'reports'/NAME/'submission.json',receipt)
    write(ROOT/'reports'/NAME/'submissions'/f'{receipt["job_id"]}.json',receipt)
    return dict(status='submitted_or_already_active',submission=receipt,collector=collect(NAME))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['start','status','pause','resume','collect','report'])
    command=parser.parse_args().command
    if command=='start':print(start())
    elif command=='collect':print(collect(NAME))
    elif command=='report':
        from scripts.cross_dataset_campaign.report_adni_mixed_losses import report
        result=report();print({k:result[k] for k in ('status','completed','total','baseline_scores','released_bytes')})
    else:
        print(remote(command))
        if command=='resume':print(collect(NAME))


if __name__=='__main__':main()
