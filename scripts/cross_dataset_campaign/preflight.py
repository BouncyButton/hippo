"""Validate frozen sources, data identities, loss recipes and all launch commands."""
import argparse
from pathlib import Path
import subprocess
import sys

from scripts.cross_dataset_campaign.common import read, write, digest, CAP, SEISMIC_CAP
from scripts.cross_dataset_campaign.controller import verify, command_for


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',type=Path,required=True);args=parser.parse_args()
    stage=args.stage.resolve();plan=verify(stage)
    subprocess.run([sys.executable,'-m','unittest','scripts.cross_dataset_campaign.tests','scripts.cross_dataset_campaign.test_resume','-v'],check=True)
    if read(stage/'full_study_plan.json')['submission_enabled'] is not False:
        raise ValueError('Long study must remain gated')
    checked=set();datasets=set()
    for task in plan['tasks']:
        if task['protocol'] in checked:continue
        checked.add(task['protocol']);folder=stage/'protocols'/task['protocol'];p=read(folder/'protocol.json')
        if p['augmentation']!='none':raise ValueError('Augmentation forbidden')
        if task['kind']=='conventional':
            if p['budget_flops']!=(SEISMIC_CAP if p['dataset']=='GeoFWI3D' else CAP) or 'Dice-only' not in p['training_loss']:
                raise ValueError('Wrong cap/base objective')
            for name,key in [('splits_10patients.json','splits_sha256'),('nnunet_architecture.json','nnunet_architecture_sha256'),('profile.json','profile_sha256')]:
                if digest(folder/name)!=p[key]:raise ValueError('Frozen protocol file differs')
            split=read(folder/'splits_10patients.json')[0]
            if len(set(split['train']))!=p['train_count'] or set(split['train'])&set(split['val']):
                raise ValueError('Training cohort mismatch/leakage')
            if p['data'] not in datasets:
                if digest(p['data'])!=p['pkl_sha256']:raise ValueError('Dataset identity differs')
                datasets.add(p['data'])
        else:
            if len(p['train_cases'])!=5 or p['updates']!=200 or p['accumulation']!=2:
                raise ValueError('Fine-tuning amount differs from MSD')
            if set(p['train_cases'])&set(p['evaluation_cases']):raise ValueError('Support leakage')
            if p['dataset']=='ADNI' and digest(p['data'])!=p['dataset_sha256']:
                raise ValueError('ADNI pickle differs')
            if p['dataset']=='GeoFWI3D':
                root=Path(p['data'])
                for name in p['train_cases']+p['evaluation_cases']:
                    row=p['manifest']['cases'][name]
                    for kind in ('image','label'):
                        if digest(root/row[kind])!=row[kind+'_sha256']:raise ValueError('Seismic data differs')
        command=command_for(stage,plan,task,Path(plan['base_staging']))
        if not Path(command[0]).exists():raise ValueError('Runtime missing')
    subprocess.run(['bash','-n',str(stage/'run.sbatch')],check=True)
    write(stage/'preflight.json',dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),
        checked_protocols=sorted(checked),runs=len(plan['tasks']),gpu_validation='First allocated run verifies actual forward/backward before subsequent cells',
        tests='Archive corruption/idempotence, no augmentation, confusion counts, cap exhaustion, source/data/split checks'))
    print('Preflight passed',flush=True)


if __name__=='__main__':main()
