"""Replicate the frozen two-sample MedSAM3 recipe on saved ADNI folds 1 and 3."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import time

from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.common import REMOTE, read, write, digest
from scripts.cross_dataset_campaign.prepare import ROOT

NAME = 'adni_medsam3_2sample_folds13_seed0_20261018'
MODULE = 'scripts.cross_dataset_campaign.adni_foundation_folds_job'
PARENT = 'foundation_two_sample_2gpu_20261015'
PARENT_SHA = '9c40972644c2bee57f2dd597e6b8c8201e6b3f9fa484d4c7ec5ef6b683e46b54'
FOLDS = (1, 3)
ARMS = ('dice', 'dice_bands', 'dice_bands_edge')
ADDITIONS = ('adni_foundation_folds_job.py', 'manage_adni_foundation_folds.py',
             'test_adni_foundation_folds.py')


def tasks():
    return [dict(id=f'ADNI_MedSAM3_2sample_{arm}_f{fold}s0', kind='foundation',
        dataset='ADNI', model='MedSAM3', fold=fold, seed=0, arm=arm,
        target=.1, protocol=f'ADNI_f{fold}') for arm in ARMS for fold in FOLDS]


def select_support(folds, by_case, fold):
    import numpy as np
    full = folds[fold]
    people = sorted({by_case[c] for c in full['train']})
    validation = sorted({by_case[c] for c in full['val']})
    if len(people) != 108 or len(validation) != 27 or set(people) & set(validation):
        raise ValueError('Saved subject split is invalid')
    chosen = sorted(np.random.RandomState(0).permutation(people)[:5].tolist())
    cases = [sorted(c for c in full['train'] if by_case[c] == person)[i % 2]
             for i, person in enumerate(chosen)][:2]
    return dict(fold=fold, support_subjects=chosen[:2], train_cases=cases,
                evaluation_cases=sorted(full['val']), validation_subjects=validation,
                five_subject_subset=chosen, full_train_subjects=people)


def protocol_for(original, selection):
    p = deepcopy(original)
    p.update(name=NAME, fold=selection['fold'], train_cases=selection['train_cases'],
        support_subjects=selection['support_subjects'], evaluation_cases=selection['evaluation_cases'],
        support_selection='Same saved-fold RandomState(0) five-subject subset rule as fold 0; first two sorted subjects, alternating L/R crops; independent of outcomes')
    return p


def parallel_source(text):
    old = "t['dataset'] == dataset"
    new = "t['protocol'] == dataset"
    if text.count(old) != 1:
        raise ValueError('Unexpected parent dispatch source')
    return text.replace(old, new)


def prepare():
    parent = ROOT/'reports'/PARENT/'frozen'
    stage = ROOT/'experiments'/NAME
    if stage.exists():
        verify(stage)
        return stage
    if digest(parent/'manifest.json') != PARENT_SHA:
        raise ValueError('Wrong original two-sample campaign')
    for name, sha in read(parent/'manifest.json').items():
        if digest(parent/name) != sha:
            raise ValueError('Parent frozen source differs: '+name)
    old = read(parent/'protocols/ADNI/protocol.json')
    provenance = read(ROOT/'reports/adni_colab_3fold_3seed_noaug_20261015/bundle/split_provenance.json')
    saved_path = ROOT/'datasets/Dataset103_ADNI/splits_final.json'
    if digest(saved_path) != provenance['saved_folds_sha256'] or old['dataset_sha256'] != provenance['dataset_sha256']:
        raise ValueError('ADNI split/data provenance differs')
    saved = read(saved_path); by_case = provenance['case_to_subject']
    selections = [select_support(saved, by_case, f) for f in (0, *FOLDS)]
    if any(selections[0][k] != old[k] for k in ('train_cases','support_subjects','evaluation_cases')):
        raise ValueError('Selection rule no longer reproduces original fold 0')
    stage.mkdir(parents=True)
    shutil.copytree(parent/'source', stage/'source', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for filename in ADDITIONS:
        shutil.copy2(ROOT/'scripts/cross_dataset_campaign'/filename, stage/'source/scripts/cross_dataset_campaign'/filename)
    parallel = parent/'source/scripts/cross_dataset_campaign/foundation_parallel.py'
    (stage/'source/scripts/cross_dataset_campaign/foundation_fold_parallel.py').write_text(parallel_source(parallel.read_text()))
    shutil.copy2(parent/'manifest.json', stage/'parent_manifest.json')
    shutil.copy2(parent/'protocols/ADNI/protocol.json', stage/'parent_protocol.json')
    shutil.copy2(parent/'source_version.json', stage/'source_version.json')
    shutil.copy2(saved_path, stage/'saved_folds.json')
    write(stage/'split_provenance.json', provenance)
    write(stage/'split_audit.json', dict(status='passed', subset_seed=0, selections=selections,
        note='Fold 3 shares the two training samples with fold 0, but uses a different held-out validation fold; do not treat these as independent training-support draws.'))
    for selection in selections[1:]:
        write(stage/'protocols'/f"ADNI_f{selection['fold']}"/'protocol.json', protocol_for(old, selection))
    plan = deepcopy(read(parent/'plan.json'))
    for key in ('predecessor','supersedes','archive_dependency_stages'):
        plan.pop(key, None)
    plan.update(name=NAME, tasks=tasks(), num_runs=6, folds=list(FOLDS), seeds=[0],
        archive_resume_module=MODULE, reference_campaign=PARENT, reference_manifest_sha256=PARENT_SHA,
        authorization='User requests two-sample MedSAM3 ADNI fine-tuning for saved folds 1 and 3 to replicate baseline, bands and bands+edge benefits. Preserve seed0, 200 updates, no augmentation and original gradient-calibration recipe.',
        exploration_followup='Only six fresh ADNI runs: folds 1 and 3, seed0, baseline/bands/bands+edge',
        parallelism={**plan['parallelism'], 'datasets':[f'ADNI_f{f}' for f in FOLDS],
            'grouping':'protocol', 'policy':'One GPU per fold; baseline then bands then bands+edge within each fold'})
    plan['storage']['archive'] = str(ROOT/'reports'/NAME)
    write(stage/'plan.json', plan)
    write(stage/'control.json', dict(pause=False, disabled_tasks=[], full_submission_enabled=False))
    batch = (parent/'run.sbatch').read_text().replace('hippo-foundation-2gpu','hippo-adni-medsam3-f13')
    batch = batch.replace('scripts.cross_dataset_campaign.foundation_two_sample_job run', MODULE+' run')
    (stage/'run.sbatch').write_text(batch)
    write(stage/'manifest.json', {str(p.relative_to(stage)):digest(p) for p in sorted(stage.rglob('*'))
        if p.is_file() and p.name not in ('manifest.json','control.json')})
    verify(stage)
    report = ROOT/'reports'/NAME
    write(report/'request_summary.json', dict(name=NAME, folds=list(FOLDS), seed=0, updates=200,
        augmentation='none', training_crops=2, training_subjects=2, validation_crops=54,
        validation_subjects=27, arms=list(ARMS), workers=2, reference_manifest_sha256=PARENT_SHA,
        calibration='Unchanged 10% reference target / 50% p95 ceiling, independently calibrated per run',
        source='All original training and GPU/archive worker sources byte-identical; only dispatcher group key changes from dataset to protocol'))
    return stage


def verify(stage):
    for name, sha in read(stage/'manifest.json').items():
        if digest(stage/name) != sha:
            raise ValueError('Frozen campaign changed: '+name)
    if digest(stage/'parent_manifest.json') != PARENT_SHA:
        raise ValueError('Parent manifest differs')
    parent = read(stage/'parent_manifest.json')
    for name, sha in parent.items():
        if name.startswith('source/') and digest(stage/name) != sha:
            raise ValueError('Original source changed: '+name)
    if digest(stage/'parent_protocol.json') != parent['protocols/ADNI/protocol.json']:
        raise ValueError('Original ADNI protocol changed')
    original_parallel = (stage/'source/scripts/cross_dataset_campaign/foundation_parallel.py').read_text()
    if (stage/'source/scripts/cross_dataset_campaign/foundation_fold_parallel.py').read_text() != parallel_source(original_parallel):
        raise ValueError('Dispatcher changes exceed grouping key')
    provenance = read(stage/'split_provenance.json'); saved = read(stage/'saved_folds.json')
    if digest(stage/'saved_folds.json') != provenance['saved_folds_sha256']:
        raise ValueError('Saved folds differ')
    old = read(stage/'parent_protocol.json')
    audit = read(stage/'split_audit.json')['selections']
    if len(audit) != 3 or [s['fold'] for s in audit] != [0,1,3]:
        raise ValueError('Fold selection audit differs')
    for selection in audit:
        f=selection['fold']; by_case=provenance['case_to_subject']; full=saved[f]
        if (selection['evaluation_cases'] != sorted(full['val']) or len(selection['train_cases']) != 2
                or len(set(selection['support_subjects'])) != 2
                or [by_case[c] for c in selection['train_cases']] != selection['support_subjects']
                or not set(selection['train_cases']) <= set(full['train'])
                or set(selection['support_subjects']) & {by_case[c] for c in full['val']}):
            raise ValueError('Invalid subject split')
        if f == 0:
            if any(selection[k] != old[k] for k in ('train_cases','support_subjects','evaluation_cases')):
                raise ValueError('Fold0 reference differs')
        elif read(stage/'protocols'/f'ADNI_f{f}'/'protocol.json') != protocol_for(old, selection):
            raise ValueError('Unrequested scientific recipe change')
    plan = read(stage/'plan.json')
    if (plan['name'] != NAME or plan['tasks'] != tasks() or plan['num_runs'] != 6
            or plan['archive_resume_module'] != MODULE or plan['archive_wait_timeout_seconds'] != 300
            or plan['max_job_hours'] != 24 or plan['parallelism']['workers'] != 2
            or plan['parallelism']['datasets'] != ['ADNI_f1','ADNI_f3']):
        raise ValueError('Scope or concurrency changed')
    return plan


def submit(stage, continuation=None):
    verify(stage)
    own = {p.stem for p in (stage/'submissions').glob('*.json')}
    active = [r for r in controller.queue() if r[0] in own]
    if continuation is None and active:
        return read(stage/'submissions'/f'{active[0][0]}.json')
    prior = controller.verify; controller.verify = verify
    try:
        return controller.submit(stage, continuation)
    finally:
        controller.verify = prior


def run(stage):
    from scripts.cross_dataset_campaign import foundation_fold_parallel as parallel
    plan = verify(stage); job = os.environ['SLURM_JOB_ID']
    def successor():
        p=stage/'successor.json'
        if p.exists():
            r=read(p)
            if (r['dependency']==job and r['manifest_sha256']==digest(stage/'manifest.json')
                    and any(x[0]==r['job_id'] and x[1]=='PENDING' for x in controller.queue())):
                return r
        return submit(stage, job)
    parallel.run(stage, plan, successor, lambda:controller.cancel_successor(stage, job))


def preflight(stage):
    plan=verify(stage); old=read(stage/'parent_protocol.json')
    for path, sha in ((plan['base_staging'],plan['base_sha256']), (old['data'],old['dataset_sha256']),
                      (old['lora_weights'],old['lora_sha256'])):
        if digest(path)!=sha:raise ValueError('Dataset or pretrained weights differ: '+path)
    for name,sha in read(stage/'source_version.json')['upstream_provenance']['python_source_sha256'].items():
        if digest(Path(old['upstream'])/name)!=sha:raise ValueError('Upstream source differs: '+name)
    subprocess.run([plan['foundation_python'],'-m','scripts.cross_dataset_campaign.test_adni_foundation_folds','--stage',str(stage)],check=True)
    subprocess.run(['bash','-n',str(stage/'run.sbatch')],check=True)
    qos=subprocess.check_output(['sacctmgr','-n','-P','show','qos','stud','format=Name,MaxTRESPerUser%80,MaxTRESPJ%80,MaxJobsPU,MaxSubmitPU'],text=True)
    limits=dict(x.split('=') for x in next(x.split('|')[1] for x in qos.splitlines() if x.startswith('stud|')).split(','))
    if int(limits['gres/gpu'])!=2 or int(limits['cpu'])<8:raise ValueError('Recheck GPU/CPU account limits')
    scheduler=subprocess.run(['sbatch','--test-only',str(stage/'run.sbatch'),str(stage)],capture_output=True,text=True,check=True)
    usage=controller.quota()
    if usage['free_gib']<plan['storage']['minimum_free_gib']:raise ValueError('Insufficient quota')
    write(stage/'preflight.json',dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),quota=usage,
        qos=qos,scheduler_test=scheduler.stdout+scheduler.stderr,checked_at=time.time(),
        checks=['Frozen trainer byte-identical','Saved subject folds and actual geometry','Matched training/calibration schedules',
                'Concurrent fold dispatch and archive/pause/failure guards','Pretrained weights/data/upstream hashes','Two-GPU scheduler acceptance']))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','verify','preflight','submit','run','status'])
    parser.add_argument('--stage',type=Path);args=parser.parse_args()
    stage=args.stage or ROOT/'experiments'/NAME
    if args.command=='prepare':print(prepare())
    elif args.command=='verify':print(verify(stage)['name'])
    elif args.command=='preflight':preflight(stage)
    elif args.command=='submit':print(submit(stage))
    elif args.command=='run':run(stage)
    else:print((stage/'status.json').read_text() if (stage/'status.json').exists() else 'Not started')


if __name__=='__main__':main()
