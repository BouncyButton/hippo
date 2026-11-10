"""Queue six original-recipe foundation runs behind the active ADNI sweep."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import time

from scripts.cross_dataset_campaign import controller, followon_job as chain
from scripts.cross_dataset_campaign import adni_mixed_losses_job as adni
from scripts.cross_dataset_campaign.common import REMOTE, read, write, digest
from scripts.cross_dataset_campaign.prepare import ROOT, freeze_source

PREVIOUS_NAME = 'foundation_adni_geofwi_bands_original10_20261015'
NAME = PREVIOUS_NAME + '_msd_native'
MODULE = 'scripts.cross_dataset_campaign.foundation_checks_job'
ARMS = ('dice', 'dice_bands', 'dice_bands_edge')
DATASETS = ('ADNI', 'GeoFWI3D')
REFERENCE = 'docs/experiments/medsam3_multiclass_replication_20260930/protocols/fold0_seed17.json'
RECIPE_KEYS = ('base_sha256', 'updates', 'accumulation', 'warmup_updates', 'learning_rate',
    'final_learning_rate', 'calibration_slabs_per_volume', 'constraint_warmup_updates',
    'gradient_ratio_target', 'gradient_ratio_p95_cap', 'calibration_seed')
BASE_SUBMIT = controller.submit
BASE_COMMAND = controller.command_for
BASE_CANCEL = controller.cancel_successor


def prepare():
    stage = ROOT/'experiments'/NAME
    if stage.exists():
        raise ValueError('Never overwrite a frozen campaign')
    stage.mkdir(parents=True)
    reference = read(ROOT/REFERENCE)
    assert reference['gradient_ratio_target'] == .1 and reference['gradient_ratio_p95_cap'] == .5
    write(stage/'original_recipe.json', reference)
    parent = ROOT/'experiments'/adni.NAME
    adni.verify(parent)
    plan = deepcopy(read(parent/'plan.json'))
    old = read(ROOT/'experiments/cross_dataset_exploration_20261014/plan.json')
    tasks = []
    for dataset in DATASETS:
        origin = ROOT/'experiments/cross_dataset_exploration_20261014/protocols'/f'{dataset}_foundation_fold0_seed0/protocol.json'
        p = deepcopy(read(origin))
        for k in RECIPE_KEYS:
            if p[k] != reference[k]:
                raise ValueError('Foundation recipe differs from original: '+k)
        p.update(name=NAME, requested_arms=list(ARMS), primary_checkpoint='final_update',
            baseline_loss='Unchanged original SAM3 native detection/presence/mask fine-tuning loss',
            reference_protocol=REFERENCE, reference_protocol_sha256=digest(ROOT/REFERENCE),
            origin_protocol_sha256=digest(origin),
            runtime_optimization=dict(transform_cache_cpu_gib=8, resident_label_supports=False,
                batched_calibration_readback=False, baseline_omits_unused_auxiliary_reductions=False,
                gpu_equivalence_required_before_training=False,
                training_path='Successful MSD-style full native/bands/edge objective; unchanged original resumable optimizer loop',
                compiler_retained_backward='AOTAutograd donated_buffer=False for original multi-loss calibration',
                validation_sampling='Original CUDA operations: finite native/bands/edge losses and repeated backward; preserve RNG, precision and sampling'))
        write(stage/'protocols'/dataset/'protocol.json', p)
        model = 'MedSAM3' if dataset == 'ADNI' else 'SAM3'
        for arm in ARMS:
            tasks.append(dict(id=f'{dataset}_{model}_{arm}_target10_f0s0', kind='foundation',
                dataset=dataset, model=model, fold=0, seed=0, arm=arm, target=.1, protocol=dataset))
    plan.update(name=NAME, num_runs=6, tasks=tasks, augmentation='none',
        recovery_of=PREVIOUS_NAME,
        foundation_python=old['foundation_python'], base_staging=old['base_staging'],
        base_sha256=old['base_sha256'], archive_resume_module=MODULE,
        archive_dependency_stages=[str(REMOTE/adni.NAME)],
        predecessor=dict(name=adni.NAME, stage=str(REMOTE/adni.NAME), manifest_sha256=digest(parent/'manifest.json')),
        authorization='User requests original-recipe MedSAM3 ADNI and SAM3 nonmedical GeoFWI3D fine-tuning, baseline/bands/bands+edge only, fold0 single seed0 no augmentation, optimization before submission.',
        comparison_policy='200 updates per arm; same initialization, support, sampling, native loss and schedule within dataset; 10% sequential auxiliary targets, 50% p95 ceiling. Final checkpoint primary.',
        exploration_followup='Only these six foundation runs; old broad follow-on and replicated campaigns remain held')
    plan['storage'].update(archive=str(ROOT/'reports'/NAME), estimated_peak_logical_gib=2,
        mirrored_peak_gib=4, reserve_gib=2, minimum_free_gib=6)
    write(stage/'plan.json', plan)
    write(stage/'control.json', dict(pause=False, disabled_tasks=[], full_submission_enabled=False))
    write(stage/'source_version.json', dict(original_recipe_sha256=digest(ROOT/REFERENCE),
        reference_engine_sha256=digest(ROOT/'scripts/cross_dataset_campaign/foundation.py'),
        upstream_provenance=read(ROOT/'reports/adni_seismic_ce60_followon_20261014/medsam3_upstream_provenance.json')))
    freeze_source(stage)
    batch = (parent/'run.sbatch').read_text().replace('hippo-adni-mixed-60-10', 'hippo-foundation-original10').replace(adni.MODULE+' run', MODULE+' run')
    (stage/'run.sbatch').write_text(batch)
    write(stage/'manifest.json', {str(f.relative_to(stage)):digest(f) for f in sorted(stage.rglob('*'))
        if f.is_file() and f.name not in ('manifest.json','control.json')})
    verify(stage)
    print(stage)


def verify(stage):
    for name, sha in read(stage/'manifest.json').items():
        if digest(stage/name) != sha:
            raise ValueError('Frozen foundation campaign changed: '+name)
    plan = read(stage/'plan.json')
    expected = [(d,a) for d in DATASETS for a in ARMS]
    if (plan['name'] != NAME or plan['num_runs'] != 6 or plan['augmentation'] != 'none'
            or [(t['dataset'],t['arm']) for t in plan['tasks']] != expected):
        raise ValueError('Unexpected foundation comparison matrix')
    reference = read(stage/'original_recipe.json')
    for task in plan['tasks']:
        p = read(stage/'protocols'/task['protocol']/'protocol.json')
        if ((task['fold'], task['seed'], task['target']) != (0,0,.1)
                or task['kind'] != 'foundation' or p['augmentation'] != 'none'
                or p['fold'] != 0 or p['seed'] != 0 or p['requested_arms'] != list(ARMS)
                or any(p[k] != reference[k] for k in RECIPE_KEYS)
                or p['gradient_ratio_target'] != .1 or p['gradient_ratio_p95_cap'] != .5
                or p['updates'] != 200 or p['accumulation'] != 2):
            raise ValueError('Original fine-tuning recipe changed')
        if len(set(p['train_cases'])) != 5 or set(p['train_cases']) & set(p['evaluation_cases']):
            raise ValueError('Incorrect support or overlapping evaluation')
        if len(p['evaluation_cases']) != (54 if task['dataset']=='ADNI' else 96):
            raise ValueError('Evaluation cohort changed')
        if p['initialization'] != ('medical_pretrained_LoRA' if task['dataset']=='ADNI' else 'SAM3_with_fresh_zero_output_LoRA'):
            raise ValueError('Foundation initialization changed')
    return plan


def check_predecessor(plan):
    dep = plan['predecessor']; stage = Path(dep['stage'])
    if dep['name'] != adni.NAME or digest(stage/'manifest.json') != dep['manifest_sha256']:
        raise ValueError('ADNI prerequisite changed')
    return stage, adni.verify(stage)


def predecessor_complete(stage, plan):
    status = read(stage/'status.json') if (stage/'status.json').exists() else {}
    if read(stage/'control.json')['pause'] or status.get('status') != 'complete' or status.get('completed') != 56:
        return False
    for task in plan['tasks']:
        root = stage/'runs'/task['id']; ack = read(root/'archive_acknowledgment.json')
        if (read(root/'completion.json')['status'] != 'complete' or read(root/'cleanup.json')['status'] != 'complete'
                or ack['status'] != 'verified' or not ack['source_verified']
                or ack['manifest_sha256'] != digest(root/'archive_manifest.json')
                or ack['source_manifest_sha256'] != digest(stage/'manifest.json')):
            raise ValueError('ADNI archive and cleanup are incomplete')
    return True


def configure_chain():
    chain.verify = verify
    chain.check_predecessor = check_predecessor
    chain.predecessor_complete = predecessor_complete
    chain.msd = adni


def submit(stage, continuation=None):
    configure_chain()
    own = {p.stem for p in (stage/'submissions').glob('*.json')}
    active = [row for row in controller.queue() if row[0] in own]
    if continuation is None and active:
        return read(stage/'submissions'/f'{active[0][0]}.json')
    if continuation is not None:
        return chain.call_submit(stage, continuation)
    if (stage/'status.json').exists() and read(stage/'status.json').get('status') == 'complete':
        raise RuntimeError('Foundation comparison already complete')
    return chain.submit(stage)


def command_for(stage, plan, task, base_path):
    command = BASE_COMMAND(stage, plan, task, base_path)
    command[2] = 'scripts.cross_dataset_campaign.foundation_msd'
    return command


def run(stage):
    configure_chain()
    plan = verify(stage); predecessor, previous = check_predecessor(plan)
    job = os.environ['SLURM_JOB_ID']
    os.environ['HIPPO_ALLOCATION_DEADLINE'] = str(time.time()+plan['graceful_stop_seconds'])
    if read(stage/'control.json')['pause'] or read(predecessor/'control.json')['pause']:
        controller.summarize(stage, plan, 'paused'); return
    if (predecessor/'status.json').exists() and read(predecessor/'status.json').get('status') == 'failed':
        controller.summarize(stage, plan, 'blocked_predecessor_failed'); return
    controller.submit = lambda ignored, continuation=None: chain.ensure_successor(stage, job)
    def cancel(phase, active_job):
        if phase == predecessor and read(phase/'status.json').get('status') == 'complete':
            return
        BASE_CANCEL(stage, active_job)
    controller.cancel_successor = cancel
    if not predecessor_complete(predecessor, previous):
        controller.verify = adni.verify
        def previous_command(phase, phase_plan, task, base_path):
            command = BASE_COMMAND(phase, phase_plan, task, base_path)
            command[2] = 'scripts.cross_dataset_campaign.adni_mixed_losses'
            env = ':'.join(str(predecessor/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
            return ['env','PYTHONPATH='+env,*command]
        controller.command_for = previous_command
        try:
            controller.run(predecessor)
        except Exception:
            controller.summarize(stage, plan, 'blocked_predecessor_failed'); BASE_CANCEL(stage, job); raise
        if not predecessor_complete(predecessor, previous):
            controller.summarize(stage, plan, read(predecessor/'status.json')['status'], active='predecessor:'+predecessor.name)
            return
    controller.verify = verify; controller.command_for = command_for
    controller.run(stage)


def preflight(stage):
    plan = verify(stage); check_predecessor(plan)
    source = read(stage/'source_version.json')['upstream_provenance']
    # Provenance records are mappings of the upstream files actually used.
    hashes = source['python_source_sha256']
    if not hashes:
        raise ValueError('Missing upstream provenance')
    for task in (plan['tasks'][0],plan['tasks'][3]):
        p = read(stage/'protocols'/task['protocol']/'protocol.json')
        if not Path(plan['foundation_python']).exists(): raise ValueError('Foundation environment missing')
        if digest(plan['base_staging']) != p['base_sha256']: raise ValueError('Base checkpoint differs')
        if digest(p['lora_weights']) != p['lora_sha256']: raise ValueError('Medical LoRA source differs')
        if p['dataset']=='ADNI':
            if digest(p['data']) != p['dataset_sha256']: raise ValueError('ADNI dataset differs')
        else:
            manifest = p['manifest']
            if p['train_cases'] != manifest['splits']['training_subsets']['5pct'][:5] or p['evaluation_cases'] != manifest['splits']['validation']:
                raise ValueError('GeoFWI3D fold0 cohort differs')
            for case in p['train_cases']+p['evaluation_cases']:
                for kind in ('image','label'):
                    row=manifest['cases'][case]
                    if digest(Path(p['data'])/row[kind]) != row[kind+'_sha256']: raise ValueError('GeoFWI3D source changed')
        for name, sha in hashes.items():
            if digest(Path(p['upstream'])/name) != sha: raise ValueError('Upstream source changed: '+name)
    subprocess.run([plan['foundation_python'],'-m','scripts.cross_dataset_campaign.test_foundation_fast','--stage',str(stage)], check=True)
    subprocess.run([plan['foundation_python'],'-m','scripts.cross_dataset_campaign.test_foundation_msd'], check=True)
    subprocess.run(['bash','-n',str(stage/'run.sbatch')],check=True)
    write(stage/'preflight.json',dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),
        checks=['Original 10% recipe and six-run scope','Data, checkpoint and split identities',
                'CPU cache, Gram arithmetic and resume tests','Actual support geometry and sampling',
                'One running / one queued ADNI handoff'],
        gpu_gate='Original MSD-style native/bands/edge finite losses and repeated backward required before training'))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['prepare','preflight','submit','run','status'])
    parser.add_argument('--stage',type=Path);args=parser.parse_args()
    if args.command=='prepare':prepare();return
    stage=args.stage.resolve()
    if args.command=='preflight':preflight(stage)
    elif args.command=='submit':print(submit(stage))
    elif args.command=='run':run(stage)
    else:print((stage/'status.json').read_text() if (stage/'status.json').exists() else 'Queued, not started')


if __name__=='__main__':main()
