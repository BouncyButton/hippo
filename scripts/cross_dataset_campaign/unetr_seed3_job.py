"""Six supplementary UNETR seed-3 checks; frozen trainer, two isolated GPUs."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess

from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.common import CAP, REMOTE, read, write, digest

ROOT = Path(__file__).resolve().parents[2]
NAME = 'msd_unetr_seed3_baseline_bands_20261015'
MODULE = 'scripts.cross_dataset_campaign.unetr_seed3_job'
BUNDLE_SHA = '3b9d0c30f71873b6f834ae3b1fd989bf33e37bef7682fdda094b656daff3620b'
INIT_SHA = '24e0605cb7bb8fecaa4840a1bcb890fbdb0ec9b31dd8743d9c43a768260d2d41'
DATA = str(REMOTE/'hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl')
ARMS = ('dice', 'dice_bands')
BASE_COMMAND = controller.command_for
BASE_SUBMIT = controller.submit


def tasks():
    return [dict(id=f'MSD_unetr_vit_b16_{arm}_f{fold}s3', kind='conventional',
        dataset='MSD', model='unetr_vit_b16', fold=fold, seed=3, arm=arm,
        target=.1, baseline_loss='Dice', protocol='unetr_vit_b16_s3')
        for fold in (0, 1, 2) for arm in ARMS]


def recipe(original):
    result = deepcopy(original)
    result.update(name=NAME, data=DATA, seeds=[3], baseline_reference=dict(
        initialization_sha256=INIT_SHA, seed=3,
        policy='Fresh matched seed-3 baseline and bands; supplementary sensitivity check, original seeds retained'))
    return result


def prepare():
    bundle = ROOT/'reports/msd_colab_3fold_3seed_noaug_20261015/bundle'
    parent = ROOT/'reports/msd_mixed_losses_nnunet60_others10_20261014/frozen'
    stage = ROOT/'experiments'/NAME
    if stage.exists():
        verify(stage)
        return stage
    assert digest(bundle/'bundle_manifest.json') == BUNDLE_SHA
    manifest = read(bundle/'bundle_manifest.json')
    for name, sha in manifest.items():
        if digest(bundle/name) != sha:
            raise ValueError('Original Colab bundle changed: '+name)
    stage.mkdir(parents=True)
    shutil.copytree(bundle/'source', stage/'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('unetr_seed3_job.py', 'unetr_seed3_parallel.py', 'unetr_seed3_gpu_worker.py'):
        shutil.copy2(ROOT/'scripts/cross_dataset_campaign'/name, stage/'source/scripts/cross_dataset_campaign'/name)
    shutil.copy2(bundle/'bundle_manifest.json', stage/'parent_bundle_manifest.json')
    original = bundle/'protocols/unetr_vit_b16_s0'
    shutil.copytree(original, stage/'parent_protocol')
    shutil.copytree(original, stage/'protocols/unetr_vit_b16_s3')
    write(stage/'protocols/unetr_vit_b16_s3/protocol.json', recipe(read(original/'protocol.json')))
    for name in ('runtime_benchmark.json', 'runtime_benchmark_manifest.json'):
        shutil.copy2(parent/name, stage/name)
    plan = deepcopy(read(parent/'plan.json'))
    for name in ('base_staging', 'base_sha256'):
        plan.pop(name, None)
    plan.pop('archive_resume_module', None)
    plan.update(name=NAME, num_runs=6, tasks=tasks(), single_allocation=True,
        parallelism=dict(workers=2, arms=list(ARMS), gpus_per_worker=1, cpus_per_worker=4,
                         gpu_type='4g.40gb', allocation_memory_gib=80),
        authorization='User requested UNETR seed3 baseline and baseline+bands, confirmed folds0/1/2, two GPU workers, and one sbatch submission. No augmentation. Preserve ongoing Colab runs and original seeds.',
        comparison_policy='Six supplementary sensitivity checks, never replace seed1 or pool into the original three-seed study. Fresh baseline for every seed3 fold. Cluster vs Colab environment remains explicit.',
        exploration_followup='Only these six seed3 runs are authorized',
        calibration_policy='Original MONAI Dice including background, AdamW, sequential10% bands target,50% p95 ceiling, calibration at update50, original50-update ramp')
    plan['storage'].update(archive=str(ROOT/'reports'/NAME), estimated_peak_logical_gib=8,
                          mirrored_peak_gib=16, reserve_gib=4, minimum_free_gib=20)
    write(stage/'plan.json', plan)
    write(stage/'control.json', dict(pause=False, disabled_tasks=[]))
    write(stage/'initialization_reference.json', dict(seed=3, model='unetr_vit_b16',
        model_sha256=INIT_SHA, parameters=92685555, torch='2.12.1+cu130', python='3.11.15',
        optimizer_updates=0, source='CPU model initialization in the original cluster environment'))
    batch = (parent/'run.sbatch').read_text().replace('hippo-msd-mixed-60-10', 'hippo-unetr-seed3-2gpu')
    batch = batch.replace('#SBATCH --gres=gpu:4g.40gb:1', '#SBATCH --nodes=1\n#SBATCH --ntasks=2\n#SBATCH --gres=gpu:4g.40gb:2')
    batch = batch.replace('#SBATCH --mem=40G', '#SBATCH --mem=80G')
    batch = batch.replace('scripts.cross_dataset_campaign.msd_mixed_losses_job run', MODULE+' run')
    (stage/'run.sbatch').write_text(batch)
    write(stage/'manifest.json', {str(p.relative_to(stage)): digest(p) for p in sorted(stage.rglob('*'))
        if p.is_file() and p.name not in ('manifest.json', 'control.json')})
    verify(stage)
    return stage


def verify(stage):
    for name, sha in read(stage/'manifest.json').items():
        if digest(stage/name) != sha:
            raise ValueError('Frozen campaign changed: '+name)
    if digest(stage/'parent_bundle_manifest.json') != BUNDLE_SHA:
        raise ValueError('Parent bundle changed')
    parent = read(stage/'parent_bundle_manifest.json')
    for name, sha in parent.items():
        if name.startswith('source/') and digest(stage/name) != sha:
            raise ValueError('Scientific source differs from Colab: '+name)
    for filename in ('protocol.json', 'profile.json', 'splits_10patients.json', 'nnunet_architecture.json'):
        if digest(stage/'parent_protocol'/filename) != parent['protocols/unetr_vit_b16_s0/'+filename]:
            raise ValueError('Parent protocol changed')
    folder = stage/'protocols/unetr_vit_b16_s3'
    p = read(folder/'protocol.json')
    if p != recipe(read(stage/'parent_protocol/protocol.json')):
        raise ValueError('Unrequested scientific change')
    for filename, key in (('profile.json','profile_sha256'), ('splits_10patients.json','splits_sha256'),
                          ('nnunet_architecture.json','nnunet_architecture_sha256')):
        if digest(folder/filename) != p[key]:
            raise ValueError('Frozen supporting artifact changed')
    if (p['folds'] != [0,1,2] or p['augmentation'] != 'none' or p['budget_flops'] != CAP
            or p['calibration_target_ratio'] != .1 or p['calibration_p95_cap'] != .5):
        raise ValueError('Wrong experimental scope')
    splits = read(folder/'splits_10patients.json')
    if len(splits) != 3 or any(len(set(s['train'])) != 10 or len(set(s['val'])) != 52
                              or set(s['train']) & set(s['val']) for s in splits):
        raise ValueError('Incorrect frozen cohorts')
    benchmark = read(stage/'runtime_benchmark.json')
    row = next(r for r in benchmark['models'] if r['model'] == 'unetr_vit_b16')
    if not row['exact_loss_gradient_optimizer_model_calibration_metrics_rng_resume']:
        raise ValueError('Missing runtime equivalence evidence')
    if digest(stage/'runtime_benchmark.json') != p['runtime_acceleration']['benchmark_summary_sha256']:
        raise ValueError('Runtime benchmark changed')
    for name, sha in read(stage/'runtime_benchmark_manifest.json')['additional_sources'].items():
        if digest(stage/'source'/name) != sha:
            raise ValueError('Runtime differs from tested acceleration')
    plan = read(stage/'plan.json')
    if (plan['name'] != NAME or plan['tasks'] != tasks() or plan['num_runs'] != 6
            or plan['augmentation'] != 'none' or plan['parallelism']['workers'] != 2
            or plan.get('single_allocation') is not True or 'archive_resume_module' in plan
            or plan['parallelism']['arms'] != list(ARMS) or plan['archive_wait_timeout_seconds'] != 300
            or plan['max_job_hours'] != 24 or plan['storage']['minimum_free_gib'] < 20):
        raise ValueError('Campaign scope or resource contract changed')
    return plan


def preflight(stage):
    plan = verify(stage)
    import hashlib
    import importlib.metadata
    import torch
    from scripts.cross_dataset_campaign import conventional as exp
    folder = stage/'protocols/unetr_vit_b16_s3'
    p = read(folder/'protocol.json')
    if digest(p['data']) != p['pkl_sha256']:
        raise ValueError('Cluster dataset differs')
    if str(torch.__version__) != '2.12.1+cu130' or importlib.metadata.version('monai') != '1.6.0':
        raise ValueError('Original cluster environment changed')
    torch.set_num_threads(2)
    exp.base.seed_everything(3)
    model = exp.build_model(p, read(folder/'nnunet_architecture.json'))
    state = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        state.update(key.encode()); state.update(value.detach().numpy().tobytes())
    if state.hexdigest() != INIT_SHA or sum(x.numel() for x in model.parameters()) != 92685555:
        raise ValueError('Seed3 initialization changed')
    loss, optimizer = exp.ORIGINAL_LOSS_OPTIMIZER('unet', model, p)
    if type(optimizer).__name__ != 'AdamW' or not loss.include_background:
        raise ValueError('Baseline recipe changed')
    for arm in ARMS:
        schedule = p['models']['unet']['objectives'][arm]
        if schedule['max_steps'] != 245 or schedule['consumed_training_flops'] > CAP:
            raise ValueError('Optimization budget differs')
    subprocess.run(['bash', '-n', str(stage/'run.sbatch')], check=True)
    qos = subprocess.check_output(['sacctmgr','-n','-P','show','qos','stud',
                                  'format=Name,MaxTRESPerUser%80,MaxTRESPJ%80'], text=True)
    write(stage/'preflight.json', dict(status='passed', manifest_sha256=digest(stage/'manifest.json'),
        initialization_sha256=INIT_SHA, dataset_sha256=p['pkl_sha256'], optimizer_updates=0,
        reused_exact_gpu_equivalence_sha256=digest(stage/'runtime_benchmark.json'), qos=qos,
        checks=['Exact scientific source, profile and cohorts from original Colab replication',
                'Seed3 initialization, unchanged Dice/AdamW/no augmentation/245-update cap',
                'Two isolated GPU bindings checked inside allocation before training',
                'Per-run verified local archive and cleanup; at least20GiB quota headroom']))


def command_for(stage, plan, task, base_path):
    command = BASE_COMMAND(stage, plan, task, base_path)
    command[2] = 'scripts.cross_dataset_campaign.msd_mixed_losses'
    return command


def submit(stage, continuation=None):
    if continuation is not None:
        raise ValueError('This campaign is authorized as a single sbatch allocation')
    # Reuse an active receipt; never duplicate a campaign allocation.
    own = {p.stem for p in (stage/'submissions').glob('*.json')}
    if continuation is None:
        for row in controller.queue():
            if row[0] in own:
                return read(stage/'submissions'/f'{row[0]}.json')
    return BASE_SUBMIT(stage, continuation)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare','verify','preflight','submit','run','status'])
    parser.add_argument('--stage', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare()); return
    stage = args.stage.resolve()
    controller.verify = verify; controller.command_for = command_for; controller.submit = submit
    if args.command == 'verify': print(verify(stage)['name'])
    elif args.command == 'preflight': preflight(stage); print('Seed3 preflight passed')
    elif args.command == 'submit': print(submit(stage))
    elif args.command == 'status': print((stage/'status.json').read_text() if (stage/'status.json').exists() else 'Not started')
    else:
        from scripts.cross_dataset_campaign import unetr_seed3_parallel as parallel
        # All six runs share this allocation. No prequeued successor or automatic
        # resubmission; interrupted work remains available for an explicit resume.
        parallel.run(stage, verify(stage), lambda: None, lambda: None)


if __name__ == '__main__':
    main()
