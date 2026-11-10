"""Two fixed-recipe categorical outer-band runs with four archived controls."""
from copy import deepcopy
from functools import partial
import fcntl
from pathlib import Path
import shutil

from scripts.cross_dataset_campaign.common import read, write, digest, REMOTE
from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints import campaign, headroom_bands as bands
from scripts.unetr_subfields import cap_extension as batch

ROOT = campaign.ROOT
NAME = 'mni_class_aware_bands_cap2_f0s0_20261107'
MODULE = 'scripts.mni_constraints.class_aware_pilot'
PARENT = 'mni_shared_cap2_baseline_bands_f0s0_20261107'
CAP = 63952901334180
MODELS = ('swinunetr', 'unetr_vit_b16')
SUMMARIZE = controller.summarize


def tasks():
    return [dict(id=f'MNI_{m}_dice_class_aware_bands_f0s0', kind='conventional', dataset='MNI',
                 model=m, arm='dice_bands', fold=0, seed=0, protocol=m,
                 condition='class_aware_bands', augmentation='xy_d4_v1', target=bands.TARGETS[m])
            for m in MODELS]


def prepare(_target=None):
    stage = ROOT/'experiments'/NAME
    parent = ROOT/'experiments'/PARENT
    if (stage/'manifest.json').exists():
        return verify(stage)
    for part in ('source', 'data'):
        shutil.copytree(parent/part, stage/part, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for filename in ('class_aware_bands.py', 'test_class_aware_bands.py', 'profile_class_aware_bands.py', 'class_aware_pilot.py'):
        shutil.copy2(ROOT/'scripts/mni_constraints'/filename, stage/'source/scripts/mni_constraints'/filename)
    references = {}; reuse = []
    for model in MODELS:
        origin = parent/'protocols'/f'{model}_dice_bands'
        folder = stage/'protocols'/model
        shutil.copytree(origin, folder)
        p = deepcopy(read(origin/'protocol.json')); references[model] = deepcopy(p)
        profile = read(ROOT/'reports'/NAME/'profiles'/f'{model}.json')
        assert profile['objective_sha256'] == digest(stage/'source/scripts/mni_constraints/objectives.py')
        write(folder/'profile.json', profile)
        s = bands.band_schedule(profile, CAP)
        old_schedule = p['models'][p['trainer_alias']]['objectives']['dice_bands']
        # The measured cost change does not change updates, prefix or LR horizon.
        for key in ('max_steps', 'prefix_updates', 'constraint_warmup_updates'):
            assert s[key] == old_schedule[key], (model, key)
        p.update(name=NAME, data=str(REMOTE/NAME/'data/MNI.pkl'),
                 models={p['trainer_alias']:dict(parameters=profile['parameters'], objectives={'dice_bands':s})},
                 profile_sha256=digest(folder/'profile.json'),
                 constraint_variant='categorical_on_original_union_band_support',
                 constraint_recipe='Four-class categorical CE on original two-step outer-contour inner/outer bands. Same half-inner/half-outer and per-case averaging; no new internal-interface support or class weighting. Base loss unchanged.',
                 categorical_source_sha256=profile['categorical_source_sha256'],
                 parent_stage=PARENT, parent_protocol_sha256=digest(origin/'protocol.json'))
        write(folder/'protocol.json', p)
        for arm in ('dice', 'dice_bands'):
            archive = ROOT/'reports'/PARENT/'runs'/f'MNI_{model}_{arm}_f0s0'
            manifest = read(archive/'archive_manifest.json'); verify_archive(archive, manifest)
            assert read(archive/'cleanup.json')['status'] == 'complete'
            control = stage/'controls'/model/arm; control.mkdir(parents=True)
            for name in ('config.json','training_final.json','validation_latest.json','validation_best.json','metrics.csv','completion.json','runtime_recipe.json','calibration.json'):
                shutil.copy2(archive/name, control/name)
            reuse.append(dict(model=model, arm=arm, archive=str(archive),
                archive_manifest_sha256=digest(archive/'archive_manifest.json'),
                files={f.name:digest(f) for f in control.iterdir()}))
    plan = deepcopy(read(parent/'plan.json'))
    plan.update(name=NAME, tasks=tasks(), num_runs=2, logical_runs=6, reused_runs=4,
        models=list(MODELS), folds=[0], seeds=[0], budget_flops=CAP,
        archive_resume_module=MODULE, recipe_name='class_aware_bands_only_fixed_cap2_recipes',
        storage=dict(minimum_free_gib=16, estimated_peak_mirrored_gib=12, archive=str(ROOT/'reports'/NAME)),
        authorization='User requested class-aware bands for SwinUNETR and UNETR on MNI, one fold and seed, same cap and recipes, compared to existing baseline and original bands. Two GPU workers, at most one running and one queued job. Per-run verified archive and cleanup.',
        criterion=dict(primary='Final-budget macro subfield Dice against both archived matched controls.',
            secondary='Per-subfield/union Dice, FN, FP, internal swaps, train/validation curves and gaps.',
            adoption='One-fold one-seed exploratory ablation; no automatic expansion or recipe tuning.'),
        limitation='Only voxel target changes on original band support. Internal-only interfaces are not newly supervised. Validation is diagnostic only and never used for calibration.')
    write(stage/'plan.json', plan); write(stage/'control.json', dict(pause=False))
    write(stage/'reference_protocols.json', references); write(stage/'control_reuse.json', reuse)
    write(stage/'source_version.json', dict(parent_manifest_sha256=digest(parent/'manifest.json'),
        changes='Categorical auxiliary term only, fresh norm calibration at unchanged targets and prefix; actual reprofiled loss arithmetic. Same network/base loss/optimizer/LR/dropout/augmentation/splits/seed/update count. Reuse four verified controls.'))
    (stage/'run.sbatch').write_text((parent/'run.sbatch').read_text().replace(
        'scripts.mni_constraints.lower_cap_pairs', MODULE).replace('hippo-mni-budget', 'hippo-mni-classbands'))
    campaign.freeze(stage)
    return verify(stage)


def verify(stage):
    for name, sha in read(stage/'manifest.json').items():
        assert digest(stage/name) == sha, name
    plan = read(stage/'plan.json'); references = read(stage/'reference_protocols.json')
    assert plan['tasks'] == tasks() and plan['num_runs'] == 2 and plan['gpu_workers'] == 2
    assert plan['budget_flops'] == CAP and len(read(stage/'control_reuse.json')) == 4
    allowed = {'name','data','models','profile_sha256','constraint_variant','constraint_recipe',
               'categorical_source_sha256','parent_stage','parent_protocol_sha256'}
    for model in MODELS:
        folder = stage/'protocols'/model; p = read(folder/'protocol.json'); old = references[model]
        assert {k for k in set(p)|set(old) if p.get(k) != old.get(k)} <= allowed
        s = p['models'][p['trainer_alias']]['objectives']['dice_bands']
        assert s == bands.band_schedule(read(folder/'profile.json'), CAP)
        assert p['lr_horizon_updates'] == s['max_steps']
        assert campaign.training_flops(s['max_steps'],s) <= CAP < campaign.training_flops(s['max_steps']+1,s)
        assert p['calibration_target_ratio'] == bands.TARGETS[model] and p['calibration_p95_cap'] == .75
        assert p['categorical_source_sha256'] == digest(stage/'source/scripts/mni_constraints/class_aware_bands.py')
        assert p['folds'] == p['seeds'] == [0] and p['augmentation'] == 'xy_d4_v1'
        for name,key in [('profile.json','profile_sha256'),('splits_10patients.json','splits_sha256'),('nnunet_architecture.json','nnunet_architecture_sha256')]:
            assert digest(folder/name) == p[key]
        assert digest(stage/'data/MNI.pkl') == p['pkl_sha256']
        for arm in ('dice','dice_bands'):
            config = read(stage/'controls'/model/arm/'config.json')
            splits = read(folder/'splits_10patients.json')[0]
            assert config['budget_flops'] == CAP and config['fold'] == config['seed'] == 0
            assert set(config['train_cases']) == set(splits['train'])
            assert set(config['validation_cases']) == set(splits['val'])
    return plan


def preflight(stage):
    from scripts.mni_constraints import class_aware_bands
    from scripts.mni_constraints.test_class_aware_bands import checks
    class_aware_bands.install()
    scientific = checks()
    old_verify, old_models = bands.verify, batch.MODELS
    try:
        bands.verify, batch.MODELS = verify, MODELS
        result = bands.preflight(stage)
    finally:
        bands.verify, batch.MODELS = old_verify, old_models
    result.update(scientific_checks=scientific, compared_controls=4,
                  recipe_check='Exact protocol difference allowlist passed; same updates, LR and prefix/ramp.')
    write(stage/'preflight.json', result)
    return result


def train(args):
    from scripts.mni_constraints import objectives, class_aware_bands
    from scripts.unetr_subfields import core
    class_aware_bands.install()
    trainer = batch.bind(args); p = read(args.protocol_dir/'protocol.json'); identity = p['model_identity']
    def declared(protocol):
        assert protocol['augmentation'] == 'xy_d4_v1'
    core.trainer.check_no_augmentation = declared
    trainer.training_loss = partial(objectives.training_loss, identity=identity)
    trainer.calibration_case = partial(objectives.calibration_case, identity=identity)
    trainer.select_weights = partial(objectives.select_weights, target=p['calibration_target_ratio'], cap=p['calibration_p95_cap'])
    trainer.geometry_weights = objectives.geometry_weights; trainer.signed_distance_maps = objectives.signed_distance_maps
    s = p['models'][args.model]['objectives'][args.objective]
    trainer.base.validation_steps = lambda maximum,count=20: sorted({next((n for n in range(1,maximum+1)
        if campaign.training_flops(n,s)>=target),maximum) for target in p['validation_target_flops']}|{maximum})
    write(args.output/'runtime_recipe.json', dict(model=identity, base_loss=p['training_loss'],
        constraints=p['constraints'], constraint_variant=p['constraint_variant'],
        categorical_source_sha256=digest(Path(class_aware_bands.__file__)),
        objective_sha256=digest(Path(objectives.__file__)), augmentation=p['augmentation'],
        protocol_sha256=digest(args.protocol_dir/'protocol.json'), initialization='fresh_from_fixed_seed',
        lr_horizon_updates=p['lr_horizon_updates'], primary_checkpoint='final_budget', baseline_only=False))
    with (args.output/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        trainer.run(args)


def evaluation(folder):
    data = read(folder/'validation_latest.json')
    matrix = [[sum(c['confusion'][i][j] for c in data['cases']) for j in range(4)] for i in range(4)]
    return dict(validation=data['summary'], training=read(folder/'training_final.json')['summary'],
        fn=sum(matrix[i][0] for i in range(1,4)), fp=sum(matrix[0][1:]),
        internal_swaps=sum(matrix[i][j] for i in range(1,4) for j in range(1,4) if i!=j))


def comparison(stage):
    rows = []
    for task in tasks():
        output = stage/'runs'/task['id']
        if not (output/'cleanup.json').exists():
            continue
        controls = {arm:evaluation(stage/'controls'/task['model']/arm) for arm in ('dice','dice_bands')}
        current = evaluation(output)
        deltas = {arm:{key:100*(current['validation'][key]-value['validation'][key])
                          for key in current['validation'] if key.endswith('hard_dice')}
                  for arm,value in controls.items()}
        rows.append(dict(model=task['model'], controls=controls, class_aware=current, delta_pp=deltas))
    return rows


def summarize(stage, plan, status, active=None):
    SUMMARIZE(stage, plan, status, active)
    state = read(stage/'status.json'); state['class_aware_comparison'] = comparison(stage)
    write(stage/'status.json', state)
    if status == 'complete':
        write(stage/'comparison.json', state['class_aware_comparison'])


def main():
    campaign.MODULE=MODULE; campaign.prepare=prepare; campaign.verify=verify
    campaign.preflight=preflight; campaign.train=train; controller.summarize=summarize
    campaign.main()


if __name__ == '__main__':
    main()
