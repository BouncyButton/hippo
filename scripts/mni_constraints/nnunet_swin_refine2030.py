"""Bounded original-bands timing/strength screen with optional paired confirmation."""
from copy import deepcopy
import math
from pathlib import Path
import shutil
from statistics import mean
from types import SimpleNamespace

from scripts.cross_dataset_campaign.common import read, write, digest, REMOTE
from scripts.cross_dataset_campaign.prepare import schedule
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints import campaign, headroom_bands
from scripts.unetr_subfields import cap_extension as batch

ROOT = campaign.ROOT
NAME = 'mni_nnunet_swin_refine2030_round2_53t_20261109'
PARENT = 'mni_verified_unetr_lower_lr_constraints14_53t_0519_20261109'
MODULE = 'scripts.mni_constraints.nnunet_swin_refine2030'
CAP = 53305109592450
MODELS = ('nnunet', 'swinunetr')
PAIRS = tuple((s, f) for s in (0, 5, 19) for f in range(3))
SCREEN = {'nnunet': ((0, 2), (5, 1), (19, 2)), 'swinunetr': ((0, 2), (19, 0))}
CANDIDATES = {
    'nnunet': (('start30', 0., .3), ('start60', 0., .6),
        ('tenth15', 1/15, .15), ('tenth30', 1/15, .3), ('tenth60', 1/15, .6),
        ('early10', None, .1), ('early15', None, .15), ('early90', None, .9),
        ('quarter30', .25, .3)),
    'swinunetr': (('start10', 0., .1), ('start20', 0., .2),
        ('halfearly10', .04, .1), ('halfearly20', .04, .2),
        ('early15', None, .15), ('early20', None, .2), ('early30', None, .3),
        ('sixth10', 1/6, .1)),
}


def tasks():
    result = []
    for model in MODELS:
        for key, _, target in CANDIDATES[model]:
            for seed, fold in PAIRS:
                result.append(dict(id=f'MNI_{model}_{key}_f{fold}s{seed}', kind='conventional',
                    dataset='MNI', model=model, arm='dice_bands', protocol=f'{model}_{key}',
                    candidate=key, target=target, seed=seed, fold=fold,
                    screen=(seed, fold) in SCREEN[model],
                    augmentation='none' if model == 'nnunet' else 'xy_d4_v1'))
    return result


def protocol(old, profile, candidate):
    key, fraction, target = candidate
    p = deepcopy(old);model = p['model_identity'];alias = p['trainer_alias']
    prior = p['models'][alias]['objectives']['dice_bands'];costs = deepcopy(profile['costs'])
    for arm, cost in costs.items():
        cost['update_flops'] += int('bands' in arm)+int('edge' in arm)
    n = CAP*fraction/costs['dice']['update_flops'] if fraction is not None else prior['prefix_updates']
    onset = min((math.floor(n), math.ceil(n)), key=lambda i: (abs(i-n), i))
    timing = dict(prefix_updates=onset, sdf_prefix_updates=onset,
                  constraint_warmup_updates=prior['constraint_warmup_updates'])
    s = schedule(costs, 'dice_bands', timing, 20, CAP)
    assert s['max_steps'] == prior['max_steps']
    p.update(name=NAME, data=str(REMOTE/NAME/'data/MNI.pkl'),
        models={alias:dict(parameters=profile['parameters'], objectives={'dice_bands':s})},
        calibration_target_ratio=target,
        timing_policy=f'Original bands only; training-only calibration after{onset} updates; unchanged{timing["constraint_warmup_updates"]}-update ramp.',
        calibration_strength_policy=f'Fixed calibrated gradient target{target}; unchanged p95guard0.75.',
        decision_rule='Complete fixed development screen, then confirm eligible shared recipes if time permits; no seed replacement or early stopping within a run.',
        selection_caveat='Adaptive development-validation screening; all9-pair confirmation required before claiming consistency. 20:30 absolute deadline; no automatic full-matrix promotion.')
    return p


def prepare(_target=None):
    stage, parent = ROOT/'experiments'/NAME, ROOT/'experiments'/PARENT
    if (stage/'manifest.json').exists():
        return verify(stage)
    original = read(parent/'manifest.json')
    for name, sha in original.items():
        assert digest(parent/name) == sha, name
    for part in ('source', 'data'):
        shutil.copytree(parent/part, stage/part, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('nnunet_swin_refine2030.py', 'nnunet_swin_refine3h.py', 'bounded_refinement_runner.py'):
        shutil.copy2(ROOT/'scripts/mni_constraints'/name, stage/'source/scripts/mni_constraints'/name)
    shutil.copy2(parent/'manifest.json', stage/'parent_manifest.json')
    shutil.copy2(parent/'nnunet_reuse_equivalence.json', stage/'nnunet_reuse_equivalence.json')
    for model in MODELS:
        origin = parent/'protocols'/f'{model}_dice_bands'
        for c in CANDIDATES[model]:
            folder = stage/'protocols'/f'{model}_{c[0]}'
            shutil.copytree(origin, folder)
            shutil.copy2(origin/'protocol.json', folder/'parent_protocol.json')
            write(folder/'protocol.json', protocol(read(origin/'protocol.json'), read(origin/'profile.json'), c))
    controls, donors = [], {}
    for model in MODELS:
        for seed, fold in PAIRS:
            for arm in ('dice', 'dice_bands'):
                identity = f'MNI_{model}_{arm}_f{fold}s{seed}'
                source = ROOT/'reports'/PARENT/'runs'/identity;remote_source = REMOTE/PARENT/'runs'/identity
                if not source.exists():
                    source = ROOT/'reports'/PARENT/'frozen/reused_runs'/identity
                    remote_source = REMOTE/PARENT/'reused_runs'/identity
                verify_archive(source, read(source/'archive_manifest.json'))
                assert read(source/'cleanup.json')['status'] == 'complete'
                target = stage/'controls'/identity;shutil.copytree(source, target)
                controls.append(dict(id=identity, model=model, arm=arm, seed=seed, fold=fold,
                    archive=str(source), archive_sha256=digest(source/'archive_manifest.json')))
                for f in target.iterdir():
                    if f.is_file() and f.name != 'archive_acknowledgment.json':
                        donors[str(f.relative_to(stage))] = str(remote_source/f.name)
    write(stage/'controls.json', controls)
    window = read(ROOT/'reports'/NAME/'authorization_window.json')
    plan = dict(name=NAME, tasks=tasks(), num_runs=len(tasks()), screen_runs=sum(t["screen"] for t in tasks()),
        seeds=[0, 5, 19], folds=[0, 1, 2], models=list(MODELS), budget_flops=CAP, gpu_workers=2,
        conventional_python=read(parent/'plan.json')['conventional_python'],
        started_at=window['started_at'], deadline=window['deadline'], maximum_seconds=window["deadline"]-window["started_at"],
        stop_buffer_seconds=180, admission_reserve_seconds=600,
        storage=dict(minimum_free_gib=18, estimated_peak_mirrored_gib=16, worst_case_mirrored_gib=32,
                     archive=str(ROOT/'reports'/NAME)),
        authorization='User extended refinement until20:30; nnUNet timing/strength tests on three selected pairs and SwinUNETR refinement; same cap,two GPUs,laptop-independent. Full matrix stays paused.',
        selection_policy='nnUNet:all3screen gainspositive,rank minimum gain then mean. Swin:both screen gainspositive,mean macro and meanCA4 improvement over existing bands,rank mean macro improvement. Confirm eligible candidates in fixed screen ranking; reject after any nonpositive baseline gain. Require9/9 for confirmation. Separate authorized20:30 full-matrix handoff; this worker never modifies the parent.',
        offline_policy='Training never waits for laptop archive. Retain unarchived artifacts; retire only after verified durable local acknowledgment. No automatic continuation beyond deadline.')
    write(stage/'plan.json', plan);write(stage/'control.json', dict(pause=False))
    write(stage/'source_version.json', dict(parent_manifest_sha256=digest(parent/'manifest.json'),
        changes='Only protocol timing/strength and an offline deadline-bounded supervisor are added. Existing training/runtime, loss definitions, optimizer recipes and graph FLOP profiles are byte-identical.'))
    script = (parent/'run.sbatch').read_text().replace('hippo-verified53', 'hippo-bands-3h').replace('--time=06:00:00', '--time=03:00:00').replace('--signal=B:USR1@900', '--signal=B:USR1@180')
    script = script.replace('scripts.mni_constraints.verified_lower_lr_full_matrix', MODULE)
    (stage/'run.sbatch').write_text(script)
    for f in stage.rglob('*'):
        if f.is_file() and original.get(str(f.relative_to(stage))) == digest(f):
            donors[str(f.relative_to(stage))] = str(REMOTE/PARENT/f.relative_to(stage))
    for model in MODELS:
        origin = parent/'protocols'/f'{model}_dice_bands'
        for c in CANDIDATES[model]:
            for f in (stage/'protocols'/f'{model}_{c[0]}').iterdir():
                if not f.is_file() or f.name == 'protocol.json':
                    continue
                source_name = 'protocol.json' if f.name == 'parent_protocol.json' else f.name
                assert digest(f) == digest(origin/source_name)
                donors[str(f.relative_to(stage))] = str(REMOTE/PARENT/'protocols'/f'{model}_dice_bands'/source_name)
    donors.pop('source/source_manifest.json', None)
    donors['parent_manifest.json'] = str(REMOTE/PARENT/'manifest.json')
    write(stage/'deployment_known_donors.json', donors)
    campaign.freeze(stage)
    return verify(stage)


def verify(stage):
    for name, sha in read(stage/'manifest.json').items():
        assert digest(stage/name) == sha, name
    plan, original = read(stage/'plan.json'), read(stage/'parent_manifest.json')
    assert plan['tasks'] == tasks() and plan['gpu_workers'] == 2 and plan['budget_flops'] == CAP
    assert plan['deadline'] == 1794252600.0 and plan['deadline']-plan['started_at'] <= 14400
    for name, sha in original.items():
        if name.startswith('source/') and name != 'source/source_manifest.json':
            assert digest(stage/name) == sha, name
    for model in MODELS:
        for c in CANDIDATES[model]:
            d = stage/'protocols'/f'{model}_{c[0]}';p = read(d/'protocol.json')
            assert digest(d/'parent_protocol.json') == original[f'protocols/{model}_dice_bands/protocol.json']
            assert p == protocol(read(d/'parent_protocol.json'), read(d/'profile.json'), c)
            s = p['models'][model]['objectives']['dice_bands'];n = s['max_steps']
            assert campaign.training_flops(n, s) <= CAP < campaign.training_flops(n+1, s)
            assert n == (149 if model == 'nnunet' else 343)
            assert p['lr_horizon_updates'] == (200 if model == 'nnunet' else 343)
            assert p['initial_lrs'][model] == (.02 if model == 'nnunet' else .0003)
            for file, key in [('profile.json','profile_sha256'), ('splits_10patients.json','splits_sha256'), ('nnunet_architecture.json','nnunet_architecture_sha256')]:
                assert digest(d/file) == p[key]
            assert digest(stage/'data/MNI.pkl') == p['pkl_sha256']
    controls = read(stage/'controls.json');assert len(controls) == 36
    for c in controls:
        d = stage/'controls'/c['id'];verify_archive(d, read(d/'archive_manifest.json'))
        assert digest(d/'archive_manifest.json') == c['archive_sha256']
        cfg, done = read(d/'config.json'), read(d/'completion.json')
        assert done['status'] == 'complete' and done['discarded_attempts'] == 0
        assert done['updates'] == done['training_attempts_charged'] == cfg['max_steps']
        assert cfg['budget_flops'] == CAP and done['consumed_training_flops'] <= CAP
        assert (cfg['model_identity'],cfg['seed'],cfg['fold'],cfg['objective']) == (c['model'],c['seed'],c['fold'],c['arm'])
    return plan


def collect_rows(stage):
    controls = {(c['model'],c['seed'],c['fold'],c['arm']):c for c in read(stage/'controls.json')}
    result = []
    for t in tasks():
        d = stage/'runs'/t['id']
        if not (d/'archive_manifest.json').exists():
            continue
        done = read(d/'completion.json');assert done['status'] == 'complete' and done['discarded_attempts'] == 0
        v = read(d/'validation_latest.json')['summary']
        b = read(stage/'controls'/controls[t['model'],t['seed'],t['fold'],'dice']['id']/'validation_latest.json')['summary']
        old = read(stage/'controls'/controls[t['model'],t['seed'],t['fold'],'dice_bands']['id']/'validation_latest.json')['summary']
        result.append(dict(**t, dice=100*v['macro_hard_dice'], baseline_dice=100*b['macro_hard_dice'],
            original_bands_dice=100*old['macro_hard_dice'], gain_pp=100*(v['macro_hard_dice']-b['macro_hard_dice']),
            gain_vs_original_pp=100*(v['macro_hard_dice']-old['macro_hard_dice']),
            ca4_gain_vs_original_pp=100*(v['CA4_DG_hard_dice']-old['CA4_DG_hard_dice']),
            per_class_gain_pp={k:100*(v[k]-b[k]) for k in ('CA1_3_hard_dice','subiculum_hard_dice','CA4_DG_hard_dice')},
            archive_sha256=digest(d/'archive_manifest.json')))
    return result


def select(rows):
    """Rank only on fixed screen pairs; never retune per seed or fold."""
    results = {};pending = []
    for model in MODELS:
        screen_tasks = [t for t in tasks() if t['model'] == model and t['screen']]
        completed = {r['id'] for r in rows}
        missing = [t for t in screen_tasks if t['id'] not in completed]
        if missing:
            results[model] = dict(phase='screen', pending=len(missing));pending.extend(missing);continue
        ranking = []
        for key, _, _ in CANDIDATES[model]:
            g = [r for r in rows if r['model'] == model and r['candidate'] == key and r['screen']]
            eligible = all(r['gain_pp'] > 0 for r in g)
            if model == 'swinunetr':
                eligible &= mean(r['gain_vs_original_pp'] for r in g) > 0 and mean(r['ca4_gain_vs_original_pp'] for r in g) >= 0
            if eligible:
                score = ((min(r['gain_pp'] for r in g), mean(r['gain_pp'] for r in g)) if model == 'nnunet'
                         else (mean(r['gain_vs_original_pp'] for r in g), min(r['gain_pp'] for r in g)))
                ranking.append((score, key))
        ranking.sort(reverse=True);winner = None;selected = None;rejected = []
        for _, key in ranking:
            g = [r for r in rows if r['model'] == model and r['candidate'] == key]
            if any(r['gain_pp'] <= 0 for r in g):
                rejected.append(key);continue
            if len(g) == 9:
                if model == 'swinunetr' and (mean(r['gain_vs_original_pp'] for r in g) <= 0 or mean(r['ca4_gain_vs_original_pp'] for r in g) < 0):
                    rejected.append(key);continue
                winner = key;break
            selected = key
            pending.extend(t for t in tasks() if t['model'] == model and t['candidate'] == key and t['id'] not in completed)
            break
        results[model] = dict(phase='confirmed' if winner else 'confirm' if selected else 'exhausted',
            winner=winner, selected=selected, ranking=[k for _,k in ranking], rejected=rejected)
    # Interleave models so both get compute before the absolute deadline.
    groups = [[t for t in pending if t['model'] == m] for m in MODELS]
    interleaved = [g[i] for i in range(max(map(len,groups),default=0)) for g in groups if i < len(g)]
    return dict(models=results, pending=interleaved, finished=not pending)


def preflight(stage):
    verify(stage)
    import torch, monai
    from scripts.mni_constraints import objectives
    from scripts.unetr_subfields import core
    from scripts.unetr_subfields.contracts import DATASETS
    torch.set_num_threads(4);assert monai.__version__ == '1.6.0'
    frame = core.base._load_pkl_dataframe(stage/'data/MNI.pkl');checks = []
    for model in MODELS:
        d = stage/'protocols'/f'{model}_{CANDIDATES[model][0][0]}';p = read(d/'protocol.json');profile = read(d/'profile.json')
        assert digest(Path(objectives.__file__)) == profile['objective_sha256']
        trainer = batch.bind(SimpleNamespace(protocol_dir=d, model=model, objective='dice_bands', fold=0))
        # Swin initialization reads scalar tensor values, so construct on CPU
        # before moving to meta for the allocation-free shape check.
        net = trainer.base.build_model(model, read(d/'nnunet_architecture.json')).to('meta')
        out = net(torch.zeros((1,1,64,64,64),device='meta'))
        shapes = [list(x.shape) for x in out] if isinstance(out,(list,tuple)) else [list(out.shape)]
        assert shapes == profile['output_shapes'] and sum(x.numel() for x in net.parameters()) == profile['parameters']
        criterion, optimizer = trainer.base.build_loss_and_optimizer(model, net, p)
        assert optimizer.param_groups[0]['lr'] == p['initial_lrs'][model]
        if model == 'nnunet':
            assert isinstance(optimizer,torch.optim.SGD) and criterion.loss.weight_ce == 1.
        else:
            assert isinstance(optimizer,torch.optim.AdamW)
        for fold, split in enumerate(read(d/'splits_10patients.json')):
            train = core.PreparedSubfields(frame, split['train'], DATASETS['MNI']);assert len(train) == 20
            assert len(core.PreparedSubfields(frame, split['val'], DATASETS['MNI'])) == 10
            torch.manual_seed(113);z = [torch.randn(shape,requires_grad=True) for shape in shapes]
            if model != 'nnunet':z = z[0]
            label = train[0]['label'][None];native = criterion if model == 'nnunet' else None
            row = objectives.calibration_case(z,label,'dice_bands',criterion=native,identity=model)
            for target in sorted({c[2] for c in CANDIDATES[model]}):
                weights,stats = objectives.select_weights([deepcopy(row)],'dice_bands',target,.75)
                loss = objectives.training_loss(z,label,'dice_bands',weights,1.,criterion=native,identity=model)
                assert torch.isfinite(loss) and torch.isfinite(torch.autograd.grad(loss,z[0] if isinstance(z,list) else z)[0]).all()
                assert stats['bands']['weighted_median_ratio'] <= target+1e-8
                checks.append(dict(model=model,fold=fold,target=target,parameters=profile['parameters']))
    result = dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),checks=checks,optimizer_updates=0)
    write(stage/'preflight.json',result);return result


def main():
    from scripts.mni_constraints import bounded_refinement_runner as runner
    runner.study = __import__(__name__, fromlist=["run"])
    run = runner.run
    campaign.MODULE = MODULE
    campaign.prepare,campaign.verify,campaign.preflight = prepare,verify,preflight
    campaign.train = headroom_bands.train;batch.run = run
    campaign.main()


if __name__ == '__main__':
    main()
