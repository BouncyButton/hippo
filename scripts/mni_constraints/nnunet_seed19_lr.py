"""Fixed-budget lower-LR pilots and held-out-fold checks for nnUNet seed19."""
import shutil
from types import SimpleNamespace

from scripts.cross_dataset_campaign.common import read, write, digest, REMOTE
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints import campaign, headroom_bands, seed_recovery as runner
from scripts.mni_constraints.shared43_study import ROOT, NAME as PARENT, CAP
from scripts.unetr_subfields import cap_extension as batch

NAME = 'mni_nnunet_seed19_lowerlr_shared43t_20261108'
MODULE = 'scripts.mni_constraints.nnunet_seed19_lr'
PREVIOUS = 'mni_seed19_all_models_shared43t_20261108'
RATES = (.005, .0025, .001, .0005)


def key(lr):
    return f'nnunet_lr{round(lr * 100000):05d}'


def tasks():
    return [dict(id=f'MNI_{key(lr)}_f{fold}s19', kind='conventional', dataset='MNI',
                 model='nnunet', arm='dice', fold=fold, seed=19, protocol=key(lr),
                 initial_lr=lr, augmentation='none') for fold in range(3) for lr in RATES]


def prepare(_target=None):
    stage = ROOT / 'experiments' / NAME
    parent = ROOT / 'experiments' / PARENT
    if (stage / 'manifest.json').exists():
        return verify(stage)
    for name, sha in read(parent / 'manifest.json').items():
        assert digest(parent / name) == sha, name
    for part in ('source', 'data'):
        shutil.copytree(parent / part, stage / part, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    source_names = {'nnunet_seed19_lr.py', 'seed_recovery.py', 'collect_seed_recovery.py', MODULE.rsplit('.', 1)[-1] + '.py'}
    for name in sorted(source_names):
        shutil.copy2(ROOT / 'scripts/mni_constraints' / name, stage / 'source/scripts/mni_constraints' / name)
    shutil.copy2(parent / 'manifest.json', stage / 'parent_manifest.json')
    rule = (f'{len(RATES)} lower initial learning rates on fold0 seed19 at the unchanged43T cap. '
            'Select the largest final-budget Dice improvement over the frozen0.01 control; '
            'ties prefer the larger learning rate. Confirm that one candidate on folds1,2. '
            'If no pilot improves, stop. Retain every attempted result. No other model or constraint run.')
    for lr in RATES:
        folder = stage / 'protocols' / key(lr)
        shutil.copytree(parent / 'protocols/nnunet_dice', folder)
        shutil.copy2(folder / 'protocol.json', folder / 'parent_protocol.json')
        p = read(folder / 'protocol.json')
        p.update(name=NAME, data=str(REMOTE / NAME / 'data/MNI.pkl'), seeds=[19],
                 initial_lrs={'nnunet': lr}, decision_rule=rule)
        write(folder / 'protocol.json', p)
    controls = []
    donors = {}
    for fold in range(3):
        identifier = f'MNI_nnunet_dice_f{fold}s19'
        src = ROOT / 'reports' / PREVIOUS / 'runs' / identifier
        verify_archive(src, read(src / 'archive_manifest.json'))
        assert read(src / 'cleanup.json')['status'] == 'complete'
        dst = stage / 'controls' / identifier
        shutil.copytree(src, dst)
        for f in dst.iterdir():
            if f.is_file() and f.name != 'archive_acknowledgment.json':
                donors[str(f.relative_to(stage))] = str(REMOTE / PREVIOUS / 'runs' / identifier / f.name)
        controls.append(dict(id=identifier, fold=fold, seed=19,
                             validation_dice=read(src / 'validation_latest.json')['summary']['macro_hard_dice'],
                             archive_manifest_sha256=digest(src / 'archive_manifest.json')))
    write(stage / 'controls.json', controls)
    write(stage / 'deployment_known_donors.json', donors)
    plan = dict(name=NAME, tasks=tasks(), num_runs=len(tasks()), maximum_executed_runs=len(RATES) + 2,
                budget_flops=CAP, gpu_workers=2, models=['nnunet'], folds=[0, 1, 2], seeds=[19],
                augmentation='none', initial_lrs=list(RATES),
                conventional_python=read(parent / 'plan.json')['conventional_python'],
                graceful_stop_seconds=20700, archive_wait_timeout_seconds=600, archive_resume_module=MODULE,
                storage=dict(minimum_free_gib=16, estimated_peak_mirrored_gib=4, archive=str(ROOT / 'reports' / NAME)),
                selection_rule=rule,
                selection_caveat='Exploratory development-validation tuning on fold0. Folds overlap; this is not an independent test estimate. Changing LR creates a new recipe and does not validate the original recipe or other seeds.',
                authorization=f'User requests lower nnUNet LR for chosen seed19 within the chosen FLOPs cap to improve performance. {len(RATES)} fold0 pilots, at most two confirmation runs, unchanged data/runtime; verified per-run archive and cleanup.')
    write(stage / 'plan.json', plan)
    write(stage / 'control.json', dict(pause=False))
    write(stage / 'source_version.json', dict(parent_manifest_sha256=digest(parent / 'manifest.json'),
          changes='Training runtime unchanged. Only initial nnUNet LR changes; preserve200-update horizon,100-update hold,121 budgeted updates.'))
    (stage / 'run.sbatch').write_text((parent / 'run.sbatch').read_text().replace(
        'scripts.mni_constraints.shared43_study', MODULE).replace('hippo-mni-14', 'hippo-nn19-lr'))
    campaign.freeze(stage)
    return verify(stage)


def verify(stage):
    for name, sha in read(stage / 'manifest.json').items():
        assert digest(stage / name) == sha, name
    parent = read(stage / 'parent_manifest.json')
    plan = read(stage / 'plan.json')
    assert plan['tasks'] == tasks() and plan['budget_flops'] == CAP and plan['gpu_workers'] == 2
    allowed = {'name', 'data', 'seeds', 'initial_lrs', 'decision_rule'}
    for lr in RATES:
        folder = stage / 'protocols' / key(lr)
        p, old = read(folder / 'protocol.json'), read(folder / 'parent_protocol.json')
        assert digest(folder / 'parent_protocol.json') == parent['protocols/nnunet_dice/protocol.json']
        assert {k: v for k, v in p.items() if k not in allowed} == {k: v for k, v in old.items() if k not in allowed}
        assert p['initial_lrs'] == {'nnunet': lr} and p['seeds'] == [19]
        assert p['budget_flops'] == CAP and p['lr_horizon_updates'] == 200
        assert p['models']['nnunet']['objectives']['dice']['max_steps'] == 121
        for name, k in [('profile.json', 'profile_sha256'), ('splits_10patients.json', 'splits_sha256'), ('nnunet_architecture.json', 'nnunet_architecture_sha256')]:
            assert digest(folder / name) == p[k]
        assert digest(stage / 'data/MNI.pkl') == p['pkl_sha256']
    for name, sha in parent.items():
        if name.startswith('source/') and name != 'source/source_manifest.json':
            assert digest(stage / name) == sha, name
    for c in read(stage / 'controls.json'):
        folder = stage / 'controls' / c['id']
        verify_archive(folder, read(folder / 'archive_manifest.json'))
        assert digest(folder / 'archive_manifest.json') == c['archive_manifest_sha256']
    return plan


def preflight(stage):
    verify(stage)
    import torch, monai
    from scripts.unetr_subfields import core
    from scripts.unetr_subfields.contracts import DATASETS
    torch.set_num_threads(4)
    assert monai.__version__ == '1.6.0'
    folder = stage / 'protocols' / key(RATES[0])
    batch.bind(SimpleNamespace(protocol_dir=folder, model='nnunet', objective='dice', fold=0))
    model = core.base.build_model('nnunet', read(folder / 'nnunet_architecture.json')).to('meta').train()
    outputs = model(torch.zeros((1, 1, 64, 64, 64), device='meta'))
    profile = read(folder / 'profile.json')
    assert [list(x.shape) for x in outputs] == profile['output_shapes']
    assert sum(x.numel() for x in model.parameters()) == profile['parameters']
    frame = core.base._load_pkl_dataframe(stage / 'data/MNI.pkl')
    for split in read(folder / 'splits_10patients.json'):
        assert len(core.PreparedSubfields(frame, split['train'], DATASETS['MNI'])) == 20
        assert len(core.PreparedSubfields(frame, split['val'], DATASETS['MNI'])) == 10
    result = dict(status='passed', manifest_sha256=digest(stage / 'manifest.json'), optimizer_updates=0,
                  rates=list(RATES), updates_per_run=121, torch=torch.__version__, monai=monai.__version__)
    write(stage / 'preflight.json', result)
    return result


def selection(stage, plan):
    controls = {c['fold']: c for c in read(stage / 'controls.json')}
    rows = []
    for task in plan['tasks']:
        folder = stage / 'runs' / task['id']
        if not (folder / 'cleanup.json').exists():
            continue
        val = read(folder / 'validation_latest.json')['summary']['macro_hard_dice']
        rows.append(dict(**task, validation_dice=val,
                         gain_pp=100 * (val - controls[task['fold']]['validation_dice'])))
    pilots = [r for r in rows if r['fold'] == 0]
    chosen = None
    if len(pilots) == len(RATES):
        promising = [r for r in pilots if r['gain_pp'] > 0]
        if promising:
            chosen = max(promising, key=lambda r: (r['validation_dice'], r['initial_lr']))['initial_lr']
    chosen_rows = [r for r in rows if r['initial_lr'] == chosen]
    complete = len(pilots) == len(RATES) and (chosen is None or len(chosen_rows) == 3)
    return dict(rows=rows, selected=dict(initial_lr=chosen), complete=complete,
                passes_every_fold50=complete and len(chosen_rows) == 3 and all(r['validation_dice'] >= .5 for r in chosen_rows),
                improves_every_fold=complete and len(chosen_rows) == 3 and all(r['gain_pp'] > 0 for r in chosen_rows),
                rule=plan['selection_rule'], caveat=plan['selection_caveat'])


def eligible(stage, plan, state):
    completed = {r['id'] for r in state['rows']}
    pilots = [t for t in plan['tasks'] if t['fold'] == 0 and t['id'] not in completed]
    if pilots:
        return pilots
    return [t for t in plan['tasks'] if t['initial_lr'] == state['selected']['initial_lr'] and t['id'] not in completed]


def main():
    runner.MODULE = MODULE
    runner.RETIREMENTS = ()
    runner.verify, runner.selection, runner.eligible = verify, selection, eligible
    campaign.MODULE = MODULE
    campaign.prepare, campaign.verify, campaign.preflight = prepare, verify, preflight
    campaign.train = headroom_bands.train
    batch.run = runner.run
    campaign.main()


if __name__ == '__main__':
    main()
