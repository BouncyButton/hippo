"""Fixed-seed, fixed-budget nnUNet optimization/loss ablations and confirmation."""
import argparse
from copy import deepcopy
import difflib
import shutil
from types import SimpleNamespace

from scripts.cross_dataset_campaign.common import read, write, digest, REMOTE
from scripts.mni_constraints import campaign, headroom_bands
from scripts.mni_constraints.common53_study import ROOT, NAME as PARENT, CAP
from scripts.unetr_subfields import cap_extension as batch

MODULE = 'scripts.mni_constraints.nnunet_repair'
PILOT = 'mni_nnunet_repair_pilot53t_20261108'
CONFIRM = 'mni_nnunet_repair_confirm53t_20261108'
RECIPES = dict(lr002=dict(lr=.02, ce=1.), ce01=dict(lr=.01, ce=.1))
CORE = 'source/scripts/cross_dataset_campaign/capped_core.py'


def runtime_source(source):
    marker = '    criterion, optimizer = base.build_loss_and_optimizer(args.model, model, protocol)\n'
    assert source.count(marker) == 1
    source = source.replace(marker, marker +
        '    assert args.model == "nnunet"\n'
        '    criterion.loss.weight_ce = protocol["nnunet_ce_weight"]\n')
    marker = '                             nnunet_sgd_momentum=optimizer.param_groups[0]["momentum"],\n'
    assert source.count(marker) == 1
    return source.replace(marker, marker + '                             nnunet_ce_weight=criterion.loss.weight_ce,\n')


def task_list(recipes, pairs):
    return [dict(id=f'MNI_nnunet_{recipe}_f{fold}s{seed}', kind='conventional', dataset='MNI',
                 model='nnunet', arm='dice', recipe=recipe, fold=fold, seed=seed,
                 protocol=recipe, augmentation='none') for fold, seed in pairs for recipe in recipes]


def changed_protocol(old, name, recipe, pairs):
    p = deepcopy(old)
    setting = RECIPES[recipe]
    p.update(name=name, data=str(REMOTE / name / 'data/MNI.pkl'),
        seeds=sorted({s for _, s in pairs}), folds=sorted({f for f, _ in pairs}),
        initial_lrs={'nnunet': setting['lr']}, nnunet_ce_weight=setting['ce'],
        training_loss=f"Native foreground Dice + {setting['ce']}*CE, original deep supervision; SGD/Nesterov momentum0.95",
        decision_rule='Final-budget validation macro Dice>=55% in every established seed/fold. Pilots change one variable each on seed19 fold0. Prefer the LR-only recipe if both pilots pass, then freeze that recipe across seeds0,5,19 and folds0,1,2. Retain all attempted results; no seed replacements.',
        loss_balance_flops_audit='loss_balance_flops_audit.json')
    return p


def prepare_stage(name, recipes, pairs):
    assert name in (PILOT, CONFIRM)
    stage, parent = ROOT / 'experiments' / name, ROOT / 'experiments' / PARENT
    if (stage / 'manifest.json').exists():
        return verify(stage)
    original = read(parent / 'manifest.json')
    for path, sha in original.items():
        assert digest(parent / path) == sha, path
    for part in ('source', 'data'):
        shutil.copytree(parent / part, stage / part, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(ROOT / 'scripts/mni_constraints/nnunet_repair.py', stage / 'source/scripts/mni_constraints/nnunet_repair.py')
    shutil.copy2(parent / 'manifest.json', stage / 'parent_manifest.json')
    ref = stage / 'reference_source/capped_core.py'
    ref.parent.mkdir(parents=True)
    shutil.copy2(stage / CORE, ref)
    before, after = ref.read_text(), runtime_source(ref.read_text())
    (stage / CORE).write_text(after)
    (stage / 'runtime.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='original', tofile=CORE)))
    audit = ROOT / 'reports' / PILOT / 'loss_balance_flops_audit.json'
    assert read(audit)['status'] == 'passed'
    shutil.copy2(audit, stage / 'loss_balance_flops_audit.json')
    for recipe in recipes:
        folder = stage / 'protocols' / recipe
        shutil.copytree(parent / 'protocols/nnunet_dice', folder)
        shutil.copy2(folder / 'protocol.json', folder / 'parent_protocol.json')
        write(folder / 'protocol.json', changed_protocol(read(folder / 'parent_protocol.json'), name, recipe, pairs))
    evidence = ROOT / 'reports/mni_nnunet_seed19_instrumented53t_f0_20261108/diagnostic_summary.json'
    shutil.copy2(evidence, stage / 'diagnostic_evidence.json')
    plan = dict(name=name, tasks=task_list(recipes, pairs), recipes=recipes, pairs=pairs,
        num_runs=len(recipes) * len(pairs), models=['nnunet'], seeds=sorted({s for _, s in pairs}), folds=sorted({f for f, _ in pairs}),
        budget_flops=CAP, gpu_workers=2, augmentation='none',
        conventional_python=read(parent / 'plan.json')['conventional_python'], graceful_stop_seconds=20700,
        archive_wait_timeout_seconds=600, archive_resume_module=MODULE,
        storage=dict(minimum_free_gib=16, estimated_peak_mirrored_gib=6, archive=str(ROOT / 'reports' / name)),
        acceptance_threshold=.55,
        authorization='User requests continued diagnosis and fixes until nnUNet reaches55% upwards under the latest agreed cap. Keep established seeds0,5,19 and folds0,1,2. One shared recipe, no seed replacement; archive/evaluate/verify/clean each run.',
        scope='Two single-variable pilots on seed19 fold0, followed by eight other seed/fold confirmations of one frozen successful recipe. Other-model sweep stays paused.',
        selection_caveat='Exploratory development-validation tuning; overlapping folds and selected seeds, not an untouched test estimate.')
    write(stage / 'plan.json', plan)
    write(stage / 'control.json', dict(pause=False))
    write(stage / 'source_version.json', dict(parent_manifest_sha256=digest(parent / 'manifest.json'),
        core_before_sha256=digest(ref), core_after_sha256=digest(stage / CORE),
        changes='Only declared initial LR or native CE coefficient changes. Same SGD/Nesterov0.95,deep supervision,data,initialization,sample order,BF16,gradient clipping,200-update LR horizon and150-update53.305T cap. Diagnostic hooks absent.'))
    (stage / 'run.sbatch').write_text((parent / 'run.sbatch').read_text().replace('scripts.mni_constraints.common53_study', MODULE).replace('hippo-common53', 'hippo-nn-repair'))
    donors = {str(f.relative_to(stage)): str(REMOTE / PARENT / f.relative_to(stage))
              for f in stage.rglob('*') if f.is_file() and original.get(str(f.relative_to(stage))) == digest(f)}
    donors.pop('source/source_manifest.json', None)
    donors['parent_manifest.json'] = str(REMOTE / PARENT / 'manifest.json')
    for recipe in recipes:
        for f in (stage / 'protocols' / recipe).iterdir():
            if f.name in ('protocol.json', 'parent_protocol.json'):
                continue
            assert digest(f) == digest(parent / 'protocols/nnunet_dice' / f.name)
            donors[str(f.relative_to(stage))] = str(REMOTE / PARENT / 'protocols/nnunet_dice' / f.name)
    write(stage / 'deployment_known_donors.json', donors)
    campaign.freeze(stage)
    return verify(stage)


def verify(stage):
    for name, sha in read(stage / 'manifest.json').items():
        assert digest(stage / name) == sha, name
    p = read(stage / 'plan.json')
    assert p['name'] in (PILOT, CONFIRM) and p['gpu_workers'] == 2 and p['budget_flops'] == CAP
    assert p['tasks'] == task_list(p['recipes'], p['pairs'])
    if p['name'] == PILOT:
        assert p['recipes'] == list(RECIPES) and p['pairs'] == [[0, 19]]
    else:
        assert len(p['recipes']) == 1 and p['recipes'][0] in RECIPES
        assert p['pairs'] == [[f, s] for s in (19, 0, 5) for f in range(3) if (f, s) != (0, 19)]
    parent = read(stage / 'parent_manifest.json')
    assert digest(stage / 'reference_source/capped_core.py') == parent[CORE]
    assert (stage / CORE).read_text() == runtime_source((stage / 'reference_source/capped_core.py').read_text())
    for name, sha in parent.items():
        if name.startswith('source/') and name not in (CORE, 'source/source_manifest.json'):
            assert digest(stage / name) == sha, name
    for recipe in p['recipes']:
        folder = stage / 'protocols' / recipe
        assert digest(folder / 'parent_protocol.json') == parent['protocols/nnunet_dice/protocol.json']
        protocol = read(folder / 'protocol.json')
        assert protocol == changed_protocol(read(folder / 'parent_protocol.json'), p['name'], recipe, p['pairs'])
        assert protocol['budget_flops'] == CAP and protocol['lr_horizon_updates'] == 200
        assert protocol['models']['nnunet']['objectives']['dice']['max_steps'] == 150
        for name, key in [('profile.json', 'profile_sha256'), ('splits_10patients.json', 'splits_sha256'), ('nnunet_architecture.json', 'nnunet_architecture_sha256')]:
            assert digest(folder / name) == protocol[key]
        assert digest(stage / 'data/MNI.pkl') == protocol['pkl_sha256']
    audit = read(stage / 'loss_balance_flops_audit.json')
    assert audit['status'] == 'passed' and audit['loss_backward_delta_flops'] == 0
    return p


def preflight(stage):
    plan = verify(stage)
    import torch
    import monai
    from scripts.unetr_subfields import core
    from scripts.unetr_subfields.contracts import DATASETS
    torch.set_num_threads(4)
    assert monai.__version__ == '1.6.0'
    frame = core.base._load_pkl_dataframe(stage / 'data/MNI.pkl')
    for recipe in plan['recipes']:
        folder = stage / 'protocols' / recipe
        p, profile = read(folder / 'protocol.json'), read(folder / 'profile.json')
        batch.bind(SimpleNamespace(protocol_dir=folder, model='nnunet', objective='dice', fold=0))
        model = core.base.build_model('nnunet', read(folder / 'nnunet_architecture.json')).to('meta').train()
        output = model(torch.zeros((1, 1, 64, 64, 64), device='meta'))
        assert [list(x.shape) for x in output] == profile['output_shapes']
        assert sum(x.numel() for x in model.parameters()) == profile['parameters']
        criterion, optimizer = core.base.build_loss_and_optimizer('nnunet', model, p)
        criterion.loss.weight_ce = p['nnunet_ce_weight']
        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]['lr'] == RECIPES[recipe]['lr']
        assert optimizer.param_groups[0]['momentum'] == .95 and optimizer.param_groups[0]['nesterov']
        assert criterion.loss.weight_dice == 1 and criterion.loss.weight_ce == RECIPES[recipe]['ce']
        for fold in plan['folds']:
            split = read(folder / 'splits_10patients.json')[fold]
            assert len(core.PreparedSubfields(frame, split['train'], DATASETS['MNI'])) == 20
            assert len(core.PreparedSubfields(frame, split['val'], DATASETS['MNI'])) == 10
    result = dict(status='passed', manifest_sha256=digest(stage / 'manifest.json'), optimizer_updates=0,
        recipes={r: RECIPES[r] for r in plan['recipes']}, updates_per_run=150, cap=CAP,
        torch=torch.__version__, monai=monai.__version__)
    write(stage / 'preflight.json', result)
    return result


def audit_loss_balance():
    import torch
    from scripts.mni_constraints.profile import audit_call, target
    from scripts.cross_dataset_campaign import conventional
    from scripts.nnunet_band_objective import native_loss
    torch.set_num_threads(4)
    origin = ROOT / 'experiments' / PARENT / 'protocols/nnunet_dice'
    p, profile = read(origin / 'protocol.json'), read(origin / 'profile.json')
    rows = []
    for geometry in ('box', 'ellipsoid'):
        for weight in (1., .1):
            torch.manual_seed(19)
            outputs = [torch.randn(shape, requires_grad=True) for shape in profile['output_shapes']]
            label = target(geometry)
            criterion, _ = conventional.ORIGINAL_LOSS_OPTIMIZER('nnunet', torch.nn.Linear(1, 1), p)
            criterion.loss.weight_ce = weight
            def call():
                loss = native_loss(outputs, label, criterion)
                loss.backward()
                assert torch.isfinite(loss)
                return float(loss)
            row, value = audit_call(call)
            rows.append(dict(geometry=geometry, ce_weight=weight, loss=value, **row))
    counts = {r['flops'] for r in rows}
    assert len(counts) == 1
    result = dict(status='passed', loss_backward_delta_flops=0, loss_backward_flops=counts.pop(),
        unchanged_update_flops=355367397283, updates=150, cap=CAP, rows=rows,
        scope='Only CE scalar coefficient changes; same native Dice+CE modules,network backward and SGD. LR changes have no tensor arithmetic effect. Loss/backward arithmetic audited under the original torch2.2.2 accounting on two target geometries.',
        source_sha256=digest(__import__('pathlib').Path(__file__)))
    write(ROOT / 'reports' / PILOT / 'loss_balance_flops_audit.json', result)
    print({k: v for k, v in result.items() if k != 'rows'})


def main():
    import sys
    if sys.argv[1] == 'audit-loss':
        audit_loss_balance()
        return
    if sys.argv[1] == 'prepare-repair':
        parser = argparse.ArgumentParser()
        parser.add_argument('command')
        parser.add_argument('--confirm', choices=list(RECIPES))
        args = parser.parse_args()
        if args.confirm:
            pairs = [[f, s] for s in (19, 0, 5) for f in range(3) if (f, s) != (0, 19)]
            result = prepare_stage(CONFIRM, [args.confirm], pairs)
        else:
            result = prepare_stage(PILOT, list(RECIPES), [[0, 19]])
        print(__import__('json').dumps(result))
        return
    campaign.MODULE = MODULE
    campaign.verify, campaign.preflight = verify, preflight
    campaign.train = headroom_bands.train
    campaign.main()


if __name__ == '__main__':
    main()
