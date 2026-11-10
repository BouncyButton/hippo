"""One unchanged 150-update nnUNet run with observational gradient probes."""
from copy import deepcopy
import difflib
import shutil
import unittest
from types import SimpleNamespace

from scripts.cross_dataset_campaign.common import read, write, digest, REMOTE
from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints import campaign, headroom_bands
from scripts.mni_constraints.common53_study import ROOT, NAME as PARENT, CAP
from scripts.mni_constraints.optimizer_diagnostics import instrument_source, PROBE_STEPS
from scripts.unetr_subfields import cap_extension as batch

NAME = 'mni_nnunet_seed19_instrumented53t_f0_20261108'
MODULE = 'scripts.mni_constraints.nnunet_seed19_instrumented'
IDENTIFIER = 'MNI_nnunet_dice_f0s19'
CORE = 'source/scripts/cross_dataset_campaign/capped_core.py'


def tasks():
    return [dict(id=IDENTIFIER, kind='conventional', dataset='MNI', model='nnunet',
                 arm='dice', fold=0, seed=19, protocol='nnunet_dice', augmentation='none')]


def protocol(old):
    p = deepcopy(old)
    p.update(name=NAME, data=str(REMOTE / NAME / 'data/MNI.pkl'), seeds=[19], folds=[0],
             decision_rule='One observational rerun. Preserve seed19 fold0,150 optimization updates,53.305T cap,LR0.01/200 horizon,SGD momentum0.95,Nesterov,native Dice+CE and all data. Measure diagnostics separately; compare the complete trajectory and final model hash with the prior run.')
    return p


def prepare(_target=None):
    stage, parent = ROOT / 'experiments' / NAME, ROOT / 'experiments' / PARENT
    if (stage / 'manifest.json').exists():
        return verify(stage)
    original = read(parent / 'manifest.json')
    for name, sha in original.items():
        assert digest(parent / name) == sha, name
    for part in ('source', 'data'):
        shutil.copytree(parent / part, stage / part, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('nnunet_seed19_instrumented.py', 'optimizer_diagnostics.py', 'test_optimizer_diagnostics.py'):
        shutil.copy2(ROOT / 'scripts/mni_constraints' / name, stage / 'source/scripts/mni_constraints' / name)
    shutil.copy2(parent / 'manifest.json', stage / 'parent_manifest.json')
    reference = stage / 'reference_source/capped_core.py'
    reference.parent.mkdir(parents=True)
    shutil.copy2(stage / CORE, reference)
    before = reference.read_text()
    after = instrument_source(before)
    (stage / CORE).write_text(after)
    (stage / 'instrumentation.patch').write_text(''.join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile='reference/capped_core.py', tofile=CORE)))
    folder = stage / 'protocols/nnunet_dice'
    shutil.copytree(parent / 'protocols/nnunet_dice', folder)
    shutil.copy2(folder / 'protocol.json', folder / 'parent_protocol.json')
    write(folder / 'protocol.json', protocol(read(folder / 'parent_protocol.json')))
    src = ROOT / 'reports' / PARENT / 'runs' / IDENTIFIER
    verify_archive(src, read(src / 'archive_manifest.json'))
    assert read(src / 'cleanup.json')['status'] == 'complete'
    shutil.copytree(src, stage / 'reference_run')
    plan = dict(name=NAME, tasks=tasks(), num_runs=1, models=['nnunet'], seeds=[19], folds=[0],
        budget_flops=CAP, gpu_workers=1, augmentation='none',
        conventional_python=read(parent / 'plan.json')['conventional_python'],
        graceful_stop_seconds=20700, archive_wait_timeout_seconds=600, archive_resume_module=MODULE,
        storage=dict(minimum_free_gib=16, estimated_peak_mirrored_gib=6, archive=str(ROOT / 'reports' / NAME)),
        parameter_probe_steps=list(PROBE_STEPS),
        authorization='User authorizes instrumented nnUNet seed19 fold0 to distinguish optimizer dynamics and loss balance before changing anything else. One fresh150-update run; broad sweep remains paused.',
        compute_accounting='Original optimization FLOPs cap unchanged; extra diagnostic gradients separately counted/profiled/timed and not presented as cap-matched performance.')
    write(stage / 'plan.json', plan)
    write(stage / 'control.json', dict(pause=False))
    write(stage / 'source_version.json', dict(parent_manifest_sha256=digest(parent / 'manifest.json'),
        reference_archive_manifest_sha256=digest(src / 'archive_manifest.json'),
        original_core_sha256=digest(reference), instrumented_core_sha256=digest(stage / CORE),
        changes='Read-only gradients before original backward; snapshot clipped gradients and actual SGD parameter deltas. No loss,optimizer,LR,data,precision or update-count changes.'))
    sbatch = (parent / 'run.sbatch').read_text().replace('scripts.mni_constraints.common53_study', MODULE)
    sbatch = sbatch.replace('hippo-common53', 'hippo-nn19-probe').replace('gpu:4g.40gb:2', 'gpu:4g.40gb:1')
    sbatch = sbatch.replace('--cpus-per-task=8', '--cpus-per-task=4').replace('--mem=48G', '--mem=32G')
    (stage / 'run.sbatch').write_text(sbatch)
    donors = {str(f.relative_to(stage)): str(REMOTE / PARENT / f.relative_to(stage))
              for f in stage.rglob('*') if f.is_file() and original.get(str(f.relative_to(stage))) == digest(f)}
    # freeze() regenerates this manifest after adding the instrumentation source.
    donors.pop('source/source_manifest.json', None)
    donors['parent_manifest.json'] = str(REMOTE / PARENT / 'manifest.json')
    for f in (stage / 'reference_run').iterdir():
        if f.is_file():
            donors[str(f.relative_to(stage))] = str(REMOTE / PARENT / 'runs' / IDENTIFIER / f.name)
    write(stage / 'deployment_known_donors.json', donors)
    campaign.freeze(stage)
    return verify(stage)


def verify(stage):
    for name, sha in read(stage / 'manifest.json').items():
        assert digest(stage / name) == sha, name
    plan = read(stage / 'plan.json')
    assert plan['tasks'] == tasks() and plan['num_runs'] == plan['gpu_workers'] == 1
    assert plan['budget_flops'] == CAP
    parent = read(stage / 'parent_manifest.json')
    original = stage / 'reference_source/capped_core.py'
    assert digest(original) == parent[CORE]
    assert (stage / CORE).read_text() == instrument_source(original.read_text())
    for name, sha in parent.items():
        if name.startswith('source/') and name not in (CORE, 'source/source_manifest.json'):
            assert digest(stage / name) == sha, name
    folder = stage / 'protocols/nnunet_dice'
    assert digest(folder / 'parent_protocol.json') == parent['protocols/nnunet_dice/protocol.json']
    p = read(folder / 'protocol.json')
    assert p == protocol(read(folder / 'parent_protocol.json'))
    assert p['models']['nnunet']['objectives']['dice']['max_steps'] == 150
    assert p['initial_lrs']['nnunet'] == .01 and p['lr_horizon_updates'] == 200
    assert digest(stage / 'data/MNI.pkl') == p['pkl_sha256']
    for name, key in [('profile.json', 'profile_sha256'), ('splits_10patients.json', 'splits_sha256'), ('nnunet_architecture.json', 'nnunet_architecture_sha256')]:
        assert digest(folder / name) == p[key]
    verify_archive(stage / 'reference_run', read(stage / 'reference_run/archive_manifest.json'))
    return plan


def preflight(stage):
    verify(stage)
    import torch
    import monai
    from scripts.unetr_subfields import core
    from scripts.unetr_subfields.contracts import DATASETS
    from scripts.mni_constraints.test_optimizer_diagnostics import InstrumentationTest
    torch.set_num_threads(4)
    assert monai.__version__ == '1.6.0'
    folder = stage / 'protocols/nnunet_dice'
    p, profile = read(folder / 'protocol.json'), read(folder / 'profile.json')
    batch.bind(SimpleNamespace(protocol_dir=folder, model='nnunet', objective='dice', fold=0))
    model = core.base.build_model('nnunet', read(folder / 'nnunet_architecture.json')).to('meta').train()
    outputs = model(torch.zeros((1, 1, 64, 64, 64), device='meta'))
    assert [list(x.shape) for x in outputs] == profile['output_shapes']
    assert sum(x.numel() for x in model.parameters()) == profile['parameters']
    frame = core.base._load_pkl_dataframe(stage / 'data/MNI.pkl')
    split = read(folder / 'splits_10patients.json')[0]
    assert len(core.PreparedSubfields(frame, split['train'], DATASETS['MNI'])) == 20
    assert len(core.PreparedSubfields(frame, split['val'], DATASETS['MNI'])) == 10
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(InstrumentationTest)
    assert unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
    result = dict(status='passed', manifest_sha256=digest(stage / 'manifest.json'),
        production_optimizer_updates=0, synthetic_equivalence_test_updates_per_arm=3,
        exact_parameter_and_momentum_equivalence=True, train_crops=20, validation_crops=10,
        torch=torch.__version__, monai=monai.__version__, updates=150, cap=CAP)
    write(stage / 'preflight.json', result)
    return result


def main():
    campaign.MODULE = MODULE
    campaign.prepare, campaign.verify, campaign.preflight = prepare, verify, preflight
    campaign.train = headroom_bands.train
    # The existing serial controller uses precisely one GPU and retains archive-before-retire.
    batch.run = controller.run
    campaign.main()


if __name__ == '__main__':
    main()
