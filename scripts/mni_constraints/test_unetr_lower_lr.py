"""Verify a pure peak-LR change against the archived full-CE cosine recipe."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import torch

from scripts.cross_dataset_campaign.common import read, write, digest, ARMS
from scripts.mni_constraints import campaign, objectives, unetr_lower_lr_study as study
from scripts.mni_constraints import unetr_plain_cosine_study as reference
from scripts.unetr_subfields import cap_extension as batch, geo_recipe


def main():
    torch.set_num_threads(4)
    runtime = study.runtime
    original = study.ROOT/'experiments'/study.source.ORIGINAL
    profile_dir = study.ROOT/'experiments'/reference.NAME/'profile_evidence'
    profile = read(profile_dir/'dropout0_original_constraints.json')
    ref_manifest = read(study.ROOT/'experiments'/reference.NAME/'manifest.json')
    assert digest(profile_dir/'dropout0_original_constraints.json') == ref_manifest[
        'profile_evidence/dropout0_original_constraints.json']
    assert profile['objective_sha256'] == digest(Path(objectives.__file__))
    for name, sha in profile['source_sha256'].items():
        assert digest(profile_dir/'source'/name) == sha
    graph_kwargs = {k:list(v) if isinstance(v, tuple) else v
                    for k, v in runtime.model_kwargs(0.).items()}
    assert graph_kwargs == profile['baseline_control']['kwargs']
    tasks = study.tasks()
    assert len(tasks) == len({t['id'] for t in tasks}) == 54
    assert len(study.CANDIDATES) == 4
    assert {t['peak_lr'] for t in tasks} == {.00045, .0003}
    allowed = {'name', 'data', 'decision_rule', 'timing_policy',
               'selection_caveat', 'primary_metric', 'initial_lrs', 'unetr_peak_lr'}
    checks = []
    with TemporaryDirectory() as temporary:
        for recipe in study.RECIPES:
            for arm in ARMS:
                old = read(original/'protocols'/('unetr_vit_b16_'+arm)/'protocol.json')
                p = study.protocol(old, profile, recipe)
                q = reference.protocol(old, profile, reference.RECIPES[0])
                differences = {k for k in set(p)|set(q) if p.get(k) != q.get(k)}
                assert differences <= allowed, differences
                assert p['initial_lrs'] == {'unet':recipe['peak_lr']}
                assert p['loss_arguments'] == geo_recipe.LOSS_ARGUMENTS
                assert p['loss_weights'] == q['loss_weights']
                schedule = p['models']['unet']['objectives'][arm]
                assert schedule == q['models']['unet']['objectives'][arm]
                assert schedule['max_steps'] == 409
                assert campaign.training_flops(409, schedule) <= study.CAP < campaign.training_flops(410, schedule)
                checks.append(dict(recipe=recipe['key'], arm=arm, updates=409))
                if arm != 'dice':
                    continue
                folder = Path(temporary)/recipe['key'];folder.mkdir()
                write(folder/'protocol.json', p)
                args = SimpleNamespace(protocol_dir=folder, model='unet', objective='dice', fold=0)
                trainer = runtime.bind(args)
                trainer.check_no_augmentation(p)
                with torch.device('meta'):
                    model = trainer.base.build_model('unet', {})
                    logits = model(torch.zeros((1,1,64,64,64)))
                assert tuple(logits.shape) == (1,4,64,64,64)
                assert sum(v.numel() for v in model.parameters()) == 92685572
                _, optimizer = trainer.base.build_loss_and_optimizer('unet', model, p)
                assert isinstance(optimizer, torch.optim.AdamW)
                assert optimizer.param_groups[0]['lr'] == recipe['peak_lr']
                for step in (0, 23, 24, 102, 205, 408):
                    lr = trainer.base.learning_rate('unet', step, 409, recipe['peak_lr'])
                    expected = batch.lr_at(step, 409, .0006, 'unetr_vit_b16', 'cap_matched')
                    assert abs(lr/expected-recipe['peak_lr']/.0006) < 1e-12
                bad = deepcopy(p);bad['unetr_peak_lr'] = .0006
                try:runtime.validate(bad)
                except AssertionError:pass
                else:raise AssertionError('Mismatched LR metadata accepted')
    torch.manual_seed(811)
    logits = torch.randn((1,4,16,16,16), requires_grad=True)
    labels = torch.randint(0,4,(1,1,16,16,16))
    actual = objectives.base_loss(logits, labels, 'unetr_vit_b16')
    expected = geo_recipe.training_loss(logits, labels, 'dice', {}, 1.)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.autograd.grad(actual, logits, retain_graph=True)[0],
                       torch.autograd.grad(expected, logits)[0])
    # Runtime binding must restore the shared trainer even when training raises.
    previous_bind, previous_train = batch.bind, runtime.ORIGINAL_TRAIN
    def fail(args):
        assert batch.bind is runtime.bind
        raise RuntimeError('intentional delegation check')
    runtime.ORIGINAL_TRAIN = fail
    try:
        try:runtime.train(None)
        except RuntimeError as error:assert str(error) == 'intentional delegation check'
        else:raise AssertionError('Exception swallowed')
        assert batch.bind is previous_bind
    finally:
        runtime.ORIGINAL_TRAIN = previous_train
    result = dict(status='passed', logical_cells=54, protocol_checks=checks,
        actual_models=2, original_base_loss_and_gradient_exact=True,
        optimizer='AdamW', peaks=[.00045,.0003], updates=409,
        profile_sha256=digest(profile_dir/'dropout0_original_constraints.json'),
        scope='Same profiled architecture, loss and optimizer graph. Only peak LR differs versus fullCE cosine; fresh baselines required. No stage prepared or job submitted.',
        test_source_sha256=digest(Path(__file__)))
    write(study.ROOT/'reports/mni_unetr_constraint_goal_20261108/lower_lr_preparation_checks.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'protocol_checks'}))


if __name__ == '__main__':
    main()
