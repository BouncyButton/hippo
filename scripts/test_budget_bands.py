"""Check native deep supervision, calibration purity, and charged-budget recovery."""
import copy
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import pytest
import torch

import nnunet_band_objective as native
import train_compute_matched_constraints as trainer
from compute_constraint_objectives import scheduled_costs, select_weights
from test_compute_matched_constraints import labels, campaign, args
from train_budget_bands import validation_updates
import run_budget_bands as controller


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def criterion():
    model = torch.nn.Conv3d(1, 3, 1)
    return trainer.base.build_loss_and_optimizer('nnunet', model,
        {'initial_lrs': {'nnunet': .01}, 'sgd_weight_decay': 3e-5, 'nnunet_sgd_momentum': .95})[0]


def test_native_bands_preserve_deep_supervision_and_change_only_full_resolution_gradient():
    y = labels()
    generator = torch.Generator().manual_seed(31)
    outputs = [torch.randn(1, 3, size, size, size, generator=generator, requires_grad=True)
               for size in [16, 8, 4, 2]]
    loss_fn = criterion()
    baseline = native.native_loss(outputs, y, loss_fn)
    assert torch.equal(native.training_loss(outputs, y, 'dice', {}, .2, loss_fn), baseline)
    constrained = native.training_loss(outputs, y, 'dice_bands', {'bands': .1}, 1, loss_fn)
    plain_grads = torch.autograd.grad(baseline, outputs, retain_graph=True, allow_unused=True)
    band_grads = torch.autograd.grad(constrained, outputs, allow_unused=True)
    assert not torch.equal(plain_grads[0], band_grads[0])
    assert all(torch.equal(plain_grads[i], band_grads[i]) for i in [1, 2])
    assert plain_grads[3] is None and band_grads[3] is None
    row = native.calibration_case(outputs, y, 'dice_bands', loss_fn)
    assert row['dice_rms'] == pytest.approx(float(plain_grads[0].square().mean().sqrt()))
    weights, stats = select_weights([dict(row) for _ in range(10)], 'dice_bands')
    assert weights['bands'] > 0
    assert stats['bands']['weighted_median_ratio'] <= .1 + 1e-8
    assert stats['bands']['weighted_p95_ratio'] <= .5 + 1e-8
    assert all(o.grad is None for o in outputs)
    with pytest.raises(ValueError, match='valid inner'):
        native.calibration_case(outputs, torch.zeros_like(y), 'dice_bands', loss_fn)


class TinyDeepSupervision(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0., .1, -.1]))

    def forward(self, image):
        full = self.logits[None, :, None, None, None].expand(image.shape[0], 3, *image.shape[2:])
        return [full] + [torch.nn.functional.avg_pool3d(full, size) for size in [2, 4, 8]]


def test_native_calibration_preserves_model_gradients_mode_and_rng():
    class StochasticTiny(TinyDeepSupervision):
        def forward(self, image):
            torch.rand(1)  # Calibration must restore even consumed random state.
            return super().forward(image)
    model = StochasticTiny().train()
    model.logits.grad = torch.tensor([.2, .3, .4])
    gradient = model.logits.grad.clone()
    anchor = trainer.model_digest(model)
    state = torch.get_rng_state().clone()
    patients = [{'image': torch.ones(1, 16, 16, 16), 'label': labels()[0], 'case_name': 'train'}]
    loss_fn = criterion()
    state = torch.get_rng_state().clone()
    report = trainer.calibrate(model, patients, 'dice_bands', torch.device('cpu'), False, loss_fn)
    assert report['status'] == 'complete' and model.training
    assert trainer.model_digest(model) == anchor
    assert torch.equal(model.logits.grad, gradient)
    assert torch.equal(torch.get_rng_state(), state)


def test_native_calibration_retry_resume_and_gradient_clipping_stay_in_budget(tmp_path):
    protocol_dir, pkl = campaign(tmp_path)
    protocol = json.loads((protocol_dir / 'protocol.json').read_text())
    protocol['models']['nnunet'] = copy.deepcopy(protocol['models']['unet'])
    protocol.update(nnunet_sgd_momentum=.95, nnunet_warmup_fraction=0)
    protocol['validation_target_flops'] = [100 * k // 20 for k in range(1, 21)]
    (protocol_dir / 'protocol.json').write_text(json.dumps(protocol))
    run_args = args(tmp_path, protocol_dir, pkl, 'dice_bands')
    run_args.model = 'nnunet'
    run_args.output = tmp_path / 'nnunet/dice_bands/fold0/seed0/run'
    clip = torch.nn.utils.clip_grad_norm_
    with patch.object(trainer.base, 'build_model', lambda *a: TinyDeepSupervision()), \
            patch.object(torch.nn.utils, 'clip_grad_norm_', wraps=clip) as clipping:
        with patch.object(trainer, 'calibrate', side_effect=RuntimeError('interrupted calibration')):
            with pytest.raises(RuntimeError, match='interrupted calibration'):
                trainer.run(run_args)
        trainer.run(run_args)
        count = clipping.call_count
        assert count > 0
        assert all(c.kwargs['error_if_nonfinite'] and c.args[1] == 12 for c in clipping.call_args_list)
        trainer.run(run_args)
        assert clipping.call_count == count
    done = json.loads((run_args.output / 'completion.json').read_text())
    assert done['calibration_attempts'] == 2 and done['calibration_flops'] == 24
    assert done['consumed_total_flops'] <= 100 and 0 <= done['unused_flops'] < 12
    assert done['updates'] == 6 and done['training_attempts_charged'] == 6
    report = json.loads((run_args.output / 'calibration.json').read_text())
    assert report['anchor_training_attempts'] == 1 and len(report['cases']) == 10
    assert set(report['weights']) == {'bands'}
    assert 'native Dice+CE' in report['scope']
    # Exercise controller verification with the completed interrupted run.
    # Training still used real native Dice+CE and real bands, on a tiny network.
    for name in ['nnunet_band_objective.py', 'train_budget_bands.py']:
        shutil.copyfile(Path(trainer.__file__).parent / name, tmp_path / 'source/scripts' / name)
    controller.write_json(run_args.output / 'runtime_recipe.json', {
        'adapter_sha256': controller.digest(tmp_path / 'source/scripts/train_budget_bands.py'),
        'protocol_sha256': controller.digest(protocol_dir / 'protocol.json'),
        'validation_updates': validation_updates(protocol, 'nnunet')})
    assert controller.verify_run(tmp_path, 'nnunet', protocol_dir)['calibration_attempts'] == 2
    done['consumed_total_flops'] += 1
    controller.write_json(run_args.output / 'completion.json', done)
    with pytest.raises(ValueError, match='budget mismatch'):
        controller.verify_run(tmp_path, 'nnunet', protocol_dir)
    done['consumed_total_flops'] -= 1
    controller.write_json(run_args.output / 'completion.json', done)
    (run_args.output / 'calibration.json').write_text('{}')
    with pytest.raises(ValueError, match='Changed completed artifact'):
        controller.verify_run(tmp_path, 'nnunet', protocol_dir)


def test_common_compute_milestones_deduct_calibration_and_prefix():
    schedule = scheduled_costs(1000, 10, 12, 100, 5)
    schedule.update(baseline_flops_per_update=10, flops_per_update=12)
    protocol = {'budget_flops': 1000, 'models': {'nnunet': {'objectives': {'dice_bands': schedule}}},
                'validation_target_flops': [40, 80, 160, 1000]}
    assert validation_updates(protocol, 'nnunet') == [4, 5, 75]
    assert validation_updates(protocol, 'nnunet', max_steps=2) == [2]
    protocol['validation_target_flops'] = [1001]
    with pytest.raises(ValueError, match='Invalid validation'):
        validation_updates(protocol, 'nnunet')
