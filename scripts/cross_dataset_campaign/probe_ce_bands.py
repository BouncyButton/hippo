"""Add bands to each matched MSD Dice+CE baseline; calibrate against Dice+CE."""
import argparse
import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.cross_dataset_campaign import conventional as experiment
from scripts.cross_dataset_campaign.common import read, write, digest
from compute_constraint_objectives import training_loss as original_loss
from nnunet_band_objective import training_loss as native_loss
from nnunet_band_objective import calibration_case as native_calibration
from thesis.new_constraints.bands.outer_boundary import OuterBoundaryBandLoss

MODELS = ('nnunet', 'swinunetr', 'unetr_vit_b16', 'basic_unet16')


def training_loss(outputs, labels, arm, weights, scale, sdf=None,
                  criterion=None, depth=None, sdf_scale=0.):
    if arm not in ('dice', 'dice_bands') or sdf is not None or depth is not None:
        raise ValueError('Only the Dice+CE / bands pilot is supported')
    if criterion is not None:
        if criterion.loss.weight_ce != 1 or criterion.loss.weight_dice != 1:
            raise ValueError('Native Dice+CE coefficients must both be one')
        return native_loss(outputs, labels, arm, weights, scale, criterion)
    return original_loss(outputs, labels, arm, weights, scale) + F.cross_entropy(
        outputs.float(), labels[:, 0].long())


def calibration_case(outputs, labels, arm, sdf=None, criterion=None, depth=None):
    if arm != 'dice_bands' or sdf is not None or depth is not None:
        raise ValueError('Expected bands calibration only')
    if criterion is not None:
        return native_calibration(outputs, labels, arm, criterion)
    z = outputs.detach().float().requires_grad_(True)
    base = training_loss(z, labels, 'dice', {}, 0.)
    gradient, = torch.autograd.grad(base, z)
    base_rms = gradient.square().mean().sqrt()
    result = OuterBoundaryBandLoss()(z, labels)
    if not bool(result.details['valid'].all()):
        raise ValueError('Invalid bands calibration geometry')
    gradient, = torch.autograd.grad(result.loss, z)
    band_rms = gradient.square().mean().sqrt()
    # Historical selector field names retained, but the denominator now includes CE.
    row = dict(dice_rms=float(base_rms), bands_rms=float(band_rms),
               bands_over_dice=float(band_rms/base_rms))
    if any(not math.isfinite(v) or v <= 0 for v in row.values()):
        raise ValueError('Nonfinite or degenerate calibration gradients')
    return row


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--protocol-dir', type=Path, required=True)
    parser.add_argument('--objective', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    p = read(args.protocol_dir/'protocol.json')
    if (p['model_identity'] not in MODELS or p['augmentation'] != 'none'
            or p['weight_ce'] != 1 or args.objective != 'dice_bands'):
        raise ValueError('Unexpected bands pilot recipe')
    experiment.configure(p, args.objective)
    experiment.base.build_loss_and_optimizer = experiment.ORIGINAL_LOSS_OPTIMIZER
    experiment.trainer.training_loss = training_loss
    experiment.trainer.calibration_case = calibration_case
    original_calibrate = experiment.trainer.calibrate
    def calibrate(*args, **kwargs):
        report = original_calibrate(*args, **kwargs)
        report.update(base_gradient='Dice+CE', legacy_dice_rms_means='Joint Dice+CE logit-gradient RMS',
            scope='All ten training volumes, no augmentation or validation; joint Dice+CE gradient versus bands at detached FP32 full-resolution logits; native deep-supervision weighting retained for nnUNet')
        return report
    experiment.trainer.calibrate = calibrate
    builder = experiment.base.build_model
    def build_model(name, architecture):
        model = builder(name, architecture)
        h = hashlib.sha256()
        for key, value in sorted(model.state_dict().items()):
            h.update(key.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        if h.hexdigest() != p['baseline_reference']['initialization_sha256']:
            raise ValueError('Model initialization differs from its Dice+CE baseline')
        write(args.output/'initialization.json', dict(model_state_sha256=h.hexdigest()))
        return model
    experiment.base.build_model = build_model
    write(args.output/'runtime_recipe.json', dict(dataset='MSD', model=p['model_identity'],
        objective='Dice+CE+bands', weight_dice=1, weight_ce=1,
        dice_include_background=p['model_identity'] != 'nnunet', augmentation='none',
        constraints=['bands'], calibration_base='Dice+CE',
        target=p['calibration_target_ratio'], p95_cap=p['calibration_p95_cap'],
        baseline_reference=p['baseline_reference'], primary_checkpoint='final_budget',
        protocol_sha256=digest(args.protocol_dir/'protocol.json')))
    experiment.trainer.main()


if __name__ == '__main__':
    main()
