"""3D UNet Dice+CE plus original bands or bands/edge/depth CE."""
import argparse
import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.cross_dataset_campaign import conventional as experiment
from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.capped_depth_delay import objectives as original

ARMS = ('dice', 'dice_bands', 'dice_bands_edge_depth')


def training_loss(logits, labels, arm, weights, scale, sdf=None,
                  criterion=None, depth=None, sdf_scale=0.):
    if arm not in ARMS or criterion is not None or sdf is not None:
        raise ValueError('Only the authorized single-output 3D UNet pilot is supported')
    return original.training_loss(logits, labels, arm, weights, scale, depth_weights=depth) + F.cross_entropy(
        logits.float(), labels[:, 0].long())


def calibration_case(outputs, labels, arm, sdf=None, criterion=None, depth=None):
    if arm not in ARMS[1:] or criterion is not None or sdf is not None:
        raise ValueError('Unexpected calibration objective')
    z = outputs.detach().float().requires_grad_(True)
    base = training_loss(z, labels, 'dice', {}, 0.)
    gradients = {'dice': torch.autograd.grad(base, z)[0]}
    bands = original.OuterBoundaryBandLoss()(z, labels)
    if not bool(bands.details['valid'].all()):
        raise ValueError('Calibration requires valid bands')
    gradients['bands'] = torch.autograd.grad(bands.loss, z)[0]
    base_rms = gradients['dice'].square().mean().sqrt()
    bands_rms = gradients['bands'].square().mean().sqrt()
    row = dict(dice_rms=float(base_rms), bands_rms=float(bands_rms),
               bands_over_dice=float(bands_rms/base_rms))
    if arm == 'dice_bands_edge_depth':
        gradients['edge'] = torch.autograd.grad(original.BoundaryEdgeConsistencyLoss()(z, labels), z)[0]
        gradients['depth'] = torch.autograd.grad(original.depth_loss(z, labels, depth), z)[0]
        row.update(edge_rms=float(gradients['edge'].square().mean().sqrt()),
            dice_bands_cross_mean=float((gradients['dice']*gradients['bands']).mean()),
            depth_rms=float(gradients['depth'].square().mean().sqrt()),
            _depth_pair_means={a+':'+b:float((ga*gb).mean())
                for a,ga in gradients.items() for b,gb in gradients.items()})
    for key, value in row.items():
        if isinstance(value, dict):
            if not all(math.isfinite(v) for v in value.values()):
                raise ValueError('Nonfinite gradient Gram matrix')
        elif not math.isfinite(value) or (not key.endswith('cross_mean') and value <= 0):
            raise ValueError('Nonfinite or degenerate calibration')
    return row


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--protocol-dir', type=Path, required=True)
    parser.add_argument('--objective', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    p = read(args.protocol_dir/'protocol.json')
    if (p['model_identity'] != 'basic_unet16' or p['augmentation'] != 'none'
            or p['weight_ce'] != 1 or args.objective not in ARMS[1:]):
        raise ValueError('Unexpected 3D UNet pilot recipe')
    experiment.configure(p, args.objective)
    experiment.trainer.ARMS = ARMS
    experiment.base.build_loss_and_optimizer = experiment.ORIGINAL_LOSS_OPTIMIZER
    experiment.trainer.training_loss = training_loss
    experiment.trainer.calibration_case = calibration_case
    original_calibrate = experiment.trainer.calibrate
    def calibrate(*args, **kwargs):
        report = original_calibrate(*args, **kwargs)
        report.update(base_gradient='Dice+CE', legacy_dice_rms_means='Joint Dice+CE logit-gradient RMS',
            scope='All ten training volumes, no augmentation or validation. Bands relative to Dice+CE; edge relative to Dice+CE+bands; depth relative to Dice+CE+bands+edge.')
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
            raise ValueError('Initialization differs from the Dice+CE baseline')
        write(args.output/'initialization.json', dict(model_state_sha256=h.hexdigest()))
        return model
    experiment.base.build_model = build_model
    write(args.output/'runtime_recipe.json', dict(dataset='MSD', model='basic_unet16',
        objective=args.objective, weight_dice=1, weight_ce=1, dice_include_background=True,
        augmentation='none', constraints=args.objective.split('_')[1:], calibration_base='Dice+CE',
        target=p['calibration_target_ratio'], p95_cap=p['calibration_p95_cap'],
        baseline_reference=p['baseline_reference'], primary_checkpoint='final_budget',
        protocol_sha256=digest(args.protocol_dir/'protocol.json')))
    experiment.trainer.main()


if __name__ == '__main__':
    main()
