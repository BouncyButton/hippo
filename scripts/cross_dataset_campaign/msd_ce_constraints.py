"""All original MSD auxiliaries on the established per-model Dice+CE baseline."""
import argparse
from functools import partial
import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.cross_dataset_campaign import conventional as experiment
from scripts.cross_dataset_campaign.common import ARMS, MODELS, read, write, digest
from scripts.capped_depth_delay import objectives as original


def training_loss(outputs, labels, arm, weights, scale, sdf=None,
                  criterion=None, depth=None, sdf_scale=0., *, sdf_policy='delayed_5_of_9'):
    if criterion is not None and (criterion.loss.weight_ce != 1 or criterion.loss.weight_dice != 1):
        raise ValueError('Native Dice+CE coefficients must both be one')
    if 'sdf' in arm and sdf_policy == 'original_shared_ramp':
        loss = original.full_loss(outputs, labels, arm, weights, scale, criterion, sdf, depth)
    else:
        loss = original.training_loss(outputs, labels, arm, weights, scale, sdf, criterion, depth, sdf_scale)
    if criterion is None:
        loss = loss + F.cross_entropy(outputs.float(), labels[:, 0].long())
    return loss


def calibration_case(outputs, labels, arm, sdf=None, criterion=None, depth=None):
    if criterion is not None:
        # The established native calibration already includes CE and deep supervision.
        return original.calibration_case(outputs, labels, arm, sdf, criterion, depth)
    z = outputs.detach().float().requires_grad_(True)
    gradients = {'dice': torch.autograd.grad(training_loss(z, labels, 'dice', {}, 0.), z)[0]}
    dr = gradients['dice'].square().mean().sqrt()
    row = dict(dice_rms=float(dr))
    if 'bands' in arm:
        band = original.OuterBoundaryBandLoss()(z, labels)
        if not bool(band.details['valid'].all()):
            raise ValueError('Invalid calibration bands')
        gradients['bands'] = torch.autograd.grad(band.loss, z)[0]
        br = gradients['bands'].square().mean().sqrt()
        row.update(bands_rms=float(br), bands_over_dice=float(br/dr))
        if 'edge' in arm or 'sdf' in arm:
            row['dice_bands_cross_mean'] = float((gradients['dice']*gradients['bands']).mean())
    if 'edge' in arm:
        gradients['edge'] = torch.autograd.grad(original.BoundaryEdgeConsistencyLoss()(z, labels), z)[0]
        er = gradients['edge'].square().mean().sqrt()
        row['edge_rms'] = float(er)
        if 'bands' not in arm:
            row['edge_over_base'] = float(er/dr)
    if 'sdf' in arm:
        gradients['sdf'] = torch.autograd.grad(original.signed_distance_loss(z, labels, sdf), z)[0]
        row['sdf_rms'] = float(gradients['sdf'].square().mean().sqrt())
        if 'edge' in arm:
            row['dice_edge_cross_mean'] = float((gradients['dice']*gradients['edge']).mean())
            if 'bands' in arm:
                row['bands_edge_cross_mean'] = float((gradients['bands']*gradients['edge']).mean())
    if 'depth' in arm:
        gradients['depth'] = torch.autograd.grad(original.depth_loss(z, labels, depth), z)[0]
        row['depth_rms'] = float(gradients['depth'].square().mean().sqrt())
        row['_depth_pair_means'] = {a+':'+b:float((ga*gb).mean())
            for a,ga in gradients.items() for b,gb in gradients.items()}
    for key, value in row.items():
        if isinstance(value, dict):
            if not all(math.isfinite(v) for v in value.values()):
                raise ValueError('Nonfinite calibration Gram matrix')
        elif not math.isfinite(value) or (not key.endswith('cross_mean') and value <= 0):
            raise ValueError('Degenerate calibration')
    return row


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--protocol-dir', type=Path, required=True)
    parser.add_argument('--objective', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    p = read(args.protocol_dir/'protocol.json')
    if (p['dataset'] != 'MSD' or p['model_identity'] not in MODELS or p['augmentation'] != 'none'
            or p['weight_ce'] != 1 or args.objective not in ARMS
            or p['calibration_target_ratio'] != .6 or p['calibration_p95_cap'] != .75):
        raise ValueError('Unexpected MSD target60 recipe')
    experiment.configure(p, args.objective)
    experiment.base.build_loss_and_optimizer = experiment.ORIGINAL_LOSS_OPTIMIZER
    experiment.trainer.training_loss = partial(training_loss, sdf_policy=p['sdf_policy'][args.objective])
    experiment.trainer.calibration_case = calibration_case
    original_calibrate = experiment.trainer.calibrate
    def calibrate(*args, **kwargs):
        report = original_calibrate(*args, **kwargs)
        report.update(base_gradient='Dice+CE', legacy_dice_rms_means='Joint Dice+CE logit-gradient RMS',
            scope='All ten training crops only; sequential bands, edge, SDF, depth against the accumulated weighted gradient; native deep-supervision weighting retained for nnUNet')
        return report
    experiment.trainer.calibrate = calibrate
    builder = experiment.base.build_model
    def build_model(name, architecture):
        model = builder(name, architecture)
        h = hashlib.sha256()
        for key, value in sorted(model.state_dict().items()):
            h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
        if h.hexdigest() != p['baseline_reference']['initialization_sha256']:
            raise ValueError('Initialization differs from the matched baseline')
        write(args.output/'initialization.json', dict(model_state_sha256=h.hexdigest()))
        return model
    experiment.base.build_model = build_model
    write(args.output/'runtime_recipe.json', dict(dataset='MSD', model=p['model_identity'],
        objective=args.objective, weight_dice=1, weight_ce=1, dice_include_background=p['model_identity'] != 'nnunet',
        augmentation='none', constraints=args.objective.split('_')[1:], calibration_base='Dice+CE',
        target=.6, p95_cap=.75, sdf_policy=p['sdf_policy'][args.objective],
        baseline_reference=p['baseline_reference'], primary_checkpoint='final_budget',
        protocol_sha256=digest(args.protocol_dir/'protocol.json')))
    experiment.trainer.main()


if __name__ == '__main__':
    main()
