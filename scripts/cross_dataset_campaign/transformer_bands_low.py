"""Matched MSD transformer bands ablation; only the calibration target changes."""
import argparse
from functools import partial
import hashlib
from pathlib import Path

from scripts.cross_dataset_campaign import msd_ce_constraints as reference
from scripts.cross_dataset_campaign.common import read, write, digest


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--protocol-dir', type=Path, required=True)
    parser.add_argument('--objective', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    p = read(args.protocol_dir/'protocol.json')
    if (p['dataset'] != 'MSD' or p['model_identity'] not in ('swinunetr', 'unetr_vit_b16')
            or p['augmentation'] != 'none' or p['weight_ce'] != 1
            or args.objective != 'dice_bands' or p['calibration_target_ratio'] not in (.1, .2)
            or p['calibration_p95_cap'] != .75):
        raise ValueError('Unexpected transformer bands ablation recipe')
    experiment = reference.experiment
    experiment.configure(p, args.objective)
    experiment.base.build_loss_and_optimizer = experiment.ORIGINAL_LOSS_OPTIMIZER
    experiment.trainer.training_loss = partial(reference.training_loss, sdf_policy=p['sdf_policy'][args.objective])
    experiment.trainer.calibration_case = reference.calibration_case
    original_calibrate = experiment.trainer.calibrate
    def calibrate(*args, **kwargs):
        report = original_calibrate(*args, **kwargs)
        report.update(base_gradient='Dice+CE', legacy_dice_rms_means='Joint Dice+CE logit-gradient RMS',
            scope='All ten training crops only; same calibration graph and prefix as the matched 60% bands run')
        return report
    experiment.trainer.calibrate = calibrate
    builder = experiment.base.build_model
    def build_model(name, architecture):
        model = builder(name, architecture)
        h = hashlib.sha256()
        for key, value in sorted(model.state_dict().items()):
            h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
        if h.hexdigest() != p['baseline_reference']['initialization_sha256']:
            raise ValueError('Initialization differs from the matched Dice+CE baseline')
        write(args.output/'initialization.json', dict(model_state_sha256=h.hexdigest()))
        return model
    experiment.base.build_model = build_model
    write(args.output/'runtime_recipe.json', dict(dataset='MSD', model=p['model_identity'],
        objective=args.objective, weight_dice=1, weight_ce=1, dice_include_background=True,
        augmentation='none', constraints=['bands'], calibration_base='Dice+CE',
        target=p['calibration_target_ratio'], p95_cap=.75, sdf_policy=p['sdf_policy'][args.objective],
        baseline_reference=p['baseline_reference'], primary_checkpoint='final_budget',
        protocol_sha256=digest(args.protocol_dir/'protocol.json'),
        matched_60_percent_reference=p['matched_60_percent_reference']))
    experiment.trainer.main()


if __name__ == '__main__':
    main()
