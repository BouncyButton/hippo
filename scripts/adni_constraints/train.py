"""Bind the audited training loop to binary ADNI data and constraint functions."""
import argparse
import functools
import json
from pathlib import Path

from scripts.adni_baselines import core as adni
from scripts.adni_constraints import objectives
from scripts.capped_depth_delay import core as trainer
from scripts.capped_depth_delay.schedule import training_flops


def validation_steps(protocol, arm, maximum=None):
    schedule = protocol['models'][protocol['trainer_alias']]['objectives'][arm]
    maximum = min(maximum or schedule['max_steps'], schedule['max_steps'])
    # Same 20 compute targets as ADNI baseline: avoid more selection chances.
    return sorted({max((n for n in range(1, maximum + 1)
                        if training_flops(n, schedule) <= target), default=1)
                   for target in protocol['validation_target_flops']} | {maximum})


def configure(protocol, arm):
    if protocol['augmentation'] != 'none':
        raise ValueError('This experiment must be unaugmented')
    base = adni.base
    identity = protocol['model_identity']
    base.build_model = lambda alias, architecture: adni.build_model(identity, architecture)
    base.PreparedPatients = adni.PreparedADNI
    base.evaluate = adni.evaluate
    base.learning_rate = lambda name, step, maximum, initial: adni._original_learning_rate(
        name, step, protocol['lr_horizon_updates'], initial, protocol.get('nnunet_warmup_fraction', 0.))
    base.validation_steps = lambda maximum, count=20: validation_steps(protocol, arm, maximum)
    trainer.ARMS = objectives.ARMS
    trainer.signed_distance_maps = objectives.signed_distance_maps
    trainer.geometry_weights = objectives.geometry_weights
    trainer.calibration_case = objectives.calibration_case
    trainer.select_weights = objectives.select_weights
    trainer.training_loss = functools.partial(objectives.training_loss, policy=protocol['sdf_policy'][arm])


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--protocol-dir', type=Path, required=True)
    parser.add_argument('--objective', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    protocol = json.loads((args.protocol_dir / 'protocol.json').read_text())
    for name, sha in protocol['training_source_sha256'].items():
        if adni.base.digest(adni.ROOT / name) != sha:
            raise ValueError('Frozen source changed: ' + name)
    configure(protocol, args.objective)
    args.output.mkdir(parents=True, exist_ok=True)
    recipe = dict(dataset='ADNI', model_identity=protocol['model_identity'], num_classes=2, augmentation='none',
                  objective=args.objective, protocol_sha256=adni.base.digest(args.protocol_dir/'protocol.json'),
                  validation_updates=validation_steps(protocol, args.objective),
                  geometry_units='source crop voxels; anatomical orientation unavailable',
                  lr_horizon_updates=protocol['lr_horizon_updates'],
                  sdf_policy=protocol['sdf_policy'][args.objective], budget_scope='optimization_only')
    path = args.output / 'runtime_recipe.json'
    if path.exists() and json.loads(path.read_text()) != recipe:
        raise ValueError('Resume recipe differs')
    adni.base.atomic_json(path, recipe)
    trainer.main()


if __name__ == '__main__':
    main()
