"""Run the MNI spatial objective on one prepared protocol."""
import fcntl

from functools import partial

from pathlib import Path

from scripts.cross_dataset_campaign.common import read, write, digest

from scripts.capped_depth_delay.schedule import training_flops

from scripts.unetr_subfields import cap_extension as batch

def train(args):
    from scripts.mni_constraints import objectives
    trainer=batch.bind(args);p=read(args.protocol_dir/'protocol.json');identity=p['model_identity']
    trainer.training_loss=partial(objectives.training_loss,identity=identity)
    trainer.calibration_case=partial(objectives.calibration_case,identity=identity)
    trainer.select_weights=partial(objectives.select_weights,target=p['calibration_target_ratio'],cap=p['calibration_p95_cap'])
    trainer.geometry_weights=objectives.geometry_weights
    trainer.signed_distance_maps=objectives.signed_distance_maps
    s=p['models'][args.model]['objectives'][args.objective]
    trainer.base.validation_steps=lambda maximum,count=20: sorted({next((n for n in range(1,maximum+1) if training_flops(n,s)>=target),maximum) for target in p['validation_target_flops']}|{maximum})
    write(args.output/'runtime_recipe.json',dict(model=identity,base_loss=p['training_loss'],constraints=p['constraints'],
        augmentation=p['augmentation'],protocol_sha256=digest(args.protocol_dir/'protocol.json'),
        objective_sha256=digest(Path(objectives.__file__)),initialization='fresh_from_fixed_seed',
        lr_horizon_updates=p['lr_horizon_updates'],primary_checkpoint='final_budget',baseline_only=False))
    with (args.output/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        trainer.run(args)

if __name__ == '__main__':
    import argparse
    from scripts.cross_dataset_campaign.common import install_signals
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'objective'):
        parser.add_argument('--' + name, required=True)
    for name in ('fold', 'seed'):
        parser.add_argument('--' + name, type=int, required=True)
    for name in ('pkl', 'protocol-dir', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.set_defaults(smoke_steps=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    install_signals()
    train(args)
