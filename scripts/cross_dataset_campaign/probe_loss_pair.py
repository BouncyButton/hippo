"""Paired nnUNet diagnostic: the CE coefficient is the only loss change."""
import argparse
import hashlib
from pathlib import Path

from scripts.cross_dataset_campaign import conventional as experiment
from scripts.cross_dataset_campaign.common import read, write, digest


def loss_optimizer(name, model, protocol):
    criterion, optimizer = experiment.ORIGINAL_LOSS_OPTIMIZER(name, model, protocol)
    criterion.loss.weight_ce = protocol['weight_ce']
    return criterion, optimizer


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--protocol-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    protocol = read(args.protocol_dir/'protocol.json')
    if (protocol['diagnostic_updates'] != 90 or protocol['augmentation'] != 'none'
            or protocol['weight_ce'] not in (0, 1)):
        raise ValueError('Only the native 90-update unaugmented pair is permitted')
    experiment.configure(protocol, 'dice')
    experiment.base.build_loss_and_optimizer = loss_optimizer
    original_builder = experiment.base.build_model
    def build_model(name, architecture):
        model = original_builder(name, architecture)
        h = hashlib.sha256()
        for key, value in sorted(model.state_dict().items()):
            h.update(key.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        write(args.output/'initialization.json', dict(model_state_sha256=h.hexdigest()))
        return model
    experiment.base.build_model = build_model
    write(args.output/'runtime_recipe.json', dict(dataset='MSD', model='nnunet',
        objective='native_foreground_Dice' + ('_plus_CE' if protocol['weight_ce'] else ''),
        weight_dice=1, weight_ce=protocol['weight_ce'], dice_include_background=False,
        augmentation='none', constraints=False, updates=90, primary_checkpoint='final_update',
        protocol_sha256=digest(args.protocol_dir/'protocol.json'),
        evaluation='Full 52-volume validation and 10-volume training cohort; Dice, FN/FP and A/P swaps'))
    experiment.trainer.main()


if __name__ == '__main__':
    main()
