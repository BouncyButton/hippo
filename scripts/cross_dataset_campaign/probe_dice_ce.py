"""Explicitly requested native Dice+CE diagnostic; no augmentation/constraints."""
import argparse
from pathlib import Path

from scripts.cross_dataset_campaign import conventional as experiment
from scripts.cross_dataset_campaign.common import read, write, digest


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--protocol-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args,_=parser.parse_known_args()
    protocol=read(args.protocol_dir/'protocol.json')
    if protocol['diagnostic_updates']!=90 or protocol['augmentation']!='none':
        raise ValueError('Only the requested 90-update unaugmented diagnostic is permitted')
    experiment.configure(protocol,'dice')
    # Restore the original native foreground Dice + CE criterion, including
    # its historical deep supervision and SGD settings.
    experiment.base.build_loss_and_optimizer=experiment.ORIGINAL_LOSS_OPTIMIZER
    write(args.output/'runtime_recipe.json',dict(dataset='MSD',model='nnunet',
        objective='native_foreground_Dice_plus_CE',augmentation='none',constraints=False,
        updates=90,primary_checkpoint='final_update',
        protocol_sha256=digest(args.protocol_dir/'protocol.json'),
        evaluation='Full 52-volume validation and 10-volume training cohort; class Dice, FN/FP and A/P swaps'))
    experiment.trainer.main()


if __name__=='__main__':main()
