"""Other-model CE ablation, preserving each established MONAI Dice recipe."""
import argparse
import hashlib
from pathlib import Path

import torch.nn.functional as F

from scripts.cross_dataset_campaign import conventional as experiment
from scripts.cross_dataset_campaign.common import read, write, digest
from compute_constraint_objectives import training_loss as original_loss

MODELS = ('swinunetr', 'unetr_vit_b16', 'basic_unet16')


def paired_loss(outputs, labels, objective, weights, scale, sdf=None,
                criterion=None, depth=None, sdf_scale=0., *, weight_ce):
    if objective != 'dice' or criterion is not None or weight_ce not in (0, 1):
        raise ValueError('Only the requested conventional baseline pair is permitted')
    loss = original_loss(outputs, labels, 'dice', weights, scale, sdf)
    if weight_ce:
        loss = loss + F.cross_entropy(outputs.float(), labels[:, 0].long())
    return loss


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--protocol-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    protocol = read(args.protocol_dir/'protocol.json')
    if (protocol['model_identity'] not in MODELS or protocol['augmentation'] != 'none'
            or protocol['weight_ce'] not in (0, 1)):
        raise ValueError('Unexpected model, augmentation or CE weight')
    experiment.configure(protocol, 'dice')
    from functools import partial
    experiment.trainer.training_loss = partial(paired_loss, weight_ce=protocol['weight_ce'])
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
    write(args.output/'runtime_recipe.json', dict(dataset='MSD', model=protocol['model_identity'],
        objective='MONAI_Dice' + ('_plus_CE' if protocol['weight_ce'] else ''),
        weight_dice=1, weight_ce=protocol['weight_ce'], dice_include_background=True,
        dice_batch=False, augmentation='none', constraints=False,
        optimizer='Unchanged AdamW and original model-specific learning-rate schedule',
        updates=protocol['models'][protocol['trainer_alias']]['objectives']['dice']['max_steps'],
        primary_checkpoint='final_budget', protocol_sha256=digest(args.protocol_dir/'protocol.json'),
        evaluation='Full 52-volume validation and 10-volume training cohort; Dice, FN/FP and A/P swaps'))
    experiment.trainer.main()


if __name__ == '__main__':
    main()
