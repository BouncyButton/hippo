"""Multiclass subfield adapters around the existing audited training loop."""
import argparse
from copy import deepcopy
import functools
import json
from pathlib import Path

import numpy as np
import torch
from monai.networks.nets import UNETR
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, NormalizeIntensityd, SpatialPadd
from torch.utils.data import Dataset

from baselines.simple_segmentation import CANDIDATES
from scripts.cross_dataset_campaign import capped_core as trainer
from scripts.cross_dataset_campaign.conventional import evaluate as evaluate_classes
from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.unetr_subfields.contracts import DATASETS

base = trainer.base
ORIGINAL_LR = base.learning_rate


def model_kwargs(classes, spatial_size):
    kwargs = deepcopy(CANDIDATES['unetr_vit_b16']['kwargs'])
    kwargs.update(out_channels=classes, img_size=tuple(spatial_size))
    return kwargs


def build_model(classes, spatial_size):
    return UNETR(**model_kwargs(classes, spatial_size))


class PreparedSubfields(Dataset):
    """Nonzero z-score and symmetric padding only; preserve every label voxel."""
    def __init__(self, dataframe, names, spec):
        rows = {str(row.subject_id): row for row in dataframe.itertuples()}
        if len(rows) != len(dataframe) or len(set(names)) != len(names):
            raise ValueError('Duplicate cases')
        transform = Compose([
            EnsureChannelFirstd(keys=['image', 'label'], channel_dim='no_channel'),
            NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
            SpatialPadd(keys=['image', 'label'], spatial_size=spec['spatial_size'], mode='constant'),
            EnsureTyped(keys=['image', 'label'], dtype=(torch.float32, torch.long), track_meta=False),
        ])
        self.items = []
        classes = len(spec['classes'])
        for name in names:
            row = rows[name]
            image, label = np.asarray(row.image_data), np.asarray(row.label_data)
            if image.shape != tuple(spec['native_shape']) or label.shape != image.shape:
                raise ValueError('Unexpected native paired geometry')
            if not np.isfinite(image).all() or not np.isfinite(label).all():
                raise ValueError('Nonfinite source arrays')
            if not np.equal(label, np.rint(label)).all() or not set(np.unique(label)) <= set(range(classes)):
                raise ValueError('Invalid label encoding')
            item = transform(dict(image=image.copy(), label=label.copy(), case_name=name,
                                  participant_id=str(row.participant_id)))
            if tuple(item['image'].shape) != (1, *spec['spatial_size']):
                raise ValueError('Input differs from profiled shape')
            for c in range(1, classes):
                if int((item['label'] == c).sum()) != int((label == c).sum()):
                    raise ValueError('Preprocessing changed a foreground class')
            self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def evaluate(model, dataset, device, amp, classes):
    result = evaluate_classes(model, dataset, device, amp, classes)
    for case in result['cases']:
        matrix = case['confusion']
        tp = sum(sum(row[1:]) for row in matrix[1:])
        denominator = 2 * tp + case['foreground_fp'] + case['foreground_fn']
        case['whole_hippocampus_hard_dice'] = 2 * tp / denominator if denominator else 1.
    result['summary']['whole_hippocampus_hard_dice'] = float(np.mean([
        case['whole_hippocampus_hard_dice'] for case in result['cases']]))
    return result


def bind(protocol):
    if protocol['model_identity'] != 'unetr_vit_b16' or protocol['requested_arm_order'] != ['dice']:
        raise ValueError('Only the frozen UNETR baseline is supported')
    spec = DATASETS[protocol['dataset']]
    if list(spec['spatial_size']) != protocol['spatial_size'] or spec['classes'] != protocol['classes']:
        raise ValueError('Protocol shape/classes differ from adapter')
    base.build_model = lambda alias, architecture: build_model(len(spec['classes']), spec['spatial_size'])
    base.PreparedPatients = functools.partial(PreparedSubfields, spec=spec)
    base.evaluate = functools.partial(evaluate, classes=spec['classes'])
    base.learning_rate = lambda name, step, maximum, initial: ORIGINAL_LR(
        name, step, protocol['lr_horizon_updates'], initial)
    cost = protocol['models']['unet']['objectives']['dice']['flops_per_update']
    base.validation_steps = lambda maximum, count: sorted({
        min(maximum, max(1, target // cost)) for target in protocol['validation_target_flops']} | {maximum})
    trainer.ARMS = ('dice',)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--protocol-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    protocol = read(args.protocol_dir / 'protocol.json')
    bind(protocol)
    write(args.output / 'runtime_recipe.json', dict(
        dataset=protocol['dataset'], model='unetr_vit_b16', objective='dice',
        augmentation='none', primary_checkpoint='final_budget',
        primary_metric='Per-volume foreground-subfield macro hard Dice, then mean across volumes',
        secondary_metric='Whole-hippocampus union Dice and per-subfield Dice/FN/FP',
        protocol_sha256=digest(args.protocol_dir / 'protocol.json')))
    trainer.main()


if __name__ == '__main__':
    main()
