"""Bind existing audited objectives to frozen unaugmented dataset recipes."""
import argparse
from contextlib import nullcontext
from copy import deepcopy
import functools
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from scripts.cross_dataset_campaign.common import ARMS, read, write, digest, confusion_metrics
from scripts.cross_dataset_campaign import capped_core as trainer
from scripts.capped_depth_delay import objectives as multiclass
from scripts.capped_depth_delay.schedule import training_flops
from scripts import campaign90
from scripts.adni_baselines import core as adni

base = trainer.base
MSD_DATASET = base.PreparedPatients
ORIGINAL_LR = base.learning_rate
ORIGINAL_BUILDER = base.build_model
ORIGINAL_LOSS_OPTIMIZER = base.build_loss_and_optimizer


def dice_only_optimizer(name, model, protocol):
    criterion, optimizer = ORIGINAL_LOSS_OPTIMIZER(name, model, protocol)
    if name == 'nnunet':
        from monai.losses import DiceLoss
        from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
        weights = np.array([1., .5, .25, 0.])
        criterion = DeepSupervisionWrapper(DiceLoss(to_onehot_y=True, softmax=True), weights/weights.sum())
    return criterion, optimizer


class SeismicDataset(Dataset):
    def __init__(self, descriptor, names):
        from scripts.geofwi3d_data import load_case
        self.items = []
        root = Path(descriptor['dataset_root'])
        manifest = read(descriptor['manifest'])
        for name in names:
            image, label = load_case(root, manifest['cases'][name])
            self.items.append(dict(image=torch.from_numpy(image[None]),
                                   label=torch.from_numpy(label[None]).long(), case_name=name))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def build_model(protocol, architecture):
    identity = protocol['model_identity']
    if protocol['dataset'] == 'MSD':
        spec = campaign90.MODELS[identity]
        if spec['candidate']:
            from baselines.simple_segmentation import build_candidate
            return build_candidate(spec['candidate'])
        return ORIGINAL_BUILDER(spec['alias'], architecture)
    if identity == 'unetr_vit_b16' and protocol['dataset'] == 'GeoFWI3D':
        from monai.networks.nets import UNETR
        from baselines.simple_segmentation import CANDIDATES
        kwargs = deepcopy(CANDIDATES[identity]['kwargs'])
        kwargs.update(out_channels=2, img_size=(96, 96, 96))
        return UNETR(**kwargs)
    return adni.build_model(identity, architecture)


@torch.no_grad()
def evaluate(model, dataset, device, amp, classes):
    model.eval()
    records = []
    for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
        with torch.autocast('cuda', dtype=torch.bfloat16) if amp else nullcontext():
            logits = model(batch['image'].to(device))
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if logits.shape[1] != len(classes) or not torch.isfinite(logits).all():
            raise ValueError('Invalid validation output')
        probabilities = logits.float().softmax(1)
        prediction = probabilities.argmax(1)
        truth = batch['label'].to(device)[:, 0].long()
        matrix = torch.bincount((truth*len(classes)+prediction).flatten(),
                                minlength=len(classes)**2).reshape(len(classes), len(classes))
        row = confusion_metrics(matrix.cpu().tolist(), classes)
        row['case_name'] = batch['case_name'][0]
        soft = []
        for c, name in enumerate(classes[1:], 1):
            gt = truth == c
            score = float(2*(probabilities[:, c]*gt).sum()/(probabilities[:, c].sum()+gt.sum()).clamp_min(1e-12))
            row[name+'_hard_dice'] = row['per_class'][name]['dice']
            row[name+'_soft_dice'] = score
            soft.append(score)
        row['macro_soft_dice'] = float(np.mean(soft))
        records.append(row)
    keys = [k for k in records[0] if k.endswith('_dice')]
    return dict(summary={k: float(np.mean([r[k] for r in records])) for k in keys}, cases=records)


def select_at_target(rows, arm, target, cap):
    import compute_constraint_objectives as original
    old_original, old_depth = original.calibrated_weight, multiclass.calibrated_weight
    def coefficient(values):
        return old_original(values, target=target, cap=cap)
    original.calibrated_weight = multiclass.calibrated_weight = coefficient
    try:
        return multiclass.select_weights(rows, arm)
    finally:
        original.calibrated_weight, multiclass.calibrated_weight = old_original, old_depth


def configure(protocol, arm):
    dataset = protocol['dataset']
    base.build_loss_and_optimizer = dice_only_optimizer
    # Every child process executes one condition, so binary loss bindings cannot
    # leak into a later MSD condition.
    base.build_model = lambda alias, arch: build_model(protocol, arch)
    if dataset == 'ADNI':
        base.PreparedPatients = adni.PreparedADNI
    elif dataset == 'GeoFWI3D':
        base._load_pkl_dataframe = read
        base.PreparedPatients = SeismicDataset
    else:
        base.PreparedPatients = MSD_DATASET
    base.evaluate = functools.partial(evaluate, classes=protocol['classes'])
    base.learning_rate = lambda name, step, maximum, initial: ORIGINAL_LR(
        name, step, protocol['lr_horizon_updates'], initial, protocol.get('nnunet_warmup_fraction', 0.))
    schedule = protocol['models'][protocol['trainer_alias']]['objectives'][arm]
    def milestones(maximum, count=20):
        return sorted({max((n for n in range(1, maximum+1) if training_flops(n, schedule) <= target), default=1)
                       for target in protocol['validation_target_flops']} | {maximum})
    base.validation_steps = milestones
    trainer.ARMS = ARMS
    if dataset != 'MSD':
        from scripts.adni_constraints import objectives as binary
        if dataset == 'GeoFWI3D':
            def depth_loss(logits, labels, weights):
                if tuple(logits.shape) != (1, 2, 96, 96, 96):
                    raise ValueError('Expected native seismic grid')
                return (torch.nn.functional.cross_entropy(logits.float(), labels[:, 0].long(), reduction='none')[0]*weights).sum()
            multiclass.depth_loss = depth_loss
        trainer.signed_distance_maps = binary.signed_distance_maps
        trainer.geometry_weights = binary.geometry_weights
        trainer.calibration_case = binary.calibration_case
        trainer.training_loss = functools.partial(binary.training_loss, policy=protocol['sdf_policy'][arm])
    else:
        def training_loss(outputs, labels, objective, weights, scale, sdf=None, criterion=None, depth=None, sdf_scale=0.):
            if 'sdf' in objective and protocol['sdf_policy'][arm] == 'original_shared_ramp':
                return multiclass.full_loss(outputs, labels, objective, weights, scale, criterion, sdf, depth)
            return multiclass.training_loss(outputs, labels, objective, weights, scale, sdf, criterion, depth, sdf_scale)
        trainer.training_loss = training_loss
    target, cap = protocol['calibration_target_ratio'], protocol['calibration_p95_cap']
    trainer.select_weights = functools.partial(select_at_target, target=target, cap=cap)
    original_calibrate = trainer.calibrate
    def calibrate(*args, **kwargs):
        report = original_calibrate(*args, **kwargs)
        report.update(target_ratio=target, p95_cap=cap)
        return report
    trainer.calibrate = calibrate


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--protocol-dir', type=Path, required=True)
    parser.add_argument('--objective', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    protocol = read(args.protocol_dir/'protocol.json')
    configure(protocol, args.objective)
    write(args.output/'runtime_recipe.json', dict(dataset=protocol['dataset'], augmentation='none',
        model=protocol['model_identity'], objective=args.objective, primary_checkpoint='final_budget',
        protocol_sha256=digest(args.protocol_dir/'protocol.json'),
        evaluation='Per-volume class Dice, integer FN/FP, A/P swaps for MSD; best is secondary'))
    trainer.main()


if __name__ == '__main__':
    main()
