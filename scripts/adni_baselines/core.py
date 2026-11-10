"""ADNI data/model adapters for the existing compute-accounted training loop."""
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import torch
from monai.networks.nets import BasicUNet, SwinUNETR, UNETR
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import train_compute_matched_baselines as base
from baselines.simple_segmentation import CANDIDATES
from baselines.swin_unetr.swin_unetr import _build_monai_dataset_from_pkl

BUDGET = 31976450667090
MODELS = {
    'swinunetr': dict(alias='swinunetr', lr=.0001, horizon=610),
    'unetr_vit_b16': dict(alias='unet', lr=.0001, horizon=477),
    'basic_unet16': dict(alias='unet', lr=.0003, horizon=1095),
    'nnunet': dict(alias='nnunet', lr=.01, horizon=174),
}
_original_builder = base.build_model
_original_learning_rate = base.learning_rate


def case_name(row):
    return f'hippocampus_adni_{row.image_id}_{row.direction}'


def select_split(df, folds):
    """Sample people from saved fold 0, never individual left/right crops."""
    by_case = {case_name(r): str(r.subject_id) for r in df.itertuples()}
    if len(by_case) != len(df):
        raise ValueError('Duplicate ADNI case IDs')
    full = folds[0]
    train_subjects = sorted({by_case[c] for c in full['train']})
    val_subjects = sorted({by_case[c] for c in full['val']})
    if set(train_subjects) & set(val_subjects):
        raise ValueError('Subject leakage in saved fold 0')
    if len(train_subjects) != 108 or len(val_subjects) != 27:
        raise ValueError('Unexpected ADNI fold size')
    count = max(1, int(.05 * len(train_subjects)))
    chosen = sorted(np.random.RandomState(0).permutation(train_subjects)[:count].tolist())
    train = sorted(c for c in full['train'] if by_case[c] in chosen)
    val = sorted(full['val'])
    if len(train) != 10 or len(val) != 54 or set(train) & set(val):
        raise ValueError('Expected 10 training and 54 validation crops')
    for subject in chosen + val_subjects:
        names = [c for c in train + val if by_case[c] == subject]
        if len(names) != 2 or {c.rsplit('_', 1)[1] for c in names} != {'L', 'R'}:
            raise ValueError('Each included subject must have both hippocampi')
    return dict(train=train, val=val), dict(
        subset_seed=0, requested_fraction=.05, actual_fraction=count / len(train_subjects),
        rounding='floor(0.05 * 108 subjects) = 5 subjects',
        selection='numpy RandomState(0) permutation of sorted subject IDs; first five',
        train_subjects=chosen, validation_subjects=val_subjects,
        full_train_subjects=train_subjects, case_to_subject=by_case)


class PreparedADNI(Dataset):
    def __init__(self, dataframe, names):
        rows = {case_name(r): i for i, r in enumerate(dataframe.itertuples())}
        if len(rows) != len(dataframe) or len(set(names)) != len(names):
            raise ValueError('Duplicate cases')
        subset = dataframe.iloc[[rows[name] for name in names]]
        for row in subset.itertuples():
            x, y = np.asarray(row.image_data), np.asarray(row.data)
            if x.shape != (32, 48, 40) or x.shape != y.shape or not np.isfinite(x).all():
                raise ValueError('Unexpected ADNI image geometry or intensity')
            if not set(np.unique(y)).issubset({0, 1}) or not np.any(y == 1):
                raise ValueError('Expected a nonempty binary hippocampus mask')
        transformed = _build_monai_dataset_from_pkl(subset, 'ADNI', 2, (64, 64, 64), False)
        self.items = []
        for item, row in zip(transformed, subset.itertuples()):
            for key in ('image', 'label'):
                if hasattr(item[key], 'as_tensor'):
                    item[key] = item[key].as_tensor()
                if tuple(item[key].shape) != (1, 64, 64, 64):
                    raise ValueError('Expected padded 64-cubed model input')
            if int(item['label'].sum()) != int(np.asarray(row.data).sum()):
                raise ValueError('Preprocessing lost foreground voxels')
            item['subject_id'] = str(row.subject_id)
            self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def build_model(identity, architecture):
    if identity == 'swinunetr':
        return SwinUNETR(in_channels=1, out_channels=2, use_checkpoint=False, drop_rate=0.)
    if identity in ('basic_unet16', 'unetr_vit_b16'):
        kwargs = deepcopy(CANDIDATES[identity]['kwargs'])
        kwargs['out_channels'] = 2
        constructor = BasicUNet if identity == 'basic_unet16' else UNETR
        return constructor(**kwargs)
    if identity != 'nnunet' or architecture['kwargs']['num_classes'] != 2:
        raise ValueError('Expected binary nnU-Net architecture')
    return _original_builder('nnunet', architecture)


@torch.no_grad()
def evaluate(model, dataset, device, amp):
    """Mean whole-hippocampus Dice; no nonexistent anterior/posterior classes."""
    model.eval()
    records = []
    for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
        with torch.autocast('cuda', dtype=torch.bfloat16) if amp else nullcontext():
            logits = model(batch['image'].to(device))
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if logits.shape[1] != 2 or not torch.isfinite(logits).all():
            raise ValueError('Expected finite binary logits')
        probabilities = logits.float().softmax(1)
        target = batch['label'].to(device)[:, 0] == 1
        prediction = probabilities.argmax(1) == 1
        intersection = (prediction & target).sum()
        hard = float(2 * intersection / (prediction.sum() + target.sum()))
        soft = float(2 * (probabilities[:, 1] * target).sum() /
                     (probabilities[:, 1].sum() + target.sum()))
        records.append(dict(case_name=batch['case_name'][0], subject_id=batch['subject_id'][0],
                            hippocampus_hard_dice=hard, hippocampus_soft_dice=soft,
                            predicted_voxels=int(prediction.sum()), target_voxels=int(target.sum())))
    hard = float(np.mean([r['hippocampus_hard_dice'] for r in records]))
    soft = float(np.mean([r['hippocampus_soft_dice'] for r in records]))
    return dict(summary=dict(hippocampus_hard_dice=hard, hippocampus_soft_dice=soft,
                             macro_hard_dice=hard, macro_soft_dice=soft), cases=records)


def bind(protocol):
    identity = protocol['model_identity']
    base.build_model = lambda alias, architecture: build_model(identity, architecture)
    base.PreparedPatients = PreparedADNI
    base.evaluate = evaluate
    base.learning_rate = lambda name, step, max_steps, initial, nnunet_warmup_fraction=0.: (
        _original_learning_rate(name, step, protocol['lr_horizon_updates'], initial,
                                nnunet_warmup_fraction))
    cost = protocol['models'][MODELS[identity]['alias']]['flops_per_update']
    cap = BUDGET // cost
    milestones = sorted({max(1, target // cost) for target in protocol['validation_target_flops']} | {cap})
    base.validation_steps = lambda max_steps, count=20: sorted({min(s, max_steps) for s in milestones})
    return milestones


def main():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--protocol-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    protocol = json.loads((args.protocol_dir / 'protocol.json').read_text())
    milestones = bind(protocol)
    args.output.mkdir(parents=True, exist_ok=True)
    recipe = dict(model=protocol['model_identity'], dataset='ADNI', num_classes=2,
                  adapter_sha256=base.digest(Path(__file__)),
                  protocol_sha256=base.digest(args.protocol_dir / 'protocol.json'),
                  validation_updates=milestones, lr_horizon_updates=protocol['lr_horizon_updates'],
                  metric='mean whole-hippocampus hard Dice across 54 crops',
                  training_unit='one hemisphere crop; 10 crops = 5 subjects per pass')
    recipe_path = args.output / 'runtime_recipe.json'
    if recipe_path.exists() and json.loads(recipe_path.read_text()) != recipe:
        raise ValueError('Resume recipe changed')
    base.atomic_json(recipe_path, recipe)
    base.main()


if __name__ == '__main__':
    main()
