"""Experiment identity and predeclared data/budget decisions."""
from pathlib import Path

NAME = 'unetr_subfields_cap_20261106'
ROOT = Path(__file__).resolve().parents[2]
CAP = 31_976_450_667_090
FOLDS = (0, 1, 2)
SEEDS = (0, 1, 2)
DATASETS = {
    'MNI': dict(folder='Dataset102_MNI', native_shape=(40, 56, 48),
                spatial_size=(64, 64, 64), train_subjects=5,
                classes=['background', 'CA1_3', 'subiculum', 'CA4_DG']),
    'COBRA': dict(folder='Dataset105_COBRA', native_shape=(48, 64, 80),
                  spatial_size=(64, 64, 96), train_subjects=4,
                  classes=['background', 'CA1', 'subiculum', 'CA4_DG', 'CA2_3', 'SR_SL_SM']),
}


def tasks():
    return [dict(id=f'{dataset}_unetr_vit_b16_dice_f{fold}s{seed}', kind='conventional',
                 dataset=dataset, model='unetr_vit_b16', fold=fold, seed=seed,
                 arm='dice', protocol=dataset)
            for fold in FOLDS for seed in SEEDS for dataset in DATASETS]


def subject(case):
    return case.rsplit('_', 1)[0]


def baseline_schedule(cost, count):
    if type(cost) is not int or cost <= 0 or cost > CAP:
        raise ValueError('Invalid measured update cost')
    updates = CAP // cost
    return dict(prefix_updates=0, sdf_prefix_updates=0, constraint_warmup_updates=1,
                sdf_warmup_updates=1, baseline_flops_per_update=cost,
                early_flops_per_update=cost, flops_per_update=cost, calibration_flops=0,
                budget_scope='optimization_only', max_steps=updates,
                consumed_training_flops=updates * cost, consumed_flops=updates * cost,
                consumed_total_flops=updates * cost, unused_flops=CAP - updates * cost,
                patient_pass_epochs=updates / count)
