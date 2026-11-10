"""Freeze exactly the supplied comparison matrix and the successful schedule."""
import hashlib
import json
from pathlib import Path
import shutil

from scripts.depth_supervision.utils import atomic_json, digest
from .objectives import entries

ROOT = Path(__file__).resolve().parents[2]
NAME = 'swin_contenders75_20261013'


def main():
    stage = ROOT/'experiments'/NAME
    if stage.exists():
        raise FileExistsError(stage)
    old = json.loads((ROOT/'docs/experiments/native_branch_constraints_20261008/protocol.json').read_text())
    inputs = json.loads((ROOT/'experiments/epoch_winner75_20261013/plan.json').read_text())
    assert old['splits']['5pct'] == inputs['split']
    p = {key:old[key] for key in ('max_epochs', 'learning_rate', 'weight_decay', 'step_size',
         'lr_gamma', 'early_stopping', 'geometry', 'calibration', 'pkl_sha256')}
    p.update(name=NAME, model='ordinary_swinunetr', model_parameters=15703029,
        fold=0, seed=0, fraction=.05, training_augmentation='none', pretrained=False,
        auxiliary_branch=False, precision='cuda bfloat16 autocast',
        pkl='/mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl',
        splits={'5pct':inputs['split']}, native_geometry=inputs['native_geometry'],
        runs=entries(), runtime_versions=inputs['runtime_versions'],
        winner_reference=dict(macro_dice=.8150157184588936, best_epoch=41, completed_epochs=49),
        loss_definitions=dict(baseline='Original three-class soft Dice, including background',
            bands='Original OuterBoundaryBandLoss, not focused bands or endpoints',
            sdf='Original full-grid A/P signed-distance loss, clip 8 mm',
            edge='Original BoundaryEdgeConsistencyLoss',
            depth='Original depth-balanced multiclass CE, 2 mm bins, 8 mm ROI, sparse-bin threshold 64',
            winner='Exact combo_depth_g100: 3/2 mm focused bands, independent 1/3 mm endpoints, depth CE'),
        calibration_policy='Training-only epoch-five gradient RMS, original sequential per-term 10% target and 50% p95 cap; preserve the winner outer coefficients and its original depth calibration.',
        interpretation='Fixed-recipe comparison on the same development fold. This ranks these configurations, not every possible tuning of each loss.',
        retention='After selected-checkpoint evaluation, verify all small records and hashes, retire completed checkpoints/predictions immediately, keep only two reference masks until paired comparisons finish.')
    p['identity_sha'] = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
    stage.mkdir(parents=True)
    parent = ROOT/'experiments/epoch_winner75_20261013/source'
    shutil.copytree(parent, stage/'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    # The established trainer/loss implementations must agree with their snapshot.
    for file in ('scripts/band_focus/combo_train.py', 'scripts/band_focus/combo_objectives.py',
                 'scripts/native_branch_constraints/objectives.py', 'scripts/depth_supervision/train.py'):
        assert digest(ROOT/file) == digest(stage/'source'/file), file
    shutil.copytree(ROOT/'scripts/swin_contenders75', stage/'source/scripts/swin_contenders75',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copyfile(ROOT/'scripts/swin_contenders75/run.sbatch', stage/'run.sbatch')
    atomic_json(stage/'protocol.json', p)
    files = {str(f.relative_to(stage/'source')):digest(f)
             for f in sorted((stage/'source').rglob('*')) if f.is_file()}
    atomic_json(stage/'source_manifest.json', files)
    atomic_json(stage/'freeze.json', {name:digest(stage/name)
                                    for name in ('protocol.json', 'source_manifest.json', 'run.sbatch')})
    print(json.dumps(dict(stage=str(stage), runs=len(p['runs']), source_files=len(files))))


if __name__ == '__main__':
    main()
