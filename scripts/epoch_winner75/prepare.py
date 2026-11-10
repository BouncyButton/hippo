"""Freeze six paired runs, reusing verified architectures, splits and loss code."""
import json
from pathlib import Path
import shutil

from scripts.depth_supervision.utils import atomic_json, digest
from scripts.capped_winner.objectives import ARM, ENTRY

ROOT = Path(__file__).resolve().parents[2]
NAME = 'epoch_winner75_20261013'


def main():
    stage = ROOT/'experiments'/NAME
    if stage.exists():
        raise FileExistsError('Refusing to overwrite a frozen experiment')
    parent = ROOT/'experiments/capped_winner_31976_20261013/source'
    old = parent/'experiments/capped_winner_31976_20261013/nnunet'
    p = json.loads((old/'protocol.json').read_text())
    plan = {k: p[k] for k in ['pkl_sha256','adamw_weight_decay','sgd_weight_decay',
        'nnunet_sgd_momentum','depth_geometry','native_geometry','precision','patient_identity_scope']}
    plan.update(name=NAME, max_epochs=75, early_stopping=dict(min_epochs=25, patience=8, min_delta=.0005),
        augmentation='none', learning_rate_policy='StepLR: initial native model LR, halved after epochs 20, 40, 60; no LR warmup',
        calibration_policy='Ten training volumes after epoch 5; linear ramp epochs 6-10; independent calibration per model',
        gradient_ratios=dict(bands=.125,pair=.075,depth=.1), gradient_ratio_p95_cap=.5,
        winner_objective=ENTRY, fold=0, seed=0, fraction=.05,
        primary_metric='Maximum validation macro anterior/posterior hard Dice, evaluated every epoch',
        stopping_selection='Patience uses 0.0005 improvement; selected checkpoint uses exact highest Dice',
        flops_cap=None, auxiliary_branch=False, pretrained_weights=False,
        training_order='Fresh seed-0 initialization and identical epoch permutations within each model pair',
        native_region_losses=dict(nnunet='Dice+CE with original deep supervision',
                                  unetr_vit_b16='Dice',basic_unet16='Dice'),
        runtime_versions=dict(torch='2.12.1+cu130',monai='1.6.0'),
        split=json.loads((old/'splits_10patients.json').read_text())[0],
        nnunet_architecture=json.loads((old/'nnunet_architecture.json').read_text()),
        parent_protocol_sha256=digest(old/'protocol.json'),
        interpretation='Paired loss comparison under a common epoch policy. Differences from capped experiments also include augmentation and LR schedule.',
        checkpoint_retention='Record and hash selected/latest checkpoint and metrics; verify result reproduction, then retire completed checkpoints.')
    plan['runs'] = [dict(id=model+'__'+arm, model=model, objective=arm, fold=0, seed=0)
                    for model in ['nnunet','unetr_vit_b16','basic_unet16'] for arm in ['dice',ARM]]
    refs = json.loads((parent/'paired_references.json').read_text())
    plan['capped_baseline_reference'] = [{k:r[k] for k in ['model','final_dice','best_dice'] if k in r}
        for r in refs if r['model'] in ['nnunet','unetr_vit_b16','basic_unet16']
        and r['fold']==r['seed']==0 and r['objective']=='dice']
    stage.mkdir(parents=True)
    shutil.copytree(parent,stage/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(ROOT/'scripts/epoch_winner75',stage/'source/scripts/epoch_winner75',
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    atomic_json(stage/'plan.json',plan)
    shutil.copyfile(ROOT/'scripts/epoch_winner75/run.sbatch',stage/'run.sbatch')
    manifest = {str(f.relative_to(stage/'source')):digest(f)
                for f in sorted((stage/'source').rglob('*')) if f.is_file()}
    atomic_json(stage/'source_manifest.json',manifest)
    atomic_json(stage/'freeze.json',{n:digest(stage/n) for n in ['plan.json','source_manifest.json','run.sbatch']})
    print(json.dumps(dict(stage=str(stage),runs=len(plan['runs']),source_files=len(manifest))))


if __name__=='__main__':
    main()
