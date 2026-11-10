"""Freeze the authorized twenty-arm 5% ordinary-SwinUNETR experiment."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/experiments/edge_ablation_20261008'
NAME = 'edge_ablation_fold0_seed0_5pct_20261008_01'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    from .objectives import experiment_matrix
    previous = ROOT/'docs/experiments/native_branch_constraints_20261008/protocol.json'
    old = json.loads(previous.read_text())
    keep = ('dataset', 'fold', 'seed', 'max_epochs', 'model', 'feature_size', 'spatial_size',
            'batch_size', 'drop_rate', 'activation_checkpointing', 'pretrained', 'training_augmentation',
            'precision', 'learning_rate', 'weight_decay', 'step_size', 'lr_gamma', 'early_stopping',
            'selection', 'geometry', 'required_dependencies', 'required_monai', 'pkl', 'pkl_sha256',
            'native_geometry', 'splits', 'low_data_fraction_actual', 'calibration')
    p = {k: copy.deepcopy(old[k]) for k in keep}
    p.update(name=NAME, baseline_loss='dice_bands', model_parameters=15703029,
             main_runs=20, fractions=['5pct'], auxiliary_branch=False,
             runs=[dict(e, fraction='5pct', seed=0, arm='ordinary_swinunetr') for e in experiment_matrix()],
             prior_protocol_sha256=sha(previous),
             resources=dict(partition='stud', gpu='4g.40gb:1', cpus=4, memory='24G', time='12:00:00'),
             minimum_free_gib=4.,
             geometry_policy='GT supervision only. Verify 1 mm voxels. Symmetric 1 and 3 mm axis-aligned pairs centered on a real annotated crossing, with exactly one transition and all intervening voxels inside the original image. No guide, ROI input, teacher, extra encoder or prediction head.',
             reduction_policy='pairs: equal mean per available distance. anchors: equal mean over inside/anterior boundary voxels, then available distances at each anchor, then valid outward directions. Missing longer distances do not remove a thin-region anchor.',
             loss_definitions=dict(dice='Original MONAI three-class Dice including background',
                 bands='Original two-step balanced foreground/background BCE, no focal/degree weights',
                 legacy_edge='Unmodified original foreground residual face-difference loss',
                 same_only='Original same-GT-foreground-class face subset in the two-step band',
                 split_equal='Equal mean of separately normalized legacy cross and same subsets',
                 probability='Squared [1-(sigmoid(score_inside)-sigmoid(score_outside))]',
                 score='softplus(score_outside-score_inside), zero ranking margin',
                 outer='score = logaddexp(z_anterior,z_posterior)-z_background',
                 ap='score = z_anterior-z_posterior; only annotated A/P crossings',
                 outer_ap='Equal raw mean of outer and AP terms, jointly calibrated to ONE edge budget'),
             calibration_policy='Five identical Dice-only epochs in every run. Training-only calibration: bands at median 0.1/p95 <=0.5 logit-gradient RMS relative to Dice; the entire edge objective at median 0.1/p95 <=0.5 relative to Dice+weighted bands. Same band coefficient across all arms; fixed coefficients; ramp epochs 6-10. Calibration is not a parameter-gradient equality guarantee.',
             evaluation='Per-case and aggregate A/P Dice, foreground Dice, missed/extra/swapped voxels, mean surface distance, HD95; deltas versus fresh Dice+bands and matched factor contrasts. Single fold/seed exploratory screen, no confirmatory claim.',
             storage_budget='Twenty best checkpoints, at most one resumable optimizer checkpoint, predictions, logs and reports; no dataset copy or automatic cleanup.')
    files = [ROOT/'baselines/swin_unetr/swin_unetr.py', ROOT/'scripts/compute_constraint_objectives.py']
    files += list((ROOT/'scripts/depth_supervision').glob('*.py'))
    # Existing calibration helpers and canonical losses; snapshot transitive imports.
    files += [ROOT/'scripts/native_branch_constraints/__init__.py',
              ROOT/'scripts/native_branch_constraints/objectives.py']
    files += list((ROOT/'thesis/new_constraints').rglob('*.py'))
    files += [f for f in (ROOT/'scripts/edge_ablation').glob('*') if f.suffix in ('.py', '.sh')]
    sources = {}
    for f in sorted(set(files)):
        if f.suffix == '.py':
            ast.parse(f.read_text(), str(f))
        sources[str(f.relative_to(ROOT))] = sha(f)
    p['source_sha256'] = sources
    p['identity_sha'] = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
    DOC.mkdir(parents=True, exist_ok=True)
    (DOC/'protocol.json').write_text(json.dumps(p, indent=2)+'\n')
    return p


if __name__ == '__main__':
    p = build()
    print(json.dumps(dict(name=p['name'], runs=p['main_runs'], identity=p['identity_sha'])))
