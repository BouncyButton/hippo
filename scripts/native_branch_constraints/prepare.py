"""Freeze the authorized matrix and unchanged native-branch training recipe."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/experiments/native_branch_constraints_20261008'
NAME = 'native_branch_constraints_fold0_seed0_5pct_20261008_01'
BASE_OBJECTIVES = ('dice', 'dice_bands', 'dice_sdf', 'dice_bands_sdf',
                   'dice_bands_edge', 'dice_bands_edge_sdf')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    previous = ROOT/'docs/experiments/tissue_profiles_20261007/protocol.json'
    old = json.loads(previous.read_text())
    keep = ['dataset', 'fold', 'seed', 'max_epochs', 'baseline_loss', 'model', 'feature_size',
            'spatial_size', 'batch_size', 'drop_rate', 'activation_checkpointing', 'pretrained',
            'training_augmentation', 'precision', 'learning_rate', 'weight_decay', 'step_size',
            'lr_gamma', 'early_stopping', 'selection', 'geometry', 'required_monai', 'pkl',
            'pkl_sha256', 'native_geometry', 'low_data_fraction_actual', 'atlas_protocol_sha256',
            'required_dependencies', 'native_mri_input', 'calibration']
    p = {k: copy.deepcopy(old[k]) for k in keep}
    p['splits'] = {'5pct': copy.deepcopy(old['splits']['5pct'])}
    names = set(p['splits']['5pct']['train']+p['splits']['5pct']['val'])
    p['native_geometry'] = {k: v for k, v in p['native_geometry'].items() if k in names}
    objectives = [b+s for b in BASE_OBJECTIVES for s in ('', '_depth')]
    p.update(name=NAME, main_runs=12, fractions=['5pct'],
        runs=[dict(id='5pct_'+o+'_s0_native_mri_context', fraction='5pct', seed=0,
                   arm='native_mri_context', objective=o) for o in objectives],
        prior_protocol_sha256=sha(previous), prior_root=old['prior_root'],
        branch_policy='Identical native_mri_context architecture and guide ROI source membership as the completed native MRI branch. MRI-only auxiliary convolutions; original voxel lattice. No plane-view or unwrapped-view tensors, no position channels, no attention gates. Native network, encoder and fusion all train from scratch.',
        baseline_policy='Fresh Dice-only native-branch control included in this twelve-run allocation, with the same seed, case order, optimizer and stopping protocol as every constrained arm.',
        loss_definitions=dict(
            dice='Original MONAI three-class soft Dice, including background.',
            bands='Original balanced foreground/background BCE in the fixed two-step six-neighbour inner/outer GT bands; focal and degree weights disabled.',
            sdf='Original full-volume classwise anterior/posterior (p-y)*phi loss; half-voxel EDT, clipped at 8 voxels and divided by 8. Dataset native voxels must be isotropic 1 mm.',
            edge='Original signed foreground face-contrast residual squared, both endpoints in the GT inner/outer band union; uniform mean.',
            depth='Original depth-balanced three-class CE on GT foreground plus 8 mm, signed 2 mm bins; equal band mass with sparse-band cap at 64 voxels. Geometry is training supervision only.'),
        calibration_policy='Five matched Dice-only epochs. Calibrate all requested terms on the same ten training cases using sequential gradient RMS relative to Dice plus previously weighted terms (bands, original edge, SDF, depth). Median target 0.1, p95 cap 0.5. Freeze coefficients; ramp every requested term in epochs 6-10. No late SDF delay or compute-cap schedule from the separate capped campaign.',
        evaluation='Best validation A/P macro Dice; also foreground Dice, anterior/posterior Dice, missed voxels, extra voxels, A/P swaps, mean surface distance, HD95 and per-depth diagnostics. Save per-case differences versus this campaign Dice control and depth-on versus depth-off for each screenshot row.',
        interpretation='Single fold and seed exploratory comparison; improvement is assessed against the identical native-branch Dice control. Twelve validation comparisons are not independent confirmation.',
        resources=dict(partition='stud', gpu='4g.40gb:1', cpus=4, memory='24G', time='12:00:00'),
        minimum_free_gib=3., storage_budget='Twelve best checkpoints, one resumable optimizer checkpoint at a time, predictions and diagnostics; shared atlas is not duplicated.')
    sources = {}
    for name, expected in old['source_sha256'].items():
        if (name.startswith('scripts/depth_supervision/') or name.startswith('scripts/unwrapped_swin/')
                or name == 'baselines/swin_unetr/swin_unetr.py'
                or name in ['scripts/tissue_profiles/__init__.py', 'scripts/tissue_profiles/mri_views.py', 'scripts/tissue_profiles/views.py']):
            if sha(ROOT/name) != expected:
                raise ValueError('Matched branch or baseline implementation changed: '+name)
            sources[name] = expected
    paths = [ROOT/'scripts/compute_constraint_objectives.py']
    paths += list((ROOT/'scripts/native_branch_constraints').glob('*'))
    # Preserve package initializers and their transitive imports exactly.
    paths += list((ROOT/'thesis/new_constraints').rglob('*.py'))
    for path in sorted(paths):
        if path.suffix not in ('.py', '.sh'):
            continue
        if path.suffix == '.py':
            ast.parse(path.read_text(), str(path))
        sources[str(path.relative_to(ROOT))] = sha(path)
    p['source_sha256'] = sources
    p['identity_sha'] = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
    DOC.mkdir(parents=True, exist_ok=True)
    (DOC/'protocol.json').write_text(json.dumps(p, indent=2)+'\n')
    return p


if __name__ == '__main__':
    p = build()
    print(json.dumps(dict(name=p['name'], runs=len(p['runs']), identity=p['identity_sha'])))
