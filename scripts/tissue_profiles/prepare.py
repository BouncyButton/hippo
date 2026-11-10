"""Freeze pooled and multiscale Dice-plus-profile experiments and saved controls."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/experiments/tissue_profiles_20261007'
NAME = 'tissue_profiles_fold0_seed0_20261007_01'
PRIOR = '/mnt/beegfsstudents/home/3160552/unwrapped_swin_fold0_seed0_20261006_02'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    previous = ROOT/'docs/experiments/unwrapped_swin_20261006_repair/protocol.json'
    old = json.loads(previous.read_text())
    keep = ['dataset', 'fold', 'max_epochs', 'baseline_loss', 'model', 'feature_size', 'spatial_size',
            'batch_size', 'drop_rate', 'activation_checkpointing', 'pretrained', 'training_augmentation',
            'precision', 'learning_rate', 'weight_decay', 'step_size', 'lr_gamma', 'early_stopping',
            'selection', 'geometry', 'plane', 'required_monai', 'splits', 'low_data_fraction_actual',
            'pkl_sha256', 'uncompressed_pickle_sha256', 'native_geometry', 'pkl']
    p = {k: copy.deepcopy(old[k]) for k in keep}
    pairs = {'5pct': ['plane_position', 'unfold_position'], 'full': ['plane_attention', 'unfold_attention']}
    runs = []
    for fraction in ('5pct', 'full'):
        for arm in ['native', *pairs[fraction]]:
            for variant in ('profile', 'multiscale'):
                suffix = '' if arm == 'native' else '_'+arm
                runs.append(dict(id=f'{fraction}_dice_{variant}_s0{suffix}', fraction=fraction, seed=0,
                                 profile_variant='pooled' if variant == 'profile' else 'multiscale', arm=arm))
    for fraction in ('5pct', 'full'):
        for variant, label in [('none', 'dice'), ('pooled', 'dice_profile'), ('multiscale', 'dice_multiscale')]:
            runs.append(dict(id=f'{fraction}_{label}_s0_informed', fraction=fraction, seed=0,
                             profile_variant=variant, arm='informed'))
    mri_pairs = {'5pct': ['plane_mri_context', 'unfold_mri_context'],
                 'full': ['plane_mri_attention', 'unfold_mri_attention']}
    for fraction, arms in mri_pairs.items():
        for arm in arms:
            runs.append(dict(id=f'{fraction}_dice_s0_{arm}', fraction=fraction, seed=0,
                             profile_variant='none', arm=arm))
    # Preserve the full planned 22-run order; add only the fourteen missing cells.
    for fraction, arms in mri_pairs.items():
        for arm in arms:
            for variant, label in [('pooled', 'profile'), ('multiscale', 'multiscale')]:
                runs.append(dict(id=f'{fraction}_dice_{label}_s0_{arm}', fraction=fraction, seed=0,
                                 profile_variant=variant, arm=arm))
    native_pairs = {'5pct': 'native_mri_context', 'full': 'native_mri_attention'}
    for fraction, arm in native_pairs.items():
        for variant, label in [('none', 'dice'), ('pooled', 'dice_profile'), ('multiscale', 'dice_multiscale')]:
            runs.append(dict(id=f'{fraction}_{label}_s0_{arm}', fraction=fraction, seed=0,
                             profile_variant=variant, arm=arm))
    p.update(name=NAME, seed=0, prior_root=PRIOR, prior_protocol_sha256=sha(previous),
        atlas_protocol_sha256=old['identity_sha'], main_runs=36, runs=runs, view_pairs=pairs,
        mri_view_pairs=mri_pairs,
        native_mri_arms=native_pairs,
        mri_view_input=dict(channels=['MRI intensity'], objectives=['Dice only', 'Dice + pooled profiles', 'Dice + multiscale profiles'],
            architecture='Match existing plane/unwrapped context branches at 5% and attention branches at full data. Both auxiliary input convolutions have one MRI channel; no longitudinal, depth or support channel is learned.',
            bookkeeping='Keep chart valid-sample masks for masked global pooling and native return indices/support for mapping and the unchanged fusion. These are geometric operations, not extra auxiliary convolution input channels.',
            initialization='Retain the seed-matched original branch weights throughout; slice the input convolution weights to their MRI channel. All parameters train jointly.',
            interpretation='Compare with existing Dice-only position-enabled branches and native MRI controls. Different-view processing, extra capacity, sampling and geometric routing remain bundled; MRI-only branches still require the frozen guide geometry.'),
        native_mri_input=dict(channels=['MRI intensity'],
            architecture='Same MRI-only auxiliary encoder, context, attention and fusion as its fraction-matched plane/unwrapped controls. Identical parameter count and seed-matched initial state.',
            sampling='Keep the exact same guide ROI source voxels and chart memberships. For each chart, sample original MRI voxel centers in the original axis orientation; crop to the chart bounding box, add one empty-voxel margin, pad dimensions to multiples of eight, and preserve separate chart support. Map back using native indices and identity-coordinate grid sampling.',
            precision='Match alternative-view input storage by casting MRI samples to float16 before encoding; no interpolation is applied to native MRI samples.',
            differences='Native and alternative-view grids differ in orientation, spacing, tensor shape, empty-cell layout and interpolation. Native pooling groups the same source membership but the raster support and effective convolution footprint differ. This controls branch capacity and chart partitions, not all sampling effects.',
            labels='No annotations are used to construct these inputs; use only native MRI and frozen guide chart membership.'),
        analysis_plan=dict(metrics=['macro_dice', 'anterior_dice', 'posterior_dice', 'foreground_dice', 'missed', 'extra', 'swaps', 'surface_distance_mean_mm', 'hd95_mm', 'parameter_count', 'runtime'],
            comparisons=['native MRI auxiliary versus ordinary SwinUNETR at each loss', 'plane/unwrapped MRI auxiliary versus native MRI auxiliary at each loss', 'position-enabled versus MRI-only auxiliary at each view and loss', 'directly informed SwinUNETR versus auxiliary architectures at each loss', 'each constrained architecture versus its own Dice control'],
            replication='Single-seed results are preliminary. Prioritize matched native-aux versus ordinary, alternative-view versus native-aux, position-enabled versus MRI-only at the winning view, and constraint versus own Dice comparisons for seeds 1 and 2 on the same split. Replicate both fractions before reliability claims; selection on this validation set requires held-out confirmation.'),
        informed_input=dict(channels=['native MRI', 'longitudinal position / 32 mm',
                                     'signed plane depth / 16 mm', 'guide ROI support'],
            source='Same fraction-specific frozen OOF guide-derived center, basis and native ROI indices as the auxiliary branches; no annotation or UV input.',
            support='Guide foreground plus 8 mm context, intersected with original MRI FOV. Position channels and support are zero outside this ROI. MRI remains unchanged across the full padded native grid.',
            architecture='One four-input-channel SwinUNETR, no auxiliary encoder, view resampling, or feature fusion. All parameters train jointly.',
            initialization='Keep the seed-matched one-channel model weights, expand its three input convolutions to four channels, initialize additional channel weights to zero and train them from the first update. No frozen parameters.',
            interpretation='Tests direct input of the same anatomical descriptors against MRI only and existing view branches. Native coordinates are exact before chart rasterization. Branch comparisons still differ in capacity, interpolation, chart neighborhoods and fusion; they cannot by themselves isolate unfolding.'),
        branch_policy='Retain all four native-only tests and add the matched plane and unwrapped input branches for both losses and both fractions. Reuse previously selected position/context branches at 5% and attention branches at full data, with ordinary fusion, no entropy gates. Train all native/branch/fusion parameters jointly from scratch. The anatomical loss is applied to the final native-grid logits identically for all views; only the input branch changes.',
        branch_baseline_root='/mnt/beegfsstudents/home/3160552/unwrapped_followup_fold0_20261007_01',
        branch_baseline_identity=json.loads((ROOT/'docs/experiments/unwrapped_followup_20261007/protocol.json').read_text())['identity_sha'],
        required_dependencies={'monai': '1.6.0'},
        profile_geometry=dict(band_mm=2., longitudinal_mm=2.),
        multiscale_geometry=[dict(band_mm=2., longitudinal_mm=4., transverse_mm=4.),
                             dict(band_mm=2., longitudinal_mm=2., transverse_mm=2.)],
        multiscale_definition='Within each signed 2 mm depth layer retain transverse strips and longitudinal bins at two resolutions: 4 mm and 2 mm. Compute the same symmetric cumulative discrepancy separately for every depth/strip/class row. Average the two scale losses equally; each has one global GT foreground volume denominator and longitudinal-extent normalization. No strip or depth balancing. Native probabilities and GT use identical fixed voxel-to-cell maps. Finite cells can still hide within-cell cancellation; Dice remains.',
        calibration=dict(prefix_epochs=5, ramp_epochs=5, target_ratio=.1, cap_ratio=.5, max_cases=32),
        permitted_losses=['Dice', 'symmetric cumulative anterior/posterior tissue-profile distance'],
        profile_definition='Sum native voxel probability times physical voxel volume in separate signed-depth and longitudinal bins. Compare forward and reverse longitudinal cumulative sums, sum absolute differences over both foreground labels and all depth bins, divide by 2 * number of longitudinal bins * max(total GT foreground mm3, 1 mm3). No per-bin reweighting, independent mass normalization, CE, SDF, focal or topology terms.',
        coordinate_policy='The loss reuses only the frozen guide-derived atlas center and orthonormal basis. Same fixed frame for probabilities and targets. Training guides are out-of-fold. All native MRI field-of-view voxels contribute, including background; artificial zero padding does not. Native controls have no branch; plane/unwrapped variants also use the existing view inputs and native return indices for feature fusion. Neither view rendering defines the loss correspondence.',
        calibration_policy='After five Dice-only epochs, compute one fixed coefficient per run on at most 32 training cases using median logit-gradient RMS ratio 0.1 capped at p95 ratio 0.5; ramp linearly in epochs 6-10. For multiscale calibrate the mean of both scales as a single term. No validation-based weight selection.',
        baseline_policy='Reuse completed native Dice-only controls for these exact case splits, seed, preprocessing, optimizer and stopping protocol; do not rerun them.',
        evaluation='Native A/P macro Dice, foreground Dice, false positives, false negatives, A/P swaps, mean surface distance and HD95; paired per-case changes versus existing Dice controls. Existing GT-depth diagnostic fields are used only during evaluation.',
        resources=dict(partition='stud', gpu='4g.40gb:1', cpus=4, memory='24G', time='23:50:00'),
        expected_predecessor='687700', minimum_free_gib=6.,
        amendment=dict(job_id='688228', original_identity='3a94effb137683b28793a379f7a7adc0d2c08c060c2feb5d45908daa7881cef9',
                       previous_identity='e263d78cf7d0f21a75766ada354fea393374a36712c319a215cea4c9fb8270c1',
                       reason='Replace the held, unstarted job with the unchanged original twelve, preserve the next ten planned input controls, then append eight constrained MRI-only view runs and six native-view MRI-only auxiliary controls.'),
        storage_budget='Reserve 6 GiB for thirty-six best checkpoints (about 2.2 GiB), one active optimizer checkpoint and atomic write overlap, predictions, diagnostics and remaining predecessor output. Frozen atlas inputs are shared, not duplicated.')
    sources = {}
    for name, expected in old['source_sha256'].items():
        if (name.startswith('scripts/depth_supervision/') or name == 'baselines/swin_unetr/swin_unetr.py'
                or name in ['scripts/unwrapped_swin/__init__.py', 'scripts/unwrapped_swin/models.py', 'scripts/unwrapped_swin/evaluate.py']):
            if sha(ROOT/name) != expected:
                raise ValueError('Matched native baseline code changed: '+name)
            sources[name] = expected
    for path in sorted((ROOT/'scripts/tissue_profiles').glob('*')):
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
    print(json.dumps(dict(protocol=str(DOC/'protocol.json'), identity=p['identity_sha'], runs=p['main_runs'])))
