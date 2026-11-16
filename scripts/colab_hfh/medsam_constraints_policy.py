"""Frozen paired HFH MedSAM3 BAND gate and conditional constraint matrix."""
import math

ARMS = ('dice', 'dice_bands', 'dice_sdf', 'dice_bands_sdf', 'dice_bands_edge',
        'dice_bands_edge_sdf', 'dice_depth', 'dice_bands_depth', 'dice_edge_depth',
        'dice_sdf_depth', 'dice_bands_edge_depth', 'dice_bands_sdf_depth',
        'dice_edge_sdf_depth', 'dice_bands_edge_sdf_depth')
PAIRS = tuple((seed, fold) for seed in range(3) for fold in range(3))
STRICT_CONTROL = dict(name='strict_native', target=.1, ramp=20, native_control=True,
                      numerics='strict_empty_grid_v1')
STRICT_RECIPES = (
    dict(name='strict_gentle_t001_r20', target=.01, ramp=20, numerics='strict_empty_grid_v1'),
    dict(name='strict_scaled_t003_c015_r20', target=.03, cap=.15, ramp=20, numerics='strict_empty_grid_v1'),
    dict(name='strict_delayed100_t001_r20', target=.01, ramp=20, delay=100,
         full_native_graph=True, numerics='strict_empty_grid_v1'),
)
# A bounded, declared exploration. No seeds, folds, native optimizer or LR tuning.
RECIPES = (
    dict(name='canonical_t010_r20', target=.1, ramp=20),
    dict(name='gentle_t003_r20', target=.03, ramp=20),
    dict(name='gentle_t001_r20', target=.01, ramp=20),
    # Added after canonical/0.03 false-positive regressions. These are labelled
    # ablations, not silently substituted definitions of canonical BAND.
    dict(name='scaled_t003_c015_r20', target=.03, cap=.15, ramp=20),
    dict(name='outer75_t003_r20', target=.03, ramp=20, inner_weight=.25),
    dict(name='outer75_t010_r20', target=.1, ramp=20, inner_weight=.25),
    dict(name='gradual_t010_r100', target=.1, ramp=100),
    # Added only after all seven initial recipes failed their paired gate.
    # Preserve a native-only first half, then introduce weak BAND at lower LR.
    dict(name='delayed100_t001_r20', target=.01, ramp=20, delay=100),
    dict(name='delayed100_t003_r20', target=.03, ramp=20, delay=100),
    # Higher median targets are omitted: canonical f0s0 already saturates the
    # unchanged p95 cap, so increasing only the target is not a distinct dose.
)


def qualifies(rows):
    """Every original pair must improve; averages never conceal a regression."""
    if len(rows) != 9 or {(r['seed'], r['fold']) for r in rows} != set(PAIRS):
        return False
    return all(math.isfinite(r['validation_dice']) and .5 <= r['validation_dice'] <= 1
               and math.isfinite(r['delta_dice_pp']) and r['delta_dice_pp'] > 0
               for r in rows)


def matrix_tasks(recipe):
    return [(recipe, arm, seed, fold) for arm in ARMS
            if arm not in ('dice', 'dice_bands') for seed, fold in PAIRS]


def make_protocol(control, schedule, history, recipe, arm, archive_sha):
    if arm not in ARMS or (arm == 'dice' and not recipe.get('native_control')):
        raise ValueError('Expected an established constraint arm')
    if control['updates'] != 200 or len(history) != 200 or len(schedule) != 400:
        raise ValueError('Incomplete paired control')
    if control['accumulation'] != 2 or control.get('arm') != 'dice':
        raise ValueError('Control recipe differs')
    result = {k: v for k, v in control.items() if k != 'arm'}
    result.update(constraint_recipe=recipe, baseline_archive_sha256=archive_sha,
                  frozen_sample_schedule=schedule,
                  frozen_learning_rates=[r['lr'] for r in history],
                  gradient_ratio_targets={'bands': recipe['target']},
                  constraint_warmup_updates=recipe['ramp'],
                  comparison='Paired fixed-200-update exploratory validation; calibration cost recorded separately')
    if 'cap' in recipe:
        result['gradient_ratio_p95_caps'] = {'bands': recipe['cap']}
    if 'numerics' in recipe:
        result['training_numerics'] = recipe['numerics']
    if recipe.get('native_control'):
        result['comparison'] = 'Fresh strict-mode native control; historical comparison is descriptive only'
    if 'sdf' in arm or 'depth' in arm:
        result['constraint_geometry'] = dict(tensor_spacing_mm=[2., 1., 1.],
            bands='Two morphological voxel steps', sdf='Physical EDT clipped at 8 mm',
            depth='Physical spacing, source-axis reference; anatomical orientation unverified')
    return result
