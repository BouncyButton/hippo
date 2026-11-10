"""Finalize the authorized eight-arm protocol from measured arithmetic costs."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from compute_constraint_objectives import ARMS, scheduled_costs


def main():
    directory = ROOT / 'experiments/compute_matched_constraints_20pct_20261004'
    report = ROOT / 'reports/compute_matched_constraints_20pct_20261004'
    updates = json.loads((report / 'unet_profiles.json').read_text())
    calibration = json.loads((report / 'unet_calibration_profiles.json').read_text())
    reference_path = ROOT / 'experiments/compute_budget_reference.json'
    reference = json.loads(reference_path.read_text())
    budget = reference['budget_flops']
    assert budget == 8_138_082_968_869
    for profiles in (updates, calibration):
        for name, expected in profiles['source_sha256'].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    reference['signed_distance_policy'] = 'Authorized full-volume anterior/posterior SDF boundary loss; clipped at +/-8 model-grid voxels, normalized by 8, no extra prediction head; calibrated after bands and original edge when present.'
    reference['constraints_schedule'] = str((directory / 'protocol.json').relative_to(ROOT))
    reference_path.write_text(json.dumps(reference, indent=2) + '\n')
    protocol = json.loads((directory / 'draft_protocol.json').read_text())
    protocol.update(status='ready', name='compute_matched_constraints20_unet_fold0_seed0',
                    pilot_scope='All eight subsets of bands/original edge/full-volume classwise SDF; U-Net, fold 0, seed 0 only.',
                    reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                    training_loss='Dice + ramp * (connected_zero + band_weight*bands + edge_weight*original_edge + sdf_weight*SDF); absent terms omitted',
                    calibration_policy='50 Dice-only updates; ten training cases with no augmentation or validation. Sequential calibration in fixed order: bands relative to Dice; edge relative to Dice plus weighted bands; SDF relative to Dice plus all preceding weighted terms. Each coefficient targets median gradient RMS ratio .1 and caps p95 at .5. All forwards/logit gradients and cross moments charged within B_ref.',
                    sources='Canonical two-step balanced BCE bands and original pooled edge consistency; full-volume classwise clipped signed-distance boundary loss.')
    protocol['signed_distance'] = protocol.pop('signed_distance_proposal')
    protocol['signed_distance'].update(status='authorized_and_implemented',
        empty_class='constant +1 normalized distance', full_class='constant -1 normalized distance',
        implementation='thesis/new_constraints/signed_distance.py',
        loss='Full-volume mean (p_c-y_c)*phi_c over classes 1 and 2; no band mask',
        computation='Recompute from transformed training labels before the training forward; cache neither augmented geometry nor validation labels.')
    protocol['profile_evidence'] = {
        'update_profiles_sha256': hashlib.sha256((report / 'unet_profiles.json').read_bytes()).hexdigest(),
        'calibration_profiles_sha256': hashlib.sha256((report / 'unet_calibration_profiles.json').read_bytes()).hexdigest(),
        'geometries': ['ellipsoid', 'box'],
        'postprocessing_scalar_flop_allowance': calibration['postprocessing_scalar_flop_allowance'],
        'distance_preprocessing': 'CPU Euclidean distance transforms excluded by the established preprocessing convention; included in reported run wall time.'}
    objectives = {}
    for arm in ARMS:
        cost = updates['schedules'][arm]['flops_per_update']
        cal = calibration['costs'][arm]
        schedule = scheduled_costs(budget, 9_580_978_301, cost, cal, 50)
        schedule.update(flops_per_update=cost, baseline_flops_per_update=9_580_978_301)
        objectives[arm] = schedule
    assert objectives['dice']['max_steps'] == 849
    assert objectives['dice_bands']['max_steps'] == objectives['dice_edge']['max_steps'] == 840
    assert objectives['dice_bands_edge']['max_steps'] == 834
    protocol['models']['unet']['objectives'] = objectives
    (directory / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    (report / 'training_schedules.json').write_text(json.dumps({
        'budget_flops': budget, 'budget_gflops': budget/1e9,
        'calibration_included': True, 'objectives': objectives}, indent=2) + '\n')
    for arm, s in objectives.items():
        print(arm, s['max_steps'], 'updates;', s['patient_pass_epochs'], 'epochs;',
              s['calibration_flops']/1e9, 'calibration GFLOPs')


if __name__ == '__main__':
    main()
