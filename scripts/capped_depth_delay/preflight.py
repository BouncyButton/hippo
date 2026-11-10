"""Verify source bindings, phase gradients, original depth CE and geometry."""
import argparse
import json
from pathlib import Path

from campaign90 import BUDGET, MODELS, digest, write_json
from scripts.capped_depth_delay.constants import CAMPAIGN, ARMS
from scripts.capped_depth_delay.schedule import training_flops, charge_for_step, sdf_scale_for_step


def check_frozen(stage, pkl):
    source = stage / 'source'
    manifest = json.loads((stage / 'source_manifest.json').read_text())
    for name, sha in manifest.items():
        if digest(source / name) != sha:
            raise ValueError('Frozen source changed: ' + name)
    plan_path = source / 'experiments' / CAMPAIGN / 'plan.json'
    plan = json.loads(plan_path.read_text())
    if (plan['total_runs'] != 44 or plan['budget_flops'] != BUDGET
            or plan['budget_scope'] != 'optimization_only' or plan['fold'] != 0 or plan['seed'] != 0
            or not plan['single_sbatch_allocation'] or plan['calibration_in_cap']):
        raise ValueError('Authorized scope differs')
    protocols = {}
    for identity, spec in MODELS.items():
        directory = source / 'experiments' / CAMPAIGN / identity
        protocol = json.loads((directory / 'protocol.json').read_text())
        if (digest(pkl) != protocol['pkl_sha256'] or protocol['budget_flops'] != BUDGET
                or protocol['model_identity'] != identity or protocol['trainer_alias'] != spec['alias']
                or protocol['requested_arm_order'] != list(ARMS)
                or protocol['lr_horizon_updates'] != spec['lr_horizon']):
            raise ValueError('Model/data/objective identity differs')
        for name, sha in protocol['training_source_sha256'].items():
            if digest(source / name) != sha:
                raise ValueError('Production/profile source binding differs')
        for name, key in [('splits_10patients.json', 'splits_sha256'),
                          ('nnunet_architecture.json', 'nnunet_architecture_sha256'),
                          ('native_geometry.json', 'native_geometry_sha256'),
                          ('calibration_references.json', 'calibration_references_sha256')]:
            if digest(directory / name) != protocol[key]:
                raise ValueError('Frozen input differs: ' + name)
        evidence = protocol['profile_evidence']
        if digest(source / evidence['file']) != evidence['sha256']:
            raise ValueError('Measured costs differ')
        for arm in ARMS:
            schedule = protocol['models'][spec['alias']]['objectives'][arm]
            n = schedule['max_steps']
            if not training_flops(n, schedule) <= BUDGET < training_flops(n + 1, schedule):
                raise ValueError('Strict update allowance differs')
            for step in range(n + 2):
                if training_flops(step, schedule) != sum(charge_for_step(i, schedule) for i in range(step)):
                    raise ValueError('Piecewise accounting differs')
            if 'sdf' in arm:
                late = schedule['sdf_prefix_updates']
                target = BUDGET * 5 // 9
                choices = [max(schedule['prefix_updates'], late - 1), late, late + 1]
                if late != min(choices, key=lambda k: (abs(training_flops(k, schedule) - target), k)):
                    raise ValueError('SDF timing not nearest to the predeclared budget fraction')
        protocols[identity] = protocol
    return plan, protocols


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--pkl', type=Path, required=True)
    args = parser.parse_args()
    plan, protocols = check_frozen(args.stage, args.pkl)
    import numpy as np
    import torch
    import monai
    from scripts.capped_depth_delay.objectives import (
        ARMS as OBJECTIVE_ARMS, training_loss, full_loss, geometry_weights,
        augmented_affine, calibration_case, select_weights)
    if OBJECTIVE_ARMS != ARMS:
        raise ValueError('Authorized arms and production objectives differ')
    if torch.__version__ != '2.12.1+cu130' or monai.__version__ != '1.6.0':
        raise ValueError('Use the previously validated production runtime')
    torch.set_num_threads(2)
    import train_compute_matched_baselines as base
    from scripts.depth_supervision.losses import components
    from thesis.new_constraints.signed_distance import signed_distance_maps
    from nnunet_multi_objective import training_loss as native_loss
    from compute_constraint_objectives import training_loss as monai_loss
    from profile_unet_constraint_budget import synthetic_target

    protocol = protocols['nnunet']
    architecture = json.loads((args.stage / 'source/experiments' / CAMPAIGN / 'nnunet/nnunet_architecture.json').read_text())
    model = base.build_model('nnunet', architecture)
    criterion, optimizer = base.build_loss_and_optimizer('nnunet', model, protocol)
    if sum(p.numel() for p in model.parameters()) != MODELS['nnunet']['parameters'] or optimizer.param_groups[0]['momentum'] != .95:
        raise ValueError('Native architecture or optimizer differs')
    labels = synthetic_target('ellipsoid')
    sdf = signed_distance_maps(labels)
    affine = np.eye(4)
    affine[:3, :3] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    depth = geometry_weights(labels, affine, protocol['depth_geometry'])
    if not torch.isfinite(depth).all() or not torch.isclose(depth.sum(), torch.tensor(1.), atol=1e-6):
        raise ValueError('Depth weights not finite/normalized')
    outputs = [torch.randn(1, 3, n, n, n, requires_grad=True) for n in (64, 32, 16, 8)]
    weights = dict(bands=.04, edge=.1, sdf=.1, depth=.1)
    from scripts.capped_depth_delay.objectives import depth_loss
    original_depth = components(outputs[0], labels, {'weights': depth.numpy()}, 'depth')['depth']
    if not torch.equal(original_depth, depth_loss(outputs[0], labels, depth)):
        raise ValueError('Original depth CE changed')
    for native_criterion, raw in ((None, outputs[0]), (criterion, outputs)):
        for arm in ARMS:
            for late_scale in ((1/9, 1.) if 'sdf' in arm else (0.,)):
                actual = training_loss(raw, labels, arm, weights, 1., sdf, native_criterion, depth, late_scale)
                adjusted = {**weights, 'sdf': weights['sdf'] * late_scale} if 'sdf' in arm else weights
                expected = full_loss(raw, labels, arm, adjusted, 1., native_criterion, sdf, depth)
                variable = outputs[:3] if native_criterion is not None else [outputs[0]]
                actual_gradient = torch.autograd.grad(actual, variable, retain_graph=True)
                expected_gradient = torch.autograd.grad(expected, variable, retain_graph=True)
                if not torch.equal(actual, expected) or any(not torch.equal(a, b) or not torch.isfinite(a).all() for a, b in zip(actual_gradient, expected_gradient)):
                    raise ValueError('Loss or phase gradients differ')
            record = calibration_case(raw, labels, arm, sdf, native_criterion, depth)
            selected, statistics = select_weights([record], arm)
            if any(not np.isfinite(value) or value <= 0 for value in selected.values()):
                raise ValueError('Invalid gradient-calibrated coefficient')
            for stat in statistics.values():
                if not 0 < stat['weighted_median_ratio'] <= .1 + 1e-8:
                    raise ValueError('Calibrated gradient ratio differs')
        for arm in ('dice_depth', 'dice_bands_edge_sdf_depth'):
            modified = full_loss(raw, labels, arm, weights, 1., native_criterion, sdf, depth)
            original = arm.removesuffix('_depth')
            unchanged = (native_loss(raw, labels, original, weights, 1., native_criterion, sdf)
                         if native_criterion is not None else monai_loss(raw, labels, original, weights, 1., sdf))
            expected = unchanged + weights['depth'] * original_depth
            if not torch.equal(modified, expected):
                raise ValueError('Depth changes another objective')
            if native_criterion is not None:
                ga = torch.autograd.grad(modified, outputs[1:3], retain_graph=True)
                gb = torch.autograd.grad(unchanged, outputs[1:3], retain_graph=True)
                if any(not torch.equal(a, b) for a, b in zip(ga, gb)):
                    raise ValueError('Auxiliary depth loss changed coarse-head supervision')
    dataframe = base._load_pkl_dataframe(args.pkl)
    split = json.loads((args.stage / 'source/experiments' / CAMPAIGN / 'nnunet/splits_10patients.json').read_text())[0]
    patients = base.PreparedPatients(dataframe, split['train'])
    accepted = 0
    for patient_index, item in enumerate(patients):
        info = protocol['native_geometry'][item['case_name']]
        import hashlib
        raw = dataframe[dataframe['subject_id'].astype(str) == item['case_name']].iloc[0]['label_data']
        if hashlib.sha256(np.asarray(raw).astype(np.uint8).tobytes()).hexdigest() != info['label_array_sha256']:
            raise ValueError('Native geometry annotation identity differs')
        generator = torch.Generator().manual_seed(2000 + patient_index)
        state = generator.get_state()
        image, target, stats = base.augment_mild(item['image'][None], item['label'][None], generator=generator)
        converted = augmented_affine(info['model_affine'], state, stats, target.shape[-3:])
        if stats['spatial_applied']:
            # Independently recover the actual nearest-neighbor label warp
            # from the physical affine, checking axis order and voxel centers.
            import torch.nn.functional as functional
            transform = np.linalg.inv(np.asarray(info['model_affine'])) @ converted
            reverse = np.eye(3)[::-1]
            matrix = reverse @ transform[:3, :3] @ reverse
            offset = reverse @ transform[:3, 3]
            size = np.asarray(target.shape[-3:][::-1], dtype=float)
            extent = np.diag(size / 2)
            center = (size - 1) / 2
            theta = np.column_stack((np.linalg.solve(extent, matrix @ extent),
                np.linalg.solve(extent, matrix @ center + offset - center)))
            grid = functional.affine_grid(torch.from_numpy(theta).float()[None], target.shape, align_corners=False)
            reconstructed = functional.grid_sample(item['label'][None].float(), grid,
                mode='nearest', padding_mode='zeros', align_corners=False).to(target.dtype)
            if not torch.equal(reconstructed, target):
                raise ValueError('Physical affine does not reproduce the actual augmented label warp')
        w = geometry_weights(target, converted, protocol['depth_geometry'])
        if not torch.isfinite(w).all() or not torch.isclose(w.sum(), torch.tensor(1.), atol=1e-6):
            raise ValueError('Augmented physical geometry invalid')
        accepted += stats['spatial_applied']
    if accepted == 0:
        raise ValueError('No accepted spatial augmentation tested')
    checks = dict(all44_identity_source_and_strict_budget_checks=True,
                  original_depth_ce_exact=True, native_coarse_head_gradients_unchanged=True,
                  delayed_sdf_phase_losses_and_gradients_exact=True,
                  calibration_positive_and_training_only=True,
                  original_geometry_all10_annotations_verified=True,
                  augmentation_physical_affine_verified=True,
                  sdf_full_volume_definition_unchanged=True,
                  budget_scope='optimization_only')
    write_json(args.stage / 'preflight.json', dict(status='passed', checks=checks,
        plan_sha256=digest(args.stage / 'source/experiments' / CAMPAIGN / 'plan.json'),
        source_manifest_sha256=digest(args.stage / 'source_manifest.json'),
        torch=torch.__version__, monai=monai.__version__, total_runs=44))
    print(json.dumps({'preflight': 'passed', 'checks': checks}), flush=True)


if __name__ == '__main__':
    main()
