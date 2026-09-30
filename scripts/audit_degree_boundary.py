#!/usr/bin/env python3
"""Paired boundary corrections from selected checkpoints; inference only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy import ndimage as ndi

CROSS = ndi.generate_binary_structure(3, 1)
KERNEL = CROSS.astype(np.int16)
KERNEL[1, 1, 1] = 0
STATES = ['correct', 'fn', 'fp', 'ap_swap']


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def supports(truth):
    fg = truth > 0
    e1 = ndi.binary_erosion(fg, structure=CROSS, iterations=1)
    e2 = ndi.binary_erosion(fg, structure=CROSS, iterations=2)
    d1 = ndi.binary_dilation(fg, structure=CROSS, iterations=1)
    d2 = ndi.binary_dilation(fg, structure=CROSS, iterations=2)
    inner, outer = fg & ~e2, d2 & ~fg
    masks = {'all': np.ones_like(fg), 'inner': inner, 'outer': outer,
             'boundary': inner | outer, 'beyond_bands': ~(inner | outer),
             'inner_layer1': fg & ~e1, 'inner_layer2': e1 & ~e2,
             'outer_layer1': d1 & ~fg, 'outer_layer2': d2 & ~d1,
             'deep_foreground': e2, 'far_background': ~d2}
    degree = ndi.convolve(fg.astype(np.int16), KERNEL, mode='constant', cval=0)
    for region, mask in [('whole', fg), ('anterior', truth == 1), ('posterior', truth == 2)]:
        for label, lo, hi in [('0_1', 0, 1), ('2_3', 2, 3), ('4_5', 4, 5), ('6', 6, 6)]:
            masks[f'{region}/whole_degree/{label}'] = mask & (degree >= lo) & (degree <= hi)
        for k in range(7):
            masks[f'{region}/whole_degree_exact/{k}'] = mask & (degree == k)
        if region != 'whole':
            local_degree = ndi.convolve(mask.astype(np.int16), KERNEL, mode='constant', cval=0)
            for k in range(7):
                masks[f'{region}/class_degree/{k}'] = mask & (local_degree == k)
            masks[f'{region}/inner'] = mask & inner
    masks['ap_interface'] = ((truth == 1) & ndi.binary_dilation(truth == 2, structure=CROSS)) | (
        (truth == 2) & ndi.binary_dilation(truth == 1, structure=CROSS))
    return masks


def error_states(truth, prediction):
    result = np.zeros_like(truth, dtype=np.uint8)
    result[(truth > 0) & (prediction == 0)] = 1
    result[(truth == 0) & (prediction > 0)] = 2
    result[(truth > 0) & (prediction > 0) & (truth != prediction)] = 3
    return result


def region_counts(state, mask):
    counts = np.bincount(state[mask], minlength=4)
    return {'voxels': int(mask.sum()), 'correct': int(counts[0]),
            'fn': int(counts[1]), 'fp': int(counts[2]), 'ap_swap': int(counts[3]),
            'union_errors': int(counts[1] + counts[2]), 'class_errors': int(counts[1:].sum())}


def transitions(old, new, mask):
    matrix = np.bincount((4 * old[mask] + new[mask]).astype(np.int64), minlength=16).reshape(4, 4)
    old_union, new_union = (old == 1) | (old == 2), (new == 1) | (new == 2)
    corrected = int((mask & old_union & ~new_union).sum())
    introduced = int((mask & ~old_union & new_union).sum())
    return {'state_transition_matrix': matrix.tolist(),
            'union_corrected': corrected, 'union_introduced': introduced,
            'union_net_corrected': corrected - introduced,
            'class_corrected': int(matrix[1:, 0].sum()),
            'class_introduced': int(matrix[0, 1:].sum()),
            'class_net_corrected': int(matrix[1:, 0].sum() - matrix[0, 1:].sum())}


def binary_metrics(truth, pred):
    denominator = int(truth.sum() + pred.sum())
    result = {'dice': float(2 * (truth & pred).sum() / denominator) if denominator else 1.0}
    a = truth & ~ndi.binary_erosion(truth, structure=CROSS)
    b = pred & ~ndi.binary_erosion(pred, structure=CROSS)
    if a.any() and b.any():
        distances = np.concatenate([ndi.distance_transform_edt(~a)[b], ndi.distance_transform_edt(~b)[a]])
        result.update(assd_mm=float(distances.mean()), hd95_mm=float(np.quantile(distances, .95)),
                      surface_dice_1mm=float((distances <= 1).mean()),
                      surface_dice_2mm=float((distances <= 2).mean()))
    else:
        result.update(assd_mm=None, hd95_mm=None, surface_dice_1mm=None, surface_dice_2mm=None)
    return result


def score_case(truth, pred, masks):
    state = error_states(truth, pred)
    regions = {key: region_counts(state, mask) for key, mask in masks.items()}
    metrics = {}
    for region, a, b in [('whole', truth > 0, pred > 0),
                          ('anterior', truth == 1, pred == 1), ('posterior', truth == 2, pred == 2)]:
        metrics.update({f'{region}/{k}': v for k, v in binary_metrics(a, b).items()})
    metrics['macro_dice'] = (metrics['anterior/dice'] + metrics['posterior/dice']) / 2
    metrics['boundary_union_errors'] = regions['boundary']['union_errors']
    metrics['boundary_class_errors'] = regions['boundary']['class_errors']
    metrics['inner_fn_rate'] = regions['inner']['fn'] / regions['inner']['voxels']
    metrics['outer_fp_rate'] = regions['outer']['fp'] / regions['outer']['voxels']
    metrics['balanced_boundary_error'] = .5 * (metrics['inner_fn_rate'] + metrics['outer_fp_rate'])
    return {'metrics': metrics, 'regions': regions}, state


def summarize(rows):
    means = {}
    for key in rows[0]['metrics']:
        values = [r['metrics'][key] for r in rows if r['metrics'][key] is not None]
        means[key] = {'mean': float(np.mean(values)) if values else None, 'valid_cases': len(values)}
    pooled = {}
    for region in rows[0]['regions']:
        pooled[region] = {key: sum(r['regions'][region][key] for r in rows)
                          for key in rows[0]['regions'][region]}
    return {'case_count': len(rows), 'mean_case_metrics': means, 'pooled_regions': pooled}


def summarize_pair(old_rows, new_rows, pair_rows):
    assert [r['case_name'] for r in old_rows] == [r['case_name'] for r in new_rows]
    bootstrap_indices = np.random.default_rng(20260924).integers(0, len(old_rows), size=(10000, len(old_rows)))
    differences = {}
    for key in old_rows[0]['metrics']:
        if any(r['metrics'][key] is None for r in old_rows + new_rows):
            continue
        delta = np.array([b['metrics'][key] - a['metrics'][key] for a, b in zip(old_rows, new_rows)])
        differences[key] = {'mean_delta': float(delta.mean()),
                            'paired_case_bootstrap_95_interval': np.quantile(delta[bootstrap_indices].mean(1), [.025, .975]).tolist(),
                            'positive_cases': int((delta > 0).sum()), 'negative_cases': int((delta < 0).sum()),
                            'equal_cases': int((delta == 0).sum())}
    pooled = {}
    for region in pair_rows[0]['regions']:
        pooled[region] = {}
        for key in pair_rows[0]['regions'][region]:
            if key == 'state_transition_matrix':
                pooled[region][key] = np.sum([r['regions'][region][key] for r in pair_rows], axis=0).tolist()
            else:
                pooled[region][key] = sum(r['regions'][region][key] for r in pair_rows)
    return {'delta_direction': 'candidate minus reference; lower error/distance is better; higher Dice is better',
            'metrics': differences, 'pooled_transitions': pooled}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'audit.lock').open('x') as f:
        f.write(str(os.getenv('SLURM_JOB_ID')))
    try:
        source = args.experiment / 'source'
        payload = json.loads((source / 'PAYLOAD.json').read_text())
        for name, expected in payload['files'].items():
            if digest(source / name) != expected:
                raise ValueError(f'Frozen source changed: {name}')
        sys.path.insert(0, str(source))
        import torch
        from thesis.new_constraints import train_swinunetr_constraints as trainer
        trainer.seed_everything(0)
        assert torch.cuda.is_available(), 'CUDA allocation required'
        device = torch.device('cuda')
        assert json.loads((args.experiment / 'completion.json').read_text())['status'] == 'complete'
        existing = json.loads((args.experiment / 'audit/summary.json').read_text())
        bindings = json.loads((args.experiment / 'reference_bindings.json').read_text())
        runs = {'dice': Path(bindings['dice']['directory']), 'bands': Path(bindings['bands']['directory']),
                'A': args.experiment / 'runs/A', 'B': args.experiment / 'runs/B'}
        configs, manifest, identity = {}, {}, None
        for name, directory in runs.items():
            config = json.loads((directory / 'config.json').read_text())
            run = config['run']
            completion = json.loads((directory / 'completion_manifest.json').read_text())
            assert completion['status'] == 'complete' and completion['run'] == run
            assert run['early_stopping_patience'] == 8 and run['early_stopping_min_epochs'] == 25
            assert run['dataset'] == 'MSD' and run['fold'] == 0 and run['seed'] == 0
            keys = ('spatial_size', 'resize', 'training_augmentation', 'amp', 'drop_rate', 'batch_size',
                    'optimizer_mode', 'learning_rate', 'weight_decay', 'step_size', 'adamw_gamma',
                    'early_stopping_patience', 'early_stopping_min_epochs', 'early_stopping_min_delta')
            settings = {k: run[k] for k in keys}
            if identity is None: identity = settings
            assert settings == identity, 'Unmatched control settings'
            for path, sha in [('pkl', 'pkl_sha256'), ('splits_json', 'splits_json_sha256')]:
                assert digest(config[path]) == config[sha]
                if configs: assert config[sha] == configs['dice'][sha]
            checkpoint = directory / 'checkpoint_best.pt'
            sha = digest(checkpoint)
            assert sha == existing[name]['checkpoint_sha256']
            assert digest(directory / 'MSD_fold0/model.pt') == completion['artifacts']['MSD_fold0/model.pt']
            configs[name] = config
            manifest[name] = {'directory': str(directory), 'checkpoint_sha256': sha,
                              'selected_epoch': completion['selected_epoch'], 'stopped_epoch': completion['epoch'],
                              'training_execution': config['execution_provenance']}
        run = configs['dice']['run']
        assert run['constraint_set'] == 'none' and run.get('supervised_loss', 'dice') == 'dice'
        data_args = SimpleNamespace(pkl=Path(configs['dice']['pkl']), splits_json=Path(configs['dice']['splits_json']),
                                   dataset='MSD', fold=0, spatial_size=run['spatial_size'], resize=run['resize'],
                                   num_classes=3, num_workers=0, batch_size=1, plain_tensors=True)
        _, loader, _, train_n, val_n = trainer.build_data(data_args, torch.Generator().manual_seed(0))
        assert (train_n, val_n) == (208, 52)
        save(args.output / 'manifest.json', {'job_id': os.getenv('SLURM_JOB_ID'), 'models': manifest,
             'audit_script_sha256': digest(__file__), 'runtime': trainer.collect_runtime_provenance(),
             'execution': trainer.collect_execution_provenance(device), 'state_order': STATES,
             'band_definition': 'GT whole foreground, 2 iterations, 6-face connectivity; same as training',
             'surface_definition': '6-face surface voxels; pooled bidirectional distances, 1-mm isotropic',
             'scope': 'One seed, reused 52-case development fold; descriptive bootstrap, no independent-test claim.'})
        predictions, truths, case_names = {}, [], []
        repeatability = {}
        for name, directory in runs.items():
            config = configs[name]
            model = trainer.build_swinunetr(tuple(run['spatial_size']), 3, device, drop_rate=run['drop_rate'], activation_checkpointing=False)
            checkpoint = torch.load(directory / 'checkpoint_best.pt', map_location='cpu', weights_only=True)
            assert json.loads(json.dumps(checkpoint['run'])) == config['run']
            assert checkpoint['epoch'] == manifest[name]['selected_epoch']
            model.load_state_dict(checkpoint['model'], strict=True)
            del checkpoint
            model.eval()
            predictions[name] = []
            with torch.no_grad():
                for index, batch in enumerate(loader):
                    truth = batch['label'][0, 0].numpy().astype(np.uint8)
                    case_name = str(batch['case_name'][0])
                    if name == 'dice':
                        truths.append(truth)
                        case_names.append(case_name)
                    else:
                        assert case_names[index] == case_name and np.array_equal(truths[index], truth)
                    with torch.autocast('cuda', enabled=run['amp']):
                        output = model(batch['image'].to(device))
                    predictions[name].append(output[0].argmax(0).cpu().numpy().astype(np.uint8))
            repeat_counts = {'changed_voxels': 0, 'changed_cases': 0, 'max_changed_voxels_in_case': 0,
                             'fn': 0, 'fp': 0, 'ap_swap': 0}
            with torch.no_grad():
                for index, batch in enumerate(loader):
                    assert str(batch['case_name'][0]) == case_names[index]
                    with torch.autocast('cuda', enabled=run['amp']):
                        output = model(batch['image'].to(device))
                    repeated = output[0].argmax(0).cpu().numpy().astype(np.uint8)
                    changed = int((repeated != predictions[name][index]).sum())
                    repeat_counts['changed_voxels'] += changed
                    repeat_counts['changed_cases'] += int(changed > 0)
                    repeat_counts['max_changed_voxels_in_case'] = max(repeat_counts['max_changed_voxels_in_case'], changed)
                    state_counts = np.bincount(error_states(truths[index], repeated).ravel(), minlength=4)
                    for key, state in [('fn', 1), ('fp', 2), ('ap_swap', 3)]:
                        repeat_counts[key] += int(state_counts[state])
            repeatability[name] = repeat_counts
            print(json.dumps({'inference_complete': name, 'cases': len(predictions[name]),
                              'repeated_inference': repeat_counts}), flush=True)
            del model
            torch.cuda.empty_cache()
        rows = {name: [] for name in runs}
        comparisons = [('dice', 'bands'), ('dice', 'A'), ('dice', 'B'), ('bands', 'A'), ('bands', 'B'), ('A', 'B')]
        pair_rows = {f'{new}_vs_{old}': [] for old, new in comparisons}
        for index, truth in enumerate(truths):
            masks = supports(truth)
            states = {}
            for name in runs:
                row, states[name] = score_case(truth, predictions[name][index], masks)
                row['case_name'] = case_names[index]
                rows[name].append(row)
            for old, new in comparisons:
                pair_rows[f'{new}_vs_{old}'].append({'case_name': case_names[index],
                    'regions': {key: transitions(states[old], states[new], mask) for key, mask in masks.items()}})
        summary = {name: summarize(values) for name, values in rows.items()}
        parity = {}
        for name, result in summary.items():
            parity[name] = {'macro_dice_delta': result['mean_case_metrics']['macro_dice']['mean'] - existing[name]['macro_dice'],
                            'count_deltas': {key: result['pooled_regions']['all'][key] - existing[name][old_key]
                                             for key, old_key in [('fn', 'total_fn'), ('fp', 'total_fp'), ('ap_swap', 'total_ap_swaps')]}}
        save(args.output / 'reproducibility.json', {'repeated_inference': repeatability, 'previous_audit_deltas': parity,
             'note': 'Checkpoints and inputs hash verified. Report observed inference differences rather than silently replacing counts.'})
        paired = {f'{new}_vs_{old}': summarize_pair(rows[old], rows[new], pair_rows[f'{new}_vs_{old}'])
                  for old, new in comparisons}
        save(args.output / 'cases.json', rows)
        save(args.output / 'paired_cases.json', pair_rows)
        save(args.output / 'summary.json', summary)
        save(args.output / 'paired_summary.json', paired)
        save(args.output / 'completion.json', {'status': 'complete', 'job_id': os.getenv('SLURM_JOB_ID'),
             'cases': len(truths), 'models': list(runs), 'training_runs': 0,
             'existing_audit_counts_reproduced': all(v == 0 for item in parity.values() for v in item['count_deltas'].values())})
        print(json.dumps({'status': 'complete', 'output': str(args.output)}), flush=True)
    except BaseException:
        import traceback
        save(args.output / 'failure.json', {'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
