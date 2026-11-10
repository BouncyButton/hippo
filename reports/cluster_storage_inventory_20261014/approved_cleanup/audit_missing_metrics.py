"""Preserve missing voxel-error diagnostics before approved model retirement."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def measures(truth, prediction):
    truth = np.asarray(truth, dtype=np.uint8)
    prediction = np.asarray(prediction, dtype=np.uint8)
    assert truth.shape == prediction.shape
    assert np.isin(truth, [0, 1, 2]).all() and np.isin(prediction, [0, 1, 2]).all()
    cm = np.bincount((truth * 3 + prediction).ravel(), minlength=9).reshape(3, 3)
    dice = [float(2 * cm[c, c] / (cm[c].sum() + cm[:, c].sum())) for c in (1, 2)]
    return dict(anterior_hard_dice=dice[0], posterior_hard_dice=dice[1],
                macro_hard_dice=float(np.mean(dice)),
                foreground_false_negative=int(cm[1, 0] + cm[2, 0]),
                foreground_false_positive=int(cm[0, 1] + cm[0, 2]),
                anterior_to_posterior=int(cm[1, 2]), posterior_to_anterior=int(cm[2, 1]),
                ap_swaps=int(cm[1, 2] + cm[2, 1]), confusion_matrix=cm.tolist())


def atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pkl', type=Path, required=True)
    args = p.parse_args()
    # Check count definitions with one of every directed error and one correct voxel/class.
    sample = measures(np.repeat([0, 1, 2], 3), np.tile([0, 1, 2], 3))
    assert sample['foreground_false_negative'] == sample['foreground_false_positive'] == sample['ap_swaps'] == 2
    assert sample['anterior_hard_dice'] == sample['posterior_hard_dice'] == 1/3
    source = args.root / 'source'
    sys.path[:0] = [str(source), str(source / 'scripts'), str(source / 'baselines/nnUNet')]
    import train_compute_matched_baselines as base
    torch.set_num_threads(4)
    base.seed_everything(0)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    frame = base._load_pkl_dataframe(args.pkl)
    dataset_sha = sha(args.pkl)
    patients = {}
    outputs = []
    capped = args.root.name.startswith('capped_depth_delay_')
    configurations = sorted(p for p in args.root.rglob('config.json') if 'source' not in p.relative_to(args.root).parts)
    expected = 44 if capped else 3
    assert len(configurations) == expected, (len(configurations), expected)
    for config_path in configurations:
        run = config_path.parent
        config = json.loads(config_path.read_text())
        assert config['dataset_sha256'] == dataset_sha
        names = config['validation_cases']
        key = tuple(names)
        if key not in patients:
            patients[key] = base.PreparedPatients(frame, names)
        identity = config.get('model_identity', config['model'])
        if capped:
            protocol = source / 'experiments/capped_depth_delay_31976_20261006' / identity
        else:
            campaign = 'budget_bands_94603_20261005' if 'budget_bands_' in args.root.name else 'budget_baselines_94603_20261005'
            protocol = source / 'experiments' / campaign / identity
        architecture = json.loads((protocol / 'nnunet_architecture.json').read_text())
        frozen_protocol = json.loads((protocol / 'protocol.json').read_text())
        if not capped and config['model'] == 'unet' and frozen_protocol.get('unet_output_head') == 'linear':
            factory_path = source / 'baselines/residual_unet.py'
            assert sha(factory_path) == frozen_protocol['unet_factory_sha256']
            from baselines.residual_unet import build_unet48
            model = build_unet48()
            assert sum(p.numel() for p in model.parameters()) == 43234201
        elif identity in ('unetr_vit_b16', 'basic_unet16'):
            from baselines.simple_segmentation import build_candidate
            model = build_candidate(identity)
        else:
            model = base.build_model(config['model'], architecture)
        model = model.cuda().eval()
        endpoints = [('best', 'checkpoint_best_model.pt' if capped else 'checkpoint_best.pt', 'validation_best.json'),
                     ('final', 'checkpoint_final_model.pt' if capped else 'checkpoint_latest.pt', 'validation_latest.json')]
        for endpoint, checkpoint_name, validation_name in endpoints:
            out = args.output / args.root.name / run.relative_to(args.root) / endpoint
            output_json = out / 'voxel_metrics.json'
            if output_json.exists():
                saved = json.loads(output_json.read_text())
                assert saved['status'] == 'verified' and sha(out / 'validation_predictions.npz') == saved['prediction_sha256']
                outputs.append(str(output_json.relative_to(args.output)))
                continue
            checkpoint = run / checkpoint_name
            payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if 'model_tensor_sha256' in payload:
                tensor_hash = hashlib.sha256()
                for tensor_name, value in payload['model'].items():
                    tensor_hash.update(tensor_name.encode())
                    tensor_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
                assert tensor_hash.hexdigest() == payload['model_tensor_sha256'], 'Original model tensor hash differs'
            model.load_state_dict(payload['model'], strict=True)
            reference = json.loads((run / validation_name).read_text())
            reference_cases = {r['case_name']: r for r in reference['cases']}
            assert set(reference_cases) == set(names)
            rows = []; predictions = {}; max_delta = 0.
            with torch.no_grad():
                for batch in base.DataLoader(patients[key], batch_size=1, shuffle=False, num_workers=0):
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        logits = model(batch['image'].cuda())
                    if isinstance(logits, (list, tuple)):
                        logits = logits[0]
                    assert torch.isfinite(logits).all()
                    prediction = logits.float().softmax(1).argmax(1)[0].cpu().numpy().astype(np.uint8)
                    row = measures(batch['label'][0, 0].numpy(), prediction)
                    name = batch['case_name'][0]
                    row['case_name'] = name
                    for metric in ('anterior_hard_dice', 'posterior_hard_dice'):
                        max_delta = max(max_delta, abs(row[metric] - reference_cases[name][metric]))
                    rows.append(row); predictions[name] = prediction
            if max_delta >= 2e-5:
                atomic(out / 'reproduction_mismatch.json', dict(checkpoint=str(checkpoint),maximum_case_dice_difference=max_delta,cases=rows,reference=reference))
                raise RuntimeError(('Archived Dice reproduction failed', checkpoint, max_delta))
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / 'validation_predictions.npz', **predictions)
            summary = {m:float(np.mean([r[m] for r in rows])) for m in ('anterior_hard_dice','posterior_hard_dice','macro_hard_dice')}
            summary.update({m:sum(r[m] for r in rows) for m in ('foreground_false_negative','foreground_false_positive','anterior_to_posterior','posterior_to_anterior','ap_swaps')})
            record = dict(status='verified', endpoint=endpoint, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                          config_sha256=sha(config_path), dataset_sha256=dataset_sha, evaluation_source_sha256=sha(__file__),
                          model_factory_source_sha256=sha(source / 'scripts/train_compute_matched_baselines.py'),
                          architecture_sha256=sha(protocol / 'nnunet_architecture.json'), maximum_case_dice_difference=max_delta,
                          original_validation_sha256=sha(run / validation_name), precision='CUDA bfloat16 autocast; original frozen model/preprocessing',
                          fold=config['fold'], seed=config['seed'], cases=rows, summary=summary,
                          torch_version=torch.__version__,cuda_version=torch.version.cuda,gpu_name=torch.cuda.get_device_name(0),node=os.environ.get('SLURMD_NODENAME'),
                          prediction_sha256=sha(out / 'validation_predictions.npz'))
            atomic(output_json, record)
            outputs.append(str(output_json.relative_to(args.output)))
            atomic(args.output / args.root.name / 'progress.json', dict(status='running', endpoints=len(outputs), expected_endpoints=2*expected))
            print(json.dumps(dict(experiment=args.root.name,run=str(run.relative_to(args.root)),endpoint=endpoint,summary=summary,max_dice_difference=max_delta)),flush=True)
            del payload, predictions
        del model
        gc.collect();torch.cuda.empty_cache()
    assert len(outputs) == expected*2
    atomic(args.output / args.root.name / 'completion.json', dict(status='verified',endpoints=len(outputs),outputs=outputs))


if __name__ == '__main__':
    main()
