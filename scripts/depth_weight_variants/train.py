"""Matched SwinUNETR runs with partial class balancing or focal depth weighting."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import fcntl
import inspect
import json
from pathlib import Path
import time

import monai
import numpy as np
from scipy import ndimage
import torch
from monai.networks.nets import SwinUNETR

from baselines.swin_unetr.swin_unetr import _build_monai_dataset_from_pkl, _load_pkl_dataframe
from .geometry import GeometryConfig, build_geometry
from .losses import ARMS, SETTINGS, partial_class_weights, calibrated_coefficient, combined_loss, components
from .utils import atomic_json, digest, restore_rng, rng_state, save_checkpoint, seed_all, stopping_state, write_history

ROOT = Path(__file__).resolve().parents[2]


class Patients:
    def __init__(self, dataframe, names, metadata):
        if len(names) != len(set(names)):
            raise ValueError('Duplicate case IDs')
        subset = dataframe[dataframe.subject_id.astype(str).isin(names)]
        if len(subset) != len(names):
            raise ValueError('Missing/duplicate dataset cases')
        dataset = _build_monai_dataset_from_pkl(subset, 'MSD', 3, spatial_size=(64, 64, 64), do_resize=False)
        items = {}
        for row, item in zip(subset.itertuples(index=False), dataset):
            for key in ('image', 'label'):
                if hasattr(item[key], 'as_tensor'):
                    item[key] = item[key].as_tensor()
                if tuple(item[key].shape) != (1, 64, 64, 64):
                    raise ValueError('Unexpected prepared shape')
            name = item['case_name']
            native = np.asarray(row.label_data)
            prepared = item['label'][0].numpy()
            if not np.isin(native, [0, 1, 2]).all() or not np.isfinite(item['image'].numpy()).all():
                raise ValueError('Invalid image/label data')
            if any(np.count_nonzero(native == c) != np.count_nonzero(prepared == c) for c in (1, 2)):
                raise ValueError('Preprocessing cropped annotated tissue')
            info = metadata[name]
            if list(native.shape) != info['native_shape']:
                raise ValueError('Native geometry metadata does not match dataset')
            if __import__('hashlib').sha256(native.astype(np.uint8).tobytes()).hexdigest() != info['label_array_sha256']:
                raise ValueError('Label voxel order differs from verified native NIfTI')
            item['affine'] = np.asarray(info['model_affine'], dtype=float)
            items[name] = item
        self.items = [items[name] for name in names]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class GeometryCache:
    def __init__(self, directory, config):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = GeometryConfig(**config)

    def get(self, item, fields=False):
        import hashlib
        labels = item['label'][0].numpy().astype(np.uint8)
        identity = hashlib.sha256(labels.tobytes() + item['affine'].tobytes()
                                  + json.dumps(asdict(self.config), sort_keys=True).encode()
                                  + digest(Path(__file__).with_name('geometry.py')).encode()
                                  + digest(Path(__file__).with_name('losses.py')).encode()).hexdigest()
        path = self.directory / f'{item["case_name"]}_{identity[:16]}_{int(fields)}.npz'
        if not path.exists():
            if fields:
                raise ValueError('These two variants do not use SDF fields')
            value = build_geometry(labels, item['affine'], self.config, fields=False)
            value['class_weights'] = partial_class_weights(labels, value)
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, **value)
            temporary.replace(path)
        with np.load(path, allow_pickle=False) as archive:
            return {k: archive[k] for k in archive.files}


def model_factory(device):
    kwargs = dict(in_channels=1, out_channels=3, feature_size=24, use_checkpoint=False, drop_rate=0.)
    if 'img_size' in inspect.signature(SwinUNETR).parameters:
        kwargs['img_size'] = (64, 64, 64)
    return SwinUNETR(**kwargs).to(device)


def autocast(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()


@torch.no_grad()
def evaluate(model, patients, device, *, geometry_cache=None, detailed=False, predictions_path=None):
    model.eval()
    rows = []
    exported = {}
    for item in patients.items:
        with autocast(device):
            logits = model(item['image'][None].to(device))
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite validation output')
        prediction = logits.float().argmax(1)[0].cpu().numpy().astype(np.uint8)
        target = item['label'][0].numpy().astype(np.uint8)
        row = {'case_name': item['case_name']}
        for c, label in ((1, 'anterior'), (2, 'posterior'), (None, 'foreground')):
            g = target > 0 if c is None else target == c
            p = prediction > 0 if c is None else prediction == c
            denominator = int(g.sum() + p.sum())
            row[label + '_dice'] = float(2 * (g & p).sum() / denominator) if denominator else 1.
        row['macro_dice'] = (row['anterior_dice'] + row['posterior_dice']) / 2
        fg_g, fg_p = target > 0, prediction > 0
        row.update(missed=int((fg_g & ~fg_p).sum()), extra=int((fg_p & ~fg_g).sum()),
                   swaps=int(((target != prediction) & fg_g & fg_p).sum()))
        if detailed:
            spacing = np.linalg.norm(item['affine'][:3, :3], axis=0)
            structure = ndimage.generate_binary_structure(3, 1)
            sg = fg_g & ~ndimage.binary_erosion(fg_g, structure=structure)
            sp = fg_p & ~ndimage.binary_erosion(fg_p, structure=structure)
            if sp.any() and sg.any():
                distances = np.r_[ndimage.distance_transform_edt(~sg, sampling=spacing)[sp],
                                   ndimage.distance_transform_edt(~sp, sampling=spacing)[sg]]
                row['surface_distance_mean_mm'] = float(distances.mean())
                row['hd95_mm'] = float(np.quantile(distances, .95))
            else:
                row['surface_distance_mean_mm'] = row['hd95_mm'] = None
            geometry = geometry_cache.get(item, fields=False)
            row['depth_bands'] = []
            union = fg_g | fg_p
            for b in np.unique(geometry['bin_ids'][union]):
                mask = geometry['bin_ids'] == b
                row['depth_bands'].append(dict(signed_bin=int(b), union=int((mask & union).sum()),
                    missed=int((mask & fg_g & ~fg_p).sum()), extra=int((mask & fg_p & ~fg_g).sum()),
                    swaps=int((mask & (target != prediction) & fg_g & fg_p).sum())))
        rows.append(row)
        if predictions_path:
            exported[item['case_name']] = prediction
    summary = {key: float(np.mean([r[key] for r in rows]))
               for key in ('anterior_dice', 'posterior_dice', 'foreground_dice', 'macro_dice')}
    summary.update({key: sum(r[key] for r in rows) for key in ('missed', 'extra', 'swaps')})
    if detailed:
        for key in ('surface_distance_mean_mm', 'hd95_mm'):
            valid = [r[key] for r in rows if r[key] is not None]
            summary[key] = float(np.mean(valid)) if valid else None
        summary['undefined_surface_cases'] = sum(r['hd95_mm'] is None for r in rows)
    if predictions_path:
        with Path(predictions_path).open('wb') as stream:
            np.savez_compressed(stream, **exported)
    return {'summary': summary, 'cases': rows}


def calibrate(model, patients, cache, device, arm, protocol):
    state = rng_state()
    model.eval()
    records = []
    try:
        indices = np.random.default_rng(0).choice(len(patients), min(32, len(patients)), replace=False)
        for i in sorted(indices):
            item = patients[int(i)]
            with torch.no_grad(), autocast(device):
                output = model(item['image'][None].to(device))
            logits = output.detach().float().requires_grad_(True)
            values = components(logits, item['label'][None].to(device), cache.get(item), arm)
            norms = {}
            predicted = logits.detach().argmax(1, keepdim=True)
            truth = item['label'][None].to(device).long()
            wrong = (predicted != truth)[:, 0]
            for key, value in values.items():
                gradient = torch.autograd.grad(value, logits, retain_graph=True)[0]
                norms[key] = float(gradient.square().mean().sqrt())
                if wrong.any():
                    direction = (gradient.gather(1, truth) - gradient.gather(1, predicted))[:, 0]
                    norms[key + '_helpful_error_fraction'] = float((direction[wrong] < 0).float().mean())
            records.append({'case_name': item['case_name'], **norms})
        terms = {name: calibrated_coefficient([r['dice'] for r in records], [r[name] for r in records],
                                               protocol['calibration']['target_ratio'],
                                               protocol['calibration']['cap_ratio'])
                 for name in values if name != 'dice'}
        return {'cases': records, 'terms': terms,
                'coefficients': {key: value['coefficient'] for key, value in terms.items()},
                'checkpoint_epoch': protocol['calibration']['prefix_epochs'],
                'basis': 'logit-gradient RMS relative to Dice; training partition only'}
    finally:
        restore_rng(state)


def verify_protocol(protocol, root):
    if protocol['fold'] != 0 or protocol['seed'] != 0 or protocol['max_epochs'] != 75:
        raise ValueError('This campaign is fixed to fold 0, seed 0, maximum 75 epochs')
    if protocol['arms'] != list(ARMS) or protocol['baseline_loss'] != 'dice':
        raise ValueError('Unexpected objectives')
    if protocol['weight_variants'] != SETTINGS:
        raise ValueError('Weight variant settings differ from frozen implementation')
    for name, sha in protocol['source_sha256'].items():
        if digest(root / name) != sha:
            raise ValueError('Frozen source changed: ' + name)


def run(protocol, fraction, arm, pkl, output, cache_dir, device, *, resume=True,
        stop_after_epoch=None, test_model_factory=None, test_patients=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config_sha = __import__('hashlib').sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    identity = dict(protocol_sha256=config_sha, fraction=fraction, arm=arm,
                    train_cases=protocol['splits'][fraction]['train'], val_cases=protocol['splits'][fraction]['val'])
    if arm not in ARMS or set(identity['train_cases']) & set(identity['val_cases']):
        raise ValueError('Invalid arm or overlapping training/validation')
    config_path = output / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text())['identity'] != identity:
        raise ValueError('Resume protocol or run identity changed')
    completion = output / 'completion.json'
    if completion.exists():
        record = json.loads(completion.read_text())
        if record['identity'] != identity:
            raise ValueError('Completed run identity mismatch')
        for name, sha in record['artifacts'].items():
            if digest(output / name) != sha:
                raise ValueError('Completed run artifact changed: ' + name)
        return record
    with (output / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        seed_all(protocol['seed'])
        if test_patients is None:
            if digest(pkl) != protocol['pkl_sha256']:
                raise ValueError('Dataset hash mismatch')
            dataframe = _load_pkl_dataframe(pkl)
            train = Patients(dataframe, identity['train_cases'], protocol['native_geometry'])
            val = Patients(dataframe, identity['val_cases'], protocol['native_geometry'])
        else:
            train, val = test_patients
        device = torch.device(device)
        if device.type == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
            raise RuntimeError('CUDA bfloat16 support required')
        model = (test_model_factory or model_factory)(device)
        parameter_count = sum(p.numel() for p in model.parameters())
        if test_model_factory is None and parameter_count != protocol['model_parameters']:
            raise ValueError('SwinUNETR parameter count changed')
        optimizer = torch.optim.AdamW(model.parameters(), lr=protocol['learning_rate'],
                                       weight_decay=protocol['weight_decay'], foreach=False)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=protocol['step_size'],
                                                    gamma=protocol['lr_gamma'])
        order_rng = torch.Generator().manual_seed(protocol['seed'])
        cache = GeometryCache(cache_dir, protocol['geometry'])
        atomic_json(config_path, {'identity': identity, 'protocol': protocol,
                                 'runtime': dict(torch=torch.__version__, monai=monai.__version__,
                                                 parameters=parameter_count, device=str(device),
                                                 gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None)})
        history, coefficients, calibration = [], {}, None
        best_score, best_epoch, updates = -1., 0, 0
        latest = output / 'checkpoint_latest.pt'
        best_path = output / 'checkpoint_best.pt'
        if latest.exists():
            if not resume:
                raise ValueError('Existing run requires resume')
            state = torch.load(latest, map_location=device, weights_only=False)
            if state['identity'] != identity:
                raise ValueError('Checkpoint identity changed')
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            history, coefficients = state['history'], state['coefficients']
            calibration = state['calibration']
            best_score, best_epoch, updates = state['best_score'], state['best_epoch'], state['updates']
            order_rng.set_state(state['order_rng'].cpu())
            restore_rng(state['rng'])
            if best_epoch == len(history) and (not best_path.exists() or
                    torch.load(best_path, map_location='cpu', weights_only=False)['epoch'] != best_epoch):
                save_checkpoint(best_path, dict(model=model.state_dict(), epoch=best_epoch, identity=identity))
        write_history(output / 'metrics.csv', history)
        start = time.monotonic()
        for epoch in range(len(history) + 1, protocol['max_epochs'] + 1):
            if stopping_state(history, **protocol['early_stopping'])['triggered']:
                break
            prefix = protocol['calibration']['prefix_epochs']
            if arm != 'dice' and epoch > prefix and calibration is None:
                calibration = calibrate(model, train, cache, device, arm, protocol)
                coefficients = calibration['coefficients']
                atomic_json(output / 'calibration.json', calibration)
            scale = min(1., max(0., (epoch - prefix) / protocol['calibration']['ramp_epochs']))
            model.train()
            totals = dict(dice=0., depth=0., total=0.)
            for i in torch.randperm(len(train), generator=order_rng).tolist():
                item = train[i]
                optimizer.zero_grad(set_to_none=True)
                with autocast(device):
                    logits = model(item['image'][None].to(device))
                active_arm = arm if scale > 0 else 'dice'
                geometry = cache.get(item) if active_arm != 'dice' else None
                values = components(logits, item['label'][None].to(device), geometry, active_arm)
                loss = combined_loss(values, coefficients, scale)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                finite_gradients = torch.stack([torch.isfinite(p.grad).all() for p in model.parameters()
                                                if p.grad is not None]).all()
                if not finite_gradients:
                    raise FloatingPointError('Nonfinite model gradient')
                optimizer.step()
                updates += 1
                for key, value in values.items():
                    totals[key] += float(value.detach())
                totals['total'] += float(loss.detach())
            lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            metrics = evaluate(model, val, device)
            score = metrics['summary']['macro_dice']
            improved = score > best_score
            if improved:
                best_score, best_epoch = score, epoch
            row = dict(epoch=epoch, updates=updates, learning_rate=lr, auxiliary_scale=scale if arm != 'dice' else 0.,
                       **{'train_' + k: v / len(train) for k, v in totals.items()},
                       validation_dice=score, validation_foreground_dice=metrics['summary']['foreground_dice'],
                       validation_missed=metrics['summary']['missed'], validation_extra=metrics['summary']['extra'],
                       validation_swaps=metrics['summary']['swaps'], elapsed_seconds=time.monotonic() - start)
            history.append(row)
            state = dict(identity=identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict(), history=history, coefficients=coefficients,
                         calibration=calibration, best_score=best_score, best_epoch=best_epoch,
                         updates=updates, order_rng=order_rng.get_state(), rng=rng_state())
            # Latest first: a crash before publishing best can be repaired exactly.
            save_checkpoint(latest, state)
            if improved:
                save_checkpoint(best_path, dict(model=model.state_dict(), epoch=epoch, identity=identity))
                atomic_json(output / 'validation_best.json', metrics)
            atomic_json(output / 'validation_latest.json', metrics)
            write_history(output / 'metrics.csv', history)
            print(json.dumps(dict(fraction=fraction, arm=arm, **row)), flush=True)
            if stop_after_epoch is not None and epoch >= stop_after_epoch:
                return {'interrupted_for_test': True, 'epoch': epoch}
        selected = torch.load(best_path, map_location=device, weights_only=False)
        if selected['epoch'] != best_epoch:
            raise ValueError('Best checkpoint chronology is inconsistent')
        model.load_state_dict(selected['model'])
        final = evaluate(model, val, device, geometry_cache=cache, detailed=True,
                         predictions_path=output / 'validation_predictions.npz')
        atomic_json(output / 'validation_best.json', final)
        record = dict(identity=identity, completed_epochs=len(history), best_epoch=best_epoch,
                      optimizer_updates=updates, early_stopping=stopping_state(history, **protocol['early_stopping']),
                      coefficients=coefficients, validation=final['summary'], artifacts={})
        artifact_names = ['config.json', 'metrics.csv', 'checkpoint_best.pt', 'checkpoint_latest.pt',
                          'validation_best.json', 'validation_latest.json', 'validation_predictions.npz']
        if calibration:
            atomic_json(output / 'calibration.json', calibration)
            artifact_names.append('calibration.json')
        record['artifacts'] = {name: digest(output / name) for name in artifact_names}
        atomic_json(completion, record)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--pkl', type=Path, required=True)
    parser.add_argument('--fraction', choices=['full', '5pct'], required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    protocol = json.loads(args.protocol.read_text())
    verify_protocol(protocol, ROOT)
    if monai.__version__ != protocol['required_monai']:
        raise RuntimeError('Production MONAI version differs from frozen protocol')
    run(protocol, args.fraction, args.arm, args.pkl, args.output, args.cache, args.device)


if __name__ == '__main__':
    main()
