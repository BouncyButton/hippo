"""Train native SwinUNETR with Dice plus one cumulative tissue-profile loss."""
import argparse
import fcntl
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from baselines.swin_unetr.swin_unetr import _load_pkl_dataframe
from scripts.depth_supervision.train import Patients, GeometryCache, autocast, model_factory, evaluate
from scripts.depth_supervision.losses import components, calibrated_coefficient, combined_loss
from scripts.depth_supervision.utils import atomic_json, digest, rng_state, restore_rng, save_checkpoint, seed_all, stopping_state, write_history
from .profiles import ProfileCache, profile_loss
from .multiscale import MultiscaleCache, multiscale_loss
from .views import DualViewSwin, FrozenAtlas, evaluate_branch
from .informed import InformedSwin, NativePlaneCache
from .mri_views import MRIOnlyViewSwin, MRI_ARMS, NativeViewAtlas


def profile_implementation(variant):
    if variant == 'none':
        return None, None
    if variant == 'pooled':
        return ProfileCache, profile_loss
    if variant == 'multiscale':
        return MultiscaleCache, multiscale_loss
    raise ValueError('Unknown profile variant: '+variant)


def model_and_inputs(root, p, fraction, device, arm):
    if arm == 'native':
        return model_factory(device), None
    if arm == 'informed':
        return InformedSwin(device), NativePlaneCache(root, p, fraction)
    if arm in MRI_ARMS:
        cache = NativeViewAtlas if arm.startswith('native_mri_') else FrozenAtlas
        return MRIOnlyViewSwin(device, arm), cache(root, p, fraction)
    return DualViewSwin(device, arm), FrozenAtlas(root, p, fraction)


def identity(p, entry):
    return dict(protocol_sha256=p['identity_sha'], run=entry, **p['splits'][entry['fraction']])


def completed_record(output, ident):
    path = Path(output)/'completion.json'
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record['identity'] != ident:
        raise ValueError('Completed run identity mismatch')
    for name, expected in record['artifacts'].items():
        if digest(Path(output)/name) != expected:
            raise ValueError('Completed artifact changed: '+name)
    return record


def forward_item(model, item, device, atlas=None):
    image = item['image'][None].to(device)
    return model(image, atlas.get(item)) if atlas is not None else model(image)


def evaluate_model(model, patients, device, atlas=None, **kwargs):
    if atlas is not None:
        return evaluate_branch(model, patients, device, atlas_cache=atlas, **kwargs)
    return evaluate(model, patients, device, **kwargs)


def calibrate(model, patients, cache, device, settings, loss_function=profile_loss, atlas=None):
    state = rng_state()
    was_training = model.training
    model.eval()
    records = []
    try:
        indices = np.random.default_rng(0).choice(len(patients), min(settings['max_cases'], len(patients)), replace=False)
        for i in sorted(indices):
            item = patients[int(i)]
            with torch.no_grad(), autocast(device):
                output = forward_item(model, item, device, atlas)
            logits = output.detach().float().requires_grad_(True)
            labels = item['label'][None].to(device)
            values = components(logits, labels, None, 'dice')
            values['profile'] = loss_function(logits, labels, cache.get(item))
            norms = {key: float(torch.autograd.grad(value, logits, retain_graph=True)[0].square().mean().sqrt())
                     for key, value in values.items()}
            records.append(dict(case_name=item['case_name'], **norms))
        result = calibrated_coefficient([r['dice'] for r in records], [r['profile'] for r in records],
                                        settings['target_ratio'], settings['cap_ratio'])
        return dict(cases=records, statistics=result, coefficient=result['coefficient'],
                    checkpoint_epoch=settings['prefix_epochs'], basis='training-only logit-gradient RMS')
    finally:
        restore_rng(state)
        model.train(was_training)


def run(p, root, entry, *, stop_after=None, device_override=None):
    root = Path(root)
    output = root/'runs'/entry['id']
    output.mkdir(parents=True, exist_ok=True)
    with (output/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(p, root, entry, output, stop_after, device_override)


def _run(p, root, entry, output, stop_after, device_override):
    ident = identity(p, entry)
    existing = completed_record(output, ident)
    if existing is not None:
        return existing
    seed_all(entry['seed'])
    split = p['splits'][entry['fraction']]
    if set(split['train']) & set(split['val']):
        raise ValueError('Train/validation overlap')
    dataframe = _load_pkl_dataframe(p['pkl'])
    train, val = [Patients(dataframe, split[k], p['native_geometry']) for k in ('train', 'val')]
    del dataframe
    device = torch.device(device_override or 'cuda')
    arm = entry.get('arm', 'native')
    model, atlas = model_and_inputs(root, p, entry['fraction'], device, arm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'], foreach=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=p['step_size'], gamma=p['lr_gamma'])
    order_rng = torch.Generator().manual_seed(entry['seed'])
    variant = entry.get('profile_variant', 'pooled')
    cache_factory, loss_function = profile_implementation(variant)
    profiles = cache_factory(root, p, entry['fraction']) if cache_factory else None
    diagnostic_cache = GeometryCache(root/'diagnostic_geometry', p['geometry'])
    history, best_score, best_epoch, updates, calibration = [], -1., 0, 0, None
    latest, best = output/'latest.pt', output/'best.pt'
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        if state['identity'] != ident:
            raise ValueError('Resume identity changed')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        history, best_score, best_epoch, updates = state['history'], state['best_score'], state['best_epoch'], state['updates']
        calibration = state['calibration']
        order_rng.set_state(state['order_rng'].cpu())
        restore_rng(state['rng'])
        del state
        if best_epoch == len(history):
            save_checkpoint(best, dict(identity=ident, model=model.state_dict(), epoch=best_epoch))
        print(json.dumps(dict(stage='resumed', run=entry['id'], epoch=len(history))), flush=True)
    config = dict(identity=ident, objective='Dice only' if variant == 'none' else 'Dice + '+variant+' tissue profile', depth_balancing=False,
                  parameters=sum(x.numel() for x in model.parameters()),
                  all_parameters_trainable=all(x.requires_grad for x in model.parameters()),
                  auxiliary_branch=arm not in ('native', 'informed'), arm=arm, profile_variant=variant,
                  input_channels=4 if arm == 'informed' else 1,
                  auxiliary_input_channels=(1 if arm in MRI_ARMS else 4) if arm not in ('native', 'informed') else 0,
                  input_policy=p.get('informed_input') if arm == 'informed' else None,
                  profile_geometry=(None if variant == 'none' else
                      p['multiscale_geometry'] if variant == 'multiscale' else p['profile_geometry']),
                  calibration=calibration)
    atomic_json(output/'config.json', config)
    started = time.monotonic()
    for epoch in range(len(history)+1, p['max_epochs']+1):
        epoch_started = time.monotonic()
        if stopping_state(history, **p['early_stopping'])['triggered']:
            break
        prefix = p['calibration']['prefix_epochs']
        scale = 0. if variant == 'none' else min(1., max(0., (epoch-prefix)/p['calibration']['ramp_epochs']))
        if scale > 0 and calibration is None:
            calibration = calibrate(model, train, profiles, device, p['calibration'], loss_function, atlas)
            config['calibration'] = calibration
            atomic_json(output/'config.json', config)
            print(json.dumps(dict(stage='calibrated', run=entry['id'], **calibration)), flush=True)
        coefficient = calibration['coefficient'] if calibration else 0.
        model.train()
        totals = []
        for i in torch.randperm(len(train), generator=order_rng).tolist():
            item = train[i]
            optimizer.zero_grad(set_to_none=True)
            with autocast(device):
                logits = forward_item(model, item, device, atlas)
            labels = item['label'][None].to(device)
            values = components(logits, labels, None, 'dice')
            if scale > 0:
                values['profile'] = loss_function(logits, labels, profiles.get(item))
            loss = combined_loss(values, {'profile': coefficient}, scale)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward()
            if not torch.stack([torch.isfinite(q.grad).all() for q in model.parameters() if q.grad is not None]).all():
                raise FloatingPointError('Nonfinite gradients')
            optimizer.step()
            updates += 1
            totals.append([float(loss.detach()), float(values['dice'].detach()),
                           float(values['profile'].detach()) if 'profile' in values else 0.])
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        metrics = evaluate_model(model, val, device, atlas)
        score = metrics['summary']['macro_dice']
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
        means = np.mean(totals, axis=0)
        row = dict(epoch=epoch, updates=updates, learning_rate=lr, train_total=float(means[0]),
                   train_dice=float(means[1]), train_profile=float(means[2]),
                   profile_scale=scale, profile_coefficient=coefficient,
                   epoch_seconds=time.monotonic()-epoch_started,
                   elapsed_seconds=time.monotonic()-started, validation_dice=score,
                   **{'validation_'+k: metrics['summary'][k] for k in ('foreground_dice', 'missed', 'extra', 'swaps')})
        history.append(row)
        save_checkpoint(latest, dict(identity=ident, model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), history=history, best_score=best_score, best_epoch=best_epoch,
            updates=updates, calibration=calibration, order_rng=order_rng.get_state(), rng=rng_state()))
        if improved:
            save_checkpoint(best, dict(identity=ident, model=model.state_dict(), epoch=best_epoch))
            atomic_json(output/'validation_best.json', metrics)
        write_history(output/'metrics.csv', history)
        print(json.dumps(dict(run=entry['id'], **row)), flush=True)
        if stop_after and epoch >= stop_after:
            return
    selected = torch.load(best, map_location=device, weights_only=False)
    if selected['identity'] != ident or selected['epoch'] != best_epoch:
        raise ValueError('Selected checkpoint mismatch')
    model.load_state_dict(selected['model'])
    del selected
    final = evaluate_model(model, val, device, atlas, geometry_cache=diagnostic_cache, detailed=True,
                     predictions_path=output/'validation_predictions.npz')
    atomic_json(output/'validation_best.json', final)
    files = ['best.pt', 'metrics.csv', 'config.json', 'validation_best.json', 'validation_predictions.npz']
    record = dict(identity=ident, completed_epochs=len(history), best_epoch=best_epoch, optimizer_updates=updates,
                  parameter_count=config['parameters'], training_seconds=sum(r['epoch_seconds'] for r in history),
                  early_stopping=stopping_state(history, **p['early_stopping']), validation=final['summary'],
                  artifacts={n: digest(output/n) for n in files})
    atomic_json(output/'completion.json', record)
    latest.unlink(missing_ok=True)
    del model, optimizer, train, val
    gc.collect()
    torch.cuda.empty_cache()
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    p = json.loads((args.root/'protocol.json').read_text())
    entry = next(e for e in p['runs'] if e['id'] == args.run_id)
    run(p, args.root, entry)


if __name__ == '__main__':
    main()
