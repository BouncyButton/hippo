"""Train twelve matched native MRI branch objectives from scratch."""
import argparse
import fcntl
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from baselines.swin_unetr.swin_unetr import _load_pkl_dataframe
from scripts.depth_supervision.train import Patients, GeometryCache, autocast, evaluate
from scripts.depth_supervision.losses import combined_loss
from scripts.depth_supervision.utils import atomic_json, digest, rng_state, restore_rng, save_checkpoint, seed_all, stopping_state, write_history
from scripts.tissue_profiles.mri_views import MRIOnlyViewSwin, NativeViewAtlas
from scripts.unwrapped_swin.evaluate import evaluate as evaluate_branch
from .objectives import Targets, components as objective_components, gradient_record, select_weights, terms, TERM_ORDER


def model_and_inputs(root, p, fraction, device, arm):
    if arm != 'native_mri_context':
        raise ValueError('Only the native MRI context branch is authorized')
    return MRIOnlyViewSwin(device, arm), NativeViewAtlas(root, p, fraction)


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


def calibrate(model, patients, cache, device, settings, objective, atlas):
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
            values = objective_components(logits, labels, objective, cache.get(item, device))
            records.append(dict(case_name=item['case_name'], **gradient_record(values, logits)))
        return select_weights(records, objective, settings)
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
    objective = entry['objective']
    targets = Targets(root, p, objective)
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
    config = dict(identity=ident, objective=objective, depth_balancing='depth' in terms(objective),
                  parameters=sum(x.numel() for x in model.parameters()),
                  all_parameters_trainable=all(x.requires_grad for x in model.parameters()),
                  auxiliary_branch=True, arm=arm, input_channels=1, auxiliary_input_channels=1,
                  calibration=calibration, loss_definitions=p.get('loss_definitions'),
                  initialization='Entire SwinUNETR, MRI-only encoder and fusion trained from scratch')
    atomic_json(output/'config.json', config)
    started = time.monotonic()
    for epoch in range(len(history)+1, p['max_epochs']+1):
        epoch_started = time.monotonic()
        if stopping_state(history, **p['early_stopping'])['triggered']:
            break
        prefix = p['calibration']['prefix_epochs']
        scale = 0. if not terms(objective) else min(1., max(0., (epoch-prefix)/p['calibration']['ramp_epochs']))
        if scale > 0 and calibration is None:
            calibration = calibrate(model, train, targets, device, p['calibration'], objective, atlas)
            config['calibration'] = calibration
            atomic_json(output/'config.json', config)
            print(json.dumps(dict(stage='calibrated', run=entry['id'], **calibration)), flush=True)
        coefficients = calibration['coefficients'] if calibration else {}
        model.train()
        totals = []
        for i in torch.randperm(len(train), generator=order_rng).tolist():
            item = train[i]
            optimizer.zero_grad(set_to_none=True)
            with autocast(device):
                logits = forward_item(model, item, device, atlas)
            labels = item['label'][None].to(device)
            active = objective if scale > 0 else 'dice'
            values = objective_components(logits, labels, active, targets.get(item, device) if scale > 0 else {})
            loss = combined_loss(values, coefficients, scale)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward()
            if not torch.stack([torch.isfinite(q.grad).all() for q in model.parameters() if q.grad is not None]).all():
                raise FloatingPointError('Nonfinite gradients')
            optimizer.step()
            updates += 1
            totals.append([float(loss.detach()), float(values['dice'].detach()),
                           *[float(values[k].detach()) if k in values else 0. for k in TERM_ORDER]])
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        metrics = evaluate_model(model, val, device, atlas)
        score = metrics['summary']['macro_dice']
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
        means = np.mean(totals, axis=0)
        row = dict(epoch=epoch, updates=updates, learning_rate=lr, train_total=float(means[0]),
                   train_dice=float(means[1]),
                   **{'train_'+k: float(means[i+2]) for i, k in enumerate(TERM_ORDER)},
                   constraint_scale=scale,
                   **{'coefficient_'+k: coefficients.get(k, 0.) for k in TERM_ORDER},
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
