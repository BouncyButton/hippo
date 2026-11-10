"""Matched SwinUNETR arms and out-of-fold guide preparation in one campaign."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from baselines.swin_unetr.swin_unetr import _load_pkl_dataframe
from scripts.depth_supervision.train import Patients, GeometryCache, autocast
from scripts.depth_supervision.losses import components, combined_loss
from scripts.depth_supervision.utils import atomic_json, digest, rng_state, restore_rng, save_checkpoint, seed_all, stopping_state, write_history
from .models import ARMS, DualViewSwin
from .evaluate import evaluate
from .cache import AtlasCache, prepare


def run_identity(p, fraction, arm, guide_fold=None):
    split = p['guide_splits'][fraction][guide_fold] if guide_fold is not None else p['splits'][fraction]
    # Geometry repairs change branch inputs, not completed native/guide training.
    protocol_sha = p.get('native_training_identity_sha', p['identity_sha']) if not ARMS[arm] else p['identity_sha']
    return dict(protocol_sha256=protocol_sha, fraction=fraction, arm=arm, guide_fold=guide_fold,
                train=split['train'], val=split['predict' if guide_fold is not None else 'val'])


def completed_record(output, identity):
    done = output / 'completion.json'
    if not done.exists():
        return None
    record = json.loads(done.read_text())
    if record['identity'] != identity:
        raise ValueError('Completion identity changed')
    for name, expected in record['artifacts'].items():
        if digest(output / name) != expected:
            raise ValueError('Completed artifact changed: ' + name)
    return record


def run(p, root, fraction, arm, *, guide_fold=None, stop_after=None, device_override=None):
    is_guide = guide_fold is not None
    if is_guide:
        division = p['guide_splits'][fraction][guide_fold]
        train_names, val_names = division['train'], division['predict']
        output = root / 'guides' / fraction / str(guide_fold)
        epochs, objective = p['guide_epochs'], 'dice'
    else:
        train_names, val_names = p['splits'][fraction]['train'], p['splits'][fraction]['val']
        output = root / 'runs' / fraction / arm
        epochs, objective = p['max_epochs'], 'dice' if arm == 'dice' else 'depth'
    output.mkdir(parents=True, exist_ok=True)
    identity = run_identity(p, fraction, arm, guide_fold)
    done = output / 'completion.json'
    record = completed_record(output, identity)
    if record is not None:
        return record
    seed_all(p['seed'])
    dataframe = _load_pkl_dataframe(p['pkl'])
    train = Patients(dataframe, train_names, p['native_geometry'])
    val = Patients(dataframe, val_names, p['native_geometry'])
    del dataframe
    device = torch.device(device_override or 'cuda')
    model = DualViewSwin(device, 'dice' if is_guide else arm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'], foreach=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=p['step_size'], gamma=p['lr_gamma'])
    order_rng = torch.Generator().manual_seed(p['seed'])
    cache = GeometryCache(root / 'loss_geometry', p['geometry'])
    atlas = AtlasCache(root / 'atlas' / fraction, p) if ARMS[arm] and not is_guide else None
    coefficient = {'depth': p['fixed_depth_coefficients'][fraction]}
    history, best_score, best_epoch, updates = [], -1., 0, 0
    latest, best = output / 'latest.pt', output / 'best.pt'
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        if state['identity'] != identity:
            raise ValueError('Resume identity changed')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        history, best_score, best_epoch, updates = state['history'], state['best_score'], state['best_epoch'], state['updates']
        order_rng.set_state(state['order_rng'].cpu()); restore_rng(state['rng'])
        del state
        # Recover an interrupted latest -> best checkpoint publication.
        if best_epoch == len(history):
            save_checkpoint(best, dict(identity=identity, model=model.state_dict(), epoch=best_epoch))
        print(json.dumps(dict(stage='checkpoint_resumed', fraction=fraction, arm=arm,
                              completed_epochs=len(history), optimizer_updates=updates)), flush=True)
    atomic_json(output / 'config.json', dict(identity=identity, settings=ARMS[arm], loss=objective,
                depth_coefficient=coefficient['depth'], parameters=sum(x.numel() for x in model.parameters()),
                guide_heldout_labels_used_for_selection=False))
    start = time.monotonic()
    for epoch in range(len(history) + 1, epochs + 1):
        if not is_guide and stopping_state(history, **p['early_stopping'])['triggered']:
            break
        model.train()
        totals, grad_norm = [], 0.
        scale = min(1., max(0., (epoch - 5) / 5)) if objective == 'depth' else 0.
        for i in torch.randperm(len(train), generator=order_rng).tolist():
            item = train[i]
            inputs = atlas.get(item) if atlas else None
            optimizer.zero_grad(set_to_none=True)
            with autocast(device):
                logits = model(item['image'][None].to(device), inputs)
            active = objective if scale > 0 else 'dice'
            values = components(logits, item['label'][None].to(device), cache.get(item, fields=False) if active == 'depth' else None, active)
            loss = combined_loss(values, coefficient, scale)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward()
            if not torch.stack([torch.isfinite(q.grad).all() for q in model.parameters() if q.grad is not None]).all():
                raise FloatingPointError('Nonfinite gradients')
            if model.encoder is not None:
                grad_norm += float(torch.stack([q.grad.detach().square().sum() for q in model.encoder.parameters() if q.grad is not None]).sum())
            optimizer.step(); updates += 1; totals.append(float(loss.detach()))
        lr = optimizer.param_groups[0]['lr']; scheduler.step()
        if is_guide:
            # Fixed epoch count; held-out labels never select guide weights.
            score = float(epoch)
            metrics = {}
        else:
            metrics = evaluate(model, val, device, atlas_cache=atlas)
            score = metrics['summary']['macro_dice']
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
        row = dict(epoch=epoch, updates=updates, learning_rate=lr, train_total=float(np.mean(totals)),
                   auxiliary_scale=scale, auxiliary_gradient_square_mean=grad_norm / len(train),
                   elapsed_seconds=time.monotonic() - start)
        if not is_guide:
            row.update(validation_dice=score, **{'validation_' + key: metrics['summary'][key] for key in ('foreground_dice', 'missed', 'extra', 'swaps')})
        history.append(row)
        if not is_guide and arm == 'dice' and epoch == p['guide_epochs']:
            evaluate(model, val, device, predictions_path=output / 'fixed_epoch_guide_predictions.npz')
        save_checkpoint(latest, dict(identity=identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), history=history, best_score=best_score, best_epoch=best_epoch,
            updates=updates, order_rng=order_rng.get_state(), rng=rng_state()))
        if improved:
            save_checkpoint(best, dict(identity=identity, model=model.state_dict(), epoch=best_epoch))
            if not is_guide:
                atomic_json(output / 'validation_best.json', metrics)
        write_history(output / 'metrics.csv', history)
        print(json.dumps(dict(fraction=fraction, arm=arm, guide_fold=guide_fold, **row)), flush=True)
        if stop_after and epoch >= stop_after:
            return
    selected = torch.load(best, map_location=device, weights_only=False)
    if selected['identity'] != identity or selected['epoch'] != best_epoch:
        raise ValueError('Best checkpoint does not match selected epoch')
    model.load_state_dict(selected['model']); del selected
    if is_guide:
        # Metrics are not computed for held-out guide cases; only predictions are produced.
        predictions = {}
        model.eval()
        with torch.no_grad():
            for item in val.items:
                with autocast(device):
                    logits = model(item['image'][None].to(device))
                predictions[item['case_name']] = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        np.savez_compressed(output / 'predictions.npz', **predictions)
        final = None
    else:
        final = evaluate(model, val, device, geometry_cache=cache, atlas_cache=atlas, detailed=True,
                         predictions_path=output / 'validation_predictions.npz')
        atomic_json(output / 'validation_best.json', final)
    # Finished runs keep best weights and metrics; discard only this run's restart state.
    # A crash before completion publication simply resumes from latest.pt.
    files = ['best.pt', 'metrics.csv', 'config.json', 'predictions.npz' if is_guide else 'validation_predictions.npz']
    if not is_guide:
        files.append('validation_best.json')
        if arm == 'dice': files.append('fixed_epoch_guide_predictions.npz')
    record = dict(identity=identity, completed_epochs=len(history), best_epoch=best_epoch, optimizer_updates=updates,
                  validation=final['summary'] if final else None, artifacts={n: digest(output / n) for n in files})
    atomic_json(done, record)
    latest.unlink(missing_ok=True)
    del model, optimizer, train, val; gc.collect(); torch.cuda.empty_cache()
    return record


def prepare_fraction(p, root, fraction):
    names = p['splits'][fraction]['train'] + p['splits'][fraction]['val']
    patients = Patients(_load_pkl_dataframe(p['pkl']), names, p['native_geometry'])
    predictions, provenance = {}, {}
    files = [(root / 'runs' / fraction / 'dice' / 'fixed_epoch_guide_predictions.npz',
              p['splits'][fraction]['train'], p['splits'][fraction]['val'])]
    for k, split in enumerate(p['guide_splits'][fraction]):
        files.append((root / 'guides' / fraction / str(k) / 'predictions.npz', split['train'], split['predict']))
    for path, trained, predicted in files:
        if set(trained) & set(predicted):
            raise ValueError('Guide training/evaluation overlap')
        with np.load(path) as archive:
            if set(archive.files) != set(predicted):
                raise ValueError('Unexpected guide cases')
            archive_hash = digest(path)
            for name in predicted:
                predictions[name] = archive[name]
                provenance[name] = dict(trained_cases=trained, fixed_epoch=p['guide_epochs'],
                                        prediction_archive_sha256=archive_hash, predicted_case_excluded_from_training=True)
    return prepare(patients, predictions, p, root / 'atlas' / fraction, provenance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--fraction', required=True, choices=['5pct', 'full'])
    parser.add_argument('--arm', default='dice', choices=list(ARMS))
    parser.add_argument('--guide-fold', type=int)
    parser.add_argument('--prepare-atlas', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    p = json.loads((args.root / 'source/docs/experiments/unwrapped_swin_20261006/protocol.json').read_text())
    if args.prepare_atlas:
        prepare_fraction(p, args.root, args.fraction)
    else:
        run(p, args.root, args.fraction, args.arm, guide_fold=args.guide_fold)


if __name__ == '__main__':
    main()
