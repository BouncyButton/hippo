"""Train only the new registered runs, with exact interruption recovery."""
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
from scripts.unwrapped_swin.cache import AtlasCache
from scripts.unwrapped_swin.evaluate import evaluate
from scripts.unwrapped_swin.train import completed_record
from .models import FollowupSwin


def identity(p, entry):
    split = p['splits'][entry['fraction']]
    return dict(protocol_sha256=p['identity_sha'], run=entry,
                train=split['train'], val=split['val'], atlas_protocol_sha256=p['atlas_protocol_sha256'])


def atlas_protocol(p):
    return dict(p, identity_sha=p['atlas_protocol_sha256'])


def run(p, root, entry, *, stop_after=None, device_override=None):
    root = Path(root)
    output = root / 'runs' / entry['id']
    output.mkdir(parents=True, exist_ok=True)
    ident = identity(p, entry)
    existing = completed_record(output, ident)
    if existing is not None:
        return existing
    seed_all(entry['seed'])
    split = p['splits'][entry['fraction']]
    dataframe = _load_pkl_dataframe(p['pkl'])
    train, val = [Patients(dataframe, split[k], p['native_geometry']) for k in ('train', 'val')]
    del dataframe
    device = torch.device(device_override or 'cuda')
    model = FollowupSwin(device, entry['arm'], entry['gate'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'], foreach=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=p['step_size'], gamma=p['lr_gamma'])
    order_rng = torch.Generator().manual_seed(entry['seed'])
    cache = GeometryCache(root / 'loss_geometry', p['geometry'])
    atlas = AtlasCache(root / 'atlas' / entry['fraction'], atlas_protocol(p)) if model.settings else None
    coefficient = {'depth': p['fixed_depth_coefficients'][entry['fraction']]}
    history, best_score, best_epoch, updates = [], -1., 0, 0
    latest, best = output / 'latest.pt', output / 'best.pt'
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        if state['identity'] != ident:
            raise ValueError('Resume identity changed')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        history, best_score, best_epoch, updates = state['history'], state['best_score'], state['best_epoch'], state['updates']
        order_rng.set_state(state['order_rng'].cpu()); restore_rng(state['rng'])
        del state
        if best_epoch == len(history):
            save_checkpoint(best, dict(identity=ident, model=model.state_dict(), epoch=best_epoch))
        print(json.dumps(dict(stage='checkpoint_resumed', run=entry['id'], epoch=len(history), updates=updates)), flush=True)
    atomic_json(output / 'config.json', dict(identity=ident, settings=model.settings, objective=entry['objective'],
                gate=entry['gate'], depth_coefficient=coefficient['depth'] if entry['objective']=='depth' else 0.,
                parameters=sum(x.numel() for x in model.parameters()),
                all_parameters_trainable=all(x.requires_grad for x in model.parameters()),
                atlas_seed=0, coefficient_seed=0))
    started = time.monotonic()
    for epoch in range(len(history)+1, p['max_epochs']+1):
        if stopping_state(history, **p['early_stopping'])['triggered']:
            break
        model.train()
        totals, grad_norm = [], 0.
        scale = min(1., max(0., (epoch-5)/5)) if entry['objective']=='depth' else 0.
        for i in torch.randperm(len(train), generator=order_rng).tolist():
            item = train[i]
            inputs = atlas.get(item) if atlas else None
            optimizer.zero_grad(set_to_none=True)
            with autocast(device):
                logits = model(item['image'][None].to(device), inputs)
            active = entry['objective'] if scale > 0 else 'dice'
            values = components(logits, item['label'][None].to(device), cache.get(item, fields=False) if active=='depth' else None, active)
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
        metrics = evaluate(model, val, device, atlas_cache=atlas)
        score = metrics['summary']['macro_dice']
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
        row = dict(epoch=epoch, updates=updates, learning_rate=lr, train_total=float(np.mean(totals)),
                   auxiliary_scale=scale, auxiliary_gradient_square_mean=grad_norm/len(train),
                   elapsed_seconds=time.monotonic()-started, validation_dice=score,
                   **{'validation_'+k:metrics['summary'][k] for k in ('foreground_dice','missed','extra','swaps')})
        history.append(row)
        save_checkpoint(latest, dict(identity=ident, model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), history=history, best_score=best_score, best_epoch=best_epoch,
            updates=updates, order_rng=order_rng.get_state(), rng=rng_state()))
        if improved:
            save_checkpoint(best, dict(identity=ident, model=model.state_dict(), epoch=best_epoch))
            atomic_json(output / 'validation_best.json', metrics)
        write_history(output / 'metrics.csv', history)
        print(json.dumps(dict(run=entry['id'], **row)), flush=True)
        if stop_after and epoch>=stop_after:
            return
    selected = torch.load(best, map_location=device, weights_only=False)
    if selected['identity'] != ident or selected['epoch'] != best_epoch:
        raise ValueError('Selected checkpoint mismatch')
    model.load_state_dict(selected['model']); del selected
    final = evaluate(model, val, device, geometry_cache=cache, atlas_cache=atlas, detailed=True,
                     predictions_path=output / 'validation_predictions.npz')
    atomic_json(output / 'validation_best.json', final)
    files = ['best.pt','metrics.csv','config.json','validation_best.json','validation_predictions.npz']
    record = dict(identity=ident, completed_epochs=len(history), best_epoch=best_epoch, optimizer_updates=updates,
                  early_stopping=stopping_state(history, **p['early_stopping']), validation=final['summary'],
                  artifacts={n:digest(output/n) for n in files})
    atomic_json(output / 'completion.json', record)
    latest.unlink(missing_ok=True)
    del model, optimizer, train, val
    gc.collect(); torch.cuda.empty_cache()
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    p = json.loads((args.root/'protocol.json').read_text())
    entry = next(e for e in p['runs'] if e['id']==args.run_id)
    run(p,args.root,entry)


if __name__ == '__main__':
    main()
