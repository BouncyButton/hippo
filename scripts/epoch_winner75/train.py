"""Native architectures/region losses with a shared 75-epoch training policy."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import torch

import train_compute_matched_baselines as base
from scripts.campaign90 import build_model, MODELS
from scripts.capped_winner.core import calibrate, model_digest
from scripts.capped_winner.geometry import make_targets
from scripts.capped_winner.objectives import ARM, training_loss
from scripts.depth_supervision.train import Patients, autocast
from scripts.depth_supervision.utils import (
    atomic_json, digest, restore_rng, rng_state, save_checkpoint, seed_all,
    stopping_state, write_history,
)


def read(path):
    return json.loads(Path(path).read_text())


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def run_epochs(p, entry, model, criterion, optimizer, train, validation, output,
               device, *, evaluator=base.evaluate, stop_after=None):
    """Commit complete epochs, including selected weights and all stochastic state."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ident = dict(protocol=p['binding'], entry=entry)
    config = dict(identity=ident, parameters=sum(v.numel() for v in model.parameters()),
                  initialization_sha256=model_digest(model), augmentation='none',
                  max_epochs=p['max_epochs'], early_stopping=p['early_stopping'],
                  learning_rate_policy=p['learning_rate_policy'], auxiliary_branch=False)
    if (output/'config.json').exists() and read(output/'config.json') != config:
        raise ValueError('Run configuration changed')
    atomic_json(output/'config.json', config)
    if (output/'completion.json').exists():
        return verify_completed(output)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=.5)
    order_rng = torch.Generator().manual_seed(entry['seed'])
    history, best, calibration, warmup = [], None, None, None
    latest = output/'latest.pt'
    if latest.exists():
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        if saved['identity'] != ident:
            raise ValueError('Resume identity changed')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        history, best = saved['history'], saved['best']
        calibration, warmup = saved['calibration'], saved['warmup']
        order_rng.set_state(saved['order_rng'])
        restore_rng(saved['rng'])
        del saved
    for epoch in range(len(history)+1, p['max_epochs']+1):
        if stopping_state(history, **p['early_stopping'])['triggered']:
            break
        started = time.monotonic()
        scale = min(1., max(0., (epoch-5)/5)) if entry['objective'] == ARM else 0.
        if scale and calibration is None:
            calibration = calibrate(model, train, ARM, device, device.type == 'cuda',
                                    native_criterion=criterion, protocol=p)
            print(json.dumps(dict(stage='calibrated', run=entry['id'],
                                  weights=calibration['weights'])), flush=True)
        weights = calibration['weights'] if calibration else {}
        model.train()
        losses = []
        for i in torch.randperm(len(train), generator=order_rng).tolist():
            item = train[i]
            labels = item['label'][None].to(device)
            target = None
            if scale:
                geo = p['native_geometry'][item['case_name']]
                target = make_targets(labels, geo['model_affine'], geo['model_affine'],
                                      geo['native_shape'], p['depth_geometry'], device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device):
                outputs = model(item['image'][None].to(device))
            loss = training_loss(outputs, labels, ARM if scale else 'dice', weights,
                                 scale, criterion=criterion, depth_weights=target)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            grads = [v.grad for v in model.parameters() if v.grad is not None]
            if not grads or not torch.stack([torch.isfinite(g).all() for g in grads]).all():
                raise FloatingPointError('Missing or nonfinite gradients')
            if entry['model'] == 'nnunet':
                torch.nn.utils.clip_grad_norm_(model.parameters(), 12., foreach=False)
            optimizer.step()
            losses.append(float(loss.detach()))
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        result = evaluator(model, validation, device, device.type == 'cuda')
        score = result['summary']['macro_hard_dice']
        if not np.isfinite(score):
            raise FloatingPointError('Nonfinite validation Dice')
        if best is None or score > best['score']:
            best = dict(epoch=epoch, score=score, model=cpu_state(model), validation=result)
        if epoch == 5:
            warmup = dict(epoch=5, model_sha256=model_digest(model))
            if p.get('expected_warmup_sha256') is not None:
                assert warmup['model_sha256'] == p['expected_warmup_sha256'], 'Paired warmup changed'
        row = dict(epoch=epoch, updates=epoch*len(train), learning_rate=lr,
                   train_loss=float(np.mean(losses)), constraint_scale=scale,
                   validation_dice=score, **result['summary'],
                   epoch_seconds=time.monotonic()-started)
        history.append(row)
        save_checkpoint(latest, dict(identity=ident, model=model.state_dict(),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), history=history,
            best=best, calibration=calibration, warmup=warmup,
            order_rng=order_rng.get_state(), rng=rng_state()))
        write_history(output/'metrics.csv', history)
        atomic_json(output/'progress.json', dict(run=entry['id'], **row,
            best_epoch=best['epoch'], best_dice=best['score']))
        print(json.dumps(dict(run=entry['id'], **row)), flush=True)
        if stop_after and epoch >= stop_after:
            return None
    if not history or best is None or warmup is None:
        raise ValueError('Incomplete training trajectory')
    model.load_state_dict(best['model'])
    confirmed = evaluator(model, validation, device, device.type == 'cuda')
    if abs(confirmed['summary']['macro_hard_dice']-best['score']) > 1e-7:
        raise ValueError('Selected checkpoint did not reproduce its score')
    atomic_json(output/'validation_best.json', confirmed)
    atomic_json(output/'calibration.json', calibration or dict(status='not_required', weights={}))
    atomic_json(output/'warmup.json', warmup)
    save_checkpoint(output/'best.pt', dict(identity=ident, model=best['model'], epoch=best['epoch']))
    files = ['config.json', 'metrics.csv', 'calibration.json', 'warmup.json',
             'validation_best.json', 'best.pt', 'latest.pt']
    record = dict(identity=ident, model=entry['model'], objective=entry['objective'],
        fold=entry['fold'], seed=entry['seed'], best_dice=best['score'], best_epoch=best['epoch'],
        final_dice=history[-1]['validation_dice'], completed_epochs=len(history),
        optimizer_updates=len(history)*len(train), initialization_sha256=config['initialization_sha256'],
        warmup_sha256=warmup['model_sha256'], training_seconds=sum(r['epoch_seconds'] for r in history),
        early_stopping=stopping_state(history, **p['early_stopping']),
        artifacts={name: digest(output/name) for name in files})
    atomic_json(output/'completion.json', record)
    return verify_completed(output)


def verify_completed(output):
    output = Path(output)
    record = read(output/'completion.json')
    retired = read(output/'retired_checkpoints.json') if (output/'retired_checkpoints.json').exists() else {}
    for name, sha in record['artifacts'].items():
        if name in retired and not (output/name).exists():
            assert retired[name] == sha
        else:
            assert digest(output/name) == sha, name
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--pkl', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    p = read(args.plan)
    p['binding'] = digest(args.plan)
    entry = next(e for e in p['runs'] if e['id'] == args.run_id)
    seed_all(entry['seed'])
    torch.set_num_threads(4)
    assert digest(args.pkl) == p['pkl_sha256']
    spec = MODELS[entry['model']]
    model = build_model(entry['model'], p['nnunet_architecture']).cuda()
    assert sum(v.numel() for v in model.parameters()) == spec['parameters']
    options = dict(p, initial_lrs={spec['alias']: spec['initial_lr']})
    criterion, optimizer = base.build_loss_and_optimizer(spec['alias'], model, options)
    criterion = criterion if entry['model'] == 'nnunet' else None
    df = base._load_pkl_dataframe(args.pkl)
    train, val = [Patients(df, p['split'][key], p['native_geometry']) for key in ('train','val')]
    del df
    output = args.root/'runs'/entry['id']
    if entry['objective'] == ARM:
        baseline = verify_completed(args.root/'runs'/(entry['model']+'__dice'))
        assert model_digest(model) == baseline['initialization_sha256'], 'Paired initialization changed'
        p['expected_warmup_sha256'] = baseline['warmup_sha256']
    output.mkdir(parents=True, exist_ok=True)
    with (output/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_epochs(p, entry, model, criterion, optimizer, train, val, output, torch.device('cuda'))


if __name__ == '__main__':
    main()
