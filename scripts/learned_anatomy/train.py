"""One matched 5% native SwinUNETR: Dice plus the audited frozen feature loss."""
import fcntl
import json
from pathlib import Path
import time
import numpy as np
import torch
from baselines.swin_unetr.swin_unetr import _load_pkl_dataframe
from scripts.depth_supervision.train import Patients, GeometryCache, model_factory, autocast, evaluate
from scripts.depth_supervision.losses import components, calibrated_coefficient
from scripts.depth_supervision.utils import atomic_json, digest, seed_all, rng_state, restore_rng, save_checkpoint, stopping_state, write_history
from .prior import LearnedLoss, ContextCache
from .verify import verify_gate


def load_data(p):
    df = _load_pkl_dataframe(p['pkl'])
    return tuple(Patients(df, p['splits']['5pct'][key], p['native_geometry']) for key in ('train', 'val'))


def calibrate(model, patients, criterion, cache, device, settings):
    state = rng_state()
    was_training = model.training
    model.eval()
    records = []
    try:
        for item in patients.items:
            with torch.no_grad(), autocast(device):
                raw = model(item['image'][None].to(device))
            logits = raw.detach().float().requires_grad_(True)
            labels = item['label'][None].to(device)
            dice = components(logits, labels)['dice']
            learned = criterion(logits, labels, cache.get(item), item['case_name'])
            norms = [float(torch.autograd.grad(v, logits, retain_graph=True)[0].square().mean().sqrt()) for v in (dice, learned)]
            records.append(dict(case_name=item['case_name'], dice=norms[0], learned=norms[1]))
        stats = calibrated_coefficient([r['dice'] for r in records], [r['learned'] for r in records], settings['target_ratio'], settings['cap_ratio'])
        return dict(coefficient=stats['coefficient'], statistics=stats, cases=records, basis='training-only logit-gradient RMS', checkpoint_epoch=settings['prefix_epochs'])
    finally:
        restore_rng(state)
        model.train(was_training)


def run(root, p, device, stop_after=None):
    root = Path(root)
    gate = verify_gate(root, p)
    output = root/'runs/5pct_dice_learned_s0'
    output.mkdir(parents=True, exist_ok=True)
    ident = dict(protocol_sha256=p['identity_sha'], encoder_sha256=gate['encoder_sha256'], fold=0, seed=0,
                 **p['splits']['5pct'])
    with (output/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/'completion.json').exists():
            c = json.loads((output/'completion.json').read_text())
            assert c['identity'] == ident
            assert all(digest(output/n)==h for n,h in c['artifacts'].items())
            return c
        seed_all(p['seed'])
        patients, validation = load_data(p)
        model = model_factory(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'], foreach=False)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=p['step_size'], gamma=p['lr_gamma'])
        order = torch.Generator().manual_seed(p['seed'])
        criterion = LearnedLoss(root/'encoder/frozen.pt', device)
        contexts = ContextCache(root, p, device)
        history, best_score, best_epoch, updates, calibration = [], -1., 0, 0, None
        latest, best = output/'latest.pt', output/'best.pt'
        if latest.exists():
            saved = torch.load(latest, map_location=device, weights_only=False)
            assert saved['identity'] == ident
            model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
            history, best_score, best_epoch, updates, calibration = (saved[k] for k in ('history','best_score','best_epoch','updates','calibration'))
            order.set_state(saved['order_rng'].cpu()); restore_rng(saved['rng'])
            if best_epoch == len(history):
                save_checkpoint(best, dict(identity=ident, model=model.state_dict(), epoch=best_epoch))
            del saved
        config = dict(identity=ident, objective='Dice + frozen learned spatial anatomy loss', input_channels=1,
            auxiliary_image_branch=False, trainable_segmentation_parameters=sum(x.numel() for x in model.parameters()),
            encoder_parameters=sum(x.numel() for x in criterion.encoder.parameters()), encoder_trainable=False,
            all_segmentation_parameters_trainable=all(x.requires_grad for x in model.parameters()),
            calibration=calibration, encoder_used_at_inference=False,
            gradient_audit_passed=gate.get('passed'), exploratory_audit_acceptance=p.get('exploratory_audit_acceptance'),
            depth_balancing=False, sdf=False, focal=False, profiles=False)
        atomic_json(output/'config.json', config)
        for epoch in range(len(history)+1, p['max_epochs']+1):
            if stopping_state(history, **p['early_stopping'])['triggered']:
                break
            started = time.monotonic()
            scale = min(1., max(0., (epoch-p['calibration']['prefix_epochs'])/p['calibration']['ramp_epochs']))
            if scale and calibration is None:
                calibration = calibrate(model, patients, criterion, contexts, device, p['calibration'])
                config['calibration'] = calibration
                atomic_json(output/'config.json', config)
                print(json.dumps(dict(stage='calibration', **calibration)), flush=True)
            coefficient = calibration['coefficient'] if calibration else 0.
            model.train()
            totals = []
            for i in torch.randperm(len(patients), generator=order).tolist():
                item = patients[i]
                optimizer.zero_grad(set_to_none=True)
                with autocast(device):
                    logits = model(item['image'][None].to(device))
                labels = item['label'][None].to(device)
                dice = components(logits, labels)['dice']
                learned = criterion(logits, labels, contexts.get(item), item['case_name']) if scale else logits.new_tensor(0.)
                total = dice + scale*coefficient*learned
                assert torch.isfinite(total)
                total.backward()
                assert all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None)
                assert all(x.grad is None for x in criterion.parameters())
                optimizer.step(); updates += 1
                totals.append([float(x.detach()) for x in (total, dice, learned)])
            lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            metrics = evaluate(model, validation, device)
            score = metrics['summary']['macro_dice']
            improved = score > best_score
            if improved:
                best_score, best_epoch = score, epoch
            means = np.mean(totals, axis=0)
            row = dict(epoch=epoch, updates=updates, learning_rate=lr, train_total=float(means[0]),
                train_dice=float(means[1]), train_learned=float(means[2]), learned_scale=scale,
                learned_coefficient=coefficient, epoch_seconds=time.monotonic()-started, validation_dice=score,
                **{'validation_'+k:metrics['summary'][k] for k in ('foreground_dice','missed','extra','swaps')})
            history.append(row)
            save_checkpoint(latest, dict(identity=ident, model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                history=history, best_score=best_score, best_epoch=best_epoch, updates=updates, calibration=calibration, order_rng=order.get_state(), rng=rng_state()))
            if improved:
                save_checkpoint(best, dict(identity=ident, model=model.state_dict(), epoch=best_epoch))
            write_history(output/'metrics.csv', history)
            print(json.dumps(dict(stage='swin', **row)), flush=True)
            if stop_after and epoch >= stop_after:
                return
        saved = torch.load(best, map_location=device, weights_only=False)
        assert saved['identity']==ident and saved['epoch']==best_epoch
        model.load_state_dict(saved['model'])
        geometry = GeometryCache(root/'diagnostic_geometry', p['geometry'])
        final = evaluate(model, validation, device, geometry_cache=geometry, detailed=True, predictions_path=output/'validation_predictions.npz')
        atomic_json(output/'validation_best.json', final)
        assert digest(root/'encoder/frozen.pt')==ident['encoder_sha256']
        files = ['best.pt','metrics.csv','config.json','validation_best.json','validation_predictions.npz']
        record = dict(identity=ident, completed_epochs=len(history), best_epoch=best_epoch, optimizer_updates=updates,
            validation=final['summary'], stopping=stopping_state(history, **p['early_stopping']),
            training_seconds=sum(x['epoch_seconds'] for x in history), artifacts={n:digest(output/n) for n in files})
        atomic_json(output/'completion.json', record)
        latest.unlink(missing_ok=True)
        return record


def preflight(root, p, device):
    verify_gate(root, p)
    state = rng_state()
    seed_all(0)
    try:
        patients, _ = load_data(p)
        model = model_factory(device)
        criterion = LearnedLoss(Path(root)/'encoder/frozen.pt', device)
        context = ContextCache(root, p, device)
        item = patients[0]
        opt = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'], foreach=False)
        with autocast(device):
            logits = model(item['image'][None].to(device))
        labels = item['label'][None].to(device)
        learned = criterion(logits, labels, context.get(item), item['case_name'])
        learned.backward()
        gradient_norm = float(torch.stack([x.grad.square().sum() for x in model.parameters() if x.grad is not None]).sum().sqrt())
        assert gradient_norm > 0 and np.isfinite(gradient_norm)
        opt.step()
        assert all(x.grad is None and not x.requires_grad for x in criterion.parameters())
        record = dict(passed=True, device=str(device), input_shape=list(item['image'].shape),
            full_native_network_trainable=all(x.requires_grad for x in model.parameters()),
            loss_only_native_gradient_norm=gradient_norm, encoder_frozen=True,
            segmentation_parameters=sum(x.numel() for x in model.parameters()))
        atomic_json(Path(root)/'swin_preflight.json', record)
        return record
    finally:
        restore_rng(state)
