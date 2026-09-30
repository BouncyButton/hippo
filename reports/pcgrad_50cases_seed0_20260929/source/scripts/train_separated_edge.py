#!/usr/bin/env python3
"""Paired pooled/separated edge experiment using frozen historical training code."""
from __future__ import annotations
import argparse
import copy
import itertools
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'scripts')]
from probe_edge_consistency import digest, save
from train_edge_lowdata import data_arguments, SPLIT_HASHES

ARMS = ('pooled', 'separated')
MONITOR_EPOCHS = {0, 5, 15, 30, 45, 60, 75}


def verify_payload():
    manifest = json.loads((REPO / 'PAYLOAD.json').read_text())
    for name, expected in manifest['files'].items():
        assert digest(REPO / name) == expected, name


def reference(base, fold, seed):
    root = base / (f'edge_lowdata_20260924_01/seed{seed}' if fold == 0 else
                   f'edge_lowdata_folds12_20260924_01/fold{fold}/seed{seed}')
    config = json.loads((root / 'run/config.json').read_text())
    complete = json.loads((root / 'run/completion_manifest.json').read_text())
    run = config['run']
    assert complete['status'] == 'complete' and complete['run'] == run
    assert (run['fold'], run['seed'], run['epochs']) == (fold, seed, 75)
    assert run['constraint_set'] == 'bands_edge' and run['training_augmentation'] == 'none'
    assert run['initial_checkpoint'] is None and run['supervised_loss'] == 'dice'
    assert run['early_stopping_min_epochs'] == 60 and run['early_stopping_patience'] == 8
    assert run['splits_json_sha256'] == SPLIT_HASHES[fold]
    assert not run['constraint_scale_knots']
    assert run['constraint_config'].get('bands_degree_alpha', 0) == 0
    for key in ('pkl', 'splits_json'):
        assert digest(config[key]) == config[key + '_sha256'], key
    for key in ('bands_calibration', 'edge_calibration'):
        assert digest(run[key]) == run[key + '_sha256'], key
    historical = json.loads(Path(run['bands_calibration']).read_text())
    assert digest(historical['checkpoint']) == historical['checkpoint_sha256']
    assert digest(historical['checkpoint_config']) == historical['checkpoint_config_sha256']
    source = config['source_provenance']['files']
    # Use the frozen original implementation for data, model, optimizer and unary bands.
    for name, expected in source.items():
        assert digest(REPO / name) == expected, name
    return config, historical, dict(directory=str(root), config_sha256=digest(root/'run/config.json'),
        checkpoint_sha256=historical['checkpoint_sha256'], original_edge_weight=run['edge_weight'])


def loaders(config):
    import torch
    from thesis.new_constraints import train_swinunetr_constraints as trainer
    dg, ag, _ = trainer.build_experiment_generators(config['run']['seed'])
    train, val, _, nt, nv = trainer.build_data(data_arguments(config), dg)
    assert (nt, nv) == (10, 52)
    names = [x['case_name'] for x in train.dataset.data]
    assert len(set(names)) == 10
    assert not set(names) & {x['case_name'] for x in val.dataset.data}
    return train, val, dg, ag


def objectives(config):
    from monai.losses import DiceLoss
    from thesis.new_constraints.objective import NewConstraintConfig, NewConstraintObjective
    return DiceLoss(to_onehot_y=True, softmax=True), NewConstraintObjective(
        NewConstraintConfig(**config['run']['constraint_config']))


def build_model(config, device):
    from thesis.new_constraints import train_swinunetr_constraints as trainer
    run = config['run']
    return trainer.build_swinunetr(tuple(run['spatial_size']), 3, device,
        drop_rate=run['drop_rate'], activation_checkpointing=run['activation_checkpointing'])


def calibrate(config, historical, device, output):
    import torch
    from torch.utils.data import DataLoader
    from thesis.new_constraints import train_swinunetr_constraints as trainer
    from thesis.new_constraints.separated_edge import edge_loss, rule_losses, matched_budget
    trainer.seed_everything(config['run']['seed'])
    train, _, _, _ = loaders(config)
    names = [x['case_name'] for x in train.dataset.data]
    assert set(names) == set(historical['training_case_ids'])
    cp = torch.load(historical['checkpoint'], map_location='cpu', weights_only=True)
    source_config = json.loads(Path(historical['checkpoint_config']).read_text())
    assert cp['epoch'] == 5 and source_config['run']['constraint_set'] == 'none'
    assert json.loads(json.dumps(cp['run'])) == source_config['run']
    for key in ('pkl_sha256', 'splits_json_sha256'):
        assert config[key] == source_config[key]
    for key in ('seed', 'fold'):
        assert config['run'][key] == source_config['run'][key]
    model = build_model(config, device)
    model.load_state_dict(cp['model'], strict=True)
    del cp
    model.eval()
    dice, bands = objectives(config)
    rows, ratios = [], {a: [] for a in ARMS}
    for batch in DataLoader(train.dataset, batch_size=1, shuffle=False, num_workers=0):
        images, labels = batch['image'].to(device), batch['label'].to(device)
        with torch.inference_mode(), torch.autocast('cuda', enabled=config['run']['amp']):
            raw = model(images)
        z = raw.clone().detach().float().requires_grad_(True)
        terms, counts = rule_losses(z, labels)
        assert all(bool((n > 0).all()) for n in counts.values())
        base = dice(z, labels) + bands(model, images, z, labels)['loss']
        gb = torch.autograd.grad(base, z, retain_graph=True)[0]
        br = float(gb.square().mean().sqrt())
        assert br > 0
        row = dict(case_name=str(batch['case_name'][0]), base_gradient_rms=br,
                   face_counts={k: int(n.sum()) for k, n in counts.items()}, rules={})
        losses = {**terms, 'pooled': edge_loss('pooled')(z, labels), 'separated': sum(terms.values())/3}
        for name, loss in losses.items():
            g = torch.autograd.grad(loss, z, retain_graph=True)[0]
            rms = float(g.square().mean().sqrt())
            assert torch.isfinite(g).all() and rms > 0
            row['rules'][name] = dict(loss=float(loss), gradient_rms=rms, ratio=rms/br)
            if name in ARMS:
                ratios[name].append(rms/br)
        rows.append(row)
    result = matched_budget(ratios, reference_weight=config['run']['edge_weight'])
    # Preserve the exact historical scalar, including its serialized precision.
    result['weights']['pooled'] = config['run']['edge_weight']
    result.update(status='complete', cases=rows, training_case_ids=names,
        checkpoint_sha256=digest(historical['checkpoint']), checkpoint=str(historical['checkpoint']),
        original_edge_weight=config['run']['edge_weight'],
        scope='Fixed epoch-five Dice checkpoint; all ten training cases; logit-gradient RMS. Both arms share a median budget and p95 cap. No validation calibration.')
    save(output, result)
    del model
    torch.cuda.empty_cache()
    return result


def historical_curves(base, fold, seed, output):
    import csv
    root = base / (f'edge_lowdata_20260924_01/seed{seed}' if fold == 0 else
                   f'edge_lowdata_folds12_20260924_01/fold{fold}/seed{seed}')
    bindings = json.loads((root/'reference_bindings.json').read_text())
    directories = {k: Path(bindings[k]['directory']) for k in ('dice','bands')}
    directories['edge'] = root/'run'
    records = {}
    for method, directory in directories.items():
        cfg = json.loads((directory/'config.json').read_text())
        done = json.loads((directory/'completion_manifest.json').read_text())
        assert done['status'] == 'complete' and done['run'] == cfg['run']
        assert (cfg['run']['fold'],cfg['run']['seed'],cfg['run']['epochs']) == (fold,seed,75)
        records[method] = dict(directory=str(directory),config_sha256=digest(directory/'config.json'),
            metrics_sha256=digest(directory/'metrics.csv'),selected_epoch=done['selected_epoch'],
            rows=list(csv.DictReader((directory/'metrics.csv').open())))
    save(output,records)


def gradient_summary(vectors):
    import torch
    norms = {k: float(v.double().norm()) for k, v in vectors.items()}
    cosines = {}
    for a, b in itertools.combinations(vectors, 2):
        denom = norms[a] * norms[b]
        cosines[a + '__' + b] = float(torch.dot(vectors[a].double(), vectors[b].double()) / denom) if denom else None
    return dict(norms=norms, cosines=cosines,
        edge_over_base=norms['weighted_edge']/norms['base'] if norms['base'] else None)


def monitor(model, dataset, config, epoch, device, dg, ag, output):
    """Probe fixed training cases without optimizer steps or RNG side effects."""
    import torch
    from thesis.new_constraints import train_swinunetr_constraints as trainer
    from thesis.new_constraints.separated_edge import rule_losses, edge_loss
    saved_rng, mode = trainer.rng_state(dg, ag), model.training
    run = config['run']
    scale = trainer.constraint_warmup_scale(max(epoch, 1), run['constraint_warmup_epochs']) if epoch else 0.
    dice, bands = objectives(config)
    params = tuple(p for p in model.parameters() if p.requires_grad)
    names = [str(x['case_name']) for x in dataset.data]
    selected = sorted(range(len(names)), key=lambda i: names[i])[:2]
    rows = []
    try:
        model.eval()
        for i in selected:
            batch = dataset[i]
            images, labels = batch['image'][None].to(device), batch['label'][None].to(device)
            # FP32 probes avoid AMP underflow in unscaled per-rule derivatives.
            raw = model(images.float())
            z = raw.float()
            terms, counts = rule_losses(z, labels)
            losses = dict(dice=dice(z, labels), bands=bands(model, images, z, labels)['loss'], **terms,
                          edge=edge_loss(run['edge_arm'])(z, labels))
            vectors = {'logits': {}, 'parameters': {}}
            for name, loss in losses.items():
                grads = torch.autograd.grad(loss, (z, *params), retain_graph=True, allow_unused=True)
                vectors['logits'][name] = grads[0].detach().float().reshape(-1)
                vectors['parameters'][name] = torch.cat([
                    (g.detach().float() if g is not None else torch.zeros_like(p)).reshape(-1)
                    for p, g in zip(params, grads[1:])])
                assert all(torch.isfinite(vectors[space][name]).all() for space in vectors)
            summaries = {}
            for space, v in vectors.items():
                v['base'] = v['dice'] + scale * v['bands']
                v['weighted_edge'] = scale * run['edge_weight'] * v['edge']
                summaries[space] = gradient_summary(v)
            rows.append(dict(case_name=names[i], losses={k: float(v.detach()) for k,v in losses.items()},
                face_counts={k: int(n.sum()) for k,n in counts.items()}, **summaries))
            del vectors, losses, raw, z, grads, terms
        save(output / f'gradients/epoch_{epoch:03d}.json', dict(epoch=epoch, constraint_scale=scale,
            arm=run['edge_arm'], cases=rows, scope='Two fixed training cases; FP32 eval-mode derivatives before AdamW; gradients are observational, not actual mixed-precision optimizer updates.'))
    finally:
        trainer.restore_rng_state(saved_rng, dg, ag)
        model.train(mode)


def audit_model(model, loaders_by_split, config, device, output):
    import numpy as np
    import torch
    from audit_edge_coherence import metrics
    rows = []
    model.eval()
    for split, loader in loaders_by_split.items():
        for batch in loader:
            with torch.inference_mode(), torch.autocast('cuda', enabled=config['run']['amp']):
                logits = model(batch['image'].to(device))
            p = logits[0].float().softmax(0).cpu().numpy()
            truth = batch['label'][0, 0].numpy().astype(np.uint8)
            rows.append(dict(split=split, case_name=str(batch['case_name'][0]),
                metrics=metrics(p[1:].sum(0), p.argmax(0).astype(np.uint8), truth)))
    save(output, rows)
    return rows


def fit(config, output, device):
    import torch
    from torch.utils.data import DataLoader
    from thesis.new_constraints import train_swinunetr_constraints as trainer
    from thesis.new_constraints.early_stopping import EarlyStopping
    from thesis.new_constraints.separated_edge import rule_losses, edge_loss
    run = config['run']
    trainer.seed_everything(run['seed'])
    train, val, dg, ag = loaders(config)
    model = build_model(config, device)
    optimizer, scheduler = trainer.build_optimizer_and_scheduler(run['optimizer_mode'], model, run['epochs'],
        adamw_gamma=run['adamw_gamma'], step_size=run['step_size'], learning_rate=run['learning_rate'], weight_decay=run['weight_decay'])
    scaler = torch.cuda.amp.GradScaler(enabled=run['amp'])
    dice, bands = objectives(config)
    edge = edge_loss(run['edge_arm'])
    stopping = EarlyStopping(patience=run['early_stopping_patience'], min_delta=run['early_stopping_min_delta'],
                             min_epochs=run['early_stopping_min_epochs'])
    best, best_epoch, start_epoch = -1., 0, 1
    latest = output/'checkpoint_latest.pt'
    if latest.exists():
        cp = torch.load(latest, map_location='cpu', weights_only=True)
        assert cp['run'] == run
        model.load_state_dict(cp['model'], strict=True)
        optimizer.load_state_dict(cp['optimizer']); scheduler.load_state_dict(cp['scheduler']); scaler.load_state_dict(cp['scaler'])
        stopping = EarlyStopping(**cp['stopping_state'])
        best, best_epoch, start_epoch = cp['best_hard_dice'], cp['best_epoch'], cp['epoch'] + 1
        trainer.restore_rng_state(cp['rng'], dg, ag)
        del cp
    else:
        monitor(model, train.dataset, config, 0, device, dg, ag, output)
    cp = None
    for epoch in range(start_epoch, run['epochs'] + 1):
        if stopping.should_stop:
            break
        started = time.perf_counter()
        model.train()
        sums = dict(total=0., dice=0., bands=0., edge=0., inner=0., outer=0., cross=0.)
        scale = trainer.constraint_warmup_scale(epoch, run['constraint_warmup_epochs'])
        for batch in train:
            images, labels = batch['image'].to(device), batch['label'].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=run['amp']):
                logits = model(images)
                d = dice(logits, labels)
                b = bands(model, images, logits, labels)['loss']
                terms, counts = rule_losses(logits, labels)
                assert all(bool((n > 0).all()) for n in counts.values())
                e = edge(logits, labels) if run['edge_arm'] == 'pooled' else sum(terms.values())/3
                loss = d + scale * (b + run['edge_weight'] * e)
            assert torch.isfinite(loss)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            for k,v in dict(total=loss, dice=d, bands=b, edge=e, **terms).items():
                sums[k] += float(v.detach())
        with torch.inference_mode():
            validation = trainer.evaluate_validation_metrics(model, val, bands, device, num_classes=3,
                amp=run['amp'], evaluate_constraints=False, all_translation_shifts=False, calibration_diagnostics=False)
        row = dict(epoch=epoch, **{'train_'+k: v/len(train) for k,v in sums.items()}, **validation,
                   learning_rate=optimizer.param_groups[0]['lr'], constraint_scale=scale,
                   epoch_seconds=time.perf_counter()-started)
        trainer.validate_epoch_metrics(row)
        save(output/f'epochs/{epoch:03d}.json', row)
        print(json.dumps(dict(fold=run['fold'], seed=run['seed'], arm=run['edge_arm'], **row)), flush=True)
        scheduler.step()
        improved = validation['val_dice_hard'] > best
        if improved:
            best, best_epoch = validation['val_dice_hard'], epoch
        stopping.update(epoch, validation['val_dice_hard'])
        terminal = stopping.should_stop or epoch == run['epochs']
        # Additional loaders/diagnostics must not consume the training RNG stream.
        state = trainer.rng_state(dg, ag)
        if epoch in MONITOR_EPOCHS or terminal:
            monitor(model, train.dataset, config, epoch, device, dg, ag, output)
            audit_model(model, {'validation': val}, config, device, output/f'coherence/epoch_{epoch:03d}.json')
        trainer.restore_rng_state(state, dg, ag)
        verify_payload()
        cp = dict(epoch=epoch, model=model.state_dict(), best_hard_dice=best, best_epoch=best_epoch,
            run=run, rng=state, stopping_state=vars(stopping), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), scaler=scaler.state_dict())
        if improved:
            trainer.save_checkpoint_payload(output/'checkpoint_best.pt', dict(epoch=epoch, model=model.state_dict(), run=run))
        trainer.save_checkpoint_payload(latest, cp)
    selected = torch.load(output/'checkpoint_best.pt', map_location='cpu', weights_only=True)
    assert selected['epoch'] == best_epoch
    model.load_state_dict(selected['model'], strict=True)
    audit_model(model, {'train': DataLoader(train.dataset,batch_size=1,shuffle=False), 'validation': val},
                config, device, output/'selected_cases.json')
    save(output/'completion.json', dict(status='complete', selected_epoch=best_epoch, stopped_epoch=stopping.epoch,
        best_val_dice_hard=best, checkpoint_sha256=digest(output/'checkpoint_best.pt'),
        selected_cases_sha256=digest(output/'selected_cases.json'), run=run,
        checkpoint_storage='Selected FP32 inference checkpoint retained; completed-arm optimizer state retired.'))
    # Only retire this experiment's own intermediate after checkpoint and audit are durable.
    latest.unlink()
    del model, optimizer, scheduler, scaler, selected, cp
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--base', type=Path, default=Path('/mnt/beegfsstudents/home/3160552'))
    parser.add_argument('--fold', type=int, choices=range(3), required=True)
    parser.add_argument('--seed', type=int, choices=range(3), required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    verify_payload()
    config, historical, binding = reference(args.base, args.fold, args.seed)
    save(args.root/'reference_binding.json', binding)
    historical_curves(args.base,args.fold,args.seed,args.root/'historical_curves.json')
    if args.preflight_only:
        print(json.dumps(dict(status='preflight_passed', fold=args.fold, seed=args.seed)))
        return
    import torch
    from thesis.new_constraints import train_swinunetr_constraints as trainer
    from run_degree_bands_ab import check_quota
    assert torch.cuda.is_available()
    device = torch.device('cuda')
    trainer.seed_everything(args.seed)
    runtime, execution = trainer.collect_runtime_provenance(), trainer.collect_execution_provenance(device)
    assert config['runtime_provenance'] == runtime
    trainer.validate_calibration_execution(config['execution_provenance'], execution, allow_mig_profile_change=True)
    calibration_path = args.root/'calibration.json'
    calibration = json.loads(calibration_path.read_text()) if calibration_path.exists() else calibrate(config,historical,device,calibration_path)
    assert calibration['checkpoint_sha256'] == historical['checkpoint_sha256']
    for arm in ARMS:
        output = args.root/arm
        if (output/'completion.json').exists():
            done = json.loads((output/'completion.json').read_text())
            assert done['status'] == 'complete'
            assert digest(output/'checkpoint_best.pt') == done['checkpoint_sha256']
            assert digest(output/'selected_cases.json') == done['selected_cases_sha256']
            continue
        output.mkdir(exist_ok=True)
        check_quota(output, 'start', .8)
        candidate = copy.deepcopy(config)
        candidate['run'].update(constraint_set='bands_edge_'+arm, edge_arm=arm, edge_weight=calibration['weights'][arm],
            edge_rule_weights=([1/3]*3 if arm=='separated' else None),
            edge_calibration=str(calibration_path),edge_calibration_sha256=digest(calibration_path))
        candidate.update(runtime_provenance=runtime,execution_provenance=execution,
            experiment_payload_sha256=digest(REPO/'PAYLOAD.json'),
            experiment_calibration_sha256=digest(calibration_path), training_script_sha256=digest(__file__))
        # Source hashes inside historical run config remain the verified original backbone provenance.
        if (output/'config.json').exists():
            assert json.loads((output/'config.json').read_text()) == candidate
        else:
            save(output/'config.json',candidate)
        fit(candidate,output,device)
    save(args.root/'completion.json',dict(status='complete',fold=args.fold,seed=args.seed,arms=list(ARMS),job_id=os.getenv('SLURM_JOB_ID')))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        print(traceback.format_exc(), flush=True)
        raise
