#!/usr/bin/env python3
"""Matched 5%-data edge-consistency run with training-only coefficient calibration."""
from __future__ import annotations
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
from probe_edge_consistency import digest, save


SPLIT_HASHES = {
    0: '7b69e54dd720b38f58f11e73302899e626ab3985d5c15d5c72c5daeb420ce8be',
    1: 'd4c2a53da8b0dd60839cd9afc6320a298c46bce22dd6a4b7907b95f04dace086',
    2: 'b3d55f8970bdaa93f123a7003a862fe7708f664642b7db823ee3c5bf974faccf',
}


def reference_directories(base, fold, seed):
    if fold == 0:
        root = base / f'low_data_noaug_seed{seed}_20260917_01'
        suffix = f'_5pct_seed{seed}_noaug'
    elif fold in (1, 2):
        root = base / f'low_data_noaug_fold{fold}_20260917_01'
        suffix = f'_seed{seed}'
    else:
        raise ValueError(f'Unsupported fold {fold}')
    return {name: root/'runs'/f'{prefix}{suffix}'
            for name, prefix in [('dice', 'baseline'), ('bands', 'bands')]}


def data_arguments(config):
    run = config['run']
    return SimpleNamespace(pkl=Path(config['pkl']), splits_json=Path(config['splits_json']), dataset='MSD',
        fold=run['fold'], spatial_size=run['spatial_size'], resize=run['resize'], batch_size=1, num_workers=0,
        num_classes=3, plain_tensors=True)


def fit(config, output, device, source, *, compact_checkpoints=False):
    import torch
    from monai.losses import DiceLoss
    from thesis.new_constraints import train_swinunetr_constraints as trainer
    from thesis.new_constraints.objective import NewConstraintConfig, NewConstraintObjective
    from thesis.new_constraints.edge_consistency import BoundaryEdgeConsistencyLoss
    from thesis.new_constraints.early_stopping import EarlyStopping
    run = config['run']
    trainer.seed_everything(run['seed'])
    data_generator, augmentation_generator, _ = trainer.build_experiment_generators(run['seed'])
    data_args = data_arguments(config)
    train_loader, val_loader, _, train_n, val_n = trainer.build_data(data_args, data_generator)
    assert (train_n, val_n) == (10, 52)
    model = trainer.build_swinunetr(tuple(run['spatial_size']), 3, device, drop_rate=run['drop_rate'],
                                   activation_checkpointing=run['activation_checkpointing'])
    optimizer, scheduler = trainer.build_optimizer_and_scheduler(run['optimizer_mode'], model, run['epochs'],
        adamw_gamma=run['adamw_gamma'], step_size=run['step_size'], learning_rate=run['learning_rate'], weight_decay=run['weight_decay'])
    scaler = torch.cuda.amp.GradScaler(enabled=run['amp'])
    dice = DiceLoss(to_onehot_y=True, softmax=True)
    base = NewConstraintObjective(NewConstraintConfig(**run['constraint_config']))
    edge = BoundaryEdgeConsistencyLoss()
    stopping = EarlyStopping(patience=run['early_stopping_patience'], min_delta=run['early_stopping_min_delta'],
                             min_epochs=run['early_stopping_min_epochs'])
    fields = ['epoch','train_loss','train_supervised_loss','train_bands_loss','train_edge_loss','constraint_scale',
              'learning_rate','val_dice_soft','val_dice_hard','epoch_seconds']
    best, best_epoch = -1., 0
    with (output / 'metrics.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for epoch in range(1, run['epochs'] + 1):
            if stopping.should_stop: break
            started = time.perf_counter()
            model.train()
            sums = dict(total=0., supervised=0., bands=0., edge=0.)
            scale = trainer.constraint_warmup_scale(epoch, run['constraint_warmup_epochs'])
            for batch in train_loader:
                images, labels = batch['image'].to(device), batch['label'].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=run['amp']):
                    logits = model(images)
                    supervised = dice(logits, labels)
                    bands_loss = base(model, images, logits, labels)['loss']
                    edge_loss = edge(logits, labels)
                    loss = supervised + scale * (bands_loss + run['edge_weight'] * edge_loss)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                for key, value in [('total', loss), ('supervised', supervised), ('bands', bands_loss), ('edge', edge_loss)]:
                    sums[key] += float(value.detach())
            metrics = trainer.evaluate_validation_metrics(model, val_loader, base, device, num_classes=3,
                amp=run['amp'], evaluate_constraints=False, all_translation_shifts=False, calibration_diagnostics=False)
            row = {'epoch': epoch, 'train_loss': sums['total']/len(train_loader),
                'train_supervised_loss': sums['supervised']/len(train_loader), 'train_bands_loss': sums['bands']/len(train_loader),
                'train_edge_loss': sums['edge']/len(train_loader), 'constraint_scale': scale,
                'learning_rate': optimizer.param_groups[0]['lr'], 'val_dice_soft': metrics['val_dice_soft'],
                'val_dice_hard': metrics['val_dice_hard'], 'epoch_seconds': time.perf_counter()-started}
            trainer.validate_epoch_metrics(row)
            writer.writerow(row); stream.flush(); os.fsync(stream.fileno())
            print(json.dumps({'seed': run['seed'], **row}), flush=True)
            scheduler.step()
            improved = metrics['val_dice_hard'] > best
            if improved: best, best_epoch = metrics['val_dice_hard'], epoch
            stopping.update(epoch, metrics['val_dice_hard'])
            trainer.validate_source_provenance_unchanged(source)
            checkpoint = {'epoch': epoch, 'model': model.state_dict(), 'best_hard_dice': best,
                'best_epoch': best_epoch, 'run': run, 'rng': trainer.rng_state(data_generator, augmentation_generator),
                'early_stopping': stopping.summary()}
            if compact_checkpoints:
                checkpoint['checkpoint_format'] = 'inference_only_no_optimizer'
            else:
                checkpoint.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict())
            trainer.save_checkpoint_payload(output / 'checkpoint_latest.pt', checkpoint)
            if improved: trainer.save_checkpoint_payload(output / 'checkpoint_best.pt', checkpoint)
    checkpoint = torch.load(output / 'checkpoint_best.pt', map_location='cpu', weights_only=True)
    if not compact_checkpoints:
        trainer.save_checkpoint_payload(output / f"MSD_fold{run['fold']}/model.pt", checkpoint['model'])
    del checkpoint, model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    save(output / 'completion_manifest.json', {'status':'complete', 'epoch':stopping.epoch,
        'selected_epoch':best_epoch, 'best_val_dice_hard':best, 'run':run, 'early_stopping':stopping.summary(),
        'checkpoint_format': 'inference_only_no_optimizer' if compact_checkpoints else 'resumable',
        'artifacts': {name:digest(output/name) for name in (
            ['checkpoint_best.pt','checkpoint_latest.pt'] if compact_checkpoints else
            [f"MSD_fold{run['fold']}/model.pt",'checkpoint_best.pt','checkpoint_latest.pt'])}})
    return train_loader, val_loader


def calibrate(config, historical, device, output):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from monai.losses import DiceLoss
    from thesis.new_constraints import train_swinunetr_constraints as trainer
    from thesis.new_constraints.objective import NewConstraintConfig, NewConstraintObjective
    from thesis.new_constraints.edge_consistency import BoundaryEdgeConsistencyLoss
    start = time.perf_counter()
    run = config['run']
    checkpoint_path = Path(historical['checkpoint'])
    assert digest(checkpoint_path) == historical['checkpoint_sha256']
    source_config = json.loads(Path(historical['checkpoint_config']).read_text())
    assert digest(historical['checkpoint_config']) == historical['checkpoint_config_sha256']
    assert source_config['runtime_provenance'] == trainer.collect_runtime_provenance()
    assert source_config['source_provenance']['files']['baselines/swin_unetr/swin_unetr.py'] == digest(REPO/'baselines/swin_unetr/swin_unetr.py')
    assert source_config['run']['seed'] == run['seed'] and source_config['run']['constraint_set'] == 'none'
    assert source_config['run']['fold'] == run['fold']
    assert source_config['pkl_sha256'] == config['pkl_sha256']
    assert source_config['splits_json_sha256'] == config['splits_json_sha256']
    assert historical['training_augmentation'] == 'none' and len(historical['training_cases']) == 10
    trainer.seed_everything(run['seed'])
    data_args = data_arguments(config)
    train, validation, _, n, nv = trainer.build_data(data_args, torch.Generator().manual_seed(run['seed']))
    assert (n, nv) == (10, 52)
    names = [item['case_name'] for item in train.dataset.data]
    assert set(names) == set(historical['training_case_ids'])
    assert not set(names) & {item['case_name'] for item in validation.dataset.data}
    model = trainer.build_swinunetr(tuple(run['spatial_size']), 3, device, drop_rate=0, activation_checkpointing=False)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    assert checkpoint['epoch'] == 5
    assert json.loads(json.dumps(checkpoint['run'])) == source_config['run']
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    model.eval()
    base = NewConstraintObjective(NewConstraintConfig(**run['constraint_config']))
    dice, edge = DiceLoss(to_onehot_y=True, softmax=True), BoundaryEdgeConsistencyLoss()
    rows = []
    for batch in DataLoader(train.dataset, batch_size=1, shuffle=False, num_workers=0):
        images, labels = batch['image'].to(device), batch['label'].to(device)
        with torch.no_grad(), torch.autocast('cuda'):
            raw = model(images)
        logits = raw.detach().float().requires_grad_(True)
        base_loss = dice(logits, labels) + base(model, images, logits, labels)['loss']
        edge_loss = edge(logits, labels)
        gb = torch.autograd.grad(base_loss, logits, retain_graph=True)[0]
        ge = torch.autograd.grad(edge_loss, logits)[0]
        assert torch.isfinite(gb).all() and torch.isfinite(ge).all()
        br, er = float(gb.square().mean().sqrt()), float(ge.square().mean().sqrt())
        assert br > 0 and er > 0
        rows.append({'case_name': str(batch['case_name'][0]), 'base_gradient_rms':br,
                     'edge_gradient_rms':er, 'edge_over_base_rms':er/br})
    ratios = [r['edge_over_base_rms'] for r in rows]
    weight = min(.1 / float(np.median(ratios)), .5 / float(np.quantile(ratios, .95)))
    result = {'status':'complete', 'fold':run['fold'], 'seed':run['seed'], 'edge_weight':weight,
              'band_weight':run['constraint_config']['bands_weight'], 'target_ratio':.1, 'cap_ratio':.5,
              'checkpoint':str(checkpoint_path), 'checkpoint_sha256':digest(checkpoint_path),
              'pkl_sha256':config['pkl_sha256'], 'splits_json_sha256':config['splits_json_sha256'],
              'training_case_ids':names, 'cases':rows, 'seconds':time.perf_counter()-start,
              'scope':'All ten training cases only, no augmentation. Existing seed-matched five-epoch Dice checkpoint. No coefficient from the full-data experiment.'}
    save(output, result)
    print(json.dumps({'calibrated_seed':run['seed'], 'edge_weight':weight, 'seconds':result['seconds']}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(0,1,2), required=True)
    parser.add_argument('--fold', type=int, choices=(0,1,2), default=0)
    parser.add_argument('--compact-checkpoints', action='store_true')
    parser.add_argument('--minimum-free-gib', type=float, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root/'run.lock').open('x') as f: f.write(str(os.getenv('SLURM_JOB_ID')))
    try:
        import torch
        from torch.utils.data import DataLoader
        from thesis.new_constraints import train_swinunetr_constraints as trainer
        from train_edge_consistency_experiment import audit
        from audit_edge_generalization import compare_splits
        from edge_lowdata_metrics import compare_histories
        from run_degree_bands_ab import check_quota
        for name, sha in json.loads((REPO/'PAYLOAD.json').read_text())['files'].items():
            assert digest(REPO/name) == sha, name
        check_quota(args.root, 'start', args.minimum_free_gib)
        assert torch.cuda.is_available()
        device = torch.device('cuda')
        runtime, execution = trainer.collect_runtime_provenance(), trainer.collect_execution_provenance(device)
        directories = reference_directories(Path('/mnt/beegfsstudents/home/3160552'), args.fold, args.seed)
        configs, histories, bindings = {}, {}, {}
        for name, directory in directories.items():
            config = json.loads((directory/'config.json').read_text())
            completed = json.loads((directory/'completion_manifest.json').read_text())
            assert completed['status']=='complete' and completed['run']==config['run']
            assert config['run']['seed']==args.seed and config['run']['fold']==args.fold
            assert config['runtime_provenance']==runtime
            trainer.validate_calibration_execution(config['execution_provenance'], execution, allow_mig_profile_change=True)
            assert config['source_provenance']['files']['baselines/swin_unetr/swin_unetr.py']==digest(REPO/'baselines/swin_unetr/swin_unetr.py')
            for key in ('pkl','splits_json'):
                assert digest(config[key])==config[key+'_sha256']
            export_name = f'MSD_fold{args.fold}/model.pt'
            exported = directory/export_name
            assert digest(exported)==completed['artifacts'][export_name]
            checkpoint = torch.load(directory/'checkpoint_best.pt',map_location='cpu',weights_only=True)
            assert checkpoint['epoch']==completed['selected_epoch']
            weights = torch.load(exported,map_location='cpu',weights_only=True)
            assert weights.keys()==checkpoint['model'].keys()
            assert all(torch.equal(v,weights[k]) for k,v in checkpoint['model'].items())
            del checkpoint,weights
            configs[name]=(directory,config)
            histories[name]=list(csv.DictReader((directory/'metrics.csv').open()))
            bindings[name]={'directory':str(directory),'config_sha256':digest(directory/'config.json'),
                            'checkpoint_sha256':digest(directory/'checkpoint_best.pt'), 'metrics_sha256':digest(directory/'metrics.csv')}
        reference = configs['bands'][1]
        for key in ('seed','fold','epochs','batch_size','spatial_size','resize','optimizer_mode','learning_rate',
                    'weight_decay','step_size','adamw_gamma','amp','initial_checkpoint','training_augmentation',
                    'supervised_loss','plain_tensors','drop_rate','activation_checkpointing','early_stopping_patience',
                    'early_stopping_min_delta','early_stopping_min_epochs','splits_json_sha256'):
            assert reference['run'][key]==configs['dice'][1]['run'][key],key
        spec = reference['run']
        assert spec['epochs']==75 and spec['training_augmentation']=='none' and spec['initial_checkpoint'] is None
        assert spec['constraint_set']=='bands' and spec['constraint_config'].get('bands_degree_alpha',0)==0
        assert spec['early_stopping_patience']==8 and spec['early_stopping_min_epochs']==60
        assert spec['supervised_loss']=='dice' and not spec['constraint_scale_knots']
        assert spec['splits_json_sha256']==SPLIT_HASHES[args.fold]
        historical = json.loads(Path(spec['bands_calibration']).read_text())
        assert digest(spec['bands_calibration'])==spec['bands_calibration_sha256']
        calibration = calibrate(reference,historical,device,args.root/'edge_calibration.json')
        torch.cuda.empty_cache()
        source = trainer.collect_source_provenance()
        config = copy.deepcopy(reference)
        config['run'].update(constraint_set='bands_edge', edge_weight=calibration['edge_weight'], calibration_diagnostics=False,
            edge_calibration=str(args.root/'edge_calibration.json'), edge_calibration_sha256=digest(args.root/'edge_calibration.json'),
            source_sha256=source['sha256'], runtime_sha256=trainer.canonical_sha256(runtime), execution_sha256=trainer.canonical_sha256(execution))
        config.update(source_provenance=source,runtime_provenance=runtime,execution_provenance=execution,
                      calibration_diagnostics=None,training_script_sha256=digest(__file__),payload_sha256=digest(REPO/'PAYLOAD.json'))
        output = args.root/'run'
        output.mkdir(exist_ok=False)
        save(output/'config.json',config)
        save(args.root/'reference_bindings.json',bindings)
        start = time.perf_counter()
        train,val = fit(config,output,device,source,compact_checkpoints=args.compact_checkpoints)
        training_seconds = time.perf_counter()-start
        histories['edge']=list(csv.DictReader((output/'metrics.csv').open()))
        save(args.root/'histories.json',histories)
        learning = compare_histories(histories)
        learning.update(fold=args.fold, seed=args.seed, training_including_io_seconds=training_seconds, calibration_seconds=calibration['seconds'])
        save(args.root/'learning_curves.json',learning)
        configs['edge']=(output,config)
        for name,loader in [('train',DataLoader(train.dataset,batch_size=1,shuffle=False)),('validation',val)]:
            audit(args.root/name,configs,device,loader)
        records={name:json.loads((args.root/name/'audit/cases.json').read_text()) for name in ('train','validation')}
        save(args.root/'generalization.json',compare_splits(records['train'],records['validation']))
        save(args.root/'completion.json',{'status':'complete','fold':args.fold,'seed':args.seed,'job_id':os.getenv('SLURM_JOB_ID'),
             'trained_runs':1,'reused_baselines':['dice','bands'],'train_cases':10,'validation_cases':52})
    except BaseException:
        import traceback
        save(args.root/'failure.json',{'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':
    main()
