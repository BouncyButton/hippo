#!/usr/bin/env python3
"""One matched Dice+bands+edge run, reusing the established training utilities."""
from __future__ import annotations
import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
from probe_edge_consistency import digest, save


def experiment_run(reference, probe, probe_path, source_sha, runtime_sha, execution_sha):
    """Retain the matched run specification and expose every added setting."""
    if not probe['gate']['passed']:
        raise ValueError('The fixed training-only gate did not pass.')
    run = copy.deepcopy(reference['run'])
    if run['constraint_set'] != 'bands' or run['constraint_config'].get('bands_degree_alpha', 0) != 0:
        raise ValueError('Expected the ordinary-bands reference.')
    if run['constraint_config']['bands_weight'] != probe['calibration']['band_weight']:
        raise ValueError('The base band coefficient changed.')
    if not math.isfinite(probe['calibration']['edge_weight']) or probe['calibration']['edge_weight'] <= 0:
        raise ValueError('The calibrated edge coefficient must be finite and positive.')
    run.update(constraint_set='bands_edge', edge_weight=probe['calibration']['edge_weight'],
               calibration_diagnostics=False,
               edge_calibration=str(probe_path), edge_calibration_sha256=digest(probe_path),
               source_sha256=source_sha, runtime_sha256=runtime_sha, execution_sha256=execution_sha)
    return run


def audit(root, configs, device, loader):
    """Score all selected checkpoints identically; retain only counts on disk."""
    import numpy as np
    import torch
    from thesis.new_constraints import train_swinunetr_constraints as trainer
    from audit_degree_boundary import supports, score_case, transitions, summarize, summarize_pair
    predictions, truths, names, records = {}, [], [], {}
    manifests = {}
    for name, (directory, config) in configs.items():
        run = config['run']
        checkpoint_path = directory / 'checkpoint_best.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        assert json.loads(json.dumps(checkpoint['run'])) == run
        model = trainer.build_swinunetr(tuple(run['spatial_size']), 3, device, drop_rate=0, activation_checkpointing=False)
        model.load_state_dict(checkpoint['model'], strict=True)
        manifests[name] = {'checkpoint_sha256': digest(checkpoint_path), 'selected_epoch': checkpoint['epoch']}
        del checkpoint
        model.eval()
        predictions[name] = []
        with torch.no_grad():
            for index, batch in enumerate(loader):
                truth = batch['label'][0, 0].numpy().astype(np.uint8)
                case_name = str(batch['case_name'][0])
                if name == 'dice':
                    truths.append(truth); names.append(case_name)
                else:
                    assert names[index] == case_name and np.array_equal(truths[index], truth)
                with torch.autocast('cuda', enabled=True):
                    output = model(batch['image'].to(device))
                predictions[name].append(output[0].float().softmax(0).argmax(0).cpu().numpy().astype(np.uint8))
        del model
        torch.cuda.empty_cache()
    pairs = {'edge_vs_dice': [], 'edge_vs_bands': []}
    records = {name: [] for name in configs}
    for index, truth in enumerate(truths):
        masks = supports(truth)
        states = {}
        for name in configs:
            row, states[name] = score_case(truth, predictions[name][index], masks)
            row['case_name'] = names[index]
            records[name].append(row)
        for reference in ('dice', 'bands'):
            pairs['edge_vs_' + reference].append({'case_name': names[index], 'regions': {
                key: transitions(states[reference], states['edge'], mask) for key, mask in masks.items()}})
    summary = {name: summarize(rows) for name, rows in records.items()}
    paired = {'edge_vs_' + name: summarize_pair(records[name], records['edge'], pairs['edge_vs_' + name])
              for name in ('dice', 'bands')}
    save(root / 'audit/cases.json', records)
    save(root / 'audit/paired_cases.json', pairs)
    save(root / 'audit/summary.json', summary)
    save(root / 'audit/comparison.json', paired)
    save(root / 'audit/checkpoints.json', manifests)
    print(json.dumps({'audit_complete': True, 'macro_dice': {name: s['mean_case_metrics']['macro_dice']['mean'] for name, s in summary.items()}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--reference-bands', type=Path, required=True)
    parser.add_argument('--reference-dice', type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / 'training.lock').open('x') as f: f.write(str(os.getenv('SLURM_JOB_ID')))
    try:
        import torch
        from monai.losses import DiceLoss
        from thesis.new_constraints import train_swinunetr_constraints as trainer
        from thesis.new_constraints.objective import NewConstraintConfig, NewConstraintObjective
        from thesis.new_constraints.edge_consistency import BoundaryEdgeConsistencyLoss
        from thesis.new_constraints.early_stopping import EarlyStopping
        from thesis.new_constraints.training_augmentation import augment_mild
        from run_degree_bands_ab import check_quota
        payload = json.loads((REPO / 'PAYLOAD.json').read_text())
        for name, sha in payload['files'].items(): assert digest(REPO / name) == sha, name
        assert digest(args.probe) == payload['edge_probe_sha256']
        probe = json.loads(args.probe.read_text())
        assert probe['status'] == 'complete' and probe['gate']['passed']
        check_quota(args.root, 'training_start', 2.0)
        trainer.seed_everything(0)
        device = torch.device('cuda')
        assert torch.cuda.is_available()
        runtime, execution = trainer.collect_runtime_provenance(), trainer.collect_execution_provenance(device)
        assert runtime == probe['runtime'] and execution == probe['execution']
        reference = json.loads((args.reference_bands / 'config.json').read_text())
        dice_config = json.loads((args.reference_dice / 'config.json').read_text())
        assert digest(args.reference_bands / 'checkpoint_best.pt') == probe['converged_checkpoint_sha256']
        for directory, config in [(args.reference_dice, dice_config), (args.reference_bands, reference)]:
            completion = json.loads((directory / 'completion_manifest.json').read_text())
            assert completion['status'] == 'complete' and completion['run'] == config['run']
            assert digest(directory / 'MSD_fold0/model.pt') == completion['artifacts']['MSD_fold0/model.pt']
            assert config['runtime_provenance'] == runtime
            trainer.validate_calibration_execution(config['execution_provenance'], execution, allow_mig_profile_change=True)
            assert config['source_provenance']['files']['baselines/swin_unetr/swin_unetr.py'] == digest(REPO / 'baselines/swin_unetr/swin_unetr.py')
            for path, sha in [('pkl', 'pkl_sha256'), ('splits_json', 'splits_json_sha256')]:
                assert digest(config[path]) == config[sha] == reference[sha]
        assert reference['run']['fold'] == reference['run']['seed'] == 0
        assert reference['run']['epochs'] == 50 and reference['run']['constraint_warmup_epochs'] == 5
        for key in ('seed', 'fold', 'epochs', 'batch_size', 'spatial_size', 'resize', 'optimizer_mode',
                    'learning_rate', 'weight_decay', 'step_size', 'adamw_gamma', 'amp', 'initial_checkpoint',
                    'training_augmentation', 'supervised_loss', 'ce_weight', 'plain_tensors', 'drop_rate',
                    'activation_checkpointing', 'early_stopping_patience', 'early_stopping_min_delta', 'early_stopping_min_epochs'):
            assert reference['run'][key] == dice_config['run'][key], key
        assert reference['run']['initial_checkpoint'] is None and not reference['run']['constraint_scale_knots']
        assert reference['run']['supervised_loss'] == 'dice' and reference['run']['training_augmentation'] == 'mild_v1'
        source = trainer.collect_source_provenance()
        run = experiment_run(reference, probe, args.probe, source['sha256'],
                             trainer.canonical_sha256(runtime), trainer.canonical_sha256(execution))
        output = args.root / 'run'
        output.mkdir(exist_ok=False)
        config = copy.deepcopy(reference)
        config.update(run=run, source_provenance=source, runtime_provenance=runtime, execution_provenance=execution,
                      calibration_diagnostics=None,
                      training_script_sha256=digest(__file__), payload_sha256=digest(REPO / 'PAYLOAD.json'),
                      experiment_note='New edge term; original bands coefficient unchanged. Frozen common model/data/optimizer/evaluation utilities.')
        save(output / 'config.json', config)
        data_generator, augmentation_generator, _ = trainer.build_experiment_generators(run['seed'])
        data_args = SimpleNamespace(pkl=Path(config['pkl']), splits_json=Path(config['splits_json']), dataset='MSD',
            fold=0, spatial_size=run['spatial_size'], resize=run['resize'], batch_size=1, num_workers=0,
            num_classes=3, plain_tensors=True)
        train_loader, val_loader, _, train_n, val_n = trainer.build_data(data_args, data_generator)
        assert (train_n, val_n) == (208, 52)
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
                    images, labels, _ = augment_mild(batch['image'].to(device), batch['label'].to(device), generator=augmentation_generator)
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
                print(json.dumps(row), flush=True)
                scheduler.step()
                improved = metrics['val_dice_hard'] > best
                if improved: best, best_epoch = metrics['val_dice_hard'], epoch
                stopping.update(epoch, metrics['val_dice_hard'])
                trainer.validate_source_provenance_unchanged(source)
                checkpoint = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'best_hard_dice': best,
                    'best_epoch': best_epoch, 'run': run, 'rng': trainer.rng_state(data_generator, augmentation_generator),
                    'early_stopping': stopping.summary()}
                trainer.save_checkpoint_payload(output / 'checkpoint_latest.pt', checkpoint)
                if improved: trainer.save_checkpoint_payload(output / 'checkpoint_best.pt', checkpoint)
        checkpoint = torch.load(output / 'checkpoint_best.pt', map_location='cpu', weights_only=True)
        trainer.save_checkpoint_payload(output / 'MSD_fold0/model.pt', checkpoint['model'])
        del checkpoint, model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()
        save(output / 'completion_manifest.json', {'status':'complete', 'epoch':stopping.epoch,
            'selected_epoch':best_epoch, 'best_val_dice_hard':best, 'run':run, 'early_stopping':stopping.summary(),
            'artifacts': {name:digest(output/name) for name in ['MSD_fold0/model.pt','checkpoint_best.pt','checkpoint_latest.pt']}})
        audit(args.root, {'dice':(args.reference_dice,dice_config), 'bands':(args.reference_bands,reference), 'edge':(output,config)}, device, val_loader)
        save(args.root / 'completion.json', {'status':'complete','job_id':os.getenv('SLURM_JOB_ID'),
            'trained_runs':1,'reused_baselines':['dice','bands'],'selected_epoch':best_epoch,'stopped_epoch':stopping.epoch})
    except BaseException:
        import traceback
        save(args.root / 'failure.json', {'traceback':traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
