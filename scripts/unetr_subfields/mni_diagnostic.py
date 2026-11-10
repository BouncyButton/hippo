"""Audited one-subject fitting diagnostic, separate from compute-matched results."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

from scripts.cross_dataset_campaign.common import read, write, digest, install_signals, stopping
from scripts.cross_dataset_campaign.storage import retire, quota


def manifest(output, task):
    metadata, bulky = {}, []
    for path in sorted(output.iterdir()):
        if not path.is_file() or path.is_symlink():
            raise ValueError('Unexpected output entry: ' + str(path))
        if path.name in ('archive_manifest.json', 'archive_acknowledgment.json', 'cleanup.json'):
            continue
        info = dict(bytes=path.stat().st_size, sha256=digest(path))
        if path.suffix in ('.pt', '.tmp'):
            bulky.append(dict(name=path.name, **info))
        elif path.suffix in ('.json', '.csv', '.jsonl', '.log', '.lock'):
            metadata[path.name] = info
        else:
            raise ValueError('Unclassified output: ' + path.name)
    write(output/'archive_manifest.json', dict(status='ready', task=task, metadata=metadata,
          deletion_candidates=bulky, created_at=time.time()))


def archive_and_retire(output, task):
    if (output/'cleanup.json').exists():
        return read(output/'cleanup.json')
    if not (output/'archive_manifest.json').exists():
        manifest(output, task)
    started = time.monotonic()
    while not (output/'archive_acknowledgment.json').exists():
        if stopping() or time.monotonic() - started > 600:
            raise SystemExit(75)
        time.sleep(5)
    return retire(output)


def fit_passes(evaluation, macro_threshold=.95, class_threshold=.90):
    return (evaluation['summary']['macro_hard_dice'] >= macro_threshold and
            all(value['dice'] >= class_threshold for case in evaluation['cases']
                for name, value in case['per_class'].items() if name != 'background'))


def train(stage, identity):
    import torch
    from scripts.unetr_subfields import core, matched_lr_runtime as runtime
    from scripts.unetr_subfields.contracts import DATASETS
    from scripts.capped_depth_delay.objectives import training_loss as native_loss
    from scripts.compute_constraint_objectives import training_loss as dice_loss
    torch.set_num_threads(4)
    assert str(torch.__version__).startswith('2.12.1')
    device = torch.device('cuda')
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    config = read(stage/'diagnostic.json')
    output = stage/'runs'/identity
    output.mkdir(parents=True, exist_ok=True)
    task = dict(id=identity, kind='one_subject_diagnostic', model=identity, dataset='MNI')
    if (output/'completion.json').exists():
        archive_and_retire(output, task)
        return
    protocol = read(stage/'protocols'/identity/'protocol.json')
    profile = read(stage/'protocols'/identity/'profile.json')
    core.base.seed_everything(config['seed'])
    model = runtime.build_model(identity, read(stage/'protocols'/identity/'nnunet_architecture.json')).to(device)
    assert sum(p.numel() for p in model.parameters()) == profile['parameters']
    criterion, optimizer = runtime.conventional.ORIGINAL_LOSS_OPTIMIZER(identity, model, protocol)
    frame = core.base._load_pkl_dataframe(stage/'data/MNI.pkl')
    assert digest(stage/'data/MNI.pkl') == protocol['pkl_sha256']
    spec = DATASETS['MNI']
    training = core.PreparedSubfields(frame, config['train_cases'], spec)
    validation = core.PreparedSubfields(frame, config['validation_cases'], spec)
    assert len(training) == 2 and len({b['participant_id'] for b in training}) == 1
    assert not set(config['train_cases']) & set(config['validation_cases'])
    audit = []
    for item in training:
        counts = torch.bincount(item['label'].flatten(), minlength=4).tolist()
        assert all(counts) and torch.isfinite(item['image']).all() and item['image'].std() > 0
        audit.append(dict(case=item['case_name'], participant=item['participant_id'], label_counts=counts,
                          shape=list(item['image'].shape)))
    cost = profile['update_flops']
    settings = dict(**config, model=identity, optimizer=str(type(optimizer).__name__),
                    optimizer_defaults=optimizer.defaults, initial_lr=protocol['initial_lrs'][identity],
                    loss='native Dice+CE with deep supervision' if identity=='nnunet' else 'MONAI Dice only, including background',
                    dataset_sha256=digest(stage/'data/MNI.pkl'), source_manifest_sha256=digest(stage/'manifest.json'),
                    update_flops=cost, data_audit=audit, torch=torch.__version__,
                    precision='BF16 autocast; FP32 parameters', ap_swaps_applicable=False)
    settings = json.loads(json.dumps(settings))
    if (output/'config.json').exists():
        assert read(output/'config.json') == settings
    write(output/'config.json', settings)
    step, consecutive, curve, losses = 0, 0, [], []
    checkpoint = output/'checkpoint_latest.pt'
    if checkpoint.exists():
        cp = torch.load(checkpoint, map_location=device, weights_only=False)
        assert cp['config_sha256'] == digest(output/'config.json')
        model.load_state_dict(cp['model']); optimizer.load_state_dict(cp['optimizer'])
        step, consecutive, curve, losses = cp['step'], cp['consecutive'], cp['curve'], cp['losses']
        torch.set_rng_state(cp['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([r.cpu() for r in cp['cuda_rng']])
        del cp
    started = time.monotonic()

    def save():
        core.base.save_checkpoint(checkpoint, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            step=step, consecutive=consecutive, curve=curve, losses=losses,
            config_sha256=digest(output/'config.json'), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all()))

    def evaluate_fit():
        nonlocal consecutive, losses
        result = core.evaluate(model, training, device, True, spec['classes'])
        passed = fit_passes(result, config['macro_threshold'], config['class_threshold'])
        consecutive = consecutive+1 if passed else 0
        row = dict(step=step, epochs=step/2, training_flops=step*cost,
                   mean_loss=sum(losses)/len(losses) if losses else None,
                   learning_rate=optimizer.param_groups[0]['lr'], passes=passed,
                   consecutive_passes=consecutive, seconds=time.monotonic()-started, **result['summary'])
        curve.append(row); losses=[]
        write(output/'training_curve.json', curve)
        write(output/'training_latest.json', result)
        write(output/'progress.json', row)
        print(json.dumps(dict(model=identity, **row)), flush=True)
        save()
        return result

    if not curve:
        evaluate_fit()
    check_steps = set(range(50, config['max_updates']+1, 50)) | {config['max_updates'], config['historical_updates'][identity]}
    while step < config['max_updates'] and consecutive < 2:
        if stopping():
            save(); raise SystemExit(75)
        # Exactly one visit to each hemisphere per two-update epoch.
        permutation = torch.randperm(2, generator=torch.Generator().manual_seed(config['seed']+1000+step//2)).tolist()
        item = training[permutation[step % 2]]
        image, label = item['image'][None].to(device), item['label'][None].to(device)
        optimizer.zero_grad(set_to_none=True); model.train()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(image)
            loss = (native_loss(logits, label, 'dice', {}, .2, criterion=criterion) if identity=='nnunet'
                    else dice_loss(logits, label, 'dice', {}, .2))
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        loss.backward()
        if identity=='nnunet':
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 12, foreach=False, error_if_nonfinite=True)
        else:
            norm = torch.linalg.vector_norm(torch.stack([p.grad.float().norm() for p in model.parameters() if p.grad is not None]))
            if not torch.isfinite(norm): raise FloatingPointError('Nonfinite gradients')
        if step == 0:
            probe = next(p for p in model.parameters() if p.grad is not None and p.grad.abs().max() > 0)
            before = probe.detach().clone()
        optimizer.step(); step += 1; losses.append(float(loss.detach()))
        if step == 1:
            changed = bool((before != probe.detach()).any())
            assert changed and float(norm) > 0
            write(output/'gradient_sanity.json', dict(first_loss=float(loss.detach()), gradient_norm=float(norm), parameter_changed=changed))
            del before
        if step in check_steps: evaluate_fit()
        elif step % 25 == 0: save()
    final = core.evaluate(model, training, device, True, spec['classes'])
    heldout = core.evaluate(model, validation, device, True, spec['classes'])
    write(output/'training_final.json', final); write(output/'validation_final.json', heldout)
    save()
    write(output/'completion.json', dict(status='complete', model=identity, updates=step, epochs=step/2,
        overfit_passed=consecutive >= 2, stopping_reason='two_consecutive_training_fit_checks' if consecutive>=2 else 'diagnostic_update_limit',
        training_macro_dice=final['summary']['macro_hard_dice'], validation_macro_dice=heldout['summary']['macro_hard_dice'],
        consumed_training_flops=step*cost, historical_cap=config['historical_cap'],
        checkpoint_sha256=digest(checkpoint), config_sha256=digest(output/'config.json'),
        heldout_used_for_stopping=False, excluded_from_compute_matched_baselines=True))
    del model, optimizer; torch.cuda.empty_cache()
    archive_and_retire(output, task)


def retire_interrupted(stage):
    import torch
    from scripts.unetr_subfields import core, matched_lr_runtime as runtime
    from scripts.unetr_subfields.contracts import DATASETS
    plan=read(stage/'plan.json'); item=plan['retirement']; output=stage/'runs'/item['output']
    task=dict(id=output.name, kind='abandoned', dataset='MNI', model='nnunet')
    if (output/'cleanup.json').exists(): return
    if not (output/'archive_manifest.json').exists():
        torch.set_num_threads(4)
        parent=Path(item['parent_stage']); folder=parent/'protocols/nnunet'
        protocol=read(folder/'protocol.json'); config=read(output/'config.json'); ledger=read(output/'budget_ledger.json')
        assert digest(folder/'protocol.json') == config['protocol_sha256']
        cp=torch.load(output/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
        assert cp['configuration']==config and cp['charged_attempts']==ledger['charged_attempts']
        model=runtime.build_model('nnunet',read(folder/'nnunet_architecture.json')).cuda()
        frame=core.base._load_pkl_dataframe(Path(protocol['data']));spec=DATASETS['MNI']
        model.load_state_dict(cp['model'])
        for label,names in [('training_interrupted',config['train_cases']),('validation_interrupted',config['validation_cases'])]:
            write(output/(label+'.json'),core.evaluate(model,core.PreparedSubfields(frame,names,spec),torch.device('cuda'),True,spec['classes']))
        best=torch.load(output/'checkpoint_best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(best['model'])
        write(output/'validation_best_poststop.json',core.evaluate(model,core.PreparedSubfields(frame,config['validation_cases'],spec),torch.device('cuda'),True,spec['classes']))
        write(output/'interrupted_protocol.json',protocol)
        write(output/'abandonment.json',dict(status='abandoned_by_user',original_run=item['original_run'],
            original_parent=str(parent), original_manifest_sha256=digest(parent/'manifest.json'),
            updates=cp['step'],consumed_training_flops=ledger['training_flops'],additional_optimizer_updates=0,
            reason='User stopped sweep to diagnose one-subject fit before establishing a new common cap.',
            excluded_from_completed_baselines=True,ap_swaps_applicable=False))
        del model,cp,best;torch.cuda.empty_cache()
    archive_and_retire(output,task)


def run(stage):
    import subprocess
    import sys
    for name,sha in read(stage/'manifest.json').items():
        assert digest(stage/name)==sha, name
    plan=read(stage/'plan.json')
    assert quota()['free_gib']>=plan['storage']['minimum_free_gib']
    os.environ['HIPPO_CONTROL']=str(stage/'control.json');install_signals()
    write(stage/'status.json',dict(status='evaluating_interrupted_run',completed=0,total=2))
    retire_interrupted(stage)
    devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    assert len(devices)==2, devices
    children=[]
    for device,identity in zip(devices,('nnunet','swinunetr')):
        child=subprocess.Popen([sys.executable,'-u','-m','scripts.unetr_subfields.mni_diagnostic','train','--stage',str(stage),'--model',identity],
            env={**os.environ,'CUDA_VISIBLE_DEVICES':device})
        children.append((identity,child))
    while any(child.poll() is None for _,child in children):
        if stopping():
            import signal
            for _,child in children:
                if child.poll() is None: child.send_signal(signal.SIGUSR1)
        write(stage/'status.json',dict(status='running',active=[name for name,c in children if c.poll() is None],
              completed=sum((stage/'runs'/name/'cleanup.json').exists() for name,_ in children),total=2))
        time.sleep(5)
    codes={name:child.returncode for name,child in children}
    state='complete' if all(code==0 for code in codes.values()) else ('paused' if stopping() or 75 in codes.values() else 'failed')
    write(stage/'status.json',dict(status=state,returncodes=codes,completed=sum(code==0 for code in codes.values()),total=2,quota=quota()))
    if state!='complete': raise SystemExit(75 if state=='paused' else 1)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['run','train'])
    parser.add_argument('--stage',type=Path,required=True);parser.add_argument('--model',choices=['nnunet','swinunetr'])
    args=parser.parse_args();install_signals()
    try:
        if args.command=='run': run(args.stage)
        else: train(args.stage,args.model)
    except Exception:
        write(args.stage/('failure_'+str(args.model)+'.json'),dict(traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
