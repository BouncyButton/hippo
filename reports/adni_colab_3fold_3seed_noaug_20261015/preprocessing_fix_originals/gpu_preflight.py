"""Diagnostic-only checks of the unchanged binary ADNI trainer on the allocated GPU."""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from runtime import (activate, child_environment, install_initialization, model_digest,
                     read, tensor_digest, verify_bundle, write)


def fingerprint(value):
    """Include every parameter, optimizer tensor, scalar and RNG byte."""
    import torch
    h = hashlib.sha256()
    def visit(item):
        h.update(type(item).__name__.encode())
        if isinstance(item, torch.Tensor):
            h.update(str((item.dtype, tuple(item.shape))).encode())
            h.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(key); visit(item[key])
        elif isinstance(item, (tuple, list)):
            for member in item:
                visit(member)
        else:
            h.update(repr(item).encode())
    visit(value)
    return h.hexdigest()


def model_check(root, output, name):
    import numpy as np
    import torch
    from scripts.cross_dataset_campaign import adni_mixed_losses as scientific
    exp = scientific.experiment
    folder = root/'protocols'/f'{name}_f0_s0'
    p = read(folder/'protocol.json')
    arm = 'dice_bands_edge_sdf_depth'
    exp.configure(p, arm)
    scientific.binary.bind()
    exp.base.build_loss_and_optimizer = exp.ORIGINAL_LOSS_OPTIMIZER
    exp.trainer.training_loss = partial(scientific.training_loss, policy=p['sdf_policy'][arm])
    exp.trainer.calibration_case = scientific.calibration_case
    install_initialization(root, exp)
    torch.set_num_threads(2)
    exp.base.seed_everything(0)
    device = torch.device('cuda')
    model = exp.build_model(p, read(folder/'nnunet_architecture.json')).to(device)
    if sum(v.numel() for v in model.parameters()) != p['models'][p['trainer_alias']]['parameters']:
        raise ValueError('Profiled model parameter count differs')
    criterion, optimizer = exp.ORIGINAL_LOSS_OPTIMIZER(p['trainer_alias'], model, p)
    native = criterion if name == 'nnunet' else None
    dataframe = exp.base._load_pkl_dataframe(root/'data/adni_hippocampus_full.pkl')
    split = read(folder/'splits_10patients.json')[0]
    train = exp.adni.PreparedADNI(dataframe, split['train'])
    validation = exp.adni.PreparedADNI(dataframe, split['val'][:2])
    before = (tensor_digest(torch.get_rng_state()), [tensor_digest(x) for x in torch.cuda.get_rng_state_all()])
    calibration = exp.trainer.calibrate(model, train, arm, device, True, native, p)
    after = (tensor_digest(torch.get_rng_state()), [tensor_digest(x) for x in torch.cuda.get_rng_state_all()])
    if before != after or [x['case_name'] for x in calibration['cases']] != split['train']:
        raise ValueError('Calibration changed RNG or the ten-crop training cohort')
    if calibration['target_ratio'] != p['calibration_target_ratio'] or calibration['p95_cap'] != p['calibration_p95_cap']:
        raise ValueError('Calibration target changed')
    weights = calibration['weights']
    schedule = p['models'][p['trainer_alias']]['objectives'][arm]
    # Exercise every auxiliary term after its original activation point. These
    # diagnostic updates and initial-model calibration never enter study ledgers.
    charged = max(schedule['prefix_updates'], schedule['sdf_prefix_updates'])
    def update(index):
        batch = train[index]
        host_label = batch['label'][None]
        affine = np.asarray(p['native_geometry'][batch['case_name']]['model_affine'])
        depth = scientific.binary.geometry_weights(host_label, affine, p['depth_geometry']).to(device)
        sdf = scientific.binary.signed_distance_maps(host_label).to(device)
        image, label = batch['image'][None].to(device), host_label.to(device)
        lr = exp.base.learning_rate(p['trainer_alias'], charged+index, schedule['max_steps'],
                                    p['initial_lrs'][p['trainer_alias']])
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        model.train()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(image)
            loss = exp.trainer.training_loss(logits, label, arm, weights, 1., sdf, native, depth, 1.)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite diagnostic loss')
        loss.backward()
        if any(not v.grad.isfinite().all() for v in model.parameters() if v.grad is not None):
            raise ValueError('Nonfinite parameter gradient')
        if native is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12, foreach=False, error_if_nonfinite=True)
        optimizer.step()
        return float(loss.detach())

    def state():
        return dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())

    first = update(0)
    with tempfile.TemporaryDirectory(prefix='adni-gpu-preflight-', dir=root) as tmp:
        checkpoint = Path(tmp)/'diagnostic.pt'
        exp.base.save_checkpoint(checkpoint, state())
        second = update(1)
        expected = fingerprint(state())
        evaluation = exp.evaluate(model, validation, device, True, p['classes'])
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
        del saved
        resumed = update(1)
        if second != resumed or fingerprint(state()) != expected:
            raise ValueError('GPU checkpoint resume changed loss, model, optimizer or RNG')
        if exp.evaluate(model, validation, device, True, p['classes']) != evaluation:
            raise ValueError('Resumed evaluation differs')
    if any('foreground_fn' not in c or 'foreground_fp' not in c or len(c['confusion']) != 2
           for c in evaluation['cases']):
        raise ValueError('Binary FN/FP evaluation missing')
    write(output/f'{name}.json', dict(status='passed', dataset='ADNI', model=name,
        diagnostic_only=True, augmentation='none', precision='BF16', loss_first=first,
        loss_second=second, exact_resume=True, final_state_sha256=expected,
        calibration=calibration, evaluation=evaluation))
    print('GPU loss, calibration, optimizer, evaluation and resume passed:', name, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model')
    args = parser.parse_args()
    root = activate(args.bundle)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.model:
        model_check(root, args.output, args.model)
        return
    plan = verify_bundle(root)
    from scripts.cross_dataset_campaign.preflight_adni_mixed_losses import check
    with tempfile.TemporaryDirectory(prefix='adni-graph-preflight-', dir=root) as tmp:
        stage = Path(tmp)
        for name in plan['models']:
            shutil.copytree(root/'protocols'/f'{name}_f0_s0', stage/'protocols'/name)
            pfile = stage/'protocols'/name/'protocol.json'
            p = read(pfile); p['data'] = str(root/'data/adni_hippocampus_full.pkl'); write(pfile, p)
        shutil.copyfile(root/'pilot_run.sbatch', stage/'run.sbatch')
        check(stage, plan, verify_data=True)
    write(args.output/'binary_graphs.json', dict(status='passed', models=plan['models'],
        arms=plan['arms'], description='Original binary losses, deep supervision, sequential targets and SDF timing'))
    from scripts.adni_baselines.core import PreparedADNI, case_name, base
    frame = base._load_pkl_dataframe(root/'data/adni_hippocampus_full.pkl')
    mapping = {case_name(row): str(row.subject_id) for row in frame.itertuples()}
    if mapping != read(root/'split_provenance.json')['case_to_subject']:
        raise ValueError('Dataset subjects differ from split provenance')
    splits = read(root/'protocols/nnunet_f0_s0/splits_10patients.json')
    names = sorted({c for s in splits for c in s['train']+s['val']})
    dataset = PreparedADNI(frame, names)
    if fingerprint(dataset[0]) != fingerprint(PreparedADNI(frame, names[:1])[0]):
        raise ValueError('Unaugmented preprocessing is not reproducible')
    write(args.output/'data.json', dict(status='passed', crops_checked=len(dataset), folds=3,
        train_subjects_per_fold=5, validation_subjects_per_fold=27, augmentation='none'))
    del dataset, frame
    for name in plan['models']:
        subprocess.run([sys.executable, str(root/'gpu_preflight.py'), '--bundle', str(root),
                        '--output', str(args.output), '--model', name],
                       cwd=root/'source', env=child_environment(root), check=True)
    write(args.output/'summary.json', dict(status='passed', models=plan['models'],
        diagnostic_only=True, dataset='ADNI', augmentation='none',
        scientific_source='unchanged frozen ADNI trainer; runtime acceleration disabled'))


if __name__ == '__main__':
    main()
