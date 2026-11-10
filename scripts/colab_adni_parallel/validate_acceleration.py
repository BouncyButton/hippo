"""Bounded ADNI equivalence checks; never starts a campaign or writes its results."""
import argparse
from contextlib import nullcontext
import fcntl
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def model_check(root, output, name, accelerated):
    """Original ADNI preflight coverage, compared across isolated processes."""
    import numpy as np
    import random
    import torch
    from runtime import install_initialization, tensor_digest
    from gpu_preflight import fingerprint
    from scripts.cross_dataset_campaign import adni_mixed_losses as scientific
    from acceleration_candidate import RuntimeAcceleration

    exp = scientific.experiment
    folder = root/'protocols'/f'{name}_f0_s0'
    p = read(folder/'protocol.json')
    arm = 'dice_bands_edge_sdf_depth'
    if p['dataset'] != 'ADNI' or p['augmentation'] != 'none' or len(p['classes']) != 2:
        raise ValueError('Expected frozen unaugmented binary ADNI')
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
        raise ValueError('Model parameter count changed')
    criterion, optimizer = exp.ORIGINAL_LOSS_OPTIMIZER(p['trainer_alias'], model, p)
    native = criterion if name == 'nnunet' else None
    accelerator = RuntimeAcceleration(exp, p, device) if accelerated else None
    context = accelerator if accelerated else nullcontext()
    print('CHECK:', name, 'candidate' if accelerated else 'original', flush=True)

    with context:
        dataframe = exp.base._load_pkl_dataframe(root/'data/adni_hippocampus_full.pkl')
        split = read(folder/'splits_10patients.json')[0]
        train = exp.base.PreparedPatients(dataframe, split['train'])
        validation = exp.base.PreparedPatients(dataframe, split['val'])
        # Same independent raw load per subprocess; no double-normalization.
        data_hashes = [fingerprint(item) for item in [*train, *validation]]
        def rng():
            numpy_state = np.random.get_state()
            return dict(torch=tensor_digest(torch.get_rng_state()),
                cuda=[tensor_digest(x) for x in torch.cuda.get_rng_state_all()],
                numpy=[numpy_state[0], hashlib.sha256(numpy_state[1].tobytes()).hexdigest(),
                       *numpy_state[2:]], python=fingerprint(random.getstate()))
        before = rng()
        print('PHASE: unchanged ten-crop scientific loss calibration', name, flush=True)
        calibration = exp.trainer.calibrate(model, train, arm, device, True, native, p)
        if before != rng() or [x['case_name'] for x in calibration['cases']] != split['train']:
            raise ValueError('Calibration altered RNG or training crop order')
        if (calibration['target_ratio'] != p['calibration_target_ratio']
                or calibration['p95_cap'] != p['calibration_p95_cap']):
            raise ValueError('Calibration target changed')
        weights = calibration['weights']
        schedule = p['models'][p['trainer_alias']]['objectives'][arm]
        charged = max(schedule['prefix_updates'], schedule['sdf_prefix_updates'])
        annotations, gradients = [], []

        def update(index):
            batch = train[index]
            host_label = batch['label'][None]
            affine = np.asarray(p['native_geometry'][batch['case_name']]['model_affine'])
            depth = exp.trainer.geometry_weights(host_label, affine, p['depth_geometry']).to(device)
            sdf = exp.trainer.signed_distance_maps(host_label).to(device)
            annotations.append(dict(case=batch['case_name'], depth=tensor_digest(depth), sdf=tensor_digest(sdf)))
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
                raise ValueError('Nonfinite diagnostic gradients')
            raw = fingerprint({k:v.grad for k,v in model.named_parameters()})
            if native is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 12, foreach=False, error_if_nonfinite=True)
            gradients.append(dict(raw=raw, clipped=fingerprint({k:v.grad for k,v in model.named_parameters()})))
            optimizer.step()
            return float(loss.detach())

        def state():
            return dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())

        print('PHASE: three disposable diagnostic updates including checkpoint replay', name, flush=True)
        first = update(0)
        with tempfile.TemporaryDirectory(prefix='probe_state_', dir=output) as temporary:
            checkpoint = Path(temporary)/'diagnostic.pt'
            exp.base.save_checkpoint(checkpoint, state())
            checkpoint_hash = fingerprint(torch.load(checkpoint, map_location='cpu', weights_only=False))
            best = dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, step=1)
            best_file = Path(temporary)/'checkpoint_best.pt'
            exp.base.save_checkpoint(best_file, best)
            first_bytes = digest(best_file)
            exp.base.save_checkpoint(best_file, best)
            if digest(best_file) != first_bytes:
                raise ValueError('Unchanged best checkpoint bytes changed')
            if fingerprint(torch.load(best_file, map_location='cpu', weights_only=False)) != fingerprint(best):
                raise ValueError('Best checkpoint contents changed')
            del best
            second = update(1)
            expected, expected_rng = fingerprint(state()), rng()
            print('PHASE: binary evaluation, original case order and batch size one', name, flush=True)
            subset = torch.utils.data.Subset(validation, [0, 1])
            evaluation = exp.base.evaluate(model, subset, device, True)
            evaluation_rng = rng()
            saved = torch.load(checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
            torch.set_rng_state(saved['torch_rng'].cpu())
            torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
            del saved
            resumed = update(1)
            if second != resumed or fingerprint(state()) != expected or rng() != expected_rng:
                raise ValueError('Checkpoint replay changed loss/state/RNG')
            if (exp.base.evaluate(model, subset, device, True) != evaluation
                    or rng() != evaluation_rng or gradients[1] != gradients[2]):
                raise ValueError('Resumed gradients, evaluation or iterator RNG differ')
        if any('foreground_fn' not in c or 'foreground_fp' not in c or len(c['confusion']) != 2
               for c in evaluation['cases']):
            raise ValueError('Expected binary FN/FP metrics')
        if data_hashes != [fingerprint(item) for item in [*train, *validation]]:
            raise ValueError('Input or annotation tensors changed')
        if accelerator is not None:
            if not all(accelerator.stats[k] > 0 for k in ('sdf_hits','depth_hits','skipped_best_checkpoint_writes')):
                raise ValueError('Candidate path was not exercised')
        scientific_result = dict(data_hashes=data_hashes, losses=[first,second,resumed],
            gradients=gradients, annotations=annotations, calibration=calibration,
            checkpoint_state_sha256=checkpoint_hash, final_state_sha256=expected,
            evaluation=evaluation, evaluation_rng=evaluation_rng,
            protocol_sha256=digest(folder/'protocol.json'), schedule=schedule,
            diagnostic_updates=3, scientific_updates=0, augmentation='none')
        write(output/(name+('_candidate.json' if accelerated else '_original.json')),
              dict(status='passed_path_checks', scientific=scientific_result,
                   acceleration=accelerator.stats if accelerator else None,
                   peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        print('PATH CHECK PASSED:', name, 'candidate' if accelerated else 'original', flush=True)


def compare(original, candidate):
    """No tolerance relaxation: all scientific outputs must match exactly."""
    a, b = original['scientific'], candidate['scientific']
    different = [key for key in a.keys() | b.keys() if a.get(key) != b.get(key)]
    if different:
        raise ValueError('Original/candidate equivalence failed: '+', '.join(sorted(different)))


def assert_idle(root):
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        if path.parent.name == str(os.getpid()):
            continue
        try:
            words = path.read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if (str(root/'entry.py') in words
                or (str(root) in words and any(w.endswith('/worker.py') or w.endswith('/scheduler.py') for w in words))):
            raise RuntimeError('An ADNI campaign is active. Wait for its clean stop and recovery uploads; this validator will not interrupt it.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--results', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model')
    parser.add_argument('--candidate', action='store_true')
    args = parser.parse_args()
    root = args.bundle.resolve()
    sys.path.insert(0, str(root))
    import runtime
    runtime.activate(root)
    if args.model:
        args.output.mkdir(parents=True, exist_ok=True)
        model_check(root, args.output, args.model, args.candidate)
        return
    with (root/'work/sweep.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('The ADNI campaign is active; no validation was started.')
        assert_idle(root)
        if args.results is None:
            raise ValueError('Existing results path is required for read-only environment checks')
        if (args.results/'.adni_controller_lock').exists():
            raise RuntimeError('The ADNI controller ownership marker is present. Wait for a clean stop; never delete it to bypass this check.')
        print('PHASE: verify existing scientific files and environment, read-only', flush=True)
        plan = runtime.verify_bundle(root)
        expected = dict(bundle_sha256=digest(root/'bundle_manifest.json'), environment=runtime.environment())
        if (runtime.read(args.results/'environment_binding.json') != expected
                or runtime.read(args.results/'preflight.json') != dict(status='passed', **expected)):
            raise RuntimeError('Existing ADNI environment/preflight does not match. No binding or receipt was changed.')
        if shutil.disk_usage(root).free < 16*1024**3:
            raise RuntimeError('At least 16 GiB of local free space is needed for disposable diagnostic checkpoints')
        package = Path(__file__).resolve().parent
        manifest = read(package/'validation_manifest.json')
        if manifest['scientific_bundle_sha256'] != expected['bundle_sha256']:
            raise ValueError('Validation package belongs to another frozen ADNI bundle')
        for name, checksum in manifest['files'].items():
            if Path(name).name != name or digest(package/name) != checksum:
                raise ValueError('Validation package hash differs: '+name)
        args.output.mkdir(parents=True, exist_ok=False)
        report = dict(status='running', binding=expected, candidate_sha256=digest(package/'acceleration_candidate.py'),
            validator_sha256=digest(Path(__file__)), models=plan['models'], checked=[],
            extra_scientific_runs=0, campaign_results_modified=False, diagnostic_updates_per_path=3,
            max_diagnostic_updates=6*len(plan['models']), comparison='exact; no tolerance changes',
            training_approval=False)
        try:
            print('BOUND: four models, two paths, three disposable diagnostic updates per path (24 maximum). No campaign jobs.', flush=True)
            for model in plan['models']:
                for accelerated in (False, True):
                    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--bundle', str(root),
                        '--output', str(args.output), '--model', model]
                    if accelerated:
                        command.append('--candidate')
                    subprocess.run(command, cwd=root/'source', env=runtime.child_environment(root), check=True)
                compare(read(args.output/(model+'_original.json')), read(args.output/(model+'_candidate.json')))
                report['checked'].append(model)
                write(args.output/'summary.json', report)
                print('EXACT EQUIVALENCE PASSED:', model, flush=True)
            report['status'] = 'passed_equivalence'
        except BaseException:
            report['status'] = 'failed_equivalence'
            report['error'] = traceback.format_exc()
            raise
        finally:
            write(args.output/'summary.json', report)
            # Reports only: temporary checkpoints have already been removed.
            with zipfile.ZipFile(args.output.with_suffix('.zip'), 'w', zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(args.output.glob('*.json')):
                    archive.write(path, path.name)
            print('REPORT ZIP:', args.output.with_suffix('.zip'), flush=True)


if __name__ == '__main__':
    main()
