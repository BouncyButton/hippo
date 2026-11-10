"""Check exact Swin initial states on CPU without any training/gradient trials."""
import argparse
import fcntl
import gc
import importlib.util
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'durable', 'control'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    root, durable, control = args.bundle.resolve(), args.durable.resolve(), args.control.resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(control))
    import runtime
    import acceleration_gate
    from initialization_compat import install, verify_receipt, RECEIPT_NAME
    runtime.activate(root)
    with (root/'work/sweep.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ('.adni_controller_lock', '.adni_seeds014_controller_lock'):
            if (durable/name).exists():
                raise RuntimeError('Existing controller ownership; preserve it and inspect before resuming.')
        print('PHASE: verify unchanged source, environment, GPU preflight and acceleration evidence', flush=True)
        plan = runtime.verify_bundle(root)
        binding = dict(bundle_sha256=runtime.digest(root/'bundle_manifest.json'), environment=runtime.environment())
        if runtime.read(durable/'environment_binding.json') != binding:
            raise RuntimeError('GPU/software changed. Restore the saved environment; nothing was reset.')
        # verify() deliberately precedes the NEW CPU receipt. existing() will
        # require both the original GPU gate and this new receipt on each run.
        acceleration_gate.verify(control, root, binding, runtime.read(durable/'preflight.json'))
        import torch
        from scripts.cross_dataset_campaign import conventional as experiment
        install(runtime)
        seeds = [4] if plan['seeds'] == [4] else [0, 1]
        reference = runtime.read(root/'initialization_reference.json')['models']['swinunetr']
        torch.set_num_threads(2)
        checks = []
        for seed in seeds:
            print(f'PHASE: CPU Swin initialization seed {seed}; zero training updates', flush=True)
            folder = root/'protocols'/f'swinunetr_f0_s{seed}'
            protocol = runtime.read(folder/'protocol.json')
            architecture = runtime.read(folder/'nnunet_architecture.json')
            # The same seed starts from the identical reference in every fold.
            for fold in (0, 1, 2):
                other = root/'protocols'/f'swinunetr_f{fold}_s{seed}'
                if (runtime.read(other/'protocol.json')['baseline_reference']['initialization_sha256']
                        != reference[str(seed)]['model_sha256']
                        or runtime.read(other/'nnunet_architecture.json') != architecture):
                    raise ValueError('Swin fold-specific initialization changed')
            torch.manual_seed(seed)
            model, audit = runtime.canonical_initialization(
                experiment.build_model, protocol, architecture, reference[str(seed)])
            checks.append(dict(seed=seed, model_sha256=runtime.model_digest(model),
                rng_after_model=runtime.tensor_digest(torch.get_rng_state()),
                portable_calls=len(audit['portable_calls']), folds_checked=[0, 1, 2],
                maximum_scratch_difference=max(row['max_abs_difference'] for row in audit['portable_calls'])))
            print(f'PASSED: seed {seed}; exact frozen model hash and RNG state', flush=True)
            del model
            gc.collect()
        receipt = dict(status='passed_cpu_initialization', binding=binding,
            control_sha256=runtime.digest(control/'control_manifest.json'),
            compat_sha256=runtime.digest(control/'initialization_compat.py'),
            seeds=seeds, checks=checks, optimizer_updates=0,
            cpu_capability=torch.backends.cpu.get_cpu_capability())
        runtime.write(durable/RECEIPT_NAME, receipt)
        if runtime.read(durable/RECEIPT_NAME) != receipt:
            raise IOError('CPU initialization receipt readback failed')
        verify_receipt(runtime, control, root, durable, binding)
        print('PASSED: original preflight/bindings untouched; CPU compatibility verified; no experiments added.', flush=True)


if __name__ == '__main__':
    main()
