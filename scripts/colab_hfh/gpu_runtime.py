"""Explicit CUDA portability overlay; original package stays immutable on Drive."""
import os
import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path


def replace_required(text, old, new):
    if new and new in text:
        return text
    if old not in text:
        if not new:
            return text
        raise ValueError('GPU portability source anchor changed: '+old[:70])
    return text.replace(old, new)


def prepare(bundle, kind, native_bf16):
    """Only unsupported-BF16 devices change arithmetic, in a separate campaign."""
    source = bundle/'source'
    changed = {}
    def patch(relative, pairs):
        path = source/relative
        before = path.read_text()
        after = before
        for old, new in pairs:
            after = replace_required(after, old, new)
        ast.parse(after)
        if before != after:
            path.write_text(after)
            changed[relative] = dict(before=hashlib.sha256(before.encode()).hexdigest(),
                                     after=hashlib.sha256(after.encode()).hexdigest())

    quota_path = source/'scripts/colab_adni/runtime.py'
    quota_source = quota_path.read_text()
    quota_start = quota_source.index('def check_drive_headroom(durable):')
    quota_end = quota_source.index('\n\nclass RecoveryWriter:', quota_start)
    quota_function = '''def check_drive_headroom(durable):
    """Query the authenticated storage remote, never Colab's compute login."""
    private = Path('/content/.hfh_storage_auth')
    if not os.path.ismount('/content/hfh_storage'):
        raise RuntimeError('Second-account storage is not mounted')
    result = subprocess.run([str(private/'rclone'), '--config', str(private/'rclone.conf'),
        'about', 'hfh_storage:', '--json'], capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise RuntimeError('Could not verify second-account storage quota')
    quota = json.loads(result.stdout)
    if 'free' not in quota:
        raise RuntimeError('Storage remote did not report free space')
    write(durable/'storage_quota.json', dict(checked_at=time.time(),
        storage_account=os.environ.get('HIPPO_STORAGE_EMAIL', ''), provider='rclone hfh_storage', **quota))
    if int(quota['free']) < 8 * 1024**3:
        raise RuntimeError('Second-account Drive headroom below 8 GiB; recovery retained')
'''
    patch('scripts/colab_adni/runtime.py', [(quota_source[quota_start:quota_end], quota_function)])

    # Small result files previously made serial upload/read/rename round trips,
    # twice per completed run. Keep verification, overlap independent transfers.
    metadata_path = source/'scripts/colab_hfh/runtime.py'
    metadata_source = metadata_path.read_text()
    metadata_start = metadata_source.index('def publish_metadata(output, durable):')
    metadata_end = metadata_source.index('\n\ndef verify_archive', metadata_start)
    metadata_function = '''def publish_metadata(output, durable):
    from concurrent.futures import ThreadPoolExecutor
    paths = [path for path in sorted(output.iterdir())
        if path.is_file() and path.suffix in ('.json', '.csv', '.log', '.jsonl', '.png')
        and path.name not in ('snapshot_manifest.json', 'supervisor_ready.json')]
    def publish(path):
        destination = durable/path.name
        if destination.exists() and digest(destination) == digest(path):
            return
        copy_verified(path, destination)
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(publish, paths))
'''
    patch('scripts/colab_hfh/runtime.py', [(metadata_source[metadata_start:metadata_end], metadata_function)])

    if not native_bf16:
        if kind == 'baselines':
            patch('scripts/cross_dataset_campaign/capped_core.py', [
                ('amp = device.type == "cuda"', 'amp = False  # Explicit FP32 portability campaign'),
            ])
            patch('scripts/colab_hfh/preflight.py', [
                ("torch.autocast('cuda', dtype=torch.bfloat16)",
                 "torch.autocast('cuda', dtype=torch.bfloat16, enabled=False)"),
            ])
            for path in (bundle/'protocols').glob('*/protocol.json'):
                value = json.loads(path.read_text())
                value['precision'] = 'CUDA FP32 parameters, optimizer and forward; no autocast'
                value['precision_campaign'] = 'fp32_portability'
                path.write_text(json.dumps(value, indent=2)+'\n')
        else:
            for relative in ('scripts/cross_dataset_campaign/foundation.py',
                             'scripts/colab_medsam3/performance_candidates.py'):
                patch(relative, [("torch.autocast('cuda', dtype=torch.bfloat16)",
                                  "torch.autocast('cuda', dtype=torch.bfloat16, enabled=False)")])

        # Baseline preflight imports this environment checker. CPU is never used
        # silently; FP32 on CUDA is explicit and source-bound above.
        patch('scripts/colab_adni/runtime.py', [
            ("if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:",
             'if not torch.cuda.is_available():'),
            ("    if not torch.cuda.is_bf16_supported():\n        raise RuntimeError('BF16 is unavailable; precision will not be changed automatically')\n", ''),
        ])

    # Record the actual GPU in each archived experiment, including resumed work.
    patch('scripts/colab_hfh/search.py', [
        ("                print('START', name, 'final Dice threshold 50%; no intermediate-loss pruning', flush=True)",
         "                hardware = os.environ.get('HIPPO_RUNTIME_ENVIRONMENT')\n"
         "                if hardware:\n"
         "                    import json\n"
         "                    write(output/'execution_environment.json', json.loads(hardware))\n"
         "                print('START', name, 'final Dice threshold 50%; no intermediate-loss pruning', flush=True)"),
        ("            if args.kind == 'baselines' and len(accepted) == 3:",
         "            failures = [r for r in rows if r['validation_dice'] < .5]\n"
         "            if args.kind == 'baselines' and len(failures) >= 3 and len({r['model'] for r in failures[-3:]}) == 1:\n"
         "                write(args.durable/'search_status.json', dict(status='needs_recipe_review',\n"
         "                    accepted_seeds=accepted, rejected_seeds=rejected, results=rows,\n"
         "                    struggling_model=failures[-1]['model'], reason='Three successive rejected seeds fail the same model; inspect curves before further seed search'))\n"
         "                print('PAUSED FOR RECIPE REVIEW:', failures[-1]['model'], flush=True)\n"
         "                return\n"
         "            if args.kind == 'baselines' and len(accepted) == 3:"),
    ])
    manifest = {str(p.relative_to(bundle)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(bundle.rglob('*')) if p.is_file()
                and p.name not in ('manifest.json', 'runtime_manifest.json', 'gpu_overlay.json')
                and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    (bundle/'runtime_manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2)+'\n')
    (bundle/'gpu_overlay.json').write_text(json.dumps(dict(kind=kind,
        precision='bf16' if native_bf16 else 'fp32', changes=changed), indent=2)+'\n')
    return hashlib.sha256((bundle/'runtime_manifest.json').read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--kind', choices=('baselines', 'medsam3'), required=True)
    args = parser.parse_args()
    import torch
    import psutil
    if not torch.cuda.is_available():
        raise RuntimeError('Connect an available NVIDIA GPU; no CPU campaign is launched automatically.')
    native = torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported()
    if psutil.virtual_memory().available < 12*1024**3:
        raise RuntimeError('At least 12 GiB free host RAM is needed; enable High-RAM.')
    properties = torch.cuda.get_device_properties(0)
    packages = ('torch', 'torchvision', 'monai', 'dynamic-network-architectures',
                'numpy', 'pandas', 'scipy', 'scikit-image', 'nibabel', 'einops', 'timm', 'Pillow')
    environment = dict(gpu=properties.name, vram_bytes=properties.total_memory,
        compute_capability=list(torch.cuda.get_device_capability()), precision='bf16' if native else 'fp32',
        cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        packages={name:importlib.metadata.version(name) for name in packages})
    manifest = prepare(args.bundle, args.kind, native)
    print(json.dumps(dict(manifest_sha256=manifest, environment=environment)))


if __name__ == '__main__':
    main()
