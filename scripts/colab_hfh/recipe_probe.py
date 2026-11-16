"""Bounded HFH recipe diagnostics on local SSD, with verified remote archives."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', default='nnunet')
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--folds', type=int, nargs='+', default=[0])
    args = parser.parse_args()
    bundle = args.root/'bundle'
    precision = json.loads((bundle/'gpu_overlay.json').read_text())['precision']
    suffix = '' if precision == 'bf16' else '_'+precision
    label = f'{args.model}_lr{args.lr:g}_poly_cap'
    work = args.root/('recipe_diagnostics'+suffix)/label
    protocol = work/'protocol'
    protocol.mkdir(parents=True, exist_ok=True)
    for path in (bundle/'protocols'/args.model).iterdir():
        if path.is_file():
            shutil.copyfile(path, protocol/path.name)
    p = json.loads((protocol/'protocol.json').read_text())
    p['initial_lrs'][p['trainer_alias']] = args.lr
    p['recipe_diagnostic'] = dict(name=label, parent='HFH_v2_cap_schedule',
        changed_fields=['initial_lrs'], selection='exploratory; all attempts retained')
    (protocol/'protocol.json').write_text(json.dumps(p, indent=2))
    env = dict(os.environ, PYTHONPATH=':'.join(map(str, [bundle/'compatibility', bundle/'source',
        bundle/'source/scripts', bundle/'source/baselines/nnUNet'])), PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', WANDB_MODE='disabled', HIPPO_HFH_GPU_CACHE='1',
        HIPPO_DEADLINE=str(time.time()+3600))
    sys.path.insert(0, str(bundle/'source'))
    from scripts.colab_hfh.runtime import archive, verify_archive
    private = Path('/content/.hfh_storage_auth')
    rc = [str(private/'rclone'), '--config', str(private/'rclone.conf'),
          '--tpslimit', '2', '--tpslimit-burst', '2', '--retries', '3', '--low-level-retries', '10']
    for fold in args.folds:
        for seed in args.seeds:
            name = f'{args.model}_f{fold}_s{seed}'
            output, durable = work/name, work/'archives'/name
            output.mkdir(parents=True, exist_ok=True)
            durable.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-m', 'scripts.colab_hfh.baseline',
                '--protocol-dir', str(protocol), '--output', str(output), '--durable', str(durable),
                '--pkl', str(bundle/'data_index.json'), '--seed', str(seed), '--fold', str(fold)]
            if (durable/'cleanup.json').exists():
                result = verify_archive(durable)
                saved_recipe = json.loads((durable/'runtime_recipe.json').read_text())
                if saved_recipe['protocol_sha256'] != hashlib.sha256((protocol/'protocol.json').read_bytes()).hexdigest():
                    raise RuntimeError('Recipe changed for an existing result; use a new campaign')
                print('REUSE VERIFIED RESULT', label, name, flush=True)
            else:
                print('DIAGNOSTIC START', label, name, flush=True)
                hardware = dict(precision=precision, gpu=subprocess.check_output([
                    'nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'], text=True).strip(),
                    packages={n:importlib.metadata.version(n) for n in ('torch','monai','dynamic-network-architectures')})
                (output/'execution_environment.json').write_text(json.dumps(hardware, indent=2))
                if not (output/'completion.json').exists():
                    with (output/'worker.log').open('a') as log:
                        subprocess.run(command, cwd=bundle/'source', env=env, stdout=log,
                                       stderr=subprocess.STDOUT, check=True)
                result = archive(output, durable, 'baselines', args.model, fold, seed, protocol)
            print('DIAGNOSTIC RESULT', json.dumps(result), flush=True)
            target = work/(name+'.zip')
            receipt = work/(name+'.remote_verified.json')
            if receipt.exists() and target.exists() and json.loads(receipt.read_text())['md5'] == hashlib.md5(target.read_bytes()).hexdigest():
                print('REUSE VERIFIED REMOTE ARCHIVE', name, flush=True)
                continue
            with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as zipped:
                for base, prefix in ((durable, 'results'), (protocol, 'protocol')):
                    for path in sorted(base.rglob('*')):
                        if path.is_file():
                            zipped.write(path, prefix+'/'+str(path.relative_to(base)))
                zipped.write(__file__, 'recipe_probe.py')
                for filename in ('manifest.json', 'runtime_manifest.json', 'gpu_overlay.json'):
                    if (bundle/filename).exists():
                        zipped.write(bundle/filename, 'source_provenance/'+filename)
                if (args.root/'gpu_runtime.py').exists():
                    zipped.write(args.root/'gpu_runtime.py', 'source_provenance/gpu_runtime.py')
            remote = 'hfh_storage:results_recipe_diagnostics_v3'+suffix+'/'+label+'/'+target.name
            subprocess.run(rc+['copyto', str(target), remote], check=True, timeout=600)
            check = subprocess.check_output(rc+['md5sum', remote], text=True, timeout=120).split()[0]
            if check != hashlib.md5(target.read_bytes()).hexdigest():
                raise RuntimeError('Remote diagnostic archive checksum mismatch')
            receipt.write_text(json.dumps(dict(remote=remote, md5=check, verified_at=time.time())))
            print('REMOTE ARCHIVE VERIFIED', remote, flush=True)


if __name__ == '__main__':
    main()
