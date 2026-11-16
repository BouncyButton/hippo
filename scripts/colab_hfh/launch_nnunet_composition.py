"""Explicit launch only, from the existing baseline Colab runtime after clearance.

This file is prepared locally. Merely importing it or viewing it starts no work.
It does not wait, queue, connect a runtime, or modify the original bundle.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def unpack(archive, destination, expected_sha):
    if sha(archive) != expected_sha:
        raise ValueError('Package ZIP differs from the reviewed build receipt')
    with zipfile.ZipFile(archive) as zipped:
        for item in zipped.infolist():
            name = Path(item.filename)
            if name.is_absolute() or '..' in name.parts or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Unsafe package member')
        zipped.extractall(destination)
    for relative, expected in read(destination/'manifest.json').items():
        name = Path(relative)
        if name.is_absolute() or '..' in name.parts or sha(destination/name) != expected:
            raise ValueError('Package member mismatch: '+relative)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start',action='store_true',help='Use only after the user explicitly says to launch')
    parser.add_argument('--package',type=Path,required=True,help='Locally uploaded package ZIP')
    parser.add_argument('--sha256',required=True,help='Expected SHA256 from build_receipt.json')
    parser.add_argument('--baseline-root',type=Path,default=Path('/content/hippo_hfh_baselines_v2'))
    parser.add_argument('--root',type=Path,default=Path('/content/hippo_nnunet_composition_v1'))
    args = parser.parse_args()
    if not args.start:
        parser.error('Prepared only. No launch without --start after explicit user go-ahead.')
    root, baseline = args.root.resolve(), args.baseline_root.resolve()
    if root == baseline or baseline in root.parents or root in baseline.parents:
        raise ValueError('Execution root must be separate from the baseline runtime')
    python = baseline/'venv/bin/python'
    if not python.exists() or not (baseline/'storage_sdk.py').exists():
        raise RuntimeError('Existing baseline runtime is unavailable; restore it separately before launch')
    # Reuse the scientific environment; do not install or upgrade anything.
    if Path(sys.executable).absolute() != python.absolute():
        os.execv(str(python),[str(python),str(Path(__file__).resolve()),*sys.argv[1:]])
    root.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (root/'launch.lock').open('a') as launch_lock:
        fcntl.flock(launch_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        package = root/'package'
        if package.exists():
            if not (root/'package_sha256.txt').exists() or (root/'package_sha256.txt').read_text().strip() != args.sha256:
                raise RuntimeError('An existing execution package differs; do not overwrite it')
        else:
            unpack(args.package,package,args.sha256)
            (root/'package_sha256.txt').write_text(args.sha256+'\n')
        source = baseline/'bundle/source'
        # Package modules can import the unchanged baseline dependencies here.
        # The definitive execution imports happen in a fresh isolated process.
        sys.path[:0] = [str(baseline/'bundle/compatibility'),str(baseline),str(source),str(source/'scripts')]
        import scripts.colab_hfh
        scripts.colab_hfh.__path__.insert(0,str(package/'source/scripts/colab_hfh'))
        from scripts.colab_hfh.nnunet_composition_campaign import check_campaign_completion, check_no_workers, verify_package
        verify_package(package)
        check_no_workers()
        if subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip() != 'NVIDIA L4':
            raise RuntimeError('An idle NVIDIA L4 is required for the matched native comparison')
        os.environ.update(HFH_DRIVE_CREDENTIALS='gcloud',HFH_DRIVE_TRANSPORT='colab_adc')
        from storage_sdk import Drive
        storage = Drive()
        medsam_path = root/'medsam_completion.json'
        storage.download('hfh_storage:results_medsam3_component_constraints125_f0s0_v1/monitor/live_status.json',medsam_path)
        check_campaign_completion(read(package/'unetr_verification.json'),read(medsam_path))
        # No VM or campaign is taken over, and no shared code is edited.
        if not (root/'bundle').exists():
            shutil.copytree(baseline/'bundle',root/'bundle',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            shutil.copytree(package/'source',root/'bundle/source',dirs_exist_ok=True)
            shutil.copyfile(baseline/'storage_sdk.py',root/'storage_sdk.py')
            execution = root/'bundle/source'
            manifest = {str(p.relative_to(execution)):sha(p) for p in sorted(execution.rglob('*'))
                        if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
            write(root/'execution_source_manifest.json',manifest)
            with zipfile.ZipFile(root/'execution_source.zip','w',zipfile.ZIP_DEFLATED) as zipped:
                for name in manifest:
                    zipped.write(execution/name,'source/'+name)
                zipped.write(root/'execution_source_manifest.json','execution_source_manifest.json')
                zipped.write(root/'storage_sdk.py','storage_sdk.py')
            provenance = dict(prepared_at=time.time(),package_sha256=args.sha256,
                baseline_root=str(baseline),execution_root=str(root),python=str(python),
                source_manifest_sha256=sha(root/'execution_source_manifest.json'),
                execution_source_zip_sha256=sha(root/'execution_source.zip'),
                storage_sdk_sha256=sha(root/'storage_sdk.py'),
                dataset_index_sha256=sha(root/'bundle/data_index.json'),
                runtime_environment=read(baseline/'unetr_restore_environment.json'))
            write(root/'launch_provenance.json',provenance)
        # A partial staging operation is never treated as a completed one.
        for required in ('execution_source_manifest.json','execution_source.zip','launch_provenance.json','storage_sdk.py'):
            if not (root/required).is_file():
                raise RuntimeError('Incomplete isolated staging: '+required)
        write(root/'clearance.json',dict(checked_at=time.time(),medsam=read(medsam_path)))
        env = dict(os.environ,PYTHONPATH=':'.join([str(root)]+[str(root/'bundle'/p) for p in
            ('compatibility','source','source/scripts','source/baselines/nnUNet')]),PYTHONUNBUFFERED='1')
        # Synchronous by design: caller may explicitly place this command in a
        # background process when launch is authorized. No automatic queue.
        subprocess.run([str(python),'-m','scripts.colab_hfh.nnunet_composition_campaign',
            '--root',str(root),'--package',str(package),'--clearance',str(root/'clearance.json')],
            cwd=root/'bundle/source',env=env,check=True)


if __name__ == '__main__':
    main()
