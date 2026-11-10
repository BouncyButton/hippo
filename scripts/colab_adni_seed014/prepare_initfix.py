"""Package a narrow external initialization fix, preserving both frozen bundles."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from scripts.colab_adni_seed014 import prepare as old

OUT = old.ROOT/'reports/adni_colab_seeds014_initfix_20261018'
ZIP_NAME = 'adni_seeds014_resume_initfix_v2_code.zip'
BOOK_NAME = 'ADNI_SEEDS_0_1_4_RESUME_INITFIX_v2_NO_AUGMENTATION.ipynb'


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Unexpected original package source: '+before)
    return text.replace(before, after, 1)


def prepare():
    OUT.mkdir(exist_ok=True)
    package = OUT/'package'
    # Exclude only local generated caches and the original dataset symlink.
    shutil.copytree(old.OUT/'package', package, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'data'))
    for name in ('control01', 'control04'):
        control = package/name
        prior = old.OUT/'package'/name
        shutil.copyfile(old.SOURCE/'initialization_compat.py', control/'initialization_compat.py')
        worker = (prior/'worker.py').read_text()
        worker = replace_once(worker,
            "        protocol = json.loads((Path(trainer_args[trainer_args.index('--protocol-dir')+1])/'protocol.json').read_text())",
            "        from initialization_compat import install\n"
            "        install(runtime)\n"
            "        stats['initialization_compat_sha256'] = sha(control/'initialization_compat.py')\n"
            "        protocol = json.loads((Path(trainer_args[trainer_args.index('--protocol-dir')+1])/'protocol.json').read_text())")
        (control/'worker.py').write_text(worker)
        gate = (prior/'acceleration_gate.py').read_text()
        if name == 'control01':
            gate = replace_once(gate, '    return binding, receipt\n',
                '    from initialization_compat import verify_receipt\n'
                '    verify_receipt(runtime, control, root, durable, binding)\n'
                '    return binding, receipt\n')
        else:
            gate = replace_once(gate, "    return binding, verify(control, root, binding, read(durable/'preflight.json'))",
                "    receipt = verify(control, root, binding, read(durable/'preflight.json'))\n"
                "    from initialization_compat import verify_receipt\n"
                "    verify_receipt(runtime, control, root, durable, binding)\n"
                "    return binding, receipt")
        (control/'acceleration_gate.py').write_text(gate)
        manifest = old.read(prior/'control_manifest.json')
        for file in ('worker.py', 'acceleration_gate.py', 'initialization_compat.py'):
            manifest['files'][file] = old.sha(control/file)
        manifest.setdefault('compatible_profile_control_sha256', []).append(old.sha(prior/'control_manifest.json'))
        manifest.setdefault('compatible_profile_worker_sha256', []).append(old.sha(prior/'worker.py'))
        manifest['initialization_compat'] = dict(scope='Swin CPU inverse-CDF arithmetic only; frozen tolerance/model/RNG gates unchanged',
            source_sha256=old.sha(control/'initialization_compat.py'), cpu_preflight_required=True)
        old.write(control/'control_manifest.json', manifest)
    shutil.copyfile(old.SOURCE/'preflight_initfix.py', package/'preflight_initfix.py')
    files = {str(p.relative_to(package)):old.sha(p) for p in sorted(package.rglob('*'))
             if p.is_file() and p.name != 'package_manifest.json' and '__pycache__' not in p.parts}
    old.write(package/'package_manifest.json', files)
    with zipfile.ZipFile(OUT/ZIP_NAME, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in [*files, 'package_manifest.json']:
            archive.write(package/name, name)
    book = old.read(old.OUT/old.BOOK_NAME)
    old_summary = old.read(old.OUT/'export_summary.json')
    for cell in book['cells']:
        text = cell['source']
        text = text.replace(old.ZIP_NAME, ZIP_NAME).replace(old_summary['code_zip_sha256'], old.sha(OUT/ZIP_NAME))
        text = text.replace("PACKAGE=Path('/content/hippo_adni_seeds014')", "PACKAGE=Path('/content/hippo_adni_seeds014_initfix_v2')")
        text = text.replace('**Before switching:** stop the old run cell once and wait for verified recovery uploads. Do not run',
            '**Before resuming:** the checked controller has already stopped. If you started another controller\n'
            'since that check, stop it once and wait for verified recovery uploads. Do not run')
        if cell['cell_type'] == 'code' and "PACKAGE/'preflight.py'" in text:
            text = """with phase('verify saved preflights and CPU Swin initialization; no training trials'):
    for root, durable, control in (
        (ORIGINAL_BUNDLE, ORIGINAL_RESULTS, PACKAGE/'control01'),
        (BUNDLE, RESULTS, PACKAGE/'control04'),
    ):
        run_logged([PYTHON, '-u', PACKAGE/'preflight_initfix.py',
                    '--bundle', root, '--durable', durable, '--control', control], 'preflight')
"""
        elif cell['cell_type'] == 'markdown' and '## 3. Required preflight' in text:
            text = """## 3. Required compatibility preflight (CPU only; no extra experiments)

Verify the unchanged source, saved GPU/software bindings, original passed GPU preflights and
A100 acceleration evidence. Then check Swin initializations for seeds 0/1/4 on CPU: the original
strict tolerance, all RNG checks and exact complete model hashes must pass. The same initialization
hashes are required in all three folds. This does not run forward/backward passes, optimizer updates
or benchmarks, and does not replace the original preflight receipts. A failed check blocks training.
Use the saved A100-SXM4-80GB environment; a different GPU/software environment is not silently accepted.
"""
        if cell['cell_type'] == 'code':
            ast.parse(text)
            cell.update(execution_count=None, outputs=[])
        cell['source'] = text
    book['cells'][0]['source'] += """

**Initialization repair v2:** replace only this notebook and upload the matching v2 code ZIP to
the existing input folder. Keep the original code ZIP, data ZIP, both results folders and recovery
files. Existing scientific source and manifests are byte-identical. The external worker uses a more
accurate CPU inverse-CDF calculation before the ORIGINAL strict check copies canonical Swin weights.
No tolerance is relaxed and no initialization/RNG/hash gate is bypassed.

The last checked snapshot had 329 completed target runs and four saved recoveries. These are not
hardcoded counts: every completed archive is checked and skipped; interrupted work is restored first.
Seven seeds-0/1 runs and 168 seed-4 runs remained at that snapshot. No new experiments are added.
After a clean stop or the confirmed drained failure, run this notebook from the top. On a fresh VM,
setup copies the existing data ZIP from Drive to local SSD; no dataset re-upload is needed.
Do not delete or reset any results, recovery, accounting, STOP or ownership files.
"""
    book['metadata']['colab']['name'] = BOOK_NAME
    old.write(OUT/BOOK_NAME, book)
    old.write(OUT/'export_summary.json', dict(status='packaged_pending_local_tests',
        notebook=BOOK_NAME, code_zip=ZIP_NAME, code_zip_sha256=old.sha(OUT/ZIP_NAME),
        notebook_sha256=old.sha(OUT/BOOK_NAME), extra_scientific_runs=0,
        original_bundle_sha256=old_summary['parent_bundle_sha256'],
        seed4_bundle_sha256=old_summary['seed4_bundle_sha256'],
        results_directories=['results_noaug_v1', 'results_seeds014_noaug_v1'],
        unchanged_scientific_bundles=True, augmentation='none', gpu_tested=False))
    print(OUT)


if __name__ == '__main__':
    prepare()
